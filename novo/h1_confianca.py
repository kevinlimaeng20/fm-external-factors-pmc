"""Teste 5: nota de confiança por previsão.
Um modelo (LightGBM) prevê o tamanho do erro percentual da combinação (mediana dos 12 modelos) usando só o que se
sabe na data da previsão: horizonte, mês, discordância entre os modelos, largura da faixa dos modelos, ruído do
histórico, erro recente já observado da própria série, aceleração recente. Treino: só previsões cujo alvo já foi
observado. Avaliação: 26 últimas origens do teste. Mede (a) se a nota separa previsões boas das ruins e
(b) se faixas de 80% construídas com a nota (conformal normalizado) são melhores que as faixas dos modelos."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from base import OUT, painel, QCOLS

d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
BASE = [m for m in d.modelo.unique() if not m.startswith("Combinação")]
K = ["unique_id", "origem", "h", "data_alvo"]
x = d[d.modelo.isin(BASE)]
g = x.groupby(K)
W = g.agg(y=("y", "first"), escala=("escala", "first"), prev=("q50", "median"), q10=("q10", "median"), q90=("q90", "median"),
          dp=("q50", "std"), mn=("q50", "min"), mx=("q50", "max")).reset_index()
W["ape"] = 100 * (W.prev - W.y).abs() / W.y
W["disc_pct"] = 100 * W.dp / W.prev; W["amp_pct"] = 100 * (W.mx - W.mn) / W.prev; W["larg80_pct"] = 100 * (W.q90 - W.q10) / W.prev
W["mes"] = W.data_alvo.dt.month
ft = pd.read_csv(f"{OUT}/previsibilidade_series.csv").set_index("unique_id")[["forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "amplitude_sazonal_pct", "brasil"]]
W = W.join(ft, on="unique_id")
p = painel(); S = {u: np.log(s.set_index("data").valor) for u, s in p.groupby("unique_id")}
origens = np.sort(W.origem.unique()); AVAL = origens[12:]
din = []
for o in origens:
    kn = W[(W.data_alvo <= o) & (W.data_alvo > o - pd.DateOffset(months=12))]
    er = kn.groupby("unique_id").ape.mean().rename("erro_recente")
    for u, s in S.items():
        yoy = s[:o].diff(12).dropna(); vol = yoy.iloc[-36:].std()
        din.append(dict(unique_id=u, origem=o, acel=(yoy.iloc[-3:].mean() - yoy.iloc[-15:-3].mean()) / vol, surpresa_ult=abs(yoy.iloc[-1] - yoy.iloc[-13:-1].mean()) / vol,
                        erro_recente=er.get(u, np.nan)))
W = W.merge(pd.DataFrame(din), on=["unique_id", "origem"])
FE = ["h", "mes", "disc_pct", "amp_pct", "larg80_pct", "forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "amplitude_sazonal_pct", "brasil", "acel", "surpresa_ult", "erro_recente"]
W["alvo"] = np.log(W.ape + 0.1)
fold = pd.factorize(W.unique_id)[0] % 5
par = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=2, random_state=0)
for c in ("nota", "f_const", "f_h", "f_nota", "f_larg"):
    W[c] = np.nan
hb = np.digitize(W.h.values, [4, 8])
for o in AVAL:
    tr = (W.data_alvo <= o).values; te = (W.origem == o).values
    m = lgb.LGBMRegressor(**par).fit(W.loc[tr, FE], W.loc[tr, "alvo"])
    W.loc[te, "nota"] = np.exp(m.predict(W.loc[te, FE]))                      # erro esperado (%, escala geométrica)
    oof = np.zeros(tr.sum())                                                # notas fora da amostra para calibrar as faixas
    ftr = fold[tr]
    for k in range(5):
        mk = lgb.LGBMRegressor(**par).fit(W.loc[tr, FE][ftr != k], W.loc[tr, "alvo"][ftr != k]); oof[ftr == k] = np.exp(mk.predict(W.loc[tr, FE][ftr == k]))
    ape_tr = W.loc[tr, "ape"].values
    W.loc[te, "f_const"] = np.quantile(ape_tr, 0.8)                           # meia-largura (%): constante
    for b in range(3):
        mb = hb[tr] == b
        W.loc[te & (hb == b), "f_h"] = np.quantile(ape_tr[mb] if mb.sum() > 300 else ape_tr, 0.8)
    W.loc[te, "f_nota"] = np.quantile(ape_tr / oof, 0.8) * W.loc[te, "nota"]   # proporcional à nota
    W.loc[te, "f_larg"] = np.quantile(ape_tr / W.loc[tr, "larg80_pct"].values, 0.8) * W.loc[te, "larg80_pct"]  # proporcional à faixa dos modelos
    print(pd.Timestamp(o).date(), flush=True)
E = W[W.origem.isin(AVAL)].copy()
# (a) a nota separa?
lin = []
for nome, col in (("Nota de confiança (modelo)", "nota"), ("Só discordância entre modelos", "disc_pct"), ("Só largura da faixa dos modelos", "larg80_pct"),
                  ("Só ruído do histórico", "ruido_pct"), ("Só erro recente da série", "erro_recente"), ("Só horizonte", "h")):
    v = E.dropna(subset=[col]); lin.append(dict(sinal=nome, spearman_com_erro=spearmanr(v[col], v.ape)[0]))
sep = pd.DataFrame(lin); sep.round(3).to_csv(f"{OUT}/t5_separacao.csv", index=False); print(sep.round(3).to_string())
E["faixa_nota"] = pd.qcut(E.nota, [0, .25, .5, .75, .9, 1], labels=["A (25% mais confiáveis)", "B", "C", "D", "E (10% menos confiáveis)"])
fx = E.groupby("faixa_nota", observed=True).agg(n=("ape", "size"), erro_esperado=("nota", "mean"), MAPE=("ape", "mean"), erro_mediano=("ape", "median"),
                                                 dentro_5=("ape", lambda v: 100 * (v <= 5).mean()), dentro_10=("ape", lambda v: 100 * (v <= 10).mean()), acima_20=("ape", lambda v: 100 * (v > 20).mean()))
fx["pct_do_erro_total"] = 100 * E.groupby("faixa_nota", observed=True).ape.sum() / E.ape.sum()
fx.round(2).to_csv(f"{OUT}/t5_faixas_de_confianca.csv"); print(fx.round(2).to_string())
top5 = E.ape >= E.ape.quantile(0.95)
print("dos 5% maiores erros, % que estavam nas faixas D/E:", round(100 * E[top5].faixa_nota.isin(["D", "E (10% menos confiáveis)"]).mean(), 1))
# (b) faixas de 80%
E["quartil_ruido"] = pd.qcut(E.ruido_pct, 4, labels=["1 (menos ruído)", "2", "3", "4 (mais ruído)"])
lin = []
for nome, lo, hi in [("Faixa dos modelos (mediana dos quantis)", 100 * (E.q10 / E.prev - 1), 100 * (E.q90 / E.prev - 1))] + \
        [(n, -E[c], E[c]) for n, c in (("Largura constante", "f_const"), ("Largura por horizonte", "f_h"), ("Proporcional à faixa dos modelos", "f_larg"), ("Proporcional à nota de confiança", "f_nota"))]:
    e = 100 * (E.y / E.prev - 1); dentro = (e >= lo) & (e <= hi); larg = hi - lo
    wink = larg + (2 / 0.2) * ((lo - e).clip(lower=0) + (e - hi).clip(lower=0))
    r = dict(faixa=nome, cobertura=100 * dentro.mean(), largura_media_pct=larg.mean(), pontuacao_intervalo=wink.mean())
    for q, v in dentro.groupby(E.quartil_ruido, observed=True).mean().items():
        r[f"cobertura | ruído {q}"] = 100 * v
    for q, v in dentro.groupby(E.faixa_nota, observed=True).mean().items():
        r[f"cobertura | nota {q[0]}"] = 100 * v
    lin.append(r)
I = pd.DataFrame(lin); I.round(2).to_csv(f"{OUT}/t5_faixas_80.csv", index=False); print(I.round(1).T.to_string())
imp = pd.Series(m.booster_.feature_importance("gain"), FE); print((100 * imp / imp.sum()).sort_values(ascending=False).round(1).to_string())
(100 * imp / imp.sum()).sort_values(ascending=False).round(2).to_csv(f"{OUT}/t5_importancia.csv")
E.to_parquet(f"{OUT}/t5_confianca.parquet", index=False)
