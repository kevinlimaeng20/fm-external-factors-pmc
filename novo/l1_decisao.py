"""Ideia 8: otimizar a decisão de estoque em vez do erro da previsão.
Problema (jornaleiro): pedir q para o mês; faltar custa cu por unidade, sobrar custa co. O ótimo é o quantil
t = cu/(cu+co) da demanda ("nível de serviço alvo"). Custo medido = t*falta + (1-t)*sobra, em % das vendas.
Regras comparadas (todas só com o que se sabe na data; calibração em alvos já observados; 26 últimas origens):
  A  pedir a previsão central (mediana dos 12 modelos), ignorando custos
  B  ingênuo sazonal + margem histórica (método tradicional)
  C  quantil dos próprios modelos (mediana dos quantis)
  D  previsão central + margem calibrada por horizonte (conformal)
  E  previsão central + margem calibrada proporcional à incerteza da série (largura da faixa dos modelos)
  F  modelo treinado direto para o custo (LightGBM quantílico no nível t, com as previsões e características)"""
import numpy as np, pandas as pd, lightgbm as lgb
from base import OUT, QCOLS, boot_ic

d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
BASE = [m for m in d.modelo.unique() if not m.startswith("Combinação")]
K = ["unique_id", "origem", "h", "data_alvo"]
x = d[d.modelo.isin(BASE)]
W = x.groupby(K).agg(y=("y", "first"), prev=("q50", "median"), dp=("q50", "std"), **{c: (c, "median") for c in QCOLS if c != "q50"}).reset_index()
W["q50"] = W.prev
W = W.merge(d[d.modelo == "Ingênuo sazonal"][K + ["q50"]].rename(columns={"q50": "sn"}), on=K)
W["e"] = W.y / W.prev - 1; W["e_sn"] = W.y / W.sn - 1
W["larg"] = (W.q90 - W.q10) / W.prev; W["disc"] = W.dp / W.prev; W["mes"] = W.data_alvo.dt.month
for c in QCOLS: W[f"r_{c}"] = W[c] / W.prev - 1
ft = pd.read_csv(f"{OUT}/previsibilidade_series.csv").set_index("unique_id")[["forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "brasil"]]
W = W.join(ft, on="unique_id")
FE = ["h", "mes", "larg", "disc", "forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "brasil"] + [f"r_{c}" for c in QCOLS if c != "q50"]
origens = np.sort(W.origem.unique()); AVAL = origens[12:]; hb = np.digitize(W.h.values, [4, 8])
TAUS = [0.5, 0.7, 0.8, 0.9, 0.95]; QM = {0.5: "q50", 0.7: "q70", 0.8: "q80", 0.9: "q90"}
REG = ["A: pedir a previsão", "B: ingênuo + margem histórica", "C: quantil dos modelos", "D: margem por horizonte", "E: margem pela incerteza da série", "F: treinado para o custo"]
for t in TAUS:
    for r in REG: W[f"{r}|{t}"] = np.nan
for o in AVAL:
    tr = (W.data_alvo <= o).values; te = (W.origem == o).values
    for t in TAUS:
        W.loc[te, f"A: pedir a previsão|{t}"] = W.prev[te]
        if t in QM: W.loc[te, f"C: quantil dos modelos|{t}"] = W[QM[t]][te]
        for b in range(3):
            mb = tr & (hb == b); mb = mb if mb.sum() > 300 else tr; mt = te & (hb == b)
            W.loc[mt, f"D: margem por horizonte|{t}"] = W.prev[mt] * (1 + np.quantile(W.e[mb], t))
            W.loc[mt, f"B: ingênuo + margem histórica|{t}"] = W.sn[mt] * (1 + np.quantile(W.e_sn[mb], t))
            W.loc[mt, f"E: margem pela incerteza da série|{t}"] = W.prev[mt] * (1 + np.quantile(W.e[mb] / W.larg[mb], t) * W.larg[mt])
        m = lgb.LGBMRegressor(objective="quantile", alpha=t, n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=2, random_state=0).fit(W.loc[tr, FE], W.e[tr])
        W.loc[te, f"F: treinado para o custo|{t}"] = W.prev[te] * (1 + m.predict(W.loc[te, FE]))
    print(pd.Timestamp(o).date(), flush=True)
E_ = W[W.origem.isin(AVAL)].copy(); E_.to_parquet(f"{OUT}/l1_decisoes.parquet", index=False)
lin = []
for t in TAUS:
    custo = {}
    for r in REG:
        q = E_[f"{r}|{t}"]
        if q.isna().all(): continue
        falta = (E_.y - q).clip(lower=0) / E_.y; sobra = (q - E_.y).clip(lower=0) / E_.y
        c = 100 * (t * falta + (1 - t) * sobra); custo[r] = c.groupby(E_.unique_id).mean()
        lin.append(dict(nivel_servico_alvo=t, regra=r, custo_pct_vendas=c.mean(), servico_atingido=100 * (E_.y <= q).mean(), falta_media_pct=100 * falta.mean(), sobra_media_pct=100 * sobra.mean(),
                        meses_com_falta_acima_10pct=100 * (falta > 0.10).mean()))
    for l in lin:
        if l["nivel_servico_alvo"] == t:
            for ref, nome in (("A: pedir a previsão", "economia_vs_A_pct"), ("C: quantil dos modelos", "economia_vs_C_pct")):
                if ref in custo:
                    pass  # ICs recalculados por bootstrap da razão de médias (ver l1_decisao.csv)
                    l[nome] = 100 * (1 - custo[l["regra"]].mean() / custo[ref].mean())
T = pd.DataFrame(lin); T.round(3).to_csv(f"{OUT}/l1_decisao.csv", index=False)
pd.set_option("display.width", 250); print(T.round(2).to_string())
# por dificuldade da série (nível 0,9)
E_["quartil_ruido"] = pd.qcut(E_.ruido_pct, 4, labels=["1 (estável)", "2", "3", "4 (ruidosa)"]); t = 0.9; lin = []
for r in REG:
    q = E_[f"{r}|{t}"]; c = 100 * (t * (E_.y - q).clip(lower=0) + (1 - t) * (q - E_.y).clip(lower=0)) / E_.y
    lin.append(pd.concat([c.groupby(E_.quartil_ruido, observed=True).mean().rename("custo"), (100 * (E_.y <= q)).groupby(E_.quartil_ruido, observed=True).mean().rename("servico")], axis=1).assign(regra=r).reset_index())
Q = pd.concat(lin); Q.round(2).to_csv(f"{OUT}/l1_decisao_por_ruido.csv", index=False)
print(Q.pivot(index="regra", columns="quartil_ruido", values="custo").round(2).to_string()); print(Q.pivot(index="regra", columns="quartil_ruido", values="servico").round(1).to_string())
