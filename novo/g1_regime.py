"""Teste 4: detector de regime ("alarme de virada") na simulação ano a ano (previsões de dez/2004 a dez/2025).
Sinais calculados SÓ com dados até a origem:
  por série: aceleração do crescimento (últimos 3 meses contra os 12 anteriores, em desvios-padrão), erro da
             combinação no ano que acabou de fechar, outliers nos últimos 12 meses;
  agregados: média da aceleração entre as séries, % de séries em queda, confiança do consumidor, câmbio, inflação, Selic.
Perguntas: (A) os sinais antecipam anos ruins? (B) trocar de modelos quando o alarme toca reduz o erro?
(C) alargar a faixa de incerteza quando o alarme toca melhora a calibração?"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from statsmodels.tsa.seasonal import STL
from base import OUT, painel, meta, QCOLS, boot_ic

p = painel(); M = meta(p)
S = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}
d = pd.read_parquet(f"{OUT}/anual_avaliado.parquet")
d = d[d.groupby(["unique_id", "origem", "modelo"]).h.transform("count") == 12]          # anos completos
EST = ["ETS", "Theta", "ARIMA"]; IA = ["LightGBM", "Chronos-2", "TimesFM 2.5"]
TODOS = [m for m in d.modelo.unique() if m not in ("Combinação", "Ingênuo sazonal")]
K = ["unique_id", "origem", "h"]
W = d[d.modelo != "Combinação"].pivot_table(index=K + ["y"], columns="modelo", values="q50").reset_index()
W["Mediana"] = W[TODOS].median(axis=1); W["Estatísticos"] = W[EST].median(axis=1); W["IA"] = W[IA].median(axis=1)
esc = {u: (s - s.shift(12)).abs().expanding().mean() for u, s in S.items()}
W["escala"] = [esc[u].get(o) for u, o in zip(W.unique_id, W.origem)]
METS = ["Mediana", "Estatísticos", "IA", "Ingênuo sazonal"] + TODOS
A = W.assign(**{m: (W[m] - W.y).abs() / W.escala for m in METS}).groupby(["unique_id", "origem"])[METS].mean().reset_index()   # MASE por série-ano
A["ape_med"] = (100 * (W.Mediana - W.y).abs() / W.y).groupby([W.unique_id, W.origem]).mean().values
A["ano"] = A.origem.dt.year + 1

# ---------- sinais por série, só com dados até a origem
lin = []
for o in sorted(A.origem.unique()):
    for u in A[A.origem == o].unique_id:
        ls = np.log(S[u][:o]); yoy = ls.diff(12).dropna()
        g3, g12, vol = yoy.iloc[-3:].mean(), yoy.iloc[-15:-3].mean(), yoy.iloc[-36:].std()
        r = STL(ls, period=12, robust=True).fit().resid; mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        lin.append(dict(unique_id=u, origem=o, cresc3=g3, acel=(g3 - g12) / vol, vol=vol, outl12=int((np.abs(r.iloc[-12:] / mad) > 3).sum())))
Fs = pd.DataFrame(lin)
A = A.merge(Fs, on=["unique_id", "origem"])
ant = A[["unique_id", "origem", "Mediana", "Ingênuo sazonal"]].copy(); ant["origem"] = ant.origem + pd.DateOffset(years=1)
ant["erro_ano_ant"] = ant.Mediana / ant["Ingênuo sazonal"]; ant["mase_ano_ant"] = ant.Mediana
A = A.merge(ant[["unique_id", "origem", "erro_ano_ant", "mase_ano_ant"]], on=["unique_id", "origem"], how="left")
# ---------- sinais agregados
cov = pd.read_parquet("dados/covariaveis.parquet"); cb = cov[cov.geo_cod == 1].set_index("data")
ag = A.groupby("origem").agg(acel_media=("acel", "mean"), pct_queda=("cresc3", lambda v: 100 * (v < 0).mean()), mase_ant_media=("mase_ano_ant", "mean")).reset_index()
for v in ("confianca", "cambio", "ipca_12m", "selic"):
    ag[v] = [cb[v].get(o) for o in ag.origem]
ag["d_selic"] = [cb.selic.get(o) - cb.selic.get(o - pd.DateOffset(months=12)) for o in ag.origem]
A = A.merge(ag, on="origem")

# ---------- (A) os sinais antecipam anos ruins?
ano = A.groupby("ano").agg(MAPE_mediana=("ape_med", "mean"), mase_med=("Mediana", "mean"), est=("Estatísticos", "mean"), ia=("IA", "mean")).join(ag.assign(ano=ag.origem.dt.year + 1).set_index("ano").drop(columns="origem"))
ano["est_sobre_ia"] = ano.est / ano.ia
ano.round(3).to_csv(f"{OUT}/t4_sinais_por_ano.csv"); print(ano.round(2).to_string())
SIN = ["acel_media", "pct_queda", "mase_ant_media", "confianca", "cambio", "ipca_12m", "d_selic"]
cor = pd.DataFrame({"corr_com_erro_do_ano_seguinte": [spearmanr(ano[s], ano.mase_med, nan_policy="omit")[0] for s in SIN],
                    "p": [spearmanr(ano[s], ano.mase_med, nan_policy="omit")[1] for s in SIN],
                    "corr_com_vantagem_estatisticos": [spearmanr(ano[s], -np.log(ano.est_sobre_ia), nan_policy="omit")[0] for s in SIN]}, index=SIN)
cor.round(3).to_csv(f"{OUT}/t4_correlacao_sinais.csv"); print(cor.round(2).to_string())
# por série-ano: quintis da aceleração
A["q_acel"] = pd.qcut(A.acel, 5, labels=["1 (forte desaceleração)", "2", "3", "4", "5 (forte aceleração)"])
qa = A.groupby("q_acel", observed=True).agg(MAPE_mediana=("ape_med", "mean"), mase_mediana=("Mediana", "mean"), mase_est=("Estatísticos", "mean"), mase_ia=("IA", "mean"), n=("ape_med", "size"))
qa["est_sobre_ia"] = qa.mase_est / qa.mase_ia; qa.round(3).to_csv(f"{OUT}/t4_por_quintil_aceleracao.csv"); print(qa.round(3).to_string())

# ---------- (B) troca de pesos guiada por sinais, sem vazamento (treina nos anos < t, aplica em t)
FE = ["acel", "cresc3", "vol", "outl12", "erro_ano_ant", "acel_media", "pct_queda", "mase_ant_media", "confianca", "cambio", "ipca_12m", "d_selic"]
A["alvo"] = np.log(A["Estatísticos"] / A.IA)                                               # > 0: IA melhor
anos = sorted(A.ano.unique()); AV = [a for a in anos if a >= 2012]
A["w_ia"] = np.nan
for a in AV:
    tr = A[A.ano < a].dropna(subset=["alvo"]); te = A.ano == a
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=10, verbose=-1, n_jobs=2, random_state=0).fit(tr[FE], tr.alvo)
    A.loc[te, "w_ia"] = 1 / (1 + np.exp(-8 * m.predict(A.loc[te, FE])))                      # peso da IA entre 0 e 1
# regra simples, fixada a priori: alarme = mais da metade das séries em queda nos últimos 3 meses -> só estatísticos
A["alarme"] = A.pct_queda > 50
Wm = W.merge(A[["unique_id", "origem", "w_ia", "alarme", "ano"]], on=["unique_id", "origem"])
Wm["Troca aprendida (IA×estatísticos)"] = Wm.w_ia * Wm.IA + (1 - Wm.w_ia) * Wm["Estatísticos"]
Wm["Mediana + troca aprendida"] = (Wm.Mediana + Wm["Troca aprendida (IA×estatísticos)"]) / 2
Wm["Regra do alarme (estatísticos se >50% em queda)"] = np.where(Wm.alarme, Wm["Estatísticos"], Wm.Mediana)
Wm["Média fixa IA+estatísticos"] = (Wm.IA + Wm["Estatísticos"]) / 2
NOV = ["Mediana", "Estatísticos", "IA", "Média fixa IA+estatísticos", "Troca aprendida (IA×estatísticos)", "Mediana + troca aprendida", "Regra do alarme (estatísticos se >50% em queda)"]
lin = []
for rot, x in (("2012–2026, todos os anos", Wm[Wm.ano >= 2012]), ("2012–2026 sem 2020–21", Wm[(Wm.ano >= 2012) & ~Wm.ano.isin([2020, 2021])]),
               ("Só anos de virada (2015, 2016, 2017)", Wm[Wm.ano.isin([2015, 2016, 2017])]), ("Só anos com alarme", Wm[(Wm.ano >= 2012) & Wm.alarme])):
    mb = ((x.Mediana - x.y).abs() / x.escala).groupby(x.unique_id).mean()
    for m in NOV:
        ms = ((x[m] - x.y).abs() / x.escala).groupby(x.unique_id).mean(); r = np.log(ms / mb); ic = boot_ic(r.values)
        lin.append(dict(recorte=rot, metodo=m, MAPE=(100 * (x[m] - x.y).abs() / x.y).mean(), razao_vs_mediana=np.exp(r.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1])))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/t4_troca_de_pesos.csv", index=False); print(T.round(3).to_string())
print("anos com alarme:", sorted(A[A.alarme].ano.unique()))
pa = Wm[Wm.ano >= 2012].groupby("ano").apply(lambda x: pd.Series({m: (100 * (x[m] - x.y).abs() / x.y).mean() for m in NOV} | {"w_ia_medio": x.w_ia.mean(), "alarme": x.alarme.iloc[0]}))
pa.round(2).to_csv(f"{OUT}/t4_por_ano.csv"); print(pa.round(2).to_string())
A.to_parquet(f"{OUT}/t4_series_ano.parquet", index=False)
