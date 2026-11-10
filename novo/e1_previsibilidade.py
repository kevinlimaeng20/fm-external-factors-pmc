"""Pergunta extra: o que torna uma série fácil ou difícil de prever?
Características medidas só com o histórico até jun/2022 (antes do teste); erro = MAPE da combinação no teste."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from base import OUT, painel, meta

p = painel(); M = meta(p)
d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
err = d[d.modelo == "Combinação (todos)"].groupby("unique_id").ape.mean().rename("MAPE_comb")
ganho = (d[d.modelo == "Combinação (todos)"].groupby("unique_id").mase.mean() / d[d.modelo == "Ingênuo sazonal"].groupby("unique_id").mase.mean()).rename("erro_rel_ingenuo")
lin = []
for u, g in p.groupby("unique_id"):
    s = np.log(g.set_index("data").valor[:"2022-06-01"])
    r = STL(s, period=12, robust=True).fit()
    vR, vS, vT = np.var(r.resid), np.var(r.seasonal + r.resid), np.var(r.trend + r.resid)
    yoy = s.diff(12).dropna()
    rec = yoy["2015-01-01":]
    lin.append(dict(unique_id=u, forca_sazonal=max(0, 1 - vR / vS), forca_tendencia=max(0, 1 - vR / vT),
                    ruido_pct=100 * np.std(r.resid), volatilidade_crescimento_pct=100 * rec.std(),
                    amplitude_sazonal_pct=100 * (r.seasonal[-12:].max() - r.seasonal[-12:].min()),
                    anos_historico=len(s) / 12, brasil=int(M.at[u, "nivel"] == "Brasil")))
X = pd.DataFrame(lin).set_index("unique_id").join(err).join(ganho).join(M[["ativ", "serie"]])
X.round(4).to_csv(f"{OUT}/previsibilidade_series.csv")
feats = ["forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "amplitude_sazonal_pct", "anos_historico", "brasil"]
Z = (X[feats] - X[feats].mean()) / X[feats].std()
mod = sm.OLS(np.log(X.MAPE_comb), sm.add_constant(Z)).fit(cov_type="HC1")
print(mod.summary().tables[1]); print("R2:", round(mod.rsquared, 3))
coef = pd.DataFrame({"efeito_%_por_1dp": 100 * (np.exp(mod.params) - 1), "p_valor": mod.pvalues}).drop("const")
coef.round(3).to_csv(f"{OUT}/previsibilidade_regressao.csv")
print(X[feats + ["MAPE_comb"]].corr(method="spearman")["MAPE_comb"].round(2))
# quartis de ruído
X["quartil_ruido"] = pd.qcut(X.ruido_pct, 4, labels=["1 (menos ruído)", "2", "3", "4 (mais ruído)"])
q = X.groupby("quartil_ruido", observed=True).agg(MAPE_comb=("MAPE_comb", "mean"), erro_rel_ingenuo=("erro_rel_ingenuo", "mean"), ruido=("ruido_pct", "mean"), n=("MAPE_comb", "size"))
q.round(2).to_csv(f"{OUT}/previsibilidade_quartis.csv"); print(q.round(2))
print(X.sort_values("MAPE_comb")[["serie", "MAPE_comb", "ruido_pct"]].head(5).round(2).to_string())
print(X.sort_values("MAPE_comb")[["serie", "MAPE_comb", "ruido_pct"]].tail(5).round(2).to_string())
