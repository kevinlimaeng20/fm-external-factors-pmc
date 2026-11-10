"""Q7 (parte 1): onde estão os outliers e quanto pesam nos erros.
Detecção: decomposição STL robusta do log de cada série (período 12); resíduo padronizado de forma robusta
(z = resíduo / (1,4826·MAD)). |z| > 3: outlier; |z| > 5: outlier extremo.
Saídas: outliers_detectados.csv, outliers_por_ano.csv, outliers_eventos.csv, outliers_por_atividade.csv,
        outliers_efeito_erro.csv"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from base import OUT, painel, meta

p = painel(); M = meta(p)
lin = []
for u, g in p.groupby("unique_id"):
    s = np.log(g.set_index("data").valor)
    r = STL(s, period=12, robust=True).fit()
    res = r.resid
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    z = res / mad
    lin.append(pd.DataFrame({"unique_id": u, "data": s.index, "z": z.values, "resid_pct": 100 * (np.exp(res.values) - 1)}))
o = pd.concat(lin, ignore_index=True).join(M, on="unique_id")
o["outlier"] = o.z.abs() > 3
o["extremo"] = o.z.abs() > 5
o["pandemia"] = o.data.between("2020-03-01", "2021-12-01")
o.to_csv(f"{OUT}/outliers_detectados.csv", index=False)

print("meses-série:", len(o), "| outliers:", round(100 * o.outlier.mean(), 2), "% | extremos:", round(100 * o.extremo.mean(), 2), "%")
print("fora da pandemia:", round(100 * o[~o.pandemia].outlier.mean(), 2), "% | na pandemia:", round(100 * o[o.pandemia].outlier.mean(), 2), "%")
ano = o.groupby(o.data.dt.year).agg(pct_outliers=("outlier", "mean"), pct_extremos=("extremo", "mean")).mul(100)
ano.round(2).to_csv(f"{OUT}/outliers_por_ano.csv"); print(ano.round(1).T.to_string())
# eventos: meses com muitas séries atípicas ao mesmo tempo (choques sistêmicos)
ev = o.groupby("data").agg(series_atipicas=("outlier", "sum"), n=("outlier", "size"), desvio_medio_pct=("resid_pct", "median"))
ev["pct"] = 100 * ev.series_atipicas / ev.n
ev = ev.sort_values("pct", ascending=False).head(25)
ev.round(2).to_csv(f"{OUT}/outliers_eventos.csv"); print(ev.round(1).to_string())
at = o.groupby("ativ").agg(pct_outliers=("outlier", "mean"), pct_extremos=("extremo", "mean")).mul(100).sort_values("pct_outliers")
at.round(2).to_csv(f"{OUT}/outliers_por_atividade.csv"); print(at.round(1).to_string())
nv = o.groupby("nivel").outlier.mean().mul(100); print(nv.round(2))

# ---------------------------------------------------------------- peso dos outliers nos erros (período de teste)
d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
ol = o.set_index(["unique_id", "data"])
d["alvo_outlier"] = ol.outlier.reindex(list(zip(d.unique_id, d.data_alvo))).values
d["alvo_extremo"] = ol.extremo.reindex(list(zip(d.unique_id, d.data_alvo))).values
# outlier recente no histórico (últimos 12 meses antes da origem)
rec = o.sort_values("data").groupby("unique_id").apply(lambda g: g.set_index("data").outlier.astype(int).rolling(12, min_periods=1).sum())
rec = rec.reset_index().rename(columns={"outlier": "n_out12"}).set_index(["unique_id", "data"]).n_out12
d["hist_com_outlier"] = rec.reindex(list(zip(d.unique_id, d.origem))).values > 0
CH = ["Combinação (todos)", "LightGBM + fatores", "Chronos-2", "Moirai 2.0", "ARIMA", "ETS", "Ingênuo sazonal"]
x = d[d.modelo.isin(CH)]
efeito = pd.concat({
    "erro_% alvo normal": x[~x.alvo_outlier].groupby("modelo").ape.mean(),
    "erro_% alvo outlier": x[x.alvo_outlier].groupby("modelo").ape.mean(),
    "erro_% alvo extremo": x[x.alvo_extremo].groupby("modelo").ape.mean(),
    "erro_% histórico recente limpo": x[~x.hist_com_outlier].groupby("modelo").ape.mean(),
    "erro_% histórico recente com outlier": x[x.hist_com_outlier].groupby("modelo").ape.mean(),
}, axis=1).loc[CH]
efeito["pct_alvos_outlier"] = 100 * x.groupby("modelo").alvo_outlier.mean()
# quanto do erro total vem dos alvos outliers?
efeito["pct_do_erro_total_em_alvos_outlier"] = 100 * x.assign(ea=x.erro.abs()).groupby("modelo").apply(lambda g: g.ea[g.alvo_outlier].sum() / g.ea.sum())
# concentração: participação dos 5% maiores erros no erro total
efeito["pct_do_erro_nos_5%_maiores"] = 100 * x.assign(ea=x.erro.abs()).groupby("modelo").apply(lambda g: g.ea[g.ea >= g.ea.quantile(0.95)].sum() / g.ea.sum())
efeito.round(2).to_csv(f"{OUT}/outliers_efeito_erro.csv"); print(efeito.round(2).to_string())
