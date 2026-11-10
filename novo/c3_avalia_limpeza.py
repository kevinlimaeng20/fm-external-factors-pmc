"""Q7 (parte 3): compara previsões com e sem limpeza de outliers no histórico (mesmos modelos, mesmas origens).
Uso: python novo/c3_avalia_limpeza.py  (lê todos os anual_limpo_*.parquet disponíveis)"""
import glob, re
import numpy as np
import pandas as pd
from base import OUT, painel, QCOLS

p = painel(); real = p.set_index(["unique_id", "data"]).valor
base = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{OUT}/anual_*.parquet")
                  if re.search(r"anual_(estatisticos|chronos-bolt-small)\.parquet$", f)], ignore_index=True)
REG = {2005: "2005–08", 2006: "2005–08", 2007: "2005–08", 2008: "2005–08", 2009: "2009", **{a: "2010–14" for a in range(2010, 2015)},
       2015: "2015–16", 2016: "2015–16", 2017: "2017–19", 2018: "2017–19", 2019: "2017–19", 2020: "2020–21", 2021: "2020–21",
       **{a: "2022–26" for a in range(2022, 2027)}}


def prep(d):
    d = d.copy(); d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
    d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
    d = d.dropna(subset=["y"]); d["ano"] = d.data_alvo.dt.year
    d["ape"] = 100 * (d.q50 - d.y).abs() / d.y
    d["dentro80"] = (d.y >= d[QCOLS].min(axis=1)) & (d.y <= d[QCOLS].max(axis=1))
    return d


B = prep(base)
lin_g, lin_r = [], []
for f in sorted(glob.glob(f"{OUT}/anual_limpo_*.parquet")):
    tag = re.search(r"anual_limpo_(.+)\.parquet", f).group(1)
    L = prep(pd.read_parquet(f))
    k = ["unique_id", "origem", "h", "modelo"]
    m = B[k + ["ano", "ape", "dentro80"]].merge(L[k + ["ape", "dentro80"]], on=k, suffixes=("_orig", "_limpo"))
    m["regime"] = m.ano.map(REG)
    # combinação (mediana dos modelos) também com/sem limpeza, se houver mais de um modelo
    for mod, g in list(m.groupby("modelo")) + [("Todos", m)]:
        x = g[~g.ano.isin([2020, 2021])]
        # razão pareada por série: MAPE limpo / MAPE original
        rs = x.groupby("unique_id")[["ape_orig", "ape_limpo"]].mean()
        lin_g.append(dict(limiar=tag, modelo=mod, MAPE_orig=x.ape_orig.mean(), MAPE_limpo=x.ape_limpo.mean(),
                          ganho_pct=100 * (1 - x.ape_limpo.mean() / x.ape_orig.mean()),
                          series_que_melhoram_pct=100 * (rs.ape_limpo < rs.ape_orig).mean(),
                          cob80_orig=100 * x.dentro80_orig.mean(), cob80_limpo=100 * x.dentro80_limpo.mean()))
        for r, y in g.groupby("regime"):
            lin_r.append(dict(limiar=tag, modelo=mod, regime=r, MAPE_orig=y.ape_orig.mean(), MAPE_limpo=y.ape_limpo.mean(),
                              ganho_pct=100 * (1 - y.ape_limpo.mean() / y.ape_orig.mean())))
G = pd.DataFrame(lin_g); R = pd.DataFrame(lin_r)
G.round(2).to_csv(f"{OUT}/q7_limpeza_geral.csv", index=False); R.round(2).to_csv(f"{OUT}/q7_limpeza_por_regime.csv", index=False)
print(G.round(2).to_string()); print(R[R.modelo == "Todos"].pivot(index="regime", columns="limiar", values="ganho_pct").round(1).to_string())
