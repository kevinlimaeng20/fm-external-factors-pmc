"""Modelos estatísticos locais (statsforecast) em origem móvel: sazonal ingênuo, ETS, Theta e ARIMA.
Uso: python 02_estatisticos.py [n_series]   (n_series opcional, para testes de tempo)"""
import sys, time
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoTheta, AutoARIMA
import os
from comum import carrega_painel, ORIGENS, FIM_USO, H, QCOLS, SUFIXO

p = carrega_painel()
df = p.rename(columns={"data": "ds", "valor": "y"})[["unique_id", "ds", "y"]]
df = df[df.ds <= FIM_USO]
if len(sys.argv) > 1:
    df = df[df.unique_id.isin(sorted(df.unique_id.unique())[: int(sys.argv[1])])]

modelos = [SeasonalNaive(season_length=12, alias="Seasonal naive"),
           AutoETS(season_length=12, alias="ETS"),
           AutoTheta(season_length=12, alias="Theta"),
           AutoARIMA(season_length=12, max_p=2, max_q=2, max_P=1, max_Q=1, approximation=True, alias="ARIMA")]
if os.environ.get("FM_MODELOS"):
    modelos = [m for m in modelos if m.alias in os.environ["FM_MODELOS"].split(",")]
NIVEIS = [20, 40, 60, 80]
t0 = time.time()
sf = StatsForecast(models=modelos, freq="MS", n_jobs=2)
cv = sf.cross_validation(df=df, h=H, step_size=1, n_windows=len(ORIGENS), level=NIVEIS)
print(f"tempo: {time.time() - t0:.0f}s")
cv = cv.reset_index() if "unique_id" not in cv.columns else cv
assert cv.cutoff.min() == ORIGENS[0] and cv.cutoff.max() == ORIGENS[-1], (cv.cutoff.min(), cv.cutoff.max())

saida = []
for m in [m.alias for m in modelos]:
    d = cv[["unique_id", "cutoff", "ds"]].rename(columns={"cutoff": "origem", "ds": "data_alvo"}).copy()
    d["modelo"] = m
    # decis a partir dos intervalos centrais: nível L -> quantis 0,5 -/+ L/200
    mapa = {"q10": f"{m}-lo-80", "q20": f"{m}-lo-60", "q30": f"{m}-lo-40", "q40": f"{m}-lo-20", "q50": m,
            "q60": f"{m}-hi-20", "q70": f"{m}-hi-40", "q80": f"{m}-hi-60", "q90": f"{m}-hi-80"}
    for q, c in mapa.items():
        d[q] = cv[c].values
    saida.append(d)
out = pd.concat(saida, ignore_index=True)
out["h"] = ((out.data_alvo.dt.year - out.origem.dt.year) * 12 + out.data_alvo.dt.month - out.origem.dt.month)
out[QCOLS] = pd.DataFrame(sorted_ := __import__("numpy").sort(out[QCOLS].values, axis=1), index=out.index)
nome = f"previsoes/estatisticos{SUFIXO}.parquet" if len(sys.argv) == 1 else "previsoes/teste_estatisticos.parquet"
out.to_parquet(nome, index=False)
print(out.groupby("modelo").size(), "\nsalvo em", nome)
