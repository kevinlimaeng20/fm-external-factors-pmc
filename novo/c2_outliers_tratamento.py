"""Q7 (parte 2): tratar os outliers do histórico melhora a previsão?
Na simulação ano a ano (origens dez/2004–dez/2025), antes de ajustar cada modelo os outliers do histórico
são detectados SÓ com dados até a origem (STL robusta no log, |z| > limite) e substituídos pelo valor
"esperado" (tendência + sazonalidade da STL). Compara com os mesmos modelos sem tratamento (b1).
Uso: python novo/c2_outliers_tratamento.py <limite_z> [modelos: estat | chronos]"""
import os, sys, time
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from base import OUT, painel, QCOLS

LIM = float(sys.argv[1]); QUAL = sys.argv[2] if len(sys.argv) > 2 else "estat"
H = 12; MIN_HIST = 36
ORIGENS = pd.date_range("2004-12-01", "2025-12-01", freq="12MS")
p = painel()
series = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}
t0 = time.time()


def limpa(s):
    ls = np.log(s)
    r = STL(ls, period=12, robust=True).fit()
    res = r.resid; mad = np.median(np.abs(res - np.median(res))) * 1.4826
    z = res / mad
    out = ls.where(z.abs() <= LIM, r.trend + r.seasonal)
    return np.exp(out), int((z.abs() > LIM).sum())


saida, contagem = [], []
if QUAL == "estat":
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, AutoARIMA
    mods = [AutoETS(season_length=12), AutoTheta(season_length=12),
            AutoARIMA(season_length=12, max_p=2, max_q=2, max_P=1, max_Q=1, approximation=True)]
    nomes = {"AutoETS": "ETS", "AutoTheta": "Theta", "AutoARIMA": "ARIMA"}
else:
    import torch
    from chronos import BaseChronosPipeline
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
for o in ORIGENS:
    lin = []
    for u, s in series.items():
        c = s[:o]
        if len(c) < MIN_HIST:
            continue
        cl, n = limpa(c)
        contagem.append(dict(origem=o, unique_id=u, n_outliers=n, n=len(c)))
        lin.append(pd.DataFrame({"unique_id": u, "ds": cl.index, "y": cl.values}))
    tr = pd.concat(lin, ignore_index=True)
    if QUAL == "estat":
        f = StatsForecast(models=mods, freq="MS", n_jobs=2).forecast(df=tr, h=H, level=[20, 40, 60, 80])
        for k, nome in nomes.items():
            Q = np.sort(np.column_stack([f[f"{k}-lo-80"], f[f"{k}-lo-60"], f[f"{k}-lo-40"], f[f"{k}-lo-20"], f[k],
                                         f[f"{k}-hi-20"], f[f"{k}-hi-40"], f[f"{k}-hi-60"], f[f"{k}-hi-80"]]), axis=1)
            g = pd.DataFrame(Q, columns=QCOLS); g["unique_id"] = f.unique_id.values; g["data_alvo"] = pd.to_datetime(f.ds.values)
            g["origem"] = o; g["h"] = ((g.data_alvo.dt.year - o.year) * 12 + g.data_alvo.dt.month - o.month).astype(int)
            g["modelo"] = nome; saida.append(g)
    else:
        us = tr.unique_id.unique().tolist()
        for k in range(0, len(us), 64):
            ctx = [torch.tensor(tr[tr.unique_id == u].y.values, dtype=torch.float32) for u in us[k:k + 64]]
            q, _ = pipe.predict_quantiles(inputs=ctx, prediction_length=H, quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            for u, Q in zip(us[k:k + 64], q):
                g = pd.DataFrame(np.sort(Q.numpy(), axis=1), columns=QCOLS); g["unique_id"] = u; g["origem"] = o; g["h"] = np.arange(1, H + 1)
                g["data_alvo"] = [o + pd.DateOffset(months=int(h)) for h in g.h]; g["modelo"] = "Chronos-Bolt P"; saida.append(g)
    print(o.date(), f"{time.time() - t0:.0f}s", flush=True)
tag = f"z{LIM:g}_{QUAL}"
pd.concat(saida, ignore_index=True).to_parquet(f"{OUT}/anual_limpo_{tag}.parquet", index=False)
pd.DataFrame(contagem).to_csv(f"{OUT}/anual_limpo_{tag}_contagem.csv", index=False)
print("ok", tag)
