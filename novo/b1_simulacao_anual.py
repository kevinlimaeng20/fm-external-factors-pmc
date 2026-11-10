"""Q6: "se tivéssemos usado estes modelos nos anos anteriores, teríamos acertado?"
Simulação histórica ano a ano: em cada dezembro (2004–2025) cada modelo usa só o histórico até ali e prevê
os 12 meses do ano seguinte. Saída: novo/saidas/anual_<modelo>.parquet (decis q10..q90).

Uso: python novo/b1_simulacao_anual.py estatisticos | lightgbm | chronos-bolt-small | chronos-bolt-base | timesfm | chronos2
"""
import os, sys, time
import numpy as np
import pandas as pd
from base import OUT, painel, QCOLS

H = 12
ORIGENS = pd.date_range("2004-12-01", "2025-12-01", freq="12MS")
MIN_HIST = 36
QUANTIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
qual = sys.argv[1]
p = painel()
series = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}
t0 = time.time()


def longo(u, o, Q, nome):
    Q = np.sort(np.asarray(Q, float), axis=1)
    df = pd.DataFrame(Q, columns=QCOLS)
    df.insert(0, "h", np.arange(1, H + 1)); df.insert(0, "origem", o); df.insert(0, "unique_id", u)
    df["data_alvo"] = [o + pd.DateOffset(months=int(h)) for h in df.h]
    df["modelo"] = nome
    return df


saida = []
if qual == "estatisticos":
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, AutoARIMA, SeasonalNaive
    mods = [SeasonalNaive(12), AutoETS(season_length=12), AutoTheta(season_length=12),
            AutoARIMA(season_length=12, max_p=2, max_q=2, max_P=1, max_Q=1, approximation=True)]
    nomes = {"SeasonalNaive": "Ingênuo sazonal", "AutoETS": "ETS", "AutoTheta": "Theta", "AutoARIMA": "ARIMA"}
    for o in ORIGENS:
        tr = p[p.data <= o]
        ok = tr.groupby("unique_id").size()
        tr = tr[tr.unique_id.isin(ok[ok >= MIN_HIST].index)][["unique_id", "data", "valor"]].rename(columns={"data": "ds", "valor": "y"})
        sf = StatsForecast(models=mods, freq="MS", n_jobs=2)
        f = sf.forecast(df=tr, h=H, level=[20, 40, 60, 80])
        for k, nome in nomes.items():
            Q = np.column_stack([f[f"{k}-lo-80"], f[f"{k}-lo-60"], f[f"{k}-lo-40"], f[f"{k}-lo-20"], f[k],
                                 f[f"{k}-hi-20"], f[f"{k}-hi-40"], f[f"{k}-hi-60"], f[f"{k}-hi-80"]])
            g = pd.DataFrame(Q, columns=QCOLS); g["unique_id"] = f.unique_id.values; g["data_alvo"] = pd.to_datetime(f.ds.values)
            g["origem"] = o; g["h"] = ((g.data_alvo.dt.year - o.year) * 12 + g.data_alvo.dt.month - o.month).astype(int); g["modelo"] = nome
            saida.append(g)
        print(o.date(), len(ok[ok >= MIN_HIST]), f"{time.time() - t0:.0f}s", flush=True)

elif qual == "lightgbm":
    sys.path.insert(0, "codigo")
    import json
    from lgb_base import X, FEATS, treino_ate, ajusta
    melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
    for o in ORIGENS:
        tr = treino_ate(o)
        alvo = X[X.origem == o].copy()
        if len(tr) < 2000 or alvo.empty:
            continue
        for q, c in zip(QUANTIS, QCOLS):
            mdl = ajusta(melhor, tr, objetivo="quantile", alpha=q)
            alvo[c] = np.exp(alvo.base_log + mdl.predict(alvo[FEATS]))
        alvo["modelo"] = "LightGBM"
        saida.append(alvo[["unique_id", "origem", "h", "data_alvo", "modelo"] + QCOLS])
        print(o.date(), len(tr), len(alvo), f"{time.time() - t0:.0f}s", flush=True)

else:
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if qual.startswith("chronos-bolt"):
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(f"amazon/{qual}", device_map="cpu")
        nome = {"chronos-bolt-small": "Chronos-Bolt P", "chronos-bolt-base": "Chronos-Bolt M"}[qual]
        def prev(ctxs):
            q, _ = pipe.predict_quantiles(inputs=[torch.tensor(c, dtype=torch.float32) for c in ctxs], prediction_length=H, quantile_levels=QUANTIS)
            return [x.numpy() for x in q]
    elif qual == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        nome = "Chronos-2"
        def prev(ctxs):
            q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in ctxs], prediction_length=H, quantile_levels=QUANTIS)
            return [np.asarray(x).reshape(-1, H, len(QUANTIS))[0] if np.asarray(x).ndim == 3 else np.asarray(x) for x in q]
    elif qual == "timesfm":
        import timesfm
        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True,
                                           force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        nome = "TimesFM 2.5"
        def prev(ctxs):
            _, qt = tfm.forecast(horizon=H, inputs=[np.asarray(c, float) for c in ctxs])
            return [qt[i][:, 1:10] for i in range(len(ctxs))]
    for o in ORIGENS:
        us, ctxs = [], []
        for u, s in series.items():
            c = s[:o].values
            if len(c) >= MIN_HIST:
                us.append(u); ctxs.append(c)
        for k in range(0, len(us), 64):
            res = prev(ctxs[k:k + 64])
            for u, Q in zip(us[k:k + 64], res):
                saida.append(longo(u, o, Q, nome))
        print(o.date(), len(us), f"{time.time() - t0:.0f}s", flush=True)

out = pd.concat(saida, ignore_index=True)
out.to_parquet(f"{OUT}/anual_{qual}.parquet", index=False)
print("ok", qual, len(out), f"{time.time() - t0:.0f}s")
