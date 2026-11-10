"""Pergunta extra: quanto histórico é preciso? Experimento controlado: mesmas séries e mesmas origens,
variando só o tamanho do histórico entregue ao modelo (3, 5, 8, 12 anos e tudo).
Origens: dez/2016–2018 e dez/2022–2024 (alvos fora da pandemia). Modelos: ETS, Theta, Chronos-Bolt P, Chronos-2."""
import os, time
import numpy as np
import pandas as pd
import torch
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta
from base import OUT, painel, QCOLS

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from chronos import BaseChronosPipeline, Chronos2Pipeline

H = 12; ORIG = pd.to_datetime(["2016-12-01", "2017-12-01", "2018-12-01", "2022-12-01", "2023-12-01", "2024-12-01"])
ANOS = [3, 5, 8, 12, 99]
p = painel(); series = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}
real = p.set_index(["unique_id", "data"]).valor
# só séries com pelo menos 12 anos em todas as origens, para a comparação ser justa
ok = [u for u, s in series.items() if (ORIG.min() - s.index.min()).days / 365.25 >= 12]
bolt = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
c2 = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
out = []; t0 = time.time()
for o in ORIG:
    for a in ANOS:
        ctx = {u: series[u][:o].iloc[-12 * a:] for u in ok}
        tr = pd.concat([pd.DataFrame({"unique_id": u, "ds": s.index, "y": s.values}) for u, s in ctx.items()])
        f = StatsForecast(models=[AutoETS(season_length=12), AutoTheta(season_length=12)], freq="MS", n_jobs=2).forecast(df=tr, h=H)
        for k, nome in (("AutoETS", "ETS"), ("AutoTheta", "Theta")):
            out.append(pd.DataFrame({"unique_id": f.unique_id.values, "data_alvo": pd.to_datetime(f.ds.values), "q50": f[k].values, "modelo": nome}).assign(origem=o, anos=a))
        ts = [torch.tensor(ctx[u].values, dtype=torch.float32) for u in ok]
        q, _ = bolt.predict_quantiles(inputs=ts, prediction_length=H, quantile_levels=QL)
        q2, _ = c2.predict_quantiles(ts, prediction_length=H, quantile_levels=QL)
        q2 = [np.asarray(x).reshape(-1, H, len(QL))[0] for x in q2]
        for nome, qq in (("Chronos-Bolt P", [x for x in q]), ("Chronos-2", q2)):
            for u, Q in zip(ok, qq):
                Q = Q.numpy() if hasattr(Q, "numpy") else np.asarray(Q)
                out.append(pd.DataFrame({"unique_id": u, "data_alvo": [o + pd.DateOffset(months=h) for h in range(1, H + 1)],
                                         "q50": Q[:, 4], "modelo": nome}).assign(origem=o, anos=a))
        print(o.date(), a, f"{time.time() - t0:.0f}s", flush=True)
d = pd.concat(out, ignore_index=True)
d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
d = d.dropna(subset=["y"]); d["ape"] = 100 * (d.q50 - d.y).abs() / d.y
d.to_parquet(f"{OUT}/curva_historico.parquet", index=False)
t = d.pivot_table(index="anos", columns="modelo", values="ape", aggfunc="mean")
t.index = t.index.map(lambda a: "tudo" if a == 99 else f"{a} anos")
t.round(2).to_csv(f"{OUT}/extra_curva_historico.csv"); print(t.round(2).to_string(), "\nséries:", len(ok))
