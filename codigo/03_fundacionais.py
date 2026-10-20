"""Modelos fundacionais em zero-shot: Chronos-Bolt (small, base) e TimesFM 2.5.
Cada origem recebe todo o histórico disponível até ela. Uso: python 03_fundacionais.py <modelo>"""
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import pandas as pd
import torch
from comum import carrega_painel, ORIGENS, H, QUANTIS, formato_longo, contexto, SUFIXO

torch.manual_seed(0)
torch.set_num_threads(2)
nome = sys.argv[1]
p = carrega_painel()
series = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}

if nome.startswith("Chronos"):
    from chronos import BaseChronosPipeline
    repo = {"Chronos-Bolt small": "amazon/chronos-bolt-small", "Chronos-Bolt base": "amazon/chronos-bolt-base"}[nome]
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)

    def prever(contextos):
        q, _ = pipe.predict_quantiles(inputs=[torch.tensor(c, dtype=torch.float32) for c in contextos],
                                      prediction_length=H, quantile_levels=QUANTIS)
        return q.numpy()  # (n, H, 9)
elif nome == "TimesFM 2.5":
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))

    def prever(contextos):
        _, qt = m.forecast(horizon=H, inputs=[np.asarray(c, dtype=float) for c in contextos])
        return qt[:, :, 1:10]  # coluna 0 é a média; 1..9 são os decis 0,1..0,9
else:
    raise SystemExit(f"modelo desconhecido: {nome}")

t0 = time.time()
saida = []
for o in ORIGENS:
    ids = [u for u, s in series.items() if s.index.min() <= o]
    ctx = [contexto(series[u], o) for u in ids]
    Q = prever(ctx)
    for u, q in zip(ids, Q):
        saida.append(formato_longo(u, o, q))
out = pd.concat(saida, ignore_index=True)
out["modelo"] = nome
os.makedirs("previsoes", exist_ok=True)
arq = "previsoes/" + nome.replace(" ", "_").replace(".", "") + SUFIXO + ".parquet"
out.to_parquet(arq, index=False)
print(f"{nome}: {len(out)} linhas em {time.time() - t0:.0f}s -> {arq}")
