"""Moirai 2.0 small em zero-shot (ambiente Python 3.11). Uso: python 03b_moirai.py"""
import os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from comum import carrega_painel, ORIGENS, H, QUANTIS, formato_longo, contexto, SUFIXO

torch.manual_seed(0)
torch.set_num_threads(2)
p = carrega_painel()
series = {u: g.set_index("data").valor for u, g in p.groupby("unique_id")}
modulo = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
CTX = 320  # cobre o histórico completo (máximo de 320 meses no painel)
model = Moirai2Forecast(module=modulo, prediction_length=H, context_length=CTX, target_dim=1,
                        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
pred = model.create_predictor(batch_size=64)

t0 = time.time()
saida = []
for o in ORIGENS:
    ids = [u for u, s in series.items() if s.index.min() <= o]
    ds = ListDataset([{"start": pd.Period(o, "M") - (len(contexto(series[u], o)) - 1), "target": contexto(series[u], o).astype("float32")}
                      for u in ids], freq="M")
    for u, fc in zip(ids, pred.predict(ds)):
        q = np.column_stack([fc.quantile(a) for a in QUANTIS])  # (H, 9)
        saida.append(formato_longo(u, o, q))
out = pd.concat(saida, ignore_index=True)
out["modelo"] = "Moirai 2.0 small"
out.to_csv(f"previsoes/Moirai_20_small{SUFIXO}.csv.gz", index=False)  # uni2ts exige numpy<2, incompatível com o pyarrow
print(f"Moirai: {len(out)} linhas em {time.time() - t0:.0f}s")
