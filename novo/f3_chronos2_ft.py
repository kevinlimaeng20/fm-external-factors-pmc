"""Teste 3: Chronos-2 além do uso "de prateleira".
Uso: python novo/f3_chronos2_ft.py <cruzado|ft_uni|ft_cov> [passos]
  cruzado — sem treino, mas prevendo as 171 séries em conjunto (cross-learning);
  ft_uni  — ajuste fino (todos os pesos) nas 171 séries, só com dados até jul/2022, depois previsão das 38 origens;
  ft_cov  — ajuste fino com 6 fatores externos (4 passados, 2 conhecidos no futuro), alinhados à data de publicação.
Saída: novo/saidas/c2_<modo>.parquet"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from base import OUT, painel, QCOLS

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from chronos import Chronos2Pipeline
from comum import ORIGENS_TESTE, QUANTIS

modo = sys.argv[1]; PASSOS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
torch.set_num_threads(2); torch.manual_seed(0)
H = 12; CORTE = pd.Timestamp("2022-07-01")
p = painel(); series = {u: g.set_index("data").valor.astype("float32") for u, g in p.groupby("unique_id")}
geo = p.drop_duplicates("unique_id").set_index("unique_id").geo_cod.astype(int)
PASS = ["desocup", "confianca", "inadimpl_pf", "credito_pf"]; FUT = ["dias_uteis", "carnaval_mes"]
if modo == "ft_cov":
    cov = pd.read_parquet("dados/covariaveis.parquet"); info = pd.read_csv("dados/covariaveis_meta.csv").set_index("variavel")
    AL = {}
    for g, d in cov.groupby("geo_cod"):
        d = d.set_index("data").sort_index()
        AL[int(g)] = pd.DataFrame({v: d[v].shift(int(info.at[v, "defasagem"])) for v in PASS + FUT}).fillna(0.0).astype("float32")


def entrada(u, ate, futuro=False):
    s = series[u][:ate]
    if modo != "ft_cov":
        return s.values
    a = AL[geo[u]]
    e = {"target": s.values, "past_covariates": {v: a[v].reindex(s.index).fillna(0).values for v in PASS + FUT}}
    if futuro:
        fd = pd.date_range(ate + pd.DateOffset(months=1), periods=H, freq="MS")
        e["future_covariates"] = {v: a[v].reindex(fd).fillna(0).values for v in FUT}
    else:
        e["future_covariates"] = {v: None for v in FUT}
    return e


pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
t0 = time.time()
if modo.startswith("ft"):
    lote = 32 if modo == "ft_uni" else 4 * (1 + len(PASS + FUT))  # 4 séries por lote (limite de memória)
    pipe = pipe.fit([entrada(u, CORTE) for u in series], prediction_length=H, num_steps=PASSOS, batch_size=lote, learning_rate=5e-6,
                    output_dir=f"/tmp/claude-0/-home-claude/c5ed94fe-e163-5a1d-adc2-22270f90eded/scratchpad/c2_{modo}", logging_steps=50, report_to="none", disable_tqdm=True)
    print("ajuste fino:", f"{time.time() - t0:.0f}s", flush=True)
out = []; us = list(series)
for o in ORIGENS_TESTE:
    kw = dict(cross_learning=True, batch_size=256) if modo == "cruzado" else dict(batch_size=256)
    q, _ = pipe.predict_quantiles([entrada(u, o, futuro=True) for u in us], prediction_length=H, quantile_levels=QUANTIS, **kw)
    for u, Q in zip(us, q):
        g = pd.DataFrame(np.sort(np.asarray(Q).reshape(-1, H, len(QUANTIS))[0], axis=1), columns=QCOLS)
        g["unique_id"] = u; g["origem"] = o; g["h"] = np.arange(1, H + 1); g["data_alvo"] = [o + pd.DateOffset(months=int(h)) for h in g.h]
        out.append(g)
    print(o.date(), f"{time.time() - t0:.0f}s", flush=True)
nome = {"cruzado": "Chronos-2 conjunto", "ft_uni": "Chronos-2 ajustado", "ft_cov": "Chronos-2 ajustado + fatores"}[modo]
pd.concat(out, ignore_index=True).assign(modelo=nome).to_parquet(f"{OUT}/c2_{modo}.parquet", index=False); print("ok", modo)
