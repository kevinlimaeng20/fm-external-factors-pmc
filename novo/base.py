"""Funções comuns do estudo novo (análise de dados; sem formato de artigo).
Carrega o painel e as previsões do período de teste (38 origens, h = 1..12) de todos os modelos,
monta as combinações e calcula medidas de erro fáceis de interpretar (erro percentual) e as técnicas (MASE)."""
import json, os, sys
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "codigo"))
os.chdir(RAIZ)
from comum import carrega_painel, QCOLS  # noqa: E402

OUT = os.path.join(RAIZ, "novo", "saidas")
os.makedirs(OUT, exist_ok=True)

NOMES = {  # nome curto em português
    "Seasonal naive": "Ingênuo sazonal", "ETS": "ETS", "Theta": "Theta", "ARIMA": "ARIMA",
    "LightGBM global": "LightGBM", "Chronos-Bolt small": "Chronos-Bolt P", "Chronos-Bolt base": "Chronos-Bolt M",
    "TimesFM 2.5": "TimesFM 2.5", "Moirai 2.0 small": "Moirai 2.0", "Chronos-2": "Chronos-2",
    "LightGBM + external factors": "LightGBM + fatores", "LightGBM + selected factors": "LightGBM + fatores sel.",
    "Combination: all models": "Combinação (todos)", "Combination: top-3 in validation": "Combinação (top-3)",
    "Combination: foundation models": "Combinação (pré-treinados)",
}
FAMILIA = {"Ingênuo sazonal": "Referência", "ETS": "Estatístico", "Theta": "Estatístico", "ARIMA": "Estatístico",
           "LightGBM": "IA treinada (ML)", "LightGBM + fatores": "IA treinada (ML)", "LightGBM + fatores sel.": "IA treinada (ML)",
           "Chronos-Bolt P": "IA pré-treinada", "Chronos-Bolt M": "IA pré-treinada", "TimesFM 2.5": "IA pré-treinada",
           "Moirai 2.0": "IA pré-treinada", "Chronos-2": "IA pré-treinada",
           "Combinação (todos)": "Combinação", "Combinação (top-3)": "Combinação", "Combinação (pré-treinados)": "Combinação"}

ARQ_TESTE = ["estatisticos.parquet", "LightGBM_global.parquet", "Chronos-Bolt_small.parquet", "Chronos-Bolt_base.parquet",
             "TimesFM_25.parquet", "Moirai_20_small.parquet", "Chronos-2_uni.parquet", "LightGBM_X_real.parquet",
             "LightGBM_X_selecionado.parquet"]


def painel():
    p = carrega_painel(imputa=False)
    return p


def meta(p):
    m = p.drop_duplicates("unique_id").set_index("unique_id")[["serie", "geo", "geo_cod", "ativ"]]
    m["nivel"] = np.where(m.geo == "Brasil", "Brasil", "Estado")
    return m


def carrega_previsoes(p=None, combinar=True):
    p = painel() if p is None else p
    ds = []
    for f in ARQ_TESTE:
        d = pd.read_parquet(f"previsoes/{f}")
        ds.append(d)
    d = pd.concat(ds, ignore_index=True)
    d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
    d = d[["unique_id", "origem", "h", "data_alvo", "modelo"] + QCOLS]
    if combinar:
        memb = json.load(open("resultados/combinacoes.json"))["membros"]
        for nome, mm in memb.items():
            c = d[d.modelo.isin(mm)].groupby(["unique_id", "origem", "h", "data_alvo"])[QCOLS].median().reset_index()
            c["modelo"] = nome
            d = pd.concat([d, c], ignore_index=True)
    d["modelo"] = d.modelo.map(NOMES).fillna(d.modelo)
    d[QCOLS] = np.sort(d[QCOLS].values, axis=1)
    # mantém só o que todos os modelos têm
    n = d.modelo.nunique()
    k = d.groupby(["unique_id", "origem", "h"]).modelo.nunique()
    d = d.merge(k[k == n].reset_index()[["unique_id", "origem", "h"]], on=["unique_id", "origem", "h"])
    real = p.set_index(["unique_id", "data"]).valor
    d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
    esc = {u: (g.set_index("data").valor - g.set_index("data").valor.shift(12)).abs().expanding().mean() for u, g in p.groupby("unique_id")}
    d["escala"] = [esc[u].get(o) for u, o in zip(d.unique_id, d.origem)]
    d["erro"] = d.q50 - d.y
    d["ape"] = 100 * (d.erro.abs() / d.y)
    d["epct"] = 100 * d.erro / d.y
    d["mase"] = d.erro.abs() / d.escala
    d["dentro80"] = (d.y >= d.q10) & (d.y <= d.q90)
    return d


def boot_ic(v, n=1000, f=np.mean, seed=2026):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    b = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])
