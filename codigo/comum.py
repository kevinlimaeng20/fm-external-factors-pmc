"""Definições comuns do experimento (painel, origens, quantis) e variantes de robustez.

Variáveis de ambiente (todas opcionais):
  FM_ORIGENS = teste (padrão) | valid   -> conjunto de origens
  FM_CTX     = número máximo de meses de contexto dos modelos fundacionais (padrão: histórico completo)
  FM_IMPUTA  = 1 para substituir mar/2020-dez/2021 por valores imputados no histórico dado aos modelos
"""
import os
import numpy as np
import pandas as pd

PAINEL = "dados/painel_pmc.csv"
H = 12
QUANTIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{int(round(q * 100)):02d}" for q in QUANTIS]
# teste: 38 origens mensais, todas com os 12 horizontes observados (último alvo = ago/2026)
ORIGENS_TESTE = pd.date_range("2022-07-01", "2025-08-01", freq="MS")
# validação: alvos anteriores à pandemia (hiperparâmetros do LightGBM e escolha das combinações)
ORIGENS_VALID = pd.date_range("2016-03-01", "2019-02-01", freq="MS")
FIM_DADOS = pd.Timestamp("2026-08-01")
PANDEMIA = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"))

CONJ = os.environ.get("FM_ORIGENS", "teste")
ORIGENS = ORIGENS_VALID if CONJ == "valid" else ORIGENS_TESTE
FIM_USO = ORIGENS[-1] + pd.DateOffset(months=H)
CTX = int(os.environ["FM_CTX"]) if os.environ.get("FM_CTX") else None
IMPUTA = os.environ.get("FM_IMPUTA") == "1"
SUFIXO = ("" if CONJ == "teste" else "_valid") + (f"_ctx{CTX}" if CTX else "") + ("_imputado" if IMPUTA else "")


def imputa_pandemia(s):
    """Substitui mar/2020-dez/2021 pela interpolação linear, em log, da série dessazonalizada; fatores
    sazonais = desvio médio do log em relação à média móvel centrada de 12 meses em 2015-2019."""
    ls = np.log(s)
    mm = ls.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    dev = (ls - mm)["2015-01-01":"2019-12-01"]
    fator = dev.groupby(dev.index.month).mean()
    fator -= fator.mean()
    saz = pd.Series(fator.reindex(ls.index.month).values, index=ls.index)
    dess = ls - saz
    dess[PANDEMIA[0]:PANDEMIA[1]] = np.nan
    dess = dess.interpolate(method="linear", limit_area="inside")
    out = np.exp(dess + saz)
    return s.where(~s.index.to_series().between(*PANDEMIA), out)


def carrega_painel(imputa=None):
    imputa = IMPUTA if imputa is None else imputa
    p = pd.read_csv(PAINEL, parse_dates=["data"])
    p = p[p.data <= FIM_DADOS].sort_values(["serie", "data"])
    ids = {s: f"s{i:03d}" for i, s in enumerate(sorted(p.serie.unique()))}
    p["unique_id"] = p.serie.map(ids)
    if imputa:
        p["valor"] = (p.groupby("unique_id", group_keys=False)
                      .apply(lambda g: imputa_pandemia(g.set_index("data").valor).set_axis(g.index)))
    return p


def contexto(serie, origem):
    """Histórico até a origem (inclusive), opcionalmente truncado aos últimos CTX meses."""
    c = serie[:origem].values
    return c[-CTX:] if CTX else c


def formato_longo(unique_id, origem, matriz_quantis):
    """matriz_quantis: array (H, 9) com os decis em ordem crescente."""
    m = np.sort(np.asarray(matriz_quantis, dtype=float), axis=1)  # evita cruzamento de quantis
    df = pd.DataFrame(m, columns=QCOLS)
    df.insert(0, "h", np.arange(1, H + 1))
    df.insert(0, "origem", origem)
    df.insert(0, "unique_id", unique_id)
    df["data_alvo"] = [origem + pd.DateOffset(months=int(h)) for h in df.h]
    return df
