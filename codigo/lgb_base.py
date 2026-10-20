"""Base do LightGBM global: atributos e funções (usada por 04_lightgbm.py e 04b_lightgbm_valid.py).
LightGBM global (um modelo para todas as séries), estratégia direta com o horizonte como atributo.
Alvo: z = ln y[t+h] - ln y[t+h-12]; previsão: y[t+h-12] * exp(z), com y[t+h-12] observado (h <= 12).
1) Optuna (40 tentativas) escolhe hiperparâmetros na validação pré-pandemia (origens mar/2016-fev/2019).
2) Modelos quantílicos (decis) com esses hiperparâmetros, reajustados anualmente no teste."""
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from comum import carrega_painel, ORIGENS_TESTE, ORIGENS_VALID, H, QUANTIS, QCOLS

SEED = 42
PANDEMIA = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"))  # alvos excluídos do treino
p = carrega_painel()
meta = p.drop_duplicates("unique_id").set_index("unique_id")[["geo_cod", "ativ_cod"]].astype(str)
cat_geo = {g: i for i, g in enumerate(sorted(meta.geo_cod.unique()))}
cat_ativ = {a: i for i, a in enumerate(sorted(meta.ativ_cod.unique()))}


def fds(mes):  # número de sábados e domingos no mês
    d = pd.date_range(mes, mes + pd.offsets.MonthEnd(0))
    return int((d.dayofweek >= 5).sum())


MESES = pd.date_range("1999-01-01", "2028-12-01", freq="MS")
FDS = pd.Series([fds(m) for m in MESES], index=MESES)

blocos = []
for u, g in p.groupby("unique_id"):
    datas = pd.DatetimeIndex(g.data.values)
    L = np.log(g.valor.values)
    n = len(L)
    d = np.full(n, np.nan)
    d[12:] = L[12:] - L[:-12]
    for i in range(23, n):
        for h in range(1, H + 1):
            j = i + h
            alvo_data = datas[i] + pd.DateOffset(months=h)
            row = {"unique_id": u, "origem": datas[i], "h": h, "data_alvo": alvo_data,
                   "mes_alvo": alvo_data.month, "geo": cat_geo[meta.at[u, "geo_cod"]],
                   "ativ": cat_ativ[meta.at[u, "ativ_cod"]],
                   "fds_dif": FDS[alvo_data] - FDS[alvo_data - pd.DateOffset(months=12)],
                   "d_mesmo_mes_ano_ant": d[j - 12], "nivel_rel": L[i] - L[i - 11:i + 1].mean(),
                   "base_log": L[j - 12], "z": (L[j] - L[j - 12]) if j < n else np.nan}
            for k in range(12):
                row[f"d_lag{k}"] = d[i - k]
            row["d_media3"] = np.mean(d[i - 2:i + 1])
            row["d_media12"] = np.mean(d[i - 11:i + 1])
            blocos.append(row)
X = pd.DataFrame(blocos)
FEATS = ["h", "mes_alvo", "geo", "ativ", "fds_dif", "d_mesmo_mes_ano_ant", "nivel_rel", "d_media3", "d_media12"] + \
        [f"d_lag{k}" for k in range(12)]
CATS = ["geo", "ativ", "mes_alvo"]
print("linhas:", len(X))

# escala do MASE: EAM do sazonal ingênuo no histórico até a origem
escala = {}
for u, g in p.groupby("unique_id"):
    s = g.set_index("data").valor
    e = (s - s.shift(12)).abs()
    escala[u] = e.expanding().mean()
X["escala"] = [escala[u].get(o, np.nan) for u, o in zip(X.unique_id, X.origem)]
X["y_real"] = np.exp(X.base_log + X.z)


def treino_ate(corte):
    m = (X.data_alvo <= corte) & X.z.notna() & ~X.data_alvo.between(*PANDEMIA)
    return X[m]


def ajusta(params, dados, objetivo="l1", alpha=None):
    kw = dict(params, objective=objetivo, random_state=SEED, n_jobs=2, verbose=-1, bagging_freq=1)
    if alpha is not None:
        kw["alpha"] = alpha
    mdl = lgb.LGBMRegressor(**kw)
    mdl.fit(dados[FEATS], dados.z, categorical_feature=CATS)
    return mdl


def mase(df, pred_z):
    yhat = np.exp(df.base_log + pred_z)
    return float(np.mean(np.abs(df.y_real - yhat) / df.escala))


