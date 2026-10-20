"""Previsões quantílicas do LightGBM global na validação (mar/2016-fev/2019), com os hiperparâmetros
escolhidos em 04_lightgbm.py e treino até a primeira origem de validação (alvos até mar/2016)."""
import json
import numpy as np
from lgb_base import X, FEATS, treino_ate, ajusta
from comum import ORIGENS_VALID, QUANTIS, QCOLS

melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
tr = treino_ate(ORIGENS_VALID[0])
alvo = X[X.origem.isin(ORIGENS_VALID)].copy()
for q, c in zip(QUANTIS, QCOLS):
    alvo[c] = np.exp(alvo.base_log + ajusta(melhor, tr, objetivo="quantile", alpha=q).predict(alvo[FEATS]))
out = alvo[["unique_id", "origem", "h", "data_alvo"] + QCOLS].copy()
out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
out["modelo"] = "LightGBM global"
out.to_parquet("previsoes/LightGBM_global_valid.parquet", index=False)
print("LightGBM validação:", len(out), "linhas")
