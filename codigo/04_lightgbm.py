"""LightGBM global: busca de hiperparâmetros na validação e teste com reajuste anual (ver lgb_base.py)."""
import json, time
import numpy as np
import pandas as pd
import optuna
from lgb_base import *  # noqa: F401,F403  (X, FEATS, treino_ate, ajusta, mase, SEED ...)
from comum import ORIGENS_TESTE, ORIGENS_VALID, QUANTIS, QCOLS

# ---------------------------------------------------------------- 1) busca de hiperparâmetros
treino_v = treino_ate(ORIGENS_VALID[0])
valid = X[X.origem.isin(ORIGENS_VALID) & X.z.notna()]
print("treino validação:", len(treino_v), "| validação:", len(valid))


def objetivo(trial):
    params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                  min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                  subsample=trial.suggest_float("subsample", 0.5, 1.0),
                  reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    mdl = ajusta(params, treino_v)
    return mase(valid, mdl.predict(valid[FEATS]))


t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
estudo.optimize(objetivo, n_trials=40)
melhor = estudo.best_params
print(f"Optuna: MASE validação {estudo.best_value:.4f} em {time.time() - t0:.0f}s; parâmetros {melhor}")
estudo.trials_dataframe().to_csv("previsoes/lightgbm_optuna_trials.csv", index=False)
json.dump({"melhor": melhor, "mase_validacao": estudo.best_value}, open("previsoes/lightgbm_hiperparametros.json", "w"), indent=1)

# ---------------------------------------------------------------- 2) teste com reajuste anual
refits = [o for o in ORIGENS_TESTE if o.month == 7] or [ORIGENS_TESTE[0]]
saida = []
for k, r in enumerate(refits):
    fim = refits[k + 1] if k + 1 < len(refits) else ORIGENS_TESTE[-1] + pd.DateOffset(months=1)
    tr = treino_ate(r)
    alvo = X[X.origem.isin([o for o in ORIGENS_TESTE if r <= o < fim])].copy()
    t0 = time.time()
    for q, c in zip(QUANTIS, QCOLS):
        mdl = ajusta(melhor, tr, objetivo="quantile", alpha=q)
        alvo[c] = np.exp(alvo.base_log + mdl.predict(alvo[FEATS]))
    print(f"reajuste {r.date()}: treino {len(tr)}, previsões {len(alvo)}, {time.time() - t0:.0f}s")
    saida.append(alvo[["unique_id", "origem", "h", "data_alvo"] + QCOLS])
out = pd.concat(saida, ignore_index=True)
out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
out["modelo"] = "LightGBM global"
out.to_parquet("previsoes/LightGBM_global.parquet", index=False)
print("LightGBM:", len(out), "linhas")
