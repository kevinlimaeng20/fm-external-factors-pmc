#!/bin/bash
# Full pipeline. Run from the repository root:  bash codigo/roda_tudo.sh
# Environments: venv (requirements.txt, Python 3.13) and venv_moirai (requirements-moirai.txt, Python 3.11).
# Runtimes on a 2-core CPU without GPU: about 10-12 hours in total (TimesFM ~53 min per run, Chronos-2 with
# covariates ~70 min per run, LightGBM with factors ~30 min per variant).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}; PYM=${PYM:-venv_moirai/bin/python}
export HF_HUB_DISABLE_XET=1
moirai() {   # Moirai runs in its own environment and writes csv.gz; convert to parquet
  $PYM codigo/03b_moirai.py
  for f in previsoes/Moirai_20_small*.csv.gz; do
    $PY -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1],parse_dates=['origem','data_alvo']); d.to_parquet(sys.argv[1].replace('.csv.gz','.parquet'),index=False)" "$f" && rm "$f"
  done
}
fm() { for m in "$@"; do $PY codigo/03_fundacionais.py "$m"; done; moirai; }

# 1) Sales panel (uses the saved SIDRA files; add --baixar to download them again)
$PY codigo/01_dados.py
# 2) Test period: statistical models, foundation models, LightGBM (Optuna search + annual refits)
$PY codigo/02_estatisticos.py
fm "Chronos-Bolt small" "Chronos-Bolt base" "TimesFM 2.5"
$PY codigo/04_lightgbm.py
# 3) Validation period (combination members) and robustness variants
export FM_ORIGENS=valid
$PY codigo/02_estatisticos.py; fm "Chronos-Bolt small" "Chronos-Bolt base" "TimesFM 2.5"; $PY codigo/04b_lightgbm_valid.py
unset FM_ORIGENS
export FM_IMPUTA=1; FM_MODELOS=ETS,ARIMA $PY codigo/02_estatisticos.py; fm "Chronos-Bolt base" "TimesFM 2.5"; unset FM_IMPUTA
for c in 60 120; do export FM_CTX=$c; fm "Chronos-Bolt base" "TimesFM 2.5"; done; unset FM_CTX
# 4) Evaluation, combinations, robustness, decision layer, figures and tables (base study)
$PY codigo/05_avaliacao.py
$PY codigo/06_combinacoes_robustez.py
$PY codigo/07_decisao.py
$PY codigo/08_figuras.py
$PY codigo/09_tabelas.py
# 5) External factors (saved raw files; run 10a_baixa_fatores.py to download them again)
$PY codigo/10_covariaveis.py
$PY codigo/11_lightgbm_x.py real
$PY codigo/11_lightgbm_x.py oraculo
$PY codigo/11_lightgbm_x.py ablacao
$PY codigo/11_lightgbm_x.py shap
$PY codigo/11_lightgbm_x.py valid
$PY codigo/11_lightgbm_x.py selecionado
for m in uni real sel; do $PY codigo/12_chronos2.py $m; done
$PY codigo/13_avaliacao_fatores.py
$PY codigo/14_figuras_fatores.py
$PY codigo/15_tabelas_fatores.py
echo "done"
