# Foundation models, external factors and production planning decisions

Reproduction package for the study **"Foundation models, external factors and production planning decisions: Evidence from 171 Brazilian retail demand series"** (Francisco Kevin Lima Alves, Independent Researcher, Brazil).

The study compares time series foundation models used zero-shot (Chronos-Bolt small and base, TimesFM 2.5, Moirai 2.0 and Chronos-2) with statistical methods (seasonal naive, ETS, Theta, ARIMA), a tuned global LightGBM model and forecast combinations. It uses 171 monthly retail sales volume series from the Brazilian Monthly Trade Survey (PMC/IBGE, January 2000 to August 2026) and a rolling-origin design with 38 origins and horizons of 1 to 12 months. Each probabilistic forecast (nine deciles) becomes a newsvendor commitment, which is evaluated on realised inventory cost, service level, lead time and the December peak. In addition, 28 external factors (economy, policy, world, climate and calendar) are aligned with their publication lags and added to LightGBM and Chronos-2.

## Contents

| Folder | Content |
|---|---|
| `codigo/` | All scripts, numbered in execution order, and `roda_tudo.sh` (full pipeline) |
| `dados/` | Raw SIDRA files for the sales data (extracted 17 Oct 2026), the panel `painel_pmc.csv`, raw external-factor files in `dados/ext/` (extracted 19 Oct 2026) and the processed factor panel `covariaveis.parquet` |
| `previsoes/` | Decile forecasts of every method and variant: test, validation, robustness and external-factor runs |
| `resultados/` | Result tables (CSV/JSON) and SHAP contributions |
| `figuras/`, `tabelas/` | Figures (PDF/EPS) and LaTeX table bodies used in the manuscript |
| `logs/` | Execution logs, including runtimes |

The two largest intermediate files, `resultados/decisoes.parquet` and `resultados/previsoes_avaliadas.parquet`, are not included because of their size. `07_decisao.py` and `05_avaliacao.py` regenerate them in a few minutes.

## Scripts

| Script | Purpose |
|---|---|
| `01_dados.py` | Builds the 171-series panel from SIDRA tables 8882 and 8880 (`--baixar` downloads them again) |
| `02_estatisticos.py` | Seasonal naive, ETS, Theta and ARIMA (StatsForecast) |
| `03_fundacionais.py`, `03b_moirai.py` | Chronos-Bolt, TimesFM 2.5 and Moirai 2.0, zero-shot |
| `04_lightgbm.py`, `04b_lightgbm_valid.py`, `lgb_base.py` | Global LightGBM: Optuna search on the validation period, annual refits on the test period |
| `05`–`09` | Evaluation, combinations, robustness, newsvendor decision layer, figures 1–4 and tables |
| `10a_baixa_fatores.py`, `10_covariaveis.py` | Download and build the external-factor panel with publication lags |
| `11_lightgbm_x.py` | LightGBM with factors: `real`, `oraculo`, `ablacao`, `shap`, `valid` (group selection), `selecionado` |
| `12_chronos2.py` | Chronos-2 without covariates (`uni`), with factors (`real`) and with selected groups (`sel`) |
| `13`–`15` | Evaluation of the external-factor models, figures 5–6 and tables |

`comum.py` defines the origins, the quantiles and the robustness variants, which are controlled by the environment variables `FM_ORIGENS`, `FM_CTX` and `FM_IMPUTA`.

## How to reproduce

```bash
python3.13 -m venv venv && venv/bin/pip install -r requirements.txt
python3.11 -m venv venv_moirai && venv_moirai/bin/pip install -r requirements-moirai.txt
bash codigo/roda_tudo.sh
```

Run all scripts from the repository root. Model weights are downloaded from Hugging Face (`amazon/chronos-bolt-small`, `amazon/chronos-bolt-base`, `amazon/chronos-2`, `google/timesfm-2.5-200m-pytorch` and `Salesforce/moirai-2.0-R-small`). The full pipeline takes about 10–12 hours on a two-core CPU without a GPU. Random seeds are fixed: 42 for LightGBM and Optuna, 2026 for the bootstrap.

To check results without re-running the models, run `05`–`09` and `13`–`15` on the forecasts provided in `previsoes/`.

`SHA256SUMS.txt` lists checksums of all data files.

## Data sources

- Sales: IBGE, Pesquisa Mensal de Comércio, SIDRA tables 8882 and 8880 (variable 7169, volume index 2022 = 100, not seasonally adjusted).
- Factors:
  - IBGE/SIDRA (IPCA; PNAD Contínua unemployment and labour income).
  - Ipeadata: series from the Central Bank of Brazil, the IMF, Fecomércio-SP and CNC, plus the real minimum wage.
  - Ministry of Social Development, VIS DATA (Bolsa Família and Auxílio Brasil transfers).
  - Baker, Bloom and Davis (2016) and Davis (2016) policy uncertainty indices (policyuncertainty.com).
  - NASA POWER: daily temperature and precipitation at the 27 state capitals.

All source data are public. They are redistributed here unchanged for reproducibility, and users should cite the original sources.

## Use of generative AI

A generative AI assistant (Claude, Anthropic) helped write and debug the code, run the computations and draft the manuscript. The author designed the study, reviewed the code and verified the results.

## License

Code: MIT (see `LICENSE`). Source data remain under the terms of their providers.

## Citation

See `CITATION.cff` and the Zenodo DOI badge of this repository.
