"""LightGBM global com fatores externos (mesmos hiperparâmetros e reajustes do LightGBM base, para isolar
o efeito das covariáveis).

Variantes:
  real     — só informação disponível na origem: último valor publicado de cada fator (respeitando a defasagem
             de publicação) + fatores conhecidos com antecedência no mês-alvo (calendário, eleições);
  oraculo  — acrescenta o valor realizado de todos os fatores no mês-alvo (teto do ganho possível).
Saídas: previsoes/LightGBM_X_real.parquet, previsoes/LightGBM_X_oraculo.parquet;
        resultados/ablacao_grupos.csv (retirada de cada grupo, modelo da mediana);
        resultados/shap_*.csv (contribuições SHAP do modelo real por grupo e atividade)."""
import json, sys, time
import numpy as np
import pandas as pd
from lgb_base import *  # noqa: F401,F403
from lgb_base import X, FEATS, CATS, treino_ate, ajusta, meta
from comum import ORIGENS_TESTE, ORIGENS_VALID, QUANTIS, QCOLS

modo = sys.argv[1] if len(sys.argv) > 1 else "real"
cov = pd.read_parquet("dados/covariaveis.parquet")
info = pd.read_csv("dados/covariaveis_meta.csv").set_index("variavel")
X["geo_cod"] = X.unique_id.map(meta.geo_cod.astype(int))
tab = cov.set_index(["geo_cod", "data"])


def busca(var, datas):
    return tab[var].reindex(pd.MultiIndex.from_arrays([X.geo_cod.values, pd.DatetimeIndex(datas)])).values


GRUPO = {}
for v, r in info.iterrows():
    if r.futuro_conhecido:
        X[f"{v}@alvo"] = busca(v, X.data_alvo); GRUPO[f"{v}@alvo"] = r.grupo
    else:
        X[f"{v}@orig"] = busca(v, X.origem - pd.DateOffset(months=int(r.defasagem))); GRUPO[f"{v}@orig"] = r.grupo
        if r.grupo == "Climate":   # média das anomalias dos últimos 3 meses
            X[f"{v}@orig3"] = np.nanmean([busca(v, X.origem - pd.DateOffset(months=k)) for k in range(3)], axis=0)
            GRUPO[f"{v}@orig3"] = r.grupo
        if modo == "oraculo":
            X[f"{v}@alvo_real"] = busca(v, X.data_alvo); GRUPO[f"{v}@alvo_real"] = r.grupo
COV = list(GRUPO)
F2 = FEATS + COV
melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
print(modo, "| atributos:", len(F2), "| covariáveis:", len(COV))


def fit(params, dados, feats, alpha):
    import lightgbm as lgb
    kw = dict(params, objective="quantile", alpha=alpha, random_state=42, n_jobs=2, verbose=-1, bagging_freq=1)
    m = lgb.LGBMRegressor(**kw)
    m.fit(dados[feats], dados.z, categorical_feature=CATS)
    return m


refits = [o for o in ORIGENS_TESTE if o.month == 7]
blocos = [(r, refits[k + 1] if k + 1 < len(refits) else ORIGENS_TESTE[-1] + pd.DateOffset(months=1))
          for k, r in enumerate(refits)]

SEL = None
if modo == "selecionado":   # grupos escolhidos na validação (ver modo "valid")
    SEL = json.load(open("resultados/fatores_selecao_validacao.json"))["grupos"]
    F2 = FEATS + [f for f in COV if GRUPO[f] in SEL]
    print("grupos selecionados:", SEL, "| atributos:", len(F2))
SO_SHAP = modo == "shap"
if SO_SHAP:
    modo = "real"
if modo in ("real", "oraculo", "selecionado"):
    saida, contrib = [], []
    for r, fim in blocos:
        tr = treino_ate(r)
        alvo = X[X.origem.isin([o for o in ORIGENS_TESTE if r <= o < fim])].copy()
        t0 = time.time()
        for q, c in zip(QUANTIS, QCOLS):
            if SO_SHAP and q != 0.5:
                continue
            mdl = fit(melhor, tr, F2, q)
            alvo[c] = np.exp(alvo.base_log + mdl.predict(alvo[F2]))
            if q == 0.5 and modo == "real" and SEL is None:
                sh = mdl.predict(alvo[F2], pred_contrib=True)[:, :-1]
                s = pd.DataFrame(np.abs(sh), columns=F2)
                s["_uid"] = alvo.unique_id.values; s["_h"] = alvo.h.values   # nomes distintos dos atributos h e mes_alvo
                s["_mes"] = alvo.mes_alvo.values
                contrib.append(s)
                # efeito com sinal das principais covariáveis (para gráficos de dependência)
                sd = pd.DataFrame(sh[:, [F2.index(v) for v in COV]], columns=COV)
                sd[[f"x:{v}" for v in COV]] = alvo[COV].values
                sd["unique_id"] = alvo.unique_id.values
                sd.to_parquet(f"resultados/shap_sinal_{r.year}.parquet", index=False)
        print(f"{modo} reajuste {r.date()}: treino {len(tr)}, previsões {len(alvo)}, {time.time() - t0:.0f}s", flush=True)
        saida.append(alvo.reindex(columns=["unique_id", "origem", "h", "data_alvo"] + QCOLS))
    out = pd.concat(saida, ignore_index=True)
    if SO_SHAP:
        out = out.iloc[:0]
    out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
    out["modelo"] = {"real": "LightGBM + external factors", "oraculo": "LightGBM + external factors (oracle)",
                     "selecionado": "LightGBM + selected factors"}[modo]
    if not SO_SHAP:
        out.to_parquet(f"previsoes/LightGBM_X_{modo}.parquet", index=False)
    if contrib and SEL is None:
        c = pd.concat(contrib, ignore_index=True)
        grupos = {"Own history": [f for f in FEATS if f not in ("geo", "ativ", "mes_alvo", "h", "fds_dif")],
                  "Series identity and season": ["geo", "ativ", "mes_alvo", "h", "fds_dif"]}
        for f in COV:
            grupos.setdefault(GRUPO[f], []).append(f)
        g = pd.DataFrame({k: c[v].sum(axis=1) for k, v in grupos.items()})
        g["unique_id"] = c._uid; g["h"] = c._h; g["mes_alvo"] = c._mes
        g.to_parquet("resultados/shap_grupos.parquet", index=False)
        c[COV + ["_uid"]].groupby("_uid").mean().rename_axis("unique_id").to_parquet("resultados/shap_variaveis_por_serie.parquet")
    print("ok", len(out))

elif modo == "ablacao":
    # modelo da mediana com todas as covariáveis e sem cada grupo; MASE nas mesmas previsões
    grupos = sorted(set(GRUPO.values()))
    linhas = []
    for nome, feats in [("Base (own history only)", FEATS), ("All external factors", F2)] + \
                        [(f"All minus {g}", [f for f in F2 if GRUPO.get(f) != g]) for g in grupos] + \
                        [(f"Base plus {g}", FEATS + [f for f in COV if GRUPO[f] == g]) for g in grupos]:
        prev = []
        for r, fim in blocos:
            tr = treino_ate(r)
            alvo = X[X.origem.isin([o for o in ORIGENS_TESTE if r <= o < fim])].copy()
            mdl = fit(melhor, tr, feats, 0.5)
            alvo["q50"] = np.exp(alvo.base_log + mdl.predict(alvo[feats]))
            prev.append(alvo[["unique_id", "origem", "h", "data_alvo", "q50", "y_real", "escala", "mes_alvo"]])
        d = pd.concat(prev)
        d["ae"] = np.abs(d.y_real - d.q50) / d.escala
        linhas.append(d.assign(variante=nome))
        print(nome, round(d.ae.mean(), 4), flush=True)
    pd.concat(linhas).to_parquet("resultados/ablacao_previsoes.parquet", index=False)

elif modo == "valid":
    # seleção de grupos sem olhar o teste: treino com alvos até mar/2016, previsão nas origens de validação
    grupos = sorted(set(GRUPO.values()))
    tr = treino_ate(ORIGENS_VALID[0])
    va = X[X.origem.isin(ORIGENS_VALID) & X.z.notna()]
    res = {}
    for nome, feats in [("Base", FEATS), ("All", F2)] + [(f"Base plus {g}", FEATS + [f for f in COV if GRUPO[f] == g])
                                                          for g in grupos]:
        mdl = fit(melhor, tr, feats, 0.5)
        res[nome] = float(np.mean(np.abs(va.y_real - np.exp(va.base_log + mdl.predict(va[feats]))) / va.escala))
        print(nome, round(res[nome], 4), flush=True)
    sel = [g for g in grupos if res[f"Base plus {g}"] < res["Base"]]
    json.dump({"mase_validacao": res, "grupos": sel}, open("resultados/fatores_selecao_validacao.json", "w"), indent=1)
    print("selecionados:", sel)
