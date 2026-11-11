"""Ideia 6: indicadores rápidos (Ipeadata / Banco Central) melhoram a previsão, sobretudo a de curto prazo?
Modelo: LightGBM global da mediana, mesmos hiperparâmetros e reajustes anuais do estudo (isola o efeito dos dados).
Momento da previsão: divulgação da PMC do mês o (cerca de dia 15 do mês o+2). Disponibilidade de cada indicador
definida de forma conservadora pelo que as APIs mostravam em 10/11/2026 (PMC de set/2026 prestes a sair):
  avanco = +1 -> já existe o mês o+1 (confiança Fecomercio, expectativa de inflação, confiança da construção; câmbio e Selic são diários)
  avanco =  0 -> último mês disponível = o;  avanco = -1 -> último mês disponível = o-1 (energia, combustíveis)
Variantes:
  V0 histórico + fatores selecionados (modelo atual)      V1 = V0 + indicadores rápidos na defasagem disponível
  V2 = V1 + mês o+1 dos indicadores comprovadamente adiantados
  V3 = V2 + mês o+1 de emplacamentos/licenciamentos (saem na 1ª semana na fonte original, mas não na API)"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from base import OUT, boot_ic
from lgb_base import X, FEATS, CATS, treino_ate, meta
from comum import ORIGENS_TESTE

R = "dados/ext/rapidos"
#        código                      nome                transf    avanço
SER = [("FENABRAVE12_VENDVETOT12", "emplacamentos",      "logyoy", 0), ("ANFAVE12_LICVETOT12", "licenciamentos", "logyoy", 0),
       ("ABPO12_PAPEL12",          "papelao",            "logyoy", 0), ("BM12_PMPPM12",        "papel_moeda",    "logyoy", 0),
       ("BM12_CCAPF12",            "concessoes_pf",      "logyoy", 0), ("PNADC12_OCUPCOMM12",  "ocupados_comercio", "logyoy", 0),
       ("CNC12_ICF12",             "intencao_consumo",   "difyoy", 0), ("CNC12_ICEC12",        "conf_comercio",  "difyoy", 0),
       ("CNC12_PEICT12",           "endividadas",        "difyoy", 0),
       ("ELETRO12_CEECOM12",       "energia_comercio",   "logyoy", -1), ("ELETRO12_CEERES12",  "energia_resid",  "logyoy", -1),
       ("ANP12_CGASOL12",          "gasolina",           "logyoy", -1), ("ANP12_COLDIE12",     "diesel",         "logyoy", -1),
       ("FCESP12_IICF12",          "expect_consumidor",  "difyoy", 1), ("BM12_IPCAEXP1212",    "expect_inflacao", "nivel", 1),
       ("FGV12_ICSTSA12",          "conf_construcao",    "difyoy", 1)]
T = {}
for cod, nome, tr, av in SER:
    v = pd.DataFrame(json.load(open(f"{R}/{cod}.json"))["value"]).dropna(subset=["VALVALOR"])
    s = pd.Series(v.VALVALOR.values, index=pd.to_datetime(v.VALDATA.str[:10])).sort_index(); s = s[~s.index.duplicated()]
    s.index = s.index.to_period("M").to_timestamp()
    T[nome] = {"logyoy": np.log(s).diff(12), "difyoy": s.diff(12), "nivel": s}[tr]
T = pd.DataFrame(T)
T.to_csv(f"{OUT}/i1_indicadores_rapidos.csv")
AV = {nome: av for _, nome, _, av in SER}
pega = lambda col, datas: col.reindex(pd.DatetimeIndex(datas)).values
N1, N2, N3 = [], [], []
for nome in T:
    ult = X.origem + pd.DateOffset(months=min(AV[nome], 0))
    X[f"{nome}@ult"] = pega(T[nome], ult); X[f"{nome}@var3"] = X[f"{nome}@ult"] - pega(T[nome], ult - pd.DateOffset(months=3)); N1 += [f"{nome}@ult", f"{nome}@var3"]
    if AV[nome] == 1:
        X[f"{nome}@o+1"] = pega(T[nome], X.origem + pd.DateOffset(months=1)); N2.append(f"{nome}@o+1")
for nome in ("emplacamentos", "licenciamentos"):
    X[f"{nome}@o+1"] = pega(T[nome], X.origem + pd.DateOffset(months=1)); N3.append(f"{nome}@o+1")
# fatores do estudo (grupos selecionados na validação) e o mês o+1 dos que são diários/imediatos
cov = pd.read_parquet("dados/covariaveis.parquet"); info = pd.read_csv("dados/covariaveis_meta.csv").set_index("variavel")
sel = json.load(open("resultados/fatores_selecao_validacao.json"))["grupos"]
X["geo_cod"] = X.unique_id.map(meta.geo_cod.astype(int)); tab = cov.set_index(["geo_cod", "data"])
busca = lambda var, datas: tab[var].reindex(pd.MultiIndex.from_arrays([X.geo_cod.values, pd.DatetimeIndex(datas)])).values
C0 = []
for v, r in info[info.grupo.isin(sel)].iterrows():
    if r.futuro_conhecido:
        X[f"{v}@alvo"] = busca(v, X.data_alvo); C0.append(f"{v}@alvo")
    else:
        X[f"{v}@orig"] = busca(v, X.origem - pd.DateOffset(months=int(r.defasagem))); C0.append(f"{v}@orig")
        if r.grupo == "Climate":
            X[f"{v}@orig3"] = np.nanmean([busca(v, X.origem - pd.DateOffset(months=k)) for k in range(3)], axis=0); C0.append(f"{v}@orig3")
for v in ("cambio", "selic", "confianca"):
    X[f"{v}@o+1"] = busca(v, X.origem + pd.DateOffset(months=1)); N2.append(f"{v}@o+1")
VAR = {"Só histórico": FEATS, "V0: histórico + fatores (atual)": FEATS + C0, "V1: + indicadores rápidos": FEATS + C0 + N1,
       "V2: + mês o+1 (adiantados)": FEATS + C0 + N1 + N2, "V3: + mês o+1 de veículos": FEATS + C0 + N1 + N2 + N3,
       "V2 sem os novos: fatores + mês o+1 de câmbio, Selic, confiança": FEATS + C0 + [f for f in N2 if f.split("@")[0] in ("cambio", "selic", "confianca")]}
melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
refits = [o for o in ORIGENS_TESTE if o.month == 7]
blocos = [(r, refits[k + 1] if k + 1 < len(refits) else ORIGENS_TESTE[-1] + pd.DateOffset(months=1)) for k, r in enumerate(refits)]
P = None; imp = {}
for nome, feats in VAR.items():
    prev = []
    for r, fim in blocos:
        tr = treino_ate(r); alvo = X[X.origem.isin([o for o in ORIGENS_TESTE if r <= o < fim])].copy()
        m = lgb.LGBMRegressor(**dict(melhor, objective="quantile", alpha=0.5, random_state=42, n_jobs=2, verbose=-1, bagging_freq=1)).fit(tr[feats], tr.z, categorical_feature=CATS)
        alvo[nome] = np.exp(alvo.base_log + m.predict(alvo[feats])); prev.append(alvo[["unique_id", "origem", "h", "y_real", "escala", nome]])
        imp[nome] = pd.Series(m.booster_.feature_importance("gain"), feats)
    d = pd.concat(prev); P = d if P is None else P.merge(d[["unique_id", "origem", "h", nome]], on=["unique_id", "origem", "h"])
    print(nome, round(((P[nome] - P.y_real).abs() / P.escala).mean(), 4), flush=True)
P.to_parquet(f"{OUT}/i1_previsoes.parquet", index=False)
ref = "V0: histórico + fatores (atual)"; lin = []
for rot, x in [("Todos os horizontes", P), ("h = 1", P[P.h == 1]), ("h = 2–3", P[P.h.between(2, 3)]), ("h = 4–12", P[P.h >= 4])]:
    mb = ((x[ref] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean()
    for nome in VAR:
        ms = ((x[nome] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean(); r = np.log(ms / mb); ic = boot_ic(r.values)
        lin.append(dict(recorte=rot, variante=nome, MAPE=(100 * (x[nome] - x.y_real).abs() / x.y_real).mean(), razao_vs_atual=np.exp(r.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_series_melhoram=100 * (ms < mb).mean()))
Tb = pd.DataFrame(lin); Tb.round(4).to_csv(f"{OUT}/i1_rapidos.csv", index=False); print(Tb.round(3).to_string())
g = imp["V3: + mês o+1 de veículos"]; g = 100 * g / g.sum()
novos = g[[f for f in g.index if f in N1 + N2 + N3]].sort_values(ascending=False)
novos.round(2).to_csv(f"{OUT}/i1_importancia_novos.csv"); print("peso total dos novos atributos (%):", round(novos.sum(), 1)); print(novos.head(15).round(2).to_string())
