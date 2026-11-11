"""Ideia 7 (testes) com o calendário de eventos de j0_eventos.py.
(a) Explicar: os meses de evento concentram outliers e erros de previsão?
(b) Prever: variáveis dos eventos anunciados com antecedência (no mês-alvo) reduzem o erro do LightGBM?"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from base import OUT, boot_ic

C = pd.read_csv(f"{OUT}/j0_calendario_eventos.csv").fillna({"geo": ""}); C["data"] = pd.to_datetime(C.mes + "-01")
# ---------------- (a) outliers em meses de evento
o = pd.read_csv(f"{OUT}/outliers_detectados.csv", parse_dates=["data"])
o["az"] = o.z.abs()
lin = []
base_out = o[~o.pandemia].outlier.mean() * 100; base_ext = o[~o.pandemia].extremo.mean() * 100
for (t, ev), g in C.groupby(["tipo", "evento"], sort=False):
    x = o[o.data.isin(g.data)]
    if g.geo.iloc[0]:
        x = x[x.geo == g.geo.iloc[0]]
    if len(x) == 0:
        continue
    lin.append(dict(tipo=t, evento=ev, antecipado=g.antecipado.iloc[0], meses=g.data.nunique(), series_mes=len(x), pct_outlier=100 * x.outlier.mean(), pct_extremo=100 * x.extremo.mean(),
                    desvio_mediano_pct=x.resid_pct.median(), vezes_o_normal=100 * x.outlier.mean() / base_out))
A = pd.DataFrame(lin).sort_values("vezes_o_normal", ascending=False)
A.round(2).to_csv(f"{OUT}/j1_outliers_por_evento.csv", index=False); print("taxa normal de outliers fora da pandemia (%):", round(base_out, 2)); print(A.round(2).to_string())
ev_nac = C[C.geo == ""]
o["em_evento"] = o.data.isin(ev_nac.data); o["em_evento_surpresa"] = o.data.isin(ev_nac[ev_nac.antecipado == 0].data); o["em_evento_antecip"] = o.data.isin(ev_nac[ev_nac.antecipado == 1].data)
np_ = o[~o.pandemia]
res = dict(pct_meses_com_evento=100 * np_.em_evento.mean(), pct_outliers_em_meses_de_evento=100 * np_[np_.outlier].em_evento.mean(),
           pct_extremos_em_meses_de_evento=100 * np_[np_.extremo].em_evento.mean(),
           pct_extremos_em_evento_surpresa=100 * np_[np_.extremo].em_evento_surpresa.mean(), pct_meses_evento_surpresa=100 * np_.em_evento_surpresa.mean())
print({k: round(v, 1) for k, v in res.items()})
# os maiores choques simultâneos (fora da pandemia) têm evento associado no calendário?
mes = np_.groupby("data").agg(pct_out=("outlier", "mean"), desvio=("resid_pct", "median")).sort_values("pct_out", ascending=False).head(15)
mes["pct_out"] *= 100; mes["evento_no_calendario"] = [", ".join(C[C.data == d].evento.unique()) or "— sem evento —" for d in mes.index]
mes.round(1).to_csv(f"{OUT}/j1_maiores_choques.csv"); print(mes.round(1).to_string())
# erro de previsão (combinação) em meses-alvo com evento, no período de teste
d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet"); d = d[d.modelo == "Combinação (todos)"]
d["rel"] = d.ape / d.groupby("unique_id").ape.transform("mean")
for rot, datas in (("evento antecipado", ev_nac[ev_nac.antecipado == 1].data), ("sem evento", None)):
    x = d[d.data_alvo.isin(datas)] if datas is not None else d[~d.data_alvo.isin(ev_nac.data)]
    print(rot, "| MAPE:", round(x.ape.mean(), 2), "| erro relativo à média da série:", round(x.rel.mean(), 3), "| n:", len(x))
te = []
for ev, g in ev_nac.groupby("evento"):
    x = d[d.data_alvo.isin(g.data)]
    if len(x):
        te.append(dict(evento=ev, meses_no_teste=x.data_alvo.nunique(), MAPE=x.ape.mean(), erro_rel_a_media_da_serie=x.rel.mean(), vies_pct=x.epct.mean()))
te = pd.DataFrame(te).sort_values("erro_rel_a_media_da_serie", ascending=False); te.round(2).to_csv(f"{OUT}/j1_erro_por_evento_teste.csv", index=False); print(te.round(2).to_string())

# ---------------- (b) eventos antecipados como variáveis de previsão
from lgb_base import X, FEATS, CATS, treino_ate, meta
from comum import ORIGENS_TESTE
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
EV = []
for t, g in ev_nac[ev_nac.antecipado == 1].groupby("tipo"):
    s = g.groupby("data").intensidade.max()
    X[f"ev_{t}"] = s.reindex(pd.DatetimeIndex(X.data_alvo)).fillna(0).values                       # no mês-alvo
    X[f"ev_{t}_ano_ant"] = s.reindex(pd.DatetimeIndex(X.data_alvo - pd.DateOffset(months=12))).fillna(0).values   # base de comparação (alvo é variação em 12 meses)
    EV += [f"ev_{t}", f"ev_{t}_ano_ant"]
melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
refits = [x for x in ORIGENS_TESTE if x.month == 7]
blocos = [(r, refits[k + 1] if k + 1 < len(refits) else ORIGENS_TESTE[-1] + pd.DateOffset(months=1)) for k, r in enumerate(refits)]
P = None
for nome, feats in {"Atual (histórico + fatores)": FEATS + C0, "Atual + eventos antecipados": FEATS + C0 + EV}.items():
    prev = []
    for r, fim in blocos:
        tr = treino_ate(r); alvo = X[X.origem.isin([x for x in ORIGENS_TESTE if r <= x < fim])].copy()
        mdl = lgb.LGBMRegressor(**dict(melhor, objective="quantile", alpha=0.5, random_state=42, n_jobs=2, verbose=-1, bagging_freq=1)).fit(tr[feats], tr.z, categorical_feature=CATS)
        alvo[nome] = np.exp(alvo.base_log + mdl.predict(alvo[feats])); prev.append(alvo[["unique_id", "origem", "h", "data_alvo", "y_real", "escala", nome]])
    x = pd.concat(prev); P = x if P is None else P.merge(x[["unique_id", "origem", "h", nome]], on=["unique_id", "origem", "h"])
P["em_evento"] = P.data_alvo.isin(ev_nac[ev_nac.antecipado == 1].data)
lin = []
for rot, x in (("Todos os meses-alvo", P), ("Meses-alvo com evento antecipado", P[P.em_evento]), ("Meses-alvo sem evento", P[~P.em_evento])):
    a, b = "Atual + eventos antecipados", "Atual (histórico + fatores)"
    ms = ((x[a] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean(); mb = ((x[b] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean()
    r = np.log(ms / mb); ic = boot_ic(r.values)
    lin.append(dict(recorte=rot, n=len(x), MAPE_atual=(100 * (x[b] - x.y_real).abs() / x.y_real).mean(), MAPE_com_eventos=(100 * (x[a] - x.y_real).abs() / x.y_real).mean(),
                    razao=np.exp(r.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_series_melhoram=100 * (ms < mb).mean()))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/j1_eventos_previsao.csv", index=False); print(T.round(3).to_string())
imp = pd.Series(mdl.booster_.feature_importance("gain"), feats); print("peso dos eventos no modelo (%):", round(100 * imp[EV].sum() / imp.sum(), 2)); print((100 * imp[EV] / imp.sum()).sort_values(ascending=False).round(3).head(8).to_string())
