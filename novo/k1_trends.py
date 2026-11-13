"""Ideia 6 (Google Trends): as buscas ajudam a prever as vendas? Lê novo/saidas/trends.csv (gerado por k0_baixa_trends.py).
Momento da previsão: divulgação da PMC do mês o (cerca de dia 15 de o+2). Nessa data o Trends dos meses o e o+1 já
está fechado. Atributos por série (termos ligados à atividade; UF se houver, senão Brasil), em variação de 12 meses do log:
  tr@o, tr@o+1, variação em 3 meses, e um índice geral ("promoção").
Testes: (1) modelo-ponte nas 12 séries do Brasil, 1 mês à frente; (2) LightGBM do estudo com e sem Trends, todos os horizontes.
Ressalvas: o Trends começa em 2004, é uma amostra (muda um pouco a cada download) e é reescalado no período inteiro."""
import json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from base import OUT, painel, boot_ic, meta as meta_pt

ARQ = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/trends.csv"
MAPA = {"Eletrodomésticos": ["geladeira", "fogão", "máquina de lavar", "televisão"], "Móveis": ["sofá", "guarda roupa"],
        "Móveis e eletrodomésticos": ["geladeira", "fogão", "máquina de lavar", "televisão", "sofá", "guarda roupa"],
        "Equipamentos e materiais para escritório": ["celular", "notebook", "impressora"], "Tecidos, vestuário": ["tênis", "vestido", "calça jeans"],
        "Artigos farmacêuticos": ["farmácia", "perfume"], "Hipermercados": ["supermercado", "cesta básica"], "Combustíveis": ["posto de gasolina"],
        "Livros, jornais": ["material escolar", "livro"], "Outros artigos": ["promoção", "perfume", "celular"], "Varejo total": ["promoção", "supermercado", "celular", "geladeira", "tênis"]}
UF = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
      31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"}
t = pd.read_csv(ARQ, parse_dates=["data"]); t["data"] = t.data.dt.to_period("M").dt.to_timestamp()
L = t.pivot_table(index="data", columns=["geo", "termo"], values="valor")
G = (np.log(L.clip(lower=1)).diff(12)).where(L.notna())                      # variação em 12 meses do log (escala some)
from lgb_base import X, FEATS, CATS, treino_ate, meta
from comum import ORIGENS_TESTE
p = painel(); meta = meta.join(meta_pt(p)[["ativ"]])
termos = lambda ativ: next((v for k, v in sorted(MAPA.items(), key=lambda kv: -len(kv[0])) if ativ.startswith(k)), MAPA["Varejo total"])
ind = {}                                                                   # índice de buscas por série
for u, r in meta.iterrows():
    geo = "BR" if int(r.geo_cod) == 1 else f"BR-{UF.get(int(r.geo_cod), '')}"
    geo = geo if geo in G.columns.get_level_values(0) else "BR"
    cols = [(geo, x) for x in termos(r.ativ) if (geo, x) in G.columns]
    ind[u] = G[cols].mean(axis=1) if cols else pd.Series(np.nan, index=G.index)
I = pd.DataFrame(ind); geral = G[("BR", "promoção")] if ("BR", "promoção") in G.columns else I.mean(axis=1)
print("termos:", sorted(t.termo.unique()), "| geografias:", t.geo.nunique(), "| período:", t.data.min().date(), "a", t.data.max().date())

# ---------- (1) modelo-ponte, Brasil, 1 mês à frente
lin = []
for u in meta[meta.geo_cod.astype(int) == 1].index:
    g = np.log(p[p.unique_id == u].set_index("data").valor).diff(12).dropna()
    D = pd.DataFrame({"y": g, "l1": g.shift(1), "l2": g.shift(2), "m3": g.shift(1).rolling(3).mean()})
    D = D[~D.index.to_series().between("2020-03-01", "2021-12-01")]
    for nome, extra in (("sem Trends", {}), ("+ Trends do próprio mês", {"x0": I[u]}), ("+ Trends do mês anterior", {"x1": I[u].shift(1)}),
                        ("+ Trends do mês e do anterior", {"x0": I[u], "x1": I[u].shift(1)})):
        Z = D.assign(**{k: v.reindex(D.index) for k, v in extra.items()}).dropna(); cols = [c for c in Z if c != "y"]; e = []
        for d in Z.index[Z.index >= "2022-08-01"]:
            tr = Z[Z.index < d]
            b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr[cols].values]), tr.y.values, rcond=None)[0]
            e.append(abs(Z.loc[d, "y"] - np.r_[1, Z.loc[d, cols].values] @ b))
        lin.append(dict(unique_id=u, ativ=meta.at[u, "ativ"][:40], variante=nome, mae=np.mean(e)))
R = pd.DataFrame(lin); R["razao"] = R.mae / R.unique_id.map(R[R.variante == "sem Trends"].set_index("unique_id").mae)
R.round(4).to_csv(f"{OUT}/k1_trends_ponte.csv", index=False)
print(R.pivot(index="ativ", columns="variante", values="razao").round(3).to_string())
print(R.groupby("variante").razao.agg(lambda v: np.exp(np.log(v).mean())).round(3).to_string())

# ---------- (2) LightGBM do estudo com e sem Trends
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
S = I.stack(dropna=False); S.index.names = ["data", "unique_id"]
pega = lambda k: S.reindex(pd.MultiIndex.from_arrays([pd.DatetimeIndex(X.origem + pd.DateOffset(months=k)), X.unique_id.values])).values
X["tr@o+1"] = pega(1); X["tr@o"] = pega(0); X["tr@var3"] = X["tr@o+1"] - pega(-2)
X["tr_geral@o+1"] = geral.reindex(pd.DatetimeIndex(X.origem + pd.DateOffset(months=1))).values
TR = ["tr@o+1", "tr@o", "tr@var3", "tr_geral@o+1"]
melhor = json.load(open("previsoes/lightgbm_hiperparametros.json"))["melhor"]
refits = [o for o in ORIGENS_TESTE if o.month == 7]
blocos = [(r, refits[k + 1] if k + 1 < len(refits) else ORIGENS_TESTE[-1] + pd.DateOffset(months=1)) for k, r in enumerate(refits)]
P = None; VAR = {"Atual (histórico + fatores)": FEATS + C0, "Atual + Google Trends": FEATS + C0 + TR, "Só histórico": FEATS, "Histórico + Google Trends": FEATS + TR}
for nome, feats in VAR.items():
    prev = []
    for r, fim in blocos:
        tr = treino_ate(r); alvo = X[X.origem.isin([o for o in ORIGENS_TESTE if r <= o < fim])].copy()
        m = lgb.LGBMRegressor(**dict(melhor, objective="quantile", alpha=0.5, random_state=42, n_jobs=2, verbose=-1, bagging_freq=1)).fit(tr[feats], tr.z, categorical_feature=CATS)
        alvo[nome] = np.exp(alvo.base_log + m.predict(alvo[feats])); prev.append(alvo[["unique_id", "origem", "h", "y_real", "escala", nome]])
        if nome == "Atual + Google Trends":
            imp = pd.Series(m.booster_.feature_importance("gain"), feats)
    d = pd.concat(prev); P = d if P is None else P.merge(d[["unique_id", "origem", "h", nome]], on=["unique_id", "origem", "h"])
    print(nome, "ok", flush=True)
P = P.join(meta[["ativ"]], on="unique_id"); P.to_parquet(f"{OUT}/k1_previsoes.parquet", index=False)
lin = []
for a, b in (("Atual + Google Trends", "Atual (histórico + fatores)"), ("Histórico + Google Trends", "Só histórico")):
    for rot, x in [("Todos os horizontes", P), ("h = 1", P[P.h == 1]), ("h = 2–3", P[P.h.between(2, 3)]), ("h = 4–12", P[P.h >= 4])] + [(f"h ≤ 3 | {k[:40]}", x[x.h <= 3]) for k, x in P.groupby("ativ")]:
        ms = ((x[a] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean(); mb = ((x[b] - x.y_real).abs() / x.escala).groupby(x.unique_id).mean()
        r = np.log(ms / mb); ic = boot_ic(r.values)
        lin.append(dict(comparacao=f"{a} × {b}", recorte=rot, n_series=len(ms), MAPE_sem=(100 * (x[b] - x.y_real).abs() / x.y_real).mean(), MAPE_com=(100 * (x[a] - x.y_real).abs() / x.y_real).mean(),
                        razao=np.exp(r.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_series_melhoram=100 * (ms < mb).mean()))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/k1_trends.csv", index=False); print(T.round(3).to_string())
print("peso do Trends no modelo (%):", (100 * imp[TR] / imp.sum()).round(2).to_dict())
