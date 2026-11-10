"""Teste 1: combinação "inteligente" contra a mediana simples.
Banco de provas: período de teste (38 origens mensais). Em cada origem o só se usam, para aprender pesos,
erros de previsões cujo alvo já foi observado (data_alvo <= o). Avaliação nas 26 últimas origens.
Tudo em escala: desvio em relação ao ingênuo sazonal dividido pela escala do MASE."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from base import OUT, boot_ic

d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
BASE = [m for m in d.modelo.unique() if not m.startswith("Combinação")]
K = ["unique_id", "origem", "h"]
W = d[d.modelo.isin(BASE)].pivot_table(index=K + ["data_alvo", "y", "escala"], columns="modelo", values="q50").reset_index()
NAO = [m for m in BASE if m != "Ingênuo sazonal"]
sn = W["Ingênuo sazonal"].values
Z = (W[BASE].values - sn[:, None]) / W.escala.values[:, None]          # previsões em escala
t = ((W.y - sn) / W.escala).values                                    # alvo em escala
feat = pd.read_csv(f"{OUT}/previsibilidade_series.csv").set_index("unique_id")[["forca_sazonal", "ruido_pct", "volatilidade_crescimento_pct", "amplitude_sazonal_pct", "anos_historico", "brasil"]]
F = feat.reindex(W.unique_id).values
mes = W.data_alvo.dt.month.values
origens = np.sort(W.origem.unique()); AVAL = origens[12:]
uid = W.unique_id.values; ucod = pd.factorize(uid)[0]; nU = ucod.max() + 1
P = {}                                                                 # previsões (em escala) de cada método


def nnls_soma1(A, b, forca=1e3):
    A2 = np.vstack([A, forca * np.ones(A.shape[1])]); b2 = np.append(b, forca)
    w, _ = nnls(A2, b2); return w / w.sum()


def Xmat(idx):
    z = Z[idx]
    return np.column_stack([z, W.h.values[idx], mes[idx], z.std(1), np.median(z, 1), F[idx]])


for nome in ("Mediana (todos)", "Média (todos)", "Peso por série (1/erro²)", "Peso por série + horizonte", "Pesos globais (NNLS)",
             "Mediana dos 5 melhores (dinâmico)", "Empilhamento LightGBM", "Mediana + empilhamento", "Mediana + pesos globais"):
    P[nome] = np.full(len(W), np.nan)
hb = np.digitize(W.h.values, [4, 8])                                   # faixas de horizonte: 1–3, 4–7, 8–12
for o in AVAL:
    tr = (W.data_alvo <= o).values; te = (W.origem == o).values
    ztr, ttr, zte = Z[tr], t[tr], Z[te]
    P["Mediana (todos)"][te] = np.median(zte, 1); P["Média (todos)"][te] = zte.mean(1)
    ae = np.abs(ztr - ttr[:, None])
    # (B) peso por série: 1/erro², encolhido para o erro global
    g = ae.mean(0)
    s = pd.DataFrame(ae).groupby(ucod[tr]).agg(["sum", "count"])
    n0 = 24
    es = np.vstack([(s[j]["sum"].reindex(range(nU)).fillna(0).values + n0 * g[j]) / (s[j]["count"].reindex(range(nU)).fillna(0).values + n0) for j in range(len(BASE))]).T
    w = 1 / es ** 2; w /= w.sum(1, keepdims=True)
    P["Peso por série (1/erro²)"][te] = (zte * w[ucod[te]]).sum(1)
    # (B2) por série e faixa de horizonte
    out = np.zeros(te.sum())
    for b in range(3):
        mtr = hb[tr] == b; mte = hb[te] == b
        if mtr.sum() < 500:
            out[mte] = (zte[mte] * w[ucod[te][mte]]).sum(1); continue
        gb = ae[mtr].mean(0); sb = pd.DataFrame(ae[mtr]).groupby(ucod[tr][mtr]).agg(["sum", "count"])
        eb = np.vstack([(sb[j]["sum"].reindex(range(nU)).fillna(0).values + n0 * gb[j]) / (sb[j]["count"].reindex(range(nU)).fillna(0).values + n0) for j in range(len(BASE))]).T
        wb = 1 / eb ** 2; wb /= wb.sum(1, keepdims=True)
        out[mte] = (zte[mte] * wb[ucod[te][mte]]).sum(1)
    P["Peso por série + horizonte"][te] = out
    # (C) pesos globais por faixa de horizonte (mínimos quadrados não negativos, soma 1)
    out = np.zeros(te.sum())
    for b in range(3):
        mtr = hb[tr] == b; mte = hb[te] == b
        wg = nnls_soma1(ztr[mtr], ttr[mtr]) if mtr.sum() >= 500 else nnls_soma1(ztr, ttr)
        out[mte] = zte[mte] @ wg
    P["Pesos globais (NNLS)"][te] = out
    # (E) mediana dos 5 melhores modelos até aqui
    top = np.argsort(g)[:5]; P["Mediana dos 5 melhores (dinâmico)"][te] = np.median(zte[:, top], 1)
    # (D) empilhamento: LightGBM prevê o alvo em escala a partir das previsões e das características
    m = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=2, random_state=0)
    m.fit(Xmat(np.where(tr)[0]), ttr)
    P["Empilhamento LightGBM"][te] = m.predict(Xmat(np.where(te)[0]))
    P["Mediana + empilhamento"][te] = (P["Mediana (todos)"][te] + P["Empilhamento LightGBM"][te]) / 2
    P["Mediana + pesos globais"][te] = (P["Mediana (todos)"][te] + P["Pesos globais (NNLS)"][te]) / 2
    print(pd.Timestamp(o).date(), tr.sum(), flush=True)

av = W.origem.isin(AVAL).values
for j, mname in enumerate(BASE):
    P[mname] = Z[:, j]
for c in ("Combinação (todos)",):                                      # combinação original do estudo, para referência
    x = d[d.modelo == c].set_index(K).q50.reindex(pd.MultiIndex.from_frame(W[K])).values
    P[c + " — original"] = (x - sn) / W.escala.values
R = W.loc[av, K + ["y", "escala"]].copy()
lin = []
mase = {}
for nome, z in P.items():
    e = np.abs(z[av] - t[av]); ape = 100 * e * R.escala.values / R.y.values
    ms = pd.Series(e).groupby(R.unique_id.values).mean(); mase[nome] = ms
    lin.append(dict(metodo=nome, MAPE=ape.mean(), erro_mediano=np.median(ape), dentro_10=100 * (ape <= 10).mean()))
T = pd.DataFrame(lin).set_index("metodo")
ref = mase["Ingênuo sazonal"]; med = mase["Mediana (todos)"]
for nome in T.index:
    T.loc[nome, "MASE_rel_ingenuo"] = np.exp(np.log(mase[nome] / ref).mean())
    r = np.log(mase[nome] / med)
    T.loc[nome, "razao_vs_mediana"] = np.exp(r.mean())
    ic = boot_ic(r.values); T.loc[nome, "IC_inf"] = np.exp(ic[0]); T.loc[nome, "IC_sup"] = np.exp(ic[1])
    T.loc[nome, "pct_series_melhor_que_mediana"] = 100 * (mase[nome] < med).mean()
T = T.sort_values("MASE_rel_ingenuo")
T.round(3).to_csv(f"{OUT}/t1_combinacao.csv"); print(T.round(3).to_string())
pd.DataFrame({k: v for k, v in P.items()}).assign(**{c: W[c].values for c in K + ["data_alvo", "y", "escala"]}).assign(sn=sn)[av].to_parquet(f"{OUT}/t1_previsoes_escala.parquet", index=False)
