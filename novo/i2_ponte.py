"""Ideia 6 (teste complementar, "teto"): modelo-ponte simples para as 12 séries do Brasil, 1 mês à frente.
Regressão em janela crescente: crescimento anual das vendas no mês o+1 ~ crescimento em o, o-1 e média de 3 meses
(+ um indicador rápido NO PRÓPRIO mês o+1, isto é, supondo que ele já estivesse disponível — o caso mais favorável).
Se nem assim o indicador reduz o erro, ele não tem informação nova sobre o varejo. Avaliação: alvos jul/2022–ago/2026."""
import numpy as np, pandas as pd
from base import OUT, painel, meta
p = painel(); M = meta(p); T = pd.read_csv(f"{OUT}/i1_indicadores_rapidos.csv", index_col=0, parse_dates=True)
cov = pd.read_parquet("dados/covariaveis.parquet"); cb = cov[cov.geo_cod == 1].set_index("data")
for v in ("cambio", "selic", "confianca", "ibcbr", "credito_pf"): T[v + " (já no estudo)"] = cb[v]
br = M[M.nivel == "Brasil"].index; lin = []
for u in br:
    g = np.log(p[p.unique_id == u].set_index("data").valor).diff(12).dropna()
    D = pd.DataFrame({"y": g, "l1": g.shift(1), "l2": g.shift(2), "m3": g.shift(1).rolling(3).mean()})
    D = D[~D.index.to_series().between("2020-03-01", "2021-12-01")]
    for ind in [None] + list(T.columns):
        X = D.copy()
        if ind: X["x"] = T[ind].reindex(X.index)
        X = X.dropna(); cols = [c for c in X.columns if c != "y"]; e = []
        for t in X.index[X.index >= "2022-08-01"]:
            tr = X[X.index < t]
            if len(tr) < 60: continue
            A = np.column_stack([np.ones(len(tr)), tr[cols].values]); b = np.linalg.lstsq(A, tr.y.values, rcond=None)[0]
            e.append(abs(X.loc[t, "y"] - np.r_[1, X.loc[t, cols].values] @ b))
        lin.append(dict(unique_id=u, ativ=M.at[u, "ativ"][:32], indicador=ind or "(sem indicador)", mae=np.mean(e) if e else np.nan))
R = pd.DataFrame(lin); b = R[R.indicador == "(sem indicador)"].set_index("unique_id").mae
R["razao"] = R.mae / R.unique_id.map(b)
S = R[R.indicador != "(sem indicador)"].groupby("indicador").agg(razao_media=("razao", lambda v: np.exp(np.log(v.dropna()).mean())), series_que_melhoram=("razao", lambda v: int((v < 0.99).sum())),
                                                                   melhor_caso=("razao", "min")).sort_values("razao_media")
S["onde_mais_ajuda"] = [R[(R.indicador == i)].sort_values("razao").ativ.iloc[0] for i in S.index]
S.round(3).to_csv(f"{OUT}/i2_ponte.csv"); print(S.round(3).to_string())
