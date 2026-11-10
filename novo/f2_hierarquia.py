"""Teste 2: reconciliação hierárquica.
Os índices da PMC não somam diretamente; os pesos de cada agregação são estimados por mínimos quadrados não
negativos (sem intercepto) com dados até jun/2022. Restrições usadas (só as que ajustam bem):
  (a) varejo total Brasil = soma ponderada do varejo total das 27 UFs;
  (b) em cada geografia (Brasil + 12 UFs): varejo total = soma ponderada de 8 atividades;
  (c) em cada geografia: móveis e eletrodomésticos = móveis + eletrodomésticos (ponderados).
Reconciliação por projeção: y~ = y^ - W C'(C W C')^-1 C y^, com W = I (OLS), escala² (WLS) ou variância dos erros
já observados (WLS-erro, só nas 26 últimas origens). Base: mediana dos 12 modelos."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from base import OUT, painel, meta, boot_ic

p = painel(); M = meta(p)
larg = p.pivot(index="data", columns="unique_id", values="valor")
ids = list(larg.columns); pos = {u: i for i, u in enumerate(ids)}
CORTE = "2022-06-01"
A8 = ["Combustíveis e lubrificantes", "Hipermercados, supermercados, produtos alimentícios, bebidas e fumo", "Tecidos, vestuário e calçados",
      "Móveis e eletrodomésticos", "Artigos farmacêuticos, médicos, ortopédicos, de perfumaria e cosméticos", "Livros, jornais, revistas e papelaria",
      "Equipamentos e materiais para escritório, informática e comunicação", "Outros artigos de uso pessoal e doméstico"]
uid = lambda ativ, geo: M[(M.ativ == ativ) & (M.geo == geo)].index[0]
restr = []
tot = "Varejo total"
restr.append(("total BR = UFs", uid(tot, "Brasil"), [uid(tot, g) for g in sorted(M[M.ativ == tot].geo.unique()) if g != "Brasil"]))
for g in sorted(M[M.ativ != tot].geo.unique()):
    restr.append((f"total = atividades | {g}", uid(tot, g), [uid(a, g) for a in A8]))
    restr.append((f"móveis+eletro | {g}", uid("Móveis e eletrodomésticos", g), [uid("Móveis", g), uid("Eletrodomésticos", g)]))
C, diag = [], []
for nome, pai, filhos in restr:
    x = larg[[pai] + filhos].dropna()
    tr = x[:CORTE].iloc[-120:]; te = x[CORTE:].iloc[1:]
    w, _ = nnls(tr[filhos].values, tr[pai].values)
    f_tr = 100 * np.abs(tr[filhos].values @ w - tr[pai].values) / tr[pai].values
    f_te = 100 * np.abs(te[filhos].values @ w - te[pai].values) / te[pai].values
    diag.append(dict(restricao=nome, n_filhos=len(filhos), soma_pesos=w.sum(), erro_ajuste_pct=f_tr.mean(), erro_fora_amostra_pct=f_te.mean()))
    if f_tr.mean() < 1.0:
        c = np.zeros(len(ids)); c[pos[pai]] = 1; c[[pos[f] for f in filhos]] = -w; C.append(c)
D = pd.DataFrame(diag); D.round(3).to_csv(f"{OUT}/t2_restricoes.csv", index=False)
print(D.round(3).to_string()); C = np.array(C); print("restrições usadas:", len(C), "| séries envolvidas:", int((np.abs(C).sum(0) > 0).sum()))

d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
BASE = [m for m in d.modelo.unique() if not m.startswith("Combinação")]
b = d[d.modelo.isin(BASE)].groupby(["unique_id", "origem", "h", "data_alvo"]).agg(q50=("q50", "median"), y=("y", "first"), escala=("escala", "first")).reset_index()
origens = np.sort(b.origem.unique())


def projeta(yh, w):
    WC = (C * w).T                                                    # W C'
    return yh - WC @ np.linalg.solve(C @ WC, C @ yh)


out = []
for o in origens:
    conhec = b[b.data_alvo <= o]
    for h in range(1, 13):
        x = b[(b.origem == o) & (b.h == h)].set_index("unique_id").reindex(ids)
        yh = x.q50.values
        r = x[["y", "escala"]].copy(); r["origem"] = o; r["h"] = h; r["base"] = yh
        r["OLS"] = projeta(yh, np.ones(len(ids)))
        r["WLS (escala²)"] = projeta(yh, x.escala.values ** 2)
        if o >= origens[12]:
            k = conhec[conhec.h == h] if (conhec.h == h).sum() >= 171 * 6 else conhec
            v = ((k.q50 - k.y) ** 2).groupby(k.unique_id).mean().reindex(ids).values
            r["WLS (erro observado)"] = projeta(yh, 0.5 * v + 0.5 * x.escala.values ** 2)
        out.append(r.reset_index())
R = pd.concat(out, ignore_index=True)
R = R.join(M[["ativ", "nivel", "geo"]], on="unique_id")
env = set(np.array(ids)[np.abs(C).sum(0) > 0]); R["envolvida"] = R.unique_id.isin(env)
R["grupo"] = np.select([(R.ativ == tot) & (R.nivel == "Brasil"), (R.ativ == tot), R.nivel == "Brasil"], ["Varejo total Brasil", "Varejo total UF", "Atividade Brasil"], "Atividade UF")
R.to_parquet(f"{OUT}/t2_reconciliado.parquet", index=False)
lin = []
for rot, x in [("Todas as 171 séries, 38 origens", R), ("26 últimas origens", R[R.origem >= origens[12]])] + [(f"38 origens | {g}", x) for g, x in R.groupby("grupo")]:
    for m in ("base", "OLS", "WLS (escala²)", "WLS (erro observado)"):
        y = x.dropna(subset=[m])
        if rot.startswith("Todas") and m == "WLS (erro observado)" or (rot.startswith("38") and m == "WLS (erro observado)"):
            continue
        ms = ((y[m] - y.y).abs() / y.escala).groupby(y.unique_id).mean(); mb = ((y.base - y.y).abs() / y.escala).groupby(y.unique_id).mean()
        rr = np.log(ms / mb); ic = boot_ic(rr.values) if len(rr) > 5 else (np.nan, np.nan)
        lin.append(dict(recorte=rot, metodo=m, n_series=len(ms), MAPE=(100 * (y[m] - y.y).abs() / y.y).mean(), razao_vs_base=np.exp(rr.mean()),
                        IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_series_melhoram=100 * (ms < mb).mean()))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/t2_hierarquia.csv", index=False); print(T.round(4).to_string())
# incoerência das previsões-base: quanto o pai previsto difere da soma ponderada dos filhos previstos
inc = []
for (o, h), x in R.groupby(["origem", "h"]):
    yh = x.set_index("unique_id").base.reindex(ids).values
    inc.append(np.abs(C @ yh) / yh[[np.argmax(c) for c in C]])
print("incoerência média das previsões-base (%):", round(100 * np.mean(inc), 2))
