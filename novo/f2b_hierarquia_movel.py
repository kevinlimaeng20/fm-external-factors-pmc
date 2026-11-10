"""Teste 2 (variante): pesos das agregações reestimados em cada origem (últimos 24 meses), pois os pesos da PMC
mudam ao longo do tempo; só entram restrições com erro de ajuste < 0,5%. Projeção WLS (escala²)."""
import numpy as np, pandas as pd
from scipy.optimize import nnls
from base import OUT, painel, meta, boot_ic
import importlib.util, io, contextlib
p = painel(); M = meta(p); larg = p.pivot(index="data", columns="unique_id", values="valor"); ids = list(larg.columns); pos = {u: i for i, u in enumerate(ids)}
R0 = pd.read_parquet(f"{OUT}/t2_reconciliado.parquet")
A8 = ["Combustíveis e lubrificantes", "Hipermercados, supermercados, produtos alimentícios, bebidas e fumo", "Tecidos, vestuário e calçados", "Móveis e eletrodomésticos",
      "Artigos farmacêuticos, médicos, ortopédicos, de perfumaria e cosméticos", "Livros, jornais, revistas e papelaria",
      "Equipamentos e materiais para escritório, informática e comunicação", "Outros artigos de uso pessoal e doméstico"]
uid = lambda a, g: M[(M.ativ == a) & (M.geo == g)].index[0]; tot = "Varejo total"
restr = [(uid(tot, "Brasil"), [uid(tot, g) for g in sorted(M[M.ativ == tot].geo.unique()) if g != "Brasil"])]
for g in sorted(M[M.ativ != tot].geo.unique()):
    restr += [(uid(tot, g), [uid(a, g) for a in A8]), (uid("Móveis e eletrodomésticos", g), [uid("Móveis", g), uid("Eletrodomésticos", g)])]
out = []; nres = []
for o, xo in R0.groupby("origem"):
    C = []
    for pai, filhos in restr:
        tr = larg.loc[:o, [pai] + filhos].dropna().iloc[-24:]
        w, _ = nnls(tr[filhos].values, tr[pai].values)
        if (100 * np.abs(tr[filhos].values @ w - tr[pai].values) / tr[pai].values).mean() < 0.5:
            c = np.zeros(len(ids)); c[pos[pai]] = 1; c[[pos[f] for f in filhos]] = -w; C.append(c)
    C = np.array(C); nres.append(len(C))
    for h, x in xo.groupby("h"):
        x = x.set_index("unique_id").reindex(ids); yh = x.base.values; WC = (C * x.escala.values ** 2).T
        x["movel"] = yh - WC @ np.linalg.solve(C @ WC, C @ yh); out.append(x.reset_index())
R = pd.concat(out); print("restrições por origem (média):", np.mean(nres))
lin = []
for rot, x in [("Todas", R)] + list(R.groupby("grupo")):
    ms = ((x.movel - x.y).abs() / x.escala).groupby(x.unique_id).mean(); mb = ((x.base - x.y).abs() / x.escala).groupby(x.unique_id).mean()
    rr = np.log(ms / mb); ic = boot_ic(rr.values) if len(rr) > 5 else (np.nan, np.nan)
    lin.append(dict(recorte=rot, n=len(ms), MAPE_base=(100 * (x.base - x.y).abs() / x.y).mean(), MAPE_reconc=(100 * (x.movel - x.y).abs() / x.y).mean(),
                    razao=np.exp(rr.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_melhoram=100 * (ms < mb).mean()))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/t2_hierarquia_movel.csv", index=False); print(T.round(4).to_string())
