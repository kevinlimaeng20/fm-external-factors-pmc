"""Avaliação: MASE, WQL, cobertura de 80%, escores relativos ao sazonal ingênuo, recortes e teste de Nemenyi (MCB)."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
from comum import carrega_painel, QUANTIS, QCOLS

os.makedirs("resultados", exist_ok=True)
p = carrega_painel()
real = p.set_index(["unique_id", "data"]).valor
meta = p.drop_duplicates("unique_id").set_index("unique_id")
meta["nivel"] = np.where(meta.geo == "Brasil", "Brasil", "UF")
meta["inicio"] = p.groupby("unique_id").data.min()

# escala do MASE: EAM do sazonal ingênuo no histórico até a origem
escala = {}
for u, g in p.groupby("unique_id"):
    s = g.set_index("data").valor
    escala[u] = (s - s.shift(12)).abs().expanding().mean()

prev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("previsoes/*.parquet"))
                  if not any(t in f for t in ("_valid", "_ctx", "_imputado", "teste_", "_X_", "Chronos-2"))], ignore_index=True)
prev = prev[[c for c in prev.columns if c in ["unique_id", "origem", "h", "data_alvo", "modelo"] + QCOLS]]
prev["origem"] = pd.to_datetime(prev.origem); prev["data_alvo"] = pd.to_datetime(prev.data_alvo)
prev["y"] = real.reindex(list(zip(prev.unique_id, prev.data_alvo))).values
prev["esc"] = [escala[u].get(o) for u, o in zip(prev.unique_id, prev.origem)]
assert prev.y.notna().all() and prev.esc.notna().all()

# conjunto comum: só (série, origem) presentes em todos os modelos
chave = ["unique_id", "origem", "h"]
nmod = prev.modelo.nunique()
cont = prev.groupby(chave).modelo.nunique()
prev = prev.merge(cont[cont == nmod].reset_index()[chave], on=chave)
print("modelos:", sorted(prev.modelo.unique()), "| previsões por modelo:", len(prev) // nmod)

Q = prev[QCOLS].values
y = prev.y.values[:, None]
alphas = np.array(QUANTIS)[None, :]
pinball = np.maximum(alphas * (y - Q), (alphas - 1) * (y - Q))
prev["ae"] = np.abs(prev.y - prev.q50)
prev["mase"] = prev.ae / prev.esc
prev["sql"] = 2 * pinball.mean(axis=1) / prev.esc          # perda quantílica escalada (aprox. CRPS / escala)
prev["wql_num"] = 2 * pinball.mean(axis=1)
prev["cob80"] = (prev.y >= prev.q10) & (prev.y <= prev.q90)
prev["ape"] = 100 * prev.ae / prev.y
prev = prev.join(meta[["nivel", "ativ", "geo", "inicio"]], on="unique_id")
prev.to_parquet("resultados/previsoes_avaliadas.parquet", index=False)


def resumo(df):
    g = df.groupby("modelo")
    r = pd.DataFrame({"MASE": g.mase.mean(), "SQL": g.sql.mean(),
                      "WQL": g.wql_num.sum() / g.y.sum(), "MAPE": g.ape.mean(),
                      "cob80": 100 * g.cob80.mean(), "n": g.size()})
    # escore relativo: média geométrica, entre séries, da razão do MASE (e SQL) do modelo sobre o sazonal ingênuo
    ps = df.groupby(["unique_id", "modelo"])[["mase", "sql"]].mean().unstack("modelo")
    for met in ("mase", "sql"):
        base = ps[met]["Seasonal naive"]
        r[f"rel_{met}"] = np.exp(np.log(ps[met].div(base, axis=0)).mean())
    return r.sort_values("rel_mase")


geral = resumo(prev)
geral.to_csv("resultados/tabela_geral.csv")
print(geral.round(3).to_string())

recortes = {"h1-3": prev.h <= 3, "h4-6": prev.h.between(4, 6), "h7-12": prev.h >= 7,
            "Brasil": prev.nivel == "Brasil", "UF": prev.nivel == "UF",
            "dezembro": prev.data_alvo.dt.month == 12,
            "alvos_2026 (após lançamento dos modelos)": prev.data_alvo >= "2026-01-01",
            "séries desde 2011": prev.inicio >= "2011-01-01"}
linhas = []
for nome, m in recortes.items():
    r = resumo(prev[m])[["rel_mase", "rel_sql", "MASE", "cob80", "n"]]
    r["recorte"] = nome
    linhas.append(r.reset_index())
pd.concat(linhas).to_csv("resultados/tabela_recortes.csv", index=False)
por_ativ = prev.groupby(["ativ", "modelo"]).mase.mean().unstack()
por_ativ.to_csv("resultados/mase_por_atividade.csv")
por_h = prev.groupby(["h", "modelo"]).mase.mean().unstack()
por_h.to_csv("resultados/mase_por_horizonte.csv")

# ranks por série (MASE médio) e Nemenyi (multiple comparisons with the best)
ps = prev.groupby(["unique_id", "modelo"]).mase.mean().unstack()
ranks = ps.rank(axis=1)
k, N = ps.shape[1], ps.shape[0]
q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
cd = q * np.sqrt(k * (k + 1) / (6 * N))
mr = ranks.mean().sort_values()
fried = stats.friedmanchisquare(*[ps[c] for c in ps.columns])
pd.DataFrame({"rank_medio": mr, "dentro_cd_do_melhor": mr <= mr.iloc[0] + cd}).to_csv("resultados/nemenyi_ranks.csv")
json.dump({"n_series": int(N), "k_modelos": int(k), "cd_nemenyi_95": float(cd), "friedman_p": float(fried.pvalue)},
          open("resultados/nemenyi_resumo.json", "w"), indent=1)
print("\nranks médios:\n", mr.round(2).to_string(), f"\nCD Nemenyi (95%) = {cd:.3f}; Friedman p = {fried.pvalue:.2e}")
