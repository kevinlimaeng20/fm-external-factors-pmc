"""Avaliação dos modelos com fatores externos: acurácia, custo de estoque, ganhos pareados por atividade,
ablação por grupo de fatores e importância (SHAP).
Saídas em resultados/: fatores_geral.csv, fatores_pareado.csv, fatores_por_atividade.csv, fatores_ablacao.csv,
fatores_shap_grupos.csv, fatores_shap_top.csv, fatores_recortes.csv"""
import glob, json, os
import numpy as np
import pandas as pd
from comum import carrega_painel, QUANTIS, QCOLS

BASE = "Seasonal naive"
CH = ["unique_id", "origem", "h"]
p = carrega_painel(imputa=False)
real = p.set_index(["unique_id", "data"]).valor
meta = p.drop_duplicates("unique_id").set_index("unique_id")[["geo", "geo_cod", "ativ", "ativ_cod"]]
escala = {u: (g.set_index("data").valor - g.set_index("data").valor.shift(12)).abs().expanding().mean()
          for u, g in p.groupby("unique_id")}

ARQ = {"Seasonal naive": "estatisticos.parquet", "LightGBM global": "LightGBM_global.parquet",
       "Moirai 2.0 small": "Moirai_20_small.parquet", "TimesFM 2.5": "TimesFM_25.parquet",
       "LightGBM + external factors": "LightGBM_X_real.parquet",
       "LightGBM + external factors (oracle)": "LightGBM_X_oraculo.parquet",
       "LightGBM + selected factors": "LightGBM_X_selecionado.parquet",
       "Chronos-2": "Chronos-2_uni.parquet", "Chronos-2 + external factors": "Chronos-2_real.parquet",
       "Chronos-2 + external factors (oracle)": "Chronos-2_oraculo.parquet",
       "Chronos-2 + selected factors": "Chronos-2_sel.parquet"}
ds = []
for m, f in ARQ.items():
    if not os.path.exists(f"previsoes/{f}"):
        print("faltando:", f); continue
    d = pd.read_parquet(f"previsoes/{f}")
    if "modelo" in d and d.modelo.nunique() > 1:
        d = d[d.modelo == m]
    d["modelo"] = m
    ds.append(d)
# membros das combinações originais (estatísticos e fundacionais do estudo base)
outros = [pd.read_parquet(f) for f in sorted(glob.glob("previsoes/*.parquet"))
          if not any(t in f for t in ("_valid", "_ctx", "_imputado", "teste_", "_X_", "Chronos-2"))]
d = pd.concat(ds + outros, ignore_index=True)
d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
d = d[CH + ["data_alvo", "modelo"] + QCOLS].drop_duplicates(CH + ["modelo"])
memb = json.load(open("resultados/combinacoes.json"))["membros"]
novos = [m for m in ["LightGBM + external factors", "Chronos-2", "Chronos-2 + external factors"] if m in d.modelo.unique()]
combos = {"Combination: all models": memb["Combination: all models"],
          "Combination: all models + external factors": memb["Combination: all models"] + novos}
for n, mm in combos.items():
    c = d[d.modelo.isin(mm)].groupby(CH + ["data_alvo"])[QCOLS].median().reset_index(); c["modelo"] = n
    d = pd.concat([d, c], ignore_index=True)
MODS = [m for m in list(ARQ) + list(combos) if m in d.modelo.unique()]
d = d[d.modelo.isin(MODS)]
d[QCOLS] = np.sort(d[QCOLS].values, axis=1)
cont = d.groupby(CH).modelo.nunique()
d = d.merge(cont[cont == len(MODS)].reset_index()[CH], on=CH)
d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
d["esc"] = [escala[u].get(o) for u, o in zip(d.unique_id, d.origem)]
Q = d[QCOLS].values; y = d.y.values[:, None]; a = np.array(QUANTIS)[None, :]
d["mase"] = np.abs(d.y - d.q50) / d.esc
d["sql"] = 2 * np.maximum(a * (y - Q), (a - 1) * (y - Q)).mean(axis=1) / d.esc
d["cob80"] = (d.y >= d.q10) & (d.y <= d.q90)
for b in (1, 4, 9):     # custo newsvendor (sobra = 1, falta = b), quantil crítico interpolado
    pc = b / (b + 1); k = min(int(np.searchsorted(QUANTIS, pc, side="right")) - 1, 7)
    Qd = Q[:, k] + (pc - QUANTIS[k]) / (QUANTIS[k + 1] - QUANTIS[k]) * (Q[:, k + 1] - Q[:, k])
    d[f"custo{b}"] = np.maximum(Qd - d.y, 0) + b * np.maximum(d.y - Qd, 0)
print("previsões por modelo:", len(d) // len(MODS), "| modelos:", len(MODS))


def rel(x, met, ref=BASE, boot=False):
    ps = x.groupby(["unique_id", "modelo"])[met].mean().unstack()
    lr = np.log(ps.div(ps[ref], axis=0))
    r = np.exp(lr.mean())
    if not boot:
        return r
    rng = np.random.default_rng(2026)
    bs = np.array([lr.iloc[rng.integers(0, len(lr), len(lr))].mean().values for _ in range(1000)])
    return r, pd.Series(np.exp(np.percentile(bs, 2.5, 0)), lr.columns), pd.Series(np.exp(np.percentile(bs, 97.5, 0)), lr.columns)


r, lo, hi = rel(d, "mase", boot=True)
g = pd.DataFrame({"rel_mase": r, "ic_inf": lo, "ic_sup": hi, "rel_sql": rel(d, "sql"),
                  "cob80": 100 * d.groupby("modelo").cob80.mean(),
                  "custo_1_1": rel(d, "custo1"), "custo_4_1": rel(d, "custo4"), "custo_9_1": rel(d, "custo9"),
                  "dez_mase": rel(d[d.data_alvo.dt.month == 12], "mase"),
                  "dez_custo_4_1": rel(d[d.data_alvo.dt.month == 12], "custo4")})
g = g.loc[MODS].sort_values("rel_mase")
g.to_csv("resultados/fatores_geral.csv")
pd.set_option("display.width", 250)
print(g.round(3).to_string())

# ---------------------------------------------------------------- ganhos pareados (com fatores vs sem)
PARES = [("LightGBM + external factors", "LightGBM global"), ("LightGBM + external factors (oracle)", "LightGBM global"),
         ("LightGBM + external factors (oracle)", "LightGBM + external factors"),
         ("LightGBM + selected factors", "LightGBM global"),
         ("Chronos-2 + external factors", "Chronos-2"), ("Chronos-2 + external factors (oracle)", "Chronos-2"),
         ("Chronos-2 + selected factors", "Chronos-2"),
         ("Combination: all models + external factors", "Combination: all models")]
PARES = [x for x in PARES if x[0] in MODS and x[1] in MODS]
rec = {"All": d.data_alvo.notna(), "h1-3": d.h <= 3, "h4-6": d.h.between(4, 6), "h7-12": d.h >= 7,
       "December": d.data_alvo.dt.month == 12, "Targets 2026": d.data_alvo >= "2026-01-01"}
lin = []
for novo, ant in PARES:
    for nr, m in rec.items():
        x = d[m & d.modelo.isin([novo, ant])]
        for met in ("mase", "sql", "custo4"):
            ps = x.groupby(["unique_id", "modelo"])[met].mean().unstack()
            lr = np.log(ps[novo] / ps[ant])
            rng = np.random.default_rng(2026)
            bs = [lr.iloc[rng.integers(0, len(lr), len(lr))].mean() for _ in range(1000)]
            lin.append(dict(novo=novo, referencia=ant, recorte=nr, metrica=met, razao=np.exp(lr.mean()),
                            ic_inf=np.exp(np.percentile(bs, 2.5)), ic_sup=np.exp(np.percentile(bs, 97.5)),
                            pct_series_melhora=100 * (lr < 0).mean()))
par = pd.DataFrame(lin)
par.to_csv("resultados/fatores_pareado.csv", index=False)
print(par[par.metrica == "mase"].round(3).to_string(index=False))

# ---------------------------------------------------------------- por atividade e por região
d["ativ"] = d.unique_id.map(meta.ativ)
d["regiao"] = d.unique_id.map(meta.geo_cod.astype(int)).map(
    lambda c: "Brazil" if c == 1 else {1: "North", 2: "Northeast", 3: "Southeast", 4: "South", 5: "Centre-West"}[c // 10])
lin = []
for novo, ant in PARES:
    for col in ("ativ", "regiao"):
        for k, x in d[d.modelo.isin([novo, ant])].groupby(col):
            ps = x.groupby(["unique_id", "modelo"]).mase.mean().unstack()
            lr = np.log(ps[novo] / ps[ant])
            lin.append(dict(novo=novo, referencia=ant, dimensao=col, grupo=k, n_series=len(lr),
                            razao_mase=np.exp(lr.mean()), pct_series_melhora=100 * (lr < 0).mean()))
pd.DataFrame(lin).to_csv("resultados/fatores_por_atividade.csv", index=False)

# ---------------------------------------------------------------- ablação por grupo (modelo da mediana)
if os.path.exists("resultados/ablacao_previsoes.parquet"):
    ab = pd.read_parquet("resultados/ablacao_previsoes.parquet")
    ab["origem"] = pd.to_datetime(ab.origem)
    ab = ab.merge(d[d.modelo == BASE][CH].drop_duplicates(), on=CH)
    ps = ab.groupby(["unique_id", "variante"]).ae.mean().unstack()
    nv = d[d.modelo == BASE].groupby("unique_id").mase.mean()
    out = []
    for v in ps.columns:
        lr = np.log(ps[v] / ps["Base (own history only)"])
        rng = np.random.default_rng(2026)
        bs = [lr.iloc[rng.integers(0, len(lr), len(lr))].mean() for _ in range(1000)]
        out.append(dict(variante=v, rel_mase_naive=np.exp(np.log(ps[v] / nv.reindex(ps.index)).mean()),
                        razao_vs_base=np.exp(lr.mean()), ic_inf=np.exp(np.percentile(bs, 2.5)),
                        ic_sup=np.exp(np.percentile(bs, 97.5)), pct_series_melhora=100 * (lr < 0).mean()))
    ab_t = pd.DataFrame(out).sort_values("razao_vs_base")
    ab_t.to_csv("resultados/fatores_ablacao.csv", index=False)
    print(ab_t.round(4).to_string(index=False))

# ---------------------------------------------------------------- SHAP
if os.path.exists("resultados/shap_grupos.parquet"):
    s = pd.read_parquet("resultados/shap_grupos.parquet")
    s["ativ"] = s.unique_id.map(meta.ativ)
    gcols = [c for c in s.columns if c not in ("unique_id", "h", "mes_alvo", "ativ")]
    tot = s[gcols].sum(axis=1)
    sh = s[gcols].div(tot, axis=0).mul(100)
    sh["ativ"] = s.ativ
    t1 = pd.concat([sh[gcols].mean().rename("All series").to_frame().T, sh.groupby("ativ")[gcols].mean()])
    t1.to_csv("resultados/fatores_shap_grupos.csv")
    print(t1.round(1).to_string())
    v = pd.read_parquet("resultados/shap_variaveis_por_serie.parquet")
    v["ativ"] = v.index.map(meta.ativ)
    top = v.groupby("ativ").mean().apply(lambda r: ", ".join(r.sort_values(ascending=False).index[:3]), axis=1)
    allv = v.drop(columns="ativ").mean().sort_values(ascending=False)
    allv.to_csv("resultados/fatores_shap_variaveis.csv")
    top.to_csv("resultados/fatores_shap_top.csv")
    print(allv.head(15).round(4).to_string()); print(top.to_string())
