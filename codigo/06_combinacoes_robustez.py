"""Combinações (membros escolhidos na validação ou definidos a priori), intervalos de confiança por bootstrap
de séries e análises de robustez dos modelos fundacionais (contexto de 60/120 meses e pandemia imputada)."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
from comum import carrega_painel, QUANTIS, QCOLS

os.makedirs("resultados", exist_ok=True)
p = carrega_painel(imputa=False)
real = p.set_index(["unique_id", "data"]).valor
escala = {}
for u, g in p.groupby("unique_id"):
    s = g.set_index("data").valor
    escala[u] = (s - s.shift(12)).abs().expanding().mean()
CHAVE = ["unique_id", "origem", "h"]
BASE = "Seasonal naive"


def carrega(padrao_inclui, exclui=("_valid", "_ctx", "_imputado", "teste_", "_X_", "Chronos-2")):
    fs = [f for f in sorted(glob.glob("previsoes/*.parquet"))
          if all(t in f for t in padrao_inclui)
          and not any(t in f for t in exclui if not any(t in q for q in padrao_inclui))]
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
    return d[CHAVE + ["data_alvo", "modelo"] + QCOLS]


def metricas(d):
    d = d.copy()
    d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
    d["esc"] = [escala[u].get(o) for u, o in zip(d.unique_id, d.origem)]
    Q = d[QCOLS].values; y = d.y.values[:, None]; a = np.array(QUANTIS)[None, :]
    pin = np.maximum(a * (y - Q), (a - 1) * (y - Q))
    d["mase"] = np.abs(d.y - d.q50) / d.esc
    d["sql"] = 2 * pin.mean(axis=1) / d.esc
    d["cob80"] = (d.y >= d.q10) & (d.y <= d.q90)
    return d


def comum(d):
    n = d.modelo.nunique()
    c = d.groupby(CHAVE).modelo.nunique()
    return d.merge(c[c == n].reset_index()[CHAVE], on=CHAVE)


def rel(d, met="mase"):
    ps = d.groupby(["unique_id", "modelo"])[met].mean().unstack()
    return np.exp(np.log(ps.div(ps[BASE], axis=0)).mean()), ps


def combina(d, membros, nome):
    c = d[d.modelo.isin(membros)].groupby(CHAVE + ["data_alvo"])[QCOLS].median().reset_index()
    c[QCOLS] = np.sort(c[QCOLS].values, axis=1)
    c["modelo"] = nome
    return c


# ------------------------------------------------------------------ escolha dos membros na validação
val = comum(metricas(carrega(["_valid"])))
rv, _ = rel(val)
rv = rv.drop(BASE).sort_values()
print("validação (MASE relativo):\n", rv.round(3).to_string())
top3 = list(rv.index[:3])
fms = ["Chronos-Bolt small", "Chronos-Bolt base", "TimesFM 2.5", "Moirai 2.0 small"]
todos = list(rv.index)
combos = {"Combination: top-3 in validation": top3, "Combination: all models": todos,
          "Combination: foundation models": fms}
json.dump({"validacao_rel_mase": rv.round(4).to_dict(), "membros": combos}, open("resultados/combinacoes.json", "w"),
          indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ teste: modelos + combinações
te = carrega([""])
te = pd.concat([te] + [combina(te, m, n) for n, m in combos.items()], ignore_index=True)
te = comum(metricas(te))
r_mase, ps = rel(te, "mase")
r_sql, ps_sql = rel(te, "sql")
g = te.groupby("modelo")
tab = pd.DataFrame({"rel_mase": r_mase, "rel_sql": r_sql, "MASE": g.mase.mean(), "cob80": 100 * g.cob80.mean()})

# IC 95% por bootstrap de séries (1.000 réplicas) para o escore relativo
rng = np.random.default_rng(2026)
lr = np.log(ps.div(ps[BASE], axis=0))
boot = np.array([lr.iloc[rng.integers(0, len(lr), len(lr))].mean().values for _ in range(1000)])
tab["ic95_inf"] = np.exp(np.percentile(boot, 2.5, axis=0))
tab["ic95_sup"] = np.exp(np.percentile(boot, 97.5, axis=0))
# Nemenyi
rk = ps.rank(axis=1)
k, N = ps.shape[1], ps.shape[0]
cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
tab["rank_medio"] = rk.mean()
tab["no_grupo_do_melhor"] = tab.rank_medio <= tab.rank_medio.min() + cd
tab = tab.sort_values("rel_mase")
tab.to_csv("resultados/tabela_com_combinacoes.csv")
print(f"\nteste (CD Nemenyi = {cd:.3f}, {N} séries, {k} métodos):\n", tab.round(3).to_string())
dez = te[te.data_alvo.dt.month == 12]
print("\ndezembro:\n", rel(dez)[0].sort_values().round(3).to_string())

# ------------------------------------------------------------------ robustez dos fundacionais (e ETS/ARIMA imputados)
linhas = []
for var in ["_ctx60", "_ctx120", "_imputado"]:
    try:
        v = carrega([var])
    except ValueError:
        continue
    v["modelo"] = v.modelo + " [" + var.strip("_") + "]"
    b = carrega([""])
    b = b[b.modelo.isin([m.split(" [")[0] for m in v.modelo.unique()] + [BASE])]
    d = comum(metricas(pd.concat([b, v], ignore_index=True)))
    rm, _ = rel(d); rs, _ = rel(d, "sql")
    for m in v.modelo.unique():
        orig = m.split(" [")[0]
        linhas.append(dict(variante=var.strip("_"), modelo=orig, rel_mase_base=rm[orig], rel_mase_variante=rm[m],
                           rel_sql_base=rs[orig], rel_sql_variante=rs[m]))
if linhas:
    rob = pd.DataFrame(linhas)
    rob.to_csv("resultados/robustez.csv", index=False)
    print("\nrobustez:\n", rob.round(3).to_string(index=False))
