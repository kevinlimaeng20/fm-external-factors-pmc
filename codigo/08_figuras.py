"""Figuras do artigo (IJPE): PDF vetorial + EPS, fontes embutidas (Liberation Sans ~ Arial), legíveis em cinza.
Fig1: escore relativo (MASE) com IC 95%; Fig2: MASE relativo por horizonte; Fig3: custo relativo por razão de custo;
Fig4: dezembro (nível de serviço atingido e custo relativo)."""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comum import carrega_painel, QCOLS

OUT = "figuras"; os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
INK, INK2, MUTED, GRID = "#111111", "#3f3e3b", "#6f6e69", "#dedcd5"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
NOMES = {"Combination: all models": "Combination (all)", "Combination: top-3 in validation": "Combination (top-3 val.)",
         "Combination: foundation models": "Combination (FMs)", "LightGBM global": "Global LightGBM",
         "Moirai 2.0 small": "Moirai 2.0", "TimesFM 2.5": "TimesFM 2.5", "ARIMA": "ARIMA", "ETS": "ETS",
         "Chronos-Bolt base": "Chronos-Bolt base", "Chronos-Bolt small": "Chronos-Bolt small", "Theta": "Theta",
         "Seasonal naive": "Seasonal naive"}
# estilos fixos por método (cor + marcador + traço: identidade nunca só pela cor)
EST = {"Combination: all models": ("#1f5fae", "o", "-"), "Combination: top-3 in validation": ("#1f5fae", "s", (0, (4, 1.5))),
       "LightGBM global": ("#14875c", "D", "-"), "Moirai 2.0 small": ("#d4561f", "^", "-"),
       "TimesFM 2.5": ("#b88200", "v", (0, (1, 1))), "ARIMA": ("#7b4fa3", "x", (0, (5, 1, 1, 1))),
       "ETS": ("#6f6e69", "+", (0, (3, 2))), "Chronos-Bolt base": ("#d4561f", "P", (0, (2, 2)))}
CHAVES = list(EST)


def salva(fig, nome):
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/{nome}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1: escore relativo com IC
t = pd.read_csv("resultados/tabela_com_combinacoes.csv", index_col=0).sort_values("rel_mase", ascending=False)
t = t.drop("Seasonal naive")
fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
y = np.arange(len(t))
cores = ["#1f5fae" if i.startswith("Combination") else ("#d4561f" if any(k in i for k in ("Chronos", "Moirai", "TimesFM"))
         else ("#14875c" if "LightGBM" in i else "#6f6e69")) for i in t.index]
ax.hlines(y, t.ic95_inf, t.ic95_sup, color=cores, lw=1.4)
MK = {"#1f5fae": "o", "#14875c": "D", "#d4561f": "^", "#6f6e69": "s"}
for yi, xv, c in zip(y, t.rel_mase, cores):
    ax.scatter(xv, yi, color=c, marker=MK[c], s=24, zorder=3, edgecolor="white", linewidth=0.6)
ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.set_yticks(y, [NOMES[i] for i in t.index])
ax.set_xlabel("Relative MASE (seasonal naive = 1)")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
leg = [Line2D([], [], color=c, marker=MK[c], lw=1.4, ms=4) for c in ("#1f5fae", "#14875c", "#d4561f", "#6f6e69")]
ax.legend(leg, ["Combination", "Global ML", "Foundation model", "Statistical"], loc="upper left",
          bbox_to_anchor=(0, -0.16), ncol=2, fontsize=7)
salva(fig, "Fig1")

# ------------------------------------------------------------------ Fig. 2: MASE relativo por horizonte
p = carrega_painel(imputa=False)
real = p.set_index(["unique_id", "data"]).valor
esc = {u: (g.set_index("data").valor - g.set_index("data").valor.shift(12)).abs().expanding().mean() for u, g in p.groupby("unique_id")}
fs = [f for f in sorted(glob.glob("previsoes/*.parquet")) if not any(s in f for s in ("_valid", "_ctx", "_imputado", "teste_", "_X_", "Chronos-2"))]
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
CH = ["unique_id", "origem", "h"]
memb = json.load(open("resultados/combinacoes.json"))["membros"]
d = pd.concat([d] + [d[d.modelo.isin(m)].groupby(CH + ["data_alvo"])[QCOLS].median().reset_index().assign(modelo=n)
                     for n, m in memb.items()], ignore_index=True)
d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
d["mase"] = (d.y - d.q50).abs() / [esc[u].get(o) for u, o in zip(d.unique_id, d.origem)]
ph = d.groupby(["h", "unique_id", "modelo"]).mase.mean().unstack()
relh = np.exp(np.log(ph.div(ph["Seasonal naive"], axis=0)).groupby(level="h").mean())
relh.to_csv("resultados/rel_mase_por_horizonte.csv")
fig, ax = plt.subplots(figsize=(90 * MM, 68 * MM))
for k in CHAVES:
    c, m, ls = EST[k]
    ax.plot(relh.index, relh[k], color=c, marker=m, ls=ls, lw=1.1, ms=3.6, mfc="white" if m not in "x+" else c,
            mew=0.9, label=NOMES[k])
ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.set_xticks(range(1, 13)); ax.set_xlabel("Forecast horizon (months ahead)"); ax.set_ylabel("Relative MASE")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, handlelength=2.4, columnspacing=1)
salva(fig, "Fig2")

# ------------------------------------------------------------------ Fig. 3: custo relativo por razão de custo
g = pd.read_csv("resultados/decisao_geral.csv")
ordem = ["1:1", "2:1", "3:1", "4:1", "9:1"]
fig, ax = plt.subplots(figsize=(90 * MM, 68 * MM))
for k in CHAVES:
    c, m, ls = EST[k]
    s = g[g.modelo == k].set_index("razao").reindex(ordem)
    ax.plot(range(5), s.custo_rel, color=c, marker=m, ls=ls, lw=1.1, ms=3.6, mfc="white" if m not in "x+" else c,
            mew=0.9, label=NOMES[k])
ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.set_xticks(range(5), [f"{r}\n(p={b / (b + 1):.2f})" for r, b in zip(ordem, (1, 2, 3, 4, 9))])
ax.set_xlabel("Shortage-to-overage cost ratio (critical fractile)")
ax.set_ylabel("Relative inventory cost")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2, handlelength=2.4, columnspacing=1)
salva(fig, "Fig3")

# ------------------------------------------------------------------ Fig. 4: dezembro, razão 4:1 (alvo 80%)
r = pd.read_csv("resultados/decisao_recortes.csv")
r4 = r[r.razao == "4:1"]
dz = r4[r4.recorte == "dezembro"].set_index("modelo")
todos = g[g.razao == "4:1"].set_index("modelo")
ks = [k for k in CHAVES] + ["Seasonal naive"]
fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
yy = np.arange(len(ks))[::-1]
ax = axes[0]
ax.barh(yy, dz.loc[ks, "servico_ciclo"], color="#1f5fae", height=0.55, label="December")
ax.scatter(todos.loc[ks, "servico_ciclo"], yy, color=INK, marker="|", s=90, zorder=3, label="All months")
ax.axvline(80, color="#d4561f", lw=1, ls=(0, (3, 2)), label="Target (80%)")
ax.set_xlim(60, 100); ax.set_yticks(yy, [NOMES[k] for k in ks]); ax.set_xlabel("Achieved cycle service level (%)")
ax.set_title("(a) Service level, cost ratio 4:1", loc="left", fontsize=8)
ax.grid(axis="y", visible=False); ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=7)
ax = axes[1]
ax.hlines(yy, todos.loc[ks, "custo_rel"], dz.loc[ks, "custo_rel"], color=MUTED, lw=1)
ax.scatter(todos.loc[ks, "custo_rel"], yy, color="white", edgecolor=INK, s=24, zorder=3, label="All months")
ax.scatter(dz.loc[ks, "custo_rel"], yy, color="#d4561f", s=24, zorder=3, label="December")
ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.set_yticks(yy, []); ax.set_xlabel("Relative inventory cost (seasonal naive = 1)")
ax.set_title("(b) Relative cost, cost ratio 4:1", loc="left", fontsize=8)
ax.grid(axis="y", visible=False); ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=7)
fig.subplots_adjust(wspace=0.05)
salva(fig, "Fig4")
print("figuras ok")
