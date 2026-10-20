"""Figuras dos fatores externos (mesmo estilo de 08_figuras.py).
Fig5: ablação por grupo (MASE relativo ao LightGBM sem fatores); Fig6: participação de cada grupo nas
contribuições SHAP por atividade."""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "figuras"; MM = 1 / 25.4
INK, INK2, MUTED, GRID = "#111111", "#3f3e3b", "#6f6e69", "#dedcd5"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
ATIV = {"Artigos farmacêuticos, médicos, ortopédicos, de perfumaria e cosméticos": "Pharmacy and cosmetics",
        "Combustíveis e lubrificantes": "Fuels and lubricants", "Eletrodomésticos": "Household appliances",
        "Equipamentos e materiais para escritório, informática e comunicação": "Office, IT and communication",
        "Hipermercados e supermercados": "Hyper- and supermarkets",
        "Hipermercados, supermercados, produtos alimentícios, bebidas e fumo": "Food, beverages and tobacco",
        "Livros, jornais, revistas e papelaria": "Books and stationery", "Móveis": "Furniture",
        "Móveis e eletrodomésticos": "Furniture and appliances", "Outros artigos de uso pessoal e doméstico":
        "Other personal and household", "Tecidos, vestuário e calçados": "Textiles, apparel and footwear",
        "Varejo total": "Total retail", "All series": "All series"}


def salva(fig, nome):
    for ext in ("pdf", "eps"):
        fig.savefig(f"{OUT}/{nome}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 5: ablação
ab = pd.read_csv("resultados/fatores_ablacao.csv").set_index("variante")
grupos = ["Economy", "Calendar", "Policy", "Climate", "World"]
fig, ax = plt.subplots(figsize=(90 * MM, 62 * MM))
y = np.arange(len(grupos))[::-1]
for k, (pref, cor, mk, lab) in enumerate([("Base plus ", "#1f5fae", "o", "Own history + this group only"),
                                         ("All minus ", "#d4561f", "s", "All factors except this group")]):
    r = ab.loc[[pref + g for g in grupos]]
    yy = y + (0.15 if k == 0 else -0.15)
    ax.hlines(yy, r.ic_inf, r.ic_sup, color=cor, lw=1.3)
    ax.scatter(r.razao_vs_base, yy, color=cor, marker=mk, s=22, zorder=3, edgecolor="white", linewidth=0.6, label=lab)
ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.axvline(ab.loc["All external factors", "razao_vs_base"], color=INK2, lw=0.8, ls=(0, (1, 1.5)))
ax.text(ab.loc["All external factors", "razao_vs_base"], len(grupos) - 0.35, " all factors", fontsize=7, color=INK2,
        va="bottom")
ax.set_yticks(y, grupos); ax.set_ylim(-0.6, len(grupos) - 0.2)
ax.set_xlabel("MASE relative to LightGBM without external factors")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=1)
salva(fig, "Fig5")

# ------------------------------------------------------------------ Fig. 6: participação SHAP por atividade
sh = pd.read_csv("resultados/fatores_shap_grupos.csv", index_col=0)
ordem = ["Own history", "Series identity and season", "Economy", "Policy", "World", "Calendar", "Climate"]
cores = ["#c9c7bf", "#8f8d86", "#1f5fae", "#7b4fa3", "#b88200", "#14875c", "#d4561f"]
sh = sh[ordem]
sh = pd.concat([sh.loc[["All series"]], sh.drop(index="All series").sort_values("Economy")])
fig, ax = plt.subplots(figsize=(190 * MM, 80 * MM))
yy = np.arange(len(sh))[::-1]
esq = np.zeros(len(sh))
for c, cor in zip(ordem, cores):
    ax.barh(yy, sh[c], left=esq, color=cor, height=0.7, edgecolor="white", linewidth=1, label=c)
    if c == "Economy":
        for yi, l, v in zip(yy, esq, sh[c]):
            ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, color="white")
    esq += sh[c].values
ax.set_yticks(yy, [ATIV[i] for i in sh.index]); ax.set_xlim(0, 100)
ax.set_xlabel("Share of mean absolute SHAP contribution (%), median model with all external factors")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=4)
salva(fig, "Fig6")

print("figuras ok")
