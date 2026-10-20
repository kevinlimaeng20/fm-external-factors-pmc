"""Gera as tabelas LaTeX do artigo diretamente dos resultados (evita transcrição manual)."""
import os
import numpy as np
import pandas as pd

os.makedirs("tabelas", exist_ok=True)
N = {"Combination: all models": "Combination (all models)", "Combination: top-3 in validation": "Combination (top-3 in validation)",
     "Combination: foundation models": "Combination (foundation models)", "LightGBM global": "Global LightGBM",
     "Moirai 2.0 small": "Moirai 2.0 small", "TimesFM 2.5": "TimesFM 2.5", "ARIMA": "ARIMA", "ETS": "ETS",
     "Chronos-Bolt base": "Chronos-Bolt base", "Chronos-Bolt small": "Chronos-Bolt small", "Theta": "Theta",
     "Seasonal naive": "Seasonal naive"}
GRUPO = {"Seasonal naive": "Statistical", "ETS": "Statistical", "Theta": "Statistical", "ARIMA": "Statistical",
         "LightGBM global": "Global ML", "Chronos-Bolt small": "Foundation", "Chronos-Bolt base": "Foundation",
         "TimesFM 2.5": "Foundation", "Moirai 2.0 small": "Foundation"}
f2 = lambda x: f"{x:.3f}"
f1 = lambda x: f"{x:.1f}"


def bold_min(vals):
    m = min(vals)
    return [f"\\textbf{{{v:.3f}}}" if np.isclose(v, m) else f"{v:.3f}" for v in vals]


# ------------------------------------------------------------ Tabela de acurácia
t = pd.read_csv("resultados/tabela_com_combinacoes.csv", index_col=0)
linhas = []
rm, rs = bold_min(t.rel_mase), bold_min(t.rel_sql)
for i, (m, r) in enumerate(t.iterrows()):
    grp = "Combination" if m.startswith("Combination") else GRUPO[m]
    top = "$\\checkmark$" if r.no_grupo_do_melhor else ""
    ic = "--" if m == "Seasonal naive" else f"[{r.ic95_inf:.3f}, {r.ic95_sup:.3f}]"
    linhas.append(f"{N[m]} & {grp} & {rm[i]} & {ic} & {rs[i]} & {r.MASE:.3f} & {r.cob80:.1f} & {r.rank_medio:.2f} & {top} \\\\")
open("tabelas/tab_acuracia.tex", "w").write("\n".join(linhas) + "\n")

# ------------------------------------------------------------ Tabela de decisão (custo relativo por razão)
g = pd.read_csv("resultados/decisao_geral.csv")
ordem = ["1:1", "2:1", "3:1", "4:1", "9:1"]
cr = g.pivot(index="modelo", columns="razao", values="custo_rel")[ordem]
sv = g.pivot(index="modelo", columns="razao", values="servico_ciclo")[ordem]
cp = g.pivot(index="modelo", columns="razao", values="custo_pct_demanda")[ordem]
ic = g[g.razao == "4:1"].set_index("modelo")[["ic95_inf", "ic95_sup"]]
cr = cr.loc[t.index]
cols = {c: bold_min(cr[c].values) for c in ordem}
linhas = []
for i, m in enumerate(cr.index):
    ci = "--" if m == "Seasonal naive" else f"[{ic.loc[m, 'ic95_inf']:.3f}, {ic.loc[m, 'ic95_sup']:.3f}]"
    linhas.append(f"{N[m]} & " + " & ".join(cols[c][i] for c in ordem) + f" & {ci} & {cp.loc[m, '4:1']:.2f} \\\\")
open("tabelas/tab_decisao.tex", "w").write("\n".join(linhas) + "\n")
linhas = [f"{N[m]} & " + " & ".join(f1(sv.loc[m, c]) for c in ordem) + " \\\\" for m in cr.index]
open("tabelas/tab_servico.tex", "w").write("\n".join(linhas) + "\n")

# ------------------------------------------------------------ Antecedência e dezembro (razão 4:1)
r = pd.read_csv("resultados/decisao_recortes.csv")
r4 = r[r.razao == "4:1"].pivot(index="modelo", columns="recorte", values="custo_rel").loc[t.index]
s4 = r[r.razao == "4:1"].pivot(index="modelo", columns="recorte", values="servico_ciclo").loc[t.index]
rh = pd.read_csv("resultados/rel_mase_por_horizonte.csv", index_col="h")
cols = {c: bold_min(r4[c].values) for c in ["h1-3", "h4-6", "h7-12", "dezembro"]}
linhas = []
for i, m in enumerate(r4.index):
    linhas.append(f"{N[m]} & " + " & ".join(cols[c][i] for c in ["h1-3", "h4-6", "h7-12", "dezembro"])
                  + f" & {s4.loc[m, 'dezembro']:.1f} \\\\")
open("tabelas/tab_antecedencia.tex", "w").write("\n".join(linhas) + "\n")

# ------------------------------------------------------------ Robustez
rob = pd.read_csv("resultados/robustez.csv")
lab = {"ctx60": "Last 60 months", "ctx120": "Last 120 months", "imputado": "Pandemic months imputed"}
linhas = []
for m in ["Moirai 2.0 small", "TimesFM 2.5", "Chronos-Bolt base", "ARIMA", "ETS"]:
    x = rob[rob.modelo == m].set_index("variante")
    base = x.rel_mase_base.iloc[0]
    cel = [f"{x.loc[v, 'rel_mase_variante']:.3f}" if v in x.index else "--" for v in ["ctx120", "ctx60", "imputado"]]
    linhas.append(f"{N[m]} & {base:.3f} & " + " & ".join(cel) + " \\\\")
open("tabelas/tab_robustez.tex", "w").write("\n".join(linhas) + "\n")

# ------------------------------------------------------------ Contaminação: alvos de 2026 vs até 2025 (h >= 5)
d = pd.read_parquet("resultados/previsoes_avaliadas.parquet")
d = d[d.h >= 5]
d["per"] = np.where(d.data_alvo >= "2026-01-01", "2026", "ate2025")


def rel(x):
    ps = x.groupby(["unique_id", "modelo"]).mase.mean().unstack()
    return np.exp(np.log(ps.div(ps["Seasonal naive"], axis=0)).mean())


c = pd.DataFrame({k: rel(gg) for k, gg in d.groupby("per")})
c.to_csv("resultados/contaminacao_h5_12.csv")
linhas = []
for m in ["Moirai 2.0 small", "TimesFM 2.5", "Chronos-Bolt base", "Chronos-Bolt small", "LightGBM global", "ARIMA", "ETS", "Theta"]:
    linhas.append(f"{N[m]} & {GRUPO[m]} & {c.loc[m, 'ate2025']:.3f} & {c.loc[m, '2026']:.3f} & {c.loc[m, '2026'] - c.loc[m, 'ate2025']:+.3f} \\\\")
open("tabelas/tab_contaminacao.tex", "w").write("\n".join(linhas) + "\n")
print("tabelas ok"); print(c.round(3))
