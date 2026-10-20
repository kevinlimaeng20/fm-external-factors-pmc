"""Tabelas LaTeX da seção de fatores externos, geradas dos resultados (sem transcrição manual)."""
import json
import numpy as np
import pandas as pd

ATIV = {"Artigos farmacêuticos, médicos, ortopédicos, de perfumaria e cosméticos": "Pharmacy and cosmetics",
        "Combustíveis e lubrificantes": "Fuels and lubricants", "Eletrodomésticos": "Household appliances",
        "Equipamentos e materiais para escritório, informática e comunicação": "Office, IT and communication",
        "Hipermercados e supermercados": "Hyper- and supermarkets",
        "Hipermercados, supermercados, produtos alimentícios, bebidas e fumo": "Food, beverages and tobacco",
        "Livros, jornais, revistas e papelaria": "Books and stationery", "Móveis": "Furniture",
        "Móveis e eletrodomésticos": "Furniture and appliances", "Outros artigos de uso pessoal e doméstico":
        "Other personal and household", "Tecidos, vestuário e calçados": "Textiles, apparel and footwear",
        "Varejo total": "Total retail"}
VAR = {"desocup": "Unemployment rate", "confianca": "Consumer confidence", "inadimpl_pf": "Household loan default",
       "credito_pf": "Household credit", "massa_renda": "Labour income", "ipca_12m": "Inflation (12 m)",
       "salmin_real": "Real minimum wage", "cambio": "Exchange rate", "renda_comprometida": "Debt-service ratio",
       "ibcbr": "Economic activity (IBC-Br)", "selic": "Policy rate (Selic)", "transf_renda": "Cash transfers",
       "soja": "Soybean price", "petroleo": "Oil price", "carnaval_mes": "Carnival month",
       "eleicao_municipal": "Municipal election year"}


def b(vals, fmt="{:.3f}"):
    m = np.nanmin(vals)
    return [("\\textbf{" + fmt.format(v) + "}") if np.isclose(v, m) else fmt.format(v) for v in vals]


# ---------------------------------------------------------------- tabela principal
g = pd.read_csv("resultados/fatores_geral.csv", index_col=0)
ORD = [("LightGBM global", "LightGBM, own history only"), ("LightGBM + external factors", "LightGBM + all factors"),
       ("LightGBM + selected factors", "LightGBM + selected factors"),
       ("LightGBM + external factors (oracle)", "LightGBM + all factors (oracle)"),
       ("Chronos-2", "Chronos-2, own history only"), ("Chronos-2 + external factors", "Chronos-2 + factors"),
       ("Chronos-2 + selected factors", "Chronos-2 + selected factors"),
       ("Combination: all models", "Combination (all models)"),
       ("Combination: all models + external factors", "Combination (all models + factor models)")]
ORD = [(k, n) for k, n in ORD if k in g.index]
t = g.loc[[k for k, _ in ORD]]
cols = ["rel_mase", "rel_sql", "custo_1_1", "custo_4_1", "custo_9_1", "dez_custo_4_1"]
bb = {c: b(t[c].values) for c in cols}
lin = []
for i, (k, n) in enumerate(ORD):
    r = t.loc[k]
    lin.append(f"{n} & {bb['rel_mase'][i]} & [{r.ic_inf:.3f}, {r.ic_sup:.3f}] & {bb['rel_sql'][i]} & {r.cob80:.1f} & "
               f"{bb['custo_1_1'][i]} & {bb['custo_4_1'][i]} & {bb['custo_9_1'][i]} & {bb['dez_custo_4_1'][i]} \\\\")
    if k in ("LightGBM + external factors (oracle)", "Chronos-2 + selected factors"):
        lin.append("\\addlinespace")
open("tabelas/tab_fatores.tex", "w").write("\n".join(lin) + "\n")

# ---------------------------------------------------------------- ganhos pareados
p = pd.read_csv("resultados/fatores_pareado.csv")
PAR = [("LightGBM + external factors", "LightGBM global", "LightGBM: all factors vs none"),
       ("LightGBM + selected factors", "LightGBM global", "LightGBM: selected factors vs none"),
       ("LightGBM + external factors (oracle)", "LightGBM + external factors", "LightGBM: oracle vs available factors"),
       ("Chronos-2 + external factors", "Chronos-2", "Chronos-2: factors vs none"),
       ("Chronos-2 + selected factors", "Chronos-2", "Chronos-2: selected factors vs none"),
       ("Combination: all models + external factors", "Combination: all models", "Combination: with vs without factor models")]
REC = ["All", "h1-3", "h4-6", "h7-12", "December"]
lin = []
for novo, ref, nome in PAR:
    x = p[(p.novo == novo) & (p.referencia == ref)]
    if x.empty:
        continue
    cel = []
    for r in REC:
        y = x[(x.recorte == r) & (x.metrica == "mase")].iloc[0]
        sig = "$^{*}$" if (y.ic_sup < 1 or y.ic_inf > 1) else ""
        cel.append(f"{y.razao:.3f}{sig}")
    c4 = x[(x.recorte == "All") & (x.metrica == "custo4")].iloc[0]
    a = x[(x.recorte == "All") & (x.metrica == "mase")].iloc[0]
    sig = "$^{*}$" if (c4.ic_sup < 1 or c4.ic_inf > 1) else ""
    lin.append(f"{nome} & " + " & ".join(cel) + f" & {c4.razao:.3f}{sig} & {a.pct_series_melhora:.0f} \\\\")
open("tabelas/tab_pareado.tex", "w").write("\n".join(lin) + "\n")

# ---------------------------------------------------------------- por atividade
a = pd.read_csv("resultados/fatores_por_atividade.csv")
a = a[(a.novo == "LightGBM + external factors") & (a.referencia == "LightGBM global") & (a.dimensao == "ativ")]
sh = pd.read_csv("resultados/fatores_shap_grupos.csv", index_col=0)
top = pd.read_csv("resultados/fatores_shap_top.csv", index_col=0).iloc[:, 0]
lin = []
for _, r in a.sort_values("razao_mase").iterrows():
    tv = ", ".join(VAR.get(v.split("@")[0], v) for v in top[r.grupo].split(", "))
    lin.append(f"{ATIV[r.grupo]} & {r.n_series} & {r.razao_mase:.3f} & {r.pct_series_melhora:.0f} & "
               f"{sh.loc[r.grupo, 'Economy']:.0f} & {tv} \\\\")
open("tabelas/tab_atividade.tex", "w").write("\n".join(lin) + "\n")
print(open("tabelas/tab_fatores.tex").read()); print(open("tabelas/tab_pareado.tex").read())
print(open("tabelas/tab_atividade.tex").read())
