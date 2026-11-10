"""Q6: avalia a simulação ano a ano (previsão feita em dezembro para os 12 meses do ano seguinte).
Medidas por ano-alvo: erro percentual médio mensal (MAPE), erro do total anual (o "orçamento" do ano),
% de meses dentro de ±5% e ±10%, cobertura da faixa de 80%, viés; e por regime econômico."""
import glob, json
import numpy as np
import pandas as pd
from base import OUT, painel, meta, QCOLS

p = painel(); M = meta(p)
real = p.set_index(["unique_id", "data"]).valor
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/anual_*.parquet")) if "limpo" not in f], ignore_index=True)
d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
d[QCOLS] = np.sort(d[QCOLS].values, axis=1)
# combinação: mediana dos modelos não ingênuos disponíveis em cada série/origem
mods = [m for m in d.modelo.unique() if m != "Ingênuo sazonal"]
c = d[d.modelo.isin(mods)].groupby(["unique_id", "origem", "h", "data_alvo"])[QCOLS].median().reset_index(); c["modelo"] = "Combinação"
d = pd.concat([d, c], ignore_index=True)
# mesma base de comparação: só pares série-origem que todos os modelos têm
n = d.modelo.nunique(); k = d.groupby(["unique_id", "origem"]).modelo.nunique()
d = d.merge(k[k == n].reset_index()[["unique_id", "origem"]], on=["unique_id", "origem"])
d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
d = d.dropna(subset=["y"])
d["ano"] = d.data_alvo.dt.year
d["ape"] = 100 * (d.q50 - d.y).abs() / d.y
d["epct"] = 100 * (d.q50 - d.y) / d.y
d["dentro80"] = (d.y >= d.q10) & (d.y <= d.q90)
d.to_parquet(f"{OUT}/anual_avaliado.parquet", index=False)
print("modelos:", sorted(d.modelo.unique()), "| séries por ano:", d.groupby("ano").unique_id.nunique().to_dict())

# mensal
mape = d.pivot_table(index="ano", columns="modelo", values="ape", aggfunc="mean")
dentro10 = d.assign(ok=d.ape <= 10).pivot_table(index="ano", columns="modelo", values="ok", aggfunc="mean") * 100
cob = d.pivot_table(index="ano", columns="modelo", values="dentro80", aggfunc="mean") * 100
vies = d.pivot_table(index="ano", columns="modelo", values="epct", aggfunc="mean")
# total anual (anos completos)
an = d[d.groupby(["unique_id", "origem", "modelo"]).h.transform("count") == 12]
tot = an.groupby(["modelo", "ano", "unique_id"]).agg(prev=("q50", "sum"), real=("y", "sum")).reset_index()
tot["ape_total"] = 100 * (tot.prev - tot.real).abs() / tot.real
mape_tot = tot.pivot_table(index="ano", columns="modelo", values="ape_total", aggfunc="mean")
for nome, t in (("mape_mensal", mape), ("dentro10_mensal", dentro10), ("cobertura80", cob), ("vies", vies), ("erro_total_anual", mape_tot)):
    t.round(2).to_csv(f"{OUT}/q6_{nome}_por_ano.csv")
print("MAPE mensal por ano:\n", mape.round(1).to_string())
print("Erro do total anual (%):\n", mape_tot.round(1).to_string())

REG = {"Crescimento 2005–2008": range(2005, 2009), "Crise 2009": [2009], "Expansão 2010–2014": range(2010, 2015),
       "Recessão 2015–2016": [2015, 2016], "Recuperação 2017–2019": range(2017, 2020), "Pandemia 2020–2021": [2020, 2021],
       "Pós-pandemia 2022–2026": range(2022, 2027)}
lin = []
for r, anos in REG.items():
    x = d[d.ano.isin(list(anos))]; xt = tot[tot.ano.isin(list(anos))]
    for m, g in x.groupby("modelo"):
        lin.append(dict(regime=r, modelo=m, MAPE_mensal=g.ape.mean(), dentro_10=100 * (g.ape <= 10).mean(),
                        erro_total_anual=xt[xt.modelo == m].ape_total.mean(), cobertura80=100 * g.dentro80.mean(), vies=g.epct.mean()))
reg = pd.DataFrame(lin)
reg.round(2).to_csv(f"{OUT}/q6_por_regime.csv", index=False)
print(reg.pivot(index="regime", columns="modelo", values="MAPE_mensal").loc[list(REG)].round(1).to_string())
print(reg.pivot(index="regime", columns="modelo", values="erro_total_anual").loc[list(REG)].round(1).to_string())
# geral (excluindo 2020–2021) e por nível
g = d[~d.ano.isin([2020, 2021])]
geral = g.groupby("modelo").agg(MAPE_mensal=("ape", "mean"), erro_mediano=("ape", "median"), dentro_5=("ape", lambda v: 100 * (v <= 5).mean()),
                                dentro_10=("ape", lambda v: 100 * (v <= 10).mean()), cobertura80=("dentro80", lambda v: 100 * v.mean()))
geral["erro_total_anual"] = tot[~tot.ano.isin([2020, 2021])].groupby("modelo").ape_total.mean()
geral["vitorias_anos"] = mape.drop(index=[2020, 2021], errors="ignore").idxmin(axis=1).value_counts().reindex(geral.index).fillna(0)
geral = geral.sort_values("MAPE_mensal")
geral.round(2).to_csv(f"{OUT}/q6_geral.csv"); print(geral.round(2).to_string())
niv = g.join(M[["nivel"]], on="unique_id").pivot_table(index="nivel", columns="modelo", values="ape", aggfunc="mean")
niv.round(2).to_csv(f"{OUT}/q6_por_nivel.csv"); print(niv.round(1).to_string())
