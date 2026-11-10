"""Q1, Q2, Q4: quão assertivas são as previsões, quais modelos são melhores e onde.
Usa as previsões do período de teste (origens jul/2022–ago/2025, h = 1..12, 171 séries)."""
import numpy as np
import pandas as pd
from base import OUT, painel, meta, FAMILIA, boot_ic

p = painel()
M = meta(p)
d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
d = d.join(M, on="unique_id")
REF = "Ingênuo sazonal"

# ---------------------------------------------------------------- 1. tabela de assertividade por modelo
def resumo(x):
    return pd.Series({
        "WAPE_%": 100 * x.erro.abs().sum() / x.y.sum(),
        "MAPE_%": x.ape.mean(), "erro_mediano_%": x.ape.median(), "erro_p90_%": x.ape.quantile(0.9),
        "dentro_5%": 100 * (x.ape <= 5).mean(), "dentro_10%": 100 * (x.ape <= 10).mean(),
        "vies_%": x.epct.mean(), "cobertura80_%": 100 * x.dentro80.mean(),
    })

tab = d.groupby("modelo").apply(resumo)
ps = d.groupby(["unique_id", "modelo"]).mase.mean().unstack()
rel = np.exp(np.log(ps.div(ps[REF], axis=0)).mean())
tab["MASE_rel_ingenuo"] = rel
lr = np.log(ps.div(ps[REF], axis=0))
ics = {m: np.exp(boot_ic(lr[m].values)) for m in ps.columns}
tab["IC95_inf"] = [ics[m][0] for m in tab.index]; tab["IC95_sup"] = [ics[m][1] for m in tab.index]
melhor = ps.drop(columns=[c for c in ps.columns if c.startswith("Combinação")]).idxmin(axis=1)
tab["%_series_onde_e_o_melhor_individual"] = 100 * melhor.value_counts(normalize=True).reindex(tab.index).fillna(0)
tab["familia"] = tab.index.map(FAMILIA)
tab = tab.sort_values("MASE_rel_ingenuo")
tab.round(3).to_csv(f"{OUT}/q1_assertividade_por_modelo.csv")
print(tab.round(2).to_string())

# ---------------------------------------------------------------- 2. por horizonte, mês-alvo, nível e atividade
CHAVE = ["Combinação (todos)", "LightGBM + fatores", "LightGBM", "Moirai 2.0", "Chronos-2", "TimesFM 2.5", "ARIMA", "ETS", REF]
hz = d[d.modelo.isin(CHAVE)].pivot_table(index="h", columns="modelo", values="ape", aggfunc="mean")[CHAVE]
hz.round(2).to_csv(f"{OUT}/q1_mape_por_horizonte.csv")
mes = d[d.modelo.isin(CHAVE)].assign(mes=d.data_alvo.dt.month).pivot_table(index="mes", columns="modelo", values="ape", aggfunc="mean")[CHAVE]
mes.round(2).to_csv(f"{OUT}/q1_mape_por_mes.csv")
niv = d[d.modelo.isin(CHAVE)].pivot_table(index="nivel", columns="modelo", values="ape", aggfunc="mean")[CHAVE]
niv.round(2).to_csv(f"{OUT}/q1_mape_por_nivel.csv")
at = d.pivot_table(index="ativ", columns="modelo", values="ape", aggfunc="mean")
at_res = pd.DataFrame({"melhor_modelo": at.idxmin(axis=1), "MAPE_melhor_%": at.min(axis=1),
                       "MAPE_combinacao_%": at["Combinação (todos)"], "MAPE_ingenuo_%": at[REF]})
at_res.sort_values("MAPE_melhor_%").round(2).to_csv(f"{OUT}/q4_melhor_por_atividade.csv")
print(hz.round(1).to_string()); print(mes.round(1).to_string()); print(niv.round(1).to_string()); print(at_res.round(1).to_string())

# ---------------------------------------------------------------- 3. famílias: estatístico x IA treinada x IA pré-treinada x combinação
fam = {"Melhor estatístico (ARIMA)": "ARIMA", "IA treinada (LightGBM + fatores)": "LightGBM + fatores",
       "IA pré-treinada (Moirai 2.0)": "Moirai 2.0", "IA pré-treinada (Chronos-2)": "Chronos-2", "Combinação (todos)": "Combinação (todos)"}
linhas = []
for a, ma in fam.items():
    for b, mb in fam.items():
        if a >= b:
            continue
        v = np.log(ps[ma] / ps[mb]).values
        lo, hi = np.exp(boot_ic(v))
        linhas.append(dict(A=a, B=b, razao_erro_A_sobre_B=np.exp(v.mean()), IC_inf=lo, IC_sup=hi, pct_series_A_melhor=100 * (v < 0).mean()))
pd.DataFrame(linhas).round(3).to_csv(f"{OUT}/q2_familias_pareado.csv", index=False)
print(pd.DataFrame(linhas).round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. escolher o modelo por série compensa?
# escolhe, em cada série, o modelo com menor erro nas 12 primeiras origens do teste e avalia nas 26 seguintes
orig = sorted(d.origem.unique()); corte = orig[11]
cand = [m for m in ps.columns if not m.startswith("Combinação")]
a1 = d[(d.origem <= corte) & d.modelo.isin(cand)].groupby(["unique_id", "modelo"]).mase.mean().unstack()
escolha = a1.idxmin(axis=1)
b2 = d[d.origem > corte].groupby(["unique_id", "modelo"]).mase.mean().unstack()
sel = pd.Series([b2.loc[u, escolha[u]] for u in b2.index], index=b2.index)
oraculo = b2[cand].min(axis=1)
comp = pd.DataFrame({
    "Escolha por série (passado recente)": np.exp(np.log(sel / b2[REF]).mean()),
    "Combinação (todos)": np.exp(np.log(b2["Combinação (todos)"] / b2[REF]).mean()),
    "LightGBM + fatores (para todas)": np.exp(np.log(b2["LightGBM + fatores"] / b2[REF]).mean()),
    "Oráculo (melhor modelo a posteriori)": np.exp(np.log(oraculo / b2[REF]).mean()),
}, index=["MASE_rel_ingenuo"]).T
comp.round(3).to_csv(f"{OUT}/q4_escolha_por_serie.csv")
print(comp.round(3)); print(escolha.value_counts())
