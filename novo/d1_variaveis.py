"""Q3: quais variáveis se relacionam com a demanda e quais ajudam a prever.
(a) Importância (SHAP) de cada variável no LightGBM com fatores, por ano de reajuste (estabilidade).
(b) Defasagem (lead-lag): correlação entre o crescimento anual das vendas e cada fator k meses antes.
(c) Causalidade de Granger (o fator ajuda a prever o crescimento das vendas além do próprio passado?)
    nas 12 séries do Brasil, 2012–2019 (antes da pandemia) e 2012–2026.
"""
import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from base import OUT, painel, meta

NOME = {"ipca_12m": "Inflação 12m", "selic": "Selic", "cambio": "Câmbio", "ibcbr": "Atividade (IBC-Br)", "credito_pf": "Crédito famílias",
        "inadimpl_pf": "Inadimplência", "confianca": "Confiança consumidor", "salmin_real": "Salário mínimo real",
        "renda_comprometida": "Renda comprometida", "desocup": "Desemprego", "massa_renda": "Massa de renda",
        "transf_renda": "Bolsa Família/Aux. Brasil", "aux_emergencial": "Auxílio Emergencial", "eleicao_geral": "Ano eleição geral",
        "eleicao_municipal": "Ano eleição municipal", "mes_eleicao": "Mês de eleição", "icms_comb_teto": "Teto ICMS combust.",
        "incerteza_br": "Incerteza Brasil", "incerteza_global": "Incerteza global", "petroleo": "Petróleo", "soja": "Soja",
        "milho": "Milho", "carne": "Carne", "temp_anom": "Temperatura (anomalia)", "temp_max_anom": "Temp. máxima (anomalia)",
        "chuva_anom": "Chuva (anomalia)", "dias_uteis": "Dias úteis", "pascoa_mes": "Páscoa", "carnaval_mes": "Carnaval"}
info = pd.read_csv("dados/covariaveis_meta.csv").set_index("variavel")

# ---------------------------------------------------------------- (a) SHAP por variável e por ano
lin = []
for f in sorted(glob.glob("resultados/shap_sinal_*.parquet")):
    ano = int(f[-12:-8])
    s = pd.read_parquet(f)
    cols = [c for c in s.columns if "@" in c and not c.startswith("x:")]
    imp = s[cols].abs().mean()
    for c, v in imp.items():
        lin.append(dict(ano=ano, variavel=c.split("@")[0], atributo=c, imp=v))
sh = pd.DataFrame(lin)
tab = sh.groupby(["variavel", "ano"]).imp.sum().unstack()
tab["media"] = tab.mean(axis=1)
tab = tab.sort_values("media", ascending=False)
tab["participacao_%"] = 100 * tab.media / tab.media.sum()
tab["grupo"] = tab.index.map(info.grupo)
# estabilidade: posição no ranking em cada ano
rank = tab[[c for c in tab.columns if isinstance(c, (int, np.integer))]].rank(ascending=False)
tab["pior_posicao"] = rank.max(axis=1); tab["melhor_posicao"] = rank.min(axis=1)
tab.index = tab.index.map(NOME)
tab.round(5).to_csv(f"{OUT}/q3_importancia_variaveis.csv")
print(tab[["grupo", "participacao_%", "melhor_posicao", "pior_posicao"]].round(1).to_string())

# ---------------------------------------------------------------- (b) defasagens
p = painel(); M = meta(p)
cov = pd.read_parquet("dados/covariaveis.parquet").set_index(["geo_cod", "data"])
p = p.sort_values(["unique_id", "data"])
p["g"] = p.groupby("unique_id").valor.transform(lambda s: np.log(s).diff(12))
p = p[(p.data >= "2012-01-01") & ~p.data.between("2020-03-01", "2021-12-01")]
VARS = [v for v in info.index if info.loc[v, "grupo"] != "Calendar" and v not in ("eleicao_geral", "eleicao_municipal", "mes_eleicao", "aux_emergencial", "icms_comb_teto")]
res = []
for v in VARS:
    for k in (0, 1, 2, 3, 6, 9, 12):
        cc = []
        for u, g in p.groupby("unique_id"):
            gc = int(M.at[u, "geo_cod"])
            x = cov[v].reindex(list(zip([gc] * len(g), g.data - pd.DateOffset(months=k)))).values
            ok = np.isfinite(x) & np.isfinite(g.g.values)
            if ok.sum() > 36:
                cc.append(np.corrcoef(x[ok], g.g.values[ok])[0, 1])
        res.append(dict(variavel=NOME[v], defasagem_meses=k, correlacao_mediana=np.median(cc), pct_series_mesmo_sinal=100 * max(np.mean(np.array(cc) > 0), np.mean(np.array(cc) < 0))))
ll = pd.DataFrame(res)
piv = ll.pivot(index="variavel", columns="defasagem_meses", values="correlacao_mediana")
piv["max_abs"] = piv.abs().max(axis=1)
piv["melhor_defasagem"] = piv.drop(columns="max_abs").abs().idxmax(axis=1)
piv = piv.sort_values("max_abs", ascending=False)
piv.round(3).to_csv(f"{OUT}/q3_correlacao_defasagens.csv"); ll.round(3).to_csv(f"{OUT}/q3_correlacao_defasagens_longo.csv", index=False)
print(piv.round(2).to_string())

# ---------------------------------------------------------------- (c) Granger nas séries do Brasil
pb = painel(); pb = pb[pb.geo == "Brasil"].sort_values(["unique_id", "data"])
pb["g"] = pb.groupby("unique_id").valor.transform(lambda s: np.log(s).diff(12))
gr = []
for jan, nome_j in ((("2012-01-01", "2019-12-01"), "2012–2019"), (("2012-01-01", "2026-08-01"), "2012–2026")):
    for v in VARS:
        sig = 0; tot = 0
        for u, g in pb.groupby("unique_id"):
            g = g[(g.data >= jan[0]) & (g.data <= jan[1])]
            x = cov[v].reindex(list(zip([1] * len(g), g.data))).values
            df = pd.DataFrame({"y": g.g.values, "x": x}).dropna()
            if len(df) < 48 or df.x.std() == 0:
                continue
            try:
                r = grangercausalitytests(df[["y", "x"]].diff().dropna(), maxlag=6)
                pmin = min(r[l][0]["ssr_ftest"][1] for l in r)
                tot += 1; sig += pmin < 0.05 / 6  # correção de Bonferroni para as 6 defasagens
            except Exception:
                pass
        gr.append(dict(periodo=nome_j, variavel=NOME[v], series_testadas=tot, pct_series_granger_sig=100 * sig / tot if tot else np.nan))
g = pd.DataFrame(gr).pivot(index="variavel", columns="periodo", values="pct_series_granger_sig").sort_values("2012–2019", ascending=False)
g.round(1).to_csv(f"{OUT}/q3_granger_brasil.csv"); print(g.round(0).to_string())
