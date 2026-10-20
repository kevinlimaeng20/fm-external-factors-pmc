"""Baixa (opcional) as tabelas da PMC/IBGE no SIDRA e monta o painel de 171 séries mensais.

Uso: python codigo/01_dados.py [--baixar]
  --baixar  baixa de novo as tabelas 8882 (atividades: Brasil e UFs) e 8880 (varejo total: Brasil e UFs);
            sem a opção, usa os arquivos já salvos em dados/ (extraídos em 17/10/2026).
Índice de volume de vendas do comércio varejista (variável 7169, número-índice 2022 = 100, sem ajuste sazonal).
Saída: dados/painel_pmc.csv (geo, geo_cod, ativ, ativ_cod, periodo, valor, data, serie)."""
import json, subprocess, sys
import pandas as pd

URL = "https://apisidra.ibge.gov.br/values/t/{t}/{n}/v/7169/p/all/c11046/56734{c}"
ARQ = {"pmc8882_br": URL.format(t=8882, n="n1/all", c="/c85/all"),
       "pmc8882_uf": URL.format(t=8882, n="n3/all", c="/c85/all"),
       "pmc8880": URL.format(t=8880, n="n1/all/n3/all", c="")}
FIM = "2026-08-01"

if "--baixar" in sys.argv:
    for nome, u in ARQ.items():
        subprocess.run(["curl", "-s", "-m", "600", "-o", f"dados/{nome}.json", u], check=True)
        print("baixado:", nome)

linhas = []
for nome in ARQ:
    d = pd.DataFrame(json.load(open(f"dados/{nome}.json"))[1:])
    x = pd.DataFrame({"geo": d.D1N, "geo_cod": d.D1C.astype(int), "periodo": d.D3C.astype(int),
                      "valor": pd.to_numeric(d.V, errors="coerce")})
    if "D5C" in d:
        x["ativ"], x["ativ_cod"] = d.D5N, d.D5C
    else:
        x["ativ"], x["ativ_cod"] = "Varejo total", "total"
    linhas.append(x)
p = pd.concat(linhas, ignore_index=True).dropna(subset=["valor"])
p["data"] = pd.to_datetime(p.periodo.astype(str), format="%Y%m")
p = p[p.data <= FIM]
p["serie"] = p.geo + " | " + p.ativ
# mantém séries completas: sem lacunas do primeiro mês publicado até o fim
ok = []
for s, g in p.groupby("serie"):
    meses = pd.date_range(g.data.min(), FIM, freq="MS")
    if len(g) == len(meses) and g.data.nunique() == len(meses):
        ok.append(s)
p = p[p.serie.isin(ok)].sort_values(["geo_cod", "ativ_cod", "data"])
p = p[["geo", "geo_cod", "ativ", "ativ_cod", "periodo", "valor", "data", "serie"]]
p.to_csv("dados/painel_pmc.csv", index=False)
print(f"{p.serie.nunique()} séries, {len(p)} observações, {p.data.min().date()} a {p.data.max().date()}")
