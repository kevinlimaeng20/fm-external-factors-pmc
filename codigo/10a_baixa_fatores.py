"""Baixa os dados brutos dos fatores externos para dados/ext/ (extraídos em 19/10/2026).
Fontes: IBGE/SIDRA (IPCA; desocupação e massa de rendimento da PNAD Contínua), Ipeadata (séries do Banco Central,
FMI, Fecomércio-SP, CNC e salário mínimo), policyuncertainty.com (EPU Brasil e global), NASA POWER (clima diário
das 27 capitais) e MDS/VIS DATA (Bolsa Família e Auxílio Brasil por UF).
Uso: python codigo/10a_baixa_fatores.py   (depois: python codigo/10_covariaveis.py)"""
import csv, json, os, subprocess, time

E = "dados/ext"
for d in ("", "/ipea", "/clima", "/mds"):
    os.makedirs(E + d, exist_ok=True)


def baixa(url, destino, tentativas=4):
    for k in range(tentativas):
        r = subprocess.run(["curl", "-s", "-m", "300", "-o", destino, "-w", "%{http_code}", url],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.stdout == "200" and os.path.getsize(destino) > 0:
            return
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"falhou: {url}")


S = "https://apisidra.ibge.gov.br/values/t/"
baixa(S + "1737/n1/all/v/63/p/all", f"{E}/ipca.json")                      # IPCA, variação mensal
baixa(S + "4099/n3/all/v/4099/p/all", f"{E}/desocup_uf.json")              # desocupação trimestral por UF
baixa(S + "6381/n1/all/v/4099/p/all", f"{E}/desocup_br_mensal.json")       # desocupação, trimestre móvel, Brasil
baixa(S + "5606/n3/all/n1/all/p/all", f"{E}/massa_uf.json")                # massa de rendimento real
IPEA = ["BM12_ERV12", "BM12_TJOVER12", "SGS12_IBCBR12", "BM12_CRLCAPF12", "BM12_CRLINPF12", "FCESP12_IIC12",
        "IFS12_PETROLEUM12", "IFS12_SOJAGP12", "IFS12_MAIZE12", "IFS12_BEEFB12", "MTE12_SALMIN12",
        "GAC12_SALMINRE12", "CNC12_PEICRC12"]
for c in IPEA:
    baixa(f"https://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{c}')", f"{E}/ipea/{c}.json")
baixa("https://www.policyuncertainty.com/media/Brazil_Policy_Uncertainty_Data.xlsx", f"{E}/epu_brasil.xlsx")
baixa("https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx", f"{E}/epu_global.xlsx")
for r in csv.DictReader(open("dados/capitais.csv")):
    baixa("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,PRECTOTCORR&community=AG"
          f"&longitude={r['lon']}&latitude={r['lat']}&start=20000101&end=20260930&format=CSV&header=false",
          f"{E}/clima/{r['uf']}.csv")

# MDS (Solr): soma por UF e mês dos valores municipais
CAMPOS = [("*", ["valor_repassado_bolsa_familia", "qtd_familias_beneficiarias_bolsa_familia_i"]),
          ("[202101 TO 202312]", ["pab_valor_pago_d", "pab_extraordinario_valor_pago_d"])]
res = {}
for r in csv.DictReader(open("dados/capitais.csv")):
    uf = r["uf"]; res[uf] = {}
    for meses, campos in CAMPOS:
        q = f"sigla_uf:{uf}" + ("" if meses == "*" else f" AND anomes_s:{meses}")
        args = ["curl", "-s", "-m", "200", "-G", "https://aplicacoes.mds.gov.br/sagi/servicos/misocial",
                "--data-urlencode", f"q={q}", "--data-urlencode", "rows=0", "--data-urlencode", "wt=json",
                "--data-urlencode", "stats=true", "--data-urlencode", "stats.facet=anomes_s"]
        for f in campos:
            args += ["--data-urlencode", f"stats.field={f}"]
        for k in range(5):
            try:
                d = json.loads(subprocess.run(args, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout)
                d = d["stats"]["stats_fields"]; break
            except Exception:
                time.sleep(5 * (k + 1))
        for f in campos:
            for m, s in (d[f].get("facets") or {}).get("anomes_s", {}).items():
                res[uf].setdefault(m, {})[f] = s["sum"]
json.dump(res, open(f"{E}/mds/mds_uf.json", "w"))
print("fatores externos baixados em", E)
