"""Monta o painel mensal de fatores externos por unidade da federação (e Brasil), 2000-2026.

Grupos: economia, política/governo, clima, mundo e calendário. Cada variável recebe:
  - uma transformação (variação anual em log, diferença anual, anomalia climática ou nível);
  - uma defasagem de publicação L (meses): o valor do mês m só é usado numa origem t se m <= t - L.
Os dados da PMC do mês t saem cerca de 40 dias depois; L = 0 indica variáveis publicadas antes disso.
Saídas: dados/covariaveis.parquet (geo_cod, data, variáveis transformadas) e dados/covariaveis_meta.csv."""
import json, glob, os
import numpy as np
import pandas as pd
from dateutil.easter import easter

E = "dados/ext"
MESES = pd.date_range("1998-01-01", "2027-12-01", freq="MS")
cap = pd.read_csv("dados/capitais.csv")
UF = dict(zip(cap.uf, cap.cod))
GEOS = [1] + list(cap.cod)


def sidra(f, geo_n1=False):
    d = pd.DataFrame(json.load(open(f))[1:])
    per = [c for c in d.columns if c.startswith("D") and c.endswith("C") and d[c].str.len().eq(6).all()][0]
    d["per"] = d[per]
    d["V"] = pd.to_numeric(d.V, errors="coerce")
    return d


def mensal_trim(per):  # 'AAAA0T' (trimestre) -> último mês do trimestre
    return pd.to_datetime([f"{p[:4]}-{int(p[4:]) * 3:02d}-01" for p in per])


def ipea(c):
    v = json.load(open(f"{E}/ipea/{c}.json"))["value"]
    s = pd.Series({pd.Timestamp(x["VALDATA"][:7] + "-01"): x["VALVALOR"] for x in v if x["VALVALOR"] is not None})
    return s.sort_index().reindex(MESES)


nac = pd.DataFrame(index=MESES)       # variáveis nacionais (iguais para todas as séries)
reg = {g: pd.DataFrame(index=MESES) for g in GEOS}   # variáveis por UF

# ---------------------------------------------------------------- economia
ipca = sidra(f"{E}/ipca.json")
ipca = pd.Series(ipca.V.values, index=pd.to_datetime(ipca.per, format="%Y%m")).reindex(MESES)
indice = (1 + ipca.fillna(0) / 100).cumprod()                       # índice de preços (deflator)
nac["ipca_12m"] = 100 * (indice / indice.shift(12) - 1)
nac["selic"] = ipea("BM12_TJOVER12")                                 # % ao mês
nac["cambio"] = ipea("BM12_ERV12")
nac["ibcbr"] = ipea("SGS12_IBCBR12")
nac["credito_pf"] = ipea("BM12_CRLCAPF12") / indice                  # concessões reais
nac["inadimpl_pf"] = ipea("BM12_CRLINPF12")
nac["confianca"] = ipea("FCESP12_IIC12")
nac["salmin_real"] = ipea("GAC12_SALMINRE12")
nac["renda_comprometida"] = ipea("CNC12_PEICRC12")
des_br = sidra(f"{E}/desocup_br_mensal.json")
nac["desocup_br"] = pd.Series(des_br.V.values, index=pd.to_datetime(des_br.per, format="%Y%m")).reindex(MESES)
des = sidra(f"{E}/desocup_uf.json")
massa = sidra(f"{E}/massa_uf.json")
massa = massa[massa.D3C == "6293"] if "D3C" in massa else massa
for g in GEOS:
    if g == 1:
        r = pd.Series(massa[massa.NC == "1"].V.values, index=mensal_trim(massa[massa.NC == "1"].per))
        reg[g]["desocup"] = nac["desocup_br"]
    else:
        x = des[des.D1C == str(g)]
        reg[g]["desocup"] = pd.Series(x.V.values, index=mensal_trim(x.per)).reindex(MESES).ffill(limit=2)
        mm = massa[(massa.NC == "3") & (massa.D1C == str(g))]
        r = pd.Series(mm.V.values, index=mensal_trim(mm.per))
    reg[g]["massa_renda"] = r.reindex(MESES).ffill(limit=2)

# ---------------------------------------------------------------- política e governo
mds = json.load(open(f"{E}/mds/mds_uf.json"))
tr = {}
for uf, d in mds.items():
    s = pd.Series({pd.Timestamp(f"{m[:4]}-{m[4:]}-01"): sum(v.get(k, 0) for k in
                   ["valor_repassado_bolsa_familia", "pab_valor_pago_d", "pab_extraordinario_valor_pago_d"])
                   for m, v in d.items()}).sort_index()
    tr[UF[uf]] = s.reindex(MESES).replace(0, np.nan)
tr[1] = pd.concat(tr.values(), axis=1).sum(axis=1, min_count=20)
for g in GEOS:
    t = tr[g] / indice                                                 # Bolsa Família / Auxílio Brasil, real
    # meses em que o Auxílio Emergencial substituiu o benefício regular: nível não comparável
    t[pd.Timestamp("2020-04-01"):pd.Timestamp("2020-12-01")] = np.nan
    t[pd.Timestamp("2021-04-01"):pd.Timestamp("2021-10-01")] = np.nan
    reg[g]["transf_renda"] = t
# Auxílio Emergencial: valor da parcela básica por família (R$) — 600 (abr-ago/2020), 300 (set-dez/2020),
# cerca de 250 em média (abr-out/2021; parcelas de 150 a 375)
ae = pd.Series(0.0, index=MESES)
ae["2020-04-01":"2020-08-01"] = 600; ae["2020-09-01":"2020-12-01"] = 300; ae["2021-04-01":"2021-10-01"] = 250
nac["aux_emergencial"] = ae
nac["eleicao_geral"] = [int(m.year % 4 == 2 and m.year >= 2002) for m in MESES]   # 2002, 2006, ..., 2026
nac["eleicao_municipal"] = [int(m.year % 4 == 0 and m.year >= 2000) for m in MESES]
nac["mes_eleicao"] = [int(m.month in (10,) and m.year % 2 == 0) for m in MESES]
nac["icms_comb_teto"] = [int(m >= pd.Timestamp("2022-07-01")) for m in MESES]       # LC 194/2022
epu = pd.read_excel(f"{E}/epu_brasil.xlsx").dropna()
nac["incerteza_br"] = pd.Series(epu.iloc[:, 2].values, index=pd.to_datetime(
    dict(year=epu.year.astype(int), month=epu.month.astype(int), day=1))).reindex(MESES)

# ---------------------------------------------------------------- mundo
gepu = pd.read_excel(f"{E}/epu_global.xlsx")
gepu = gepu[pd.to_numeric(gepu.Year, errors="coerce").notna() & gepu.Month.notna()]
nac["incerteza_global"] = pd.Series(gepu.GEPU_current.values, index=pd.to_datetime(
    dict(year=gepu.Year.astype(int), month=gepu.Month.astype(int), day=1))).reindex(MESES)
nac["petroleo"] = ipea("IFS12_PETROLEUM12")
nac["soja"] = ipea("IFS12_SOJAGP12")
nac["milho"] = ipea("IFS12_MAIZE12")
nac["carne"] = ipea("IFS12_BEEFB12")

# ---------------------------------------------------------------- clima (NASA POWER, capitais)
clima = {}
for f in glob.glob(f"{E}/clima/*.csv"):
    c = pd.read_csv(f)
    c["data"] = pd.to_datetime(c.YEAR.astype(str), format="%Y") + pd.to_timedelta(c.DOY - 1, unit="D")
    c = c.replace(-999, np.nan).set_index("data")
    m = pd.DataFrame({"temp": c.T2M.resample("MS").mean(), "temp_max": c.T2M_MAX.resample("MS").mean(),
                      "chuva": c.PRECTOTCORR.resample("MS").sum()})
    clima[UF[os.path.basename(f)[:2]]] = m.reindex(MESES)
clima[1] = sum(clima[g] for g in clima) / len(clima)
for g in GEOS:
    m = clima[g]
    norm = m["2001-01-01":"2020-12-01"].groupby(m["2001-01-01":"2020-12-01"].index.month).mean()
    reg[g]["temp_anom"] = m.temp - norm.temp.reindex(m.index.month).values
    reg[g]["temp_max_anom"] = m.temp_max - norm.temp_max.reindex(m.index.month).values
    reg[g]["chuva_anom"] = np.log((m.chuva + 10) / (norm.chuva.reindex(m.index.month).values + 10))

# ---------------------------------------------------------------- calendário (conhecido com antecedência)
def dias_uteis(m):
    d = pd.date_range(m, m + pd.offsets.MonthEnd(0))
    return int((d.dayofweek < 5).sum())


nac["dias_uteis"] = [dias_uteis(m) for m in MESES]
nac["pascoa_mes"] = [int(easter(m.year).month == m.month) for m in MESES]
carn = {y: pd.Timestamp(easter(y)) - pd.Timedelta(47, unit='D') for y in range(1998, 2028)}
nac["carnaval_mes"] = [int(carn[m.year].month == m.month) for m in MESES]

# ---------------------------------------------------------------- transformações e metadados
META = {  # variável: (grupo, transformação, defasagem de publicação, conhecida no futuro?)
    "ipca_12m": ("Economy", "nivel", 0, 0), "selic": ("Economy", "nivel", 0, 0), "cambio": ("Economy", "logyoy", 0, 0),
    "ibcbr": ("Economy", "logyoy", 2, 0), "credito_pf": ("Economy", "logyoy", 1, 0),
    "inadimpl_pf": ("Economy", "nivel", 1, 0), "confianca": ("Economy", "logyoy", 0, 0),
    "salmin_real": ("Economy", "logyoy", 0, 0), "renda_comprometida": ("Economy", "nivel", 0, 0),
    "desocup": ("Economy", "difyoy", 2, 0), "massa_renda": ("Economy", "logyoy", 2, 0),
    "transf_renda": ("Policy", "logyoy", 0, 0), "aux_emergencial": ("Policy", "nivel", 0, 0),
    "eleicao_geral": ("Policy", "nivel", 0, 1), "eleicao_municipal": ("Policy", "nivel", 0, 1),
    "mes_eleicao": ("Policy", "nivel", 0, 1), "icms_comb_teto": ("Policy", "nivel", 0, 0),
    "incerteza_br": ("Policy", "logyoy", 0, 0),
    "incerteza_global": ("World", "logyoy", 0, 0), "petroleo": ("World", "logyoy", 0, 0),
    "soja": ("World", "logyoy", 0, 0), "milho": ("World", "logyoy", 0, 0), "carne": ("World", "logyoy", 0, 0),
    "temp_anom": ("Climate", "nivel", 0, 0), "temp_max_anom": ("Climate", "nivel", 0, 0),
    "chuva_anom": ("Climate", "nivel", 0, 0),
    "dias_uteis": ("Calendar", "difyoy", 0, 1), "pascoa_mes": ("Calendar", "difyoy", 0, 1),
    "carnaval_mes": ("Calendar", "difyoy", 0, 1),
}


def transf(s, t):
    if t == "logyoy":
        return np.log(s) - np.log(s.shift(12))
    if t == "difyoy":
        return s - s.shift(12)
    return s


out = []
for g in GEOS:
    df = pd.concat([nac.drop(columns=["desocup_br"]), reg[g]], axis=1)
    df = pd.DataFrame({v: transf(df[v].astype(float), META[v][1]) for v in META}, index=MESES)
    df.insert(0, "geo_cod", g)
    out.append(df.rename_axis("data").reset_index())
cov = pd.concat(out, ignore_index=True)
cov.to_parquet("dados/covariaveis.parquet", index=False)
pd.DataFrame([dict(variavel=k, grupo=v[0], transformacao=v[1], defasagem=v[2], futuro_conhecido=v[3])
              for k, v in META.items()]).to_csv("dados/covariaveis_meta.csv", index=False)
x = cov[cov.data.between("2000-01-01", "2026-08-01")]
print(cov.shape)
print(x.groupby("geo_cod").size().head(3))
print((x[x.geo_cod.isin([1, 23, 35])].groupby("geo_cod").apply(lambda d: d.drop(columns=["geo_cod", "data"]).notna().mean()).T.round(2)))
print(x[x.geo_cod == 23].set_index("data").loc["2020-01-01":"2023-12-01":6, ["transf_renda", "temp_anom", "desocup", "massa_renda", "ibcbr"]].round(3))
