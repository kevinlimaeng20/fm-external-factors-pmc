"""Baixa o Google Trends (buscas mensais desde 2004) para os termos ligados às atividades do varejo.
Rode no SEU computador (o ambiente do estudo não tem acesso ao Google):

    pip install pytrends pandas
    python k0_baixa_trends.py            # só Brasil  (~20 consultas, uns 5 minutos)
    python k0_baixa_trends.py estados    # Brasil + 27 UFs (~560 consultas, pode levar horas; pode ser interrompido e retomado)

Gera o arquivo trends.csv na mesma pasta (colunas: termo, geo, data, valor). Envie esse arquivo de volta.
Uma consulta por termo, para que cada termo fique na sua própria escala 0–100."""
import os, sys, time, random
import pandas as pd
from pytrends.request import TrendReq

TERMOS = ["geladeira", "fogão", "máquina de lavar", "televisão", "sofá", "guarda roupa", "celular", "notebook", "impressora",
          "tênis", "vestido", "calça jeans", "farmácia", "perfume", "supermercado", "cesta básica", "posto de gasolina",
          "material escolar", "livro", "promoção"]
UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
GEOS = ["BR"] + ([f"BR-{u}" for u in UFS] if len(sys.argv) > 1 and sys.argv[1] == "estados" else [])
ARQ = "trends.csv"
feito = set()
if os.path.exists(ARQ):
    a = pd.read_csv(ARQ); feito = set(zip(a.termo, a.geo))
py = TrendReq(hl="pt-BR", tz=180, timeout=(10, 30))
for geo in GEOS:
    for t in TERMOS:
        if (t, geo) in feito:
            continue
        for tentativa in range(6):
            try:
                py.build_payload([t], timeframe="all", geo=geo)
                d = py.interest_over_time()
                break
            except Exception as e:                       # o Google limita consultas seguidas (erro 429): espera e tenta de novo
                espera = 60 * (tentativa + 1)
                print(f"  {t} / {geo}: {type(e).__name__}; aguardando {espera}s"); time.sleep(espera)
        else:
            print(f"  desisti de {t} / {geo}"); continue
        if len(d):
            out = pd.DataFrame({"termo": t, "geo": geo, "data": d.index.strftime("%Y-%m-%d"), "valor": d[t].values})
            out.to_csv(ARQ, mode="a", header=not os.path.exists(ARQ), index=False, encoding="utf-8")
        print(t, geo, len(d)); time.sleep(8 + random.random() * 7)
print("pronto:", ARQ)
