"""Ideia 7: calendário de eventos (2000–2026) montado com IA de linguagem a partir de conhecimento geral e
conferência pontual na web. É um rascunho para revisão humana: datas em nível de mês, valores aproximados.
antecipado = 1 se o evento era anunciado/conhecido antes de acontecer (pode entrar como variável de previsão)."""
import pandas as pd
from base import OUT

m = lambda a, b: [d.strftime("%Y-%m") for d in pd.date_range(a, b, freq="MS")]
E = [  # tipo, nome, meses, antecipado, intensidade (R$ bi/mês quando faz sentido; senão 1), geo ("" = Brasil), atividades mais afetadas
    ("copa", "Copa do Mundo 2002", m("2002-06", "2002-06"), 1, 1, "", "eletro"), ("copa", "Copa do Mundo 2006", m("2006-06", "2006-07"), 1, 1, "", "eletro"),
    ("copa", "Copa do Mundo 2010", m("2010-06", "2010-07"), 1, 1, "", "eletro"), ("copa", "Copa do Mundo 2014 (no Brasil)", m("2014-06", "2014-07"), 1, 1, "", "eletro"),
    ("copa", "Copa do Mundo 2018", m("2018-06", "2018-07"), 1, 1, "", "eletro"), ("copa", "Copa do Mundo 2022", m("2022-11", "2022-12"), 1, 1, "", "eletro"),
    ("copa", "Copa do Mundo 2026", m("2026-06", "2026-07"), 1, 1, "", "eletro"),
    ("saque_fgts", "FGTS contas inativas (R$ 44 bi)", m("2017-03", "2017-07"), 1, 8.8, "", "geral"),
    ("saque_fgts", "FGTS saque imediato (≈R$ 28 bi)", m("2019-09", "2020-03"), 1, 4.0, "", "geral"),
    ("saque_fgts", "FGTS saque emergencial (≈R$ 38 bi)", m("2020-06", "2020-11"), 1, 6.3, "", "geral"),
    ("saque_fgts", "FGTS saque extraordinário (≈R$ 24 bi sacados)", m("2022-04", "2022-06"), 1, 7.9, "", "geral"),
    ("saque_fgts", "FGTS saldo retido do saque-aniversário (≈R$ 12 bi)", m("2025-03", "2025-06"), 1, 3.0, "", "geral"),
    ("estimulo_duraveis", "IPI reduzido autos e linha branca", m("2008-12", "2010-03"), 1, 1, "", "eletro"),
    ("estimulo_duraveis", "IPI reduzido linha branca, móveis e autos", m("2011-12", "2013-12"), 1, 1, "", "eletro"),
    ("estimulo_duraveis", "Minha Casa Melhor (crédito para móveis e eletro)", m("2013-06", "2014-12"), 1, 1, "", "eletro"),
    ("decimo_inss_antecipado", "13º do INSS antecipado", m("2020-04", "2020-05") + m("2021-05", "2021-06") + m("2022-04", "2022-05") + m("2023-05", "2023-06")
     + m("2024-04", "2024-05") + m("2025-04", "2025-05") + m("2026-04", "2026-05"), 1, 1, "", "geral"),
    ("renegociacao_dividas", "Desenrola Brasil", m("2023-07", "2024-05"), 1, 1, "", "geral"),
    ("imposto_importados", "Imposto sobre compras internacionais até US$ 50", m("2024-08", "2025-07"), 1, 1, "", "vestuario"),
    ("alivio_ir", "Isenção de IR até R$ 5 mil", m("2026-01", "2026-12"), 1, 1, "", "geral"),
    ("racionamento", "Racionamento de energia", m("2001-06", "2002-02"), 1, 1, "", "eletro"),
    ("olimpiada", "Olimpíada do Rio", m("2016-08", "2016-08"), 1, 1, "Rio de Janeiro", "geral"),
    ("greve", "Greve dos caminhoneiros", m("2018-05", "2018-06"), 0, 1, "", "combustiveis"),
    ("crise_financeira", "Crise financeira global", m("2008-10", "2009-03"), 0, 1, "", "eletro"),
    ("protestos", "Manifestações de junho", m("2013-06", "2013-06"), 0, 1, "", "geral"),
    ("pandemia", "Covid-19: fechamentos", m("2020-03", "2020-07") + m("2021-03", "2021-04"), 0, 1, "", "geral"),
    ("desastre", "Enchentes no Rio Grande do Sul", m("2024-05", "2024-06"), 0, 1, "Rio Grande do Sul", "geral"),
    ("desastre", "Apagão no Amapá", m("2020-11", "2020-11"), 0, 1, "Amapá", "geral"),
    ("combustiveis_imposto", "Teto do ICMS e desoneração de combustíveis", m("2022-07", "2022-12"), 0, 1, "", "combustiveis"),
]
lin = [dict(tipo=t, evento=n, mes=x, antecipado=a, intensidade=i, geo=g, atividades=at) for t, n, ms, a, i, g, at in E for x in ms]
C = pd.DataFrame(lin); C.to_csv(f"{OUT}/j0_calendario_eventos.csv", index=False)
print(C.groupby(["tipo", "antecipado"]).agg(eventos=("evento", "nunique"), meses=("mes", "count")).to_string())
