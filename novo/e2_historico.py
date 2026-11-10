"""Pergunta extra: quanto histórico é preciso? Na simulação ano a ano, relaciona o erro com os anos de dados
disponíveis na origem (comparando a mesma série em origens diferentes, fora da pandemia), e testa o efeito
de cortar o contexto dos modelos pré-treinados não é feito aqui — só o efeito observacional."""
import numpy as np
import pandas as pd
from base import OUT, painel

p = painel()
ini = p.groupby("unique_id").data.min()
d = pd.read_parquet(f"{OUT}/anual_avaliado.parquet")
d = d[~d.ano.isin([2020, 2021])].copy()
d["anos_hist"] = ((d.origem - d.unique_id.map(ini)).dt.days / 365.25).round().astype(int)
d["faixa"] = pd.cut(d.anos_hist, [0, 4, 6, 9, 12, 16, 30], labels=["3–4", "5–6", "7–9", "10–12", "13–16", "17+"])
t = d.pivot_table(index="faixa", columns="modelo", values="ape", aggfunc="mean", observed=True)
# controle por série: erro relativo à média da própria série (remove diferença de dificuldade entre séries)
d["ape_rel"] = d.ape / d.groupby(["unique_id", "modelo"]).ape.transform("mean")
tr = d.pivot_table(index="faixa", columns="modelo", values="ape_rel", aggfunc="mean", observed=True)
n = d[d.modelo == "Combinação"].groupby("faixa", observed=True).unique_id.nunique()
t.round(2).to_csv(f"{OUT}/extra_historico_mape.csv"); tr.round(3).to_csv(f"{OUT}/extra_historico_relativo.csv")
print(t.round(1).to_string()); print(tr.round(2).to_string()); print(n)
