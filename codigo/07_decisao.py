"""Avaliação orientada à decisão: cada previsão quantílica vira uma decisão de produção/estoque.

Para cada série, origem e horizonte, a quantidade Q comprometida é o quantil crítico p = b/(b+h) da distribuição
prevista (newsvendor), com h = custo unitário de sobra e b = custo unitário de falta. O custo realizado é
C = h·max(Q−y, 0) + b·max(y−Q, 0). Razões b:h avaliadas: 1:1, 2:1, 3:1, 4:1 e 9:1 (p = 0,50; 0,67; 0,75; 0,80; 0,90).
Saídas: custo relativo ao sazonal ingênuo (média geométrica entre séries, IC 95% por bootstrap de séries),
custo como % da demanda, nível de serviço de ciclo atingido P(y ≤ Q) e taxa de atendimento (fill rate),
por antecedência (horizonte) e para dezembro."""
import glob, json
import numpy as np
import pandas as pd
from comum import carrega_painel, QUANTIS, QCOLS

BASE = "Seasonal naive"
RAZOES = {"1:1": 1, "2:1": 2, "3:1": 3, "4:1": 4, "9:1": 9}
CHAVE = ["unique_id", "origem", "h"]
p = carrega_painel(imputa=False)
real = p.set_index(["unique_id", "data"]).valor

fs = [f for f in sorted(glob.glob("previsoes/*.parquet"))
      if not any(t in f for t in ("_valid", "_ctx", "_imputado", "teste_", "_X_", "Chronos-2"))]
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d["origem"] = pd.to_datetime(d.origem); d["data_alvo"] = pd.to_datetime(d.data_alvo)
d = d[CHAVE + ["data_alvo", "modelo"] + QCOLS]

# combinações (mesmas definições de 06_combinacoes_robustez.py; membros escolhidos na validação)
memb = json.load(open("resultados/combinacoes.json"))["membros"]
comb = []
for nome, m in memb.items():
    c = d[d.modelo.isin(m)].groupby(CHAVE + ["data_alvo"])[QCOLS].median().reset_index()
    c["modelo"] = nome
    comb.append(c)
d = pd.concat([d] + comb, ignore_index=True)
d[QCOLS] = np.sort(d[QCOLS].values, axis=1)
n = d.modelo.nunique()
cont = d.groupby(CHAVE).modelo.nunique()
d = d.merge(cont[cont == n].reset_index()[CHAVE], on=CHAVE)
d["y"] = real.reindex(list(zip(d.unique_id, d.data_alvo))).values
assert d.y.notna().all()

Qm = d[QCOLS].values
alf = np.array(QUANTIS)
linhas = []
for rot, b in RAZOES.items():
    pc = b / (b + 1)
    k = min(int(np.searchsorted(alf, pc, side="right")) - 1, len(alf) - 2)  # interpolação linear entre decis
    Qd = Qm[:, k] + (pc - alf[k]) / (alf[k + 1] - alf[k]) * (Qm[:, k + 1] - Qm[:, k])
    sobra = np.maximum(Qd - d.y.values, 0)
    falta = np.maximum(d.y.values - Qd, 0)
    x = d[CHAVE + ["data_alvo", "modelo", "y"]].copy()
    x["custo"] = sobra + b * falta
    x["atendido"] = d.y.values <= Qd
    x["falta"] = falta
    x["razao"] = rot
    x["p_alvo"] = pc
    linhas.append(x)
dec = pd.concat(linhas, ignore_index=True)
dec.to_parquet("resultados/decisoes.parquet", index=False)


def resumo(x, boot=True):
    out = []
    for (rot, pc), g in x.groupby(["razao", "p_alvo"]):
        ps = g.groupby(["unique_id", "modelo"]).custo.mean().unstack()
        lr = np.log(ps.div(ps[BASE], axis=0))
        r = pd.DataFrame({"custo_rel": np.exp(lr.mean())})
        if boot:
            rng = np.random.default_rng(2026)
            bs = np.array([lr.iloc[rng.integers(0, len(lr), len(lr))].mean().values for _ in range(1000)])
            r["ic95_inf"] = np.exp(np.percentile(bs, 2.5, axis=0)); r["ic95_sup"] = np.exp(np.percentile(bs, 97.5, axis=0))
        gm = g.groupby("modelo")
        r["custo_pct_demanda"] = 100 * gm.custo.sum() / gm.y.sum()
        r["servico_ciclo"] = 100 * gm.atendido.mean()
        r["fill_rate"] = 100 * (1 - gm.falta.sum() / gm.y.sum())
        r["razao"], r["p_alvo"] = rot, pc
        out.append(r.reset_index())
    return pd.concat(out, ignore_index=True)


geral = resumo(dec)
geral.to_csv("resultados/decisao_geral.csv", index=False)
pd.set_option("display.width", 220)
print(geral.pivot(index="modelo", columns="razao", values="custo_rel").sort_values("4:1").round(3).to_string())
print("\nnível de serviço de ciclo atingido (%), alvo = 50, 67, 75, 80, 90:")
print(geral.pivot(index="modelo", columns="razao", values="servico_ciclo").round(1).to_string())

recs = {"h1-3": dec.h <= 3, "h4-6": dec.h.between(4, 6), "h7-12": dec.h >= 7, "dezembro": dec.data_alvo.dt.month == 12}
rr = []
for nome, m in recs.items():
    r = resumo(dec[m], boot=False); r["recorte"] = nome; rr.append(r)
rr = pd.concat(rr, ignore_index=True)
rr.to_csv("resultados/decisao_recortes.csv", index=False)
print("\ncusto relativo com razão 4:1, por recorte:")
print(rr[rr.razao == "4:1"].pivot(index="modelo", columns="recorte", values="custo_rel").round(3).to_string())
print("\nnível de serviço em dezembro (%), razão 4:1 (alvo 80):")
print(rr[(rr.razao == "4:1") & (rr.recorte == "dezembro")].set_index("modelo").servico_ciclo.round(1).to_string())
