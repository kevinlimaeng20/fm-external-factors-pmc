"""Chronos-2 (amazon/chronos-2), zero-shot, com e sem fatores externos.

Uso: python 12_chronos2.py [uni|real|oraculo|sel]
  uni      — só o histórico de vendas;
  real     — covariáveis passadas (cada fator deslocado pela sua defasagem de publicação, de modo que a linha do
             mês m só contém o que estava publicado em m) + fatores conhecidos no futuro (calendário, eleições);
  oraculo  — também recebe o valor realizado de todos os fatores nos meses previstos.
Saída: previsoes/Chronos-2_<modo>{SUFIXO}.parquet"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from comum import carrega_painel, ORIGENS, H, QUANTIS, QCOLS, SUFIXO

modo = sys.argv[1] if len(sys.argv) > 1 else "uni"
torch.set_num_threads(2)
p = carrega_painel()
cov = pd.read_parquet("dados/covariaveis.parquet")
info = pd.read_csv("dados/covariaveis_meta.csv").set_index("variavel")
# conjunto compacto (o custo do Chronos-2 cresce com o número de variáveis): principais fatores de cada grupo
info = info.loc[["selic", "cambio", "ibcbr", "credito_pf", "confianca", "desocup", "massa_renda", "transf_renda",
                 "aux_emergencial", "incerteza_br", "petroleo", "temp_anom", "chuva_anom",
                 "dias_uteis", "pascoa_mes", "carnaval_mes", "eleicao_geral"]]
if modo == "sel":   # só os grupos escolhidos na validação do LightGBM (economia, clima, calendário)
    import json
    sel = json.load(open("resultados/fatores_selecao_validacao.json"))["grupos"]
    info = info[info.grupo.isin(sel)]
    print("covariáveis:", list(info.index))
geo = p.drop_duplicates("unique_id").set_index("unique_id").geo_cod.astype(int)

# covariáveis alinhadas à disponibilidade: coluna v na data m = valor de v em m - defasagem
al = []
for g, d in cov.groupby("geo_cod"):
    d = d.set_index("data").sort_index()
    x = pd.DataFrame(index=d.index)
    for v, r in info.iterrows():
        x[v] = d[v].shift(int(r.defasagem))
        if modo == "oraculo" and not r.futuro_conhecido:
            x[v + "_real"] = d[v]          # valor realizado (só usado no futuro, como teto)
    x["geo_cod"] = g
    al.append(x.reset_index())
al = pd.concat(al).fillna(0.0)
PASSADAS = [v for v, r in info.iterrows() if not r.futuro_conhecido]
FUTURAS = [v for v, r in info.iterrows() if r.futuro_conhecido]
if modo == "oraculo":
    FUTURAS = FUTURAS + [v + "_real" for v in PASSADAS]

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
base = p[["unique_id", "data", "valor"]].rename(columns={"unique_id": "id", "data": "timestamp", "valor": "target"})
base["geo_cod"] = base.id.map(geo)
if modo != "uni":
    base = base.merge(al, left_on=["geo_cod", "timestamp"], right_on=["geo_cod", "data"], how="left").drop(columns="data")
    base[PASSADAS + FUTURAS] = base[PASSADAS + FUTURAS].fillna(0.0)
saida = []
t0 = time.time()
TMP = f"previsoes/tmp_c2_{modo}{SUFIXO}"; os.makedirs(TMP, exist_ok=True)   # checkpoint por origem
for o in ORIGENS[:int(os.environ.get("FM_MAXO", len(ORIGENS)))]:
    ck = f"{TMP}/{o.date()}.parquet"
    if os.path.exists(ck):
        saida.append(pd.read_parquet(ck)); continue
    ctx = base[base.timestamp <= o]
    fut = None
    cols = ["id", "timestamp", "target"]
    if modo != "uni":
        cols += PASSADAS + FUTURAS
        datas = pd.date_range(o + pd.DateOffset(months=1), periods=H, freq="MS")
        fut = pd.DataFrame([(u, d) for u in geo.index for d in datas], columns=["id", "timestamp"])
        fut["geo_cod"] = fut.id.map(geo)
        fut = fut.merge(al[["geo_cod", "data"] + FUTURAS], left_on=["geo_cod", "timestamp"], right_on=["geo_cod", "data"],
                        how="left")[["id", "timestamp"] + FUTURAS].fillna(0.0)
    if modo == "oraculo":   # no passado, a coluna "_real" também é conhecida (é história)
        ctx = ctx.copy()
    o_df = pipe.predict_df(ctx[cols], future_df=fut, id_column="id", timestamp_column="timestamp", target="target",
                           prediction_length=H, quantile_levels=QUANTIS, batch_size=64)
    o_df = o_df.rename(columns={str(q): c for q, c in zip(QUANTIS, QCOLS)})
    o_df = o_df.rename(columns={q: c for q, c in zip(QUANTIS, QCOLS)})
    o_df["origem"] = o
    o_df["h"] = ((o_df.timestamp.dt.year - o.year) * 12 + o_df.timestamp.dt.month - o.month).astype(int)
    saida.append(o_df.rename(columns={"id": "unique_id", "timestamp": "data_alvo"})[
        ["unique_id", "origem", "h", "data_alvo"] + QCOLS])
    saida[-1].to_parquet(ck, index=False)
    if len(saida) in (1, 5) or len(saida) % 10 == 0:
        print(f"{modo}: {len(saida)}/{len(ORIGENS)} origens, {time.time() - t0:.0f}s", flush=True)
out = pd.concat(saida, ignore_index=True)
out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
nome = {"uni": "Chronos-2", "real": "Chronos-2 + external factors",
        "oraculo": "Chronos-2 + external factors (oracle)", "sel": "Chronos-2 + selected factors"}[modo]
out["modelo"] = nome
out.to_parquet(f"previsoes/Chronos-2_{modo}{SUFIXO}.parquet", index=False)
print(nome, len(out), f"linhas em {time.time() - t0:.0f}s")
