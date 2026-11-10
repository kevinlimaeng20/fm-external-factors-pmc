"""Teste 3 (avaliação): variantes do Chronos-2 contra o Chronos-2 de prateleira, e efeito na combinação."""
import glob
import numpy as np
import pandas as pd
from base import OUT, boot_ic

d = pd.read_parquet(f"{OUT}/previsoes_teste.parquet")
K = ["unique_id", "origem", "h"]
BASE = [m for m in d.modelo.unique() if not m.startswith("Combinação")]
W = d[d.modelo.isin(BASE)].pivot_table(index=K + ["y", "escala"], columns="modelo", values="q50").reset_index()
novos = []
for f in sorted(glob.glob(f"{OUT}/c2_*.parquet")):
    x = pd.read_parquet(f); n = x.modelo.iloc[0]; novos.append(n)
    x["origem"] = pd.to_datetime(x.origem)
    W = W.merge(x[K + ["q50"]].rename(columns={"q50": n}), on=K, how="left")
W["Mediana (12 modelos)"] = W[BASE].median(axis=1)
for n in novos:
    W[f"Mediana trocando Chronos-2 por {n}"] = W[[m for m in BASE if m != "Chronos-2"] + [n]].median(axis=1)
    W[f"Mediana (12) + {n}"] = W[BASE + [n]].median(axis=1)
mase = lambda c: ((W[c] - W.y).abs() / W.escala).groupby(W.unique_id).mean()
ref = mase("Ingênuo sazonal")
lin = []
for c, base in [(n, "Chronos-2") for n in ["Chronos-2", "LightGBM + fatores sel."] + novos] + [(c, "Mediana (12 modelos)") for c in W.columns if c.startswith("Mediana")]:
    r = np.log(mase(c) / mase(base)); ic = boot_ic(r.values)
    lin.append(dict(modelo=c, MAPE=(100 * (W[c] - W.y).abs() / W.y).mean(), MASE_rel_ingenuo=np.exp(np.log(mase(c) / ref).mean()), comparado_com=base,
                    razao=np.exp(r.mean()), IC_inf=np.exp(ic[0]), IC_sup=np.exp(ic[1]), pct_series_melhor=100 * (mase(c) < mase(base)).mean()))
T = pd.DataFrame(lin); T.round(4).to_csv(f"{OUT}/t3_chronos2.csv", index=False); print(T.round(3).to_string())
h = pd.DataFrame({c: (100 * (W[c] - W.y).abs() / W.y).groupby(W.h).mean() for c in ["Chronos-2"] + novos + ["LightGBM + fatores sel."]})
h.round(2).to_csv(f"{OUT}/t3_por_horizonte.csv"); print(h.round(2).to_string())
