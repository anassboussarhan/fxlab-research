"""2026 check diagnostic: for USD/TRY D1 and W1 counting R1 rules, every configuration's 2026 return with and without
financing, and the mean |side| of the rule's output (statistics only)."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import dev_end; import r23_LW as LW; import intraday17 as I17
dev_end.apply("2026-10-31", 2026)
from scipy.stats import qmc
import zlib
import r23_expose as XP
from fxlab.data.store import load_m1, resample
m1 = load_m1("USDTRY"); m1 = m1[m1.index <= pd.Timestamp("2026-10-31 23:59", tz="UTC")]
S = {e.id: e for e in I17.strategies()}
r24 = pd.read_csv("res/r24_streams.csv.gz", low_memory=False)
c = r24[(r24.counts == True) & (r24.symbol == "USDTRY")]
units = ["S.BOOK.BW_FRACTAL|D1", "S.BO.RETEST|W1"]
out = {}
d1 = resample(m1, "D1")
for u in units:
    key, tf = u.split("|")
    b = d1 if tf == "D1" else LW.higher(d1, "W1")
    e = S[key]
    fn, consts = XP.expose(e)
    probe = b
    ok = bool(consts) and XP.verified(e, fn, probe.iloc[-3000:])
    if not ok:
        fn, consts = e.fn, []
    sps = LW.strategy_specs(e, consts)
    seed = zlib.crc32(f"LW|{key}|{tf}".encode()) & 0x7FFFFFFF
    U = qmc.Sobol(len(sps) + LW.N_EXEC, scramble=True, seed=seed).random(128)
    rows = []
    for i, uu in enumerate(U[:64]):
        params = {s["name"]: LW.draw(s, uu[j]) for j, s in enumerate(sps)}
        try:
            LW.G["financing"] = "rates_oecd"; rf = LW.evaluate_one(e, fn, b, "USDTRY", params, uu[len(sps):])
            LW.G["financing"] = None; r0 = LW.evaluate_one(e, fn, b, "USDTRY", params, uu[len(sps):])
            side = fn(b.copy(), **params)["side"]
        except Exception as ex:  # noqa: BLE001
            continue
        m26 = rf.index >= pd.Timestamp("2026-01-01", tz="UTC")
        if float(rf[m26].sum()) > 1.0 and "trades" not in out:
            from fxlab.evaluate import evaluate
            d_, mode_, ent_, risk_ = LW.execution(uu[len(sps):])
            out["trades"] = {"unit": u, "i": i, "mode": mode_, "entry": str(ent_), "risk": str(risk_)[:300], "params": str(params)[:300]}
            rr = rf[m26]
            out["trades"]["top_bars"] = [(str(k), round(float(v), 4)) for k, v in rr.sort_values().tail(8).items()]
            out["trades"]["n_nonzero_2026"] = int((rr != 0).sum())
            sig_ = fn(b.copy(), **params); sig_ = sig_.copy(); sig_["side"] = sig_["side"] * d_
            res = evaluate(e.id, b, "USDTRY", signals=sig_[["side"]], mode_override="trade", risk_override=risk_, exec_cfg=ent_, financing="rates_oecd")
            T = res["trades"]
            T = T[pd.to_datetime(T.iloc[:, 0] if "entry_time" not in T else T["entry_time"], utc=True) >= pd.Timestamp("2026-01-01", tz="UTC")] if len(T) else T
            out["trades"]["trade_columns"] = list(T.columns)
            out["trades"]["n_trades_2026"] = int(len(T))
            out["trades"]["trades_2026_head"] = T.head(12).astype(str).values.tolist()
            sd = __import__("fxlab.execution.risk", fromlist=["x"]).stop_realized_vol(b, **risk_["stop_params"])
            out["trades"]["stop_dist_2026_quantiles"] = [float(x) for x in sd[sd.index >= pd.Timestamp("2026-01-01", tz="UTC")].quantile([0, 0.1, 0.5, 0.9]).values]
            bb = b[b.index >= pd.Timestamp("2026-01-01", tz="UTC")]
            out["trades"]["bars_2026"] = [(str(k), round(float(x), 4), round(float(y), 4), round(float(z), 4)) for k, x, y, z in zip(bb.index[:3], bb.open[:3], bb.high[:3], bb.low[:3])]
            big = rr.sort_values().index[-1]
            w = b.loc[big - pd.Timedelta(days=10): big + pd.Timedelta(days=3)]
            out["trades"]["around_top"] = [(str(k), round(float(o), 4), round(float(h), 4), round(float(l), 4), round(float(c_), 4)) for k, o, h, l, c_ in zip(w.index, w.open, w.high, w.low, w.close)]
        rows.append({"ret26_fin": float(rf[m26].sum()), "ret26_nofin": float(r0[m26].sum()), "ret2025_fin": float(rf[(rf.index.year == 2025)].sum()),
                     "mean_abs_side": float(side.abs().mean()), "max_abs_side": float(side.abs().max()), "mode": LW.execution(uu[len(sps):])[1]})
    R = pd.DataFrame(rows)
    out[u] = {"configs": len(R), "ret26_fin_median": float(R.ret26_fin.median()), "ret26_nofin_median": float(R.ret26_nofin.median()),
              "share_ret26_pos": float((R.ret26_fin > 0).mean()), "max_abs_side_max": float(R.max_abs_side.max()),
              "mean_abs_side_median": float(R.mean_abs_side.median()), "ret26_fin_max": float(R.ret26_fin.max()),
              "fin_effect_median": float((R.ret26_fin - R.ret26_nofin).median())}
print(json.dumps(out, indent=1))
