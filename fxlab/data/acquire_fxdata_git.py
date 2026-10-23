"""Acquire real Dukascopy bid/ask tick data from the FX-Data GitHub mirror and
aggregate it to validated M1 bid/ask bars.

Source: https://github.com/FX-Data/FX-Data-<PAIR>-DS, one branch per <PAIR>-<YEAR>,
hourly CSV files ``<PAIR>/<YEAR>/<MM>/<YYYY-MM-DD>--<HH>h_ticks.csv`` with columns
``time(UTC, ms), bid, ask, bid_volume, ask_volume`` (column order verified
empirically: column 2 >= column 1 on 100% of ticks).

Design:
* one pair-year at a time, shallow single-branch clone into a scratch dir,
  processed month by month, raw clone deleted immediately (bounded disk/RAM);
* every pair-year appends a JSON line to data/manifests/acquisition.jsonl with
  provenance (branch commit SHA), raw/clean tick counts, drops by reason,
  first/last timestamp and the SHA-256 of the written parquet;
* resumable: a pair-year with an ``ok`` manifest record and an existing parquet
  whose hash matches is skipped.

No observation is ever synthesised: minutes without ticks are simply absent.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from fxlab.config import M1_DIR, MANIFEST_DIR

REPO_URL = "https://github.com/FX-Data/FX-Data-{pair}-DS"
MANIFEST = MANIFEST_DIR / "acquisition.jsonl"

PRIORITY = ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
            "XAGUSD", "EURJPY", "EURGBP", "GBPJPY", "EURCHF", "AUDJPY", "EURAUD", "CHFJPY",
            "CADJPY", "GBPCHF", "EURCAD", "AUDNZD", "EURSEK", "EURNOK", "USDSEK", "USDNOK",
            "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD", "AUDCHF", "AUDCAD", "CADCHF", "NZDJPY",
            "NZDCHF", "NZDCAD"]

# The mirror's CSVs divide Dukascopy points by 10^5 for EVERY instrument, but JPY-quoted pairs and metals
# use 3 decimal digits (points / 10^3). Their prices therefore arrive 100x too small (USDJPY ~1.03 instead of
# ~103, XAUUSD ~12.7 instead of ~1270). Detected by the price-plausibility audit; corrected here.
def price_scale(pair: str) -> float:
    return 100.0 if pair.endswith("JPY") or pair.startswith(("XAU", "XAG")) else 1.0


M1_SCHEMA_COLS = ["ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c",
                  "n_ticks", "spread_mean", "spread_max", "spread_min", "bid_liq", "ask_liq",
                  "up_ticks", "down_ticks"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_years(pair: str) -> list[int]:
    out = subprocess.run(["git", "ls-remote", "--heads", REPO_URL.format(pair=pair)],
                         capture_output=True, text=True, timeout=120)
    years = []
    for line in out.stdout.splitlines():
        ref = line.split("\t")[-1]
        tail = ref.rsplit("/", 1)[-1]
        if tail.startswith(pair + "-") and tail[len(pair) + 1:].isdigit():
            years.append(int(tail[len(pair) + 1:]))
    return sorted(years)


def load_manifest() -> dict[tuple[str, int], dict]:
    recs: dict[tuple[str, int], dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["pair"], int(r["year"]))] = r  # last record wins
    return recs


def parquet_path(pair: str, year: int) -> Path:
    return M1_DIR / pair / f"{pair}_{year}.parquet"


def _read_ticks(files: list[str]) -> pl.DataFrame:
    df = pl.read_csv(files, has_header=False,
                     new_columns=["ts", "bid", "ask", "bid_vol", "ask_vol"],
                     schema_overrides={"ts": pl.Utf8, "bid": pl.Float64, "ask": pl.Float64,
                                       "bid_vol": pl.Float64, "ask_vol": pl.Float64},
                     ignore_errors=True)  # unparsable numeric fields -> null -> dropped and counted
    # strict=False: malformed timestamps become null and are dropped (and counted) by clean_ticks
    return df.with_columns(pl.col("ts").str.strptime(pl.Datetime("ms", "UTC"),
                                                     "%Y.%m.%d %H:%M:%S%.3f", strict=False))


def clean_ticks(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    n0 = df.height
    drops = {}
    m_null = df.select(pl.any_horizontal(pl.all().is_null())).to_series()
    drops["null"] = int(m_null.sum())
    df = df.filter(~m_null)
    m_nonpos = (df["bid"] <= 0) | (df["ask"] <= 0)
    drops["non_positive"] = int(m_nonpos.sum())
    df = df.filter(~m_nonpos)
    m_crossed = df["ask"] < df["bid"]
    drops["crossed"] = int(m_crossed.sum())
    df = df.filter(~m_crossed)
    n1 = df.height
    df = df.unique(maintain_order=True)
    drops["exact_duplicate"] = n1 - df.height
    df = df.sort("ts")
    drops["total"] = n0 - df.height
    return df, drops


def ticks_to_m1(df: pl.DataFrame) -> pl.DataFrame:
    mid = (df["bid"] + df["ask"]) / 2
    df = df.with_columns(spread=(pl.col("ask") - pl.col("bid")), dmid=mid.diff())
    out = (df.group_by_dynamic("ts", every="1m", closed="left", label="left")
           .agg(bid_o=pl.col("bid").first(), bid_h=pl.col("bid").max(),
                bid_l=pl.col("bid").min(), bid_c=pl.col("bid").last(),
                ask_o=pl.col("ask").first(), ask_h=pl.col("ask").max(),
                ask_l=pl.col("ask").min(), ask_c=pl.col("ask").last(),
                n_ticks=pl.len().cast(pl.Int32),
                spread_mean=pl.col("spread").mean(), spread_max=pl.col("spread").max(),
                spread_min=pl.col("spread").min(),
                bid_liq=pl.col("bid_vol").sum().cast(pl.Float32),
                ask_liq=pl.col("ask_vol").sum().cast(pl.Float32),
                up_ticks=(pl.col("dmid") > 0).sum().cast(pl.Int32),
                down_ticks=(pl.col("dmid") < 0).sum().cast(pl.Int32)))
    return out.select(M1_SCHEMA_COLS)


def process_pair_year(pair: str, year: int, scratch: Path) -> dict:
    t0 = time.time()
    rec: dict = {"pair": pair, "year": year, "source": REPO_URL.format(pair=pair),
                 "branch": f"{pair}-{year}", "started_utc": datetime.now(timezone.utc).isoformat()}
    clone = scratch / f"{pair}-{year}"
    if clone.exists():
        shutil.rmtree(clone)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    for attempt in range(2):
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--single-branch", "--branch",
                            f"{pair}-{year}", REPO_URL.format(pair=pair), str(clone)],
                           capture_output=True, text=True, timeout=3600, env=env)
        if r.returncode == 0:
            break
        if "429" in r.stderr and attempt == 0:
            time.sleep(5)
            shutil.rmtree(clone, ignore_errors=True)
            continue
        raise RuntimeError(f"git clone failed: {r.stderr.strip()[-500:]}")
    rec["commit"] = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
    rec["clone_seconds"] = round(time.time() - t0, 1)
    files = sorted(glob.glob(str(clone / "**" / "*_ticks.csv"), recursive=True))
    rec["n_files"] = len(files)
    if not files:
        raise RuntimeError("no tick files in branch")
    by_month: dict[str, list[str]] = {}
    for f in files:
        by_month.setdefault(Path(f).parent.name, []).append(f)
    parts, drops_total, raw_total = [], {}, 0
    empty_files = 0
    for month, fs in sorted(by_month.items()):
        nonempty = [f for f in fs if os.path.getsize(f) > 0]
        empty_files += len(fs) - len(nonempty)
        if not nonempty:
            continue
        ticks = _read_ticks(nonempty)
        sc = price_scale(pair)
        if sc != 1.0:
            ticks = ticks.with_columns(pl.col("bid") * sc, pl.col("ask") * sc)
        raw_total += ticks.height
        ticks, drops = clean_ticks(ticks)
        for k, v in drops.items():
            drops_total[k] = drops_total.get(k, 0) + v
        # keep only ticks that belong to the branch year (guards mislabelled files)
        ticks = ticks.filter(pl.col("ts").dt.year() == year)
        parts.append(ticks_to_m1(ticks))
        del ticks
    m1 = pl.concat(parts).sort("ts").unique(subset="ts", keep="first", maintain_order=True)
    out = parquet_path(pair, year)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    m1.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True)
    os.replace(tmp, out)
    shutil.rmtree(clone, ignore_errors=True)
    rec.update({"status": "ok", "price_scale_applied": price_scale(pair), "n_ticks_raw": raw_total, "drops": drops_total,
                "empty_files": empty_files, "n_minutes": m1.height,
                "first_ts": str(m1["ts"].min()), "last_ts": str(m1["ts"].max()),
                "parquet": str(out.relative_to(M1_DIR.parent.parent)),
                "parquet_bytes": out.stat().st_size, "sha256": sha256_file(out),
                "seconds": round(time.time() - t0, 1)})
    return rec


def rescale_existing(pairs=None) -> list[dict]:
    """One-off migration for pair-years written before the scale fix: multiply price/spread columns by
    price_scale(pair), rewrite atomically, and append a manifest record documenting the correction."""
    done = load_manifest()
    out = []
    for (pair, year), rec in sorted(done.items()):
        if rec.get("status") != "ok" or price_scale(pair) == 1.0 or rec.get("price_scale_applied"):
            continue
        if pairs and pair not in pairs:
            continue
        p = parquet_path(pair, year)
        df = pl.read_parquet(p)
        sc = price_scale(pair)
        cols = [c for c in df.columns if c.startswith(("bid_o", "bid_h", "bid_l", "bid_c", "ask_", "spread_"))]
        df = df.with_columns([pl.col(c) * sc for c in cols])
        tmp = p.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True)
        os.replace(tmp, p)
        new = dict(rec) | {"price_scale_applied": sc, "rescaled_from_sha256": rec.get("sha256"),
                           "sha256": sha256_file(p), "parquet_bytes": p.stat().st_size,
                           "rescaled_utc": datetime.now(timezone.utc).isoformat()}
        append_manifest(new)
        out.append({"pair": pair, "year": year, "scale": sc})
    return out


def append_manifest(rec: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(pairs: list[str], scratch: Path, max_pair_years: int | None = None,
        years_filter: set[int] | None = None) -> None:
    done = load_manifest()
    count = 0
    for pair in pairs:
        try:
            years = list_years(pair)
        except Exception as e:  # network hiccup: record and move on
            append_manifest({"pair": pair, "year": -1, "status": "failed_listing", "error": repr(e)})
            continue
        for year in years:
            if years_filter and year not in years_filter:
                continue
            prev = done.get((pair, year))
            p = parquet_path(pair, year)
            if prev and prev.get("status") == "ok" and p.exists() and sha256_file(p) == prev.get("sha256"):
                continue
            if max_pair_years is not None and count >= max_pair_years:
                return
            count += 1
            print(f"[{datetime.now():%H:%M:%S}] {pair} {year} ...", flush=True)
            try:
                rec = process_pair_year(pair, year, scratch)
                print(f"   ok {rec['n_minutes']} minutes, {rec['n_ticks_raw']} ticks, {rec['seconds']}s", flush=True)
            except Exception as e:
                rec = {"pair": pair, "year": year, "status": "failed", "error": repr(e),
                       "traceback": traceback.format_exc()}
                print(f"   FAILED {e!r}", flush=True)
                shutil.rmtree(scratch / f"{pair}-{year}", ignore_errors=True)
            append_manifest(rec)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=PRIORITY)
    ap.add_argument("--scratch", default=os.environ.get("FXLAB_SCRATCH", "/tmp/fxlab_scratch"))
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    run(a.pairs, scratch, a.max, set(a.years) if a.years else None)


if __name__ == "__main__":
    sys.exit(main())
