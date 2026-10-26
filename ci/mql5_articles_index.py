"""Index every English MQL5 article (www.mql5.com/en/articles) for the research question "did we miss a strategy":
title, date, category, the strategy families each article discusses (keyword taxonomy), whether it reports
out-of-sample / forward tests, and the tester figures quoted in its text (profit factor, Sharpe, drawdown, net
profit), each with a short context snippet. Full article text is NOT stored or published (copyright): only this
derived index. Respects robots.txt and waits between requests.
Usage: python ci/mql5_articles_index.py --out DIR [--delay 1.2] [--max-pages 400]"""
import argparse
import csv
import html
import json
import os
import re
import sys
import time
import urllib.request
import urllib.robotparser

BASE = "https://www.mql5.com"
UA = "Mozilla/5.0 (compatible; research-index/1.0; one request per second; github.com/anassboussarhan)"

FAMILIES = {
    "grid": r"\bgrid\b", "martingale": r"martingale|anti-martingale", "hedging": r"\bhedg",
    "averaging": r"averaging down|cost averaging|\baveraging\b",
    "breakout": r"breakout|break-out|opening range|donchian", "trend_following": r"trend[- ]follow|trend trading|turtle",
    "ma_crossover": r"moving average cross|ma cross|crossover", "mean_reversion": r"mean[- ]revers|reversion to the mean",
    "rsi": r"\brsi\b|relative strength index", "macd": r"\bmacd\b", "bollinger": r"bollinger",
    "stochastic": r"stochastic", "ichimoku": r"ichimoku", "atr": r"\batr\b|average true range",
    "pivot": r"pivot point", "support_resistance": r"support and resistance|support/resistance|supply and demand",
    "candlestick": r"candlestick pattern|engulfing|pin bar|doji|price action",
    "harmonic": r"harmonic pattern|gartley|butterfly pattern", "elliott": r"elliott wave",
    "fibonacci": r"fibonacci", "smc_ict": r"order block|fair value gap|smart money|liquidity sweep|\bict\b|break of structure",
    "volume_profile": r"volume profile|market profile|vwap|volume[- ]weighted",
    "order_flow": r"order flow|depth of market|\bdom\b|order book|footprint|cumulative delta",
    "seasonality": r"seasonal|time of day|day of the week|day-of-week|calendar effect|month[- ]end|turn of the month",
    "sessions": r"asian session|london session|new york session|tokyo session|session (?:high|low|range)|killzone",
    "news": r"news trading|economic calendar|non-?farm|\bnfp\b|interest rate decision|news event",
    "gotobi_fix": r"gotobi|tokyo fix|london fix|wm/reuters|fixing",
    "carry_swap": r"carry trade|positive swap|\bswap\b.*harvest|rollover",
    "arbitrage": r"arbitrage|latency arb", "pairs_cointegration": r"pairs trading|cointegrat|spread trading",
    "stat_arb": r"statistical arbitrage|stat arb",
    "correlation_basket": r"currency strength|basket trading|correlation matrix|multi-currency",
    "scalping": r"scalp", "hft_tick": r"high[- ]frequency|tick data|tick-by-tick|microstructure",
    "machine_learning": r"machine learning|neural network|deep learning|lstm|transformer|random forest|xgboost|"
                        r"gradient boost|catboost|lightgbm|svm|support vector",
    "reinforcement_learning": r"reinforcement learning|q-learning|\bdqn\b|\bppo\b|actor-critic",
    "genetic_optimisation": r"genetic algorithm|evolutionary|particle swarm|optimization algorithm|metaheuristic",
    "regime_hmm": r"hidden markov|regime|markov chain", "kalman": r"kalman", "hurst_fractal": r"hurst|fractal",
    "wavelet_fourier": r"wavelet|fourier|spectral", "entropy_information": r"entropy|mutual information",
    "garch_volatility": r"garch|volatility forecast|realized volatility|realised volatility",
    "portfolio": r"portfolio optimi|markowitz|risk parity|mean-variance",
    "position_sizing": r"kelly|position sizing|money management|risk per trade|fixed fractional",
    "walk_forward": r"walk[- ]forward", "overfitting": r"overfit|curve[- ]fit|data snooping|multiple testing",
    "options": r"\boption(?:s)? (?:pricing|strateg|trading|greeks)|black-scholes|implied volatility|straddle|strangle",
    "gold": r"\bgold\b|xauusd", "crypto": r"bitcoin|btcusd|crypto", "indices": r"s&p ?500|nasdaq|dax|us30|us500|nas100|index futures",
    "llm_ai": r"\bllm\b|chatgpt|gpt-|large language model|openai",
}
PERF = {
    "profit_factor": r"profit factor[^0-9\-]{0,25}(-?[0-9]+(?:[.,][0-9]+)?)",
    "sharpe": r"sharpe(?: ratio)?[^0-9\-]{0,25}(-?[0-9]+(?:[.,][0-9]+)?)",
    "drawdown_pct": r"drawdown[^0-9\-%]{0,40}?([0-9]+(?:[.,][0-9]+)?)\s?%",
    "net_profit": r"(?:total )?net profit[^0-9\-]{0,25}(-?[0-9][0-9 ,]*(?:[.][0-9]+)?)",
    "win_rate_pct": r"(?:win rate|winning trades|profit trades)[^0-9\-]{0,40}?([0-9]+(?:[.,][0-9]+)?)\s?%",
}
OOS = r"out[- ]of[- ]sample|forward test|forward period|walk[- ]forward|unseen data|live (?:account|trading|signal)"


def get(url, rp, delay):
    if rp is not None and not rp.can_fetch(UA, url):
        return None, "robots"
    time.sleep(delay)
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace"), None
        except Exception as ex:  # noqa: BLE001
            err = repr(ex)[:120]
            time.sleep(5 * (k + 1))
    return None, err


def text_of(page):
    page = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", page)
    m = re.search(r'(?is)<div[^>]+class="[^"]*\b(?:content|article__body|text)\b[^"]*"[^>]*>(.*)', page)
    body = m.group(1) if m else page
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    t = re.sub(r"\s+", " ", t)
    for stop in ("Attached files", "Translated from", "Other articles by this author", "Last comments"):
        i = t.find(stop)
        if i > 2000:
            t = t[:i]
    return t


def snippet(t, m, w=90):
    a, b = max(0, m.start() - w), min(len(t), m.end() + w)
    return t[a:b].strip()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--max-pages", type=int, default=400)
    ap.add_argument("--max-articles", type=int, default=100000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "crawl_log.txt"), "w")
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(BASE + "/robots.txt")
    try:
        rp.read()
        print("robots.txt read; /en/articles allowed:", rp.can_fetch(UA, BASE + "/en/articles"), file=log, flush=True)
    except Exception as ex:  # noqa: BLE001
        print("robots.txt unreadable", repr(ex), file=log, flush=True)
    if not rp.can_fetch(UA, BASE + "/en/articles"):
        print("robots.txt disallows /en/articles: stopping", file=log)
        json.dump({"stopped": "robots"}, open(os.path.join(a.out, "summary.json"), "w"))
        return
    urls, pages = [], 0
    seen = set()
    for p in range(1, a.max_pages + 1):
        lst = BASE + "/en/articles" + ("" if p == 1 else f"/page{p}")
        page, err = get(lst, rp, a.delay)
        if page is None:
            print("listing", lst, "failed", err, file=log, flush=True)
            break
        found = [u for u in re.findall(r'href="(/en/articles/\d+)"', page)]
        new = [u for u in dict.fromkeys(found) if u not in seen]
        pages += 1
        print("listing", p, len(new), file=log, flush=True)
        if not new:
            break
        for u in new:
            seen.add(u)
            urls.append(u)
    rows, perf_rows = [], []
    for i, u in enumerate(urls[: a.max_articles]):
        page, err = get(BASE + u, rp, a.delay)
        if page is None:
            print("article", u, "failed", err, file=log, flush=True)
            continue
        title = html.unescape(re.sub(r"\s+", " ", (re.search(r"(?is)<title>(.*?)</title>", page) or [None, ""])[1])).strip()
        title = re.sub(r"\s*-\s*MQL5 Articles\s*$", "", title)
        date = (re.search(r'(\d{4}\.\d{2}\.\d{2})', page) or [None, ""])[1]
        cats = sorted(set(re.findall(r'href="/en/articles/([a-z_]+)"', page)) - {"page", "new"})
        t = text_of(page)
        low = t.lower()
        fam = {k: len(re.findall(v, low)) for k, v in FAMILIES.items()}
        r = {"url": BASE + u, "title": title, "date": date, "categories": ";".join(cats[:8]), "chars": len(t),
             "oos_mentions": len(re.findall(OOS, low)), **{f"k_{k}": v for k, v in fam.items()}}
        for k, pat in PERF.items():
            ms = list(re.finditer(pat, low))
            r[f"n_{k}"] = len(ms)
            for m in ms[:4]:
                perf_rows.append({"url": BASE + u, "title": title, "metric": k, "value": m.group(1).replace(",", "."),
                                  "context": snippet(t, m)})
        rows.append(r)
        if i % 50 == 0:
            print("article", i, len(urls), title[:80], file=log, flush=True)
    if rows:
        with open(os.path.join(a.out, "articles_index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if perf_rows:
        with open(os.path.join(a.out, "performance_quotes.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(perf_rows[0].keys()))
            w.writeheader()
            w.writerows(perf_rows)
    json.dump({"listing_pages": pages, "article_urls": len(urls), "articles_indexed": len(rows),
               "performance_quotes": len(perf_rows)}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("done", pages, len(urls), len(rows), file=log, flush=True)


if __name__ == "__main__":
    sys.exit(main())
