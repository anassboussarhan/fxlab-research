"""Build the sortable feature-predictability page (HTML) from feature_ic_report.py outputs.
Usage: python scripts/feature_ic_page.py SUMMARY_DIR OUT_HTML"""
import json
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
F = pd.read_csv(f"{src}/features.csv")
T = pd.read_csv(f"{src}/by_tf.csv")
C = pd.read_csv(f"{src}/by_category.csv")
S = json.load(open(f"{src}/summary.json"))
tfs = [c[4:] for c in F.columns if c.startswith("sig_")]


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


rows = [{k: clean(v) for k, v in r.items()} for r in F.to_dict("records")]
tfrows = [{k: clean(v) for k, v in r.items()} for r in T.to_dict("records")]
cats = [{k: clean(v) for k, v in r.items()} for r in C.to_dict("records")]
data = json.dumps({"rows": rows, "tfs": tfs, "by_tf": tfrows, "cats": cats, "summary": S})

html = """<title>FX Feature Predictability</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* layout: one reading column; summary strip, timeframe ladder, then the full sortable feature ledger */
:root{
  --bg:#f6f7f5; --panel:#ffffff; --ink:#17201c; --muted:#5d6b64; --line:#d9dfdb;
  --accent:#1f6f5c; --accent-soft:#dcefe8; --warn:#9a6a12; --warn-soft:#f6ecd6; --none:#8a948f; --none-soft:#eceeed;
  --heat0:#eef1ef; --heat1:#cfe7de; --heat2:#9fd0bf; --heat3:#5fae96; --heat4:#1f6f5c;
  --display:"Fraunces", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#111614; --panel:#18201d; --ink:#e7eeea; --muted:#9aa8a1; --line:#2b3632;
  --accent:#6cc7a9; --accent-soft:#1d3a31; --warn:#e2b45c; --warn-soft:#3a2f17; --none:#7d8984; --none-soft:#232b28;
  --heat0:#1c2421; --heat1:#1e3d33; --heat2:#256049; --heat3:#2f8a69; --heat4:#6cc7a9; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#111614; --panel:#18201d; --ink:#e7eeea; --muted:#9aa8a1; --line:#2b3632;
  --accent:#6cc7a9; --accent-soft:#1d3a31; --warn:#e2b45c; --warn-soft:#3a2f17; --none:#7d8984; --none-soft:#232b28;
  --heat0:#1c2421; --heat1:#1e3d33; --heat2:#256049; --heat3:#2f8a69; --heat4:#6cc7a9; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:14px;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding-block:28px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1{font-family:var(--display);font-size:clamp(26px,4vw,36px);margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font-size:15px;font-weight:600;margin:0 0 10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.lede{max-width:72ch;color:var(--muted);margin:8px 0 0}
.stats{display:flex;flex-wrap:wrap;gap:12px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 16px;min-width:150px;flex:1 1 150px}
.stat b{display:block;font-family:var(--mono);font-size:22px;font-weight:500;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:12.5px}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:middle}
th{font-size:12px;font-weight:600;color:var(--muted);background:var(--panel);position:sticky;top:0;z-index:1}
th.sort{cursor:pointer;user-select:none} th.sort:hover{color:var(--ink)} th.sort:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
td.num{font-family:var(--mono);font-size:12.5px;text-align:right}
td.mono{font-family:var(--mono);font-size:12px}
td.wrap{white-space:normal;min-width:220px;max-width:340px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:500}
.p-pred{background:var(--accent-soft);color:var(--accent)} .p-weak{background:var(--warn-soft);color:var(--warn)}
.p-rare{background:var(--none-soft);color:var(--ink)} .p-none{background:var(--none-soft);color:var(--none)}
.heat{display:inline-block;width:22px;height:16px;border-radius:3px;vertical-align:middle}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
input,select{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:var(--ink)}
input{min-width:220px;flex:1 1 220px}
.count{color:var(--muted);font-size:12.5px}
.legend{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:12px;flex-wrap:wrap}
.grid2{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px}
@media (max-width:820px){.grid2{grid-template-columns:minmax(0,1fr)}}
.note{color:var(--muted);font-size:12.5px;max-width:90ch}
</style>
<div class="wrap">
  <header>
    <h1>FX Feature Predictability</h1>
    <p class="lede" id="lede"></p>
  </header>
  <section class="stats" id="stats"></section>
  <section class="grid2">
    <div><h2>By timeframe</h2><div class="scroll"><table id="tft"></table></div></div>
    <div><h2>By category</h2><div class="scroll"><table id="catt"></table></div></div>
  </section>
  <section>
    <h2>All features</h2>
    <div class="controls">
      <input id="q" type="search" placeholder="Search feature, name or category" aria-label="Search features">
      <select id="cat" aria-label="Category"><option value="">All categories</option></select>
      <select id="ver" aria-label="Verdict"><option value="">All verdicts</option></select>
      <span class="count" id="count"></span>
    </div>
    <div class="legend">Share of tests significant per timeframe:
      <span class="heat" style="background:var(--heat0)"></span>0%
      <span class="heat" style="background:var(--heat1)"></span>&lt;2%
      <span class="heat" style="background:var(--heat2)"></span>2-5%
      <span class="heat" style="background:var(--heat3)"></span>5-15%
      <span class="heat" style="background:var(--heat4)"></span>&ge;15%</div>
    <div class="scroll" style="margin-top:8px;max-height:72vh"><table id="ft"></table></div>
    <p class="note" id="method"></p>
  </section>
</div>
<script>
const D = __DATA__;
const pct = v => v == null ? "" : (100 * v).toFixed(v < 0.01 && v > 0 ? 2 : 1) + "%";
const f3 = v => v == null ? "" : v.toFixed(3);
const f2 = v => v == null ? "" : v.toFixed(2);
const heat = v => v == null ? "var(--heat0)" : v >= .15 ? "var(--heat4)" : v >= .05 ? "var(--heat3)" : v >= .02 ? "var(--heat2)" : v > 0 ? "var(--heat1)" : "var(--heat0)";
const pill = v => {const c = v.startsWith("predictive") ? "p-pred" : v.startsWith("weakly") ? "p-weak" : v.startsWith("rarely") ? "p-rare" : "p-none";
  const t = v.startsWith("predictive") ? "Predictive" : v.startsWith("weakly") ? "Weak" : v.startsWith("rarely") ? "Rare" : "None";
  return `<span class="pill ${c}" title="${v}">${t}</span>`};
const S = D.summary;
document.getElementById("lede").textContent =
  `Out-of-sample information each of the ${S.features} features carries about the next 1, 6 and 24 bars, on ${S.pairs} currency pairs and metals across ${S.tfs.join(", ")}. ` +
  `A test is significant when it beats 20 time-shifted copies of the same feature after Benjamini-Hochberg correction over all ${S.tests.toLocaleString()} tests. ` +
  `"Covers costs" asks whether the best significant test's predicted move exceeds a retail round trip.`;
const v = S.verdicts || {};
const stats = [[S.features, "features tested"], [S.tests.toLocaleString(), "feature x pair x timeframe x horizon tests"],
  [pct(S.share_significant), "of tests significant (chance: under 0.1%)"],
  [(v["predictive (significant in >=5% of its tests)"] || 0), "features predictive in 5% or more of their tests"],
  [S.significant_covering_cost.toLocaleString(), "significant tests whose move exceeds the round-trip cost"]];
document.getElementById("stats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
document.getElementById("tft").innerHTML = "<tr><th>Timeframe</th><th>Tests</th><th>Significant</th><th>z &gt; 3</th><th>Median IC</th><th>Best IC</th><th>Cost (bp)</th><th>Edge / cost, significant</th><th>Cover cost</th></tr>" +
  D.by_tf.map(r => `<tr><td class="mono">${r.tf}</td><td class="num">${r.tests.toLocaleString()}</td><td class="num">${pct(r.share_sig)}</td><td class="num">${pct(r.share_z3)}</td><td class="num">${f3(r.median_ic)}</td><td class="num">${f3(r.best_ic)}</td><td class="num">${f2(r.median_cost)}</td><td class="num">${f2(r.sig_median_edge_over_cost)}</td><td class="num">${r.sig_covering_cost}</td></tr>`).join("");
document.getElementById("catt").innerHTML = "<tr><th>Category</th><th>Features</th><th>Predictive</th><th>Weak</th><th>Median share significant</th></tr>" +
  D.cats.map(r => `<tr><td>${r.category}</td><td class="num">${r.features}</td><td class="num">${r.predictive}</td><td class="num">${r.weak}</td><td class="num">${pct(r.median_share_sig)}</td></tr>`).join("");
const catSel = document.getElementById("cat"), verSel = document.getElementById("ver");
[...new Set(D.rows.map(r => r.category))].sort().forEach(c => catSel.insertAdjacentHTML("beforeend", `<option>${c}</option>`));
[...new Set(D.rows.map(r => r.verdict))].forEach(c => verSel.insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`));
const cols = [["rank", "#", "num"], ["feature", "Feature", "mono"], ["name", "Name", "wrap"], ["category", "Category", ""],
  ["verdict", "Verdict", "pill"], ["share_significant_bh", "Significant", "pct"], ["share_z_gt_3", "z > 3", "pct"],
  ["median_ic", "Median IC", "f3"], ...D.tfs.map(t => ["sig_" + t, t, "heat"]), ["best_z", "Best z", "f2"],
  ["best_test", "Best test", "mono"], ["best_sig_edge_test", "Best significant edge vs cost", "mono"], ["covers_costs", "Covers costs", ""]];
let sortKey = "rank", asc = true;
function render() {
  const q = document.getElementById("q").value.toLowerCase(), c = catSel.value, ve = verSel.value;
  let rs = D.rows.filter(r => (!c || r.category === c) && (!ve || r.verdict === ve) &&
    (!q || (r.feature + " " + r.name + " " + r.category).toLowerCase().includes(q)));
  rs.sort((a, b) => {const x = a[sortKey], y = b[sortKey]; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * (asc ? 1 : -1)});
  document.getElementById("count").textContent = `${rs.length} of ${D.rows.length} features`;
  const head = "<tr>" + cols.map(([k, l]) => `<th class="sort" tabindex="0" data-k="${k}">${l}${sortKey === k ? (asc ? " ▲" : " ▼") : ""}</th>`).join("") + "</tr>";
  const body = rs.map(r => "<tr>" + cols.map(([k, , t]) => {const x = r[k];
    if (t === "pct") return `<td class="num">${pct(x)}</td>`;
    if (t === "f3") return `<td class="num">${f3(x)}</td>`;
    if (t === "f2") return `<td class="num">${f2(x)}</td>`;
    if (t === "num") return `<td class="num">${x}</td>`;
    if (t === "heat") return `<td title="${pct(x)}"><span class="heat" style="background:${heat(x)}"></span></td>`;
    if (t === "pill") return `<td>${pill(x)}</td>`;
    return `<td class="${t}">${x == null ? "" : x}</td>`}).join("") + "</tr>").join("");
  const tb = document.getElementById("ft"); tb.innerHTML = head + body;
  tb.querySelectorAll("th.sort").forEach(th => {const go = () => {const k = th.dataset.k; asc = sortKey === k ? !asc : true; sortKey = k; render()};
    th.addEventListener("click", go); th.addEventListener("keydown", e => {if (e.key === "Enter" || e.key === " ") {e.preventDefault(); go()}})});
}
["q", "cat", "ver"].forEach(id => document.getElementById(id).addEventListener("input", render));
render();
document.getElementById("method").textContent =
  "Method: walk-forward with 3-year training and 6-month test windows from 2010 on 2007-2018 data (EURUSD to 2022). In each fold the sign of the feature's relation with the next move is learned on training data and measured on the test window, with the feature standardised by training statistics only. " +
  "The null shifts the target in time by 20 random offsets, which keeps the feature's own persistence. IC is the out-of-sample correlation, equal to the normalised P&L of holding a position proportional to the feature, before costs. " +
  "Edge vs cost compares half the top-minus-bottom-decile spread of the next move with a round trip (spread x 1.25, slippage, commission). M1 is not included: building every feature on about 4 million one-minute bars per pair exceeds the cloud machines' memory.";
</script>
"""
open(out, "w").write(html.replace("__DATA__", data))
print("written", out, len(html) + len(data))
