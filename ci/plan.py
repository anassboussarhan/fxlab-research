"""Compute the job matrices of the cloud sweep from ci/sweep_config.json (mode = smoke | full) and write them to
$GITHUB_OUTPUT."""
import json
import os

c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_config.json")))
m = c["mode"]
pairs = c[f"pairs_{m}"]
hi, lo = c[f"hi_tfs_{m}"], c[f"lo_tfs_{m}"]
first = c["smoke_first_year"] if m == "smoke" else c["first_year"]
out = {
    "mode": m, "first_year": str(first), "last_year": str(c["last_year"]), "pairs_csv": "," + ",".join(pairs) + ",",
    "data": json.dumps({"pair": pairs}),
    "hi": json.dumps({"pair": pairs, "tf": hi}) if hi else "", "lo": json.dumps({"pair": pairs, "tf": lo}) if lo else "",
    "models": json.dumps({"pair": pairs}) if c[f"models_{m}"] else "",
}
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
    for k, v in out.items():
        f.write(f"{k}={v}\n")
print(json.dumps(out, indent=1))
