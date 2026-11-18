"""Round 23 LW relaunch: the wave-1 units not finished in run OLD (cancelled because its H1/M30/M15 jobs ran past
the 350-minute job limit and lost their output). Group 0 jobs are rerun as they were; each unfinished group 1 job is
split into one job per timeframe. Prints the matrix JSON. Usage: python ci/r23a2_list.py OLD_RUN_ID"""
import json
import os
import subprocess
import sys

old = sys.argv[1]
repo = os.environ["GITHUB_REPOSITORY"]
out = subprocess.run(["gh", "api", f"repos/{repo}/actions/runs/{old}/jobs?per_page=100", "--paginate", "-q",
                      '.jobs[] | select(.conclusion=="success") | .name'], capture_output=True, text=True, check=True).stdout
ok = {n.rsplit(", ", 1)[-1].rstrip(")") for n in out.splitlines() if n.startswith("lw (")}
inc = []
for j in json.load(open("ci/r23a_wave1.json"))["include"]:
    if j["tag"] in ok:
        continue
    if j["tag"].endswith("-1"):
        for tf in j["tfs"].split(","):
            inc.append({"symbol": j["symbol"], "tfs": tf, "tag": f"{j['tag']}{tf.lower()}"})
    else:
        inc.append(j)
print(json.dumps({"include": inc}))
print(f"finished in {old}: {len(ok)}; to run: {len(inc)}", file=sys.stderr)
