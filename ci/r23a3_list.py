"""Round 23 LW relaunch 2 (docs/round23_preregistration.md, addendum 6): every wave-1 LW job not finished in full in
runs RUN... A tag counts as finished when its job succeeded and its log reports "skipped 0". Group 0 tags are run as
they were; an unfinished group 1 tag is split into one job per timeframe (unless that split tag finished in full).
Prints the matrix JSON. Usage: python ci/r23a3_list.py RUN [RUN ...]"""
import json
import os
import re
import subprocess
import sys

repo = os.environ["GITHUB_REPOSITORY"]


def gh(*a):
    for _ in range(4):
        r = subprocess.run(["gh", "api", *a], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    raise RuntimeError(f"gh api {a}: {r.stderr[:300]}")


full, partial = set(), {}
for run in sys.argv[1:]:
    out = gh(f"repos/{repo}/actions/runs/{run}/jobs?per_page=100", "--paginate", "-q",
             '.jobs[] | select(.conclusion=="success") | "\\(.id)\\t\\(.name)"')
    for line in out.splitlines():
        jid, name = line.split("\t", 1)
        if not name.startswith("lw ("):
            continue
        tag = name.rsplit(", ", 1)[-1].rstrip(")")
        m = re.search(r" units (\d+) done (\d+) skipped (\d+) ", gh(f"repos/{repo}/actions/jobs/{jid}/logs"))
        if m and int(m.group(3)) == 0:
            full.add(tag)
        else:
            partial[tag] = m.group(0).strip() if m else "no summary line"
inc = []
for j in json.load(open("ci/r23a_wave1.json"))["include"]:
    if j["tag"] in full:
        continue
    if j["tag"].endswith("-1"):
        for tf in j["tfs"].split(","):
            t = f"{j['tag']}{tf.lower()}"
            if t not in full:
                inc.append({"symbol": j["symbol"], "tfs": tf, "tag": t})
    else:
        inc.append(j)
print(json.dumps({"include": inc}))
print(f"finished in full: {len(full)}; partial (rerun): {partial}; to run: {len(inc)}", file=sys.stderr)
