#!/usr/bin/env bash
# Usage: ci/dl_each.sh RUN REGEX DIR -> every artifact of run RUN whose name matches REGEX, one by one into DIR/NAME,
# each with retries (artifact downloads fail at times with "HTTP 503 Egress is over the account limit"). Exits 1 when
# any matching artifact could not be downloaded, so a job never runs on part of its data.
set -u
R=$1; RX=$2; D=$3
names=""
for k in 1 2 3 4 5 6; do
  names=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$R/artifacts?per_page=100" --paginate -q '.artifacts[].name' 2>/dev/null) && break
  sleep $((15 * k))
done
fail=0; n=0
for a in $(echo "$names" | grep -E "$RX"); do
  n=$((n + 1)); ok=0
  for k in 1 2 3 4 5 6 7 8; do
    rm -rf "$D/$a"
    if gh run download $R -n "$a" -D "$D/$a" 2>/tmp/dl_err; then ok=1; break; fi
    echo "retry $k $a: $(tail -c 160 /tmp/dl_err)"; sleep $((30 * k))
  done
  [ $ok = 1 ] || { echo "FAILED $a"; fail=1; }
done
echo "downloaded $n artifacts matching $RX from run $R (failed: $fail)"
[ $n -gt 0 ] && [ $fail = 0 ]
