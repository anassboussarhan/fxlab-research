# Round 22: robustness of the fair-value-gap family under harsher execution (development data only).
set -u
mkdir -p out
for S in USATECHIDXUSD USA500IDXUSD USDJPY USDCHF EURUSD; do bash ci/r22_place.sh $S > /dev/null 2>&1; done
pids=""
for S in USATECHIDXUSD USA500IDXUSD USDJPY USDCHF EURUSD; do
  python -W ignore scripts/r22_fvg_check.py --symbol $S --out out > out/log_$S.txt 2>&1 &
  pids="$pids $!"
done
wait $pids
tail -5 out/log_*.txt
