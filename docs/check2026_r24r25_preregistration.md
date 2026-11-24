# 2026 check of the round 24-25 counting streams (pre-registered 2026-11-24, before any 2026 data was read)

The rules of rounds 24 and 25: "a stream that counts gets one pre-registered check on 2026-01-01 to 2026-10-31 with
its frozen rule, written before 2026 data is read". This is the sixth use of 2026. No 2026 price, rate or funding
value has been read for these instruments.

## What is checked

1. **USD/TRY, round 24 R1:** the equal-weight portfolio of the 271 counting R1 unit streams.
2. **USD/TRY, round 25 part P:** the equal-weight portfolio of the 326 counting part-P USD/TRY streams. The 31 FX
   panels are not checked separately: they are this effect.
3. **The five single round-25 streams:**
   - ALT GALAUSDT S.DAILY.DOUBLE_SEVEN H1 (default);
   - ALT ADAUSDT S.VOL.COMPRESSION H1 (default);
   - ALT WAVESUSDT S.SEAS.WEEKEND_GAP_FADE H8 (default);
   - P EUSIDXEUR S.BO.RANGE_EXPANSION W1 (nested-plus);
   - P CHEIDXCHF S.BO.RANGE_EXPANSION W1 (nested-plus).
4. **R7 BTC basis** is not checked: its placebos pass as often, so it is the known cash-and-carry.

## Frozen rules

Each unit is rerun with its own code, seeds and configurations on data to 2026-10-31. The selection that trades in
2026 is the one each stream makes at its 2026 re-selection points from data before them:
- default: top 5 by expanding t, re-selected each January;
- nested / nested-plus: the setting chosen each January from inner streams ending before it.

Nothing is refitted on 2026. The 2019-2025 part of every rerun must match the published stream (t within 0.01);
otherwise that unit is reported as not reproduced and left out.

## Data

- **USD/TRY, EUSIDXEUR, CHEIDXCHF:** Dukascopy hour candles for 2026-01 to 2026-10 (the round-23 downloader),
  appended to the stores used in rounds 24-25.
- **Rates:** BIS policy rates for 2026, with the same one-month lag.
- **Altcoins:** Binance archive 1-hour klines and funding for 2026-01 to 2026-10. A coin with no 2026 data in the
  archive (delisted) is reported as such.

## Criteria (one-sided, as the counting rule)

| Stream | Passes if |
|---|---|
| each USD/TRY portfolio (items 1 and 2) | 2026 mean > 0 and t ≥ 1.645 at carry x1 |
| each single stream | 2026 mean > 0 and t ≥ 1.645 |

For the USD/TRY portfolios, the result at carry x1.5 and buy-and-hold USD/TRY with carry over the same months are
also reported.

10 months is short, so a failure or a pass at these thresholds says little on its own.

Not financial advice.
