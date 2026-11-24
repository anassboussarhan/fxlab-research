# 2026 check of the round 24-25 counting streams: results (2026-11-24)

Pre-registration: `check2026_r24r25_preregistration.md` (with addendum 1). Run: check2026 45331755459, data to
2026-10-31.

| Stream | 2026 | Result |
|---|---|---|
| USD/TRY R1 portfolio (271 streams) | 0.83% a day, t 14.7 | **void:** trade-engine fault (round 24 addendum 9) |
| USD/TRY part-P portfolio (326 streams) | 0.49% a day, t 13.4 | **void:** same fault |
| ADAUSDT compression, H1 | t 1.35 | fails |
| EUSIDXEUR range expansion, W1 | flat all year | fails |
| CHEIDXCHF range expansion, W1 | flat all year | fails |
| GALAUSDT Double Seven, H1 | not a 2026 member (top 30 by 2025 volume) | does not trade |
| WAVESUSDT weekend-gap fade, H8 | not a 2026 member | does not trade |

**How the fault was found:**
- Buy-and-hold USD/TRY with carry lost in 2026 (t -7.3) on a 15% rise, yet the portfolio showed 0.83% a day.
- One configuration's 2026 returns came from a long opened in 2019. The engine marked it against its 2019 entry
  price, so it carried about nine times its exposure.

**On the fixed engine** no USD/TRY stream counts, so there is nothing left to check on 2026.

**The 2026 data used:**
- the Dukascopy hour candles: no crossed quotes, normal spreads, no break at the year boundary;
- the BIS rates to 2026-10;
- the Binance archive.

This is the sixth use of 2026.

Not financial advice.
