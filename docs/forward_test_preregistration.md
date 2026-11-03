# Forward test from November 2026: pre-registration

Written and committed on 2026-11-02, before the first Gotobi night after the 2026 holdout (2026-11-05; entry
2026-11-04 23:00 Tokyo = 14:00 UTC). No price, positioning or market data after 2026-10-31 had been downloaded or
read, and the frozen model below was fitted on development nights only (2007-2025). Nothing is re-tuned after
this commit. A coding or data error found later is fixed and reported next to the first result, never in its place.

## Frozen rules

| ID | Rule | Development evidence |
|---|---|---|
| T | **The trade:** long USD/JPY and EUR/JPY, equal weight, from 23:00 Tokyo the evening before to 09:53 Tokyo on Gotobi days (5th, 10th, 15th, 20th, 25th, month end; the previous Tokyo business day when that is a weekend or a Japanese holiday); no trade when the entry has no quote within 10 minutes (Gotobi Mondays). Costs, fills and swap as in development | +4.6 bp a night, 2007-2025; +2.0 bp in the 2026 holdout |
| S1 | **Disaster stop, 80 bp** below each pair's entry (resting order; a gap fills at the next price) | the 2007-2018 pick for the worst case; 2019-2025 +4.96 bp vs +4.77 without, worst -81 vs -141 bp (`docs/untested_batch.md` 9.15) |
| S2 | Disaster stop, 100 bp (secondary) | 2007-2018 / 2019-2025 +4.68 / +4.71 bp |
| FV1 | **Winning-night filter forward-v1:** `results/frozen_forward_v1.json` (sha256 `b5b5c96a7e8f0aca123eac17074f69f687dece2fa9def46974ac70e6e771f8a7`), logistic regression fitted on all 1,067 development nights; trade the night only when its predicted probability of a win is above 0.5126 (the lowest-tercile cut of the fit). Features computed exactly as `features()` in `scripts/gotobi_accuracy_lab.py` at commit b90884b (USD/JPY M1 bars, Yahoo ^VIX, ^GSPC, ^N225, ^TNX, CFTC legacy JPY reports); scored with `score()` in `scripts/freeze_forward.py` | 2019-2025 (model fitted on 2007-2018): kept 63% winners, +7.5 bp vs +4.8 all; walk-forward 2012-2025: kept +8.4 bp, skipped -0.8 bp, kept above skipped in 13 of 14 years |

## Periods

- **First look (primary):** Gotobi nights 2026-11-05 to 2027-10-29 (about 53 traded nights).
- **Second look:** to 2028-10-31. Interim results may be reported for information; they change nothing.

## Tests and verdicts, decided now

| Test | Statistic | Confirmed | Consistent | Not supported | Contradicted |
|---|---|---|---|---|---|
| **FT1 (primary): the filter** | mean no-stop net of kept nights minus skipped nights, Welch t | > 0 and t >= 1.645 | > 0 | <= 0 | t <= -1.645 |
| FT2: kept nights | mean no-stop net of kept nights, t | t >= 1.645 | > 0 | <= 0 | |
| FT3: the 80 bp stop | paired difference S1 - no stop, all nights | "keeps the gain" if >= -0.5 bp a night; "cuts the loss" if both the worst trade and CVaR5 of the open loss are smaller | | | |
| FT4: the trade itself | mean no-stop net, all nights, t | t >= 1.645 | > 0 | <= 0 | below +4.6 bp with t of the difference <= -1.645 |

S2 is reported next to S1 with the same statistics, as information.

## Power, stated before the run

With about 35 kept and 18 skipped nights in the first year, a per-night sd of 30-35 bp and the walk-forward gap
(+8.4 vs -0.8 bp), the expected FT1 t is about 1.0 (a 25-30% chance of "confirmed"); after two years about 1.5
(about 45%). "Consistent" is the most likely outcome even if the filter works; a contradiction is what would count
against it.

## Procedure

- Data: Dukascopy M1 for USD/JPY and EUR/JPY (downloaded in CI after each month), Yahoo daily closes, CFTC
  reports. The evaluation code is written later, uses the frozen JSON without refitting, and must first reproduce
  the development numbers on 2025 nights (dry run) before any forward night is evaluated.
- A second filter (forward-v2) may be frozen before 2026-11-04 14:00 UTC from the losing-night analysis of
  `scripts/gotobi_losses_lab.py`; it will be tested alongside forward-v1, never in its place.

## Addendum (2026-11-02, still before the first forward night): forward-v2

The losing-night lab (run 41431532005) applied its pre-set rule: the extended win model qualifies as forward-v2 when
its walk-forward kept-minus-skipped gap (2012-2025) and its 2019-2025 AUC both beat forward-v1's (+9.2 bp, 0.58).
It did, narrowly (+9.3 bp, 0.61; in the same run forward-v1's features gave +8.8 bp and 0.58, and the walk-forward
AUCs are 0.58 vs 0.59), so it is frozen and tested as a secondary filter:

| ID | Rule |
|---|---|
| FV2 | `results/frozen_forward_v2.json` (sha256 `27e57a992a21994e90d8ee0f911d966d5e9840c08780cae7c98b77f60e677835`): the same fitting on all 1,067 development nights with 80 features known at entry (`features()` of `scripts/gotobi_accuracy_lab.py` at b90884b joined with `more_features()` of `scripts/gotobi_losses_lab.py` at b82707d, columns starting `night_` excluded); trade the night when the predicted probability of a win is above 0.4832 |

FT5 (secondary): FT1 for forward-v2, with the same verdicts. FT1 (forward-v1) stays the primary test.

## Dry run (2026-11-02): passed

`scripts/forward_eval.py --dry 2025` (run 41434905713) rebuilt the 58 Gotobi nights of 2025 from the development
store with the forward code: trades (no stop and 80 bp stop) identical to the development files (largest difference
1e-12 bp), all 24 forward-v1 and 80 forward-v2 features identical, predicted probabilities identical (1e-13) and the
same trade / skip decision on every night for both filters. The monthly job (`.github/workflows/forward_test.yml`)
runs the same code on the forward nights.

## Addendum 2 (2026-11-03, before the first forward night): FW1, sizing by the filter's probability (exploratory)

The edges batch tested E1 (`docs/edges_batch_preregistration.md`, `scripts/edges_e1.py`, `docs/edges_batch/e1.json`).
On 2019-2025 walk-forward nights, sizing each night by the forward-v1 probability raised the FTMO 2-Step 5x chance
of being paid at the 2026-shifted edge from 49% (all nights) and 44% (skip a third) to 60%. It did **not** meet its
pre-set rule, which also required the weighted night's t (3.62) to beat the skip rule's (3.95). So it is not a
confirmed rule. It is added to the forward report as information only, with no verdict:

| ID | Rule |
|---|---|
| FW1 | weight = min(p_v1 / 0.55764, 2), with p_v1 from the frozen forward-v1 model and 0.55764 its mean over the 1,067 development nights; reported: weighted vs unweighted mean of the 80 bp stop nights and their paired difference |

## Addendum 3 (2026-11-03, before the first forward night): FW2, the 5th-20th Gotobi days (exploratory)

In development, the 25th and month-end Gotobi days earned +1.1 bp against +4.5 bp for all days
(`docs/untested_batch.md` 9.4, T4a). That split was seen before this line was written, so it is judged only on
forward nights.

**FW2:** the mean of the 80 bp stop nights on the 5th, 10th, 15th and 20th, against the 25th and month-end, with
the Welch t of the difference. The slot comes from the lab's `day_25` and `month_end` features. Information only,
no verdict.
