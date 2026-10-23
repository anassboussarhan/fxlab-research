"""Transaction-cost model. Every component is explicit and stress-testable.

Price-based costs come from the REAL quoted bid/ask of the execution bar
(longs buy the ask, shorts sell the bid). On top of that:
  * slippage   = slip_spread_frac * quoted spread + slip_fixed_pips * pip  (market & stop orders; limit orders 0)
  * commission = commission_bps_per_side * notional  (default 0.35 bp ≈ $3.5 per $100k per side)
  * financing  = swap_pips_per_night * pip charged per NY-17:00 rollover held (x3 on the Wednesday rollover);
                 historical swap rates are NOT available from the reachable data sources, so a conservative
                 negative carry is charged on both sides by default (documented limitation)
  * latency    = execution delayed by `latency_bars` bars after the decision bar
  * spread_mult = multiplier applied to the quoted spread (stress testing / spread widening)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class CostModel:
    spread_mult: float = 1.0
    slip_spread_frac: float = 0.25
    slip_fixed_pips: float = 0.1
    commission_bps_per_side: float = 0.35
    swap_pips_per_night: float = 0.5
    latency_bars: int = 0
    name: str = "base"

    def stressed(self, factor: float) -> "CostModel":
        return replace(self, spread_mult=self.spread_mult * factor, slip_spread_frac=self.slip_spread_frac * factor,
                       slip_fixed_pips=self.slip_fixed_pips * factor,
                       commission_bps_per_side=self.commission_bps_per_side * factor,
                       swap_pips_per_night=self.swap_pips_per_night * factor, name=f"x{factor:g}")

    def to_dict(self):
        return asdict(self)


ZERO = CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0, "zero")  # diagnostics only; never the primary result
BASE = CostModel()
STRESS = {f: BASE.stressed(f) for f in (1.5, 2.0, 3.0)}
LATENCY = {k: replace(BASE, latency_bars=k, name=f"latency{k}") for k in (1, 2, 5)}
