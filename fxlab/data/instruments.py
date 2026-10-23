"""Instrument metadata. Tradable assets vs explanatory-only inputs are explicit."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    base: str
    quote: str
    pip: float              # price increment treated as one pip
    contract: float         # units of base per 1.0 lot
    asset_class: str        # "fx" | "metal" | "index" | "commodity"
    tradable: bool = True   # False => explanatory input only
    commission_per_lot_side_usd: float = 3.5  # ECN-style default, overridable

    @property
    def is_jpy(self) -> bool:
        return self.quote == "JPY"


def _fx(sym: str, **kw) -> Instrument:
    base, quote = sym[:3], sym[3:]
    pip = 0.01 if quote == "JPY" else 0.0001
    return Instrument(sym, base, quote, pip, 100_000.0, "fx", **kw)


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
CROSSES = ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD", "GBPJPY", "GBPCHF",
           "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD", "CADJPY",
           "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD"]
EXOTICS = ["USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDTRY", "USDCNH", "EURSEK", "EURNOK"]
METALS = ["XAUUSD", "XAGUSD"]

INSTRUMENTS: dict[str, Instrument] = {s: _fx(s) for s in MAJORS + CROSSES + EXOTICS}
INSTRUMENTS["XAUUSD"] = Instrument("XAUUSD", "XAU", "USD", 0.01, 100.0, "metal")
INSTRUMENTS["XAGUSD"] = Instrument("XAGUSD", "XAG", "USD", 0.001, 5000.0, "metal")

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]


def get(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError as e:
        raise KeyError(f"Unknown instrument {symbol}") from e


def pip_value_usd(symbol: str, price: float, usd_quote_rate: float | None = None) -> float:
    """USD value of one pip for one lot.

    For XXXUSD it is pip*contract. For USDXXX it is pip*contract/price. For crosses
    the caller must pass usd_quote_rate = USD per one unit of quote currency.
    """
    ins = get(symbol)
    raw = ins.pip * ins.contract  # in quote currency
    if ins.quote == "USD":
        return raw
    if ins.base == "USD":
        return raw / price
    if usd_quote_rate is None:
        raise ValueError(f"{symbol}: need USD per {ins.quote} to value a pip")
    return raw * usd_quote_rate
