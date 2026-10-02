"""Indian equity delivery (CNC) trading costs.

Defaults approximate Upstox delivery charges in 2026. Check the broker's
current charge sheet before going live and update these numbers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryCosts:
    brokerage_per_order: float = 20.0      # flat ₹ per executed order
    brokerage_pct_cap: float = 0.025        # brokerage capped at 2.5% of order value
    stt_pct: float = 0.001                  # 0.1% on buy and sell (delivery)
    exchange_txn_pct: float = 0.0000297     # NSE transaction charge
    sebi_fee_pct: float = 0.000001          # ₹10 per crore
    stamp_duty_buy_pct: float = 0.00015     # 0.015% on buy side only
    gst_pct: float = 0.18                   # on brokerage + exchange + SEBI fees
    dp_charge_per_sell: float = 20.0        # depository charge per scrip sold (approx, incl. GST)
    slippage_pct: float = 0.0015            # assumed fill worse than the open price

    def charges(self, value: float, side: str) -> float:
        """Statutory and broker charges in ₹ for an order of `value` rupees (excludes slippage)."""
        if value <= 0:
            return 0.0
        brokerage = min(self.brokerage_per_order, value * self.brokerage_pct_cap)
        stt = value * self.stt_pct
        exch = value * self.exchange_txn_pct
        sebi = value * self.sebi_fee_pct
        gst = (brokerage + exch + sebi) * self.gst_pct
        total = brokerage + stt + exch + sebi + gst
        if side == "buy":
            total += value * self.stamp_duty_buy_pct
        else:
            total += self.dp_charge_per_sell
        return total

    def fill_price(self, price: float, side: str) -> float:
        return price * (1 + self.slippage_pct) if side == "buy" else price * (1 - self.slippage_pct)
