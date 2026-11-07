
from dataclasses import dataclass


@dataclass
class TransactionCostConfig:
    # All rates are configurable.
    # Rates must be entered as decimals:
    # 0.001 = 0.1%, 0.18 = 18%.

    brokerage_per_order: float = 0.0
    exchange_txn_rate: float = 0.0
    stt_sell_rate: float = 0.0
    sebi_rate: float = 0.0
    gst_rate: float = 0.0
    stamp_buy_rate: float = 0.0

    # Estimated slippage in basis points.
    # 10 basis points = 0.10%.
    slippage_bps: float = 0.0

    # Current simulator uses one simulated unit.
    # Later, configure this with the actual contract lot size.
    contract_multiplier: float = 1.0


class TransactionCostCalculator:

    def __init__(self, config=None):
        self.config = config or TransactionCostConfig()

    def calculate(self, position):

        if position.status == "OPEN" or position.exit_price is None:
            raise ValueError(
                "Transaction costs can only be calculated "
                "for closed positions."
            )

        config = self.config

        quantity = float(position.quantity)
        multiplier = float(config.contract_multiplier)

        if quantity <= 0 or multiplier <= 0:
            raise ValueError("Quantity and multiplier must be positive.")

        units = quantity * multiplier

        entry_price = float(position.entry_price)
        exit_price = float(position.exit_price)

        buy_turnover = entry_price * units
        sell_turnover = exit_price * units
        total_turnover = buy_turnover + sell_turnover

        # Gross P&L before transaction costs.
        gross_pnl = float(position.pnl) * multiplier

        # Brokerage is charged on both entry and exit orders.
        brokerage = config.brokerage_per_order * 2

        exchange_charges = (
            total_turnover * config.exchange_txn_rate
        )

        stt = sell_turnover * config.stt_sell_rate

        sebi_charges = (
            total_turnover * config.sebi_rate
        )

        stamp_duty = (
            buy_turnover * config.stamp_buy_rate
        )

        # GST applies to eligible service charges.
        gst = (
            brokerage + exchange_charges + sebi_charges
        ) * config.gst_rate

        # Estimated slippage on entry and exit turnover.
        slippage = (
            total_turnover * config.slippage_bps / 10000
        )

        total_charges = (
            brokerage
            + exchange_charges
            + stt
            + sebi_charges
            + stamp_duty
            + gst
        )

        total_cost = total_charges + slippage

        net_pnl = gross_pnl - total_cost

        return {
            "gross_pnl": round(gross_pnl, 2),
            "brokerage": round(brokerage, 2),
            "exchange_charges": round(exchange_charges, 2),
            "stt": round(stt, 2),
            "sebi_charges": round(sebi_charges, 2),
            "stamp_duty": round(stamp_duty, 2),
            "gst": round(gst, 2),
            "total_charges": round(total_charges, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
            "net_pnl": round(net_pnl, 2),
        }

    @staticmethod
    def display(result):

        print("\nTRANSACTION COST REPORT")
        print("=" * 45)

        print("Gross P&L:", f'Rs.{result["gross_pnl"]:.2f}')

        print("\nCHARGES")
        print("-" * 45)

        print("Brokerage:", f'Rs.{result["brokerage"]:.2f}')
        print("Exchange charges:", f'Rs.{result["exchange_charges"]:.2f}')
        print("STT:", f'Rs.{result["stt"]:.2f}')
        print("SEBI charges:", f'Rs.{result["sebi_charges"]:.2f}')
        print("Stamp duty:", f'Rs.{result["stamp_duty"]:.2f}')
        print("GST:", f'Rs.{result["gst"]:.2f}')

        print("Total charges:", f'Rs.{result["total_charges"]:.2f}')
        print("Estimated slippage:", f'Rs.{result["slippage"]:.2f}')

        print("\nFINAL RESULT")
        print("-" * 45)

        print("Total estimated cost:", f'Rs.{result["total_cost"]:.2f}')
        print("Net P&L:", f'Rs.{result["net_pnl"]:.2f}')
