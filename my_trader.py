import json
from collections import defaultdict, deque
from datamodel import Order, TradingState

POSITION_LIMIT = {
    "INTARIAN_PEPPER_ROOT": 20,
    "ASH_COATED_OSMIUM": 20,
}

# =========================
# INTARIAN_PEPPER_ROOT
# FV(day, ts) = 9998.5 + 0.001 * ts + 1000 * (day + 2)
# =========================
IPR_BASE = 9998.5
IPR_SLOPE = 0.001
IPR_DAY_STEP = 1000
IPR_MAX_DAY = 20

# Fractions of the day rather than hardcoded timestamps.
IPR_OPEN_AGGR_FRAC = 0.20
IPR_EOD_START_FRAC = 0.94
IPR_EOD_HARD_FRAC = 0.985
IPR_PASSIVE_OFFSET = 1

# =========================
# ASH_COATED_OSMIUM
# stable around 10000, wide spread, MM preferred
# =========================
ACO_FAIR = 10000
ACO_BASE_SIZE = 5
ACO_INVENTORY_SOFT = 10
ACO_INVENTORY_HARD = 16
ACO_EOD_START_FRAC = 0.95
ACO_EOD_HARD_FRAC = 0.995

OBI_BETA_INIT = 2.0
OBI_W1, OBI_W2 = 0.4, 0.6
OBI_WARMUP = 100
OBI_UPDATE_N = 50


class Trader:
    def __init__(self):
        self._ipr_day = -2
        self._last_ts = None
        self._day_max_ts_seen = 100_000  # safe default for both 1k-tick and 10k-tick styles

        self._obi_pairs_aco = deque(maxlen=2000)
        self._last_obi_aco = 0.0
        self._last_mid_aco = None
        self._obi_beta_aco = OBI_BETA_INIT
        self._obi_samples = 0

        self._bot_pos = defaultdict(lambda: defaultdict(float))

    def _load(self, data: str) -> None:
        if not data:
            return
        try:
            d = json.loads(data)
            self._ipr_day = d.get("ipr_day", -2)
            self._last_ts = d.get("last_ts", None)
            self._day_max_ts_seen = d.get("day_max_ts_seen", 100_000)
            self._obi_beta_aco = d.get("obi_beta_aco", OBI_BETA_INIT)
            self._obi_samples = d.get("obi_samples", 0)
            self._last_mid_aco = d.get("last_mid_aco", None)
            self._last_obi_aco = d.get("last_obi_aco", 0.0)
            for b, pos in d.get("bot_pos", {}).items():
                for s, v in pos.items():
                    self._bot_pos[b][s] = v
            for pair in d.get("obi_pairs_aco", []):
                self._obi_pairs_aco.append(tuple(pair))
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "ipr_day": self._ipr_day,
            "last_ts": self._last_ts,
            "day_max_ts_seen": self._day_max_ts_seen,
            "obi_beta_aco": round(self._obi_beta_aco, 4),
            "obi_samples": self._obi_samples,
            "last_mid_aco": self._last_mid_aco,
            "last_obi_aco": round(self._last_obi_aco, 4),
            "bot_pos": {b: dict(s) for b, s in self._bot_pos.items()},
            "obi_pairs_aco": list(self._obi_pairs_aco)[-300:],
        })

    def _update_signals(self, market_trades: dict) -> None:
        for symbol, trades in market_trades.items():
            for t in trades:
                for bot in ["Olivia", "Pablo", "Camilla"]:
                    if t.buyer == bot:
                        self._bot_pos[bot][symbol] += t.quantity
                    if t.seller == bot:
                        self._bot_pos[bot][symbol] -= t.quantity
        for sym in POSITION_LIMIT:
            self._bot_pos["Olivia"][sym] *= 0.99
            self._bot_pos["Pablo"][sym] *= 0.95
            self._bot_pos["Camilla"][sym] *= 0.99

    def _update_obi_beta(self) -> None:
        if len(self._obi_pairs_aco) < OBI_WARMUP:
            return
        obis = [p[0] for p in self._obi_pairs_aco]
        diffs = [p[1] for p in self._obi_pairs_aco]
        n = len(obis)
        o_mean = sum(obis) / n
        d_mean = sum(diffs) / n
        cov = sum((o - o_mean) * (d - d_mean) for o, d in zip(obis, diffs))
        var = sum((o - o_mean) ** 2 for o in obis) + 1e-8
        self._obi_beta_aco = max(0.5, min(6.0, cov / var))

    def _ipr_fv(self, timestamp: int) -> float:
        return IPR_BASE + IPR_SLOPE * timestamp + IPR_DAY_STEP * (self._ipr_day + 2)

    def _infer_ipr_day_from_book(self, depth, timestamp: int) -> None:
        if not depth.buy_orders or not depth.sell_orders:
            return
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        raw = (mid - IPR_BASE - IPR_SLOPE * timestamp) / IPR_DAY_STEP - 2
        inferred = int(round(raw))
        inferred = max(-2, min(IPR_MAX_DAY, inferred))
        self._ipr_day = inferred

    def _day_threshold(self, frac: float) -> int:
        base = max(10_000, int(self._day_max_ts_seen))
        return int(base * frac)

    def _ipr_orders(self, depth, pos: int, limit: int, timestamp: int) -> list[Order]:
        """
        ROOT is a drift asset, not a trading asset.
        Best execution here is simple:
        1) get long quickly early in the day,
        2) hold through the session,
        3) flatten near the close.
        """
        orders = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        fv = self._ipr_fv(timestamp)

        eod_start_ts = self._day_threshold(IPR_EOD_START_FRAC)
        eod_hard_ts = self._day_threshold(IPR_EOD_HARD_FRAC)
        open_aggr_ts = self._day_threshold(IPR_OPEN_AGGR_FRAC)

        # Near close: flatten decisively. The edge is in holding the drift,
        # not in trying to save the final tick.
        if pos > 0 and timestamp >= eod_start_ts:
            if best_bid is None:
                return orders
            if timestamp >= eod_hard_ts:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -pos))
            else:
                to_sell = max(1, (pos + 1) // 2)
                px = best_bid if timestamp >= (eod_start_ts + eod_hard_ts) // 2 else best_bid + 1
                orders.append(Order("INTARIAN_PEPPER_ROOT", px, -min(pos, to_sell)))
            return orders

        if pos >= limit or best_ask is None:
            return orders

        buy_cap = limit - pos
        ask_vol = abs(depth.sell_orders.get(best_ask, 0)) if best_ask in depth.sell_orders else buy_cap
        takeable = min(buy_cap, ask_vol if ask_vol > 0 else buy_cap)

        # True aggressive entry: for this product the drift is the edge.
        # Do not wait for impossible "cheap ask" conditions.
        if timestamp <= open_aggr_ts:
            if takeable > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, takeable))
            remaining = buy_cap - takeable
            if remaining > 0 and best_bid is not None:
                # Keep a top-of-book bid to finish the fill quickly.
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid + IPR_PASSIVE_OFFSET, remaining))
            return orders

        # After the opening window, only top up if still not full.
        if buy_cap > 0 and best_bid is not None and timestamp < eod_start_ts:
            # If ask is still reasonably close to fair, cross for the remainder.
            if best_ask <= fv + 8:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, takeable))
                remaining = buy_cap - takeable
                if remaining > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", min(best_bid + 1, int(fv - 1)), remaining))
            else:
                orders.append(Order("INTARIAN_PEPPER_ROOT", min(best_bid + 1, int(fv - 1)), buy_cap))

        return orders

    def _aco_orders(self, depth, pos: int, limit: int, timestamp: int) -> list[Order]:
        orders = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        mid = (best_bid + best_ask) / 2.0

        # OBI update
        if self._last_mid_aco is not None:
            delta_mid = mid - self._last_mid_aco
            self._obi_pairs_aco.append((self._last_obi_aco, delta_mid))
            self._obi_samples += 1
            if self._obi_samples >= OBI_UPDATE_N:
                self._update_obi_beta()
                self._obi_samples = 0

        bid_levels = sorted(depth.buy_orders.items(), reverse=True)
        ask_levels = sorted(depth.sell_orders.items())
        bid_vol1 = sum(v for _, v in bid_levels[:1])
        ask_vol1 = sum(abs(v) for _, v in ask_levels[:1])
        bid_vol2 = bid_levels[1][1] if len(bid_levels) > 1 else 0
        ask_vol2 = abs(ask_levels[1][1]) if len(ask_levels) > 1 else 0
        deep_bid = OBI_W1 * bid_vol1 + OBI_W2 * bid_vol2
        deep_ask = OBI_W1 * ask_vol1 + OBI_W2 * ask_vol2
        deep_obi = (deep_bid - deep_ask) / (deep_bid + deep_ask + 1e-6)
        self._last_obi_aco = deep_obi
        self._last_mid_aco = mid

        bot_alpha = (
            self._bot_pos["Olivia"]["ASH_COATED_OSMIUM"] * 0.6 +
            self._bot_pos["Camilla"]["ASH_COATED_OSMIUM"] * 1.0 -
            self._bot_pos["Pablo"]["ASH_COATED_OSMIUM"] * 0.4
        )
        bot_alpha = max(-1.5, min(1.5, bot_alpha))

        fair = ACO_FAIR + self._obi_beta_aco * deep_obi + bot_alpha

        eod_start_ts = self._day_threshold(ACO_EOD_START_FRAC)
        eod_hard_ts = self._day_threshold(ACO_EOD_HARD_FRAC)

        # End of day flattening
        if timestamp >= eod_start_ts and pos != 0:
            if pos > 0:
                px = best_bid if timestamp >= eod_hard_ts else best_bid + 1
                orders.append(Order("ASH_COATED_OSMIUM", px, -pos))
            else:
                px = best_ask if timestamp >= eod_hard_ts else max(best_bid + 1, best_ask - 1)
                orders.append(Order("ASH_COATED_OSMIUM", px, -pos))
            return orders

        # Inventory-aware discrete quoting.
        bid_shift = 0
        ask_shift = 0
        if pos > ACO_INVENTORY_SOFT:
            bid_shift -= 1
        if pos < -ACO_INVENTORY_SOFT:
            ask_shift += 1

        if deep_obi > 0.35:
            bid_shift += 1
            ask_shift += 1
        elif deep_obi < -0.35:
            bid_shift -= 1
            ask_shift -= 1

        if fair >= best_ask - 1:
            bid_shift += 1
        elif fair <= best_bid + 1:
            ask_shift -= 1

        my_bid = best_bid + 1 + bid_shift
        my_ask = best_ask - 1 + ask_shift

        # stay non-crossing unless hard inventory pressure
        my_bid = min(my_bid, best_ask - 1)
        my_ask = max(my_ask, best_bid + 1)
        if my_bid >= my_ask:
            return orders

        buy_cap = limit - pos
        sell_cap = limit + pos

        if abs(pos) >= ACO_INVENTORY_HARD:
            if pos > 0 and sell_cap > 0:
                orders.append(Order("ASH_COATED_OSMIUM", max(best_bid + 1, my_ask), -min(sell_cap, ACO_BASE_SIZE + 2)))
            elif pos < 0 and buy_cap > 0:
                orders.append(Order("ASH_COATED_OSMIUM", min(best_ask - 1, my_bid), min(buy_cap, ACO_BASE_SIZE + 2)))
            return orders

        skew = abs(pos) / max(1, limit)
        if pos >= 0:
            buy_sz = max(1, int(ACO_BASE_SIZE * (1 - 0.6 * skew)))
            sell_sz = max(1, int(ACO_BASE_SIZE * (1 + 0.8 * skew)))
        else:
            buy_sz = max(1, int(ACO_BASE_SIZE * (1 + 0.8 * skew)))
            sell_sz = max(1, int(ACO_BASE_SIZE * (1 - 0.6 * skew)))

        if buy_cap > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid, min(buy_sz, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask, -min(sell_sz, sell_cap)))
        return orders

    def run(self, state: TradingState):
        self._load(state.traderData)
        self._update_signals(state.market_trades)

        ts = state.timestamp

        # Detect new day by timestamp reset.
        if self._last_ts is not None and ts < self._last_ts:
            self._ipr_day += 1
            self._day_max_ts_seen = max(self._day_max_ts_seen, self._last_ts)
        self._last_ts = ts
        self._day_max_ts_seen = max(self._day_max_ts_seen, ts)

        # Prefer direct inference from current ROOT book each tick.
        ipr_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if ipr_depth is not None:
            self._infer_ipr_day_from_book(ipr_depth, ts)

        result = {}
        for symbol, depth in state.order_depths.items():
            pos = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 20)

            if symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._ipr_orders(depth, pos, limit, ts)
            elif symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._aco_orders(depth, pos, limit, ts)

        return result, 0, self._save()
