import json
from collections import defaultdict, deque
from datamodel import Order, TradingState

# ============================================================
#  基础配置
# ============================================================
POSITION_LIMIT = {
    "INTARIAN_PEPPER_ROOT": 20,
    "ASH_COATED_OSMIUM":    20,
}

# ============================================================
#  INTARIAN_PEPPER_ROOT (IPR) 参数
#  模型解码：FV = 9998.5 + 0.001×ts + 1000×(day+2)
#  三天 R²=1.0000，残差 std=2.0
#  策略：开局立刻满仓买入，持有全天，日末卖出
# ============================================================
IPR_SLOPE      = 0.001      # pts/ts，精确值
IPR_DAY_STEP   = 1000       # 每天涨多少
IPR_BASE       = 9998.5     # day=-2, ts=0 时的 FV
IPR_EOD_TS     = 950000     # 日末开始被动卖出（最后 500 tick，ts 0→999900）
IPR_ENTRY_MARGIN = 8        # 只要 ask < FV + margin 就买（正常 ask 高 FV 约 7.5）

# ============================================================
#  ASH_COATED_OSMIUM (ACO) 参数
#  模型解码：OU 过程，FV=10000，theta=0.24，spread 主导=16
#  OBI 相关性 0.65，用于调整 reservation price
#  策略：被动夹单做市，OBI 调整偏向，EOD 平仓
# ============================================================
ACO_FV         = 10000.0
ACO_GAMMA      = 0.15       # 库存风险厌恶
ACO_QUOTE_SIZE = 5
ACO_INVENTORY_ALERT = 0.70
ACO_MIN_SPREAD = 2

# OBI 参数（在线估计）
OBI_BETA_INIT  = 2.5        # ACO OBI 相关性 0.65，比 TOMATOES 强，初值略高
OBI_W1, OBI_W2 = 0.4, 0.6
OBI_WARMUP     = 100        # 对数
OBI_UPDATE_N   = 50

ACO_EOD_TS     = 950000     # 日末平仓时间戳（ts 0→999900）


class Trader:

    def __init__(self):
        # ── IPR 状态 ─────────────────────────────────────────
        self._ipr_day        = -2     # 当前天（从 traderData 读）
        self._ipr_entered    = False  # 是否已建仓

        # ── ACO 在线学习 ─────────────────────────────────────
        self._obi_pairs_aco  = deque(maxlen=2000)
        self._last_obi_aco   = 0.0
        self._last_mid_aco   = None
        self._obi_beta_aco   = OBI_BETA_INIT
        self._obi_samples    = 0

        # ── Bot 信号 ─────────────────────────────────────────
        self._bot_pos = defaultdict(lambda: defaultdict(float))

    # ----------------------------------------------------------
    #  持久化
    # ----------------------------------------------------------
    def _load(self, data: str) -> None:
        if not data:
            return
        try:
            d = json.loads(data)
            self._ipr_day      = d.get("ipr_day", -2)
            self._ipr_entered  = d.get("ipr_entered", False)
            self._obi_beta_aco = d.get("obi_beta_aco", OBI_BETA_INIT)
            self._obi_samples  = d.get("obi_samples", 0)
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
            "ipr_day":      self._ipr_day,
            "ipr_entered":  self._ipr_entered,
            "obi_beta_aco": round(self._obi_beta_aco, 4),
            "obi_samples":  self._obi_samples,
            "last_mid_aco": self._last_mid_aco,
            "last_obi_aco": round(self._last_obi_aco, 4),
            "bot_pos":      {b: dict(s) for b, s in self._bot_pos.items()},
            "obi_pairs_aco": list(self._obi_pairs_aco)[-300:],
        })

    # ----------------------------------------------------------
    #  Bot 信号追踪
    # ----------------------------------------------------------
    def _update_signals(self, market_trades: dict) -> None:
        for symbol, trades in market_trades.items():
            for t in trades:
                for bot in ["Olivia", "Pablo", "Camilla"]:
                    if t.buyer  == bot: self._bot_pos[bot][symbol] += t.quantity
                    if t.seller == bot: self._bot_pos[bot][symbol] -= t.quantity
        for sym in POSITION_LIMIT:
            self._bot_pos["Olivia" ][sym] *= 0.99
            self._bot_pos["Pablo"  ][sym] *= 0.95
            self._bot_pos["Camilla"][sym] *= 0.99

    # ----------------------------------------------------------
    #  在线 OBI beta 估计
    # ----------------------------------------------------------
    def _update_obi_beta(self) -> None:
        if len(self._obi_pairs_aco) < OBI_WARMUP:
            return
        obis  = [p[0] for p in self._obi_pairs_aco]
        diffs = [p[1] for p in self._obi_pairs_aco]
        n = len(obis)
        o_mean = sum(obis)  / n
        d_mean = sum(diffs) / n
        cov = sum((o - o_mean) * (d - d_mean) for o, d in zip(obis, diffs))
        var = sum((o - o_mean) ** 2 for o in obis) + 1e-8
        self._obi_beta_aco = max(0.5, min(8.0, cov / var))

    # ----------------------------------------------------------
    #  IPR 公允价计算
    #  FV(ts, day) = 9998.5 + 0.001×ts + 1000×(day+2)
    # ----------------------------------------------------------
    def _ipr_fv(self, timestamp: int) -> float:
        return IPR_BASE + IPR_SLOPE * timestamp + IPR_DAY_STEP * (self._ipr_day + 2)

    # ----------------------------------------------------------
    #  INTARIAN_PEPPER_ROOT 策略
    #
    #  核心逻辑：
    #  1. 每天开始（ts 很小且未建仓）→ 以 ask 价主动买入满仓
    #     只要 ask < FV + margin（正常 ask 高于 FV 约 7.5，margin=8 基本总满足）
    #  2. 持有全天（不做任何被动做市，防止被卖出）
    #  3. EOD（ts > 195000）→ 以 bid 价被动卖出全部仓位
    #     最后 50 tick（ts > 199500）才用市价确保清仓
    #
    #  理论日 PnL = 20 × (1000 − 7.5 − 4.5) ≈ 19600
    # ----------------------------------------------------------
    def _ipr_orders(self, depth, pos: int, limit: int,
                    timestamp: int) -> list[Order]:
        orders = []

        if not depth.buy_orders and not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        fv = self._ipr_fv(timestamp)

        # ── EOD 平仓 ─────────────────────────────────────────
        if timestamp >= IPR_EOD_TS and pos > 0:
            if timestamp >= 995000:
                # 最后 50 tick：市价卖出
                if best_bid:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -pos))
            else:
                # 被动挂单：bid + 1（FV 还在涨，不急于踩踏）
                if best_bid:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid + 1, -pos))
            return orders

        # ── 建仓：每天开局尽快买满 ───────────────────────────
        if pos < limit:
            buy_cap = limit - pos
            if best_ask and best_ask < fv + IPR_ENTRY_MARGIN:
                # ask 合理（低于 FV + 8），主动买入
                ask_vol = abs(depth.sell_orders.get(best_ask, 0))
                vol = min(buy_cap, ask_vol if ask_vol > 0 else buy_cap)
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, vol))

                # 同时挂被动 bid+1 补充（如果还没满仓）
                remaining = buy_cap - vol
                if remaining > 0 and best_bid:
                    orders.append(Order("INTARIAN_PEPPER_ROOT",
                                        int(round(fv - 4)), remaining))
            elif best_bid:
                # ask 偏贵，只挂被动买单在 FV 附近
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                                    int(round(fv - 4)), buy_cap))

        return orders

    # ----------------------------------------------------------
    #  ASH_COATED_OSMIUM 策略
    #
    #  核心逻辑（完全复用修复后的 TOMATOES 框架）：
    #  1. OBI 只影响 FV → reservation_price → 被动报价偏向
    #     不做主动吃单（8pts crossing cost > OBI 预期收益 ~7pts）
    #  2. OU 均值回归：theta=0.24，但 crossing_cost=8，
    #     需要偏离 33pts 才回本，实际从未发生 → 不做 OU 主动吃单
    #  3. 被动夹单：bid+1 / ask-1，spread=16 → 利润=14/轮
    #  4. 库存警戒/EOD 用被动单不用市价
    # ----------------------------------------------------------
    def _aco_orders(self, depth, pos: int, limit: int,
                    timestamp: int) -> list[Order]:
        orders = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread   = best_ask - best_bid
        mid      = (best_bid + best_ask) / 2.0

        # ── 更新 OBI 配对 ──────────────────────────────────────
        if self._last_mid_aco is not None:
            delta_mid = mid - self._last_mid_aco
            self._obi_pairs_aco.append((self._last_obi_aco, delta_mid))
            self._obi_samples += 1
            if self._obi_samples >= OBI_UPDATE_N:
                self._update_obi_beta()
                self._obi_samples = 0

        # ── 计算 deep OBI ──────────────────────────────────────
        bid_vol1 = sum(v for v in depth.buy_orders.values())
        ask_vol1 = sum(abs(v) for v in depth.sell_orders.values())
        sorted_bids = sorted(depth.buy_orders.items(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.items())
        bid_vol2 = sorted_bids[1][1] if len(sorted_bids) > 1 else 0
        ask_vol2 = abs(sorted_asks[1][1]) if len(sorted_asks) > 1 else 0

        deep_bid = OBI_W1 * bid_vol1 + OBI_W2 * bid_vol2
        deep_ask = OBI_W1 * ask_vol1 + OBI_W2 * ask_vol2
        deep_obi = (deep_bid - deep_ask) / (deep_bid + deep_ask + 1e-6)

        self._last_obi_aco = deep_obi
        self._last_mid_aco = mid

        # Bot alpha
        alpha = (
            self._bot_pos["Olivia" ]["ASH_COATED_OSMIUM"] * 0.8 +
            self._bot_pos["Camilla"]["ASH_COATED_OSMIUM"] * 1.2 -
            self._bot_pos["Pablo"  ]["ASH_COATED_OSMIUM"] * 0.5
        )
        alpha = max(-2.0, min(2.0, alpha))

        # FV = ACO_FV(10000) + OBI 信号 + alpha
        # OBI 只影响 FV，不触发市价单
        obi_signal = deep_obi * self._obi_beta_aco
        fv = ACO_FV + obi_signal + alpha

        # ── EOD 被动平仓 ────────────────────────────────────────
        if timestamp >= ACO_EOD_TS and pos != 0:
            urgent = timestamp >= 999500
            if pos > 0:
                price = best_bid if urgent else best_bid + 1
                orders.append(Order("ASH_COATED_OSMIUM", price, -pos))
            else:
                price = best_ask if urgent else best_ask - 1
                orders.append(Order("ASH_COATED_OSMIUM", price, -pos))
            return orders

        # ── spread 太窄，放弃 ──────────────────────────────────
        if spread <= ACO_MIN_SPREAD:
            return orders

        my_bid = best_bid + 1
        my_ask = best_ask - 1
        if my_bid >= my_ask:
            return orders

        # ── 库存警戒：被动单（不用市价） ─────────────────────
        if pos >= limit * ACO_INVENTORY_ALERT:
            reduce = int(pos - limit * 0.5)
            if reduce > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_bid + 1, -reduce))
            return orders
        if pos <= -limit * ACO_INVENTORY_ALERT:
            reduce = int(-pos - limit * 0.5)
            if reduce > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_ask - 1, reduce))
            return orders

        # ── Reservation price + 非对称仓位 ─────────────────────
        reservation = fv - pos * ACO_GAMMA
        want_buy    = (my_bid <= reservation + ACO_GAMMA * 0.5)
        want_sell   = (my_ask >= reservation - ACO_GAMMA * 0.5)

        skew = abs(pos) / limit
        if pos >= 0:
            buy_sz  = max(1, int(ACO_QUOTE_SIZE * (1 - skew)))
            sell_sz = max(1, int(ACO_QUOTE_SIZE * (1 + skew)))
        else:
            buy_sz  = max(1, int(ACO_QUOTE_SIZE * (1 + skew)))
            sell_sz = max(1, int(ACO_QUOTE_SIZE * (1 - skew)))

        buy_cap  = limit - pos
        sell_cap = limit + pos

        if want_buy  and buy_cap  > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid,  min(buy_sz,  buy_cap)))
        if want_sell and sell_cap > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask, -min(sell_sz, sell_cap)))

        return orders

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------
    def run(self, state: TradingState):
        self._load(state.traderData)
        self._update_signals(state.market_trades)

        # 从 timestamp 推断当前 day（跨天时 timestamp 重置为 0）
        # TradingState 有 state.timestamp，但 day 需要从外部传入
        # 这里用 traderData 中存储的 day 值，由外部环境在每天开始时重置
        # 如果环境提供 state.traderData 含 day 信息可直接读取

        ts = state.timestamp

        result = {}
        for symbol, depth in state.order_depths.items():
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 20)

            if symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._ipr_orders(depth, pos, limit, ts)

            elif symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._aco_orders(depth, pos, limit, ts)

        return result, ts, self._save()
