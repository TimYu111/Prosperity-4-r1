import json
from collections import defaultdict, deque
from datamodel import Order, TradingState

# ============================================================
#  基础配置
# ============================================================
POSITION_LIMIT = {"EMERALDS": 20, "TOMATOES": 20}
EMERALD_FV     = 10000

# ============================================================
#  EMERALDS 参数（Markov 模型，数据解码完成）
# ============================================================
# 静态 bot：9992/10008，跳变到 10000 后 97% 概率下一 tick 回撤
# 跳变时完全忽略 reservation price，全力买卖

# ============================================================
#  TOMATOES 做市参数
# ============================================================
GAMMA         = 0.2
QUOTE_SIZE    = 5
INVENTORY_ALERT = 0.70
MIN_SPREAD    = 2

# OBI beta（在线估计，初值来自历史数据）
OBI_BETA_INIT = 2.0
OBI_W1, OBI_W2 = 0.4, 0.6

# 在线估计窗口：积累多少对 (OBI_t, Δmid_{t+1}) 后开始更新
OBI_WARMUP   = 100    # 对数
OBI_UPDATE_N = 50     # 每积累多少新样本更新一次 beta

# Drift 检测参数
DRIFT_WARMUP     = 300   # 至少多少 tick 后才开始用 drift 信号（约 30000 ts）
DRIFT_R2_THRESH  = 0.30  # R² 阈值：超过才认为趋势显著
DRIFT_WINDOW     = 1000  # OLS 使用最近多少 tick

# 日末平仓：最后 5% 的 tick 开始强制减仓
# 一天 2000 tick → 最后 100 tick → timestamp > 190000
EOD_START_TS = 190000

# ============================================================
#  spread=5 信号（数据验证：E[Δmid_{t+1}] = +4.0）
# ============================================================
NARROW_SPREAD_THRESHOLD = 6   # spread ≤ 这个值触发主动吃单
NARROW_SPREAD_AGGRESSIVE = True  # 直接主动 market order

class Trader:

    def __init__(self):
        # ── EMERALDS Markov ──────────────────────────────────
        self._em_bid_state = "normal"   # "normal" | "up"
        self._em_ask_state = "normal"   # "normal" | "down"

        # ── TOMATOES 在线学习 ────────────────────────────────
        # OBI beta 在线估计
        self._obi_pairs   = deque(maxlen=2000)   # (obi_t, delta_mid_t+1)
        self._last_obi_tom  = 0.0
        self._last_mid_tom  = None
        self._obi_beta      = OBI_BETA_INIT
        self._obi_samples_since_update = 0

        # Drift 检测
        self._mid_ts_tom  = deque(maxlen=DRIFT_WINDOW)  # (timestamp, mid)
        self._drift_slope = 0.0
        self._drift_sig   = False  # 是否显著

        # Tick 计数（用于 EOD 检测）
        self._tick_count = 0

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
            self._em_bid_state  = d.get("em_bid", "normal")
            self._em_ask_state  = d.get("em_ask", "normal")
            self._obi_beta      = d.get("obi_beta", OBI_BETA_INIT)
            self._drift_slope   = d.get("drift_slope", 0.0)
            self._drift_sig     = d.get("drift_sig", False)
            self._tick_count    = d.get("tick_count", 0)
            for b, pos in d.get("bot_pos", {}).items():
                for s, v in pos.items():
                    self._bot_pos[b][s] = v
            for pair in d.get("obi_pairs", []):
                self._obi_pairs.append(tuple(pair))
            for item in d.get("mid_ts_tom", []):
                self._mid_ts_tom.append(tuple(item))
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "em_bid":      self._em_bid_state,
            "em_ask":      self._em_ask_state,
            "obi_beta":    round(self._obi_beta, 4),
            "drift_slope": round(self._drift_slope, 6),
            "drift_sig":   self._drift_sig,
            "tick_count":  self._tick_count,
            "bot_pos":     {b: dict(s) for b, s in self._bot_pos.items()},
            "obi_pairs":   list(self._obi_pairs)[-500:],   # 只存最近 500 对
            "mid_ts_tom":  list(self._mid_ts_tom)[-500:],
        })

    # ----------------------------------------------------------
    #  Bot 信号
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
    #  在线 OBI beta 估计（最小二乘）
    # ----------------------------------------------------------
    def _update_obi_beta(self) -> None:
        if len(self._obi_pairs) < OBI_WARMUP:
            return
        obis  = [p[0] for p in self._obi_pairs]
        diffs = [p[1] for p in self._obi_pairs]
        n = len(obis)
        o_mean = sum(obis)  / n
        d_mean = sum(diffs) / n
        cov = sum((o - o_mean) * (d - d_mean) for o, d in zip(obis, diffs))
        var = sum((o - o_mean) ** 2 for o in obis) + 1e-8
        beta = cov / var
        self._obi_beta = max(0.5, min(6.0, beta))   # 限幅防止极端值

    # ----------------------------------------------------------
    #  Drift 检测（OLS on mid price levels）
    #
    #  对最近 DRIFT_WINDOW 个 (timestamp, mid) 做线性回归
    #  slope 单位：pts/tick
    #  R² > DRIFT_R2_THRESH 且样本足够 → 认为趋势显著
    # ----------------------------------------------------------
    def _update_drift(self) -> None:
        data = list(self._mid_ts_tom)
        n    = len(data)
        if n < DRIFT_WARMUP:
            self._drift_sig = False
            return

        # 标准化 timestamp（以 tick 为单位，第一个=0）
        t0   = data[0][0]
        ts_  = [(d[0] - t0) / 100 for d in data]   # 每 100ms 一 tick
        mids = [d[1] for d in data]

        t_mean = sum(ts_)  / n
        m_mean = sum(mids) / n
        stt = sum((t - t_mean) ** 2 for t in ts_) + 1e-10
        stm = sum((t - t_mean) * (m - m_mean) for t, m in zip(ts_, mids))

        b = stm / stt                # slope: pts/tick

        # R²
        ss_tot = sum((m - m_mean) ** 2 for m in mids) + 1e-10
        ss_res = sum((m - (m_mean + b * (t - t_mean))) ** 2
                     for t, m in zip(ts_, mids))
        r2 = 1.0 - ss_res / ss_tot

        self._drift_slope = b
        self._drift_sig   = (r2 > DRIFT_R2_THRESH) and (n >= DRIFT_WARMUP)

    # ----------------------------------------------------------
    #  EMERALDS：Markov 状态机 + 最优吃单
    #
    #  优化点 1：Markov 跳变时完全忽略 reservation price，全力操作
    #  优化点 2：跳变时先清仓再反向建仓（同一 tick 一起提交）
    # ----------------------------------------------------------
    def _emerald_orders(self, depth, pos: int, limit: int,
                        timestamp: int) -> list[Order]:
        orders = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # 更新 Markov 状态
        self._em_bid_state = "up"   if best_bid == 10000 else "normal"
        self._em_ask_state = "down" if best_ask == 10000 else "normal"

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # ── EOD 平仓 ────────────────────────────────────────
        if timestamp >= EOD_START_TS:
            return self._eod_unwind("EMERALDS", depth, pos, limit)

        # ── Markov exploit: ask 跳到 10000 ─────────────────
        # 这是罕见的价格压低机会（正常 10008），97% 下一 tick 回撤
        # 策略：先卖出所有多头（如果有），再以 10000 买入到 limit
        if self._em_ask_state == "down":
            # 1. 如果持有多头，先以 best_bid 卖出（只能等下一 tick，所以跳过）
            #    实际做法：直接以 10000 全力买入（不受 reservation 约束）
            if buy_cap > 0:
                # 吃掉 10000 的 ask 上所有量（主动成交）
                ask_vol = abs(depth.sell_orders.get(10000, 0))
                vol = min(buy_cap, ask_vol if ask_vol > 0 else limit)
                if vol > 0:
                    orders.append(Order("EMERALDS", 10000, vol))
            return orders   # 本 tick 只做这一件事

        # ── Markov exploit: bid 跳到 10000 ─────────────────
        # 正常 bid=9992，现在 10000，97% 下一 tick 回到 9992
        # 以 10000 卖出，等下一 tick bid 回到 9992 可再买回
        if self._em_bid_state == "up":
            if sell_cap > 0:
                bid_vol = depth.buy_orders.get(10000, 0)
                vol = min(sell_cap, bid_vol if bid_vol > 0 else limit)
                if vol > 0:
                    orders.append(Order("EMERALDS", 10000, -vol))
            return orders

        # ── 正常状态：被动夹单 9993/10007 ───────────────────
        if buy_cap > 0:
            orders.append(Order("EMERALDS", best_bid + 1,  min(5, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", best_ask - 1, -min(5, sell_cap)))

        return orders

    # ----------------------------------------------------------
    #  TOMATOES：OBI + Drift + spread=5 exploit + 日末平仓
    # ----------------------------------------------------------
    def _tomato_orders(self, depth, pos: int, limit: int,
                       timestamp: int) -> list[Order]:
        orders = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread   = best_ask - best_bid
        mid      = (best_bid + best_ask) / 2.0

        # ── 记录 mid 历史（用于 drift 检测）───────────────────
        self._mid_ts_tom.append((timestamp, mid))

        # ── 更新 OBI 配对 ──────────────────────────────────
        if self._last_mid_tom is not None:
            delta_mid = mid - self._last_mid_tom
            self._obi_pairs.append((self._last_obi_tom, delta_mid))
            self._obi_samples_since_update += 1
            if self._obi_samples_since_update >= OBI_UPDATE_N:
                self._update_obi_beta()
                self._obi_samples_since_update = 0

        # ── 每 100 tick 更新 drift ─────────────────────────
        if self._tick_count % 100 == 0:
            self._update_drift()

        # ── 计算 deep OBI ──────────────────────────────────
        bid_vol1 = sum(v for v in depth.buy_orders.values())
        ask_vol1 = sum(abs(v) for v in depth.sell_orders.values())
        sorted_bids = sorted(depth.buy_orders.items(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.items())
        bid_vol2 = sorted_bids[1][1] if len(sorted_bids) > 1 else 0
        ask_vol2 = abs(sorted_asks[1][1]) if len(sorted_asks) > 1 else 0

        deep_bid = OBI_W1 * bid_vol1 + OBI_W2 * bid_vol2
        deep_ask = OBI_W1 * ask_vol1 + OBI_W2 * ask_vol2
        deep_obi = (deep_bid - deep_ask) / (deep_bid + deep_ask + 1e-6)

        # 记录本 tick 的 OBI，供下 tick 配对用
        self._last_obi_tom = deep_obi
        self._last_mid_tom = mid

        # Bot alpha
        alpha = (
            self._bot_pos["Olivia" ]["TOMATOES"] * 0.8 +
            self._bot_pos["Camilla"]["TOMATOES"] * 1.2 -
            self._bot_pos["Pablo"  ]["TOMATOES"] * 0.5
        )
        alpha = max(-2.0, min(2.0, alpha))

        # FV = mid + OBI 预测 + alpha
        obi_signal = deep_obi * self._obi_beta
        fv = mid + obi_signal + alpha

        # ── EOD 平仓 ────────────────────────────────────────
        if timestamp >= EOD_START_TS:
            return self._eod_unwind("TOMATOES", depth, pos, limit)

        # ── 优化 1：spread 极窄时主动吃单（强方向信号）──────
        # 数据验证：spread=5 → E[Δmid_{t+1}] = +4.0（极强信号）
        # spread=6/7 → E[Δmid] ≈ +2.1（中等信号）
        if NARROW_SPREAD_AGGRESSIVE and spread <= NARROW_SPREAD_THRESHOLD:
            # OBI 辅助确认方向
            if spread <= 5 or (spread <= 7 and deep_obi > 0.1):
                # 强买信号：主动吃 ask
                if pos < limit:
                    vol = min(QUOTE_SIZE, limit - pos)
                    orders.append(Order("TOMATOES", best_ask, vol))
                return orders   # 本 tick 只做这一件事
            elif spread <= 5:
                # spread=5 但 OBI 偏空：不买，什么都不做
                return orders

        # ── 优化 2：日内 drift 显著时切换方向模式 ─────────
        # drift_slope < 0 → 下跌趋势 → 只挂卖单 / 不买
        # drift_slope > 0 → 上涨趋势 → 只挂买单 / 不卖
        if self._drift_sig and self._tick_count >= DRIFT_WARMUP:
            # 方向模式：跟随趋势，只做一侧
            slope = self._drift_slope
            if slope < -0.01:   # 明确下跌
                # 只挂卖单，并考虑主动做空
                if pos > -limit:
                    # 被动卖 ask-1
                    if spread > MIN_SPREAD:
                        orders.append(Order("TOMATOES", best_ask - 1,
                                            -min(QUOTE_SIZE, limit + pos)))
                    # OBI 确认下跌时主动吃 bid
                    if deep_obi < -0.3 and pos > -limit:
                        orders.append(Order("TOMATOES", best_bid,
                                            -min(3, limit + pos)))
                return orders
            elif slope > 0.01:  # 明确上涨
                if pos < limit:
                    if spread > MIN_SPREAD:
                        orders.append(Order("TOMATOES", best_bid + 1,
                                            min(QUOTE_SIZE, limit - pos)))
                    if deep_obi > 0.3 and pos < limit:
                        orders.append(Order("TOMATOES", best_ask,
                                            min(3, limit - pos)))
                return orders
            # slope ≈ 0：不显著，走普通做市

        # ── 优化 3：OBI 强信号时主动吃单 ──────────────────
        # corr(deep_OBI, Δmid) = 0.59，信号极强时直接 market order
        if abs(deep_obi) > 0.6 and spread <= 10:
            if deep_obi > 0.6 and pos < limit:
                orders.append(Order("TOMATOES", best_ask, min(3, limit - pos)))
            elif deep_obi < -0.6 and pos > -limit:
                orders.append(Order("TOMATOES", best_bid, -min(3, limit + pos)))

        # ── 普通被动夹单做市 ────────────────────────────────
        if spread <= MIN_SPREAD:
            return orders

        my_bid = best_bid + 1
        my_ask = best_ask - 1
        if my_bid >= my_ask:
            return orders

        # 库存警戒
        if pos >= limit * INVENTORY_ALERT:
            reduce = int(pos - limit * 0.5)
            if reduce > 0:
                orders.append(Order("TOMATOES", best_bid, -reduce))
            return orders
        if pos <= -limit * INVENTORY_ALERT:
            reduce = int(-pos - limit * 0.5)
            if reduce > 0:
                orders.append(Order("TOMATOES", best_ask, reduce))
            return orders

        # Reservation price + 非对称仓位
        reservation = fv - pos * GAMMA
        want_buy    = (my_bid <= reservation + GAMMA * 0.5)
        want_sell   = (my_ask >= reservation - GAMMA * 0.5)

        skew = abs(pos) / limit
        if pos >= 0:
            buy_sz, sell_sz = max(1, int(QUOTE_SIZE*(1-skew))), max(1, int(QUOTE_SIZE*(1+skew)))
        else:
            buy_sz, sell_sz = max(1, int(QUOTE_SIZE*(1+skew))), max(1, int(QUOTE_SIZE*(1-skew)))

        buy_cap  = limit - pos
        sell_cap = limit + pos

        if want_buy  and buy_cap  > 0:
            orders.append(Order("TOMATOES", my_bid,  min(buy_sz,  buy_cap)))
        if want_sell and sell_cap > 0:
            orders.append(Order("TOMATOES", my_ask, -min(sell_sz, sell_cap)))

        return orders

    # ----------------------------------------------------------
    #  日末强制平仓（最后 5% tick）
    #
    #  策略：越接近收盘越激进，最后 50 tick 直接市价清仓
    # ----------------------------------------------------------
    def _eod_unwind(self, symbol: str, depth, pos: int,
                    limit: int) -> list[Order]:
        orders = []
        if pos == 0:
            return orders

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        if pos > 0 and best_bid is not None:
            orders.append(Order(symbol, best_bid, -pos))  # 市价卖出
        elif pos < 0 and best_ask is not None:
            orders.append(Order(symbol, best_ask, -pos))  # 市价买入

        return orders

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------
    def run(self, state: TradingState):
        self._load(state.traderData)
        self._update_signals(state.market_trades)
        self._tick_count += 1

        ts = state.timestamp

        result = {}
        for symbol, depth in state.order_depths.items():
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT[symbol]

            if symbol == "EMERALDS":
                result[symbol] = self._emerald_orders(depth, pos, limit, ts)

            elif symbol == "TOMATOES":
                result[symbol] = self._tomato_orders(depth, pos, limit, ts)

        return result, ts, self._save()
