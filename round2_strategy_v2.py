import json
from collections import defaultdict, deque
from datamodel import Order, TradingState

# ============================================================
#  Round 2 变化摘要
#  - Position limit: 20 → 80
#  - ACO_JUMP_BOOST 关闭（R2 ±8跳后无回撤，P=35.8%，不可用）
#  - IPR 被动入场窗口删除（静态bot不会主动填 bid+1，浪费380pts）
#  - 新增 bid() → MAF = 1500（downside=0，赢得额外25%成交量）
# ============================================================

POSITION_LIMIT = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# ============================================================
#  INTARIAN_PEPPER_ROOT
#  FV(ts, day) = 9998.5 + 0.001*ts + 1000*(day+2)  R²=1.0000
#  策略：开盘立刻积极建满仓，持有，97.5% 进度被动 EOD
# ============================================================
IPR_BASE              = 9998.5
IPR_SLOPE             = 0.001
IPR_DAY_STEP          = 1000
IPR_MAX_DAY           = 20

IPR_OPEN_AGGR_FRAC    = 0.20    # 前20%：积极+被动双轨
IPR_EOD_START_FRAC    = 0.975   # 97.5%进度开始EOD
IPR_EOD_HARD_FRAC     = 0.995   # 99.5%进度市价清仓
IPR_EOD_PASSIVE_OFFSET = 4.0   # EOD被动挂单偏移（FV - 4）

# ============================================================
#  ASH_COATED_OSMIUM
#  OU过程，FV=10000，theta≈0.26，spread=16，OBI相关=0.65
#  R2变化：±8跳变后无回撤（P=35.8%），关闭JUMP_BOOST
# ============================================================
ACO_FAIR              = 10000
ACO_BASE_SIZE         = 8      # 匹配市场深度（均值14手），原20手报量超出导致成交率降低
ACO_INVENTORY_SOFT    = 20     # 调低至实际持仓范围（原40从未触发）
ACO_INVENTORY_HARD    = 40     # 调低（原64从未触发）
ACO_EOD_START_FRAC    = 0.985
ACO_EOD_HARD_FRAC     = 0.997
ACO_PRE_EOD_BIAS_FRAC = 0.965

ACO_STRONG_SIGNAL     = 0.90
ACO_EXTREME_SIGNAL    = 1.40

OBI_BETA_INIT         = 2.0
OBI_W1, OBI_W2        = 0.4, 0.6
OBI_WARMUP            = 100
OBI_UPDATE_N          = 50

# R2: ±8跳变后无均值回归（P=35.8% < 50%），关闭
ACO_JUMP_THRESHOLD    = 8.0
ACO_JUMP_BOOST        = 0      # ← R2 关闭（R1=2，R2跳后随机方向）

# OBI极端单侧报价：|OBI|>0.6 时停止反向报价
ACO_OBI_ONESIDED      = 0.60


class Trader:

    def bid(self):
        """
        MAF 竞标：赢得额外25%成交量。
        top 50%赢 → 支付此费用。输了不扣钱。
        额外25%量 → ACO额外PnL ≈ 2640，出价1500净赚≈1140。
        """
        return 1500

    def __init__(self):
        # IPR
        self._ipr_day         = -2
        self._last_ts         = None
        self._day_max_ts_seen = 100_000

        # ACO
        self._obi_pairs_aco   = deque(maxlen=2000)
        self._last_obi_aco    = 0.0
        self._last_mid_aco    = None
        self._last_delta_aco  = 0.0
        self._obi_beta_aco    = OBI_BETA_INIT
        self._obi_samples     = 0

        self._bot_pos = defaultdict(lambda: defaultdict(float))

    # ----------------------------------------------------------
    #  持久化
    # ----------------------------------------------------------
    def _load(self, data: str) -> None:
        if not data:
            return
        try:
            d = json.loads(data)
            self._ipr_day         = d.get("ipr_day", -2)
            self._last_ts         = d.get("last_ts")
            self._day_max_ts_seen = d.get("day_max_ts_seen", 100_000)
            self._obi_beta_aco    = d.get("obi_beta_aco", OBI_BETA_INIT)
            self._obi_samples     = d.get("obi_samples", 0)
            self._last_mid_aco    = d.get("last_mid_aco")
            self._last_obi_aco    = d.get("last_obi_aco", 0.0)
            self._last_delta_aco  = d.get("last_delta_aco", 0.0)
            for b, pos in d.get("bot_pos", {}).items():
                for s, v in pos.items():
                    self._bot_pos[b][s] = v
            for pair in d.get("obi_pairs_aco", []):
                self._obi_pairs_aco.append(tuple(pair))
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "ipr_day":         self._ipr_day,
            "last_ts":         self._last_ts,
            "day_max_ts_seen": self._day_max_ts_seen,
            "obi_beta_aco":    round(self._obi_beta_aco, 4),
            "obi_samples":     self._obi_samples,
            "last_mid_aco":    self._last_mid_aco,
            "last_obi_aco":    round(self._last_obi_aco, 4),
            "last_delta_aco":  round(self._last_delta_aco, 4),
            "bot_pos":         {b: dict(s) for b, s in self._bot_pos.items()},
            "obi_pairs_aco":   list(self._obi_pairs_aco)[-300:],
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
    #  OBI beta 在线估计
    # ----------------------------------------------------------
    def _update_obi_beta(self) -> None:
        if len(self._obi_pairs_aco) < OBI_WARMUP:
            return
        obis  = [p[0] for p in self._obi_pairs_aco]
        diffs = [p[1] for p in self._obi_pairs_aco]
        n = len(obis)
        om = sum(obis) / n
        dm = sum(diffs) / n
        cov = sum((o-om)*(d-dm) for o, d in zip(obis, diffs))
        var = sum((o-om)**2 for o in obis) + 1e-8
        self._obi_beta_aco = max(0.5, min(6.0, cov / var))

    # ----------------------------------------------------------
    #  IPR 工具
    # ----------------------------------------------------------
    def _ipr_fv(self, timestamp: int) -> float:
        return IPR_BASE + IPR_SLOPE * timestamp + IPR_DAY_STEP * (self._ipr_day + 2)

    def _infer_ipr_day(self, depth, timestamp: int) -> None:
        if not depth.buy_orders or not depth.sell_orders:
            return
        bid = max(depth.buy_orders.keys())
        ask = min(depth.sell_orders.keys())
        mid = (bid + ask) / 2.0
        raw = (mid - IPR_BASE - IPR_SLOPE * timestamp) / IPR_DAY_STEP - 2
        self._ipr_day = max(-2, min(IPR_MAX_DAY, int(round(raw))))

    def _day_threshold(self, frac: float) -> int:
        return int(max(10_000, self._day_max_ts_seen) * frac)

    # ----------------------------------------------------------
    #  INTARIAN_PEPPER_ROOT
    #
    #  R2 修复：删掉被动入场窗口（静态bot不填 bid+1，浪费380pts）
    #  直接从 ts=0 开始积极买入，前20%进度积极+被动双轨确保建仓
    # ----------------------------------------------------------
    def _ipr_orders(self, depth, pos: int, limit: int,
                    timestamp: int) -> list:
        orders = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_bid is None and best_ask is None:
            return orders

        fv            = self._ipr_fv(timestamp)
        eod_start_ts  = self._day_threshold(IPR_EOD_START_FRAC)
        eod_hard_ts   = self._day_threshold(IPR_EOD_HARD_FRAC)
        open_aggr_ts  = self._day_threshold(IPR_OPEN_AGGR_FRAC)

        # ── EOD 平仓 ─────────────────────────────────────────
        if pos > 0 and timestamp >= eod_start_ts:
            if best_bid is None:
                return orders
            if timestamp >= eod_hard_ts:
                # 最后 0.5%：市价清仓
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -pos))
            else:
                # 被动挂 bid+1（FV-4 通常高于 bid，无人成交会拖到市价）
                target_px = best_bid + 1
                orders.append(Order("INTARIAN_PEPPER_ROOT", target_px, -pos))
            return orders

        if pos >= limit or best_ask is None:
            return orders

        buy_cap  = limit - pos
        ask_vol  = abs(depth.sell_orders.get(best_ask, 0))
        takeable = min(buy_cap, ask_vol if ask_vol > 0 else buy_cap)

        if timestamp <= open_aggr_ts:
            # 前20%：积极吃 ask（确保快速建仓）+ 被动 bid+1 补余量
            if takeable > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, takeable))
            remaining = buy_cap - takeable
            if remaining > 0 and best_bid is not None:
                passive = min(best_bid + 1, int(fv - 1))
                if passive > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", passive, remaining))
        elif timestamp < eod_start_ts:
            # 中段：仍未满仓则继续补（ask <= FV+8 才积极）
            if best_ask <= fv + 8 and takeable > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, takeable))
                remaining = buy_cap - takeable
                if remaining > 0 and best_bid is not None:
                    passive = min(best_bid + 1, int(fv - 1))
                    if passive > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", passive, remaining))
            elif best_bid is not None:
                passive = min(best_bid + 1, int(fv - 1))
                if passive > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", passive, buy_cap))

        return orders

    # ----------------------------------------------------------
    #  ASH_COATED_OSMIUM
    #
    #  R2 修复：ACO_JUMP_BOOST=0（R2跳变后无回撤，加量会亏）
    #  其余逻辑保持（OBI单侧、strong/extreme signal、pre-EOD bias）
    #  参数按 limit=80 等比扩大
    # ----------------------------------------------------------
    def _aco_orders(self, depth, pos: int, limit: int,
                    timestamp: int) -> list:
        orders = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread   = best_ask - best_bid
        if spread <= 1:
            return orders

        mid = (best_bid + best_ask) / 2.0

        # OBI 更新
        curr_delta = (mid - self._last_mid_aco) if self._last_mid_aco is not None else 0.0
        prev_delta = self._last_delta_aco
        # R2: 跳变后无回撤，不再触发 JUMP_BOOST
        jumped_up   = prev_delta >= ACO_JUMP_THRESHOLD and ACO_JUMP_BOOST > 0
        jumped_down = prev_delta <= -ACO_JUMP_THRESHOLD and ACO_JUMP_BOOST > 0

        if self._last_mid_aco is not None:
            self._obi_pairs_aco.append((self._last_obi_aco, curr_delta))
            self._obi_samples += 1
            if self._obi_samples >= OBI_UPDATE_N:
                self._update_obi_beta()
                self._obi_samples = 0

        # Deep OBI
        bid_levels = sorted(depth.buy_orders.items(), reverse=True)
        ask_levels = sorted(depth.sell_orders.items())
        bv1 = sum(v for _, v in bid_levels[:1])
        av1 = sum(abs(v) for _, v in ask_levels[:1])
        bv2 = bid_levels[1][1] if len(bid_levels) > 1 else 0
        av2 = abs(ask_levels[1][1]) if len(ask_levels) > 1 else 0
        db  = OBI_W1 * bv1 + OBI_W2 * bv2
        da  = OBI_W1 * av1 + OBI_W2 * av2
        deep_obi = (db - da) / (db + da + 1e-6)

        self._last_obi_aco   = deep_obi
        self._last_mid_aco   = mid
        self._last_delta_aco = curr_delta

        # Bot alpha
        bot_alpha = (
            self._bot_pos["Olivia" ]["ASH_COATED_OSMIUM"] * 0.6 +
            self._bot_pos["Camilla"]["ASH_COATED_OSMIUM"] * 1.0 -
            self._bot_pos["Pablo"  ]["ASH_COATED_OSMIUM"] * 0.4
        )
        bot_alpha = max(-1.5, min(1.5, bot_alpha))

        fair   = ACO_FAIR + self._obi_beta_aco * deep_obi + bot_alpha
        signal = fair - ACO_FAIR
        agree  = (deep_obi > 0.10 and bot_alpha > 0.10) or \
                 (deep_obi < -0.10 and bot_alpha < -0.10)

        strong_buy   = signal >= ACO_STRONG_SIGNAL and agree
        strong_sell  = signal <= -ACO_STRONG_SIGNAL and agree
        extreme_buy  = signal >= ACO_EXTREME_SIGNAL and deep_obi > 0
        extreme_sell = signal <= -ACO_EXTREME_SIGNAL and deep_obi < 0

        eod_start_ts     = self._day_threshold(ACO_EOD_START_FRAC)
        eod_hard_ts      = self._day_threshold(ACO_EOD_HARD_FRAC)
        pre_eod_bias_ts  = self._day_threshold(ACO_PRE_EOD_BIAS_FRAC)

        # ── EOD 平仓 ─────────────────────────────────────────
        if timestamp >= eod_start_ts and pos != 0:
            if pos > 0:
                px = best_bid if timestamp >= eod_hard_ts else best_bid + 1
                orders.append(Order("ASH_COATED_OSMIUM", px, -pos))
            else:
                px = best_ask if timestamp >= eod_hard_ts else max(best_bid + 1, best_ask - 1)
                orders.append(Order("ASH_COATED_OSMIUM", px, -pos))
            return orders

        # ── bid/ask shift 计算 ────────────────────────────────
        bid_shift = 0
        ask_shift = 0

        # 库存偏向
        if pos > ACO_INVENTORY_SOFT:
            bid_shift -= 1
            ask_shift -= 1
        if pos < -ACO_INVENTORY_SOFT:
            bid_shift += 1
            ask_shift += 1

        # OBI 方向（正向指标：OBI>0.6 → 价格涨 +6.9pts）
        if deep_obi > 0.35:
            bid_shift += 1
            ask_shift += 1
        elif deep_obi < -0.35:
            bid_shift -= 1
            ask_shift -= 1

        # fair vs best_ask/bid
        if fair >= best_ask - 1:
            bid_shift += 1
        elif fair <= best_bid + 1:
            ask_shift -= 1

        # ±8 跳变位置调整（R2 中 JUMP_BOOST=0，只做 shift 调整）
        if jumped_up and pos > 0:
            ask_shift -= 1   # 高点时 ask 稍低，更容易卖出
        elif jumped_down and pos < 0:
            bid_shift += 1   # 低点时 bid 稍高，更容易买入

        # Strong/extreme signal
        if strong_buy:
            bid_shift += 1
            ask_shift += 1
        elif strong_sell:
            bid_shift -= 1
            ask_shift -= 1

        if extreme_buy:
            bid_shift += 1
            ask_shift += 1
        elif extreme_sell:
            bid_shift -= 1
            ask_shift -= 1

        # Pre-EOD bias（渐进式单侧平仓）
        only_buy = only_sell = False
        if timestamp >= pre_eod_bias_ts:
            denom    = max(1, eod_hard_ts - pre_eod_bias_ts)
            progress = (timestamp - pre_eod_bias_ts) / denom
            unwind   = 1 + int(2 * progress)
            if pos > 0:
                bid_shift -= unwind
                ask_shift -= unwind
                only_sell  = True
            elif pos < 0:
                bid_shift += unwind
                ask_shift += unwind
                only_buy   = True

        my_bid = best_bid + 1 + bid_shift
        my_ask = best_ask - 1 + ask_shift
        my_bid = min(my_bid, best_ask - 1)
        my_ask = max(my_ask, best_bid + 1)
        if my_bid >= my_ask:
            return orders

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # 硬库存警戒
        if abs(pos) >= ACO_INVENTORY_HARD:
            if pos > 0 and sell_cap > 0:
                orders.append(Order("ASH_COATED_OSMIUM",
                                    max(best_bid + 1, my_ask),
                                    -min(sell_cap, ACO_BASE_SIZE + 4)))
            elif pos < 0 and buy_cap > 0:
                orders.append(Order("ASH_COATED_OSMIUM",
                                    min(best_ask - 1, my_bid),
                                    min(buy_cap, ACO_BASE_SIZE + 4)))
            return orders

        # 报价量（按仓位非对称）
        skew = abs(pos) / max(1, limit)
        if pos >= 0:
            buy_sz  = max(1, int(ACO_BASE_SIZE * (1 - 0.6 * skew)))
            sell_sz = max(1, int(ACO_BASE_SIZE * (1 + 0.8 * skew)))
        else:
            buy_sz  = max(1, int(ACO_BASE_SIZE * (1 + 0.8 * skew)))
            sell_sz = max(1, int(ACO_BASE_SIZE * (1 - 0.6 * skew)))

        # Signal 调量
        if strong_buy:
            buy_sz  += 2
            sell_sz  = max(1, sell_sz - 2)
        elif strong_sell:
            sell_sz += 2
            buy_sz   = max(1, buy_sz - 2)

        if extreme_buy and pos <= 0 and spread >= 4:
            sell_sz = 0
        elif extreme_sell and pos >= 0 and spread >= 4:
            buy_sz  = 0

        if only_buy:
            sell_sz = 0
        elif only_sell:
            buy_sz  = 0

        # ── OBI 极端单侧报价（隐藏 FV，防信息泄露）─────────
        # OBI>0.6 → 价格即将涨 → 停止卖出，集中买入
        if not only_buy and not only_sell:
            if deep_obi > ACO_OBI_ONESIDED and pos < int(limit * 0.75):
                sell_sz = 0
                buy_sz  = min(buy_cap, buy_sz + 2)
            elif deep_obi < -ACO_OBI_ONESIDED and pos > -int(limit * 0.75):
                buy_sz  = 0
                sell_sz = min(sell_cap, sell_sz + 2)

        if buy_cap  > 0 and buy_sz  > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid,
                                 min(buy_sz, buy_cap)))
        if sell_cap > 0 and sell_sz > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask,
                                 -min(sell_sz, sell_cap)))
        return orders

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------
    def run(self, state: TradingState):
        self._load(state.traderData)
        self._update_signals(state.market_trades)

        ts = state.timestamp

        # 新天检测（ts 大幅回落）
        if self._last_ts is not None and ts < self._last_ts:
            self._ipr_day        += 1
            self._day_max_ts_seen = max(self._day_max_ts_seen, self._last_ts)
        self._last_ts         = ts
        self._day_max_ts_seen = max(self._day_max_ts_seen, ts)

        # 每 tick 从 LOB 实时推断 IPR day
        ipr_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if ipr_depth is not None:
            self._infer_ipr_day(ipr_depth, ts)

        result = {}
        for symbol, depth in state.order_depths.items():
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 80)

            if symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._ipr_orders(depth, pos, limit, ts)
            elif symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._aco_orders(depth, pos, limit, ts)

        return result, 0, self._save()
