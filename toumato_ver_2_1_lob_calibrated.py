import json
from collections import defaultdict
from datamodel import Order, TradingState

# ============================================================
#  基础配置（根据历史数据校准）
# ============================================================
POSITION_LIMIT = {"EMERALDS": 20, "TOMATOES": 20}

# EMERALDS：历史数据确认 FV = 10000，静态 bot 永远挂 9992/10008
# TOMATOES：历史数据显示主要 spread = 13-14，无固定公允价，用 AR(4) 辅助
EMERALD_FV = 10000

# TOMATOES AR(4) 参数（用于计算 reservation price 的参考 FV，不用于直接报价）
TOM_DIM       = 4
TOM_COEF      = [-0.02, 0.04, 0.16, 0.82]
TOM_INTERCEPT = 0.0

# ============================================================
#  做市商参数（数据驱动校准）
# ============================================================
# 库存风险厌恶：每持有 1 手多头，reservation price 下移 GAMMA 点
# EMERALDS：mid 非常稳定（std=0.72），GAMMA 可以小一点
# TOMATOES：mid 波动大（std=19.75），GAMMA 需要大一点来对冲库存风险
GAMMA = {"EMERALDS": 0.1, "TOMATOES": 0.2}

# 每次最多挂多少手（防止一口气耗尽 limit）
# 历史数据：每次成交量 2-8 手，limit=20，挂 5 手比较合理
QUOTE_SIZE = {"EMERALDS": 5, "TOMATOES": 5}

# 库存警戒线：超过这个比例时启动单边减仓模式
INVENTORY_ALERT = 0.70

# 最小有效 spread（spread <= 这个值时夹单无利润，放弃）
# EMERALDS 主要 spread=16，TOMATOES 主要 spread=13-14，MIN=2 安全
MIN_SPREAD = 2

# ============================================================
#  关键发现（来自历史数据分析）
# ============================================================
# EMERALDS：
#   - 静态 bot 永远在 9992/10008（占 98.4% 的 tick）
#   - 我们报 9993/10007：profit = 14/轮，且始终队列第一
#   - 原代码用 FV±1=9999/10001 只能赚 2/轮，严重低估
#
# TOMATOES：
#   - spread 主要为 13-14（93% 的时间），夹单利润 ≈ 11
#   - LOB 每 tick 移动（日内范围 60-70 点），不能用固定参考价
#   - 直接读 best_bid/ask 内缩 1 档，结合 reservation price 防单边
#   - 偶发 spread 收窄至 5-9（7%，平均仅 1.1 tick），不影响整体
# ============================================================


class Trader:

    def __init__(self):
        self._tom_cache = []
        self._bot_pos   = defaultdict(lambda: defaultdict(float))

    # ----------------------------------------------------------
    #  持久化
    # ----------------------------------------------------------
    def _load(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            d = json.loads(trader_data)
            self._tom_cache = d.get("tom_cache", [])
            for b, positions in d.get("bot_pos", {}).items():
                for s, v in positions.items():
                    self._bot_pos[b][s] = v
        except Exception:
            pass

    def _save(self) -> str:
        self._tom_cache = self._tom_cache[-10:]
        return json.dumps({
            "tom_cache": self._tom_cache,
            "bot_pos":   {b: dict(s) for b, s in self._bot_pos.items()}
        })

    # ----------------------------------------------------------
    #  Bot 信号追踪（Olivia 正向，Pablo 反向，Camilla 内幕）
    # ----------------------------------------------------------
    def _update_signals(self, market_trades: dict) -> None:
        target_bots = ["Olivia", "Pablo", "Camilla"]
        for symbol, trades in market_trades.items():
            for t in trades:
                if t.buyer  in target_bots: self._bot_pos[t.buyer ][symbol] += t.quantity
                if t.seller in target_bots: self._bot_pos[t.seller][symbol] -= t.quantity
        for symbol in POSITION_LIMIT:
            self._bot_pos["Olivia" ][symbol] *= 0.99
            self._bot_pos["Pablo"  ][symbol] *= 0.95
            self._bot_pos["Camilla"][symbol] *= 0.99

    # ----------------------------------------------------------
    #  Tomatoes AR(4) FV（仅用于 reservation price 参考）
    # ----------------------------------------------------------
    def _get_fv_tom(self, mid: float | None) -> float:
        if mid is not None:
            self._tom_cache.append(mid)
        if len(self._tom_cache) < TOM_DIM + 1:
            return mid or 5000.0
        diffs     = [self._tom_cache[i] - self._tom_cache[i-1] for i in range(-TOM_DIM, 0)]
        pred_diff = sum(c * d for c, d in zip(TOM_COEF, diffs)) + TOM_INTERCEPT
        return self._tom_cache[-1] + pred_diff

    def _mid(self, depth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    # ----------------------------------------------------------
    #  核心：LOB 内缩做市引擎
    #
    #  每个 tick 的逻辑：
    #  1. 读取实时 best_bid / best_ask
    #  2. 计算 my_bid = best_bid + 1，my_ask = best_ask - 1
    #     → 始终队列第一，所有吃单先打到我们
    #  3. Reservation price 过滤（防单边层 1）
    #     只有当 my_bid ≤ reservation 才挂买单
    #     只有当 my_ask ≥ reservation 才挂卖单
    #  4. 非对称挂单量（防单边层 2）
    #     持多 → 买单缩小 / 卖单放大
    #  5. 紧急清仓（防单边层 3）
    #     持仓 ≥ INVENTORY_ALERT * limit 时停止做市，强制减仓
    # ----------------------------------------------------------
    def _mm_orders(self,
                   symbol:      str,
                   depth,
                   fair_value:  float,
                   position:    int,
                   limit:       int) -> list[Order]:

        orders = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread   = best_ask - best_bid

        # ── 层 3：紧急清仓（优先于一切）────────────────────
        if position >= limit * INVENTORY_ALERT:
            reduce = int(position - limit * 0.5)
            if reduce > 0:
                orders.append(Order(symbol, best_bid, -reduce))
            return orders

        if position <= -limit * INVENTORY_ALERT:
            reduce = int(-position - limit * 0.5)
            if reduce > 0:
                orders.append(Order(symbol, best_ask, reduce))
            return orders

        # ── 利润空间检查 ────────────────────────────────────
        if spread <= MIN_SPREAD:
            return orders

        # ── 我们的报价位置（LOB 内缩 1 档）──────────────────
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        # 确保双边不交叉
        if my_bid >= my_ask:
            return orders

        # ── 层 1：Reservation price 过滤 ─────────────────────
        # 持多 → reservation 低于 FV → 不愿意再买
        # 持空 → reservation 高于 FV → 不愿意再卖
        gamma       = GAMMA[symbol]
        reservation = fair_value - position * gamma

        want_buy  = (my_bid  <= reservation + gamma * 0.5)
        want_sell = (my_ask  >= reservation - gamma * 0.5)

        # ── 层 2：非对称挂单量 ──────────────────────────────
        base_size = QUOTE_SIZE[symbol]
        skew      = abs(position) / limit   # 0=空仓 → 1=满仓

        if position >= 0:  # 持多：买少卖多
            buy_sz  = max(1, int(base_size * (1 - skew)))
            sell_sz = max(1, int(base_size * (1 + skew)))
        else:              # 持空：买多卖少
            buy_sz  = max(1, int(base_size * (1 + skew)))
            sell_sz = max(1, int(base_size * (1 - skew)))

        # ── 实际下单（受持仓上限约束）──────────────────────
        buy_cap  = limit - position
        sell_cap = limit + position

        if want_buy and buy_cap > 0:
            vol = min(buy_sz, buy_cap)
            orders.append(Order(symbol, my_bid, vol))

        if want_sell and sell_cap > 0:
            vol = min(sell_sz, sell_cap)
            orders.append(Order(symbol, my_ask, -vol))

        return orders

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------
    def run(self, state: TradingState):
        self._load(state.traderData)
        self._update_signals(state.market_trades)

        result = {}

        for symbol, depth in state.order_depths.items():
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT[symbol]

            # Bot alpha 信号（叠加到 FV 上，影响 reservation price）
            alpha = (
                self._bot_pos["Olivia" ][symbol] * 0.8 +
                self._bot_pos["Camilla"][symbol] * 1.2 -
                self._bot_pos["Pablo"  ][symbol] * 0.5
            )
            alpha = max(-2.0, min(2.0, alpha))

            if symbol == "EMERALDS":
                # ── EMERALDS ─────────────────────────────────
                # 数据确认：静态 bot 永远报 9992/10008
                # 我们报 9993/10007 → profit=14（原代码 FV±1 只有 2）
                # FV=10000 确认正确（mid 的 std 仅 0.72）
                # reservation 会根据持仓自动决定单边停挂
                fv = EMERALD_FV + alpha

            elif symbol == "TOMATOES":
                # ── TOMATOES ─────────────────────────────────
                # LOB 移动频繁，直接读 best_bid/ask 内缩
                # AR(4) FV 仅用于 reservation price 参考
                mid   = self._mid(depth)
                fv_ar = self._get_fv_tom(mid)

                # 订单簿失衡（OBI）补偿
                bid_vol   = sum(depth.buy_orders.values()) if depth.buy_orders else 0
                ask_vol   = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

                fv = fv_ar + imbalance * 1.5 + alpha

            else:
                continue

            result[symbol] = self._mm_orders(symbol, depth, fv, pos, limit)

        return result, state.timestamp, self._save()
