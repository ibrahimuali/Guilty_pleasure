import numpy as np
import collections
from collections import defaultdict
import copy
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import string
import json
import pandas as pd
import statistics
import math
from typing import List, Dict, Tuple


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

AMET = "AMETHYSTS"
STAR = "STARFRUIT"

PRODUCTS = [
    AMET,
    STAR,
]

DEFAULT_PRICES = {
    AMET : 10000,
    STAR : 5000,
}

class Trader:
    def __init__(self):
        self.position = defaultdict(int)
        self.POSITION_LIMIT = {STAR: 20, AMET: 20}
        self.starfruit_cache = []
        self.starfruit_dim = 4
        self.coef = [0.18898843, 0.20770677, 0.26106908, 0.34176867]
        self.intercept = 2.3564943532519464
        self.ema_prices = defaultdict(float)
        self.cash = 0
        self.round = 0

    def calc_next_price_starfruit(self):
        if len(self.starfruit_cache) < self.starfruit_dim:
            return None
        nxt_price = self.intercept
        for i in range(self.starfruit_dim):
            nxt_price += self.starfruit_cache[i] * self.coef[i]
        return int(round(nxt_price))

    def compute_orders_starfruit(self, order_depth, state):
        orders = []
        acc_bid, acc_ask = 5000, 5000  # Example values; adjust based on prediction

        # Calculate the current mid price from available order depth
        current_mid_price = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
        self.starfruit_cache.append(current_mid_price)  # Maintain price history for prediction
        self.ema_prices['STAR'] = self.update_ema('STAR', current_mid_price)  # Update EMA

        predicted_price = self.calc_next_price_starfruit()
        if predicted_price is not None:
            acc_bid = predicted_price - 1
            acc_ask = predicted_price + 1

        # Check current position and calculate allowed quantity to buy or sell
        current_position = self.position[STAR]
        max_buy = self.POSITION_LIMIT[STAR] - current_position  # Max quantity we can buy
        max_sell = current_position + self.POSITION_LIMIT['STAR']  # Max quantity we can sell if we are short
        
        if max_buy > 0:
            # Only place buy order if we have room within the limit
            buy_quantity = min(max_buy, some_default_buy_volume)  # some_default_buy_volume is an example placeholder
            orders.append(Order(STAR, acc_bid, buy_quantity))

        if max_sell > 0:
            # Only place sell order if we have room within the limit
            sell_quantity = min(max_sell, some_default_sell_volume)  # some_default_sell_volume is an example placeholder
            orders.append(Order(STAR, acc_ask, -sell_quantity))

        return orders


    def compute_orders_ameth(self, state):
        # Placeholder for AMETHYSTS order computation logic
        return []

    def update_ema(self, product, current_price):
        alpha = 0.1  # Example smoothing factor
        old_ema = self.ema_prices[product]
        new_ema = alpha * current_price + (1 - alpha) * old_ema
        return new_ema

    def pnl(self, state):
        # Placeholder for PnL calculation logic
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        trader_data = "Little_Trader"
        result = {STAR: [], AMET: []}

        self.round += 1
        pnl = self.pnl(state)
        #self.ema_prices(state)

        logger.print(f"Log round {self.round}")
        logger.print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    logger.print(trade)

        logger.print(f"\tCash {self.cash}")

        PRODUCTS = [STAR, AMET]
       # for product in PRODUCTS:
           # print(f"\tProduct {product}, Position {self.position[product]}, Midprice {self.mid_price(product, state)}, Value {self.value_on_product(product, state)}, EMA {self.ema_prices[product]}")

        logger.print(f"\tPnL {pnl}")

        for product in PRODUCTS:
            order_depth: OrderDepth = state.order_depths[product]
            if product == STAR:
                orders_starfruit = self.compute_orders_starfruit(order_depth, state)
                result[STAR] += orders_starfruit
            else:
                orders_amet = self.compute_orders_ameth(state)
                result[AMET] += orders_amet

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
