from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import string
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

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


class Trader:
    
    def improved_strategy_starfruit(self, order_depth: OrderDepth):
        orders = []
        best_ask, _ = self.get_best_ask(order_depth)
        best_bid, _ = self.get_best_bid(order_depth)
        
        buy_pressure, sell_pressure = self.calculate_order_book_imbalance(order_depth)
        if buy_pressure > sell_pressure * 1.2 and best_ask is not None:  # Significant buy sentiment
            orders.append(Order('STARFRUIT', best_ask, 1))  # Buy 1 unit
        elif sell_pressure > buy_pressure * 1.2 and best_bid is not None:  # Significant sell sentiment
            orders.append(Order('STARFRUIT', best_bid, -1))  # Sell 1 unit
        
        return orders

    def improved_strategy_amethysts(self, order_depth: OrderDepth):
        orders = []
        best_ask, _ = self.get_best_ask(order_depth)
        best_bid, _ = self.get_best_bid(order_depth)
        
        if best_ask is not None and best_bid is not None:
            # Attempt to buy slightly above the best bid and sell slightly below the best ask
            orders.append(Order('AMETHYSTS', best_bid + 0.01, 1))  # Buy closer to mid-price
            orders.append(Order('AMETHYSTS', best_ask - 0.01, -1))  # Sell closer to mid-price
        
        return orders

    def get_best_ask(self, order_depth: OrderDepth):
        if order_depth.sell_orders:
            best_ask_price = min(order_depth.sell_orders.keys())
            best_ask_amount = order_depth.sell_orders[best_ask_price]
            return best_ask_price, best_ask_amount
        return None, None

    def get_best_bid(self, order_depth: OrderDepth):
        if order_depth.buy_orders:
            best_bid_price = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid_price]
            return best_bid_price, best_bid_amount
        return None, None

    def calculate_order_book_imbalance(self, order_depth: OrderDepth):
        buy_pressure = sum(order_depth.buy_orders.values())
        sell_pressure = sum(order_depth.sell_orders.values())
        return buy_pressure, sell_pressure
    
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}
        
        for product, order_depth in state.order_depths.items():
            if product == 'STARFRUIT':
                orders = self.improved_strategy_starfruit(order_depth)
            elif product == 'AMETHYSTS':
                orders = self.improved_strategy_amethysts(order_depth)
            else:
                continue  # Skip if the product is not STARFRUIT or AMETHYSTS
            
            if orders:
                result[product] = orders
                
        return result, 0, ""  # Assuming conversions and trader_data are placeholders

    