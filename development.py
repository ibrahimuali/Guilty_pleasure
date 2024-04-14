from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import string
import json
import pandas as pd
import statistics
import math

SUBMIT = "LITTLETRADER"
AMET = "AMETHYSTS"
STAR = "STARFRUIT"

PRODUCTS = [
    AMET,
    STAR,
]

DEFAULT_PRICES = {
    AMET: 10000,
    STAR: 5000,
}

class Trader:
    def __init__(self) -> None:
        self.position_limit = {
            AMET: 20,
            STAR: 20,
        }
        self.round = 0
        self.cash = 0
        self.past_prices = {product: [] for product in PRODUCTS}
        self.ema_prices = {product: 0 for product in PRODUCTS}
        self.ema_param = 0.5

    def position(self, product, state: TradingState):
        return state.position.get(product, 0)
    
    def mid_price(self, product, state: TradingState):
        default_price = self.ema_prices[product] or DEFAULT_PRICES[product]
        if product not in state.order_depths:
            return default_price
        
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        if not market_bids or not market_asks:
            return default_price
        
        return (max(market_bids) + min(market_asks)) / 2
    
    def adjust_ema_param(self, state: TradingState):
        recent_prices = self.past_prices[STAR][-10:]
        # Ensure there are at least two data points before calculating volatility
        if len(recent_prices) > 1:
            volatility = statistics.stdev(recent_prices)
            self.ema_param = min(max(0.1, 1 / volatility), 0.9)
        else:
            # Default or fallback value for ema_param if not enough data is available
            self.ema_param = 0.5

    
    def detect_trend(self):
        # Simple trend detection
        short_term_ema = self.ema_prices[STAR]
        long_term_ema = statistics.mean(self.past_prices[STAR][-30:]) if len(self.past_prices[STAR]) > 30 else short_term_ema
        if short_term_ema > long_term_ema:
            return "uptrend"
        elif short_term_ema < long_term_ema:
            return "downtrend"
        return "sideways"
    
    def ema_price(self, state: TradingState):
        for product in PRODUCTS:
            mid_price = self.mid_price(product, state)
            self.past_prices[product].append(mid_price)
            if self.ema_prices[product] == 0:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]
        self.adjust_ema_param(state)  # Adjust EMA parameter dynamically
    
    def compute_orders_starfruit(self, state: TradingState) -> List[Order]:
        position_star = self.position(STAR, state)

        # Assuming OrderBook is accessible via state.order_depths[STAR]
        order_depth = state.order_depths[STAR]
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)

        # Calculate undercut prices
        undercut_buy = best_bid + 1
        undercut_sell = best_ask - 1 if best_ask > 0 else 0  # Ensure not negative

        bid_volume = self.position_limit[STAR] - position_star
        ask_volume = -self.position_limit[STAR] - position_star

        orders = []

        # Adjust your EMA prices with respect to the market conditions
        ema_bid_price = math.floor(self.ema_prices[STAR] - 1)
        ema_ask_price = math.ceil(self.ema_prices[STAR] + 1)

        # Adjust prices based on the current best bid/ask to make them more competitive
        adjusted_bid_price = max(ema_bid_price, undercut_buy)
        adjusted_ask_price = min(ema_ask_price, undercut_sell) if undercut_sell > 0 else ema_ask_price + 1

        if position_star == 0:
            orders.append(Order(STAR, adjusted_bid_price, bid_volume))
            orders.append(Order(STAR, adjusted_ask_price, ask_volume))
    
        if position_star > 0:
            orders.append(Order(STAR, adjusted_bid_price - 2, bid_volume))
            orders.append(Order(STAR, adjusted_ask_price, ask_volume))

        if position_star < 0:
            orders.append(Order(STAR, adjusted_bid_price, bid_volume))
            orders.append(Order(STAR, adjusted_ask_price + 2, ask_volume))

        return orders
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        self.ema_price(state)  # Update EMA prices
        conversions = 0
        trader_data = SUBMIT
        
        result = {AMET: [], STAR: self.compute_orders_starfruit(state)}
        
        return result, conversions, trader_data
