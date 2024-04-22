#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:19:52 2024

@author: ibrahimuali
"""

from typing import List, Dict, Tuple, Union
import string
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import Any
import math
import pandas as pd
import numpy as np
import statistics

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

SUBMIT = "SUBMISSION"
AMET = "AMETHYSTS"
STAR = "STARFRUIT"
ORC = 'ORCHIDS'
BASK = 'GIFT_BASKET'
ICHIGO = 'STRAWBERRIES'
CHOCO = 'CHOCOLATE'
ROSE = 'ROSES'

PRODUCTS = [
    AMET,
    STAR,
    ORC,
    BASK,
    ICHIGO,
    CHOCO,
    ROSE
]

DEFAULT_PRICES = {
    AMET : 10000,
    STAR : 5000,
    ORC : 1100,
    CHOCO : 8000,
    ICHIGO : 4000,
    ROSE : 15000,
    BASK : 71000
}

POSITION_LIMIT = {
    BASK: 60,
    ROSE: 60,
    CHOCO: 250,
    ICHIGO: 350
}

VOLUME_BASKET = 2
WINDOW = 200


class Trader:
    def __init__(self) -> None:

        self.position_limit = {
            AMET : 20,
            STAR : 20,
            ORC : 100,
            CHOCO : 250,
            ICHIGO : 350,
            ROSE : 60,
            BASK : 60
            }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        
        self.prices : Dict[PRODUCTS, pd.Series] = {
            "SPREAD_GIFT": pd.Series(),
            }
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()

        for product in PRODUCTS:
            self.past_prices[product] = []
        
        self.ema_prices = dict()
        
        for product in PRODUCTS:
            self.ema_prices[product] = 0
            
        self.ema_param = 0.5
        
        self.p_mid = []
        self.p_spread = []

    def position(self, product, state : TradingState):
        return state.position.get(product, 0) 
   
    def mid_price(self, product, state : TradingState):
        
        default_price = self.ema_prices[product]
        
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            return default_price
    
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            return default_price
    
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        
        return (best_bid + best_ask)/2
    
    def save_prices_product(self, product, state: TradingState, price: Union[float, int, None] = None):
        if not price:
            price = self.mid_price(product, state)
    
        if product not in self.prices:
            self.prices[product] = pd.Series({state.timestamp: price})
        else:
            new_series = pd.Series({state.timestamp: price})
            # Check if the new series is not empty
            if not new_series.empty:
                # Filter out empty Series before concatenating
                non_empty_series = [s for s in [self.prices[product], new_series] if not s.empty]
                self.prices[product] = pd.concat(non_empty_series)

   
    def value_on_product(self, product, state : TradingState):
        """
        Returns the amount of Cash currently held on the product.  
        """
        value = self.position(product, state) * self.mid_price(product, state)
         
        return value
 
    def pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        
        def value_on_positions():
            value = 0
            for product in state.position:
                value += self.value_on_product(product, state)
            return value
        
        def new_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                    # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMIT:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMIT:
                        self.cash += trade.quantity * trade.price
           
        return self.cash + value_on_positions()
    
    def ema_price(self, state : TradingState):
        """
        Update the exponential moving average of the prices of Starfruit
        """
        for product in PRODUCTS:
            mid_price = self.mid_price(product, state)
            if mid_price == 0:
                continue

            # Update ema price
            if self.ema_prices[product] == 0:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

    def compute_orders_ameth(self, state: TradingState) -> List[Order]:
        product = 'AMETHYSTS'
        limit = 20
        order_depth: OrderDepth = state.order_depths[product]
        
        D = {(9996.0, 10004.0): [2955, 2860], (9996.0, 9998.0): [379, 2417], (9996.0, 10002.0): [631, 1318], (9996.0, 10000.0): [177, 154], (10000.0, 10004.0): [139, 77], (10000.0, 10005.0): [65, 25], (10000.0, 10002.0): [44, 37], (9995.0, 10005.0): [1932, 1869], (9995.0, 9998.0): [294, 1280], (9995.0, 10002.0): [457, 624], (9995.0, 10000.0): [27, 80], (9998.0, 10004.0): [1413, 762], (9998.0, 10005.0): [673, 456], (9998.0, 10000.0): [46, 47], (10002.0, 10004.0): [2482, 353], (10002.0, 10005.0): [1303, 221]}

        orders: List[Order] = []

        if product in state.position.keys():
            q = state.position[product]
        else:
            q = 0

        if len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            tot = 0
            vol = 0
            ctr = 0
            for key in sorted(order_depth.sell_orders.keys()):
                if ctr == 3:
                    break
                else:
                    tot += key * order_depth.sell_orders[key]
                    vol += order_depth.sell_orders[key]
                    ctr += 1
            ask_vwap = tot / vol

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            tot = 0
            vol = 0
            ctr = 0
            for key in sorted(order_depth.buy_orders.keys()):
                if ctr == 3:
                    break
                else:
                    tot += key * order_depth.buy_orders[key]
                    vol += order_depth.buy_orders[key]
                    ctr += 1
            bid_vwap = tot / vol

        self.p_mid.append((bid_vwap + ask_vwap) / 2)
        self.p_spread.append(best_ask - best_bid)

        mu = round(statistics.fmean(self.p_mid))
        spread = round(statistics.fmean(self.p_spread))

        if (best_bid, best_ask) not in D.keys():
            orders.append(Order(product, mu + spread // 2, (-q - limit)))
            orders.append(Order(product, mu - spread // 2, (limit - q)))
        else:
            p = D[(best_bid, best_ask)][0] / (D[(best_bid, best_ask)][0] + D[(best_bid, best_ask)][1])
            if p == 1:
                p = 0.99
            func = (p * (limit - q) + q) / (1 - p)
            sell_qty = math.floor(min(limit + q, max(0, func)))
            buy_qty = math.floor(((1 - p) * sell_qty - q) / p)

            buy_qty = min(limit - q, max(0, buy_qty))
            f = 0

            if best_ask > mu and best_bid < mu:
                if q < 0:
                    orders.append(Order(product, best_ask, -math.ceil(sell_qty * f)))
                    orders.append(Order(product, best_ask - 1, -math.floor(sell_qty * (1 - f))))
                    orders.append(Order(product, best_bid + 1, math.floor(buy_qty * (1 - f))))
                    orders.append(Order(product, best_bid, math.floor(buy_qty * f)))
                if q >= 0:
                    orders.append(Order(product, best_ask, -math.ceil(sell_qty * f)))
                    orders.append(Order(product, best_ask - 1, -math.floor(sell_qty * (1 - f))))
                    orders.append(Order(product, best_bid + 1, math.floor(buy_qty * (1 - f))))
                    orders.append(Order(product, best_bid, math.floor(buy_qty * f)))
            elif best_bid >= mu:
                if best_bid == mu:
                    if q >= 0:
                        orders.append(Order(product, best_bid + 1, -sell_qty))
                    else:
                        orders.append(Order(product, best_bid + 1, -math.ceil(sell_qty)))
                else:
                    if q >= 0:
                        orders.append(Order(product, best_bid, -math.ceil(sell_qty * f)))
                        orders.append(Order(product, best_bid - 1, -math.floor(sell_qty * (1 - f))))
                    else:
                        orders.append(Order(product, best_bid, -math.ceil(sell_qty * f)))
                        orders.append(Order(product, best_bid - 1, -math.floor(sell_qty * (1 - f))))
            else:
                if best_ask == mu:
                    if q <= 0:
                        orders.append(Order(product, best_ask - 1, buy_qty))
                    else:
                        orders.append(Order(product, best_ask - 1, math.ceil(buy_qty)))
                else:
                    if q <= 0:
                        orders.append(Order(product, best_ask, math.ceil(buy_qty * f)))
                        orders.append(Order(product, best_ask + 1, math.ceil(buy_qty * (1 - f))))
                    else:
                        orders.append(Order(product, best_ask, math.ceil(buy_qty * f)))
                        orders.append(Order(product, best_ask + 1, math.floor(buy_qty * (1 - f))))

        return orders

    def compute_orders_starfruit(self, state: TradingState) -> List[Order]:
        
        order_depth: OrderDepth = state.order_depths[STAR]
        position_star = self.position(STAR, state)

        sell_orders = list(order_depth.sell_orders.items())
        buy_orders = list(order_depth.buy_orders.items())
    
        best_bid = max(buy_orders, key=lambda x: x[0], default=(0, 0))[0] if buy_orders else 0
        best_ask = min(sell_orders, key=lambda x: x[0], default=(0, 0))[0] if sell_orders else 0

        bid_volume = math.floor(self.position_limit[STAR] - position_star)
        ask_volume = math.ceil(-self.position_limit[STAR] - position_star)

        orders = []
   
        ema_bid_price = math.floor(self.ema_prices[STAR]) 
        ema_ask_price = math.ceil(self.ema_prices[STAR])
            
        order_bid_price = ema_bid_price - 1
        order_ask_price = ema_ask_price + 1

        if position_star == 0 or position_star > 0 or position_star < 0:
            bid_volume = max(0, bid_volume)
            ask_volume = max(0, ask_volume)
        

            if bid_volume > 0:
                orders.append(Order(STAR, order_bid_price, bid_volume))
            
            if ask_volume > 0:
                orders.append(Order(STAR, order_ask_price, ask_volume))
                
        return orders
    
    def volumes(self, product, state: TradingState):
        sell_orders, buy_orders, vol_buy, vol_sell = {}, {}, {}, {}
        
        
        sell_orders[product] = list(state.order_depth[product].sell_orders.items())
        buy_orders[product] = list(state.order_depth[product].buy_orders.items())
        
        vol_buy[product], vol_sell[product] = 0, 0
        
        for price, vol in buy_orders[product].items():
           vol_buy += vol 
           
           if vol_buy[product] >= self.POSITION_LIMIT[product]/10:
               break
           
        for price, vol in sell_orders[product].items():
           vol_sell += -vol 
           
           if vol_sell[product] >= self.POSITION_LIMIT[product]/10:
               break
     
    def compute_orders_gift(self, state: TradingState)-> List[Order]: 
        orders_choco = []
        orders_ichigo = []
        orders_rose = []
        orders_bask = []
    
        def create_orders(buy_basket: bool) -> List[List[Order]]:
            if buy_basket:
                sign = 1
                price_basket = 10000000
                price_others = 1
            else:
                sign = -1
                price_basket = 1
                price_others = 10000000
        
            orders_bask.append(Order(BASK, price_basket, sign*VOLUME_BASKET))
            orders_choco.append(Order(CHOCO, price_others, -sign*4*VOLUME_BASKET))
            orders_ichigo.append(Order(ICHIGO, price_others, -sign*6*VOLUME_BASKET))
            orders_rose.append(Order(ROSE, price_others, -sign*VOLUME_BASKET))

            #return orders_baguette, orders_basket, orders_dip, orders_ukulele 
    

        price_bask = self.mid_price(BASK, state)
        price_choco = self.mid_price(CHOCO, state)
        price_ichigo= self.mid_price(ICHIGO, state)
        price_rose = self.mid_price(ROSE, state)

        position_bask = self.position(BASK, state)
        position_choco = self.position(CHOCO, state)
        position_ichigo = self.position(ICHIGO, state)
        position_rose = self.position(ROSE, state)

        spread = price_bask - (4*price_choco + 6*price_ichigo + price_rose)
        
        #Introducing Premium 
        prem = price_bask - (4*price_choco + 6*price_ichigo + price_rose + 379)
        
        self.save_prices_product("SPREAD_GIFT", state, spread)

        avg_spread = self.prices["SPREAD_GIFT"].rolling(WINDOW).mean()
        std_spread = self.prices["SPREAD_GIFT"].rolling(WINDOW).std()
        spread_5 = self.prices["SPREAD_GIFT"].rolling(5).mean()

        if not np.isnan(avg_spread.iloc[-1]):
            avg_spread = avg_spread.iloc[-1]
            std_spread = std_spread.iloc[-1]
            spread_5 = spread_5.iloc[-1]
            #logger.print(f"Average spread: {avg_spread}, Spread5: {spread_5}, Std: {std_spread}")

            if abs(position_bask) <= POSITION_LIMIT[BASK] - 2:
                if spread_5 < avg_spread - 2*std_spread:  # buy basket
                    buy_bask = True
                    create_orders(buy_bask)

                elif spread_5 > avg_spread + 2*std_spread: # sell basket
                    sell_bask = False 
                    create_orders(sell_bask)
            
            else:
                if position_bask > 0 : # sell basket
                    if spread_5 > avg_spread + 2*std_spread:
                        sell_bask = False
                        create_orders(sell_bask)

                else: # buy basket
                    if spread_5 < avg_spread - 2*std_spread:
                        buy_bask = True
                        create_orders(buy_bask)

        return orders_bask, orders_choco, orders_ichigo, orders_rose

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = "Little_Trader"
        result = { AMET : [], STAR : [], BASK : [], CHOCO : [], ICHIGO : [], ROSE : []} 
        
        self.round += 1
        pnl = self.pnl(state)
        self.ema_price(state)

        logger.print(f"Log round {self.round}")

        logger.print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    logger.print(trade)

        logger.print(f"\tCash {self.cash}")
        
        for product in PRODUCTS:
            logger.print(f"\tProduct {product}, Position {self.position(product, state)}, Midprice {self.mid_price(product, state)}, Value {self.value_on_product(product, state)}, EMA {self.ema_prices[product]}")
            
        logger.print(f"\tPnL {pnl}")
     
        try:  
            result[STAR] = self.compute_orders_starfruit(state)
        except Exception as e:
            logger.print(e)
            
        try:    
            result[AMET] = self.compute_orders_ameth(state)
        except Exception as e:
            logger.print(e)
        
        try:             
            result[BASK], _, _, _ = self.compute_orders_gift(state)
        except Exception as e:
            logger.print(e)
            
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data