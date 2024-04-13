
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import string
import json
import pandas as pd
import statistics
import math

 # storing string as const to avoid typos
SUBMIT = "SUBMISSION"
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

    
    def __init__(self) -> None:
    
        print("Starting Trader")

        self.position_limit = {
            AMET : 20,
            STAR: 20,
            }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
    
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        
        for product in PRODUCTS:
            self.ema_prices[product] = 0

        self.ema_param = 0.5

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
   
    def value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
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

        D = {(9996.0, 10004.0): [2955, 2860], (9996.0, 9998.0): [379, 2417], (9996.0, 10002.0): [631, 1318], (9996.0, 10000.0): [177, 154], (10000.0, 10004.0): [139, 77], (10000.0, 10005.0): [65, 25], (10000.0, 10002.0): [44, 37], (9995.0, 10005.0): [1932, 1869], (9995.0, 9998.0): [294, 1280], (9995.0, 10002.0): [457, 624], (9995.0, 10000.0): [27, 80], (9998.0, 10004.0): [1413, 762], (9998.0, 10005.0): [673, 456], (9998.0, 10000.0): [46, 47], (10002.0, 10004.0): [2482, 353], (10002.0, 10005.0): [1303, 221]}

        order_depth: OrderDepth = state.order_depths[product]
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
        
      position_star = self.position(STAR, state)

      bid_volume = self.position_limit[STAR] - position_star
      ask_volume = - self.position_limit[STAR] - position_star

      orders: List[Order] = []

      if position_star == 0:
          # Not long nor short
          orders.append(Order(STAR, math.floor(self.ema_prices[STAR]-0.5), bid_volume))
          orders.append(Order(STAR, math.ceil(self.ema_prices[STAR]+0.5), ask_volume))
      
      if position_star > 0:
          # Long position
          orders.append(Order(STAR, math.floor(self.ema_prices[STAR]-1), bid_volume))
          orders.append(Order(STAR, math.ceil(self.ema_prices[STAR]), ask_volume))

      if position_star < 0:
          # Short position
          orders.append(Order(STAR, math.floor(self.ema_prices[STAR]), bid_volume))
          orders.append(Order(STAR, math.ceil(self.ema_prices[STAR]+1), ask_volume))

      return orders
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        result = { AMET : [], STAR : []} 
        
        self.round += 1
        pnl = self.pnl(state)
        self.ema_price(state)

        print(f"Log round {self.round}")

        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        
        for product in PRODUCTS:
            print(f"\tProduct {product}, Position {self.position(product, state)}, Midprice {self.mid_price(product, state)}, Value {self.value_on_product(product, state)}, EMA {self.ema_prices[product]}")
            
        print(f"\tPnL {pnl}")
       
        
        # Call the compute_orders_ameth function
        #result['AMETHYSTS'] = self.compute_orders_ameth(state)
        result[STAR] = self.compute_orders_starfruit(state)
      
      
        return result, conversions, trader_data