from typing import List, Dict, Tuple, Union
import string
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import Any
from math import exp, log, sqrt, erf
import pandas as pd
import numpy as np
import statistics
import math

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
COCO = 'COCONUT'
COUP = 'COCONUT_COUPON'

PRODUCTS = [
    AMET,
    STAR,
    ORC,
    BASK,
    ICHIGO,
    CHOCO,
    ROSE,
    COCO,
    COUP
]

DEFAULT_PRICES = {
    AMET : 10000,
    STAR : 5000,
    ORC : 1100,
    CHOCO : 8000,
    ICHIGO : 4000,
    ROSE : 15000,
    BASK : 71000,
    COCO: 10000, 
    COUP: 637.63
}

POSITION_LIMIT = {
    BASK: 60,
    ROSE: 60,
    CHOCO: 250,
    ICHIGO: 350
}

VOLUME_BASKET = 2
WINDOW = 200

INF = int(1e9)

class Trader:
    def __init__(self) -> None:

        self.position_limit = {
            AMET : 20,
            STAR : 20,
            ORC : 100,
            CHOCO : 250,
            ICHIGO : 350,
            ROSE : 60,
            BASK : 60,
            COCO: 300, 
            COUP: 600
            }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        
        self.prices : Dict[PRODUCTS, pd.Series] = {
            "SPREAD_GIFT": pd.Series()
            }
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()

        for product in PRODUCTS:
            self.past_prices[product] = []
            
        self.starfruit_cache = []
        self.starfruit_dim = 4
        
        self.p_mid = []
        self.p_spread = []
        
        self.timestamp = 0
        self.total_days = 250

    def position(self, product, state : TradingState):
        return state.position.get(product, 0) 
   
    def mid_price(self, product, state : TradingState):
        
        default_price = 0
        
        if default_price == 0:
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
           
        def value_on_positions():
            value = 0
            for product in state.position:
                value += self.value_on_product(product, state)
            return value
           
        new_cash()
        return self.cash + value_on_positions()
    
    def values_extract(self, order_dict, buy = 0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_ameth(self, state: TradingState, acc_bid, acc_ask) -> List[Order]:
        
        orders: list[Order] = []
        
        sell_orders, buy_orders, vol_buy, vol_sell, best_sell, best_buy = {}, {}, {}, {}, {}, {}
        
        sorted_sell_orders = sorted(state.order_depths[AMET].sell_orders.items())
        sorted_buy_orders = sorted(state.order_depths[AMET].buy_orders.items(), reverse=True)
            
        sell_orders = {price: vol for price, vol in sorted_sell_orders}
        buy_orders = {price: vol for price, vol in sorted_buy_orders}
            
        vol_sell, best_sell = self.values_extract(sell_orders)
        vol_buy, best_buy = self.values_extract(buy_orders, 1)
        
        current_position = self.position(AMET, state)
        
        mx_with_buy = -1
        
        for ask, vol in sell_orders.items():
            if ((ask < acc_bid) or ((self.position(AMET, state) < 0) and (ask == acc_bid))) and current_position < self.position_limit[AMET]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.position_limit[AMET] - current_position)
                current_position += order_for
                assert(order_for >= 0)
                orders.append(Order(AMET, ask, order_for))
        
        undercut_buy = best_buy + 1
        undercut_sell = best_sell - 1
        
        bid_pr = min(undercut_buy, acc_bid - 1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)
        
        if (current_position < self.position_limit[AMET]) and (self.position(AMET, state) < 0):
            num = min(20, self.position_limit[AMET] - current_position)
            orders.append(Order(AMET, min(undercut_buy + 1, acc_bid - 1), num))
            current_position += num

        if (current_position < self.position_limit[AMET]) and (self.position(AMET, state) > 15):
            num = min(20, self.position_limit[AMET] - current_position)
            orders.append(Order(AMET, min(undercut_buy - 1, acc_bid - 1), num))
            current_position += num

        if current_position < self.position_limit[AMET]:
            num = min(20, self.position_limit[AMET] - current_position)
            orders.append(Order(AMET, bid_pr, num))
            current_position += num
        
        current_position = self.position(AMET, state)

        for bid, vol in buy_orders.items():
            if ((bid > acc_ask) or ((self.position(AMET, state) > 0) and (bid == acc_ask))) and current_position > -self.position_limit[AMET]:
                order_for = max(-vol, -self.position_limit[AMET]-current_position)
                # order_for is a negative number denoting how much we will sell
                current_position += order_for
                assert(order_for <= 0)
                orders.append(Order(AMET, bid, order_for))

        if (current_position > -self.position_limit[AMET]) and (self.position(AMET, state) > 0):
            num = max(-20, -self.position_limit[AMET]-current_position)
            orders.append(Order(AMET, max(undercut_sell-1, acc_ask+1), num))
            current_position += num

        if (current_position > -self.position_limit[AMET]) and (self.position(AMET, state) < -15):
            num = max(-20, -self.position_limit[AMET]-current_position)
            orders.append(Order(AMET, max(undercut_sell+1, acc_ask+1), num))
            current_position += num

        if current_position > -self.position_limit[AMET]:
            num = max(-20, -self.position_limit[AMET]-current_position)
            orders.append(Order(AMET, sell_pr, num))
            current_position += num

        return orders

    def calc_next_price_starfruit(self):
        if len(self.starfruit_cache) != 4:
            logger.print("Error: Starfruit cache does not contain exactly 4 elements.")
            return None

        # If the length is correct, proceed with the calculation
        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]  # Adjusted coefficients
        intercept = 4.481696494462085  # Adjusted intercept
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def compute_orders_starfruit(self, state: TradingState, acc_bid, acc_ask) -> List[Order]:
        
        orders: list[Order] = []
        sell_orders, buy_orders, vol_buy, vol_sell, best_sell, best_buy = {}, {}, {}, {}, {}, {}
    
        sorted_sell_orders = sorted(state.order_depths[STAR].sell_orders.items())
        sorted_buy_orders = sorted(state.order_depths[STAR].buy_orders.items(), reverse=True)
            
        sell_orders = {price: vol for price, vol in sorted_sell_orders}
        buy_orders = {price: vol for price, vol in sorted_buy_orders}
            
        vol_sell, best_sell = self.values_extract(sell_orders)
        vol_buy, best_buy = self.values_extract(buy_orders, 1)
                
        current_position = self.position(STAR, state)
         
        for ask, vol in sell_orders.items():
            if ((ask <= acc_bid) or ((self.position(STAR, state) < 0) and (ask == acc_bid+1))) and current_position < self.position_limit[STAR]:
                order_for = min(-vol, self.position_limit[STAR] - current_position)
                current_position += order_for
                assert(order_for >= 0)
                orders.append(Order(STAR, ask, order_for)) 
        
        undercut_buy = best_buy + 1
        undercut_sell = best_sell - 1
        
        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)
        
        if current_position < self.position_limit[STAR]:
            num = self.position_limit[STAR] - current_position
            orders.append(Order(STAR, bid_pr, num))
            current_position += num
        
        current_position = self.position(STAR, state)
        
        for bid, vol in buy_orders.items():
            if ((bid >= acc_ask) or ((self.position(STAR, state)>0) and (bid+1 == acc_ask))) and current_position > -self.position_limit[STAR]:
                order_for = max(-vol, -self.position_limit[STAR]-current_position)
                # order_for is a negative number denoting how much we will sell
                current_position += order_for
                assert(order_for <= 0)
                orders.append(Order(STAR, bid, order_for))

        if current_position > -self.position_limit[STAR]:
            num = -self.position_limit[STAR]-current_position
            orders.append(Order(STAR, sell_pr, num))
            current_position += num
        
        return orders
     
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
    
    def normal_cdf(self, d_2: float, state: TradingState) -> float:
        return (1.0 + erf(d_2/sqrt(2.0)))/2.0

    def black_scholes(self, S0: float, K: float, T: float, r: float, sigma: float, state: TradingState) -> float:
         """Calculate the Black-Scholes option pricing."""
         if S0 <= 0 or K <= 0:
            return 0  # Return a default value or handle as appropriate
         if T <= 0 or sigma <= 0:
            return 0  # Return a default value or handle as appropriate
    
         d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
         d2 = d1 - sigma * sqrt(T)
         
         return S0 * self.normal_cdf(d1, state) - K * exp(-r * T) * self.normal_cdf(d2, state)
   
    def implied_volatility(self, state: TradingState, S, K, T, r, market_price, tol=1e-6, max_iterations=1000)-> float:
        sigma_low = 0.0001
        sigma_high = 2.0
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high)/2
            
            price_mid = self.black_scholes(S, K, T, r, sigma_mid, state)
            
            if abs(price_mid - market_price) < tol:
                return sigma_mid
            elif price_mid > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        return (sigma_low + sigma_high) / 2
     
    def update_day(self, state: TradingState):
         """Call this method at every 1,000,000 timestamps to decrement the day."""
         self.total_days -= 1
    
    def compute_orders_coco(self, state) -> List[Order]:
       """Calculate orders for coconut and coconut coupons based on Black-Scholes pricing and market conditions."""
       sell_orders, buy_orders, vol_buy, vol_sell, best_sell, best_buy, worst_sell, worst_buy = {}, {}, {}, {}, {}, {}, {}, {}
       products = [COCO, COUP]
       orders_coco = []
       orders_coup = []
       current_price_coconut = int(self.mid_price(COCO, state))
       current_price_coupon = int(self.mid_price(COUP, state))
       
       self.timestamp += 1
       if self.timestamp % 1000000 == 0:  # Check if a 'day' has passed every 1,000,000 timestamps
           self.update_day(state)
            
       r = 0 # risk-free rate, made zero
       #logger.print("Days until exp: {}".format(self.total_days))
       T = 247/365  # normalize by the number of trading days per year 
       strike_price = 10000
       
       if self.timestamp == 1:
           sigma = self.implied_volatility(state, DEFAULT_PRICES[COCO],strike_price, T, r, DEFAULT_PRICES[COUP])
       else:
           sigma = self.implied_volatility(state, current_price_coconut,strike_price, T, r, current_price_coupon)
           
       logger.print("Implied Volatility:{}".format(sigma))
       
       price_coupon = round(self.black_scholes(current_price_coconut, strike_price, T, r, sigma, state))
       logger.print("Coupon Price:{}".format(price_coupon))
       
       for product in products:
           sorted_sell_orders = sorted(state.order_depths[product].sell_orders.items())
           sorted_buy_orders = sorted(state.order_depths[product].buy_orders.items(), reverse=True)
           
           sell_orders[product] = {price: vol for price, vol in sorted_sell_orders}
           buy_orders[product] = {price: vol for price, vol in sorted_buy_orders}
           
           best_sell[product] = next(iter(sell_orders[product]))
           best_buy[product] = next(iter(buy_orders[product]))
           
           worst_sell[product] = next(reversed(sell_orders[product]))
           worst_buy[product] = next(reversed(buy_orders[product]))
           
           vol_buy[product], vol_sell[product] = 0, 0
              
           # Iterate over the buy orders
           for price, vol in buy_orders[product].items():
               vol_buy[product] += vol 
               if vol_buy[product] >= self.position_limit[product]/10:
                   break

           # Iterate over the sell orders
           for price, vol in buy_orders[product].items():
               vol_sell[product] += -vol  # Note: Ensure vol is subtracted correctly
               if vol_sell[product] >= self.position_limit[product]/10:
                   break
           
       logger.print("Volumes to buy and sell: {}, {}".format(vol_buy, vol_sell))
       logger.print("Best to buy and sell: {}, {}".format(best_buy, best_sell))
       
       buy_quantity_coupon = math.floor(min(self.position_limit[COUP] - self.position(COUP, state), vol_buy[COUP]))
       sell_quantity_coupon = math.ceil(min(self.position(COUP, state) + self.position_limit[COUP], -vol_sell[COUP]))
       buy_quantity_coconut = math.floor(min(self.position_limit[COCO] - self.position(COCO, state), vol_buy[COCO]))
       sell_quantity_coconut = math.ceil(min(self.position_limit[COCO] + self.position(COCO, state), -vol_sell[COCO]))
       '''
       if price_coupon + strike_price < current_price_coconut:
           if price_coupon <= worst_sell[COUP] and buy_quantity_coupon > 0:
               orders_coup.append(Order(COUP, price_coupon, buy_quantity_coupon))
               logger.print(f"Placed BUY order for COUPON at {price_coupon} for {buy_quantity_coupon} units.")

           if current_price_coconut >= worst_buy[COCO] and sell_quantity_coconut > 0:
               orders_coco.append(Order(COCO, current_price_coconut, -sell_quantity_coconut))
               logger.print(f"Placed SELL order for COCONUT at {current_price_coconut} for {sell_quantity_coconut} units.")

       elif price_coupon + strike_price > current_price_coconut:
           if price_coupon >= worst_sell[COUP] and sell_quantity_coupon > 0:
               orders_coup.append(Order(COUP, price_coupon, -sell_quantity_coupon))
               logger.print(f"Placed SELL order for COUPON at {price_coupon} for {sell_quantity_coupon} units.")

           if current_price_coconut <= worst_buy[COCO] and buy_quantity_coconut > 0:
               orders_coco.append(Order(COCO, current_price_coconut, buy_quantity_coconut))
               logger.print(f"Placed BUY order for COCONUT at {current_price_coconut} for {buy_quantity_coconut} units.")
               '''
       
       if price_coupon + strike_price < current_price_coconut:
           buy_quantity_coupon = math.floor(min(self.position_limit[COUP] - self.position(COUP, state), vol_buy[COUP]))
           sell_quantity_coconut = math.ceil(min(self.position_limit[COCO] + self.position(COCO, state()), -vol_sell[COCO]))
           if buy_quantity_coupon > 0:
               orders_coup.append(Order(COUP, price_coupon, buy_quantity_coupon))
           if sell_quantity_coconut > 0:
               orders_coco.append(Order(COCO, current_price_coconut, -sell_quantity_coconut))
               
       elif price_coupon + strike_price > current_price_coconut:
           sell_quantity_coupon = math.ceil(min(self.position_limit[COUP] + self.position(COUP, state), -vol_sell[COUP]))
           buy_quantity_coconut = math.floor(min(self.position_limit[COCO] - self.position(COCO, state), vol_buy[COCO]))
           if sell_quantity_coupon > 0:
               orders_coup.append(Order(COUP, price_coupon, -sell_quantity_coupon))
           if buy_quantity_coconut > 0:
               orders_coco.append(Order(COCO, current_price_coconut, buy_quantity_coconut))
             
       return orders_coco, orders_coup

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = "Little_Trader"
        result = {AMET : [], STAR : [], BASK : [], CHOCO : [], ICHIGO : [], ROSE : [], COCO : [], COUP : []} 
        
        self.round += 1
        pnl = self.pnl(state)
        
        logger.print(f"Log round {self.round}")
        
        logger.print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    logger.print(trade)

        logger.print(f"\tCash {self.cash}")
        
        for product in PRODUCTS:
            logger.print(f"\tProduct {product}, Position {self.position(product, state)}, Midprice {self.mid_price(product, state)}, Value {self.value_on_product(product, state)}")
            
        logger.print(f"\tPnL {pnl}")
       
        #Starfruit part
        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)
            
        self.starfruit_cache.append(self.mid_price(STAR, state))
    
        starfruit_lb = -INF
        starfruit_ub = INF
        
        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1
        
        amethyst_lb = 10000
        amethyst_ub = 10000

        # CHANGE FROM HERE

        acc_bid = {AMET : amethyst_lb, STAR : starfruit_lb} 
        acc_ask = {AMET : amethyst_ub, STAR : starfruit_ub}

        try:  
            result[STAR] = self.compute_orders_starfruit(state, acc_bid[STAR], acc_ask[STAR])
        except Exception as e:
            logger.print(e)
           
        try:    
            result[AMET] = self.compute_orders_ameth(state, acc_bid[AMET], acc_ask[AMET])
        except Exception as e:
            logger.print(e)
        
        try:             
            result[BASK], _, _, _ = self.compute_orders_gift(state)
        except Exception as e:
            logger.print(e)
        
        try:             
            result[COCO], result[COUP] = self.compute_orders_coco(state)
        except Exception as e:
            logger.print(e)
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data