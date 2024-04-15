from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Tuple
import string
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
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

ORC = 'ORCHIDS'

UBMIT = "SUBMISSION"
AMET = "AMETHYSTS"
STAR = "STARFRUIT"
ORC = 'ORCHIDS'

PRODUCTS = [
    AMET,
    STAR,
    ORC
]

DEFAULT_PRICES = {
    AMET : 10000,
    STAR : 5000,
    ORC : 1100
}

SUNLIGHT_THRESHOLD = 2500 
SUNLIGHT_HOURS_EQUIVALENT = 7 * 60
DECREASE_FOR_EACH_10MIN_BELOW_THRESHOLD = 0.04  
HUMIDITY_IDEAL_RANGE = (60, 80)  
DECREASE_FOR_EACH_5PP_OUTSIDE_RANGE = 0.02 
STORAGE_COST = 0.1

class Trader:
    def __init__(self) -> None:

        self.position_limit = {
            AMET : 20,
            STAR : 20,
            ORC : 100
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
        
        self.p_mid = []
        self.p_spread = []
        
        

    def position(self, product, state : TradingState):
        return state.position.get(product, 0) 
        
    def calculate_sunlight_effect(self, sunlight):
        if sunlight >= SUNLIGHT_THRESHOLD:
            return 0
        minutes_below = (SUNLIGHT_THRESHOLD - sunlight) / (SUNLIGHT_THRESHOLD / SUNLIGHT_HOURS_EQUIVALENT)
        ten_min_segments = minutes_below / 10
        return ten_min_segments * DECREASE_FOR_EACH_10MIN_BELOW_THRESHOLD
    
    def calculate_humidity_effect(self, humidity):
        if HUMIDITY_IDEAL_RANGE[0] <= humidity <= HUMIDITY_IDEAL_RANGE[1]:
            return 0
        if humidity < HUMIDITY_IDEAL_RANGE[0]:
            five_segments = (HUMIDITY_IDEAL_RANGE[0] - humidity) / 5
        else:
            five_segments = (humidity - HUMIDITY_IDEAL_RANGE[1]) / 5
        return five_segments * DECREASE_FOR_EACH_5PP_OUTSIDE_RANGE
    
    def compute_orders_orchids(self, state: TradingState) -> (List[Order], int):
        # Access the ConversionObservation for ORCHIDS
        orchids_data = state.observations.conversionObservations[ORC]

        bidPrice = orchids_data.bidPrice
        askPrice = orchids_data.askPrice
        transportFees = orchids_data.transportFees
        exportTariff = orchids_data.exportTariff
        importTariff = orchids_data.importTariff
        sunlight = orchids_data.sunlight
        humidity = orchids_data.humidity
        order_depth = state.order_depths[ORC]

        bid_volume = math.floor(sum(quantity for price, quantity in order_depth.buy_orders.items() if price <= bidPrice))
        ask_volume = math.ceil(sum(quantity for price, quantity in order_depth.sell_orders.items() if price >= askPrice))
    
        sunlight_effect = self.calculate_sunlight_effect(sunlight)
        humidity_effect = self.calculate_humidity_effect(humidity)
        
        total_decrease = sunlight_effect + humidity_effect
        potential_price_increase = bidPrice * (1 + total_decrease)
    
        shipping_costs = transportFees + max(exportTariff, importTariff)
    
        arbitrage_opportunity = potential_price_increase - askPrice
        
        # Position check
        position_orchids = self.position(ORC, state)
        orders = []
        conversions = 0

   # Decide buy or sell based on arbitrage opportunity and market volumes
        if arbitrage_opportunity > 0:
            # Buy if there is a net arbitrage opportunity and room to buy within position limits
            buy_volume = min(self.position_limit[ORC] - position_orchids, ask_volume)
            if buy_volume > 0:
                orders.append(Order(ORC, askPrice, buy_volume))
                conversions += buy_volume

    # Consider selling if holding ORCHIDS and the selling conditions are favorable
        if position_orchids > 0:
            # Only sell if effective sell price after storage costs is still profitable
            effective_sell_price = bidPrice - (sunlight_effect + humidity_effect) * bidPrice / 100 - shipping_costs
            sell_volume = min(position_orchids, bid_volume)
            if effective_sell_price > askPrice and sell_volume > 0:  # Ensure selling price is above the asking price to make a profit
                orders.append(Order(ORC, effective_sell_price, -sell_volume))
                conversions += sell_volume

        return orders, conversions
    
    def run(self, state: TradingState) -> Tuple[Dict[str, int], int, str]:
        orders = []
        conversions = 0
        trader_data = "Orchid_Trader"
        result = {ORC : []} 
        
        for product in PRODUCTS:
            order_depth: OrderDepth = state.order_depths[product]
            if product == ORC:   
                orders, conversions_orchids = self.compute_orders_orchids(state)
                result[ORC] += orders
                conversions += conversions_orchids
                
        logger.print("Converstions at current timestamp: {}".format(conversions))
                    
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data