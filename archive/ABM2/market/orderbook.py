"""
Order Book and Market Clearing
Handles order matching and price discovery
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass
class Order:
    """Represents a trading order"""
    agent_id: str
    ticker: str
    quantity: int  # Positive = buy, negative = sell
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None  # For limit orders
    timestamp: int = 0


class OrderBook:
    """
    Order book for a single asset

    Maintains buy and sell orders, matches trades
    """

    def __init__(self, ticker: str):
        """
        Initialize order book

        Args:
            ticker: Asset ticker symbol
        """
        self.ticker = ticker

        # Order storage
        self.buy_orders = []  # List of (price, quantity, agent_id)
        self.sell_orders = []  # List of (price, quantity, agent_id)
        self.market_buy_orders = []  # Market orders
        self.market_sell_orders = []

        # Trade history
        self.executed_trades = []
        self.daily_volume = 0

    def add_order(self, order: Order):
        """
        Add order to book

        Args:
            order: Order instance
        """
        if order.order_type == 'market':
            if order.quantity > 0:
                self.market_buy_orders.append(order)
            else:
                self.market_sell_orders.append(order)
        else:  # limit order
            if order.quantity > 0:
                self.buy_orders.append((order.price, order.quantity, order.agent_id))
            else:
                self.sell_orders.append((order.price, abs(order.quantity), order.agent_id))

    def clear_market(self, current_price: float) -> Tuple[float, int, List[Dict]]:
        """
        Clear market and determine new equilibrium price

        Simple clearing mechanism:
        1. Calculate net order flow (buy pressure - sell pressure)
        2. Adjust price based on order imbalance
        3. Match orders at clearing price
        4. Return new price, volume, and executed trades

        Args:
            current_price: Current market price

        Returns:
            Tuple of (new_price, total_volume, executed_trades)
        """
        # Calculate order flow
        total_buy_quantity = sum(q for _, q, _ in self.buy_orders) + \
                           sum(abs(o.quantity) for o in self.market_buy_orders)

        total_sell_quantity = sum(q for _, q, _ in self.sell_orders) + \
                            sum(abs(o.quantity) for o in self.market_sell_orders)

        # Net order flow
        net_flow = total_buy_quantity - total_sell_quantity

        # Price impact (simplified linear impact model)
        # impact = order_imbalance / liquidity
        liquidity = 1000  # Base liquidity parameter
        price_impact = (net_flow / liquidity) * current_price

        # Apply dampening to prevent extreme moves
        max_move = current_price * 0.10  # Max 10% move per day
        price_impact = np.clip(price_impact, -max_move, max_move)

        # New equilibrium price
        new_price = current_price + price_impact

        # Ensure positive price
        new_price = max(new_price, current_price * 0.1)

        # Volume is the min of buy and sell (matched volume)
        volume = min(total_buy_quantity, total_sell_quantity)

        # Execute trades (simplified - match at clearing price)
        executed_trades = self._execute_trades(new_price, volume)

        # Clear order book for next day
        self._clear_orders()

        return new_price, volume, executed_trades

    def _execute_trades(self, price: float, max_volume: int) -> List[Dict]:
        """
        Execute trades at clearing price

        Args:
            price: Clearing price
            max_volume: Maximum volume to trade

        Returns:
            List of executed trade dicts
        """
        executed_trades = []
        remaining_volume = max_volume

        # Match market buy orders
        for order in self.market_buy_orders:
            if remaining_volume <= 0:
                break

            quantity = min(order.quantity, remaining_volume)
            executed_trades.append({
                'agent_id': order.agent_id,
                'ticker': order.ticker,
                'quantity': quantity,
                'price': price,
                'side': 'buy'
            })
            remaining_volume -= quantity

        # Match market sell orders
        for order in self.market_sell_orders:
            if remaining_volume <= 0:
                break

            quantity = min(abs(order.quantity), remaining_volume)
            executed_trades.append({
                'agent_id': order.agent_id,
                'ticker': order.ticker,
                'quantity': -quantity,  # Negative for sells
                'price': price,
                'side': 'sell'
            })
            remaining_volume -= quantity

        # TODO: Match limit orders within price range

        self.executed_trades.extend(executed_trades)
        self.daily_volume += sum(abs(t['quantity']) for t in executed_trades)

        return executed_trades

    def _clear_orders(self):
        """Clear all orders (end of day)"""
        self.buy_orders = []
        self.sell_orders = []
        self.market_buy_orders = []
        self.market_sell_orders = []

    def get_bid_ask_spread(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current bid-ask spread

        Returns:
            Tuple of (best_bid, best_ask)
        """
        best_bid = max((p for p, _, _ in self.buy_orders), default=None)
        best_ask = min((p for p, _, _ in self.sell_orders), default=None)
        return best_bid, best_ask


class Market:
    """
    Multi-asset market with order books and clearing

    Coordinates trading across multiple assets
    """

    def __init__(self, asset_universe):
        """
        Initialize market

        Args:
            asset_universe: AssetUniverse instance
        """
        self.asset_universe = asset_universe
        self.assets = asset_universe.get_all_assets()

        # Create order book for each asset
        self.order_books = {
            asset.ticker: OrderBook(asset.ticker)
            for asset in self.assets
        }

        # Pending orders (submitted but not cleared)
        self.pending_orders = []

    def submit_order(self, order_dict: Dict):
        """
        Submit order to market

        Args:
            order_dict: Order dictionary with agent_id, ticker, quantity, etc.
        """
        order = Order(
            agent_id=order_dict['agent_id'],
            ticker=order_dict['ticker'],
            quantity=order_dict['quantity'],
            order_type=order_dict.get('order_type', 'market'),
            price=order_dict.get('price'),
            timestamp=order_dict.get('timestamp', 0)
        )

        # Add to appropriate order book
        if order.ticker in self.order_books:
            self.order_books[order.ticker].add_order(order)
            self.pending_orders.append(order)

    def clear_orders(self) -> Dict[str, Tuple[float, int]]:
        """
        Clear all markets and determine new prices

        Returns:
            Dict of {ticker: (new_price, volume)}
        """
        price_updates = {}

        # Clear each market
        for ticker, order_book in self.order_books.items():
            asset = self.asset_universe.get_asset(ticker)
            if not asset:
                continue

            # Get current price
            current_price = asset.price

            # Clear market
            new_price, volume, executed_trades = order_book.clear_market(current_price)

            # Record price update
            price_updates[ticker] = (new_price, volume)

            # Execute trades (notify agents)
            self._execute_agent_trades(executed_trades)

        # Update asset prices
        self.asset_universe.update_all_prices(price_updates)

        # Clear pending orders
        self.pending_orders = []

        return price_updates

    def _execute_agent_trades(self, executed_trades: List[Dict]):
        """
        Notify agents of executed trades

        Args:
            executed_trades: List of trade dicts
        """
        # Group trades by agent
        agent_trades = defaultdict(list)
        for trade in executed_trades:
            agent_trades[trade['agent_id']].append(trade)

        # Notify each agent (agent will update portfolio)
        # This will be connected to agents via model reference
        # For now, store for later processing
        pass

    def get_asset(self, ticker: str):
        """
        Get asset by ticker

        Args:
            ticker: Asset ticker

        Returns:
            Asset instance
        """
        return self.asset_universe.get_asset(ticker)

    def get_market_depth(self, ticker: str) -> Dict:
        """
        Get order book depth for asset

        Args:
            ticker: Asset ticker

        Returns:
            Dict with bid/ask levels
        """
        if ticker not in self.order_books:
            return {}

        order_book = self.order_books[ticker]
        best_bid, best_ask = order_book.get_bid_ask_spread()

        return {
            'ticker': ticker,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'buy_orders': len(order_book.buy_orders),
            'sell_orders': len(order_book.sell_orders)
        }

    def get_market_summary(self) -> Dict:
        """
        Get summary of all markets

        Returns:
            Market statistics dict
        """
        summary = self.asset_universe.get_market_summary()

        # Add order book info
        for ticker, order_book in self.order_books.items():
            if ticker in summary:
                summary[ticker]['daily_trades'] = len(order_book.executed_trades)

        return summary

    def __repr__(self):
        return f"Market({len(self.order_books)} assets, {len(self.pending_orders)} pending orders)"
