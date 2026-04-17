"""
Financial Instruments
Asset class definitions and price dynamics
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque


class Asset:
    """
    Represents a tradable asset with price dynamics

    Tracks:
    - Current price
    - Price history
    - Volume
    - Volatility
    """

    def __init__(
        self,
        ticker: str,
        name: str,
        initial_price: float,
        volatility: float,
        asset_category: str = "equity"
    ):
        """
        Initialize asset

        Args:
            ticker: Asset ticker symbol (e.g., 'TECH')
            name: Full name
            initial_price: Starting price
            volatility: Daily volatility (std dev of returns)
            asset_category: Type of asset
        """
        self.ticker = ticker
        self.name = name
        self.asset_category = asset_category

        # Price tracking
        self.price = initial_price
        self.initial_price = initial_price
        self.price_history = [initial_price]

        # Volatility
        self.base_volatility = volatility
        self.realized_volatility = volatility

        # Volume tracking
        self.daily_volume = 0
        self.volume_history = []

        # Order flow
        self.bid_ask_spread = initial_price * 0.001  # 0.1% spread initially
        self.last_trade_price = initial_price

    def update_price(self, new_price: float, volume: int):
        """
        Update asset price after market clearing

        Args:
            new_price: New equilibrium price
            volume: Trading volume
        """
        self.price = new_price
        self.last_trade_price = new_price
        self.daily_volume = volume

        # Record history
        self.price_history.append(new_price)
        self.volume_history.append(volume)

        # Update realized volatility (rolling window)
        if len(self.price_history) > 20:
            returns = np.diff(self.price_history[-21:]) / self.price_history[-21:-1]
            self.realized_volatility = np.std(returns)

    def add_noise_drift(self, drift: float = 0.0):
        """
        Add random walk component to price (for days with no trades)

        Args:
            drift: Expected return (usually 0 or small positive)
        """
        # Random walk with drift
        shock = np.random.normal(drift, self.base_volatility)
        new_price = self.price * (1 + shock)

        # Ensure price stays positive
        new_price = max(new_price, self.initial_price * 0.1)

        self.update_price(new_price, 0)

    def get_returns(self, window: Optional[int] = None) -> np.ndarray:
        """
        Calculate historical returns

        Args:
            window: Number of periods to include (None = all)

        Returns:
            Array of returns
        """
        prices = np.array(self.price_history[-window:] if window else self.price_history)
        if len(prices) < 2:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    def get_total_return(self) -> float:
        """
        Calculate total return since inception

        Returns:
            Total return (%)
        """
        return (self.price / self.initial_price - 1) * 100

    def __repr__(self):
        return f"{self.ticker} (${self.price:.2f}, vol={self.realized_volatility:.3f})"


class AssetUniverse:
    """
    Collection of tradable assets with correlations
    """

    def __init__(self, asset_configs: List[Dict], correlations: Dict):
        """
        Initialize asset universe

        Args:
            asset_configs: List of asset configuration dicts
            correlations: Correlation structure
        """
        self.assets = {}
        self.correlations = correlations

        # Create assets
        for config in asset_configs:
            asset = Asset(
                ticker=config['ticker'],
                name=config['name'],
                initial_price=config['initial_price'],
                volatility=config['volatility']
            )
            self.assets[config['ticker']] = asset

    def get_asset(self, ticker: str) -> Optional[Asset]:
        """
        Get asset by ticker

        Args:
            ticker: Asset ticker symbol

        Returns:
            Asset instance or None
        """
        return self.assets.get(ticker)

    def get_all_assets(self) -> List[Asset]:
        """
        Get all assets

        Returns:
            List of all assets
        """
        return list(self.assets.values())

    def update_all_prices(self, price_updates: Dict[str, tuple]):
        """
        Update prices for all assets after market clearing

        Args:
            price_updates: Dict of {ticker: (new_price, volume)}
        """
        for ticker, (price, volume) in price_updates.items():
            if ticker in self.assets:
                self.assets[ticker].update_price(price, volume)

    def apply_correlated_shocks(self, mean_drift: float = 0.0):
        """
        Apply correlated random shocks to all assets

        Used when market doesn't clear (no trades)

        Args:
            mean_drift: Expected return
        """
        # Generate correlated shocks based on correlation structure
        # Simplified: apply independent shocks for now
        # TODO: Implement proper correlation matrix
        for asset in self.assets.values():
            asset.add_noise_drift(mean_drift)

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Calculate realized correlation matrix from price history

        Returns:
            Correlation matrix
        """
        tickers = list(self.assets.keys())
        n = len(tickers)

        if n == 0:
            return np.array([])

        # Get returns for each asset
        returns_dict = {}
        min_length = float('inf')

        for ticker in tickers:
            returns = self.assets[ticker].get_returns()
            if len(returns) > 0:
                returns_dict[ticker] = returns
                min_length = min(min_length, len(returns))

        if min_length == float('inf') or min_length < 2:
            return np.eye(n)  # Return identity if insufficient data

        # Align returns to same length
        aligned_returns = np.array([
            returns_dict[ticker][-int(min_length):] for ticker in tickers
        ])

        # Calculate correlation
        corr_matrix = np.corrcoef(aligned_returns)
        return corr_matrix

    def get_market_summary(self) -> Dict:
        """
        Get summary statistics for all assets

        Returns:
            Dictionary of market statistics
        """
        summary = {}

        for ticker, asset in self.assets.items():
            summary[ticker] = {
                'price': asset.price,
                'return': asset.get_total_return(),
                'volatility': asset.realized_volatility,
                'volume': asset.daily_volume
            }

        return summary

    def __repr__(self):
        return f"AssetUniverse({len(self.assets)} assets)"
