"""
Market Data Collector
Tracks and stores simulation metrics for analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .model import MarketModel


class MarketDataCollector:
    """
    Collects time-series and cross-sectional data during simulation

    Tracks:
    - Market metrics (prices, volumes, volatility)
    - Agent metrics (portfolios, returns, positions)
    - Aggregate metrics (sentiment, herding indices)
    """

    def __init__(self, model: 'MarketModel'):
        """
        Initialize data collector

        Args:
            model: Parent market model
        """
        self.model = model

        # Time-series storage (list of dicts, one per time step)
        self.market_data = []
        self.agent_data = []

        # Aggregate metrics
        self.summary_stats = {}

    def collect(self, model: 'MarketModel'):
        """
        Collect data for current time step

        Args:
            model: Market model instance
        """
        day = model.current_day

        # Collect market metrics
        market_metrics = self._collect_market_metrics(model)
        market_metrics['day'] = day
        self.market_data.append(market_metrics)

        # Collect agent metrics if configured
        if model.config['data_collection']['track_individual_agents']:
            agent_metrics = self._collect_agent_metrics(model)
            agent_metrics['day'] = day
            self.agent_data.append(agent_metrics)

    def _collect_market_metrics(self, model: 'MarketModel') -> Dict[str, Any]:
        """
        Collect market-level metrics

        Returns:
            Dictionary of market metrics
        """
        metrics = {}

        # Prices (to be implemented with market module)
        if model.market:
            for asset in model.market.assets:
                metrics[f'price_{asset.ticker}'] = asset.price
                metrics[f'volume_{asset.ticker}'] = asset.daily_volume
                metrics[f'volatility_{asset.ticker}'] = asset.realized_volatility

        return metrics

    def _collect_agent_metrics(self, model: 'MarketModel') -> Dict[str, Any]:
        """
        Collect agent-level metrics

        Returns:
            Dictionary of aggregate agent metrics
        """
        metrics = {}

        if not model.agents:
            return metrics

        # Aggregate by agent type
        by_type = defaultdict(list)
        for agent in model.agents:
            if agent.active:
                by_type[agent.agent_type].append(agent)

        # Calculate metrics by type
        for agent_type, agents in by_type.items():
            portfolios = [a.portfolio_value for a in agents]
            metrics[f'avg_portfolio_{agent_type}'] = np.mean(portfolios)
            metrics[f'count_{agent_type}'] = len(agents)

        # Overall market metrics
        all_portfolios = [a.portfolio_value for a in model.agents if a.active]
        if all_portfolios:
            metrics['total_market_value'] = sum(all_portfolios)
            metrics['avg_portfolio_value'] = np.mean(all_portfolios)

        return metrics

    def get_time_series(self) -> pd.DataFrame:
        """
        Get market time series as pandas DataFrame

        Returns:
            DataFrame with market metrics over time
        """
        return pd.DataFrame(self.market_data)

    def get_agent_time_series(self) -> pd.DataFrame:
        """
        Get agent time series as pandas DataFrame

        Returns:
            DataFrame with agent metrics over time
        """
        return pd.DataFrame(self.agent_data)

    def calculate_returns(self, asset_ticker: str) -> pd.Series:
        """
        Calculate returns for specific asset

        Args:
            asset_ticker: Asset ticker symbol

        Returns:
            Series of daily returns
        """
        df = self.get_time_series()
        price_col = f'price_{asset_ticker}'

        if price_col not in df.columns:
            return pd.Series()

        prices = df[price_col]
        returns = prices.pct_change().dropna()
        return returns

    def calculate_volatility(self, asset_ticker: str, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility

        Args:
            asset_ticker: Asset ticker symbol
            window: Rolling window size

        Returns:
            Series of rolling volatility
        """
        returns = self.calculate_returns(asset_ticker)
        if len(returns) == 0:
            return pd.Series()

        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility

    def get_stylized_facts(self) -> Dict[str, float]:
        """
        Calculate stylized facts for validation

        Returns:
            Dictionary of empirical metrics
        """
        facts = {}

        df = self.get_time_series()
        if len(df) < 50:  # Need sufficient data
            return facts

        # For each asset, calculate stylized facts
        for asset_config in self.model.config['market']['assets']:
            ticker = asset_config['ticker']
            returns = self.calculate_returns(ticker)

            if len(returns) < 50:
                continue

            # Return distribution properties
            facts[f'{ticker}_return_mean'] = returns.mean()
            facts[f'{ticker}_return_std'] = returns.std()
            facts[f'{ticker}_return_skew'] = returns.skew()
            facts[f'{ticker}_return_kurtosis'] = returns.kurtosis()

            # Volatility clustering (autocorrelation of squared returns)
            squared_returns = returns ** 2
            facts[f'{ticker}_vol_clustering'] = squared_returns.autocorr(lag=1)

        return facts

    def summary(self) -> str:
        """
        Generate summary statistics report

        Returns:
            Formatted summary string
        """
        df = self.get_time_series()

        if len(df) == 0:
            return "No data collected yet"

        summary = []
        summary.append(f"Simulation Days: {len(df)}")
        summary.append(f"\\nMarket Metrics:")

        # Asset summaries
        for asset_config in self.model.config['market']['assets']:
            ticker = asset_config['ticker']
            price_col = f'price_{ticker}'

            if price_col in df.columns:
                initial_price = df[price_col].iloc[0]
                final_price = df[price_col].iloc[-1]
                total_return = (final_price / initial_price - 1) * 100

                summary.append(f"  {ticker}:")
                summary.append(f"    Return: {total_return:.2f}%")
                summary.append(f"    Final Price: ${final_price:.2f}")

        return "\\n".join(summary)
