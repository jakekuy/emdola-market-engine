"""
Price formation — log-linear excess demand mechanism (spec §4).

    log p(t+1) = log p(t) + λ_k * ED(t) / N

This module is a thin functional layer; the actual state update lives in
MarketEnvironment.apply_price_update().  These helpers are exposed for
unit testing individual components.
"""

from __future__ import annotations

import numpy as np

from backend.config.sector_config import NUM_SECTORS


def compute_log_price_update(
    lambdas: np.ndarray,
    excess_demand: np.ndarray,
    total_agent_count: int,
) -> np.ndarray:
    """
    Compute the log-price change for all sectors this tick.

    Parameters
    ----------
    lambdas :
        Per-sector price sensitivity (shape: NUM_SECTORS).
    excess_demand :
        Net signed trade magnitude per sector this tick (shape: NUM_SECTORS).
    total_agent_count :
        N — normalisation factor.

    Returns
    -------
    np.ndarray
        Δ log p per sector (shape: NUM_SECTORS).
    """
    assert len(lambdas) == NUM_SECTORS
    assert len(excess_demand) == NUM_SECTORS
    N = max(total_agent_count, 1)
    return lambdas * excess_demand / N


def apply_log_price_update(
    log_prices: np.ndarray,
    delta_log_prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a log-price delta and return (new_log_prices, new_prices).
    Prices remain positive because exp() is always > 0.
    """
    new_log_prices = log_prices + delta_log_prices
    new_prices = np.exp(new_log_prices)
    return new_log_prices, new_prices


def compute_mispricing(prices: np.ndarray, p0: float = 100.0) -> np.ndarray:
    """
    Mispricing metric per sector: (price(t) - P0) / P0 (spec §13.4).
    Positive = overpriced vs starting price; negative = underpriced.
    """
    return (prices - p0) / p0
