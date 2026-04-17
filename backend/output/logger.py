"""
RunLogger — converts a completed RunResult into aligned NumPy arrays for the
aggregator.  All runs in a batch log the same set of ticks (determined by the
granularity setting and shock onset positions), so arrays can be stacked
cross-run without alignment issues.

Spec references: §11 (output architecture).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from backend.config.sector_config import NUM_SECTORS, AGENT_TYPES
from backend.models.results import RunResult


@dataclass
class RunArrays:
    """
    Numeric arrays extracted from one RunResult — the format consumed by
    the aggregator.  All arrays share the same tick axis (axis 0).

    Attributes
    ----------
    ticks : np.ndarray, shape (T,)
        Tick indices that were logged.
    prices : np.ndarray, shape (T, NUM_SECTORS)
    volume : np.ndarray, shape (T, NUM_SECTORS)
    volatility : np.ndarray, shape (T, NUM_SECTORS)
    excess_demand : np.ndarray, shape (T, NUM_SECTORS)
    shock_active : np.ndarray, shape (T,), bool
    activity_ratio : np.ndarray, shape (T, num_types)
        Fraction of each agent type that was active per logged tick.
    net_direction : np.ndarray, shape (T, num_types, NUM_SECTORS)
        Net signed magnitude traded by each type per sector per logged tick.
    """
    ticks: np.ndarray
    prices: np.ndarray
    volume: np.ndarray
    volatility: np.ndarray
    excess_demand: np.ndarray
    shock_active: np.ndarray
    activity_ratio: np.ndarray
    net_direction: np.ndarray
    run_number: int
    final_prices: np.ndarray
    final_mispricing: np.ndarray


def extract_run_arrays(result: RunResult) -> RunArrays:
    """
    Convert a RunResult into RunArrays.  The RunResult's tick_snapshots and
    type_summaries must cover the same set of logged ticks (guaranteed when
    both are produced by the same SimulationRun instance).
    """
    snaps = result.tick_snapshots
    summaries = result.type_summaries
    T = len(snaps)

    ticks = np.array([s.tick for s in snaps], dtype=int)
    prices = np.array([s.prices for s in snaps], dtype=float)
    volume = np.array([s.volume for s in snaps], dtype=float)
    volatility = np.array([s.volatility for s in snaps], dtype=float)
    excess_demand = np.array([s.excess_demand for s in snaps], dtype=float)
    shock_active = np.array([s.shock_active for s in snaps], dtype=bool)

    num_types = len(AGENT_TYPES)
    type_index = {t: i for i, t in enumerate(AGENT_TYPES)}

    activity_ratio = np.zeros((T, num_types), dtype=float)
    net_direction = np.zeros((T, num_types, NUM_SECTORS), dtype=float)

    for summary in summaries:
        # Find the position of this tick in the logged ticks array.
        tick_positions = np.where(ticks == summary.tick)[0]
        if len(tick_positions) == 0:
            continue
        t_idx = int(tick_positions[0])
        type_idx = type_index.get(summary.agent_type)
        if type_idx is None:
            continue
        activity_ratio[t_idx, type_idx] = summary.activity_ratio
        net_direction[t_idx, type_idx, :] = summary.net_direction

    return RunArrays(
        ticks=ticks,
        prices=prices,
        volume=volume,
        volatility=volatility,
        excess_demand=excess_demand,
        shock_active=shock_active,
        activity_ratio=activity_ratio,
        net_direction=net_direction,
        run_number=result.run_number,
        final_prices=np.array(result.final_prices, dtype=float),
        final_mispricing=np.array(result.final_mispricing, dtype=float),
    )
