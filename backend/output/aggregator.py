"""
Aggregator — computes cross-run statistics from a list of RunResults.

Produces an AggregatedStats object (price trajectories, mispricing,
type activity) and serialises a compact JSON summary for the LLM
narrative call input.

Spec references: §11 (output architecture, §9.5 (narrative input).
"""

from __future__ import annotations

import json
import numpy as np

from backend.config.sector_config import NUM_SECTORS, SECTORS, AGENT_TYPES
from backend.models.results import RunResult, AggregatedStats
from backend.output.logger import extract_run_arrays, RunArrays


def aggregate_runs(
    results: list[RunResult],
    shocked_sector_indices: list[int] | None = None,
) -> AggregatedStats:
    """
    Compute cross-run aggregate statistics from a completed batch.

    All runs must have been produced with the same CalibrationInput
    (identical granularity → identical logged ticks).

    Returns
    -------
    AggregatedStats
        Mean/median/p10/p90 price and mispricing trajectories per sector,
        plus type activity summary and the compact narrative input JSON.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty list of RunResults.")

    arrays = [extract_run_arrays(r) for r in results]

    # Verify all runs logged the same ticks (should always be true for a batch).
    ref_ticks = arrays[0].ticks
    for arr in arrays[1:]:
        if not np.array_equal(arr.ticks, ref_ticks):
            raise ValueError(
                "Run results have mismatched logged tick sequences — "
                "cannot aggregate."
            )

    # Stack prices across runs: shape (num_runs, T, NUM_SECTORS).
    prices_stack = np.stack([a.prices for a in arrays], axis=0)
    mispricing_stack = (prices_stack - 100.0) / 100.0

    mean_prices = _to_sector_dict(np.mean(prices_stack, axis=0))
    median_prices = _to_sector_dict(np.median(prices_stack, axis=0))
    p10_prices = _to_sector_dict(np.percentile(prices_stack, 10, axis=0))
    p90_prices = _to_sector_dict(np.percentile(prices_stack, 90, axis=0))

    mean_mispricing = _to_sector_dict(np.mean(mispricing_stack, axis=0))
    p10_mispricing = _to_sector_dict(np.percentile(mispricing_stack, 10, axis=0))
    p90_mispricing = _to_sector_dict(np.percentile(mispricing_stack, 90, axis=0))

    # Type activity: mean activity ratio and net direction across runs.
    activity_stack = np.stack([a.activity_ratio for a in arrays], axis=0)
    type_activity_summary = _build_type_activity_summary(activity_stack)

    # Compact narrative input.
    narrative_json = _build_narrative_input(
        arrays=arrays,
        results=results,
        mean_prices=mean_prices,
        mean_mispricing=mean_mispricing,
        type_activity_summary=type_activity_summary,
        shocked_sector_indices=shocked_sector_indices or [],
    )

    return AggregatedStats(
        num_runs=len(results),
        sectors=list(SECTORS),
        mean_prices=mean_prices,
        median_prices=median_prices,
        p10_prices=p10_prices,
        p90_prices=p90_prices,
        mean_mispricing=mean_mispricing,
        p10_mispricing=p10_mispricing,
        p90_mispricing=p90_mispricing,
        type_activity_summary=type_activity_summary,
        narrative_input_json=narrative_json,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_sector_dict(arr_T_K: np.ndarray) -> dict[str, list[float]]:
    """
    Convert a (T, NUM_SECTORS) array to a dict keyed by sector name.
    Each value is a list of floats over logged ticks.
    """
    return {
        sector: arr_T_K[:, k].tolist()
        for k, sector in enumerate(SECTORS)
    }


def _build_type_activity_summary(
    activity_stack: np.ndarray,  # shape (num_runs, T, num_types)
) -> dict:
    """
    Summarise agent type activity across runs and ticks.
    Returns a dict with mean, max activity ratio per type.
    """
    summary = {}
    for i, atype in enumerate(AGENT_TYPES):
        type_activity = activity_stack[:, :, i]  # (num_runs, T)
        summary[atype] = {
            "mean_activity_ratio": float(np.mean(type_activity)),
            "max_activity_ratio": float(np.max(type_activity)),
            "std_activity_ratio": float(np.std(type_activity)),
        }
    return summary


def _build_narrative_input(
    arrays: list[RunArrays],
    results: list[RunResult],
    mean_prices: dict[str, list[float]],
    mean_mispricing: dict[str, list[float]],
    type_activity_summary: dict,
    shocked_sector_indices: list[int],
) -> str:
    """
    Build a compact JSON string suitable as input to the LLM narrative call.

    Contains: final mispricing summary, shock-window dynamics, type activity
    with directional breakdown, and trace flavour.
    """
    # Final mispricing statistics across runs.
    final_misps = np.stack([a.final_mispricing for a in arrays], axis=0)
    final_mispricing_summary = {
        sector: {
            "mean": round(float(np.mean(final_misps[:, k])), 4),
            "std": round(float(np.std(final_misps[:, k])), 4),
            "p10": round(float(np.percentile(final_misps[:, k], 10)), 4),
            "p90": round(float(np.percentile(final_misps[:, k], 90)), 4),
        }
        for k, sector in enumerate(SECTORS)
    }

    payload: dict = {
        "num_runs": len(arrays),
        "sectors": list(SECTORS),
        "final_mispricing": final_mispricing_summary,
        "type_activity": {
            atype: {
                "mean_activity": round(v["mean_activity_ratio"], 3),
                "max_activity": round(v["max_activity_ratio"], 3),
            }
            for atype, v in type_activity_summary.items()
        },
    }

    # Shock-window dynamics — only if shock ticks exist.
    shock_dyn = _shock_window_dynamics(arrays)
    if shock_dyn:
        payload["shock_dynamics"] = shock_dyn

    # Type directional behaviour during shock window.
    type_dirs = _type_shock_direction(arrays)
    if type_dirs:
        payload["type_shock_direction"] = type_dirs

    # Trace flavour — compact behavioural texture from agent traces.
    trace_fl = _trace_flavour(results, shocked_sector_indices)
    if trace_fl:
        payload["trace_flavour"] = trace_fl

    return json.dumps(payload, separators=(",", ":"))


def _shock_window_dynamics(arrays: list[RunArrays]) -> dict:
    """
    Per sector (|peak mispricing| > 1%), return [pre_shock, peak, mid, end]
    signed mispricing values averaged across runs.  Gives the narrative
    the shape of the price path through the shock window.
    """
    # Find shock tick indices (consistent across runs — same batch).
    shock_mask = arrays[0].shock_active  # shape (T,)
    shock_idxs = np.where(shock_mask)[0]
    if len(shock_idxs) == 0:
        return {}

    # Pre-shock: last non-shock tick before first shock tick.
    first_shock = int(shock_idxs[0])
    pre_idx = first_shock - 1 if first_shock > 0 else 0

    mid_idx = int(shock_idxs[len(shock_idxs) // 2])
    last_idx = -1  # final logged tick

    result = {}
    for k, sector in enumerate(SECTORS):
        # Mispricing = (price - 100) / 100 at each reference tick, mean over runs.
        def mean_misp(t_idx: int) -> float:
            vals = [(a.prices[t_idx, k] - 100.0) / 100.0 for a in arrays]
            return float(np.mean(vals))

        peak_misps = np.mean(
            [np.max(np.abs((a.prices[shock_idxs, k] - 100.0) / 100.0)) for a in arrays]
        )
        if peak_misps < 0.01:
            continue  # near-zero dynamics — skip

        # Sign of peak from mean mispricing at peak tick.
        peak_vals = [(a.prices[shock_idxs, k] - 100.0) / 100.0 for a in arrays]
        peak_mean = float(np.mean([p[np.argmax(np.abs(p))] for p in peak_vals]))

        result[sector] = [
            round(mean_misp(pre_idx), 4),
            round(peak_mean, 4),
            round(mean_misp(mid_idx), 4),
            round(mean_misp(last_idx), 4),
        ]

    return result


def _type_shock_direction(arrays: list[RunArrays]) -> dict:
    """
    Per agent type, sectors where net trading direction during shock ticks
    is notable (top relative magnitude).  Returns sign only: "+" or "-".
    Omits types and sectors with negligible activity.
    """
    shock_mask = arrays[0].shock_active
    shock_idxs = np.where(shock_mask)[0]
    if len(shock_idxs) == 0:
        return {}

    # Mean net direction per type per sector during shock, averaged across runs.
    # net_direction shape: (T, num_types, NUM_SECTORS)
    mean_net = np.mean(
        np.stack([a.net_direction[shock_idxs].sum(axis=0) for a in arrays], axis=0),
        axis=0,
    )  # shape: (num_types, NUM_SECTORS)

    result = {}
    for ti, atype in enumerate(AGENT_TYPES):
        type_net = mean_net[ti]  # shape: (NUM_SECTORS,)
        max_abs = float(np.max(np.abs(type_net)))
        if max_abs < 1e-6:
            continue
        # Include sectors where |net| > 15% of this type's max — filters noise.
        threshold = max_abs * 0.15
        sectors_dir = {}
        for k, sector in enumerate(SECTORS):
            if abs(type_net[k]) >= threshold:
                sectors_dir[sector] = "+" if type_net[k] > 0 else "-"
        if sectors_dir:
            result[atype] = sectors_dir

    return result


def _trace_flavour(
    results: list[RunResult],
    shocked_sector_indices: list[int],
) -> dict:
    """
    Compact behavioural texture from agent traces across all runs.
    Per type (only where notable): shock direction, reversal count, coupling flag.
    """
    from collections import defaultdict

    shocked_set = set(shocked_sector_indices)

    # Collect all traces across runs, grouped by agent_type.
    by_type: dict[str, list] = defaultdict(list)
    for run in results:
        for tr in run.agent_traces:
            by_type[tr.agent_type].append(tr)

    result = {}
    for atype in AGENT_TYPES:
        traces = by_type[atype]
        if not traces:
            continue

        shock_trades = [t for t in traces if t.shock_active]
        if not shock_trades:
            continue  # no shock activity — skip this type

        # Net direction of shock-active trades.
        net = sum(t.direction for t in shock_trades)
        shock_dir = "+" if net > 0 else ("-" if net < 0 else None)
        if shock_dir is None:
            continue

        # Reversal count: direction change in same sector, per agent across runs.
        prev: dict[tuple, int] = {}  # (run, agent_id, sector) → last direction
        reversals = 0
        for t in sorted(traces, key=lambda x: (x.agent_id, x.tick)):
            key = (t.agent_id, t.sector_index)
            if key in prev and prev[key] != t.direction:
                reversals += 1
            prev[key] = t.direction

        # Coupling: shock-active trade in a non-shocked sector.
        coupling = any(
            t.shock_active and t.sector_index not in shocked_set
            for t in traces
        )

        entry: dict = {"shock_dir": shock_dir}
        if reversals > 0:
            entry["reversals"] = reversals
        if coupling:
            entry["coupling"] = True
        result[atype] = entry

    return result
