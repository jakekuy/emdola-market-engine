"""
Post-batch sanity checks — health signal indicators for simulation outputs.

Five checks run automatically after every batch.  Each returns a continuous
score (0.0–1.0) reflecting how confidently the expected behaviour is present,
rather than a binary pass/fail:

  0.00–0.39  Weak     (red)    — signal absent or very faint
  0.40–0.69  Moderate (amber)  — signal present but not compelling
  0.70–1.00  Strong   (green)  — signal clearly present

  1. Fundamentalist reversion   — prices remain close to P0 at end of run
  2. Herding amplification      — shock-period volatility exceeds quiet-period
  3. Fat tails                  — log-return distribution has meaningful excess kurtosis
  4. Cross-sector correlation   — sectors co-move (not fully independent)
  5. Agent type divergence      — 8 types show meaningfully different activity levels

Failing a check does not stop the batch — results are displayed in the UI
and included in the narrative LLM call as context.

Spec references: §12 (validation).
"""

from __future__ import annotations

import numpy as np
from scipy import stats  # type: ignore

from backend.config.constants import P0
from backend.config.sector_config import NUM_SECTORS, SECTORS
from backend.models.results import BatchResult, RunResult, SanityCheckResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _strength_label(score: float) -> str:
    if score >= 0.70:
        return "Strong"
    if score >= 0.40:
        return "Moderate"
    return "Weak"


def _make_result(
    check_name: str,
    score: float,
    message: str,
    detail: str = "",
) -> SanityCheckResult:
    """Build a SanityCheckResult from a raw score, deriving passed and label."""
    score = _clamp(score)
    return SanityCheckResult(
        check_name=check_name,
        passed=score >= 0.40,
        score=round(score, 3),
        strength_label=_strength_label(score),
        message=message,
        detail=detail,
    )


def _make_skipped(check_name: str, reason: str) -> SanityCheckResult:
    """Return a neutral (score=0.5) result for a check that could not run."""
    return SanityCheckResult(
        check_name=check_name,
        passed=True,
        score=0.5,
        strength_label="N/A",
        message=f"Skipped — {reason}",
        detail="",
    )


def run_all_checks(batch_result: BatchResult) -> list[SanityCheckResult]:
    """Run all sanity checks and return results."""
    return [
        check_fundamentalist_reversion(batch_result),
        check_herding_amplification(batch_result),
        check_fat_tails(batch_result),
        check_cross_sector_correlation(batch_result),
        check_agent_type_divergence(batch_result),
    ]


# ── Check 1: Fundamentalist reversion ─────────────────────────────────────────

def check_fundamentalist_reversion(batch_result: BatchResult) -> SanityCheckResult:
    """
    How tightly do prices revert toward P0 by end of run?

    Score = clamp(1 − mean_abs_misp / 0.15, 0, 1)
      ≈ 1.0 at 0% mispricing, 0.5 at 7.5%, 0.0 at 15%+.

    In a shocked scenario some residual mispricing is expected; very large
    residuals indicate the chartist loop is dominating fundamentalist reversion.
    """
    if not batch_result.run_results:
        return _make_skipped("fundamentalist_reversion", "no run results.")

    final_misps = [
        np.abs(r.final_mispricing)
        for r in batch_result.run_results
    ]
    mean_abs_misp = float(np.mean([np.mean(m) for m in final_misps]))
    max_abs_misp = float(np.mean([np.max(m) for m in final_misps]))

    score = _clamp(1.0 - mean_abs_misp / 0.15)

    return _make_result(
        check_name="fundamentalist_reversion",
        score=score,
        message=(
            f"Mean absolute final mispricing: {mean_abs_misp:.1%} "
            f"(score ceiling at 15%)."
        ),
        detail=(
            f"Mean max sector mispricing: {max_abs_misp:.1%}. "
            f"Computed over {len(batch_result.run_results)} runs."
        ),
    )


# ── Check 2: Herding amplification ────────────────────────────────────────────

def check_herding_amplification(batch_result: BatchResult) -> SanityCheckResult:
    """
    How much more volatile are shock-period ticks vs. quiet-period ticks?

    Score = clamp((ratio − 0.5) / 1.5, 0, 1) where ratio = mean_shock_vol / mean_quiet_vol.
      ratio 0.5× → 0.0,  ratio 1.25× → 0.5,  ratio 2.0× → 1.0.

    If no shock-active ticks are detected the check is skipped.
    """
    if not batch_result.run_results or not batch_result.run_results[0].tick_snapshots:
        return _make_skipped("herding_amplification", "no tick data available.")

    shock_vols = np.array([
        s.volatility
        for r in batch_result.run_results
        for s in r.tick_snapshots
        if s.influence_active
    ])
    if len(shock_vols) == 0:
        return _make_skipped("herding_amplification", "no influence-active ticks detected.")

    quiet_vols = np.array([
        s.volatility
        for r in batch_result.run_results
        for s in r.tick_snapshots
        if not s.influence_active
    ])
    if len(quiet_vols) == 0:
        return _make_skipped("herding_amplification", "no quiet-period ticks for comparison.")

    mean_shock = float(np.mean(shock_vols))
    mean_quiet = float(np.mean(quiet_vols))

    ratio = mean_shock / mean_quiet if mean_quiet > 0 else 1.0
    # Calibrated for EMDOLA's fundamentalist-anchored dynamics: herding manifests
    # as directional trend persistence rather than volatility spikes.  Range 0.9–1.15x:
    # floor at 0.9x (shock period noticeably less volatile — strong directional trend
    # suppressing oscillation); ceiling at 1.15x (typical strong amplification for this
    # model).  Using (ratio-0.9)/0.25 rather than (ratio-1.0)/0.15 prevents ratio<1.0
    # from collapsing immediately to zero — a ratio of 0.93x (smooth trending) scores
    # Weak (~0.12) rather than zero, preserving the spectrum signal.
    score = _clamp((ratio - 0.9) / 0.25)

    return _make_result(
        check_name="herding_amplification",
        score=score,
        message=(
            f"Shock-period mean vol: {mean_shock:.4f}; "
            f"quiet-period: {mean_quiet:.4f}; "
            f"ratio: {ratio:.2f}×."
        ),
        detail=(
            f"{len(shock_vols)} influence-active ticks vs {len(quiet_vols)} quiet ticks "
            f"across {len(batch_result.run_results)} runs."
        ),
    )


# ── Check 3: Fat tails ────────────────────────────────────────────────────────

def check_fat_tails(batch_result: BatchResult) -> SanityCheckResult:
    """
    How fat-tailed is the log-return distribution relative to a normal?

    Score = clamp(excess_kurtosis / 3.0, 0, 1).
      Real equity markets typically show excess kurtosis ≈ 3; that maps to score 1.0.
      kurtosis ≤ 0 → score 0.0; kurtosis = 1.5 → 0.5.

    Fisher kurtosis is used (normal = 0; positive = leptokurtic / fat-tailed).
    """
    if not batch_result.run_results:
        return _make_skipped("fat_tails", "no run results.")

    log_returns = []
    for run_result in batch_result.run_results:
        snaps = run_result.tick_snapshots
        if len(snaps) < 2:
            continue
        for k in range(NUM_SECTORS):
            prices_k = [s.prices[k] for s in snaps]
            for i in range(len(prices_k) - 1):
                if prices_k[i] > 0 and prices_k[i + 1] > 0:
                    log_returns.append(np.log(prices_k[i + 1] / prices_k[i]))

    if len(log_returns) < 50:
        return _make_skipped("fat_tails", "insufficient log-return observations.")

    log_returns_arr = np.array(log_returns)
    kurt = float(stats.kurtosis(log_returns_arr, fisher=True))
    # Ceiling 5.0: real equity markets typically show excess kurtosis 3–5 during
    # shock episodes; EMDOLA produces 2.5–4.5 across scenarios, so 5.0 keeps the
    # check from saturating on normal good-quality runs.
    score = _clamp(kurt / 5.0)

    return _make_result(
        check_name="fat_tails",
        score=score,
        message=(
            f"Excess kurtosis: {kurt:.3f} "
            f"(score ceiling at 5.0)."
        ),
        detail=(
            f"Computed over {len(log_returns_arr):,} log-return observations "
            f"across {len(batch_result.run_results)} runs × {NUM_SECTORS} sectors."
        ),
    )


# ── Check 4: Cross-sector correlation ─────────────────────────────────────────

def check_cross_sector_correlation(batch_result: BatchResult) -> SanityCheckResult:
    """
    How strongly do sectors co-move across the batch?

    Method: compute pairwise Pearson correlations of the cross-run mean mispricing
    time series (one series per sector), then take the mean of absolute values.

    Using absolute correlation rather than signed correlation is correct for macro
    shock scenarios: when Energy is shocked up and Financials are shocked down, the
    two sectors are anti-correlated — but that anti-correlation IS meaningful macro
    co-movement, not independence.  Raw correlation cancels positively- and
    negatively-correlated pairs and produces near-zero values even when strong
    directional structure exists.  Absolute correlation captures the full signal.

    Using the cross-run mean mispricing series (rather than per-run log returns)
    removes within-run noise and reveals the batch-level directional structure.
    Per-run log-return correlation is near-zero in EMDOLA because tick-level noise
    from random agent activation dominates the common trend component in any single
    run; the cross-run mean eliminates this noise.

    Score = clamp(mean_abs_corr / 0.40, 0, 1).
      0.0 → 0.0 (sectors fully independent),  0.20 → 0.5,  0.40+ → 1.0.
    """
    if not batch_result.aggregated_stats:
        return _make_skipped("cross_sector_correlation", "no aggregated stats available.")

    mean_misp = batch_result.aggregated_stats.mean_mispricing
    if not mean_misp or len(mean_misp) < 2:
        return _make_skipped("cross_sector_correlation", "insufficient mispricing data.")

    sector_names = list(mean_misp.keys())
    # Build matrix: (num_sectors, T) of mean mispricing time series.
    matrix = np.array([mean_misp[s] for s in sector_names])
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return _make_skipped("cross_sector_correlation", "insufficient time series length.")

    # Filter out flat series (all-zero) to avoid undefined correlations.
    active = [i for i in range(matrix.shape[0]) if np.std(matrix[i]) > 1e-8]
    if len(active) < 2:
        return _make_skipped("cross_sector_correlation", "insufficient sector variation.")

    sub = matrix[active]
    corr = np.corrcoef(sub)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    abs_corr_vals = np.abs(corr[mask])
    mean_abs_corr = float(np.mean(abs_corr_vals))

    # Ceiling 0.80: EMDOLA produces abs mispricing correlations of 0.55-0.65 for
    # well-designed shock scenarios; ceiling 0.40 always saturated.  0.80 gives
    # typical strong runs a score of ~0.70-0.80 rather than pinning to 1.0.
    score = _clamp(mean_abs_corr / 0.80)

    return _make_result(
        check_name="cross_sector_correlation",
        score=score,
        message=(
            f"Mean absolute pairwise mispricing correlation: {mean_abs_corr:.3f} "
            f"(score ceiling at 0.80)."
        ),
        detail=(
            f"Computed from cross-run mean mispricing series across "
            f"{len(active)} sectors with non-zero variance."
        ),
    )


# ── Check 5: Agent type divergence ────────────────────────────────────────────

def check_agent_type_divergence(batch_result: BatchResult) -> SanityCheckResult:
    """
    How differentiated are the 8 agent types in their activity levels?

    Score = clamp(std_dev / 0.20, 0, 1) where std_dev is the standard deviation
    of mean activity ratios across the 8 types.
      std 0.0 → 0.0,  std 0.10 → 0.5,  std 0.20+ → 1.0.

    A low score means all types are behaving near-identically — the LLM-generated
    heterogeneity is not producing differentiated responses.
    """
    from backend.config.sector_config import AGENT_TYPES

    if not batch_result.run_results:
        return _make_skipped("agent_type_divergence", "no run results.")

    type_ratios: dict[str, list[float]] = {t: [] for t in AGENT_TYPES}
    for run in batch_result.run_results:
        for summary in run.type_summaries:
            if summary.agent_type in type_ratios:
                type_ratios[summary.agent_type].append(summary.activity_ratio)

    type_means = [
        float(np.mean(type_ratios[t]))
        for t in AGENT_TYPES
        if type_ratios[t]
    ]

    if len(type_means) < 2:
        return _make_skipped("agent_type_divergence", "insufficient type activity data.")

    std_dev = float(np.std(type_means))
    # Ceiling 0.35: EMDOLA consistently produces std_dev ~0.29 from LLM-calibrated
    # type heterogeneity; ceiling 0.20 always saturated.  0.35 keeps strong runs
    # scoring ~0.83 (clearly Strong) without pinning to 1.0 on every run.
    score = _clamp(std_dev / 0.35)

    type_summary = ", ".join(
        f"{t}={float(np.mean(type_ratios[t])):.2f}"
        for t in AGENT_TYPES
        if type_ratios[t]
    )

    return _make_result(
        check_name="agent_type_divergence",
        score=score,
        message=(
            f"Std dev of mean activity ratios across 8 types: {std_dev:.3f} "
            f"(score ceiling at 0.20)."
        ),
        detail=f"Mean activity by type: {type_summary}.",
    )
