"""
Signal computation — market channel, influence channel, and sector affinity.

All signal functions operate on a single agent × single sector at a time,
called from the activation function in activation.py.

Spec references: §3, §7.5 Step 1, §7.12.
"""

from __future__ import annotations

import numpy as np
from collections import deque

from backend.config.constants import (
    P0,
    SIGMA_BACKGROUND,
    CHARTIST_SIGNAL_SCALE,
    DEFENSIVE_SECTOR_INDICES,
    HIGH_VOL_SECTOR_INDICES,
    RISK_AVERSION_AFFINITY_SHIFT,
    TIME_HORIZON_AFFINITY_DAMPEN,
    AFFINITY_MIN,
)
from backend.config.sector_config import NUM_SECTORS, BASE_AFFINITY


# ── Fundamentalist signal (spec §7.5 Step 1) ──────────────────────────────────

def compute_fundamentalist_signal(price_k: float, p0_effective: float = P0) -> float:
    """
    F_i(t,k) = (P0_effective − price_k) / P0_effective

    Positive when price is below the current fair-value belief (buy signal).
    Negative when above (sell signal).

    p0_effective is updated each tick by the ShockProcessor: a market-channel
    shock shifts it away from P0=100 to reflect changed economic beliefs.
    In the absence of shocks p0_effective = P0 = 100.
    """
    if p0_effective <= 0:
        return 0.0
    return (p0_effective - price_k) / p0_effective


# ── Chartist signal (spec §7.5 Step 1) ────────────────────────────────────────

def compute_decay_weights(
    memory_ticks: int,
    recency_bias: float,
    is_institutional: bool,
) -> np.ndarray:
    """
    Compute decay weights for the chartist signal memory window.

    Retail (exponential decay): w_j = exp(−j / τ)
        τ = memory_ticks / 3 / (1 + recency_bias)
        Higher recency_bias → smaller τ → steeper decay → more recent-dominant.

    Institutional (power law): w_j = (j+1)^(−α)
        α = 1.0 + recency_bias × 1.5
        Higher recency_bias → higher α → steeper decay.

    Returns normalised weights (sum = 1), most recent first (index 0).
    """
    if memory_ticks <= 0:
        return np.array([1.0])

    indices = np.arange(memory_ticks, dtype=float)  # 0 = most recent

    if not is_institutional:
        # Exponential — retail
        tau = (memory_ticks / 3.0) / max(1.0 + recency_bias, 0.01)
        weights = np.exp(-indices / max(tau, 0.01))
    else:
        # Power law — institutional
        alpha = 1.0 + recency_bias * 1.5
        weights = (indices + 1.0) ** (-alpha)

    total = weights.sum()
    if total <= 0:
        return np.ones(memory_ticks) / memory_ticks
    return weights / total


def compute_chartist_signal(
    price_memory: deque,
    memory_ticks: int,
    recency_bias: float,
    time_horizon: float,
    is_institutional: bool,
    beta_c: float,
) -> float:
    """
    C_i(t,k) = tanh(Σ w_j · Δ log p_k(t-j))

    Sum of recency-weighted log-price changes, bounded by tanh.
    Time horizon attenuates the chartist contribution: multiply by
    (1 − time_horizon × 0.5) so long-horizon agents discount short-term trends.

    Returns a scalar in (−1, +1).  Positive = upward momentum.
    """
    mem = list(price_memory)  # oldest first
    if len(mem) < 2:
        return 0.0

    # Compute log-returns from memory (most recent last in mem list).
    log_returns = [
        np.log(mem[i + 1] / mem[i])
        for i in range(len(mem) - 1)
        if mem[i] > 0 and mem[i + 1] > 0
    ]
    if not log_returns:
        return 0.0

    # Most recent return first for weight alignment.
    log_returns_arr = np.array(log_returns[::-1])  # flip: [0] = most recent
    n = min(len(log_returns_arr), memory_ticks)
    log_returns_arr = log_returns_arr[:n]

    weights = compute_decay_weights(n, recency_bias, is_institutional)[:n]
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    weighted_sum = float(np.dot(weights, log_returns_arr))
    # Scale before tanh so small-but-real price moves register as meaningful
    # momentum signals.  Without amplification, 0.5%/day trends produce C ≈ 0.005
    # (negligible); at 8× they produce C ≈ 0.04–0.2, enabling the chartist
    # feedback loop.  tanh saturation bounds the signal for large shock moves.
    signal = float(np.tanh(weighted_sum * CHARTIST_SIGNAL_SCALE))

    # Time horizon attenuation: longer-horizon agents discount momentum.
    time_attenuation = 1.0 - time_horizon * 0.5
    return signal * time_attenuation


# ── Influence channel signal (spec §7.5 Step 1) ───────────────────────────────

def compute_influence_signal(
    broadcast_direction: float,
    broadcast_intensity: float,
    herding_sensitivity: float,
) -> float:
    """
    H_i(t,k) = broadcast_direction × broadcast_intensity × herding_sensitivity

    broadcast_direction: +1 (bullish) or -1 (bearish) or 0 (no signal)
    broadcast_intensity: 0.0–1.0, set by ShockProcessor
    herding_sensitivity: agent characteristic (0–1)

    Returns a scalar in [−1, +1].
    """
    return broadcast_direction * broadcast_intensity * herding_sensitivity


def compute_background_influence(
    non_market_signal_intensity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the ambient background influence channel signal for all sectors
    when no active shock is targeting the influence channel (spec §3.2).

    Draws noise ~ N(0, SIGMA_BACKGROUND) per sector.
    Direction is derived from sign of the draw.
    Intensity = |z-score| × non_market_signal_intensity, clipped to [0, 1].
    non_market_signal_intensity thus directly scales how strongly background
    sentiment drives agent behaviour (0 = no background signal, 1 = full).

    Returns (directions, intensities) arrays of shape (NUM_SECTORS,).
    """
    noise = rng.normal(0, SIGMA_BACKGROUND, NUM_SECTORS)
    directions = np.sign(noise)
    # Normalise by σ to get z-scores, then scale by NMI so the parameter
    # actually controls background influence strength (previously it cancelled out).
    intensities = np.clip(np.abs(noise) / SIGMA_BACKGROUND * non_market_signal_intensity, 0.0, 1.0)
    return directions, intensities


# ── Sector affinity (spec §7.12) ───────────────────────────────────────────────

def compute_sector_affinity_weights(
    agent_type: str,
    risk_aversion: float,
    belief_formation: float,
    time_horizon: float,
    information_quality: float,
    price_memory_all: list[deque],
    current_prices: np.ndarray,
    p0: float = P0,
    p0_effective: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute A_i(k) for all 11 sectors.

    A_i(k) = base_affinity(type, k) × characteristic_modifier(agent, k)
    Then normalised so the sum of A_i across sectors equals NUM_SECTORS
    (preserving relative attention without globally inflating signal magnitudes).

    Component 1: base_affinity — engine-defined per type (from sector_config.py).
    Component 2: characteristic modifier — four characteristics modulate attention:
      - risk_aversion: boosts defensive sectors, reduces high-vol sectors
      - belief_formation (chartist end): chases highest recent momentum sector
      - belief_formation (fundamentalist end): attracted to most deviated sector
      - time_horizon: dampens high-vol sectors
      - information_quality: concentrates existing affinities (amplifies strong, dampens weak)
    """
    base = np.array(BASE_AFFINITY[agent_type], dtype=float)
    modifier = np.ones(NUM_SECTORS, dtype=float)

    # ── Risk aversion: defensive boost / high-vol penalty ─────────────────────
    for k in DEFENSIVE_SECTOR_INDICES:
        modifier[k] += risk_aversion * RISK_AVERSION_AFFINITY_SHIFT
    for k in HIGH_VOL_SECTOR_INDICES:
        modifier[k] -= risk_aversion * RISK_AVERSION_AFFINITY_SHIFT

    # ── Time horizon: dampens high-vol sectors ────────────────────────────────
    for k in HIGH_VOL_SECTOR_INDICES:
        modifier[k] -= time_horizon * TIME_HORIZON_AFFINITY_DAMPEN

    # ── Belief formation: chartist chases momentum, fundamentalist chases gap ─
    chartist_weight = belief_formation
    fundamentalist_weight = 1.0 - belief_formation

    if chartist_weight > 0.5:
        # Chartist: boost sector with highest recent momentum (largest C signal proxy).
        momentums = _compute_sector_momentums(price_memory_all)
        if momentums is not None:
            best_momentum_sector = int(np.argmax(np.abs(momentums)))
            momentum_boost = (chartist_weight - 0.5) * 0.6
            modifier[best_momentum_sector] += momentum_boost

    if fundamentalist_weight > 0.5:
        # Fundamentalist: boost sector showing largest price gap from P0_effective.
        # Use p0_effective if provided (shock-shifted fair value), else fall back to P0.
        p0_ref = p0_effective if p0_effective is not None else np.full(NUM_SECTORS, p0)
        safe_p0 = np.where(p0_ref > 0, p0_ref, p0)
        gaps = np.abs(current_prices - safe_p0) / safe_p0
        best_gap_sector = int(np.argmax(gaps))
        gap_boost = (fundamentalist_weight - 0.5) * 0.6
        modifier[best_gap_sector] += gap_boost

    # ── Information quality: concentrating effect ─────────────────────────────
    # High IQ agents amplify their already-strong base affinities;
    # low IQ agents flatten toward equal attention.
    # concentrating_factor ∈ [0.5, 1.0]: multiplies the distance from neutral (1.0).
    concentrate = 0.5 + information_quality * 0.5
    deviation = base * modifier - 1.0  # distance from neutral
    final = 1.0 + deviation * concentrate

    # Apply minimum floor and normalise.
    final = np.maximum(final, AFFINITY_MIN)
    # Normalise so sum = NUM_SECTORS (mean = 1.0), preserving relative magnitudes.
    final = final / final.mean()

    return final


def _compute_sector_momentums(price_memory_all: list[deque]) -> np.ndarray | None:
    """
    Compute a simple momentum proxy for each sector from price memory:
    sign-weighted sum of the last few log-returns.
    Returns None if insufficient history.
    """
    momentums = np.zeros(NUM_SECTORS)
    has_data = False
    for k, mem in enumerate(price_memory_all):
        prices = list(mem)
        if len(prices) >= 2:
            has_data = True
            returns = [np.log(prices[i+1] / prices[i])
                      for i in range(len(prices)-1) if prices[i] > 0]
            if returns:
                momentums[k] = float(np.mean(returns[-5:]))  # last 5 ticks
    return momentums if has_data else None


# ── Composite signal (Step 1 full formula) ────────────────────────────────────

def compute_composite_signal(
    # Agent characteristics
    beta_f: float,
    beta_c: float,
    herding_sensitivity: float,
    information_quality: float,
    recency_bias: float,
    time_horizon: float,
    is_institutional: bool,
    memory_ticks: int,
    # Signal inputs
    price_k: float,
    price_memory_k: deque,
    broadcast_direction_k: float,
    broadcast_intensity_k: float,
    # Environment
    sector_affinity_k: float,
    rng: np.random.Generator,
    # Effective fundamental price for this sector (from ShockProcessor).
    p0_effective: float = P0,
) -> float:
    """
    Compute the full composite signal S_eff(i,t,k) for one agent × one sector.

    S_i(t,k) = β_f·F·q_i  +  β_c·C  +  H  +  ε_i(k)
    S_eff(i,t,k) = S_i(t,k) × A_i(k)

    Signal components and their information-quality treatment:
      F — fundamentalist analysis (private research quality): scaled by q_i.
          High-IQ agents produce more reliable fundamental estimates.
      C — chartist / momentum (public price data): NOT scaled by q_i.
          Price history is equally observable by all agents.  Removing the IQ
          attenuation enables the chartist feedback loop that drives emergence:
          price move → C grows → larger trades → bigger move → more divergence.
      H — herding/influence (broadcast sentiment): NOT scaled by q_i.
          herding_sensitivity already encodes agent-type susceptibility; high-IQ
          institutional types have low herding_sensitivity by calibration design.
      ε — private noise: inversely proportional to q_i (less noise for better info).

    This decomposition is the key driver of ABM emergence.  In the old formula
    (all three scaled by q_i), R4 agents (beta_c=0.8, IQ=0.175) had their
    chartist signal multiplied by 0.175 — an 82.5% reduction — making the
    feedback loop structurally impossible regardless of price momentum.  With
    the new formula, R4's C contribution is fully expressed, enabling momentum
    cascades that cause runs to diverge.

    Returns S_eff — the effective signal after sector affinity scaling.
    """
    F = compute_fundamentalist_signal(price_k, p0_effective)
    C = compute_chartist_signal(
        price_memory_k, memory_ticks, recency_bias, time_horizon, is_institutional, beta_c
    )
    H = compute_influence_signal(broadcast_direction_k, broadcast_intensity_k, herding_sensitivity)

    # Decomposed raw signal (spec §7.5 Step 1, revised):
    #   F quality-dependent; C and H directly expressed (see docstring above).
    S_raw = beta_f * F * information_quality + beta_c * C + H

    # Noise term: inversely proportional to information quality.
    noise_sigma = max(0.0, 1.0 - information_quality) * 0.10
    epsilon = float(rng.normal(0.0, noise_sigma)) if noise_sigma > 0 else 0.0

    S = S_raw + epsilon

    # Sector affinity scaling.
    S_eff = S * sector_affinity_k

    return float(S_eff)
