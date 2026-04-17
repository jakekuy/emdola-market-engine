"""
Five-step activation function — the per-agent, per-sector decision kernel.

Called from the run loop for each agent that passes the Layer 1 evaluation
probability gate (handled externally).  This module processes all 11 sectors
for one agent and returns the trades it executes this tick.

Spec references: §7.5 (5-step activation), §7.12 (sector affinity).

Step 1 — Signal formation       : composite signal S_eff from F, C, H channels
Step 2 — Bias distortions       : confirmation bias, anchoring, salience, pattern
Step 3 — Raw demand             : D_i = S_biased / (risk_aversion × σ²_perceived)
Step 4 — Sigmoid gate           : θ_eff with behavioural modifiers → trade/no-trade
Step 5 — Trade execution        : direction + magnitude → portfolio + environment
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque

from backend.config.constants import (
    # Variance / demand
    SIGMA_SQUARED_FLOOR,
    # Sigmoid gate
    SIGMOID_STEEPNESS,
    SIGNAL_GATE_CAP,
    # Bias distortion
    CONFIRMATION_BIAS_ATTENUATION,
    SALIENCE_BIAS_AMPLIFICATION,
    SALIENCE_SIGMA_THRESHOLD,
    PATTERN_MATCHING_AMPLIFICATION,
    # Theta modifiers
    LOSS_AVERSION_THETA_RAISE,
    WINNER_SELLING_THETA_SHIFT,
    REGRET_AVERSION_THETA_RAISE,
    FOMO_THETA_LOWER,
    FOMO_INFLUENCE_THRESHOLD,
    INSTITUTIONAL_INERTIA_THETA_RAISE,
    OWNERSHIP_BIAS_THETA_RAISE,
    DECISION_STYLE_NOISE_SIGMA,
    OVERCONFIDENCE_THETA_LOWER,
    # Step 5
    INSTITUTIONAL_INERTIA_MAGNITUDE_DAMPEN,
    MENTAL_ACCOUNTING_NOISE_RANGE,
    # Anchoring
    ANCHOR_LOOKBACK_TICKS,
)
from backend.config.sector_config import NUM_SECTORS
from backend.simulation.signals import (
    compute_composite_signal,
    compute_sector_affinity_weights,
)

# Type hints only — avoid circular imports at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.simulation.agent import Agent
    from backend.simulation.environment import MarketEnvironment


# ── Public entry point ────────────────────────────────────────────────────────

def run_agent_activation(
    agent: "Agent",
    env: "MarketEnvironment",
    current_tick: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, float]]:
    """
    Run the full 5-step activation for one agent across all 11 sectors.

    Strategy adaptability is checked once before the sector loop.
    Sector affinity weights are computed once and reused across sectors.

    Returns
    -------
    list of (sector_index, direction, actual_magnitude) for each executed trade.
    """
    # Strategy adaptability check: may shift beta_f / beta_c.
    agent.check_strategy_adaptability(current_tick)

    # Compute sector affinity weights once for all sectors (spec §7.12).
    p = agent.profile
    sector_affinities = compute_sector_affinity_weights(
        agent_type=agent.agent_type,
        risk_aversion=p.risk_aversion,
        belief_formation=p.belief_formation,
        time_horizon=p.time_horizon,
        information_quality=p.information_quality,
        price_memory_all=agent.price_memory,
        current_prices=env.prices,
        p0_effective=env.p0_effective,
    )

    trades: list[tuple[int, int, float]] = []

    for k in range(NUM_SECTORS):
        result = _run_sector_activation(
            agent, k, env, current_tick, rng, sector_affinities[k]
        )
        if result is not None:
            direction, magnitude = result
            trades.append((k, direction, magnitude))

    return trades


# ── Per-sector activation pipeline ───────────────────────────────────────────

def _run_sector_activation(
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    current_tick: int,
    rng: np.random.Generator,
    sector_affinity_k: float,
) -> tuple[int, float] | None:
    """
    Steps 1–5 for one agent × one sector.
    Returns (direction, actual_magnitude) or None if the gate did not fire.
    """
    # Step 1 — Signal formation.
    S_eff = _step1_form_signal(agent, sector_k, env, rng, sector_affinity_k)

    # Step 2 — Bias distortions.
    S_biased = _step2_apply_bias_distortions(S_eff, agent, sector_k, env)

    if S_biased == 0.0:
        return None

    # Step 3 — Raw demand (direction only; magnitude uses S_biased directly in Step 5).
    D_i = _step3_compute_raw_demand(S_biased, agent, sector_k, env, rng)

    if D_i == 0.0:
        return None

    # Step 4 — Sigmoid gate: gate on signal strength, not exploded D_i.
    fired = _step4_evaluate_gate(D_i, S_biased, agent, sector_k, env, rng)
    if not fired:
        return None

    # Step 5 — Trade execution: magnitude proportional to signal strength.
    return _step5_execute_trade(D_i, S_biased, agent, sector_k, env, current_tick)


# ── Step 1: Signal formation ──────────────────────────────────────────────────

def _step1_form_signal(
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    rng: np.random.Generator,
    sector_affinity_k: float,
) -> float:
    """
    S_eff(i,t,k) = [β_f·F + β_c·C + β_h·H] · q_i + ε_i(k) × A_i(k)

    Delegates to compute_composite_signal in signals.py.
    """
    p = agent.profile
    return compute_composite_signal(
        beta_f=agent.beta_f,
        beta_c=agent.beta_c,
        herding_sensitivity=p.herding_sensitivity,
        information_quality=p.information_quality,
        recency_bias=p.recency_bias,
        time_horizon=p.time_horizon,
        is_institutional=agent.is_institutional,
        memory_ticks=agent.memory_ticks,
        price_k=env.prices[sector_k],
        price_memory_k=agent.price_memory[sector_k],
        broadcast_direction_k=env.influence_signal.direction[sector_k],
        broadcast_intensity_k=env.influence_signal.intensity[sector_k],
        sector_affinity_k=sector_affinity_k,
        rng=rng,
        p0_effective=float(env.p0_effective[sector_k]),
    )


# ── Step 2: Bias distortions ──────────────────────────────────────────────────

def _step2_apply_bias_distortions(
    S: float,
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
) -> float:
    """
    Apply four bias distortions to the raw signal (spec §7.5 Step 2).

    Order: confirmation bias → anchoring → salience bias → pattern matching.
    Each operates multiplicatively or additively on S.
    """
    p = agent.profile

    # ── Confirmation bias ─────────────────────────────────────────────────────
    # Attenuate signal that contradicts an existing long position.
    # No short positions in Stage 1, so only long × sell-signal case applies.
    if p.confirmation_bias > 0.0:
        has_position = agent.sector_weights[sector_k] > 1e-6
        signal_dir = math.copysign(1.0, S) if S != 0.0 else 0.0
        if has_position and signal_dir < 0:
            # Sell signal while holding a long position: attenuate.
            S *= (1.0 - p.confirmation_bias * CONFIRMATION_BIAS_ATTENUATION)

    # ── Anchoring ─────────────────────────────────────────────────────────────
    # Blend signal toward an anchor-price-implied return.
    if p.anchoring > 0.0:
        anchor_price = _get_anchor_price(agent, sector_k, env, S)
        if anchor_price > 0.0:
            anchor_signal = (anchor_price - env.prices[sector_k]) / anchor_price
            S = S * (1.0 - p.anchoring) + anchor_signal * p.anchoring

    # ── Salience bias ─────────────────────────────────────────────────────────
    # Amplify signal when the most recent price move was an extreme outlier (>2σ).
    if p.salience_bias > 0.0:
        sigma = env.get_price_history_sigma(sector_k)
        if sigma > 0.0:
            mem = agent.price_memory[sector_k]
            if len(mem) >= 2:
                prices = list(mem)
                last_log_return = abs(
                    np.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0.0
                )
                if last_log_return > SALIENCE_SIGMA_THRESHOLD * sigma:
                    S *= (1.0 + p.salience_bias * SALIENCE_BIAS_AMPLIFICATION)

    # ── Pattern matching ──────────────────────────────────────────────────────
    # Amplify signal when recent price trend direction matches signal direction.
    if p.pattern_matching_bias > 0.0 and S != 0.0:
        trend_dir = _compute_trend_direction(agent.price_memory[sector_k])
        signal_dir = math.copysign(1.0, S)
        if trend_dir * signal_dir > 0:
            # Signal aligns with recent trend — amplify.
            S *= (1.0 + p.pattern_matching_bias * PATTERN_MATCHING_AMPLIFICATION)

    return S


def _get_anchor_price(
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    signal: float,
) -> float:
    """
    Return the anchor price for sector k.

    If a position is held: weighted average cost (purchase_price).
    If no position: recent high (for bearish signals) or recent low (for bullish).
    Returns the current price if no useful anchor is available.
    """
    if agent.sector_weights[sector_k] > 1e-6:
        return float(agent.purchase_price[sector_k])

    recent = list(agent.price_memory[sector_k])[-ANCHOR_LOOKBACK_TICKS:]
    if not recent:
        return env.prices[sector_k]

    if signal > 0:
        return float(min(recent))  # bullish: anchored to recent low
    else:
        return float(max(recent))  # bearish: anchored to recent high


def _compute_trend_direction(price_memory: deque) -> float:
    """
    Return +1 if recent prices are trending up, -1 if down, 0 if flat.
    Uses the mean of the last few log-returns as a trend proxy.
    """
    prices = list(price_memory)
    if len(prices) < 3:
        return 0.0
    recent = prices[-min(5, len(prices)):]
    log_returns = [
        np.log(recent[i + 1] / recent[i])
        for i in range(len(recent) - 1)
        if recent[i] > 0 and recent[i + 1] > 0
    ]
    if not log_returns:
        return 0.0
    mean_return = float(np.mean(log_returns))
    return math.copysign(1.0, mean_return) if mean_return != 0.0 else 0.0


# ── Step 3: Raw demand ────────────────────────────────────────────────────────

def _step3_compute_raw_demand(
    S_biased: float,
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    rng: np.random.Generator,
) -> float:
    """
    D_i = S_biased / (risk_aversion_eff × σ²)

    Only sign(D_i) = sign(S_biased) is consumed downstream (gate and magnitude
    both use S_biased directly — see Step 4 docstring).  Overconfidence effect
    is captured via OVERCONFIDENCE_THETA_LOWER in Step 4.
    Mental accounting (retail only): adds per-sector noise to effective risk_aversion.
    """
    p = agent.profile

    # Observed variance (log-return σ²) for this sector.
    sigma = env.get_price_history_sigma(sector_k)
    sigma_sq = max(sigma ** 2, SIGMA_SQUARED_FLOOR)

    # Effective risk aversion — mental accounting adds per-sector noise for retail.
    ra = max(p.risk_aversion, 0.05)  # floor to avoid division by zero
    if p.mental_accounting is not None and p.mental_accounting > 0.0:
        noise = float(rng.uniform(-1.0, 1.0)) * MENTAL_ACCOUNTING_NOISE_RANGE * p.mental_accounting
        ra = max(0.05, ra + noise)

    D_i = S_biased / (ra * sigma_sq)
    return float(D_i)


# ── Step 4: Sigmoid gate ──────────────────────────────────────────────────────

def _step4_evaluate_gate(
    D_i: float,
    S_biased: float,
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    rng: np.random.Generator,
) -> bool:
    """
    Compute θ_eff with all behavioural modifiers, then evaluate the gate:
        P(act) = sigmoid(|S_biased| − θ_eff)

    Uses S_biased (the biased composite signal, bounded ≈ ±1.5) rather than D_i.
    D_i = S/(ra × σ²) explodes to ~100,000 due to the σ²_floor=1e-6, making
    min(|D_i|, cap) always equal to cap — the gate becomes a constant per agent
    type, completely independent of signal strength.  Using S_biased directly
    makes the firing probability proportional to actual signal intensity so that
    a background-noise tick and a shock tick produce meaningfully different outcomes.
    Returns True if the agent fires a trade this sector.
    """
    p = agent.profile
    direction = 1 if D_i > 0 else -1
    current_price = env.prices[sector_k]
    sector_return = agent.get_current_return_for_sector(sector_k, current_price)

    theta = p.conviction_threshold

    # ── Loss aversion: harder to crystallise a loss ───────────────────────────
    # Applies when selling a losing position.
    if direction < 0 and sector_return < 0.0:
        theta += p.loss_aversion * LOSS_AVERSION_THETA_RAISE

    # ── Winner/loser (disposition effect) ────────────────────────────────────
    # Lower threshold to sell winners (eager to lock in gains);
    # raise threshold to sell losers (reluctance to realise losses).
    if direction < 0:
        wslh = p.winner_selling_loser_holding
        if sector_return >= 0.0:
            theta -= wslh * WINNER_SELLING_THETA_SHIFT   # easier to sell winner
        else:
            theta += wslh * WINNER_SELLING_THETA_SHIFT   # harder to sell loser

    # ── Regret aversion: hesitate when action is non-consensus ───────────────
    # Non-consensus = own signal direction opposes the influence channel broadcast.
    influence_dir = env.influence_signal.direction[sector_k]
    if influence_dir != 0.0 and influence_dir * direction < 0:
        # Acting against the broadcast — higher conviction required.
        theta += p.regret_aversion * REGRET_AVERSION_THETA_RAISE

    # ── FOMO: lower threshold when influence channel is strong ────────────────
    influence_intensity = env.influence_signal.intensity[sector_k]
    if influence_intensity > FOMO_INFLUENCE_THRESHOLD:
        theta -= p.fomo * FOMO_THETA_LOWER

    # ── Institutional inertia (institutions only) ─────────────────────────────
    if p.institutional_inertia is not None:
        theta += p.institutional_inertia * INSTITUTIONAL_INERTIA_THETA_RAISE

    # ── Ownership bias (retail only): reluctance to sell a held position ──────
    if p.ownership_bias is not None and direction < 0:
        if agent.sector_weights[sector_k] > 1e-6:
            theta += p.ownership_bias * OWNERSHIP_BIAS_THETA_RAISE

    # ── Decision style noise ──────────────────────────────────────────────────
    noise_sigma = p.decision_style * DECISION_STYLE_NOISE_SIGMA
    if noise_sigma > 0.0:
        theta += float(rng.normal(0.0, noise_sigma))

    # ── Overconfidence: lowers effective conviction threshold ─────────────────
    # Overconfident agents perceive less risk and act more readily.
    # Applied unconditionally — always active when overconfidence > 0.
    if p.overconfidence > 0.0:
        theta -= p.overconfidence * OVERCONFIDENCE_THETA_LOWER

    # ── Sigmoid gate ──────────────────────────────────────────────────────────
    # P(act) = sigmoid(|S_biased| − θ_eff).
    # S_biased is naturally bounded ≈ [−1.5, 1.5] (tanh-bounded C, ±1 F and H,
    # sector affinity ≈ 1, bias multipliers ≤ 1.4×).  theta is in [0, 1].
    # Strong signal → high p_act even for cautious agents.
    # Weak background noise → only low-theta agents fire.
    d_gated = min(abs(S_biased), SIGNAL_GATE_CAP)
    x = SIGMOID_STEEPNESS * (d_gated - theta)
    p_act = 1.0 / (1.0 + math.exp(-x))

    return bool(rng.random() < p_act)


# ── Step 5: Trade execution ───────────────────────────────────────────────────

def _step5_execute_trade(
    D_i: float,
    S_biased: float,
    agent: "Agent",
    sector_k: int,
    env: "MarketEnvironment",
    current_tick: int,
) -> tuple[int, float]:
    """
    Convert signal into a concrete trade and apply it to agent portfolio + environment.

    direction  = sign(D_i)  [= sign(S_biased), since ra and σ² are both positive]
    magnitude  = min(|S_biased| / risk_aversion, 1.0) × AUM × position_scale

    Using S_biased for magnitude (rather than D_i) preserves proportionality between
    signal strength and trade size.  D_i = S/(ra × σ²_floor) ≈ 100,000 for any
    non-zero signal, making min(|D_i|, 1.0) ≡ 1.0 always — all trades execute at
    maximum magnitude regardless of signal intensity, killing the chartist feedback
    loop that drives emergent dynamics.  S_biased is bounded ≈ ±1.5, so weak noise
    signals produce tiny trades and strong shock/momentum signals produce large ones.
    Dividing by risk_aversion preserves the spec intent that less risk-averse agents
    take proportionally larger positions for the same signal.

    Institutional inertia further dampens magnitude.
    Agent.execute_trade() clamps buys to available cash.
    """
    direction = 1 if D_i > 0 else -1
    p = agent.profile

    # Base magnitude: signal strength / risk_aversion as fraction of position, capped at 1.
    ra = max(p.risk_aversion, 0.05)
    magnitude = min(abs(S_biased) / ra, 1.0) * agent.aum * agent.position_scale

    # Institutional inertia dampens execution magnitude.
    if p.institutional_inertia is not None and p.institutional_inertia > 0.0:
        magnitude *= (1.0 - p.institutional_inertia * INSTITUTIONAL_INERTIA_MAGNITUDE_DAMPEN)

    magnitude = max(0.0, magnitude)

    # Execute on portfolio (agent.execute_trade handles cash clamping).
    actual_magnitude = agent.execute_trade(
        sector=sector_k,
        direction=direction,
        magnitude=magnitude,
        current_tick=current_tick,
    )

    # Register trade in the environment's excess demand accumulator.
    signed_magnitude = direction * actual_magnitude
    env.accumulate_trade(sector_k, signed_magnitude)

    return direction, actual_magnitude
