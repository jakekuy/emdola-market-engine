"""
Agent — holds all per-agent state and provides portfolio management methods.

An agent is a representative unit of one of the 8 EMDOLA investor types.
Its behaviour is entirely determined by:
  - Layer 1: structural attributes set here from config (AUM, position_scale, etc.)
  - Layer 2: characteristics from an LLM-generated persona profile

Spec references: §7.4–7.11.
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque

from backend.config.constants import (
    MEMORY_SCORE_TO_TICKS,
    P0,
    DEFAULT_TOTAL_TICKS,
    MAX_CASH_DEPLOY_FRACTION_PER_TICK,
)
from backend.config.agent_config import (
    AUM_PER_AGENT,
    EVALUATION_PROBABILITY,
    POSITION_SCALE,
    IS_INSTITUTIONAL,
    STARTING_CASH_WEIGHT,
)
from backend.config.sector_config import NUM_SECTORS
from backend.models.profile import AgentCharacteristics


def memory_score_to_ticks(score: float) -> int:
    """
    Convert a memory_window characteristic score (0.0–1.0) to a tick count
    using linear interpolation over the MEMORY_SCORE_TO_TICKS anchor table.
    """
    anchors = MEMORY_SCORE_TO_TICKS
    if score <= anchors[0][0]:
        return anchors[0][1]
    if score >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        s0, t0 = anchors[i]
        s1, t1 = anchors[i + 1]
        if s0 <= score <= s1:
            frac = (score - s0) / (s1 - s0)
            return max(1, round(t0 + frac * (t1 - t0)))
    return 40  # fallback: midpoint default


class Agent:
    """
    Represents one investor agent in the simulation.

    Key attributes
    --------------
    agent_type : str
        One of R1–R4, I1–I4.
    agent_id : int
        Unique integer ID within the simulation run.
    profile : AgentCharacteristics
        LLM-generated characteristic values (Layer 2).
    aum : float
        Current total assets under management (mark-to-market each tick).
    cash_weight : float
        Fraction of AUM held in cash (0.0–1.0).
    sector_weights : np.ndarray
        11-element array of sector allocation weights; all ≥ 0.
        cash_weight + sum(sector_weights) ≈ 1.0 at all times.
    beta_f, beta_c : float
        Mutable fundamentalist/chartist weights. Initialised from
        (1 - belief_formation) and belief_formation; may shift via
        strategy_adaptability.
    """

    def __init__(
        self,
        agent_type: str,
        agent_id: int,
        profile: AgentCharacteristics,
        calibration_aum: float | None = None,
        calibration_cash_weight: float | None = None,
        calibration_sector_pcts: list[float] | None = None,
    ) -> None:
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.profile = profile
        self.is_institutional = IS_INSTITUTIONAL[agent_type]

        # ── Layer 1 structural attributes ──────────────────────────────────────
        self.eval_probability: float = EVALUATION_PROBABILITY[agent_type]
        self.position_scale: float = POSITION_SCALE[agent_type]

        # AUM: use calibration override if supplied, else type default.
        base_aum = calibration_aum if calibration_aum is not None else AUM_PER_AGENT[agent_type]
        self.aum: float = float(base_aum)
        self._initial_aum: float = self.aum

        # ── Portfolio state ────────────────────────────────────────────────────
        cash_w = (
            calibration_cash_weight
            if calibration_cash_weight is not None
            else STARTING_CASH_WEIGHT[agent_type]
        )
        self.cash_weight: float = float(cash_w)

        # Convert sector percentages → fractional weights of TOTAL portfolio.
        if calibration_sector_pcts is not None:
            pcts = calibration_sector_pcts
        else:
            from backend.config.agent_config import STARTING_SECTOR_PCT
            pcts = STARTING_SECTOR_PCT[agent_type]

        equity_fraction = 1.0 - self.cash_weight
        self.sector_weights: np.ndarray = np.array(
            [p / 100.0 * equity_fraction for p in pcts], dtype=float
        )
        assert len(self.sector_weights) == NUM_SECTORS

        # ── Purchase price tracking (for anchoring and winner/loser detection) ─
        # Weighted average cost per sector; None if no position held.
        self.purchase_price: np.ndarray = np.full(NUM_SECTORS, P0, dtype=float)

        # ── Mutable strategy betas (strategy_adaptability can shift these) ─────
        bf = profile.belief_formation
        self.beta_f: float = 1.0 - bf   # fundamentalist weight
        self.beta_c: float = bf         # chartist/momentum weight

        # ── Memory window ──────────────────────────────────────────────────────
        self.memory_ticks: int = memory_score_to_ticks(profile.memory_window)

        # Signal memory: per-sector deque of (log_price, log_return) tuples.
        # Length = memory_ticks; oldest entries drop automatically.
        self.price_memory: list[deque[float]] = [
            deque(maxlen=self.memory_ticks) for _ in range(NUM_SECTORS)
        ]
        # Influence channel memory: per-sector deque of (direction, intensity) tuples.
        self.influence_memory: list[deque[tuple[float, float]]] = [
            deque(maxlen=self.memory_ticks) for _ in range(NUM_SECTORS)
        ]

        # ── Trace flags ────────────────────────────────────────────────────────
        # Unused sentinel kept for compatibility; all trade capture is handled
        # at the run level (run.py post-loop filtering across all agents).
        self.is_traced: bool = False
        # Archetype variant number (1, 2, or 3); set by population builder.
        self.archetype_num: int = 0

        # ── Action history ─────────────────────────────────────────────────────
        # Deque of (tick, sector, direction, magnitude) tuples.
        self.action_history: deque[tuple[int, int, int, float]] = deque(
            maxlen=self.memory_ticks
        )

        # ── Performance history ────────────────────────────────────────────────
        # Rolling AUM values for return computation.
        self.aum_history: deque[float] = deque(maxlen=self.memory_ticks)
        self.aum_history.append(self.aum)

        # ── Strategy adaptability state ────────────────────────────────────────
        # Tracks whether a shift is currently pending to avoid repeated triggers.
        self._last_strategy_shift_tick: int = -1

    # ── Portfolio management ───────────────────────────────────────────────────

    def execute_trade(
        self,
        sector: int,
        direction: int,        # +1 = buy, -1 = sell
        magnitude: float,      # absolute monetary value (B USD)
        current_tick: int,
    ) -> float:
        """
        Update portfolio weights for an executed trade.

        Returns the actual magnitude executed (may be reduced if insufficient cash
        for buys or if sell exceeds current position).
        """
        weight_change = magnitude / self.aum

        if direction > 0:
            # Buy: draw from cash, rate-limited to MAX_CASH_DEPLOY_FRACTION_PER_TICK
            # of available cash.  Without this cap agents spend their full cash on
            # the first shock tick — subsequent ticks have zero buying power, volume
            # collapses and prices barely move during the shock window.
            available_cash = self.cash_weight * self.aum
            per_tick_limit = available_cash * MAX_CASH_DEPLOY_FRACTION_PER_TICK
            actual_magnitude = min(magnitude, per_tick_limit)
            actual_weight_change = actual_magnitude / self.aum
            self.sector_weights[sector] += actual_weight_change
            self.cash_weight -= actual_weight_change
            # Update purchase price (weighted average cost).
            old_holding = (self.sector_weights[sector] - actual_weight_change) * self.aum
            new_investment = actual_magnitude
            total_holding = old_holding + new_investment
            if total_holding > 0:
                old_price = self.purchase_price[sector]
                current_price_implied = self.aum * (
                    self.sector_weights[sector] - actual_weight_change
                ) / max(old_holding, 1e-9)
                # Simplified: use P0 as purchase price reference if no prior holding
                self.purchase_price[sector] = (
                    old_holding * old_price + new_investment * self._get_sector_price_approx(sector)
                ) / total_holding
        else:
            # Sell: reduce existing position toward zero (no shorting in Stage 1).
            current_position = self.sector_weights[sector] * self.aum
            actual_magnitude = min(magnitude, current_position)
            actual_weight_change = actual_magnitude / self.aum
            self.sector_weights[sector] = max(
                0.0, self.sector_weights[sector] - actual_weight_change
            )
            self.cash_weight += actual_weight_change
            actual_magnitude = actual_magnitude  # for consistency

        # Record in action history.
        self.action_history.append((current_tick, sector, direction, actual_magnitude))

        # Clamp weights to avoid floating-point drift below zero.
        self.sector_weights = np.clip(self.sector_weights, 0.0, None)
        self.cash_weight = max(0.0, self.cash_weight)

        return actual_magnitude

    def _get_sector_price_approx(self, sector: int) -> float:
        """Return the most recent logged price for a sector, or P0 if unknown."""
        mem = self.price_memory[sector]
        if mem:
            return float(mem[-1])
        return P0

    def mark_to_market(self, prices: np.ndarray) -> None:
        """
        Update AUM based on current sector prices (mark-to-market).
        AUM = cash_portion + sum(sector_holdings_at_current_prices).

        Because price changes affect sector holdings' value but not weights
        directly, we rebalance weights proportionally.
        """
        # Monetary value of each sector at current prices.
        # Sector weight × AUM × (current_price / reference_price).
        # We use log-price changes to scale holdings.
        # Simpler: sector_value_i = sector_weight_i * old_aum * (price_i / old_price_i)
        # We track purchase prices, so use them as reference.
        # For simplicity: AUM changes with weighted price return.
        #
        # New sector value = (sector_weight × AUM) × (current_price / last_price)
        # where last_price comes from memory.
        new_sector_values = np.zeros(NUM_SECTORS)
        for k in range(NUM_SECTORS):
            holding = self.sector_weights[k] * self.aum
            last_price = self._get_sector_price_approx(k)
            if last_price > 0:
                new_sector_values[k] = holding * (prices[k] / last_price)
            else:
                new_sector_values[k] = holding

        cash_value = self.cash_weight * self.aum
        new_total_aum = cash_value + new_sector_values.sum()

        if new_total_aum > 0:
            # Recompute weights from new values.
            self.sector_weights = new_sector_values / new_total_aum
            self.cash_weight = cash_value / new_total_aum
            self.aum = new_total_aum

        self.aum_history.append(self.aum)

    # ── Memory update ──────────────────────────────────────────────────────────

    def update_memory(
        self,
        prices: np.ndarray,
        influence_directions: np.ndarray,
        influence_intensities: np.ndarray,
    ) -> None:
        """
        Append current market and influence channel observations to signal memory.
        Called once per tick for every agent that evaluated (Stage 1 fired).
        """
        for k in range(NUM_SECTORS):
            self.price_memory[k].append(float(prices[k]))
            self.influence_memory[k].append(
                (float(influence_directions[k]), float(influence_intensities[k]))
            )

    # ── Performance and strategy adaptability ─────────────────────────────────

    def compute_rolling_return(self) -> float:
        """
        Mean return over the last (25% of memory_window) ticks.
        Returns 0.0 if insufficient history.
        """
        window = max(1, int(self.memory_ticks * 0.25))
        hist = list(self.aum_history)
        if len(hist) < window + 1:
            return 0.0
        recent = hist[-window - 1:]
        returns = [
            (recent[i + 1] - recent[i]) / recent[i]
            for i in range(len(recent) - 1)
            if recent[i] > 0
        ]
        return float(np.mean(returns)) if returns else 0.0

    def check_strategy_adaptability(self, current_tick: int) -> None:
        """
        Trigger β_f / β_c shift if sustained poor performance detected.
        Trigger condition: rolling return < −5% AND not triggered recently
        (cooldown = memory_ticks to avoid continuous flickering).
        Spec §7.5 Step 5 — strategy adaptability.
        """
        from backend.config.constants import (
            STRATEGY_ADAPTABILITY_RETURN_THRESHOLD,
            STRATEGY_ADAPTABILITY_MAX_SHIFT,
        )
        sa = self.profile.strategy_adaptability
        if sa <= 0.0:
            return
        cooldown = self.memory_ticks
        if current_tick - self._last_strategy_shift_tick < cooldown:
            return

        rolling_return = self.compute_rolling_return()
        if rolling_return < STRATEGY_ADAPTABILITY_RETURN_THRESHOLD:
            # Determine which component correlates better with recent returns.
            # Simplified: if price is above P0 on average, chartist was right;
            # if below P0, fundamentalist was right.
            avg_price = np.mean([
                float(mem[-1]) for mem in self.price_memory if mem
            ]) if any(self.price_memory) else P0

            shift = sa * STRATEGY_ADAPTABILITY_MAX_SHIFT
            if avg_price < P0:
                # Prices have been below fundamental — shift toward fundamentalist.
                self.beta_f = min(1.0, self.beta_f + shift)
                self.beta_c = max(0.0, self.beta_c - shift)
            else:
                # Prices trending above — shift toward chartist.
                self.beta_c = min(1.0, self.beta_c + shift)
                self.beta_f = max(0.0, self.beta_f - shift)

            self._last_strategy_shift_tick = current_tick

    # ── Helpers ────────────────────────────────────────────────────────────────

    def get_current_return_for_sector(self, sector: int, current_price: float) -> float:
        """
        Return the unrealised return on the current position in a sector.
        Positive = in profit (winner), negative = in loss (loser).
        Returns 0.0 if no position held.
        """
        if self.sector_weights[sector] <= 0.0:
            return 0.0
        pp = self.purchase_price[sector]
        if pp <= 0.0:
            return 0.0
        return (current_price - pp) / pp

    @property
    def total_portfolio_weight(self) -> float:
        """cash_weight + sum(sector_weights) — should ≈ 1.0."""
        return float(self.cash_weight + self.sector_weights.sum())

    def __repr__(self) -> str:
        return (
            f"Agent(type={self.agent_type}, id={self.agent_id}, "
            f"aum={self.aum:.0f}B, cash={self.cash_weight:.1%})"
        )
