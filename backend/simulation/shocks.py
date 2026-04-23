"""
Shock processor — translates ShockDefinition objects into per-tick market
and influence channel signals used by the simulation run loop.

Spec references: §10 (shock model), §3.2 (influence channel), §4 (price formation).

Market channel: shifts the per-sector effective fundamental price (P0_effective).
  Fundamentalist agents observe (P0_eff - price) / P0_eff as their F signal.
  Prices rise or fall EMERGENTLY as agents respond — no prices are forced directly.
  This is the correct ABM mechanism: the shock changes beliefs, agents create the
  price impact.  Different runs respond differently (some overshoot, some undershoot),
  producing the Monte Carlo spread that is the model's analytical output.

Influence channel: sets broadcast direction + intensity for the relevant tick(s).
  Agents with high herding_sensitivity respond to the broadcast signal.

Precomputes all tick × sector values at construction time for O(1) lookup
during the run loop.
"""

from __future__ import annotations

import math
import numpy as np

from backend.config.constants import P0, INFLUENCE_INTENSITY_SCALE
from backend.config.sector_config import NUM_SECTORS
from backend.models.calibration import ShockDefinition
from backend.simulation.environment import InfluenceSignal


class ShockProcessor:
    """
    Converts ShockDefinition objects into per-tick, per-sector signals.

    Market channel — stored as per-tick P0_effective values (absolute prices):
      No reversion:      P0_eff ramps from P0 to P0×(1+mag) over duration ticks,
                         then stays at that level.
      Acute (reversion): P0_eff jumps to P0×(1+mag) at onset_tick, then linearly
                         reverts to P0 over the next `duration` ticks.
      Chronic:           P0_eff ramps linearly from P0 to P0×(1+mag) over duration.

    Influence channel — direction ∈ {−1, 0, +1} and intensity ∈ [0, 1]:
      Active for all ticks in [onset_tick, onset_tick + duration).
      Multiple overlapping shocks on the same sector: highest intensity wins.
    """

    def __init__(
        self,
        shocks: list[ShockDefinition],
        total_ticks: int,
        narrative_half_life: int = 10,
    ) -> None:
        T = total_ticks
        # Delta accumulator per tick per sector: each shock adds its own
        # contribution; P0_effective = P0 + _delta, so multiple shocks on
        # the same sector stack correctly rather than overwriting each other.
        self._delta: np.ndarray = np.zeros((T, NUM_SECTORS), dtype=float)
        # Influence channel direction and intensity per tick per sector.
        self._inf_directions: np.ndarray = np.zeros((T, NUM_SECTORS))
        self._inf_intensities: np.ndarray = np.zeros((T, NUM_SECTORS))
        # Bool mask: True if any shock is active on this tick.
        self._active: np.ndarray = np.zeros(T, dtype=bool)

        # Influence decay rate derived from narrative half-life:
        # intensity(age) = base * exp(-age * decay_rate); at age=half_life, intensity=0.5*base.
        self._influence_decay_rate: float = math.log(2) / max(narrative_half_life, 1)
        self._total_ticks = T
        self._precompute(shocks)
        # Build P0_effective from accumulated deltas.
        self._p0_effective: np.ndarray = P0 + self._delta

    # ── Precomputation ────────────────────────────────────────────────────────

    def _precompute(self, shocks: list[ShockDefinition]) -> None:
        T = self._total_ticks
        for shock in shocks:
            onset = shock.onset_tick
            dur = max(shock.duration, 1)
            mag = shock.magnitude
            sectors = shock.affected_sectors
            inf_end = min(onset + dur, T)

            # ── Market channel: accumulate delta onto P0_effective ────────────
            # Each shock adds its own signed delta (P0 * mag) to _delta rather
            # than writing an absolute value.  Multiple shocks on the same sector
            # therefore stack correctly: a +0.15 closure shock followed by a
            # -0.06 normalisation shock on the same sector produces a net +0.09
            # effect, not a -0.06 overwrite.
            if shock.channel in ("market", "both"):
                delta_target = P0 * mag  # e.g. P0=100, mag=0.15 → delta=+15

                if shock.shock_type == "acute":
                    if onset < T:
                        for k in sectors:
                            self._delta[onset, k] += delta_target
                    if shock.reversion:
                        # This shock's delta linearly decays back to 0 over dur ticks.
                        for t in range(onset + 1, min(onset + dur + 1, T)):
                            frac = (t - onset) / dur   # 0 < frac ≤ 1
                            for k in sectors:
                                self._delta[t, k] += delta_target * (1.0 - frac)
                        # After full reversion this shock contributes 0 — no further add.
                    else:
                        # No reversion: delta holds at delta_target for all later ticks.
                        for t in range(onset + 1, T):
                            for k in sectors:
                                self._delta[t, k] += delta_target

                else:  # chronic: ramp up over duration, then hold
                    for t in range(onset, min(onset + dur, T)):
                        frac = (t - onset + 1) / dur   # 1/dur … 1.0
                        for k in sectors:
                            self._delta[t, k] += delta_target * frac
                    # After ramp-up, hold at full delta_target.
                    for t in range(min(onset + dur, T), T):
                        for k in sectors:
                            self._delta[t, k] += delta_target

            # ── Influence channel ─────────────────────────────────────────────
            if shock.channel in ("influence", "both"):
                direction = 1.0 if mag > 0 else (-1.0 if mag < 0 else 0.0)
                base_intensity = min(abs(mag) / INFLUENCE_INTENSITY_SCALE, 1.0)
                for t in range(onset, inf_end):
                    # Exponential decay: sharp initial consensus → agents diverge
                    # as the news is processed.  Prevents 25-tick constant consensus
                    # that kills two-sided flow for the entire shock window.
                    age = t - onset
                    intensity = base_intensity * float(np.exp(-age * self._influence_decay_rate))
                    for k in sectors:
                        # Higher intensity wins if shocks overlap on same sector/tick.
                        if intensity > self._inf_intensities[t, k]:
                            self._inf_directions[t, k] = direction
                            self._inf_intensities[t, k] = intensity

            # ── Mark active ticks ─────────────────────────────────────────────
            self._active[onset:inf_end] = True

    # ── Per-tick query API ────────────────────────────────────────────────────

    def get_p0_effective(self, tick: int) -> np.ndarray:
        """
        Return the per-sector P0_effective values for this tick.

        These are the 'fair values' that fundamentalist agents target.
        A market-channel shock shifts them away from P0=100 to reflect
        changed economic beliefs; agents then bid prices toward those beliefs
        emergently, with path-dependent results that vary across MC runs.

        Safe to call for any tick — returns all-P0 if out of range.
        """
        if tick < 0 or tick >= self._total_ticks:
            return np.full(NUM_SECTORS, P0, dtype=float)
        return self._p0_effective[tick].copy()

    def get_influence_signal(self, tick: int) -> InfluenceSignal | None:
        """
        Return the shock-driven influence signal for this tick, or None if
        no shock is affecting the influence channel at this tick.

        Returns None signals the run loop to use the ambient background signal.
        """
        if tick < 0 or tick >= self._total_ticks:
            return None
        if not self._active[tick]:
            return None
        # Only return a shock signal if at least one sector has non-zero intensity.
        if not np.any(self._inf_intensities[tick]):
            return None
        return InfluenceSignal(
            direction=self._inf_directions[tick].copy(),
            intensity=self._inf_intensities[tick].copy(),
        )

    def is_active(self, tick: int) -> bool:
        """True if any shock is active at this tick."""
        if tick < 0 or tick >= self._total_ticks:
            return False
        return bool(self._active[tick])
