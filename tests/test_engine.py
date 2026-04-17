"""
Integration tests for the simulation run loop (Phase 4 checkpoint).

Tests verify:
- Fundamentalist-only agents produce price reversion toward P0 without shocks.
- A market-channel shock causes the expected price response in affected sectors.
- A chronic shock produces a sustained price drift.
- RunResult structure is valid.
"""

import numpy as np
import pytest

from backend.config.constants import P0
from backend.config.sector_config import NUM_SECTORS, SECTORS
from backend.models.calibration import CalibrationInput, ShockDefinition, RunConfig
from backend.models.profile import ProfileSet
from backend.simulation.run import SimulationRun
from backend.simulation.population import build_population

from tests.conftest import (
    make_characteristics,
    make_fundamentalist_characteristics,
    make_persona,
)


TECH = 7   # sector index for Technology


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fundamentalist_profile_set() -> ProfileSet:
    """ProfileSet with pure fundamentalist personas for all 8 types."""
    from backend.config.sector_config import AGENT_TYPES
    from backend.config.agent_config import IS_INSTITUTIONAL
    chars_base = make_fundamentalist_characteristics()
    personas: dict = {}
    for atype in AGENT_TYPES:
        inertia = 0.5 if IS_INSTITUTIONAL[atype] else None
        mental = None if IS_INSTITUTIONAL[atype] else 0.5
        ownership = None if IS_INSTITUTIONAL[atype] else 0.4
        chars = make_characteristics(
            belief_formation=0.0,
            herding_sensitivity=0.05,
            conviction_threshold=0.1,
            overconfidence=0.0,
            fomo=0.0,
            anchoring=0.0,
            confirmation_bias=0.0,
            salience_bias=0.0,
            pattern_matching_bias=0.0,
            decision_style=0.0,
            information_quality=1.0,
            institutional_inertia=inertia,
            mental_accounting=mental,
            ownership_bias=ownership,
        )
        personas[atype] = [make_persona(atype, chars, num=i + 1) for i in range(3)]
    return ProfileSet(personas=personas)


def make_run(
    total_ticks: int = 50,
    shocks: list[ShockDefinition] | None = None,
    pop_seed: int = 0,
) -> tuple[SimulationRun, str]:
    """Build a SimulationRun with fundamentalist agents and optional shocks."""
    calibration = CalibrationInput(
        shocks=shocks or [],
        run=RunConfig(
            total_ticks=total_ticks,
            num_runs=1,
            data_granularity="full_fidelity",
        ),
    )
    profile_set = make_fundamentalist_profile_set()
    agents, persona_seed = build_population(calibration, profile_set, pop_seed)
    sim = SimulationRun(calibration=calibration, agents=agents)
    return sim, persona_seed


# ── Fundamentalist reversion ──────────────────────────────────────────────────

class TestFundamentalistReversion:
    def test_prices_remain_near_p0_without_shocks(self):
        """
        With only fundamentalist agents and no shocks, prices should not drift
        far from P0. Pure noise from gate randomness should average out.
        """
        sim, persona_seed = make_run(total_ticks=80)
        result = sim.run(numpy_seed=42, persona_seed=persona_seed, run_number=1)
        final_prices = result.final_prices
        for k, price in enumerate(final_prices):
            assert abs(price - P0) < 30.0, (
                f"Sector {k}: price {price:.2f} drifted too far from P0={P0} "
                "without shocks (fundamentalist agents should stabilise)."
            )

    def test_price_returns_near_p0_after_reversion_shock(self):
        """
        An acute shock with reversion=True should return prices toward P0
        by the end of the simulation.
        """
        onset = 10
        duration = 10
        shock = ShockDefinition(
            onset_tick=onset,
            magnitude=0.10,
            duration=duration,
            affected_sectors=[TECH],
            shock_type="acute",
            channel="market",
            reversion=True,
        )
        sim, persona_seed = make_run(total_ticks=60, shocks=[shock])
        result = sim.run(numpy_seed=7, persona_seed=persona_seed, run_number=1)
        final_price = result.final_prices[TECH]
        assert abs(final_price - P0) < 20.0, (
            f"Tech price {final_price:.2f} should return near P0 after reversion shock."
        )


# ── Shock response ────────────────────────────────────────────────────────────

class TestShockResponse:
    def test_acute_bullish_shock_raises_price(self):
        """
        An acute market-channel bullish shock should immediately raise the price
        in the affected sector at the onset tick.
        """
        onset = 5
        shock = ShockDefinition(
            onset_tick=onset,
            magnitude=0.15,   # +15% target
            duration=1,
            affected_sectors=[TECH],
            shock_type="acute",
            channel="market",
            reversion=False,
        )
        sim, persona_seed = make_run(total_ticks=20, shocks=[shock])
        result = sim.run(numpy_seed=0, persona_seed=persona_seed, run_number=1)

        # Find prices at tick before and after onset.
        price_before = None
        price_after = None
        for snap in result.tick_snapshots:
            if snap.tick == onset - 1:
                price_before = snap.prices[TECH]
            if snap.tick == onset:
                price_after = snap.prices[TECH]

        assert price_before is not None and price_after is not None, (
            "Ticks before and at onset should be logged (full_fidelity mode)."
        )
        assert price_after > price_before, (
            f"Bullish shock should raise Tech price: before={price_before:.2f}, "
            f"after={price_after:.2f}."
        )

    def test_acute_bearish_shock_lowers_price(self):
        onset = 5
        shock = ShockDefinition(
            onset_tick=onset,
            magnitude=-0.15,
            duration=1,
            affected_sectors=[TECH],
            shock_type="acute",
            channel="market",
            reversion=False,
        )
        sim, persona_seed = make_run(total_ticks=20, shocks=[shock])
        result = sim.run(numpy_seed=1, persona_seed=persona_seed, run_number=1)

        price_before = next(
            (s.prices[TECH] for s in result.tick_snapshots if s.tick == onset - 1), None
        )
        price_at = next(
            (s.prices[TECH] for s in result.tick_snapshots if s.tick == onset), None
        )
        assert price_at < price_before

    def test_unaffected_sectors_not_impacted_by_shock(self):
        """Shock on TECH should not directly move Energy (sector 0)."""
        shock = ShockDefinition(
            onset_tick=5,
            magnitude=0.20,
            duration=5,
            affected_sectors=[TECH],
            shock_type="chronic",
            channel="market",
            reversion=False,
        )
        sim, persona_seed = make_run(total_ticks=30, shocks=[shock])
        result = sim.run(numpy_seed=2, persona_seed=persona_seed, run_number=1)

        price_at_onset = next(
            (s.prices[0] for s in result.tick_snapshots if s.tick == 5), None
        )
        # Energy price should stay close to P0 at onset (no direct shock).
        assert abs(price_at_onset - P0) < 5.0, (
            f"Energy price {price_at_onset:.2f} should not be directly impacted "
            "by a Tech-only shock."
        )

    def test_chronic_shock_produces_sustained_drift(self):
        """
        A chronic market shock should produce a gradual drift over its duration,
        with price higher at end of shock vs start.
        """
        onset = 10
        duration = 20
        shock = ShockDefinition(
            onset_tick=onset,
            magnitude=0.20,
            duration=duration,
            affected_sectors=[TECH],
            shock_type="chronic",
            channel="market",
            reversion=False,
        )
        sim, persona_seed = make_run(total_ticks=40, shocks=[shock])
        result = sim.run(numpy_seed=3, persona_seed=persona_seed, run_number=1)

        price_at_start = next(
            (s.prices[TECH] for s in result.tick_snapshots if s.tick == onset), None
        )
        price_at_end = next(
            (s.prices[TECH] for s in result.tick_snapshots
             if s.tick == onset + duration - 1), None
        )
        assert price_at_end > price_at_start, (
            "Chronic bullish shock should produce sustained upward price drift."
        )


# ── RunResult structure ───────────────────────────────────────────────────────

class TestRunResultStructure:
    def test_result_has_correct_final_prices_length(self):
        sim, persona_seed = make_run(total_ticks=10)
        result = sim.run(0, persona_seed, 1)
        assert len(result.final_prices) == NUM_SECTORS
        assert len(result.final_mispricing) == NUM_SECTORS

    def test_final_prices_positive(self):
        sim, persona_seed = make_run(total_ticks=20)
        result = sim.run(0, persona_seed, 1)
        assert all(p > 0 for p in result.final_prices)

    def test_tick_snapshots_logged_in_full_fidelity(self):
        """full_fidelity mode should log every tick."""
        total = 10
        sim, persona_seed = make_run(total_ticks=total)
        result = sim.run(0, persona_seed, 1)
        assert len(result.tick_snapshots) == total

    def test_seed_record_stored_correctly(self):
        sim, persona_seed = make_run(total_ticks=10)
        result = sim.run(numpy_seed=1234, persona_seed=persona_seed, run_number=3)
        assert result.seed_record.numpy_seed == 1234
        assert result.seed_record.persona_seed == persona_seed
        assert result.seed_record.run_number == 3

    def test_type_summaries_cover_all_types(self):
        """Each logged tick should have one TypeActionSummary per agent type."""
        from backend.config.sector_config import AGENT_TYPES
        sim, persona_seed = make_run(total_ticks=10)
        result = sim.run(0, persona_seed, 1)
        # Full fidelity: 10 ticks × 8 types = 80 summaries.
        assert len(result.type_summaries) == 10 * 8
        types_logged = {s.agent_type for s in result.type_summaries}
        assert types_logged == set(AGENT_TYPES)

    def test_activity_ratio_in_valid_range(self):
        sim, persona_seed = make_run(total_ticks=10)
        result = sim.run(0, persona_seed, 1)
        for s in result.type_summaries:
            assert 0.0 <= s.activity_ratio <= 1.0, (
                f"{s.agent_type} tick {s.tick}: activity_ratio {s.activity_ratio} out of range."
            )

    def test_deterministic_with_same_seed(self):
        """Two runs with identical seeds must produce identical results."""
        sim1, persona_seed = make_run(total_ticks=15)
        sim2, _ = make_run(total_ticks=15)
        r1 = sim1.run(numpy_seed=99, persona_seed=persona_seed, run_number=1)
        r2 = sim2.run(numpy_seed=99, persona_seed=persona_seed, run_number=1)
        assert r1.final_prices == r2.final_prices, (
            "Identical seeds should produce identical final prices."
        )
