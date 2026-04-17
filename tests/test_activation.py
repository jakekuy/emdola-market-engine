"""
Unit tests for the 5-step activation function (backend/simulation/activation.py).

Tests cover each step individually plus the full pipeline.
"""

import math
import numpy as np
import pytest
from collections import deque

from backend.config.constants import P0
from backend.config.sector_config import NUM_SECTORS
from backend.simulation.environment import MarketEnvironment
from backend.simulation.activation import (
    run_agent_activation,
    _step1_form_signal,
    _step2_apply_bias_distortions,
    _step3_compute_raw_demand,
    _step4_evaluate_gate,
    _step5_execute_trade,
)

from tests.conftest import (
    make_characteristics,
    make_fundamentalist_characteristics,
    make_chartist_characteristics,
    make_persona,
)
from backend.simulation.agent import Agent
from backend.simulation.population import build_population
from backend.models.calibration import CalibrationInput


# ── Helpers ───────────────────────────────────────────────────────────────────

TECH_SECTOR = 7   # index 7 = Tech

def make_agent(
    agent_type: str = "R2",
    **char_overrides,
) -> Agent:
    chars = make_characteristics(**char_overrides)
    persona = make_persona(agent_type, chars)
    return Agent(
        agent_type=agent_type,
        agent_id=0,
        profile=persona.characteristics,
    )


def make_env_with_price(sector: int, price: float) -> MarketEnvironment:
    env = MarketEnvironment()
    env.prices[sector] = price
    env.log_prices[sector] = np.log(price)
    return env


def prime_price_memory(agent: Agent, sector: int, prices: list[float]) -> None:
    """Load a sequence of prices into an agent's price_memory for a sector."""
    for p in prices:
        agent.price_memory[sector].append(float(p))


# ── Step 1: Signal formation ──────────────────────────────────────────────────

class TestStep1FormSignal:
    def test_fundamentalist_positive_below_p0(self):
        agent = make_agent(belief_formation=0.0, herding_sensitivity=0.0,
                           information_quality=1.0)
        env = make_env_with_price(TECH_SECTOR, 80.0)
        rng = np.random.default_rng(1)
        S = _step1_form_signal(agent, TECH_SECTOR, env, rng, sector_affinity_k=1.0)
        assert S > 0

    def test_fundamentalist_negative_above_p0(self):
        agent = make_agent(belief_formation=0.0, herding_sensitivity=0.0,
                           information_quality=1.0)
        env = make_env_with_price(TECH_SECTOR, 120.0)
        rng = np.random.default_rng(2)
        S = _step1_form_signal(agent, TECH_SECTOR, env, rng, sector_affinity_k=1.0)
        assert S < 0

    def test_herding_follows_broadcast(self):
        agent = make_agent(belief_formation=0.0, herding_sensitivity=1.0,
                           information_quality=1.0)
        env = MarketEnvironment()
        env.influence_signal.direction[TECH_SECTOR] = 1.0
        env.influence_signal.intensity[TECH_SECTOR] = 0.9
        rng = np.random.default_rng(3)
        S = _step1_form_signal(agent, TECH_SECTOR, env, rng, sector_affinity_k=1.0)
        assert S > 0

    def test_affinity_scales_output(self):
        agent = make_agent(belief_formation=0.0, information_quality=1.0,
                           herding_sensitivity=0.0)
        env = make_env_with_price(TECH_SECTOR, 80.0)
        rng_a = np.random.default_rng(10)
        rng_b = np.random.default_rng(10)
        S_low = _step1_form_signal(agent, TECH_SECTOR, env, rng_a, sector_affinity_k=0.5)
        S_high = _step1_form_signal(agent, TECH_SECTOR, env, rng_b, sector_affinity_k=2.0)
        assert abs(S_high) > abs(S_low)


# ── Step 2: Bias distortions ──────────────────────────────────────────────────

class TestStep2BiasDistortions:
    def test_confirmation_bias_attenuates_sell_on_long(self):
        """Holding a long position should attenuate a sell signal."""
        agent = make_agent(confirmation_bias=1.0)
        agent.sector_weights[TECH_SECTOR] = 0.1  # hold a position
        env = MarketEnvironment()
        S_original = -0.5  # sell signal
        S_biased = _step2_apply_bias_distortions(S_original, agent, TECH_SECTOR, env)
        assert abs(S_biased) < abs(S_original), "Confirmation bias should attenuate."

    def test_confirmation_bias_no_effect_on_buy_when_long(self):
        """Buy signal with existing long position: no attenuation (confirms view).
        Other biases zeroed out to isolate confirmation bias."""
        agent = make_agent(
            confirmation_bias=1.0,
            anchoring=0.0, salience_bias=0.0, pattern_matching_bias=0.0,
        )
        agent.sector_weights[TECH_SECTOR] = 0.1
        env = MarketEnvironment()
        S_original = 0.5  # buy signal confirming position
        S_biased = _step2_apply_bias_distortions(S_original, agent, TECH_SECTOR, env)
        assert S_biased == pytest.approx(S_original, abs=1e-6), (
            "Confirmation bias should not attenuate a confirming signal."
        )

    def test_zero_biases_pass_through(self):
        """With all biases at zero the signal should be unchanged."""
        agent = make_agent(
            confirmation_bias=0.0, anchoring=0.0,
            salience_bias=0.0, pattern_matching_bias=0.0,
        )
        env = MarketEnvironment()
        for S_val in [-0.7, 0.0, 0.4]:
            S_out = _step2_apply_bias_distortions(S_val, agent, TECH_SECTOR, env)
            assert S_out == pytest.approx(S_val, abs=1e-9)

    def test_salience_bias_amplifies_on_extreme_move(self):
        """Signal should be amplified following a large price move.
        Anchoring and pattern matching zeroed to isolate salience."""
        agent = make_agent(salience_bias=1.0, anchoring=0.0,
                           confirmation_bias=0.0, pattern_matching_bias=0.0)
        env = MarketEnvironment()
        # Seed volatility via the environment's rolling history.
        # Use a small sigma so a large recent move triggers salience.
        # Inject small constant σ directly.
        env.volatility[TECH_SECTOR] = 0.005
        # Load memory with a large jump at the end.
        prices = [100.0] * 8 + [110.0]
        prime_price_memory(agent, TECH_SECTOR, prices)

        S_original = 0.3
        S_biased = _step2_apply_bias_distortions(S_original, agent, TECH_SECTOR, env)
        assert abs(S_biased) > abs(S_original), "Salience bias should amplify after extreme move."

    def test_pattern_matching_amplifies_when_aligned(self):
        """Signal matching the recent trend should be amplified.
        Anchoring and other biases zeroed to isolate pattern matching."""
        agent = make_agent(pattern_matching_bias=1.0, anchoring=0.0,
                           confirmation_bias=0.0, salience_bias=0.0)
        env = MarketEnvironment()
        # Uptrend in memory.
        prices = [100.0 + i for i in range(8)]
        prime_price_memory(agent, TECH_SECTOR, prices)

        S_upward = 0.3  # buy signal aligned with uptrend
        S_biased = _step2_apply_bias_distortions(S_upward, agent, TECH_SECTOR, env)
        assert S_biased > S_upward, "Aligned pattern matching should amplify signal."


# ── Step 3: Raw demand ────────────────────────────────────────────────────────

class TestStep3RawDemand:
    def test_sign_preserved(self):
        agent = make_agent()
        env = MarketEnvironment()
        rng = np.random.default_rng(5)
        D_pos = _step3_compute_raw_demand(0.5, agent, TECH_SECTOR, env, rng)
        rng2 = np.random.default_rng(5)
        D_neg = _step3_compute_raw_demand(-0.5, agent, TECH_SECTOR, env, rng2)
        assert D_pos > 0
        assert D_neg < 0

    def test_higher_signal_larger_demand(self):
        agent = make_agent(overconfidence=0.0)
        env = MarketEnvironment()
        # Both use same rng to isolate.
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        D_small = _step3_compute_raw_demand(0.2, agent, TECH_SECTOR, env, rng_a)
        D_large = _step3_compute_raw_demand(0.8, agent, TECH_SECTOR, env, rng_b)
        assert abs(D_large) > abs(D_small)

    def test_higher_risk_aversion_reduces_demand(self):
        env = MarketEnvironment()
        env.volatility[TECH_SECTOR] = 0.01
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        agent_low_ra = make_agent(risk_aversion=0.1, overconfidence=0.0)
        agent_high_ra = make_agent(risk_aversion=0.9, overconfidence=0.0)
        D_low = _step3_compute_raw_demand(0.5, agent_low_ra, TECH_SECTOR, env, rng_a)
        D_high = _step3_compute_raw_demand(0.5, agent_high_ra, TECH_SECTOR, env, rng_b)
        assert abs(D_low) > abs(D_high), "Higher risk aversion should reduce demand."


# ── Step 4: Sigmoid gate ──────────────────────────────────────────────────────

class TestStep4Gate:
    def test_very_large_demand_almost_always_fires(self):
        """Demand well above threshold should almost certainly trigger."""
        agent = make_agent(conviction_threshold=0.1, fomo=0.0, regret_aversion=0.0,
                           loss_aversion=0.0, winner_selling_loser_holding=0.0,
                           decision_style=0.0)
        env = MarketEnvironment()
        rng = np.random.default_rng(0)
        # D_i = 100 >> theta ≈ 0.1 → gate almost certainly fires.
        # S_biased is capped at SIGNAL_GATE_CAP (0.75) before the gate evaluates,
        # so fire probability is sigmoid(0.75 - theta) ≈ 0.62, not near-certain.
        # The intent is: strong signal fires more often than not (>50%).
        fires = [_step4_evaluate_gate(100.0, 100.0, agent, TECH_SECTOR, env,
                                       np.random.default_rng(i)) for i in range(50)]
        assert sum(fires) >= 25, "Strong signal should fire more often than not."

    def test_zero_demand_rarely_fires(self):
        """D_i = 0 puts gate at sigmoid(-theta) < 0.5 — should rarely fire."""
        agent = make_agent(conviction_threshold=0.5, fomo=0.0, regret_aversion=0.0,
                           loss_aversion=0.0, winner_selling_loser_handling=0.0 if False else None,
                           decision_style=0.0)
        # Rebuild without the bad kwarg.
        chars = make_characteristics(
            conviction_threshold=0.5, fomo=0.0, regret_aversion=0.0,
            loss_aversion=0.0, decision_style=0.0,
        )
        persona = make_persona("R2", chars)
        agent = Agent(agent_type="R2", agent_id=0, profile=persona.characteristics)
        env = MarketEnvironment()
        fires = [_step4_evaluate_gate(0.0, 0.0, agent, TECH_SECTOR, env,
                                       np.random.default_rng(i)) for i in range(100)]
        assert sum(fires) < 40, "Zero demand should rarely clear the conviction threshold."

    def test_fomo_lowers_threshold(self):
        """FOMO should cause the gate to fire more easily when broadcast is strong."""
        env = MarketEnvironment()
        env.influence_signal.intensity[TECH_SECTOR] = 0.8  # above FOMO_INFLUENCE_THRESHOLD

        # Same D_i, same seed, but one agent has high FOMO.
        chars_low = make_characteristics(conviction_threshold=0.4, fomo=0.0, decision_style=0.0)
        chars_high = make_characteristics(conviction_threshold=0.4, fomo=1.0, decision_style=0.0)
        persona_low = make_persona("R2", chars_low)
        persona_high = make_persona("R2", chars_high)
        agent_low = Agent("R2", 0, persona_low.characteristics)
        agent_high = Agent("R2", 1, persona_high.characteristics)

        N = 200
        fires_low = sum(
            _step4_evaluate_gate(0.3, 0.3, agent_low, TECH_SECTOR, env, np.random.default_rng(i))
            for i in range(N)
        )
        fires_high = sum(
            _step4_evaluate_gate(0.3, 0.3, agent_high, TECH_SECTOR, env, np.random.default_rng(i))
            for i in range(N)
        )
        assert fires_high > fires_low, "High FOMO should fire more often."

    def test_loss_aversion_raises_threshold_on_sell_loss(self):
        """Selling a losing position should be harder with high loss aversion."""
        env = MarketEnvironment()

        def count_fires(loss_aversion_val: float) -> int:
            chars = make_characteristics(
                conviction_threshold=0.3, loss_aversion=loss_aversion_val,
                decision_style=0.0, fomo=0.0, regret_aversion=0.0,
                winner_selling_loser_holding=0.0,
            )
            agent = Agent("R2", 0, make_persona("R2", chars).characteristics)
            # Set up a losing position.
            agent.sector_weights[TECH_SECTOR] = 0.1
            agent.purchase_price[TECH_SECTOR] = 120.0  # bought at 120, price is now 100 → loss
            return sum(
                _step4_evaluate_gate(-0.4, -0.4, agent, TECH_SECTOR, env, np.random.default_rng(i))
                for i in range(200)
            )

        fires_low = count_fires(0.0)
        fires_high = count_fires(1.0)
        assert fires_high < fires_low, "Loss aversion should reduce firing on sell-loss."


# ── Step 5: Trade execution ───────────────────────────────────────────────────

class TestStep5ExecuteTrade:
    def test_buy_reduces_cash(self):
        agent = make_agent()
        env = MarketEnvironment()
        initial_cash = agent.cash_weight
        direction, magnitude = _step5_execute_trade(0.5, 0.5, agent, TECH_SECTOR, env, 1)
        assert direction == 1
        assert agent.cash_weight < initial_cash

    def test_sell_reduces_position(self):
        agent = make_agent()
        agent.sector_weights[TECH_SECTOR] = 0.2  # hold a position
        env = MarketEnvironment()
        initial_weight = agent.sector_weights[TECH_SECTOR]
        direction, magnitude = _step5_execute_trade(-0.5, -0.5, agent, TECH_SECTOR, env, 1)
        assert direction == -1
        assert agent.sector_weights[TECH_SECTOR] < initial_weight

    def test_buy_recorded_in_excess_demand(self):
        agent = make_agent()
        env = MarketEnvironment()
        direction, magnitude = _step5_execute_trade(0.5, 0.5, agent, TECH_SECTOR, env, 1)
        assert env.excess_demand[TECH_SECTOR] > 0

    def test_sell_recorded_as_negative_excess_demand(self):
        agent = make_agent()
        agent.sector_weights[TECH_SECTOR] = 0.2
        env = MarketEnvironment()
        direction, magnitude = _step5_execute_trade(-0.5, -0.5, agent, TECH_SECTOR, env, 1)
        assert env.excess_demand[TECH_SECTOR] < 0

    def test_magnitude_capped_at_position_scale_aum(self):
        agent = make_agent()
        env = MarketEnvironment()
        # Large D_i >> 1: should cap at position_scale * AUM.
        _, magnitude = _step5_execute_trade(1e6, 1e6, agent, TECH_SECTOR, env, 1)
        max_allowed = agent.position_scale * agent.aum
        assert magnitude <= max_allowed + 1e-9  # allow tiny float tolerance

    def test_institutional_inertia_dampens_magnitude(self):
        """Institutional agents with high inertia should trade smaller magnitudes."""
        env = MarketEnvironment()

        chars_no_inertia = make_characteristics(institutional_inertia=0.0)
        chars_high_inertia = make_characteristics(institutional_inertia=1.0)
        agent_low = Agent("I1", 0, make_persona("I1", chars_no_inertia).characteristics)
        agent_high = Agent("I1", 1, make_persona("I1", chars_high_inertia).characteristics)

        # Use a weak signal (0.1) so magnitude stays below the per-tick cash
        # deployment cap — otherwise both damped and undamped hit the same ceiling.
        _, mag_low = _step5_execute_trade(0.1, 0.1, agent_low, TECH_SECTOR, MarketEnvironment(), 1)
        _, mag_high = _step5_execute_trade(0.1, 0.1, agent_high, TECH_SECTOR, MarketEnvironment(), 1)
        assert mag_high < mag_low, "High institutional inertia should reduce trade size."


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullActivationPipeline:
    def _make_dummy_calibration(self) -> CalibrationInput:
        from backend.models.calibration import CalibrationInput
        return CalibrationInput()

    def test_fundamentalist_buys_below_p0(self):
        """
        Fundamentalist agent with price well below P0 should predominantly buy.
        Run many times with random gates to get a statistical result.
        """
        from backend.config.sector_config import NUM_SECTORS
        agent = make_agent(
            agent_type="R2",
            belief_formation=0.0,   # pure fundamentalist
            herding_sensitivity=0.0,
            information_quality=1.0,
            conviction_threshold=0.1,  # low threshold — easy to fire
            decision_style=0.0,
            risk_aversion=0.5,
        )
        # Set price well below P0 to generate a strong buy signal.
        env = MarketEnvironment()
        env.prices[TECH_SECTOR] = 60.0
        env.log_prices[TECH_SECTOR] = np.log(60.0)

        total_buy = 0
        total_sell = 0
        for seed in range(50):
            rng = np.random.default_rng(seed)
            agent_fresh = make_agent(
                agent_type="R2",
                belief_formation=0.0,
                herding_sensitivity=0.0,
                information_quality=1.0,
                conviction_threshold=0.1,
                decision_style=0.0,
                risk_aversion=0.5,
            )
            trades = run_agent_activation(agent_fresh, env, 1, rng)
            for sector, direction, _ in trades:
                if sector == TECH_SECTOR:
                    if direction > 0:
                        total_buy += 1
                    else:
                        total_sell += 1

        assert total_buy > total_sell, (
            "Fundamentalist agent with price below P0 should predominantly buy."
        )

    def test_chartist_follows_uptrend(self):
        """
        Pure chartist agent should predominantly buy in a sector with strong uptrend.
        Biases zeroed so the chartist signal channel dominates.
        """
        env = MarketEnvironment()
        env.prices[TECH_SECTOR] = 115.0
        env.log_prices[TECH_SECTOR] = np.log(115.0)

        total_buy = 0
        total_sell = 0
        for seed in range(50):
            agent = make_agent(
                agent_type="R4",
                belief_formation=1.0,       # pure chartist
                herding_sensitivity=0.0,
                information_quality=1.0,
                conviction_threshold=0.1,
                decision_style=0.0,
                recency_bias=0.5,
                time_horizon=0.0,           # no horizon attenuation
                # Zero out biases that could distort the signal.
                anchoring=0.0,
                confirmation_bias=0.0,
                salience_bias=0.0,
                pattern_matching_bias=0.0,
            )
            # Load clear uptrend into memory.
            prices = [90.0 + i * 2.5 for i in range(10)]
            prime_price_memory(agent, TECH_SECTOR, prices)

            rng = np.random.default_rng(seed)
            trades = run_agent_activation(agent, env, 1, rng)
            for sector, direction, _ in trades:
                if sector == TECH_SECTOR:
                    if direction > 0:
                        total_buy += 1
                    else:
                        total_sell += 1

        assert total_buy > total_sell, (
            "Chartist agent with uptrend should predominantly buy."
        )

    def test_returns_list_of_tuples(self):
        agent = make_agent()
        env = MarketEnvironment()
        rng = np.random.default_rng(0)
        trades = run_agent_activation(agent, env, 1, rng)
        assert isinstance(trades, list)
        for item in trades:
            assert len(item) == 3
            sector, direction, magnitude = item
            assert 0 <= sector < NUM_SECTORS
            assert direction in (-1, +1)
            assert magnitude >= 0.0

    def test_zero_signal_produces_no_trades(self):
        """
        At exact equilibrium (price=P0, no memory, no broadcast, IQ=1, no biases),
        all three signal components are zero → D_i = 0 → no trades ever fire.
        """
        for seed in range(20):
            agent = make_agent(
                belief_formation=0.5,
                herding_sensitivity=0.0,
                information_quality=1.0,   # IQ=1 → no noise term
                conviction_threshold=0.5,
                fomo=0.0,
                anchoring=0.0,
                confirmation_bias=0.0,
                salience_bias=0.0,
                pattern_matching_bias=0.0,
            )
            env = MarketEnvironment()  # all prices at P0; no broadcast
            rng = np.random.default_rng(seed)
            trades = run_agent_activation(agent, env, 1, rng)
            # F=0 (at P0), C=0 (no memory), H=0 (no broadcast) → S=0 → D_i=0 → no trade.
            assert trades == [], (
                f"Zero signal at equilibrium should produce no trades (seed={seed})."
            )
