"""
Unit tests for signal computation (backend/simulation/signals.py).

Tests cover: fundamentalist signal, chartist signal, decay weights,
influence signal, background influence, sector affinity, and composite signal.
"""

import math
import numpy as np
import pytest
from collections import deque

from backend.config.constants import P0, SIGMA_BACKGROUND
from backend.config.sector_config import NUM_SECTORS, AGENT_TYPES
from backend.simulation.signals import (
    compute_fundamentalist_signal,
    compute_decay_weights,
    compute_chartist_signal,
    compute_influence_signal,
    compute_background_influence,
    compute_sector_affinity_weights,
    compute_composite_signal,
)


# ── Fundamentalist signal ─────────────────────────────────────────────────────

class TestFundamentalistSignal:
    def test_positive_when_below_p0(self):
        F = compute_fundamentalist_signal(90.0)
        assert F > 0, "Below P0: undervalued → buy signal (positive)"

    def test_negative_when_above_p0(self):
        F = compute_fundamentalist_signal(110.0)
        assert F < 0, "Above P0: overvalued → sell signal (negative)"

    def test_zero_at_p0(self):
        F = compute_fundamentalist_signal(P0)
        assert F == pytest.approx(0.0)

    def test_magnitude_scales_with_deviation(self):
        F_small = compute_fundamentalist_signal(99.0)   # 1% below
        F_large = compute_fundamentalist_signal(80.0)   # 20% below
        assert abs(F_large) > abs(F_small)

    def test_symmetric_around_p0(self):
        F_below = compute_fundamentalist_signal(90.0)
        F_above = compute_fundamentalist_signal(110.0)
        # Asymmetric in formula ((P0 - p)/P0 vs (p - P0)/P0) but magnitude comparable.
        # Below P0 at 90: (100-90)/100 = 0.10
        # Above P0 at 110: (100-110)/100 = -0.10
        assert F_below == pytest.approx(-F_above, abs=1e-6)


# ── Decay weights ─────────────────────────────────────────────────────────────

class TestDecayWeights:
    def test_sum_to_one(self):
        for memory_ticks in [5, 20, 50]:
            w = compute_decay_weights(memory_ticks, recency_bias=0.5, is_institutional=False)
            assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_institutional_sum_to_one(self):
        w = compute_decay_weights(20, recency_bias=0.5, is_institutional=True)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_retail_most_recent_heaviest(self):
        """Weight[0] (most recent) should be highest for retail with positive recency_bias."""
        w = compute_decay_weights(10, recency_bias=0.5, is_institutional=False)
        assert w[0] >= w[1] >= w[-1], "Retail weights should decrease from most recent."

    def test_institutional_most_recent_heaviest(self):
        """Power law: w_j = (j+1)^(-α) → index 0 (j+1=1) has highest weight."""
        w = compute_decay_weights(10, recency_bias=0.5, is_institutional=True)
        assert w[0] >= w[1], "Institutional power-law weights should decrease from most recent."

    def test_higher_recency_bias_steepens_retail(self):
        """Higher recency_bias → smaller τ → steeper decay → larger ratio w[0]/w[-1]."""
        w_low = compute_decay_weights(10, recency_bias=0.1, is_institutional=False)
        w_high = compute_decay_weights(10, recency_bias=0.9, is_institutional=False)
        ratio_low = w_low[0] / w_low[-1]
        ratio_high = w_high[0] / w_high[-1]
        assert ratio_high > ratio_low

    def test_zero_ticks_returns_unit(self):
        w = compute_decay_weights(0, recency_bias=0.5, is_institutional=False)
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0)

    def test_length_matches_memory_ticks(self):
        for n in [1, 5, 25]:
            w = compute_decay_weights(n, recency_bias=0.5, is_institutional=False)
            assert len(w) == n


# ── Chartist signal ───────────────────────────────────────────────────────────

class TestChartistSignal:
    def _make_memory(self, prices: list[float]) -> deque:
        mem = deque(maxlen=len(prices) + 5)
        for p in prices:
            mem.append(p)
        return mem

    def test_positive_in_uptrend(self):
        prices = [100.0 + i for i in range(10)]
        mem = self._make_memory(prices)
        C = compute_chartist_signal(mem, memory_ticks=9, recency_bias=0.3,
                                    time_horizon=0.5, is_institutional=False, beta_c=0.5)
        assert C > 0, "Consistent uptrend should give positive chartist signal."

    def test_negative_in_downtrend(self):
        prices = [110.0 - i for i in range(10)]
        mem = self._make_memory(prices)
        C = compute_chartist_signal(mem, memory_ticks=9, recency_bias=0.3,
                                    time_horizon=0.5, is_institutional=False, beta_c=0.5)
        assert C < 0, "Consistent downtrend should give negative chartist signal."

    def test_bounded_by_tanh(self):
        # Very steep uptrend — signal should still be in (-1, +1).
        prices = [100.0 * (1.05 ** i) for i in range(20)]
        mem = self._make_memory(prices)
        C = compute_chartist_signal(mem, memory_ticks=19, recency_bias=0.5,
                                    time_horizon=0.0, is_institutional=False, beta_c=0.5)
        assert -1.0 < C < 1.0

    def test_flat_prices_near_zero(self):
        prices = [100.0] * 10
        mem = self._make_memory(prices)
        C = compute_chartist_signal(mem, memory_ticks=9, recency_bias=0.3,
                                    time_horizon=0.5, is_institutional=False, beta_c=0.5)
        assert abs(C) < 1e-9, "Flat prices should produce zero chartist signal."

    def test_long_horizon_attenuates(self):
        """Agent with longer time_horizon discounts short-term momentum."""
        prices = [100.0 + i for i in range(10)]
        mem = self._make_memory(prices)
        C_short = compute_chartist_signal(mem, memory_ticks=9, recency_bias=0.3,
                                          time_horizon=0.0, is_institutional=False, beta_c=0.5)
        C_long = compute_chartist_signal(mem, memory_ticks=9, recency_bias=0.3,
                                         time_horizon=1.0, is_institutional=False, beta_c=0.5)
        assert abs(C_long) < abs(C_short), "Long-horizon agent should have smaller chartist signal."

    def test_insufficient_history_returns_zero(self):
        mem = deque(maxlen=10)
        mem.append(100.0)  # Only one price — no return computable.
        C = compute_chartist_signal(mem, memory_ticks=5, recency_bias=0.3,
                                    time_horizon=0.5, is_institutional=False, beta_c=0.5)
        assert C == 0.0


# ── Influence signal ──────────────────────────────────────────────────────────

class TestInfluenceSignal:
    def test_positive_bullish_signal(self):
        H = compute_influence_signal(
            broadcast_direction=1.0,
            broadcast_intensity=0.8,
            herding_sensitivity=0.5,
        )
        assert H > 0

    def test_negative_bearish_signal(self):
        H = compute_influence_signal(
            broadcast_direction=-1.0,
            broadcast_intensity=0.8,
            herding_sensitivity=0.5,
        )
        assert H < 0

    def test_zero_when_no_broadcast(self):
        H = compute_influence_signal(0.0, 0.5, 0.5)
        assert H == 0.0

    def test_scales_with_herding_sensitivity(self):
        H_low = compute_influence_signal(1.0, 1.0, 0.1)
        H_high = compute_influence_signal(1.0, 1.0, 0.9)
        assert H_high > H_low

    def test_scales_with_intensity(self):
        H_weak = compute_influence_signal(1.0, 0.2, 0.5)
        H_strong = compute_influence_signal(1.0, 0.8, 0.5)
        assert H_strong > H_weak


# ── Background influence ──────────────────────────────────────────────────────

class TestBackgroundInfluence:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        dirs, intensities = compute_background_influence(0.5, rng)
        assert dirs.shape == (NUM_SECTORS,)
        assert intensities.shape == (NUM_SECTORS,)

    def test_intensities_clipped_0_1(self):
        rng = np.random.default_rng(0)
        _, intensities = compute_background_influence(1.0, rng)
        assert np.all(intensities >= 0.0)
        assert np.all(intensities <= 1.0)

    def test_directions_are_signs(self):
        rng = np.random.default_rng(1)
        dirs, _ = compute_background_influence(0.5, rng)
        # Directions should be in {-1, 0, +1}.
        assert np.all(np.isin(dirs, [-1.0, 0.0, 1.0]))

    def test_zero_intensity_gives_zero_intensities(self):
        # Direction is always computed from the noise draw (sign of random normal).
        # When non_market_signal_intensity=0, intensities are zero — the direction
        # value is irrelevant since agents multiply direction × intensity.
        rng = np.random.default_rng(2)
        _, intensities = compute_background_influence(0.0, rng)
        np.testing.assert_array_equal(intensities, np.zeros(NUM_SECTORS))


# ── Sector affinity weights ───────────────────────────────────────────────────

class TestSectorAffinityWeights:
    def _empty_memories(self) -> list[deque]:
        return [deque(maxlen=10) for _ in range(NUM_SECTORS)]

    def test_normalised_mean_equals_one(self):
        """After normalisation, mean affinity across sectors should equal 1.0."""
        for agent_type in AGENT_TYPES:
            A = compute_sector_affinity_weights(
                agent_type=agent_type,
                risk_aversion=0.5,
                belief_formation=0.5,
                time_horizon=0.5,
                information_quality=0.5,
                price_memory_all=self._empty_memories(),
                current_prices=np.full(NUM_SECTORS, P0),
            )
            assert A.mean() == pytest.approx(1.0, abs=1e-9), (
                f"{agent_type}: mean affinity should be 1.0, got {A.mean()}"
            )

    def test_all_positive(self):
        for agent_type in AGENT_TYPES:
            A = compute_sector_affinity_weights(
                agent_type=agent_type,
                risk_aversion=0.5,
                belief_formation=0.5,
                time_horizon=0.5,
                information_quality=0.5,
                price_memory_all=self._empty_memories(),
                current_prices=np.full(NUM_SECTORS, P0),
            )
            assert np.all(A > 0), f"{agent_type}: all affinities should be positive."

    def test_correct_length(self):
        A = compute_sector_affinity_weights(
            agent_type="R1",
            risk_aversion=0.5,
            belief_formation=0.5,
            time_horizon=0.5,
            information_quality=0.5,
            price_memory_all=self._empty_memories(),
            current_prices=np.full(NUM_SECTORS, P0),
        )
        assert len(A) == NUM_SECTORS

    def test_high_info_quality_amplifies_spread(self):
        """High IQ agents should have more concentrated affinities (larger spread)."""
        kwargs = dict(
            agent_type="I1",
            risk_aversion=0.5,
            belief_formation=0.5,
            time_horizon=0.5,
            price_memory_all=self._empty_memories(),
            current_prices=np.full(NUM_SECTORS, P0),
        )
        A_low_iq = compute_sector_affinity_weights(information_quality=0.0, **kwargs)
        A_high_iq = compute_sector_affinity_weights(information_quality=1.0, **kwargs)
        spread_low = A_low_iq.max() - A_low_iq.min()
        spread_high = A_high_iq.max() - A_high_iq.min()
        assert spread_high >= spread_low, "High IQ should amplify affinity spread."


# ── Composite signal ──────────────────────────────────────────────────────────

class TestCompositeSignal:
    def _make_memory(self, prices: list[float]) -> deque:
        mem = deque(maxlen=50)
        for p in prices:
            mem.append(p)
        return mem

    def test_fundamentalist_dominates_below_p0(self):
        """Pure fundamentalist agent sees positive signal when price < P0."""
        rng = np.random.default_rng(42)
        S = compute_composite_signal(
            beta_f=1.0, beta_c=0.0, herding_sensitivity=0.0,
            information_quality=1.0, recency_bias=0.3, time_horizon=0.5,
            is_institutional=False, memory_ticks=10,
            price_k=80.0,
            price_memory_k=self._make_memory([80.0] * 5),
            broadcast_direction_k=0.0, broadcast_intensity_k=0.0,
            sector_affinity_k=1.0,
            rng=rng,
        )
        assert S > 0, "Fundamentalist: price below P0 should yield positive signal."

    def test_chartist_dominates_uptrend(self):
        """Pure chartist agent sees positive signal in an uptrend."""
        rng = np.random.default_rng(7)
        prices = [90.0 + i * 2 for i in range(10)]
        S = compute_composite_signal(
            beta_f=0.0, beta_c=1.0, herding_sensitivity=0.0,
            information_quality=1.0, recency_bias=0.5, time_horizon=0.0,
            is_institutional=False, memory_ticks=9,
            price_k=prices[-1],
            price_memory_k=self._make_memory(prices),
            broadcast_direction_k=0.0, broadcast_intensity_k=0.0,
            sector_affinity_k=1.0,
            rng=rng,
        )
        assert S > 0, "Chartist: uptrend should yield positive signal."

    def test_herding_signal_positive_on_bullish_broadcast(self):
        """Herding agent follows bullish broadcast."""
        rng = np.random.default_rng(99)
        S = compute_composite_signal(
            beta_f=0.0, beta_c=0.0, herding_sensitivity=1.0,
            information_quality=1.0, recency_bias=0.3, time_horizon=0.5,
            is_institutional=False, memory_ticks=5,
            price_k=P0,
            price_memory_k=self._make_memory([P0] * 3),
            broadcast_direction_k=1.0, broadcast_intensity_k=0.9,
            sector_affinity_k=1.0,
            rng=rng,
        )
        assert S > 0

    def test_affinity_scales_signal(self):
        """Higher sector affinity should scale up signal magnitude."""
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)  # same seed
        kwargs = dict(
            beta_f=0.5, beta_c=0.0, herding_sensitivity=0.0,
            information_quality=1.0, recency_bias=0.3, time_horizon=0.5,
            is_institutional=False, memory_ticks=5,
            price_k=90.0,
            price_memory_k=self._make_memory([90.0] * 5),
            broadcast_direction_k=0.0, broadcast_intensity_k=0.0,
        )
        S_low = compute_composite_signal(**kwargs, sector_affinity_k=0.5, rng=rng_a)
        S_high = compute_composite_signal(**kwargs, sector_affinity_k=2.0, rng=rng_b)
        assert abs(S_high) > abs(S_low), "Higher affinity should amplify signal magnitude."

    def test_zero_information_quality_produces_noise_only(self):
        """With IQ=0, signal is purely noise; no coherent direction guaranteed."""
        rng = np.random.default_rng(123)
        # No meaningful assert — just confirm it doesn't raise and returns a float.
        S = compute_composite_signal(
            beta_f=0.5, beta_c=0.5, herding_sensitivity=0.5,
            information_quality=0.0, recency_bias=0.3, time_horizon=0.5,
            is_institutional=False, memory_ticks=5,
            price_k=P0, price_memory_k=deque([P0] * 3, maxlen=10),
            broadcast_direction_k=0.0, broadcast_intensity_k=0.0,
            sector_affinity_k=1.0, rng=rng,
        )
        assert isinstance(S, float)
