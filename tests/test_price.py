"""
Unit tests for price formation (backend/simulation/price.py and environment.py).
"""

import math
import numpy as np
import pytest

from backend.config.constants import P0
from backend.config.sector_config import NUM_SECTORS, LAMBDA_DEFAULTS
from backend.simulation.price import (
    compute_log_price_update,
    apply_log_price_update,
    compute_mispricing,
)
from backend.simulation.environment import MarketEnvironment


class TestLogPriceUpdate:
    def test_positive_excess_demand_raises_price(self):
        lambdas = np.array(LAMBDA_DEFAULTS)
        ed = np.zeros(NUM_SECTORS)
        ed[7] = 100.0  # net buying in Tech
        delta = compute_log_price_update(lambdas, ed, total_agent_count=240)
        assert delta[7] > 0, "Positive excess demand should increase log-price."
        assert delta[0] == 0.0, "Unaffected sectors should not change."

    def test_negative_excess_demand_lowers_price(self):
        lambdas = np.array(LAMBDA_DEFAULTS)
        ed = np.zeros(NUM_SECTORS)
        ed[7] = -100.0
        delta = compute_log_price_update(lambdas, ed, total_agent_count=240)
        assert delta[7] < 0

    def test_zero_excess_demand_no_change(self):
        lambdas = np.array(LAMBDA_DEFAULTS)
        ed = np.zeros(NUM_SECTORS)
        delta = compute_log_price_update(lambdas, ed, total_agent_count=240)
        np.testing.assert_array_equal(delta, np.zeros(NUM_SECTORS))

    def test_prices_remain_positive(self):
        """Realistic excess demand should not produce non-positive prices.
        The log-linear mechanism uses exp(), which is always > 0 for finite values.
        Test with a realistic worst case: 30% of 240 agents trade 20% of avg AUM
        in one tick, all selling the same sector."""
        lambdas = np.array(LAMBDA_DEFAULTS)
        log_prices = np.log(np.full(NUM_SECTORS, P0))
        avg_aum = 536.0   # rough mean across types (B USD)
        active_agents = int(0.30 * 240)
        realistic_extreme = active_agents * avg_aum * 0.20  # B
        ed = np.full(NUM_SECTORS, -realistic_extreme)
        delta = compute_log_price_update(lambdas, ed, total_agent_count=240)
        new_log, new_prices = apply_log_price_update(log_prices, delta)
        assert np.all(new_prices > 0), "Prices must remain positive for realistic magnitudes."

    def test_lambda_scaling(self):
        """Higher lambda → larger price impact for same excess demand."""
        ed = np.zeros(NUM_SECTORS)
        ed[0] = 100.0  # Energy (lambda=0.15)
        ed[10] = 100.0  # Real Estate (lambda=0.25)
        delta = compute_log_price_update(np.array(LAMBDA_DEFAULTS), ed, 240)
        assert delta[10] > delta[0], "Higher lambda should produce larger price move."


class TestMispricing:
    def test_at_p0_mispricing_is_zero(self):
        prices = np.full(NUM_SECTORS, P0)
        mp = compute_mispricing(prices)
        np.testing.assert_array_almost_equal(mp, np.zeros(NUM_SECTORS))

    def test_above_p0_positive_mispricing(self):
        prices = np.full(NUM_SECTORS, 110.0)
        mp = compute_mispricing(prices)
        np.testing.assert_array_almost_equal(mp, np.full(NUM_SECTORS, 0.10))

    def test_below_p0_negative_mispricing(self):
        prices = np.full(NUM_SECTORS, 90.0)
        mp = compute_mispricing(prices)
        np.testing.assert_array_almost_equal(mp, np.full(NUM_SECTORS, -0.10))


class TestMarketEnvironment:
    def test_initial_prices_at_p0(self):
        env = MarketEnvironment()
        np.testing.assert_array_equal(env.prices, np.full(NUM_SECTORS, P0))

    def test_price_rises_with_positive_excess_demand(self):
        env = MarketEnvironment()
        env.accumulate_trade(7, 500.0)   # net buy in Tech
        env.apply_price_update(total_agent_count=240)
        assert env.prices[7] > P0, "Tech price should rise with net buying."
        assert env.prices[0] == pytest.approx(P0), "Energy price should be unchanged."

    def test_price_falls_with_negative_excess_demand(self):
        env = MarketEnvironment()
        env.accumulate_trade(7, -500.0)
        env.apply_price_update(total_agent_count=240)
        assert env.prices[7] < P0

    def test_volume_accumulates_correctly(self):
        env = MarketEnvironment()
        env.accumulate_trade(0, 100.0)
        env.accumulate_trade(0, -200.0)
        env.accumulate_trade(0, 50.0)
        assert env.volume[0] == pytest.approx(350.0)

    def test_excess_demand_clears_each_tick(self):
        env = MarketEnvironment()
        env.accumulate_trade(7, 500.0)
        env.apply_price_update(240)
        env.clear_tick_accumulators()
        assert env.excess_demand[7] == 0.0
        assert env.volume[7] == 0.0

    def test_volatility_increases_after_price_moves(self):
        env = MarketEnvironment()
        for i in range(25):
            env.clear_tick_accumulators()
            sign = 1 if i % 2 == 0 else -1
            env.accumulate_trade(7, sign * 500.0)
            env.apply_price_update(240)
        assert env.volatility[7] > 0.0, "Alternating prices should produce non-zero vol."

    def test_reset_returns_to_initial_state(self):
        env = MarketEnvironment()
        env.accumulate_trade(0, 1000.0)
        env.apply_price_update(240)
        env.reset()
        np.testing.assert_array_equal(env.prices, np.full(NUM_SECTORS, P0))
        assert env.tick == 0
