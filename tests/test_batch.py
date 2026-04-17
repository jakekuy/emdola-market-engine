"""
Phase 5 checkpoint — integration tests for the batch engine, aggregator,
storage, and sanity checks.

These tests run a small 3-run batch with dummy profiles (no LLM calls) and
verify that the full pipeline produces structurally valid output.

Note: each test creates its own temp directory for disk storage so tests
are isolated and leave no permanent state.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.config.sector_config import NUM_SECTORS, SECTORS, AGENT_TYPES
from backend.models.calibration import CalibrationInput, RunConfig, ShockDefinition
from backend.models.results import BatchResult
from backend.simulation.engine import BatchEngine
from backend.output.aggregator import aggregate_runs
from backend.output.storage import save_batch, load_batch, list_batches
from backend.validation.sanity_checks import run_all_checks

from tests.conftest import make_characteristics, make_persona
from tests.test_engine import make_fundamentalist_profile_set


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_batch_calibration(
    num_runs: int = 3,
    total_ticks: int = 20,
    shocks: list | None = None,
) -> CalibrationInput:
    return CalibrationInput(
        scenario_name="TestBatch",
        shocks=shocks or [],
        run=RunConfig(
            total_ticks=total_ticks,
            num_runs=num_runs,
            data_granularity="full_fidelity",
        ),
    )


def run_mini_batch(
    num_runs: int = 3,
    total_ticks: int = 20,
    shocks: list | None = None,
    save_dir: Path | None = None,
) -> BatchResult:
    calibration = make_batch_calibration(num_runs=num_runs, total_ticks=total_ticks, shocks=shocks)
    profiles = make_fundamentalist_profile_set()
    engine = BatchEngine(save_results=(save_dir is not None), max_parallel_workers=2, enable_narrative=False)
    if save_dir is not None:
        from backend.output import storage as _storage
        _storage.DEFAULT_DATA_DIR  # import side-effect only
        # Monkey-patch save location for this call.
        result = engine.run_batch(calibration, profiles, batch_id="test_batch_001")
        save_batch(result, base_dir=save_dir)
        return result
    return engine.run_batch(calibration, profiles, batch_id="test_batch_001")


# ── BatchEngine: structural correctness ──────────────────────────────────────

class TestBatchEngine:
    def test_produces_correct_run_count(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=3)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "b1")
        assert len(result.run_results) == 3

    def test_run_numbers_sequential(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=4)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "b2")
        run_nums = [r.run_number for r in result.run_results]
        assert run_nums == [1, 2, 3, 4]

    def test_seeds_are_unique_per_run(self):
        """Each run should have a distinct numpy seed (random policy)."""
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=5)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "b3")
        seeds = [r.seed_record.numpy_seed for r in result.run_results]
        assert len(set(seeds)) == len(seeds), "Each run should have a unique numpy seed."

    def test_fixed_seed_policy_reproducible(self):
        """
        Two batches with seed_policy='fixed' and the same fixed_seed should
        produce identical final prices on every run.
        """
        calibration = CalibrationInput(
            scenario_name="FixedSeedTest",
            run=RunConfig(
                total_ticks=15,
                num_runs=2,
                data_granularity="full_fidelity",
                seed_policy="fixed",
                fixed_seed=42,
            ),
        )
        profiles = make_fundamentalist_profile_set()
        engine1 = BatchEngine(save_results=False, enable_narrative=False)
        engine2 = BatchEngine(save_results=False, enable_narrative=False)
        r1 = engine1.run_batch(calibration, profiles, "fixed_a")
        r2 = engine2.run_batch(calibration, profiles, "fixed_b")
        for run_a, run_b in zip(r1.run_results, r2.run_results):
            assert run_a.final_prices == run_b.final_prices, (
                "Fixed-seed runs must be deterministic across batches."
            )

    def test_aggregate_only_mode_drops_run_results(self):
        calibration = CalibrationInput(
            scenario_name="AggOnly",
            run=RunConfig(
                total_ticks=10,
                num_runs=2,
                data_granularity="full_fidelity",
                storage_mode="aggregate_only",
            ),
        )
        profiles = make_fundamentalist_profile_set()
        engine = BatchEngine(save_results=False, enable_narrative=False)
        result = engine.run_batch(calibration, profiles, "agg_only")
        assert result.run_results == [], (
            "aggregate_only mode should clear per-run results after aggregation."
        )
        assert result.aggregated_stats is not None

    def test_batch_result_has_sanity_checks(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=2)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "sc_test")
        assert len(result.sanity_checks) == 5, "Should have 5 sanity checks."
        names = {sc.check_name for sc in result.sanity_checks}
        assert names == {"fundamentalist_reversion", "herding_amplification", "fat_tails", "cross_sector_correlation", "agent_type_divergence"}

    def test_replay_produces_identical_result(self):
        """
        Replaying a run with its stored seeds must produce the same final prices.
        """
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=1, total_ticks=15)
        profiles = make_fundamentalist_profile_set()
        batch = engine.run_batch(calibration, profiles, "replay_test")

        original = batch.run_results[0]
        replayed = engine.run_single(
            calibration=calibration,
            profiles=profiles,
            numpy_seed=original.seed_record.numpy_seed,
            run_number=1,
        )
        assert original.final_prices == replayed.final_prices, (
            "Replay must be fully deterministic."
        )


# ── Aggregator ────────────────────────────────────────────────────────────────

class TestAggregator:
    def _get_run_results(self, num_runs: int = 3) -> list:
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=num_runs)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "agg_unit")
        return result.run_results

    def test_aggregated_stats_has_all_sectors(self):
        runs = self._get_run_results()
        stats = aggregate_runs(runs)
        assert set(stats.mean_prices.keys()) == set(SECTORS)

    def test_price_trajectories_have_correct_length(self):
        """Each sector's price list should have one entry per logged tick."""
        runs = self._get_run_results()
        stats = aggregate_runs(runs)
        expected_ticks = len(runs[0].tick_snapshots)
        for sector, prices in stats.mean_prices.items():
            assert len(prices) == expected_ticks, (
                f"{sector}: expected {expected_ticks} ticks, got {len(prices)}."
            )

    def test_mean_within_p10_p90_bounds(self):
        """Mean price should lie between p10 and p90 for all sectors and ticks."""
        runs = self._get_run_results()
        stats = aggregate_runs(runs)
        for sector in SECTORS:
            mean = np.array(stats.mean_prices[sector])
            p10 = np.array(stats.p10_prices[sector])
            p90 = np.array(stats.p90_prices[sector])
            assert np.all(mean >= p10 - 1e-9), f"{sector}: mean < p10."
            assert np.all(mean <= p90 + 1e-9), f"{sector}: mean > p90."

    def test_narrative_input_is_valid_json(self):
        runs = self._get_run_results()
        stats = aggregate_runs(runs)
        parsed = json.loads(stats.narrative_input_json)
        assert "final_mispricing" in parsed
        assert "num_runs" in parsed

    def test_type_activity_covers_all_types(self):
        runs = self._get_run_results()
        stats = aggregate_runs(runs)
        assert set(stats.type_activity_summary.keys()) == set(AGENT_TYPES)


# ── Storage ───────────────────────────────────────────────────────────────────

class TestStorage:
    def test_save_and_load_round_trip(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=2)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "round_trip_batch")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            save_batch(result, base_dir=tmp)
            loaded = load_batch(result.batch_id, base_dir=tmp)

        assert loaded.batch_id == result.batch_id
        assert loaded.num_runs == result.num_runs

    def test_results_json_written(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=2)
        profiles = make_fundamentalist_profile_set()
        result = engine.run_batch(calibration, profiles, "json_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            batch_dir = save_batch(result, base_dir=tmp)
            assert (batch_dir / "results.json").exists()

    def test_list_batches_returns_saved_ids(self):
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=1)
        profiles = make_fundamentalist_profile_set()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for batch_id in ["batch_a", "batch_b"]:
                result = engine.run_batch(calibration, profiles, batch_id)
                # Override batch_id in the result for saving.
                result = result.model_copy(update={"batch_id": batch_id})
                save_batch(result, base_dir=tmp)
            saved = list_batches(base_dir=tmp)
        assert set(saved) == {"batch_a", "batch_b"}

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_batch("nonexistent_batch", base_dir=Path(tmpdir))


# ── Sanity checks ─────────────────────────────────────────────────────────────

class TestSanityChecks:
    def _get_batch(self, shocks=None) -> BatchResult:
        engine = BatchEngine(save_results=False, enable_narrative=False)
        calibration = make_batch_calibration(num_runs=2, total_ticks=30, shocks=shocks)
        profiles = make_fundamentalist_profile_set()
        return engine.run_batch(calibration, profiles, "sanity_test")

    def test_all_three_checks_present(self):
        batch = self._get_batch()
        names = {sc.check_name for sc in batch.sanity_checks}
        assert names == {"fundamentalist_reversion", "herding_amplification", "fat_tails", "cross_sector_correlation", "agent_type_divergence"}

    def test_fundamentalist_reversion_passes_without_shocks(self):
        batch = self._get_batch()
        rev_check = next(
            sc for sc in batch.sanity_checks
            if sc.check_name == "fundamentalist_reversion"
        )
        assert rev_check.passed, (
            "Fundamentalist agents without shocks should keep prices near P0."
        )

    def test_sanity_check_results_have_messages(self):
        batch = self._get_batch()
        for sc in batch.sanity_checks:
            assert sc.message, f"{sc.check_name} has no message."
