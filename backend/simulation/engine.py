"""
BatchEngine — orchestrates a complete multi-run simulation batch.

Responsibilities:
  - Generate persona seeds and numpy seeds for each run.
  - Execute run 1 (the "display run") in a thread with an optional
    tick_callback for live WebSocket streaming.
  - Execute remaining runs in parallel via ProcessPoolExecutor (true CPU
    parallelism — bypasses the GIL for Python-bound simulation work).
  - Aggregate results, run sanity checks, save to disk.
  - Expose replay: re-execute any single run with its exact seed pair.

Spec references: §9.3 (seeds), §9.4 (saved artefacts), §11 (output).
"""

from __future__ import annotations

import os
import threading
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable

from backend.models.calibration import CalibrationInput
from backend.models.profile import ProfileSet
from backend.models.results import BatchResult, RunResult, SeedRecord
from backend.simulation.population import build_population
from backend.simulation.run import SimulationRun
from backend.output.aggregator import aggregate_runs
from backend.output.storage import save_batch
from backend.validation.sanity_checks import run_all_checks
from backend.config.sector_config import AGENT_TYPES, SECTORS


def _execute_run_subprocess(
    calibration: CalibrationInput,
    profiles: ProfileSet,
    seed_record: SeedRecord,
    run_number: int,
) -> RunResult:
    """
    Module-level function for ProcessPoolExecutor — must be picklable.

    Identical to BatchEngine._execute_run but without callbacks, so it can
    be submitted to a subprocess pool for true CPU parallelism.
    """
    agents, persona_seed = build_population(calibration, profiles, seed_record.numpy_seed)
    actual_seed = SeedRecord(
        run_number=seed_record.run_number,
        persona_seed=persona_seed,
        numpy_seed=seed_record.numpy_seed,
    )
    sim = SimulationRun(calibration=calibration, agents=agents)
    return sim.run(
        numpy_seed=actual_seed.numpy_seed,
        persona_seed=actual_seed.persona_seed,
        run_number=run_number,
        tick_callback=None,
        pause_event=None,
    )


class BatchEngine:
    """
    Executes a batch of simulation runs from a CalibrationInput + ProfileSet.

    Parameters
    ----------
    save_results : bool
        If True (default), saves the BatchResult to disk after completion.
    max_parallel_workers : int
        Maximum threads for background (non-display) runs.
    """

    def __init__(
        self,
        save_results: bool = True,
        max_parallel_workers: int | None = None,
        enable_narrative: bool = True,
    ) -> None:
        self.save_results = save_results
        # None → use all logical CPU cores (capped at 32 to avoid memory pressure).
        self.max_parallel_workers = max_parallel_workers or min(32, os.cpu_count() or 4)
        self.enable_narrative = enable_narrative

    # ── Main entry point ───────────────────────────────────────────────────────

    def run_batch(
        self,
        calibration: CalibrationInput,
        profiles: ProfileSet,
        batch_id: str,
        tick_callback: Callable[[dict], None] | None = None,
        background_tick_callback: Callable[[dict], None] | None = None,
        run_complete_callback: Callable | None = None,
        run_start_callback: Callable | None = None,
        pre_narrative_callback: Callable | None = None,
        narrative_status_callback: Callable[[str], None] | None = None,
        pause_event: threading.Event | None = None,
    ) -> BatchResult:
        """
        Run a full simulation batch and return the BatchResult.

        Run 1 is the "display run": it executes synchronously and sends
        tick events via tick_callback (for WebSocket streaming in the API).
        Runs 2..N execute in parallel in a ThreadPoolExecutor.

        Parameters
        ----------
        calibration :
            Full user-configured setup.
        profiles :
            LLM-generated (or dummy) agent personas.
        batch_id :
            Unique identifier for this batch — used as the disk folder name.
        tick_callback :
            Optional callable invoked after each tick of the display run.
            Signature: ``callback(data: dict) -> None``.
        """
        num_runs = calibration.run.num_runs
        seeds = self._generate_seeds(calibration, num_runs)

        run_results: list[RunResult] = []

        # ── Dispatch: run 1 in a thread (tick_callback), runs 2..N in processes ─
        # ProcessPoolExecutor spawns separate Python interpreters → bypasses the
        # GIL and gives true CPU parallelism for compute-bound simulation work.
        # Background tick callbacks cannot cross process boundaries — the live
        # spaghetti shows only run 1 during execution; all runs appear in results.
        workers = min(self.max_parallel_workers, num_runs)
        bg_workers = max(1, workers - 1)

        all_futures: dict = {}
        thread_exec = ThreadPoolExecutor(max_workers=1)
        proc_exec = ProcessPoolExecutor(max_workers=bg_workers) if num_runs > 1 else None

        try:
            for i, seed in enumerate(seeds):
                run_n = i + 1
                if run_n == 1:
                    future = thread_exec.submit(
                        self._execute_run,
                        calibration=calibration,
                        profiles=profiles,
                        seed_record=seed,
                        run_number=1,
                        tick_callback=tick_callback,
                        run_start_callback=run_start_callback,
                        pause_event=pause_event,
                    )
                else:
                    future = proc_exec.submit(
                        _execute_run_subprocess,
                        calibration,
                        profiles,
                        seed,
                        run_n,
                    )
                all_futures[future] = run_n

            for future in as_completed(all_futures):
                result = future.result()
                run_results.append(result)
                if run_complete_callback is not None:
                    run_complete_callback(result)
        finally:
            thread_exec.shutdown(wait=True)
            if proc_exec is not None:
                proc_exec.shutdown(wait=True)

        # Sort by run_number — parallel execution may return out of order.
        run_results.sort(key=lambda r: r.run_number)

        # ── Aggregate ─────────────────────────────────────────────────────────
        # Collect directly shocked sector indices before aggregation so they
        # can be passed for trace/shock-window enrichment of narrative input.
        _shocked_indices: list[int] = []
        for shock in calibration.shocks:
            for idx in shock.affected_sectors:
                if idx < len(SECTORS) and idx not in _shocked_indices:
                    _shocked_indices.append(idx)

        agg_stats = aggregate_runs(run_results, shocked_sector_indices=_shocked_indices)

        # ── Build interim BatchResult for sanity checks ───────────────────────
        # Collect directly shocked sector names (deduplicated, order-preserving).
        _shocked = [SECTORS[i] for i in _shocked_indices]

        batch = BatchResult(
            batch_id=batch_id,
            scenario_name=calibration.scenario_name,
            num_runs=num_runs,
            total_ticks=calibration.run.total_ticks,
            shocked_sectors=_shocked,
            run_results=run_results,
            aggregated_stats=agg_stats,
            sanity_checks=[],
            narrative="",
            persona_narrative_md="",
        )

        # ── Sanity checks ─────────────────────────────────────────────────────
        sanity = run_all_checks(batch)
        batch = batch.model_copy(update={"sanity_checks": sanity})

        # ── LLM narrative ─────────────────────────────────────────────────────
        if pre_narrative_callback is not None:
            pre_narrative_callback()

        narrative = ""
        if self.enable_narrative:
            narrative = self._generate_narrative(
                calibration, profiles, agg_stats, narrative_status_callback
            )
        if narrative:
            batch = batch.model_copy(update={"narrative": narrative})

        # ── Trim run-level detail if aggregate-only mode ───────────────────────
        if calibration.run.storage_mode == "aggregate_only":
            batch = batch.model_copy(update={"run_results": []})

        # ── Save to disk ───────────────────────────────────────────────────────
        if self.save_results:
            save_batch(batch)

        return batch

    # ── Replay ────────────────────────────────────────────────────────────────

    def run_single(
        self,
        calibration: CalibrationInput,
        profiles: ProfileSet,
        numpy_seed: int,
        run_number: int = 1,
        tick_callback: Callable[[dict], None] | None = None,
    ) -> RunResult:
        """
        Re-execute a single run from its numpy_seed.  Used by the Replay button.
        The population and all stochastic elements are reproduced from numpy_seed.
        The result is not saved — the caller decides what to do with it.
        """
        seed_record = SeedRecord(
            run_number=run_number,
            persona_seed="",  # generated during population build
            numpy_seed=numpy_seed,
        )
        return self._execute_run(
            calibration=calibration,
            profiles=profiles,
            seed_record=seed_record,
            run_number=run_number,
            tick_callback=tick_callback,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _execute_run(
        self,
        calibration: CalibrationInput,
        profiles: ProfileSet,
        seed_record: SeedRecord,
        run_number: int,
        tick_callback: Callable[[dict], None] | None,
        run_start_callback: Callable | None = None,
        pause_event: threading.Event | None = None,
    ) -> RunResult:
        """Build fresh agents and execute one simulation run."""
        agents, persona_seed = build_population(calibration, profiles, seed_record.numpy_seed)
        actual_seed = SeedRecord(
            run_number=seed_record.run_number,
            persona_seed=persona_seed,
            numpy_seed=seed_record.numpy_seed,
        )
        if run_start_callback is not None:
            run_start_callback(run_number, actual_seed.persona_seed)
        sim = SimulationRun(calibration=calibration, agents=agents)
        return sim.run(
            numpy_seed=actual_seed.numpy_seed,
            persona_seed=actual_seed.persona_seed,
            run_number=run_number,
            tick_callback=tick_callback,
            pause_event=pause_event,
        )

    def _generate_seeds(
        self,
        calibration: CalibrationInput,
        num_runs: int,
    ) -> list[SeedRecord]:
        """
        Generate one SeedRecord per run.

        Persona seed: 8 random digits (1–3), one per agent type in order
        R1, R2, R3, R4, I1, I2, I3, I4.  Drawn from a master RNG so the
        full set of seeds is itself reproducible from the batch setup.

        Numpy seed:
          - seed_policy="random" → independent random int per run.
          - seed_policy="fixed"  → fixed_seed for all runs (identical outputs).
        """
        # Use a deterministic master RNG for seed generation, seeded from
        # the hash of the calibration's scenario name to give stable seeds
        # for the same scenario across re-runs of the API.
        master_seed = abs(hash(calibration.scenario_name)) % (2**32)
        master_rng = np.random.default_rng(master_seed)

        records = []
        for i in range(num_runs):
            # Numpy seed — controls both population archetype assignment and
            # all stochastic simulation elements.
            if calibration.run.seed_policy == "fixed" and calibration.run.fixed_seed is not None:
                numpy_seed = calibration.run.fixed_seed
            else:
                numpy_seed = int(master_rng.integers(0, 2**31))

            # persona_seed is generated during population build (_execute_run).
            records.append(SeedRecord(
                run_number=i + 1,
                persona_seed="",
                numpy_seed=numpy_seed,
            ))

        return records

    def _generate_narrative(
        self,
        calibration: CalibrationInput,
        profiles: ProfileSet,
        agg_stats,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Generate the post-batch investment briefing narrative via LLM."""
        from backend.llm.narrative_generator import generate_narrative
        return generate_narrative(calibration, profiles, agg_stats, status_callback)
