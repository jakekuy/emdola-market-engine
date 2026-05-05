"""
Pydantic v2 models for simulation run and batch results.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from backend.config.sector_config import NUM_SECTORS, SECTORS


class SeedRecord(BaseModel):
    """Stores both seeds required to exactly reproduce a run (spec §9.3)."""
    model_config = ConfigDict(frozen=True)

    run_number: int
    persona_seed: str = Field(
        ...,
        description=(
            "Archetype distribution string recording how many agents of each type "
            "were assigned to each archetype (a/b/c) in this run. "
            "Format: 'R1a5b15c10R2a2b19c9...'. Generated from numpy_seed at "
            "population build time — numpy_seed is sufficient to reproduce the run."
        )
    )
    numpy_seed: int = Field(
        ...,
        description="NumPy random seed that fixes all stochastic simulation elements."
    )


class TickSnapshot(BaseModel):
    """Environment state at a single logged tick."""
    model_config = ConfigDict(frozen=True)

    tick: int
    prices: list[float] = Field(..., description=f"{NUM_SECTORS} sector prices.")
    volume: list[float] = Field(..., description=f"{NUM_SECTORS} sector volumes.")
    volatility: list[float] = Field(..., description=f"{NUM_SECTORS} rolling σ.")
    excess_demand: list[float] = Field(..., description=f"{NUM_SECTORS} net excess demand.")
    shock_active: bool
    influence_active: bool = False


class TypeActionSummary(BaseModel):
    """Aggregate action summary for one agent type at one logged tick."""
    model_config = ConfigDict(frozen=True)

    agent_type: str
    tick: int
    activity_ratio: float = Field(
        ..., description="Fraction of agents of this type that acted this tick."
    )
    net_direction: list[float] = Field(
        ...,
        description=(
            f"{NUM_SECTORS} net signed magnitude per sector (positive=buy, negative=sell)."
        )
    )
    buy_count: int
    sell_count: int
    hold_count: int


class RunResult(BaseModel):
    """Complete result for one simulation run."""
    model_config = ConfigDict(frozen=True)

    run_number: int
    seed_record: SeedRecord
    tick_snapshots: list[TickSnapshot] = Field(
        default_factory=list,
        description="Environment state at each logged tick."
    )
    type_summaries: list[TypeActionSummary] = Field(
        default_factory=list,
        description="Per-type action summaries at each logged tick."
    )
    final_prices: list[float] = Field(
        ..., description="Sector prices at the final tick."
    )
    final_mispricing: list[float] = Field(
        ...,
        description=(
            f"{NUM_SECTORS} mispricing values at final tick: "
            "(price(T) - 100) / 100."
        )
    )
    agent_traces: list[AgentTraceRecord] = Field(
        default_factory=list,
        description=(
            "Interesting trade events for the 2 traced agents per type (16 total). "
            "Filtered to shock-active ticks, large trades, and direction reversals. "
            "Empty for old saved results."
        )
    )


class AggregatedStats(BaseModel):
    """
    Cross-run aggregate statistics computed by the aggregator (spec §11).
    This is the primary analytical output and the input to the narrative LLM call.
    """
    model_config = ConfigDict(frozen=True)

    num_runs: int
    sectors: list[str] = Field(default_factory=lambda: list(SECTORS))

    # Price trajectory statistics — shape: [num_ticks, num_sectors] collapsed to statistics
    mean_prices: dict[str, list[float]] = Field(
        ..., description="Mean price per sector per logged tick, keyed by sector name."
    )
    median_prices: dict[str, list[float]] = Field(...)
    p10_prices: dict[str, list[float]] = Field(...)
    p90_prices: dict[str, list[float]] = Field(...)

    # Mispricing: (price - 100) / 100
    mean_mispricing: dict[str, list[float]] = Field(...)
    p10_mispricing: dict[str, list[float]] = Field(...)
    p90_mispricing: dict[str, list[float]] = Field(...)

    # Per-type activity across runs
    type_activity_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent type → summary statistics of activity ratio and direction."
    )

    # Compact JSON summary for narrative call input
    narrative_input_json: str = Field(
        "",
        description="Serialized compact summary for the LLM narrative call."
    )


class AgentTraceRecord(BaseModel):
    """One interesting trade event captured for a traced agent."""
    model_config = ConfigDict(frozen=True)

    agent_id: int
    agent_type: str
    archetype_num: int = Field(..., description="Archetype variant: 1, 2, or 3 (A, B, C).")
    tick: int
    sector_index: int
    direction: int = Field(..., description="+1 = buy, -1 = sell.")
    magnitude: float = Field(..., description="Absolute trade magnitude in billions USD.")
    shock_active: bool


class SanityCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_name: str
    passed: bool
    score: float = Field(
        0.0,
        description=(
            "Continuous health signal strength: 0.0–1.0. "
            "0.0–0.39 = Weak, 0.40–0.69 = Moderate, 0.70–1.0 = Strong."
        )
    )
    strength_label: str = Field(
        "",
        description="Human-readable label derived from score: 'Weak', 'Moderate', or 'Strong'."
    )
    message: str
    detail: str = ""


class BatchResult(BaseModel):
    """
    Complete result for a full batch run.
    Saved to data/runs/{batch_id}/results.json.
    """
    model_config = ConfigDict(frozen=True)

    batch_id: str
    scenario_name: str
    num_runs: int
    total_ticks: int
    shocked_sectors: list[str] = Field(
        default_factory=list,
        description="Sector names that received at least one direct shock. Used by the UI to distinguish directly shocked signals from coupling/emergent effects."
    )
    run_results: list[RunResult] = Field(
        default_factory=list,
        description="Individual run results (only present when storage_mode='full')."
    )
    aggregated_stats: AggregatedStats | None = None
    sanity_checks: list[SanityCheckResult] = Field(default_factory=list)
    narrative: str = Field("", description="LLM-generated investment briefing note.")
    persona_narrative_md: str = Field(
        "", description="LLM-generated persona descriptions in markdown."
    )


class BatchStatus(BaseModel):
    """Live status of a running batch — returned by GET /api/batch/{id}/status."""
    model_config = ConfigDict(frozen=True)

    batch_id: str
    status: str  # "running" | "complete" | "error"
    runs_complete: int
    total_runs: int
    current_tick: int = 0
    error_message: str = ""
