"""
Pydantic v2 models for the calibration input — everything the user configures
in the Setup view before a batch run begins.  This is the Layer 3 (user-defined
contextual environment) input described in spec §7.1.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

from backend.config.sector_config import NUM_SECTORS, SECTORS, LAMBDA_DEFAULTS
from backend.config.agent_config import (
    STARTING_CASH_WEIGHT,
    STARTING_SECTOR_PCT,
    AUM_PER_AGENT,
)
from backend.config.constants import (
    DEFAULT_TOTAL_TICKS,
    DEFAULT_AGENTS_PER_TYPE,
    DEFAULT_NON_MARKET_SIGNAL_INTENSITY,
)


# ── Shock definition (spec §10) ───────────────────────────────────────────────

class ShockDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    onset_tick: int = Field(
        ..., ge=0,
        description="Tick at which the shock begins. Must be < total_ticks."
    )
    magnitude: float = Field(
        ...,
        description=(
            "% perturbation to starting price (P0=100). "
            "Positive = bullish, negative = bearish. E.g. 0.10 = +10%."
        )
    )
    duration: int = Field(
        ..., ge=1,
        description="Number of ticks over which the shock is applied."
    )
    affected_sectors: list[int] = Field(
        ...,
        description=f"Sector indices (0–{NUM_SECTORS - 1}) receiving this shock."
    )
    shock_type: Literal["acute", "chronic"] = Field(
        "acute",
        description=(
            "'acute' = concentrated at onset (discrete event); "
            "'chronic' = distributed per tick (slow-building trend)."
        )
    )
    channel: Literal["market", "influence", "both"] = Field(
        "both",
        description="Which signal channel(s) the shock targets."
    )
    reversion: bool = Field(
        False,
        description=(
            "Acute shocks only: if True, the perturbation decays back toward "
            "baseline over the duration period after peak."
        )
    )

    @field_validator("affected_sectors")
    @classmethod
    def validate_sectors(cls, v: list[int]) -> list[int]:
        for idx in v:
            if not (0 <= idx < NUM_SECTORS):
                raise ValueError(
                    f"Sector index {idx} out of range (0–{NUM_SECTORS - 1})."
                )
        return sorted(set(v))  # deduplicate and sort


# ── Tab 1: Market environment ──────────────────────────────────────────────────

class MarketEnvironmentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lambdas: list[float] = Field(
        default_factory=lambda: list(LAMBDA_DEFAULTS),
        description=(
            f"Price sensitivity (λ) per sector — {NUM_SECTORS} values, one per GICS sector "
            f"in order: {', '.join(SECTORS)}."
        )
    )
    non_market_signal_intensity: float = Field(
        DEFAULT_NON_MARKET_SIGNAL_INTENSITY,
        ge=0.0, le=1.0,
        description=(
            "Scales the influence channel. 0.0 = no non-market signals; "
            "1.0 = maximum influence channel strength. Default 0.40 reflects "
            "normal US equity market conditions."
        )
    )
    narrative_half_life: int = Field(
        10, ge=1,
        description=(
            "Half-life of the influence channel broadcast in ticks. "
            "Controls how long the shock narrative sustains agent herding behaviour "
            "before consensus dissolves and agents diverge. "
            "Internally converted to decay_rate = ln(2) / narrative_half_life. "
            "Default 10 ticks (~2 weeks at daily resolution). "
            "Shorter for fast-moving events (central bank announcements: 5); "
            "longer for sustained macro narratives (armed conflict: 15; policy shift: 20+)."
        )
    )
    pre_shock_ticks: int = Field(
        50, ge=0,
        description=(
            "Informational: approximate number of ticks before the first shock "
            "onset. Does not change simulation length — use onset_tick in shocks."
        )
    )

    @field_validator("lambdas")
    @classmethod
    def validate_lambdas(cls, v: list[float]) -> list[float]:
        if len(v) != NUM_SECTORS:
            raise ValueError(f"lambdas must have exactly {NUM_SECTORS} values.")
        if any(lam <= 0 for lam in v):
            raise ValueError("All lambda values must be positive.")
        return v


# ── Tab 2: Shock definitions ───────────────────────────────────────────────────

class ShocksConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shocks: list[ShockDefinition] = Field(
        default_factory=list,
        description="List of shocks to apply during the simulation."
    )


# ── Tab 3: Agent configuration ─────────────────────────────────────────────────

class AgentTypeOverride(BaseModel):
    """Optional per-type overrides for agent count, AUM, and sector weights."""
    model_config = ConfigDict(frozen=True)

    agent_count: int | None = Field(
        None, ge=30, le=200,
        description="Number of agents of this type. Minimum 30, maximum 200."
    )
    aum_override: float | None = Field(
        None, gt=0,
        description="AUM per agent in billions USD. Overrides the engine default."
    )
    cash_weight_override: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Starting cash weight fraction (0–1). Overrides engine default."
    )
    sector_weights_override: list[float] | None = Field(
        None,
        description=(
            f"Starting sector weights — {NUM_SECTORS} values summing to 100 "
            "(percentages of equity allocation)."
        )
    )

    @field_validator("sector_weights_override")
    @classmethod
    def validate_sector_weights(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != NUM_SECTORS:
            raise ValueError(f"sector_weights_override must have {NUM_SECTORS} values.")
        if abs(sum(v) - 100.0) > 0.5:
            raise ValueError(f"sector_weights_override must sum to 100 (got {sum(v):.2f}).")
        return v


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    overrides: dict[str, AgentTypeOverride] = Field(
        default_factory=dict,
        description="Per-type overrides keyed by agent type (R1–R4, I1–I4)."
    )


# ── Tab 4: Run configuration ───────────────────────────────────────────────────

class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_ticks: int = Field(
        DEFAULT_TOTAL_TICKS, ge=10,
        description="Total number of simulation ticks per run."
    )
    num_runs: int = Field(
        10, ge=1,
        description="Number of simulation runs in this batch."
    )
    data_granularity: Literal["end_state", "every_5", "full_fidelity"] = Field(
        "full_fidelity",
        description=(
            "Logging granularity: 'end_state' = final tick only; "
            "'every_5' = every 5 ticks; "
            "'full_fidelity' = every tick (default)."
        )
    )
    storage_mode: Literal["aggregate_only", "full"] = Field(
        "full",
        description=(
            "'aggregate_only' = post-batch summaries only; "
            "'full' = retain per-run tick data for drill-down and replay."
        )
    )
    tick_length_label: str = Field(
        "1 trading day",
        description="Human-readable label for what one tick represents."
    )
    seed_policy: Literal["random", "fixed"] = Field(
        "random",
        description=(
            "'random' = new numpy seed each run; "
            "'fixed' = use fixed_seed for reproducible runs."
        )
    )
    fixed_seed: int | None = Field(
        None,
        description="Numpy seed to use when seed_policy='fixed'."
    )


# ── Top-level calibration input ────────────────────────────────────────────────

class CalibrationInput(BaseModel):
    """
    Complete setup parameters for one EMDOLA batch run.
    Passed to the batch engine and to the LLM profile generation calls.
    Corresponds to the Layer 3 (user-defined contextual environment) input.
    """
    model_config = ConfigDict(frozen=True)

    # Human-readable context (used in LLM prompt)
    scenario_name: str = Field(
        "EMDOLA Run",
        description="Short name for this calibration scenario."
    )
    scenario_description: str = Field(
        "",
        description=(
            "Free-text description of the shock event being studied. "
            "Passed to the narrative LLM call. Not used for profile generation "
            "(agents would not know the shock is coming)."
        )
    )
    market_context: str = Field(
        "",
        description=(
            "Free-text description of the prevailing macro/market environment — "
            "the regime agents are operating in before any shock occurs. "
            "Passed to the LLM profile generation call to produce contextually "
            "calibrated behavioral archetypes. Only meaningful when profiles are "
            "generated or respawned for this run."
        )
    )

    market: MarketEnvironmentConfig = Field(default_factory=MarketEnvironmentConfig)
    shocks: list[ShockDefinition] = Field(default_factory=list)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def get_agent_count(self, agent_type: str) -> int:
        """Returns agent count for a type, applying overrides if set."""
        override = self.agents.overrides.get(agent_type)
        if override and override.agent_count is not None:
            return override.agent_count
        return DEFAULT_AGENTS_PER_TYPE

    def get_aum(self, agent_type: str) -> float:
        """Returns AUM per agent in billions USD, applying overrides if set."""
        override = self.agents.overrides.get(agent_type)
        if override and override.aum_override is not None:
            return override.aum_override
        from backend.config.agent_config import AUM_PER_AGENT
        return AUM_PER_AGENT[agent_type]

    def get_cash_weight(self, agent_type: str) -> float:
        override = self.agents.overrides.get(agent_type)
        if override and override.cash_weight_override is not None:
            return override.cash_weight_override
        return STARTING_CASH_WEIGHT[agent_type]

    def get_sector_pcts(self, agent_type: str) -> list[float]:
        override = self.agents.overrides.get(agent_type)
        if override and override.sector_weights_override is not None:
            return override.sector_weights_override
        return STARTING_SECTOR_PCT[agent_type]
