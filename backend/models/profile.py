"""
Pydantic v2 models for LLM-generated agent persona profiles.
These represent the Layer 2 (LLM-defined characterisation) output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator, ConfigDict


class AgentCharacteristics(BaseModel):
    """
    The 21 behavioural characteristics that define an agent's profile.
    All scores are 0.0–1.0.  N/A fields (institutional_inertia for retail;
    mental_accounting and ownership_bias for institutional) are set to None
    and excluded from LLM prompting for the relevant types.
    """
    model_config = ConfigDict(frozen=True)

    # ── Stable traits ──────────────────────────────────────────────────────────
    risk_aversion: float = Field(..., ge=0.0, le=1.0)
    time_horizon: float = Field(..., ge=0.0, le=1.0)
    belief_formation: float = Field(..., ge=0.0, le=1.0)
    information_quality: float = Field(..., ge=0.0, le=1.0)
    herding_sensitivity: float = Field(..., ge=0.0, le=1.0)
    decision_style: float = Field(..., ge=0.0, le=1.0)
    strategy_adaptability: float = Field(..., ge=0.0, le=1.0)
    conviction_threshold: float = Field(..., ge=0.0, le=1.0)
    institutional_inertia: float | None = Field(None, ge=0.0, le=1.0)
    memory_window: float = Field(..., ge=0.0, le=1.0)

    # ── Cognitive biases ───────────────────────────────────────────────────────
    overconfidence: float = Field(..., ge=0.0, le=1.0)
    anchoring: float = Field(..., ge=0.0, le=1.0)
    confirmation_bias: float = Field(..., ge=0.0, le=1.0)
    recency_bias: float = Field(..., ge=0.0, le=1.0)
    salience_bias: float = Field(..., ge=0.0, le=1.0)
    pattern_matching_bias: float = Field(..., ge=0.0, le=1.0)

    # ── Emotional biases ───────────────────────────────────────────────────────
    loss_aversion: float = Field(..., ge=0.0, le=1.0)
    winner_selling_loser_holding: float = Field(..., ge=0.0, le=1.0)
    regret_aversion: float = Field(..., ge=0.0, le=1.0)
    fomo: float = Field(..., ge=0.0, le=1.0)
    mental_accounting: float | None = Field(None, ge=0.0, le=1.0)
    ownership_bias: float | None = Field(None, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, float]:
        """Return characteristics as a plain dict, omitting None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class PersonaProfile(BaseModel):
    """
    One LLM-generated persona for a specific agent type.
    3 personas per type are generated per model calibration.
    """
    model_config = ConfigDict(frozen=True)

    agent_type: str = Field(..., description="Agent type code: R1–R4 or I1–I4.")
    persona_number: int = Field(..., ge=1, le=3)
    narrative: str = Field(
        ...,
        description="One-paragraph description of this investor persona."
    )
    characteristics: AgentCharacteristics


class ProfileSet(BaseModel):
    """
    The full set of LLM-generated personas for one model calibration —
    3 personas × 8 agent types = 24 personas total.
    """
    model_config = ConfigDict(frozen=True)

    personas: dict[str, list[PersonaProfile]] = Field(
        ...,
        description="Keyed by agent type (R1–R4, I1–I4); each has 3 PersonaProfiles."
    )

    @model_validator(mode="after")
    def validate_completeness(self) -> "ProfileSet":
        from backend.config.sector_config import AGENT_TYPES
        missing = [t for t in AGENT_TYPES if t not in self.personas]
        if missing:
            raise ValueError(f"ProfileSet missing agent types: {missing}")
        for atype, profiles in self.personas.items():
            if len(profiles) != 3:
                raise ValueError(
                    f"Expected 3 personas for {atype}, got {len(profiles)}."
                )
        return self
