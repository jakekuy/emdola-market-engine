"""
Shared pytest fixtures for EMDOLA tests.
"""

import pytest
import numpy as np

from backend.config.sector_config import NUM_SECTORS
from backend.models.profile import AgentCharacteristics, PersonaProfile, ProfileSet


def make_characteristics(**overrides) -> AgentCharacteristics:
    """
    Create an AgentCharacteristics with sensible defaults.
    Override any field by passing it as a keyword argument.
    """
    defaults = dict(
        risk_aversion=0.5,
        time_horizon=0.5,
        belief_formation=0.5,
        information_quality=0.5,
        herding_sensitivity=0.3,
        decision_style=0.3,
        strategy_adaptability=0.3,
        conviction_threshold=0.4,
        institutional_inertia=None,
        memory_window=0.5,
        overconfidence=0.3,
        anchoring=0.4,
        confirmation_bias=0.4,
        recency_bias=0.4,
        salience_bias=0.4,
        pattern_matching_bias=0.4,
        loss_aversion=0.5,
        winner_selling_loser_holding=0.4,
        regret_aversion=0.4,
        fomo=0.3,
        mental_accounting=None,
        ownership_bias=None,
    )
    defaults.update(overrides)
    return AgentCharacteristics(**defaults)


def make_fundamentalist_characteristics() -> AgentCharacteristics:
    """Pure fundamentalist profile (belief_formation=0, low biases)."""
    return make_characteristics(
        belief_formation=0.0,
        herding_sensitivity=0.1,
        conviction_threshold=0.3,
        overconfidence=0.1,
        fomo=0.05,
        strategy_adaptability=0.1,
    )


def make_chartist_characteristics() -> AgentCharacteristics:
    """Pure chartist profile (belief_formation=1, trend-following)."""
    return make_characteristics(
        belief_formation=1.0,
        herding_sensitivity=0.7,
        conviction_threshold=0.2,
        overconfidence=0.6,
        fomo=0.6,
    )


def make_persona(agent_type: str, characteristics: AgentCharacteristics, num: int = 1) -> PersonaProfile:
    return PersonaProfile(
        agent_type=agent_type,
        persona_number=num,
        narrative=f"Test persona for {agent_type}",
        characteristics=characteristics,
    )


@pytest.fixture
def default_chars():
    return make_characteristics()


@pytest.fixture
def fundamentalist_chars():
    return make_fundamentalist_characteristics()


@pytest.fixture
def dummy_profile_set():
    """
    A ProfileSet with 3 identical test personas per agent type.
    Suitable for integration tests that don't require LLM calls.
    """
    from backend.config.sector_config import AGENT_TYPES
    chars = make_characteristics()
    personas = {}
    for atype in AGENT_TYPES:
        inertia = 0.5 if atype.startswith("I") else None
        mental = None if atype.startswith("I") else 0.5
        ownership = None if atype.startswith("I") else 0.4
        type_chars = make_characteristics(
            institutional_inertia=inertia,
            mental_accounting=mental,
            ownership_bias=ownership,
        )
        personas[atype] = [
            make_persona(atype, type_chars, num=i + 1)
            for i in range(3)
        ]
    return ProfileSet(personas=personas)
