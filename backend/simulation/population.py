"""
Population builder — constructs the agent population for one simulation run
from a ProfileSet and a numpy seed.

Each agent is independently assigned one of the 3 archetypes for its type,
drawn uniformly at random using the provided numpy_seed. This produces
within-type heterogeneity: agents of the same type may hold different
behavioural calibrations within a single run, and the archetype mix varies
across runs, producing genuine distributional coverage in the Monte Carlo.

The returned persona_seed string records the archetype counts per type:
  "R1a5b15c10R2a2b19c9R3a24b3c3..."
where a/b/c are the counts of agents assigned to archetypes 1/2/3 for each
type. This string is stored in SeedRecord for logging and display; population
is reproducible from the numpy_seed alone.

Spec references: §7.9, §9.3.
"""

from __future__ import annotations

import numpy as np

from backend.config.constants import ARCHETYPE_DIRICHLET_ALPHA, TYPE_DIRICHLET_ALPHA, MIN_AGENTS_PER_TYPE
from backend.config.sector_config import AGENT_TYPES
from backend.models.calibration import CalibrationInput
from backend.models.profile import ProfileSet
from backend.simulation.agent import Agent


def build_population(
    calibration: CalibrationInput,
    profile_set: ProfileSet,
    numpy_seed: int,
) -> tuple[list[Agent], str]:
    """
    Build and return the complete agent population for one simulation run,
    along with the persona_seed string recording the archetype distribution.

    Each agent is independently assigned one of the 3 archetypes for its
    type, drawn uniformly using numpy_seed.  Runs with the same numpy_seed
    produce the same population; different seeds produce different archetype
    mixes, including highly unbalanced distributions.

    Parameters
    ----------
    calibration :
        Contains agent counts, AUM overrides, and starting allocation overrides.
    profile_set :
        The LLM-generated archetypes for all 8 agent types (3 per type).
    numpy_seed :
        Seed for the population RNG — controls archetype assignment.

    Returns
    -------
    agents : list[Agent]
        All agents in a flat list.  Order is R1×N, R2×N … I4×N.
        Agents are numbered sequentially from 0.
    persona_seed : str
        Archetype count string, e.g. "R1a5b15c10R2a2b19c9...".
    """
    rng = np.random.default_rng(numpy_seed)
    agents: list[Agent] = []
    agent_id = 0
    seed_parts: list[str] = []

    # ── Type-level Dirichlet: vary agent count per type across MC runs ─────────
    # Sample proportions across the 8 agent types from a fat-tailed Dirichlet.
    # α=0.5 (same as archetype Dirichlet) makes extreme splits common — some runs
    # will have many R4/I2 agents (chartist-dominated) and few I1/I3 agents
    # (fundamentalist-dominated), and vice versa.  This is what makes MC paths
    # diverge: different type compositions respond differently to the same shock.
    # Only applied when the user has not overridden any type counts explicitly.
    has_count_overrides = any(
        calibration.agents.overrides.get(t) is not None
        and calibration.agents.overrides[t].agent_count is not None
        for t in AGENT_TYPES
    )
    if has_count_overrides:
        # Respect explicit user configuration for all types.
        type_counts = {t: calibration.get_agent_count(t) for t in AGENT_TYPES}
    else:
        n_types = len(AGENT_TYPES)
        total_agents = sum(calibration.get_agent_count(t) for t in AGENT_TYPES)
        # Each type gets at least MIN_AGENTS_PER_TYPE; the remainder is Dirichlet-allocated.
        min_total = MIN_AGENTS_PER_TYPE * n_types
        extra_pool = max(0, total_agents - min_total)
        type_props = rng.dirichlet([TYPE_DIRICHLET_ALPHA] * n_types)
        extra_alloc = np.round(type_props * extra_pool).astype(int)
        # Adjust rounding so totals match exactly.
        diff = extra_pool - int(extra_alloc.sum())
        if diff != 0:
            extra_alloc[int(np.argmax(extra_alloc))] += diff
        type_counts = {
            t: MIN_AGENTS_PER_TYPE + int(extra_alloc[i])
            for i, t in enumerate(AGENT_TYPES)
        }

    for agent_type in AGENT_TYPES:
        count = type_counts[agent_type]
        aum = calibration.get_aum(agent_type)
        cash_weight = calibration.get_cash_weight(agent_type)
        sector_pcts = calibration.get_sector_pcts(agent_type)

        # Sample archetype proportions from Dirichlet(α, α, α), then assign
        # each agent by drawing from those proportions.  α < 1 gives fat-tailed
        # distributions — extreme splits are common, balanced splits are possible.
        props = rng.dirichlet([ARCHETYPE_DIRICHLET_ALPHA] * 3)
        archetype_assignments = rng.choice(3, size=count, p=props)
        counts = [0, 0, 0]

        for archetype_idx in archetype_assignments:
            counts[int(archetype_idx)] += 1
            persona = profile_set.personas[agent_type][int(archetype_idx)]
            agent = Agent(
                agent_type=agent_type,
                agent_id=agent_id,
                profile=persona.characteristics,
                calibration_aum=aum,
                calibration_cash_weight=cash_weight,
                calibration_sector_pcts=sector_pcts,
            )
            agent.archetype_num = int(archetype_idx) + 1  # 1, 2, or 3
            agents.append(agent)
            agent_id += 1

        seed_parts.append(f"{agent_type}n{count}a{counts[0]}b{counts[1]}c{counts[2]}")

    persona_seed = "".join(seed_parts)
    return agents, persona_seed


def get_total_agent_count(calibration: CalibrationInput) -> int:
    """Return the total number of agents across all types."""
    return sum(calibration.get_agent_count(t) for t in AGENT_TYPES)


def agents_by_type(agents: list[Agent]) -> dict[str, list[Agent]]:
    """Group agents by type — used for type-level logging."""
    result: dict[str, list[Agent]] = {t: [] for t in AGENT_TYPES}
    for agent in agents:
        result[agent.agent_type].append(agent)
    return result
