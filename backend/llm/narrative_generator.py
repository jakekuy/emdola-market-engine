"""
LLM narrative generator — one post-batch call producing an investment briefing.

Entry point: `generate_narrative(calibration, profiles, agg_stats) -> str`

The output is a plain-text investment briefing note with:
  - One paragraph per agent type summarising their behaviour in the simulation
  - One overall finding (~one page): what happened, why, and what it implies

Input to the LLM:
  - Compact Pandas summary JSON from `agg_stats.narrative_input_json`
  - Persona descriptions (narrative field from each ProfileSet persona)
  - Calibration summary (scenario name, shocks)

The narrative is stored in `BatchResult.narrative` and written to
`data/runs/{batch_id}/personas.md`.

Spec references: §9.5 (narrative output).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import anthropic

# ── Retry configuration ────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4, 8)
_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)

from backend.config.sector_config import AGENT_TYPES, SECTORS
from backend.models.calibration import CalibrationInput
from backend.models.profile import ProfileSet
from backend.output.aggregator import AggregatedStats

_MODEL = "claude-sonnet-4-6"

_TYPE_LABELS: dict[str, str] = {
    "R1": "Passive / long-term retail",
    "R2": "Active self-directed retail",
    "R3": "High net worth retail",
    "R4": "Speculative / momentum retail",
    "I1": "Long-only asset manager",
    "I2": "Hedge fund",
    "I3": "Pension fund / insurance",
    "I4": "Sovereign wealth fund",
}


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_narrative(
    calibration: CalibrationInput,
    profiles: ProfileSet,
    agg_stats: AggregatedStats,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Generate an investment briefing note from batch simulation results.

    Returns the narrative as a plain string (markdown-formatted).
    Returns an empty string (silently) if ANTHROPIC_API_KEY is not set —
    this allows test runs without an API key to complete without error.
    Retries up to _MAX_RETRIES times on transient API errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = _build_system_prompt()
    user_message = _build_user_message(calibration, profiles, agg_stats)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except _RETRYABLE_ERRORS as exc:
            if attempt < _MAX_RETRIES:
                msg = (
                    f"API error generating investment analysis "
                    f"(attempt {attempt}/{_MAX_RETRIES}) — retrying in {_RETRY_DELAYS[attempt - 1]}s…"
                )
                if status_callback:
                    status_callback(msg)
                time.sleep(_RETRY_DELAYS[attempt - 1])
            else:
                raise RuntimeError(
                    f"Anthropic API error generating narrative after "
                    f"{_MAX_RETRIES} attempts: {exc}"
                ) from exc
        except anthropic.APIError as exc:
            raise RuntimeError(
                f"Anthropic API error generating simulation narrative: {exc}"
            ) from exc

    return ""  # unreachable, satisfies type checker


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are a senior investment analyst writing a post-simulation briefing note "
        "for an agent-based market simulation. Write in a direct, analytical style. "
        "Do not make any assumptions about who the audience is.\n\n"
        "IMPORTANT — framing:\n"
        "All outputs — price movements, mispricing signals, cross-run patterns — are modelling "
        "outputs, not observed market facts. The value of the simulation is identifying possible "
        "market mechanisms and price dynamics worth investigating. Write consistently in that register: "
        "this is what the model suggests could happen, and why it might be interesting. "
        "Do not distinguish between 'confirmed' price findings and 'hypothesised' mechanisms — "
        "everything is a model output. Do not overclaim.\n\n"
        "IMPORTANT — interpreting prices and mispricing figures:\n"
        "All prices are indexed to a baseline of 100 that proxies prevailing market consensus for "
        "the scenario — 100 is not a fixed absolute price. A sector ending at 106 does not mean its "
        "absolute price rose 6%; it means agents priced it 6% above the market-implied baseline, "
        "ceteris paribus relative to whatever else the market is pricing in. The ±% figures are "
        "relative (mis)pricing pressure: systematic behavioural deviation above or below the "
        "consensus baseline. Always frame price and mispricing readings this way. Do not describe "
        "movements as absolute gains or losses.\n\n"
        "IMPORTANT — what the analysis must focus on:\n"
        "Your analysis must describe what the model's agents actually did — not restate what the "
        "scenario was designed to represent. The scenario description and shock design are inputs "
        "you have been given for context. The analysis is about the emergent outputs: how agents "
        "responded, which agent types drove the signal, what the price path shape reveals (front-loaded "
        "vs gradual, sustained vs reverting), what the spread across runs tells you about conviction "
        "vs uncertainty, and where sectors diverged in unexpected ways relative to each other or to "
        "their shock design. Do not present the scenario framing as something the model discovered. "
        "The model received a signal; describe how agents processed it.\n\n"
        "IMPORTANT — distinguishing direct shocks from coupling effects:\n"
        "The user message lists the sectors that received direct shocks. Any sector NOT in that list "
        "that shows mispricing is most likely a cross-sector coupling artefact — the model's "
        "macro sentiment spillover mechanism bleeds signal from shocked sectors into adjacent ones. "
        "Do NOT invent a sector-specific mechanism for non-shocked sectors. Instead, flag them as "
        "potential coupling effects. For directly shocked sectors, focus on the agent behaviour and "
        "price dynamics — not on restating the shock rationale from the scenario description.\n\n"
        "CRITICAL FORMATTING RULES:\n"
        "- Do NOT add any distribution classification, confidentiality notices, or disclaimers.\n"
        "- Do NOT write preamble or introductory sentences about what you are about to do.\n"
        "- Start directly with the first section heading.\n"
        "- Total length: under 500 words.\n\n"
        "Structure your output in markdown as follows:\n\n"
        "## Investment Signal\n"
        "2–3 sentences, plain language, no jargon. State what the simulation found and "
        "what it means for an investor. Be concrete and specific — name sectors and "
        "directions. Do not hedge or caveat.\n\n"
        "## Emergent Market Dynamics\n"
        "2–3 paragraphs on what the model produced. Focus on agent behaviour and interaction "
        "dynamics: which agent types drove the signal (based on the persona descriptions and "
        "activity data), what the price path shape reveals (front-loaded vs gradual, sustained "
        "vs reverting), what the spread tells you about cross-run conviction, and where sectors "
        "diverged unexpectedly relative to each other or to their shock design. Do not restate "
        "the scenario framing — describe what agents did with the signal they received.\n\n"
        "## Cross-Run Patterns\n"
        "Bullet points: what was consistent across most runs; what varied; any unexpected sector "
        "behaviour worth flagging. Note spread width where relevant.\n\n"
        "## Mispricing Signal\n"
        "Which sectors consistently ended above or below starting price? Structural (low spread, "
        "consistent direction) or noisy (wide spread)? Give direction and magnitude.\n\n"
        "## Investment Thesis\n"
        "What does the model suggest is worth investigating in real markets? Be concrete about "
        "sectors and direction. This is a starting point for further research, not a prediction. "
        "Let the number of signals emerge from the data. Lead with the strongest finding."
    )


def _build_user_message(
    calibration: CalibrationInput,
    profiles: ProfileSet,
    agg_stats: AggregatedStats,
) -> str:
    sections = []

    # Scenario context
    sections.append(f"**Scenario:** {calibration.scenario_name}")
    if calibration.scenario_description:
        sections.append(f"**Scenario Context:**\n{calibration.scenario_description}")
    sections.append(f"**Ticks:** {calibration.run.total_ticks} | **Runs:** {calibration.run.num_runs}")

    if calibration.shocks:
        shock_lines = []
        for s in calibration.shocks:
            sector_names = [SECTORS[i] for i in s.affected_sectors if i < len(SECTORS)]
            sectors = ", ".join(sector_names) if sector_names else "all sectors"
            shock_lines.append(
                f"- {s.shock_type} {s.channel}-channel shock: {sectors}, "
                f"magnitude {s.magnitude:+.0%}, onset tick {s.onset_tick}"
            )
        sections.append("**Shocks:**\n" + "\n".join(shock_lines))
    else:
        sections.append("**Shocks:** none")

    # Persona narratives
    persona_lines = []
    for agent_type in AGENT_TYPES:
        label = _TYPE_LABELS.get(agent_type, agent_type)
        type_personas = profiles.personas.get(agent_type, [])
        persona_lines.append(f"\n### {agent_type} — {label}")
        for p in type_personas:
            persona_lines.append(f"Persona {p.persona_number}: {p.narrative}")
    sections.append("**Agent Personas:**" + "\n".join(persona_lines))

    # Quantitative results
    sections.append(
        f"**Simulation Results (JSON summary):**\n```json\n{agg_stats.narrative_input_json}\n```"
    )

    sections.append(
        "Write the investment briefing note based on the above. "
        "Focus on what the price and mispricing data shows. "
        "Where you propose mechanisms, frame them as hypotheses."
    )

    return "\n\n".join(sections)
