"""
LLM profile generator — 8 sequential calls to create investor personas.

Entry point: `generate_profiles(calibration) -> ProfileSet`

One call per agent type (R1, R2, R3, R4, I1, I2, I3, I4). Each call
generates 3 personas for that type using Anthropic structured output
(tool use). Prior personas are visible in context to enforce diversity.

Error handling:
  - Any characteristic value outside its valid range → substitute range
    midpoint and emit a warning (does not abort the batch).
  - API errors propagate as RuntimeError with context.

Spec references: §9.2 (profile generation call structure), §12.1 (BARS).
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable

import anthropic

# ── Retry configuration ────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4, 8)   # seconds between attempts
_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)

from backend.config.sector_config import AGENT_TYPES, SECTORS
from backend.models.calibration import CalibrationInput
from backend.models.profile import AgentCharacteristics, PersonaProfile, ProfileSet
from backend.llm.schemas import (
    build_type_schema,
    get_na_fields_for_type,
    get_range_midpoint,
    is_in_range,
)

# Model used for profile generation.
_MODEL = "claude-sonnet-4-6"

# Human-readable labels for agent types — shown in system prompt.
_TYPE_LABELS: dict[str, str] = {
    "R1": "Passive / long-term retail (pension savers, index fund holders)",
    "R2": "Active self-directed retail (platform traders, DIY investors)",
    "R3": "High net worth / sophisticated retail",
    "R4": "Speculative / momentum-driven retail",
    "I1": "Long-only institutional asset manager",
    "I2": "Hedge fund",
    "I3": "Pension fund or insurance company (liability-driven)",
    "I4": "Sovereign wealth fund or central bank",
}


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_profiles(
    calibration: CalibrationInput,
    progress_callback: Callable[[str, int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> ProfileSet:
    """
    Generate all 24 agent personas (3 per type, 8 types) via sequential
    LLM calls.  Returns a validated ProfileSet.

    progress_callback(agent_type, step, total) is called after each type
    completes, where step is 1-indexed and total is 8.

    status_callback(message) is called on retry attempts so the caller
    can surface retry status to the user.

    Requires ANTHROPIC_API_KEY to be set in the environment (via .env).
    """
    client = _make_client()
    calibration_summary = _summarise_calibration(calibration)

    all_personas: dict[str, list[PersonaProfile]] = {}

    for step, agent_type in enumerate(AGENT_TYPES, 1):
        personas = _generate_type_personas(
            client=client,
            agent_type=agent_type,
            calibration_summary=calibration_summary,
            status_callback=status_callback,
        )
        all_personas[agent_type] = personas
        if progress_callback is not None:
            progress_callback(agent_type, step, len(AGENT_TYPES))

    return ProfileSet(personas=all_personas)


# ── Per-type generation ────────────────────────────────────────────────────────

def _generate_type_personas(
    client: anthropic.Anthropic,
    agent_type: str,
    calibration_summary: str,
    status_callback: Callable[[str], None] | None = None,
) -> list[PersonaProfile]:
    """
    Make one API call for the given agent type, generating all 3 personas.
    Validates and clamps out-of-range values before returning.
    Retries up to _MAX_RETRIES times on transient API errors.
    """
    tool_schema = build_type_schema(agent_type)
    na_fields = get_na_fields_for_type(agent_type)
    type_label = _TYPE_LABELS[agent_type]

    system_prompt = _build_system_prompt(agent_type, type_label, na_fields)
    user_message = _build_user_message(agent_type, type_label, calibration_summary)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_schema],
                tool_choice={"type": "auto"},
            )
            break  # success — exit retry loop
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                msg = (
                    f"API error generating {agent_type} profiles "
                    f"(attempt {attempt}/{_MAX_RETRIES}) — retrying in {_RETRY_DELAYS[attempt - 1]}s…"
                )
                if status_callback:
                    status_callback(msg)
                time.sleep(_RETRY_DELAYS[attempt - 1])
            else:
                raise RuntimeError(
                    f"Anthropic API error generating {agent_type} personas after "
                    f"{_MAX_RETRIES} attempts: {exc}"
                ) from exc
        except anthropic.APIError as exc:
            raise RuntimeError(
                f"Anthropic API error generating {agent_type} personas: {exc}"
            ) from exc

    raw_personas = _extract_tool_result(response, agent_type)
    personas = _parse_and_validate_personas(raw_personas, agent_type, na_fields)
    return personas


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_system_prompt(
    agent_type: str,
    type_label: str,
    na_fields: list[str],
) -> str:
    na_note = (
        f" Note: the following characteristics are not applicable for {agent_type} "
        f"and are excluded from the schema: {', '.join(na_fields)}."
        if na_fields
        else ""
    )
    return (
        f"You are generating behavioural calibration types for a financial market "
        f"agent-based simulation. You will create 3 distinct behavioural variants of "
        f"the {agent_type} agent type ({type_label}).\n\n"
        "These are NOT fictional individual investors. They are reusable behavioural "
        "archetypes — specific calibrations of this agent type that represent "
        "meaningfully different ways this type of investor processes information and "
        "makes decisions. Each variant will be used as a calibration template across "
        "thousands of simulation runs.\n\n"
        "The narrative field must describe the BEHAVIOURAL ARCHETYPE:\n"
        "- What signals does this variant attend to, and how strongly?\n"
        "- What cognitive and emotional patterns dominate its decision-making?\n"
        "- How does it respond to market stress, momentum, or exogenous shocks?\n"
        "- What is its dominant behavioural failure mode "
        "  (e.g. paralysed by loss aversion, trend-chases, ignores signals entirely)?\n"
        "Do NOT describe a fictional person, their biography, occupation, or age.\n\n"
        "IMPORTANT — reasoning requirement:\n"
        "Before calling the tool, think through each variant carefully. Consider:\n"
        "1. What is the core behavioural distinction of this variant within the type?\n"
        "2. How do its characteristics interact coherently "
        "   (e.g. high herding + low information quality → susceptible to market panics)?\n"
        "3. Are all three variants meaningfully different in their dominant behavioural driver?\n\n"
        "Scoring guidance:\n"
        "- Use the full range allowed for each characteristic. Avoid clustering at round numbers.\n"
        "- Scores must reflect realistic, coherent investor psychology, not random assignments.\n"
        "- Each characteristic's description contains BARS anchor text — use it to reason "
        "  about the appropriate score before assigning it.\n"
        "- The 3 variants must differ in at least their primary driver "
        "  (e.g. one more fundamentalist, one more momentum-driven, one more reactive).\n"
        + na_note
    )


def _build_user_message(
    agent_type: str,
    type_label: str,
    calibration_summary: str,
) -> str:
    return (
        f"Generate 3 distinct behavioural variants of the {agent_type} ({type_label}) "
        "agent type for this simulation scenario.\n\n"
        f"Scenario context:\n{calibration_summary}\n\n"
        "Each variant must represent a meaningfully different behavioural archetype within "
        "this investor type — a distinct pattern of how this type processes signals and acts, "
        "not a fictional individual. Call the tool with all 3 variants."
    )


# ── Parsing and validation ─────────────────────────────────────────────────────

def _extract_tool_result(
    response: anthropic.types.Message,
    agent_type: str,
) -> dict:
    """
    Extract the tool use result dict from the Anthropic response.
    Raises RuntimeError if the LLM did not call the tool.
    """
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(
        f"LLM did not call the tool for {agent_type} persona generation. "
        f"Stop reason: {response.stop_reason}. "
        f"Content types: {[b.type for b in response.content]}."
    )


def _parse_and_validate_personas(
    raw: dict,
    agent_type: str,
    na_fields: list[str],
) -> list[PersonaProfile]:
    """
    Parse the tool call result into PersonaProfile objects. Validates each
    characteristic value; substitutes midpoint for out-of-range values.
    """
    personas = []
    for idx in range(1, 4):
        key = f"persona_{idx}"
        persona_data = raw.get(key)
        if persona_data is None:
            raise RuntimeError(
                f"Tool result for {agent_type} missing '{key}'."
            )

        narrative = persona_data.get("narrative", "")
        raw_chars = persona_data.get("characteristics", {})

        validated_chars = _validate_characteristics(raw_chars, agent_type, na_fields, idx)

        # Set N/A fields to None for the model.
        for field in na_fields:
            validated_chars[field] = None

        personas.append(PersonaProfile(
            agent_type=agent_type,
            persona_number=idx,
            narrative=narrative,
            characteristics=AgentCharacteristics(**validated_chars),
        ))

    return personas


def _validate_characteristics(
    raw_chars: dict,
    agent_type: str,
    na_fields: list[str],
    persona_number: int,
) -> dict:
    """
    Validate and clamp characteristic values. Returns a clean dict with all
    21 fields (N/A fields set to None by the caller).

    - Missing fields that are N/A: set to None (handled by caller).
    - Missing fields that are required: warn and substitute midpoint.
    - Out-of-range values: warn and substitute midpoint.
    """
    from backend.config.agent_config import CHARACTERISTIC_NAMES

    result = {}

    for name in CHARACTERISTIC_NAMES:
        if name in na_fields:
            # Caller handles N/A; skip range validation.
            continue

        raw_value = raw_chars.get(name)

        if raw_value is None:
            midpoint = get_range_midpoint(agent_type, name)
            warnings.warn(
                f"{agent_type} persona {persona_number}: '{name}' missing from LLM output; "
                f"substituting midpoint {midpoint:.3f}.",
                stacklevel=4,
            )
            result[name] = midpoint
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            midpoint = get_range_midpoint(agent_type, name)
            warnings.warn(
                f"{agent_type} persona {persona_number}: '{name}' is not a number "
                f"(got {raw_value!r}); substituting midpoint {midpoint:.3f}.",
                stacklevel=4,
            )
            result[name] = midpoint
            continue

        if not is_in_range(agent_type, name, value):
            midpoint = get_range_midpoint(agent_type, name)
            warnings.warn(
                f"{agent_type} persona {persona_number}: '{name}={value:.3f}' is outside "
                f"valid range; substituting midpoint {midpoint:.3f}.",
                stacklevel=4,
            )
            result[name] = midpoint
        else:
            result[name] = value

    return result


# ── Calibration summariser ─────────────────────────────────────────────────────

def _summarise_calibration(calibration: CalibrationInput) -> str:
    """
    Produce a compact human-readable summary of the calibration for the LLM.
    Keeps token usage low — the LLM needs context, not full JSON.
    """
    lines = [f"Scenario: {calibration.scenario_name}"]
    if calibration.market_context:
        lines.append(
            f"Market environment: {calibration.market_context}\n"
            "(This describes the prevailing macro regime agents are operating in — "
            "use it to calibrate their baseline behavioral dispositions.)"
        )
    lines.append(f"Ticks: {calibration.run.total_ticks} (≈1 tick per trading day)")

    if calibration.shocks:
        shock_strs = []
        for shock in calibration.shocks:
            sector_names = [SECTORS[i] for i in shock.affected_sectors if i < len(SECTORS)]
            sectors = ", ".join(sector_names) if sector_names else "all sectors"
            shock_strs.append(
                f"{shock.shock_type} {shock.channel}-channel shock on {sectors}: "
                f"magnitude {shock.magnitude:+.0%}, onset tick {shock.onset_tick}"
            )
        lines.append("Shocks: " + "; ".join(shock_strs))
    else:
        lines.append("Shocks: none")

    return "\n".join(lines)


# ── Client factory ─────────────────────────────────────────────────────────────

def _make_client() -> anthropic.Anthropic:
    """Create an Anthropic client. Reads ANTHROPIC_API_KEY from the environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file and ensure python-dotenv loads it."
        )
    return anthropic.Anthropic(api_key=api_key)
