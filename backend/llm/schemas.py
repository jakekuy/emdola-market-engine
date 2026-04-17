"""
JSON schemas for LLM-structured output in profile generation.

Each agent type gets its own tool schema: a JSON object with 21 characteristics
(minus the N/A ones for that type), each field carrying:
  - type: "number"
  - minimum / maximum: from CHARACTERISTIC_RANGES in agent_config.py
  - description: BARS rubric anchor text (used by the LLM to reason within range)

`build_type_schema(agent_type)` returns an Anthropic tool definition dict,
ready to pass to `client.messages.create(tools=[...])`.

`build_all_schemas()` returns all eight schemas keyed by agent type.
"""

from __future__ import annotations

from backend.config.agent_config import CHARACTERISTIC_RANGES

# ── BARS rubric text per characteristic ───────────────────────────────────────
# Each entry is the anchor text shown to the LLM. Inline for clarity.

_BARS: dict[str, str] = {
    "risk_aversion": (
        "Score 0.0–1.0. "
        "0.0=takes maximum positions in highest-volatility sectors; no downside limits. "
        "0.25=large positions; tolerates significant drawdowns. "
        "0.5=balances return and volatility; trims on material volatility spikes. "
        "0.75=limits sizes; avoids high-vol sectors; reduces on adverse moves. "
        "1.0=minimises all positions; moves to cash on any uncertainty."
    ),
    "time_horizon": (
        "Score 0.0–1.0. "
        "0.0=exits on any adverse move; daily evaluation. "
        "0.25=reviews weekly–monthly; responsive to short-term news. "
        "0.5=holds through normal volatility; reassesses on material change over months. "
        "0.75=holds through market cycles; ignores short-term noise. "
        "1.0=holds indefinitely absent structural change; insensitive to short-term moves."
    ),
    "belief_formation": (
        "Score 0.0–1.0 from fundamentalist to chartist. "
        "0.0=buys when price below intrinsic value; trends irrelevant. "
        "0.25=primarily value-driven; occasionally factors strong momentum. "
        "0.5=weighs both fundamental value and price trends equally. "
        "0.75=follows price trends as primary signal; fundamentals as loose constraint. "
        "1.0=buys into rising prices; sells into falling; fundamentals have no bearing."
    ),
    "information_quality": (
        "Score 0.0–1.0. "
        "0.0=public news and social signals only; no structured research. "
        "0.25=general financial media and basic market data; limited depth. "
        "0.5=structured research, analyst reports, standard data; moderate capability. "
        "0.75=proprietary research, specialist data, or direct sector expertise. "
        "1.0=deep proprietary research; primary sources, expert networks, specialist data."
    ),
    "herding_sensitivity": (
        "Score 0.0–1.0. "
        "0.0=increases position when crowd moves against it; views consensus as contrary indicator. "
        "0.25=maintains independent view; crowd has negligible influence. "
        "0.5=sometimes aligns with observed market direction; sometimes holds independent view. "
        "0.75=regularly adjusts toward observed crowd direction; consensus carries significant weight. "
        "1.0=systematically follows crowd; takes positions primarily because others do."
    ),
    "decision_style": (
        "Score 0.0–1.0 from systematic to intuitive. "
        "0.0=applies predefined quantitative rules mechanically; no deviation. "
        "0.25=structured analytical process; discretion only in exceptional circumstances. "
        "0.5=combines formal analysis with discretionary overlay; roughly equal weight. "
        "0.75=relies primarily on experience and intuition; process is a loose reference. "
        "1.0=decides entirely on gut feel and emotion; no formal process applied."
    ),
    "strategy_adaptability": (
        "Score 0.0–1.0. "
        "0.0=applies identical strategy regardless of conditions or track record. "
        "0.25=reviews only under extreme sustained underperformance; very rare adjustments. "
        "0.5=reviews and adjusts periodically in response to sustained signals. "
        "0.75=regularly updates based on recent performance feedback and market signals. "
        "1.0=revises strategy after each new signal or outcome; approach changes continuously."
    ),
    "conviction_threshold": (
        "Score 0.0–1.0. "
        "0.0=acts on any perceived signal regardless of strength; minimal evidence required. "
        "0.25=acts on moderate signals; requires only weak confirmation. "
        "0.5=requires a consistent and reasonably clear signal from at least one reliable source. "
        "0.75=acts only on strong signals confirmed by multiple sources or sustained over time. "
        "1.0=acts only on the most unambiguous multi-source confirmed signals; long inactivity normal."
    ),
    "institutional_inertia": (
        "Score 0.0–1.0. Institutional types only. "
        "0.0=changes positions immediately on new signals; no committee or mandate friction. "
        "0.25=some internal review required; responds to signals within a short window. "
        "0.5=meaningful committee or mandate review; position changes take weeks. "
        "0.75=major changes require extended review and broad consensus; resistant to deviation. "
        "1.0=position changes almost never occur; mandate or career risk prevents deviation."
    ),
    "memory_window": (
        "Score 0.0–1.0. "
        "0.0=retains only the most recent tick; no historical context. "
        "0.25=short lookback ~5–15 ticks; recent signals dominate almost entirely. "
        "0.5=medium lookback ~20–60 ticks; reasonable historical context. "
        "0.75=long lookback ~100–200 ticks; historical patterns carry significant weight. "
        "1.0=very long lookback 250+ ticks; resistant to short-term signal dominance."
    ),
    "overconfidence": (
        "Score 0.0–1.0. "
        "0.0=sizes positions to reflect genuine uncertainty; regularly acknowledges being wrong. "
        "0.25=slight tendency toward larger positions than evidence warrants. "
        "0.5=regularly takes positions larger than evidence warrants; moderate overtrading. "
        "0.75=highly concentrated positions; dismisses contradictory evidence. "
        "1.0=maximum concentration regardless of evidence quality; uncertainty never acknowledged."
    ),
    "anchoring": (
        "Score 0.0–1.0. "
        "0.0=assesses value purely on current information; historical prices play no role. "
        "0.25=occasionally references historical prices; readily updates with new evidence. "
        "0.5=reference prices meaningfully influence buy/sell decisions. "
        "0.75=strongly tied to reference prices; resists revising targets even under compelling evidence. "
        "1.0=decisions governed almost entirely by reference prices; cannot revise assessment."
    ),
    "confirmation_bias": (
        "Score 0.0–1.0. "
        "0.0=actively seeks contrary evidence; tests views against most challenging counterarguments. "
        "0.25=generally balanced; mild preference for confirming information. "
        "0.5=noticeably more receptive to confirming information; filters some contrary signals. "
        "0.75=largely dismisses contrary signals; seeks confirming information. "
        "1.0=processes only information consistent with existing position; contrary signals filtered entirely."
    ),
    "recency_bias": (
        "Score 0.0–1.0. "
        "0.0=weights recent and historical data equally; long-run patterns carry as much weight. "
        "0.25=slight over-weighting of recent events; maintains meaningful historical reference. "
        "0.5=recent events weighted more heavily; historical patterns informative but secondary. "
        "0.75=decisions primarily driven by recent events; historical context largely discounted. "
        "1.0=decisions based almost entirely on most recent events; no meaningful historical reference."
    ),
    "salience_bias": (
        "Score 0.0–1.0. "
        "0.0=probability estimates based on base rates and systematic evidence; vividness has no influence. "
        "0.25=occasionally over-weights vivid events; generally maintains evidence-based probability. "
        "0.5=probability assessment noticeably influenced by dramatic or widely-reported events. "
        "0.75=significantly over-weights dramatic events; base rates largely ignored. "
        "1.0=probability assessment almost entirely driven by the most salient and memorable events."
    ),
    "pattern_matching_bias": (
        "Score 0.0–1.0. "
        "0.0=probability assessed from base rates and evidence; surface similarities ignored. "
        "0.25=mild tendency to draw analogies; generally relies on base rates over pattern matching. "
        "0.5=meaningful tendency to judge situations by resemblance to known historical patterns. "
        "0.75=frequently bases decisions on surface similarity to historical patterns. "
        "1.0=decisions almost entirely driven by pattern matching to known analogues."
    ),
    "loss_aversion": (
        "Score 0.0–1.0. "
        "0.0=treats potential gains and losses symmetrically; decisions based purely on expected value. "
        "0.25=losses slightly more aversive than equivalent gains; mild asymmetry in exits. "
        "0.5=losses feel approximately twice as painful as equivalent gains; extends losing positions. "
        "0.75=strong asymmetry; holds losing positions well past rational exit; cuts winners early. "
        "1.0=refuses to realise losses under almost any circumstances; holds to zero."
    ),
    "winner_selling_loser_holding": (
        "Score 0.0–1.0. "
        "0.0=sells losing positions promptly when thesis invalidated; holds winners as fundamentals support. "
        "0.25=slight tendency to realise gains early and extend losing positions slightly too long. "
        "0.5=meaningfully tends to sell winners at modest gains and extend losers past rational exit. "
        "0.75=systematically realises gains quickly and holds losing positions through major adverse moves. "
        "1.0=almost never realises a loss; sells winners at first available gain."
    ),
    "regret_aversion": (
        "Score 0.0–1.0. "
        "0.0=takes positions based on best analysis regardless of whether others agree. "
        "0.25=mild tendency to avoid positions with high visible regret potential. "
        "0.5=meaningfully biased toward consensus positions to avoid the regret of a visible mistake. "
        "0.75=strongly avoids non-consensus positions; primary concern is not being visibly wrong. "
        "1.0=almost never deviates from consensus; being wrong alone feared more than suboptimal returns."
    ),
    "fomo": (
        "Score 0.0–1.0. "
        "0.0=entirely unmoved by others' gains; no reaction to perceived missed opportunities. "
        "0.25=occasionally notices others' gains; decisions not materially affected. "
        "0.5=periodically enters positions driven by fear of missing a move others are profiting from. "
        "0.75=frequently acts on fear of missing out; enters because others appear to be gaining. "
        "1.0=decisions almost entirely driven by FOMO; cannot sit out a move others are participating in."
    ),
    "mental_accounting": (
        "Score 0.0–1.0. Retail types only. "
        "0.0=treats all capital as a single fungible pool; consistent risk rules across all holdings. "
        "0.25=slight tendency to treat some capital differently by source. "
        "0.5=meaningfully applies different risk rules to different capital pools (e.g. 'house money'). "
        "0.75=different capital pots operate under materially different risk tolerances. "
        "1.0=completely inconsistent behaviour across pools; each mental account under entirely separate rules."
    ),
    "ownership_bias": (
        "Score 0.0–1.0. Retail types only. "
        "0.0=values held and unheld assets identically; sells at the same price willing to buy. "
        "0.25=slight tendency to demand a modest premium to sell versus buy. "
        "0.5=meaningfully overvalues current holdings; requires clear margin above fair value before selling. "
        "0.75=strongly attached to current holdings; significantly reluctant to sell even at fair value. "
        "1.0=treats current holdings as uniquely valuable; almost never willingly sells."
    ),
}


# ── Schema builder ────────────────────────────────────────────────────────────

def build_type_schema(agent_type: str) -> dict:
    """
    Return an Anthropic tool definition dict for the given agent type.

    The tool input_schema has one top-level property per generated persona
    (persona_1, persona_2, persona_3), each containing a narrative field and
    an object of characteristics.  N/A characteristics for the type are
    omitted from the schema.

    The returned dict is ready to be placed in the `tools` list of a
    `client.messages.create()` call.
    """
    ranges = CHARACTERISTIC_RANGES[agent_type]
    char_properties = _build_characteristic_properties(agent_type, ranges)

    persona_schema = {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": (
                    "One paragraph describing this behavioural archetype: "
                    "which signals it attends to and how strongly, what cognitive and emotional "
                    "patterns dominate its decision-making, how it responds to market stress or "
                    "momentum, and what its dominant failure mode is. "
                    "Do NOT describe a fictional individual — describe the behavioural pattern. "
                    "Must be consistent with the characteristic scores assigned."
                ),
            },
            "characteristics": {
                "type": "object",
                "properties": char_properties,
                "required": list(char_properties.keys()),
                "additionalProperties": False,
            },
        },
        "required": ["narrative", "characteristics"],
        "additionalProperties": False,
    }

    return {
        "name": f"generate_{agent_type}_personas",
        "description": (
            f"Generate exactly 3 distinct {agent_type} investor personas for this simulation. "
            "The 3 personas must be clearly different from each other in their approach and characteristics. "
            "Score each characteristic according to the BARS rubric in each field description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "persona_1": persona_schema,
                "persona_2": persona_schema,
                "persona_3": persona_schema,
            },
            "required": ["persona_1", "persona_2", "persona_3"],
            "additionalProperties": False,
        },
    }


def build_all_schemas() -> dict[str, dict]:
    """Return all eight type-specific tool schemas keyed by agent type."""
    from backend.config.sector_config import AGENT_TYPES
    return {atype: build_type_schema(atype) for atype in AGENT_TYPES}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_characteristic_properties(
    agent_type: str,
    ranges: dict[str, tuple[float, float] | None],
) -> dict[str, dict]:
    """
    Build the properties dict for the characteristics object.
    Excludes fields where ranges[name] is None (N/A for this type).
    """
    props = {}
    for name, range_val in ranges.items():
        if range_val is None:
            continue  # N/A for this type — omit from schema
        lo, hi = range_val
        props[name] = {
            "type": "number",
            "minimum": lo,
            "maximum": hi,
            "description": (
                f"[{agent_type} range: {lo:.2f}–{hi:.2f}] "
                + _BARS[name]
            ),
        }
    return props


def get_na_fields_for_type(agent_type: str) -> list[str]:
    """Return the list of characteristic names that are N/A for this type."""
    ranges = CHARACTERISTIC_RANGES[agent_type]
    return [name for name, val in ranges.items() if val is None]


def get_range_midpoint(agent_type: str, characteristic: str) -> float:
    """
    Return the midpoint of the valid range for a characteristic.
    Used as the fallback substitution value when the LLM returns an out-of-range score.
    """
    ranges = CHARACTERISTIC_RANGES[agent_type]
    val = ranges.get(characteristic)
    if val is None:
        raise ValueError(
            f"Characteristic '{characteristic}' is N/A for type '{agent_type}'."
        )
    lo, hi = val
    return (lo + hi) / 2.0


def is_in_range(agent_type: str, characteristic: str, value: float) -> bool:
    """Return True if value is within the valid range for this type and characteristic."""
    ranges = CHARACTERISTIC_RANGES[agent_type]
    val = ranges.get(characteristic)
    if val is None:
        return False
    lo, hi = val
    return lo <= value <= hi
