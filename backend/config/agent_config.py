"""
Per-agent-type configuration: AUM, evaluation probability, position scale,
starting cash weights, and default sector allocations.

All values are Layer 1 (engine-defined, hard-coded) structural parameters.
AUM is equity-focused global AUM ÷ 30 agents (see build notes for derivation).
Sector weights are defaults and are user-overridable in Setup Tab 3.
"""

from __future__ import annotations

# ── AUM per agent unit (billions USD) ─────────────────────────────────────────
# Equity-focused global AUM ÷ 30 representative agents per type.
AUM_PER_AGENT: dict[str, float] = {
    "R1": 500.0,    # $15T passive equity / 30
    "R2": 133.0,    # $4T self-directed equity / 30
    "R3": 133.0,    # $4T HNW equity / 30
    "R4": 50.0,     # $1.5T speculative equity / 30
    "I1": 2667.0,   # ~$80T active AM equity / 30
    "I2": 100.0,    # ~$3T HF net equity / 30
    "I3": 583.0,    # ~$17.5T pension equity sleeve / 30
    "I4": 121.0,    # ~$3.6T SWF public equity / 30
}

# ── Evaluation probability (Stage 1 activation, spec §6.3) ───────────────────
# Probability that an agent evaluates its portfolio on any given tick.
EVALUATION_PROBABILITY: dict[str, float] = {
    "R1": 0.15,   # Passive — rarely reviews
    "R2": 0.80,   # Active self-directed — monitors daily
    "R3": 0.50,   # HNW — periodic review
    "R4": 0.90,   # Speculative — constant attention
    "I1": 1.00,   # Institutional — dedicated daily process
    "I2": 1.00,
    "I3": 1.00,
    "I4": 1.00,
}

# ── Position scale: max trade as fraction of AUM (spec §7.5 Step 5) ──────────
POSITION_SCALE: dict[str, float] = {
    "R1": 0.03,   # Rare, small rebalancing moves
    "R2": 0.10,   # Active but retail-sized
    "R3": 0.08,   # Meaningful but considered
    "R4": 0.20,   # Concentrated, high-conviction
    "I1": 0.05,   # Benchmark-relative, controlled sizing
    "I2": 0.15,   # High-conviction concentrated positions
    "I3": 0.02,   # Liability-constrained, small incremental moves
    "I4": 0.03,   # Large AUM; small % = large absolute
}

# ── Is institutional (affects inertia modifier and ownership/mental accounting) ─
IS_INSTITUTIONAL: dict[str, bool] = {
    "R1": False, "R2": False, "R3": False, "R4": False,
    "I1": True,  "I2": True,  "I3": True,  "I4": True,
}

# ── Starting cash weight (fraction of AUM) ────────────────────────────────────
# Remainder (1 - cash_weight) is distributed across equity sectors per
# STARTING_SECTOR_WEIGHTS below.
STARTING_CASH_WEIGHT: dict[str, float] = {
    "R1": 0.03,
    "R2": 0.15,
    "R3": 0.12,
    "R4": 0.30,
    "I1": 0.05,
    "I2": 0.30,
    "I3": 0.08,
    "I4": 0.10,
}

# ── Starting sector weights (as % of equity allocation, pre-cash) ─────────────
# Sector order: En Mat Ind CDi Sta HC Fin Tech CS Uti RE (indices 0–10).
# These are percentages that sum to 100 — they are multiplied by (1 - cash_weight)
# to get the actual portfolio weight per sector.
STARTING_SECTOR_PCT: dict[str, list[float]] = {
    # R1: market-cap proxy
    "R1": [5, 4, 9, 10, 7, 12, 14, 25, 8,  3, 3],
    # R2: Tech/Fin tilt
    "R2": [3, 2, 7, 12, 4, 10, 18, 35, 8,  0, 1],
    # R3: market-cap + quality tilt (slight HC/Staples overweight)
    "R3": [5, 4, 9,  9, 8, 14, 14, 22, 8,  4, 3],
    # R4: momentum concentrated
    "R4": [0, 0, 5, 25, 0,  0, 15, 45, 10, 0, 0],
    # I1: benchmark-hugging, slight Tech overweight
    "I1": [5, 4, 9, 10, 7, 12, 13, 27, 8,  3, 2],
    # I2: Tech/Fin overweight
    "I2": [5, 5, 5, 15, 0, 10, 25, 25, 10, 0, 0],
    # I3: equity sleeve — defensives (HC, Fin, Uti, Staples)
    "I3": [0, 0, 9,  0, 18, 28, 25, 0,  0, 20, 0],
    # I4: infrastructure/energy diversified
    "I4": [18, 12, 18, 0, 7, 10, 15, 15, 0, 5, 0],
}

# Validate all weights sum to 100 (±0.1 for floating-point tolerance).
for _atype, _pcts in STARTING_SECTOR_PCT.items():
    _total = sum(_pcts)
    assert abs(_total - 100.0) < 0.1, (
        f"STARTING_SECTOR_PCT['{_atype}'] sums to {_total}, expected 100."
    )

# ── Characteristic ranges per agent type (spec §7.7, matrix) ─────────────────
# Used by the LLM schema builder (llm/schemas.py) to enforce per-type min/max.
# Format: {characteristic_name: (min, max)}

CHARACTERISTIC_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "R1": {
        "risk_aversion":              (0.35, 0.65),
        "time_horizon":               (0.70, 1.00),
        "belief_formation":           (0.10, 0.40),
        "information_quality":        (0.00, 0.25),
        "herding_sensitivity":        (0.25, 0.55),
        "decision_style":             (0.10, 0.50),
        "strategy_adaptability":      (0.00, 0.30),
        "conviction_threshold":       (0.55, 0.90),
        "institutional_inertia":      None,           # N/A for retail
        "memory_window":              (0.10, 0.40),
        "overconfidence":             (0.10, 0.40),
        "anchoring":                  (0.30, 0.65),
        "confirmation_bias":          (0.30, 0.65),
        "recency_bias":               (0.40, 0.75),
        "salience_bias":              (0.40, 0.75),
        "pattern_matching_bias":      (0.30, 0.65),
        "loss_aversion":              (0.50, 0.85),
        "winner_selling_loser_holding": (0.30, 0.55),
        "regret_aversion":            (0.50, 0.85),
        "fomo":                       (0.15, 0.50),
        "mental_accounting":          (0.50, 0.85),
        "ownership_bias":             (0.30, 0.65),
    },
    "R2": {
        "risk_aversion":              (0.15, 0.50),
        "time_horizon":               (0.10, 0.45),
        "belief_formation":           (0.45, 0.80),
        "information_quality":        (0.15, 0.45),
        "herding_sensitivity":        (0.45, 0.80),
        "decision_style":             (0.40, 0.75),
        "strategy_adaptability":      (0.40, 0.70),
        "conviction_threshold":       (0.10, 0.45),
        "institutional_inertia":      None,
        "memory_window":              (0.15, 0.45),
        "overconfidence":             (0.50, 0.85),
        "anchoring":                  (0.35, 0.70),
        "confirmation_bias":          (0.50, 0.80),
        "recency_bias":               (0.55, 0.85),
        "salience_bias":              (0.50, 0.80),
        "pattern_matching_bias":      (0.50, 0.80),
        "loss_aversion":              (0.45, 0.80),
        "winner_selling_loser_holding": (0.55, 0.85),
        "regret_aversion":            (0.20, 0.60),
        "fomo":                       (0.50, 0.85),
        "mental_accounting":          (0.40, 0.75),
        "ownership_bias":             (0.40, 0.70),
    },
    "R3": {
        "risk_aversion":              (0.30, 0.60),
        "time_horizon":               (0.50, 0.85),
        "belief_formation":           (0.15, 0.50),
        "information_quality":        (0.35, 0.70),
        "herding_sensitivity":        (0.15, 0.45),
        "decision_style":             (0.15, 0.50),
        "strategy_adaptability":      (0.20, 0.55),
        "conviction_threshold":       (0.40, 0.70),
        "institutional_inertia":      None,
        "memory_window":              (0.20, 0.55),
        "overconfidence":             (0.35, 0.70),
        "anchoring":                  (0.25, 0.60),
        "confirmation_bias":          (0.25, 0.60),
        "recency_bias":               (0.25, 0.60),
        "salience_bias":              (0.30, 0.65),
        "pattern_matching_bias":      (0.30, 0.65),
        "loss_aversion":              (0.35, 0.70),
        "winner_selling_loser_holding": (0.25, 0.60),
        "regret_aversion":            (0.25, 0.60),
        "fomo":                       (0.15, 0.50),
        "mental_accounting":          (0.25, 0.60),
        "ownership_bias":             (0.20, 0.55),
    },
    "R4": {
        "risk_aversion":              (0.00, 0.30),
        "time_horizon":               (0.00, 0.30),
        "belief_formation":           (0.60, 1.00),
        "information_quality":        (0.00, 0.35),
        "herding_sensitivity":        (0.60, 1.00),
        "decision_style":             (0.60, 1.00),
        "strategy_adaptability":      (0.60, 0.95),
        "conviction_threshold":       (0.00, 0.30),
        "institutional_inertia":      None,
        "memory_window":              (0.05, 0.30),
        "overconfidence":             (0.65, 1.00),
        "anchoring":                  (0.40, 0.75),
        "confirmation_bias":          (0.65, 1.00),
        "recency_bias":               (0.65, 1.00),
        "salience_bias":              (0.65, 1.00),
        "pattern_matching_bias":      (0.55, 0.90),
        "loss_aversion":              (0.55, 0.90),
        "winner_selling_loser_holding": (0.65, 1.00),
        "regret_aversion":            (0.05, 0.40),
        "fomo":                       (0.70, 1.00),
        "mental_accounting":          (0.55, 0.90),
        "ownership_bias":             (0.45, 0.80),
    },
    "I1": {
        "risk_aversion":              (0.25, 0.55),
        "time_horizon":               (0.40, 0.70),
        "belief_formation":           (0.10, 0.45),
        "information_quality":        (0.55, 0.85),
        "herding_sensitivity":        (0.25, 0.65),
        "decision_style":             (0.15, 0.45),
        "strategy_adaptability":      (0.20, 0.50),
        "conviction_threshold":       (0.55, 0.85),
        "institutional_inertia":      (0.35, 0.65),
        "memory_window":              (0.35, 0.65),
        "overconfidence":             (0.50, 0.85),
        "anchoring":                  (0.35, 0.65),
        "confirmation_bias":          (0.40, 0.70),
        "recency_bias":               (0.25, 0.55),
        "salience_bias":              (0.20, 0.50),
        "pattern_matching_bias":      (0.30, 0.60),
        "loss_aversion":              (0.30, 0.60),
        "winner_selling_loser_holding": (0.20, 0.50),
        "regret_aversion":            (0.50, 0.85),
        "fomo":                       (0.20, 0.55),
        "mental_accounting":          None,           # N/A for institutional
        "ownership_bias":             None,
    },
    "I2": {
        "risk_aversion":              (0.15, 0.50),
        "time_horizon":               (0.20, 0.60),
        "belief_formation":           (0.10, 0.55),
        "information_quality":        (0.65, 1.00),
        "herding_sensitivity":        (0.05, 0.40),
        "decision_style":             (0.10, 0.45),
        "strategy_adaptability":      (0.30, 0.65),
        "conviction_threshold":       (0.60, 0.90),
        "institutional_inertia":      (0.05, 0.40),
        "memory_window":              (0.25, 0.60),
        "overconfidence":             (0.55, 0.90),
        "anchoring":                  (0.30, 0.65),
        "confirmation_bias":          (0.35, 0.70),
        "recency_bias":               (0.20, 0.55),
        "salience_bias":              (0.15, 0.45),
        "pattern_matching_bias":      (0.30, 0.60),
        "loss_aversion":              (0.25, 0.55),
        "winner_selling_loser_holding": (0.10, 0.40),
        "regret_aversion":            (0.10, 0.45),
        "fomo":                       (0.10, 0.40),
        "mental_accounting":          None,
        "ownership_bias":             None,
    },
    "I3": {
        "risk_aversion":              (0.65, 0.90),
        "time_horizon":               (0.85, 1.00),
        "belief_formation":           (0.00, 0.25),
        "information_quality":        (0.50, 0.80),
        "herding_sensitivity":        (0.05, 0.35),
        "decision_style":             (0.05, 0.35),
        "strategy_adaptability":      (0.00, 0.20),
        "conviction_threshold":       (0.70, 1.00),
        "institutional_inertia":      (0.70, 0.95),
        "memory_window":              (0.50, 0.85),
        "overconfidence":             (0.10, 0.40),
        "anchoring":                  (0.40, 0.75),
        "confirmation_bias":          (0.30, 0.65),
        "recency_bias":               (0.10, 0.40),
        "salience_bias":              (0.10, 0.40),
        "pattern_matching_bias":      (0.20, 0.50),
        "loss_aversion":              (0.60, 0.90),
        "winner_selling_loser_holding": (0.15, 0.45),
        "regret_aversion":            (0.50, 0.80),
        "fomo":                       (0.00, 0.20),
        "mental_accounting":          None,
        "ownership_bias":             None,
    },
    "I4": {
        "risk_aversion":              (0.45, 0.75),
        "time_horizon":               (0.85, 1.00),
        "belief_formation":           (0.00, 0.30),
        "information_quality":        (0.60, 0.90),
        "herding_sensitivity":        (0.00, 0.30),
        "decision_style":             (0.05, 0.35),
        "strategy_adaptability":      (0.00, 0.15),
        "conviction_threshold":       (0.75, 1.00),
        "institutional_inertia":      (0.80, 1.00),
        "memory_window":              (0.55, 0.90),
        "overconfidence":             (0.10, 0.40),
        "anchoring":                  (0.50, 0.80),
        "confirmation_bias":          (0.30, 0.65),
        "recency_bias":               (0.05, 0.35),
        "salience_bias":              (0.10, 0.40),
        "pattern_matching_bias":      (0.25, 0.55),
        "loss_aversion":              (0.55, 0.85),
        "winner_selling_loser_holding": (0.10, 0.40),
        "regret_aversion":            (0.65, 0.95),
        "fomo":                       (0.00, 0.15),
        "mental_accounting":          None,
        "ownership_bias":             None,
    },
}

# Ordered list of all 21 characteristics (the LLM produces exactly these fields).
CHARACTERISTIC_NAMES: list[str] = [
    "risk_aversion",
    "time_horizon",
    "belief_formation",
    "information_quality",
    "herding_sensitivity",
    "decision_style",
    "strategy_adaptability",
    "conviction_threshold",
    "institutional_inertia",
    "memory_window",
    "overconfidence",
    "anchoring",
    "confirmation_bias",
    "recency_bias",
    "salience_bias",
    "pattern_matching_bias",
    "loss_aversion",
    "winner_selling_loser_holding",
    "regret_aversion",
    "fomo",
    "mental_accounting",
    "ownership_bias",
]
