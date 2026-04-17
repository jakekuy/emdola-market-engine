"""
GICS sector definitions, price sensitivity (lambda) defaults, and sector
affinity base weights per agent type.

Sector index order (used throughout the engine):
  0  Energy
  1  Materials
  2  Industrials
  3  Consumer Discretionary
  4  Consumer Staples
  5  Health Care
  6  Financials
  7  Information Technology
  8  Communication Services
  9  Utilities
  10 Real Estate
"""

from __future__ import annotations

# ── Sector list ────────────────────────────────────────────────────────────────

SECTORS: list[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]

NUM_SECTORS: int = len(SECTORS)

SECTOR_INDEX: dict[str, int] = {name: i for i, name in enumerate(SECTORS)}

# ── Lambda defaults (price sensitivity per sector) ────────────────────────────
# Higher λ → lower liquidity → larger price move per unit excess demand.
# User-adjustable in Setup Tab 1; these are the engine defaults.
#
# Values are grounded in relative Amihud (2002) illiquidity ratios across GICS
# sectors: IT and Financials are the most liquid (highest daily turnover, largest
# float); Materials and Utilities are less liquid; Real Estate (listed REITs) sits
# alongside Utilities as a smaller, lower-volume, rate-sensitive sector.
# NOTE: the GICS Real Estate sector is listed REITs — not physical property.
# λ is calibrated to equity-market illiquidity, not direct property illiquidity.
# Range 0.80–2.00 gives a plausible 2.5× spread between most and least liquid,
# consistent with empirical Amihud ratios for large-cap vs. REIT equities.

LAMBDA_DEFAULTS: list[float] = [
    1.50,  # Energy          — moderate liquidity; commodity-driven moves
    2.00,  # Materials       — smaller float; higher price impact per unit
    1.20,  # Industrials     — mid-cap heavy; moderate liquidity
    1.20,  # Consumer Discretionary — large, liquid sector
    1.00,  # Consumer Staples — highly liquid defensive
    1.20,  # Health Care     — broad sector; moderate liquidity
    0.80,  # Financials      — very high liquidity; large daily volume
    0.80,  # Information Technology — highest liquidity by market cap
    1.20,  # Communication Services — moderate; mixed large/mid cap
    1.80,  # Utilities       — lower liquidity; rate-sensitive
    1.80,  # Real Estate     — listed REITs; similar liquidity profile to Utilities
]

assert len(LAMBDA_DEFAULTS) == NUM_SECTORS, "Lambda defaults must have one entry per sector."

# ── Sector affinity base weights per agent type ────────────────────────────────
# Relative attention multipliers: 1.0 = neutral; >1.0 = heightened; <1.0 = reduced.
# These are Layer 1 (engine-defined, empirically grounded) affinity weights.
# Applied as Component 1 in A_i(k) = base_affinity(type, k) × char_modifier(i, k).
# Normalization is applied at compute time in signals.py.
#
# Sector index order: En Mat Ind CDi Sta HC  Fin Tech CS  Uti RE

BASE_AFFINITY: dict[str, list[float]] = {
    # R1 — Passive: tracks market-cap weights → near-equal across sectors.
    "R1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

    # R2 — Active self-directed: heightened Tech/CDi/CS; reduced Utilities/RE.
    "R2": [0.8, 0.7, 0.9, 1.3, 0.7, 1.0, 1.2, 1.8, 1.2, 0.4, 0.5],

    # R3 — HNW: Financials/HC/Staples; below-average Materials/Utilities.
    "R3": [0.9, 0.7, 1.0, 1.0, 1.2, 1.4, 1.3, 1.1, 1.0, 0.7, 0.9],

    # R4 — Speculative: strong Tech/CDi; near-zero Staples/Utilities/RE.
    "R4": [0.5, 0.4, 0.7, 1.6, 0.3, 0.5, 1.1, 2.0, 1.3, 0.2, 0.3],

    # I1 — Long-only AM: benchmark-relative; slight Tech overweight, no structural avoidance.
    "I1": [1.0, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 0.9, 0.9],

    # I2 — Hedge fund: opportunistic, no fixed affinity.
    "I2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

    # I3 — Pension/insurance: defensives and Financials; below-average Tech/Materials.
    "I3": [0.6, 0.5, 1.0, 0.5, 1.5, 1.6, 1.5, 0.5, 0.6, 1.6, 0.8],

    # I4 — SWF: Energy/Industrials (infrastructure); broad diversification.
    "I4": [1.5, 1.2, 1.5, 0.7, 0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 0.8],
}

AGENT_TYPES: list[str] = list(BASE_AFFINITY.keys())

for _atype, _weights in BASE_AFFINITY.items():
    assert len(_weights) == NUM_SECTORS, (
        f"BASE_AFFINITY['{_atype}'] must have {NUM_SECTORS} entries."
    )
