"""
Engine constants — fixed values that govern simulation mechanics.
All deferred-to-build-time decisions from the spec are resolved here.
See build/EMDOLA_build_notes.md for rationale on each value.
"""

# ── Activation function ────────────────────────────────────────────────────────

# Floor for perceived variance — prevents division by zero in demand calculation.
SIGMA_SQUARED_FLOOR: float = 1e-6

# Sigmoid gate: P(act) = sigmoid(|D| - theta).  A larger value here means the
# sigmoid transitions more sharply around the threshold.
SIGMOID_STEEPNESS: float = 1.0

# ── Signal formation ───────────────────────────────────────────────────────────

# Starting price for all sectors (P_0).  Mispricing is measured as deviation
# from this reference: mispricing(t) = (price(t) - P0) / P0
P0: float = 100.0

# Ambient background influence channel noise level (spec §3.2).
# Noise drawn as N(0, SIGMA_BACKGROUND); intensity scaled by non_market_signal_intensity.
# 0.05 gives meaningful pre/post-shock volatility without overwhelming shock signals.
SIGMA_BACKGROUND: float = 0.05

# ── Market environment ─────────────────────────────────────────────────────────

# Rolling window for realised volatility computation (ticks).
VOLATILITY_WINDOW: int = 20

# ── Strategy adaptability ──────────────────────────────────────────────────────

# Trigger threshold: mean rolling return must fall below this to trigger
# strategy shift (spec §7.5 Step 5).
STRATEGY_ADAPTABILITY_RETURN_THRESHOLD: float = -0.05

# Fraction of memory window used for the rolling return lookback.
STRATEGY_ADAPTABILITY_WINDOW_FRACTION: float = 0.25

# Maximum β_f / β_c shift per trigger event.
STRATEGY_ADAPTABILITY_MAX_SHIFT: float = 0.2

# ── Bias distortion constants (spec §7.5 Step 2) ──────────────────────────────

CONFIRMATION_BIAS_ATTENUATION: float = 0.5   # S *= (1 - cb * 0.5)
SALIENCE_BIAS_AMPLIFICATION: float = 0.4     # S *= (1 + sb * 0.4) if extreme in memory
SALIENCE_SIGMA_THRESHOLD: float = 2.0        # |Δp| > N * σ triggers salience
PATTERN_MATCHING_AMPLIFICATION: float = 0.3  # S *= (1 + pm * 0.3 * trend_match)

# ── Theta modifier constants (spec §7.5 Step 4) ───────────────────────────────

LOSS_AVERSION_THETA_RAISE: float = 0.3       # θ += loss_aversion * 0.3 when crystallising a loss
WINNER_SELLING_THETA_SHIFT: float = 0.2      # θ ± winner_selling * 0.2 (lower for gains, raise for losses)
REGRET_AVERSION_THETA_RAISE: float = 0.2     # θ += regret_aversion * 0.2 for non-consensus action
FOMO_THETA_LOWER: float = 0.3                # θ -= FOMO * 0.3 when influence signal is strong
INSTITUTIONAL_INERTIA_THETA_RAISE: float = 0.2
OWNERSHIP_BIAS_THETA_RAISE: float = 0.2      # θ += ownership_bias * 0.2 for selling a held position
DECISION_STYLE_NOISE_SIGMA: float = 0.1      # θ += N(0, decision_style * 0.1)
OVERCONFIDENCE_THETA_LOWER: float = 0.2      # θ -= overconfidence * 0.2 (unconditional)

# Threshold for "strong" influence signal (triggers FOMO lowering).
FOMO_INFLUENCE_THRESHOLD: float = 0.3

# ── Step 5 execution constants (spec §7.5 Step 5) ─────────────────────────────

INSTITUTIONAL_INERTIA_MAGNITUDE_DAMPEN: float = 0.3  # magnitude *= (1 - inertia * 0.3)
# Per-sector noise on retail risk aversion, modelling mental accounting
# (Thaler 1985, 1999): retail investors treat each sector position as an
# independent mental account, producing inconsistent risk tolerance across
# holdings.  ε_k ~ Uniform(-0.5, 0.5) scaled by the agent's mental_accounting
# characteristic.  Magnitude derived from design reasoning — large enough to
# produce meaningful per-sector variation, small enough not to dominate the
# signal.  Not fitted to data; formal calibration is future work.
MENTAL_ACCOUNTING_NOISE_RANGE: float = 0.5

# ── Sector affinity (spec §7.12) ───────────────────────────────────────────────

RISK_AVERSION_AFFINITY_SHIFT: float = 0.4    # defensive boost / high-vol penalty
TIME_HORIZON_AFFINITY_DAMPEN: float = 0.3    # reduces high-vol sector affinity
# Minimum affinity weight before normalization (prevents a sector being fully zeroed).
AFFINITY_MIN: float = 0.1

# ── Memory decay (build notes) ────────────────────────────────────────────────

# Memory window score → tick count: linear interpolation across these anchors.
# (score, ticks) pairs matching the rubric anchor descriptions.
MEMORY_SCORE_TO_TICKS: list[tuple[float, int]] = [
    (0.00, 1),
    (0.25, 10),
    (0.50, 40),
    (0.75, 150),
    (1.00, 250),
]

# Anchor window for anchor price calculation when no position is held:
# use the high/low of the most recent N ticks from signal memory.
ANCHOR_LOOKBACK_TICKS: int = 5

# ── Population archetype assignment ───────────────────────────────────────────

# Concentration parameter for the Dirichlet distribution used to sample
# archetype proportions when building each run's agent population.
# α < 1 produces fat-tailed distributions — extreme splits (e.g. 25:3:2)
# are common, exploring the full space of plausible cohort compositions.
# α = 0.5 (Jeffreys prior) is the default: fat tails without making
# balanced distributions impossible.  Lower → more extreme; higher → more
# central tendency toward equal thirds.
ARCHETYPE_DIRICHLET_ALPHA: float = 0.5

# ── Type-level Dirichlet for Monte Carlo population composition ────────────────
# Samples how many agents of each TYPE appear in a given run (not just which
# persona within a type).  α < 1 produces fat-tailed distributions — some runs
# are chartist-dominated (many R4/I2), others fundamentalist-dominated (many I1/I3).
# This is what makes MC paths diverge: different type compositions respond
# differently to the same shock, creating the spaghetti spread.
TYPE_DIRICHLET_ALPHA: float = 0.5
# Minimum agents of each type per run: keeps all 8 types represented.
MIN_AGENTS_PER_TYPE: int = 10

# ── Chartist signal amplification ────────────────────────────────────────────
# Scales the weighted log-return sum before tanh in compute_chartist_signal.
# Derived from design reasoning: small background moves (0.5%/tick) should
# produce negligible chartist signal; a sustained 3-tick trend should activate
# the feedback loop.  At 8×, tanh(0.12) ≈ 0.12 — sufficient activation.
# tanh saturation prevents runaway for large shock-driven moves.
# Not fitted to real-world data; formal calibration against stylized facts
# (fat tails, volatility clustering) is future work.
# This is the primary lever for chartist feedback strength.
CHARTIST_SIGNAL_SCALE: float = 8.0

# ── Per-tick cash deployment cap ─────────────────────────────────────────
# Limits buys to this fraction of available cash per tick.  Without this cap,
# agents spend their full cash on tick 1 of a shock and have no buying power
# for the remaining shock window — volume collapses to near zero after the
# first tick even though agents keep evaluating (gate fires but magnitude=0).
# 0.25 spreads a full deployment over ~4 ticks, generating sustained excess
# demand and meaningful price movement throughout the shock window.
# Derived from design reasoning: a ~4-tick spread is short enough to produce
# a sharp price response while long enough to sustain two-sided flow.
# Not fitted to data; formal calibration against real order-flow data is
# future work.
MAX_CASH_DEPLOY_FRACTION_PER_TICK: float = 0.25

# ── Cross-sector coupling ─────────────────────────────────────────────────────
# Global constant governing how much market-wide sentiment spills across sectors.
# Each tick, the average signed influence signal across all sectors is computed
# and added to every sector's influence signal scaled by this factor.
# Mechanism: large macro shocks create a "risk appetite" dimension — institutional
# agents reduce broad equity exposure, not just exposure to directly shocked sectors.
# This produces realistic cross-sector co-movement without hardcoding correlations.
# At 0.0: sectors are fully independent (current behaviour without this feature).
# At 0.20: a strong average signal (0.5) adds ±0.10 to every sector.
# At 0.35: risk-off would become dominant — use carefully.
# Kept as a global constant (not user-configurable) because cross-sector contagion
# is endogenous to how markets work, not a scenario design choice.
CROSS_SECTOR_COUPLING: float = 0.20

# ── Signal gate cap ───────────────────────────────────────────────────────────
# The conviction gate evaluates sigmoid(min(|S_biased|, cap) − θ_eff).
# Capping |S_biased| prevents very large shock-driven signals from saturating
# the sigmoid and making the gate fire at near-certainty for all agents.
# At 0.75 the cap is above typical background noise (|S_biased| ≈ 0.1–0.3) but
# below the upper range of strong-shock signals (|S_biased| ≈ 1.0–1.5).
# Effective firing rates at cap 0.75 (before behavioural theta modifiers):
#   R4 (θ≈0.15): fires ~65% of ticks  →  chartist amplification active
#   I3 (θ≈0.85): fires ~47% of ticks  →  institutional inertia represented
SIGNAL_GATE_CAP: float = 0.75

# ── Shock influence scaling ───────────────────────────────────────────────────
# Maps economic shock magnitude to influence channel intensity.
# base_intensity = min(|magnitude| / INFLUENCE_INTENSITY_SCALE, 1.0)
# At 0.20: a 20% shock → intensity = 1.0 (full narrative saturation).
# At 0.10: a 10% shock → intensity = 0.5 (moderate narrative pressure).
# Governs how strongly the influence channel responds to a given shock size.
INFLUENCE_INTENSITY_SCALE: float = 0.20

# ── Agent trace selection ─────────────────────────────────────────────────────
# Redundancy decay for greedy trace selection.  Each additional trade with the
# same primary tag reduces its effective score by this amount, producing natural
# diversity without hard per-category caps.  At 1.0: the 2nd coupling trade
# scores (4-1)=3, equal to a dissent-only trade; the 3rd scores 2, equal to a
# reversal.  If a scenario genuinely has nothing but coupling signals they still
# appear — just not at the expense of burying any dissent or reversals present.
TRACE_REDUNDANCY_DECAY: float = 1.0

# Maximum traces to surface per run.
TRACE_MAX_COUNT: int = 30

# ── Simulation defaults ────────────────────────────────────────────────────────

DEFAULT_TOTAL_TICKS: int = 250
DEFAULT_AGENTS_PER_TYPE: int = 30
DEFAULT_NON_MARKET_SIGNAL_INTENSITY: float = 0.40

# ── GICS sector classification ─────────────────────────────────────────────────

# Indices correspond to SECTORS list in sector_config.py (0-based, 0=Energy … 10=Real Estate).
DEFENSIVE_SECTOR_INDICES: frozenset[int] = frozenset({4, 5, 9})  # Staples, HC, Utilities
HIGH_VOL_SECTOR_INDICES: frozenset[int] = frozenset({3, 7})       # Consumer Disc, Tech
