"""
diagnose_dynamics.py — single-run diagnostic for shock-period dynamics.

Runs one simulation with a fixed seed from the last Iran batch, instrumenting
the activation pipeline to capture per-tick:
  - S_biased distribution (signal strength before gate)
  - Gate firing rate (% of evaluations that produce a trade)
  - Magnitude: requested vs actual (cash constraint impact)
  - Direction balance (buy fraction)
  - Agent cash weight distribution

Then prints a comparison table: quiet period (ticks 20-39) vs shock period
(ticks 40-64) vs post-shock (ticks 65-89).

Run from the project root:
    python diagnose_dynamics.py
"""

import sys, json, math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Make backend importable ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.calibration import CalibrationInput
from backend.models.profile import ProfileSet
from backend.simulation.population import build_population
from backend.simulation.environment import MarketEnvironment, InfluenceSignal
from backend.simulation.shocks import ShockProcessor
from backend.simulation.signals import (
    compute_background_influence,
    compute_composite_signal,
    compute_sector_affinity_weights,
)
from backend.simulation.agent import Agent
from backend.config.sector_config import NUM_SECTORS, AGENT_TYPES, SECTORS
from backend.config.constants import (
    KAPPA, SIGMA_SQUARED_FLOOR, SIGMOID_STEEPNESS, SIGNAL_GATE_CAP,
    CONFIRMATION_BIAS_ATTENUATION, SALIENCE_BIAS_AMPLIFICATION,
    SALIENCE_SIGMA_THRESHOLD, PATTERN_MATCHING_AMPLIFICATION,
    LOSS_AVERSION_THETA_RAISE, WINNER_SELLING_THETA_SHIFT,
    REGRET_AVERSION_THETA_RAISE, FOMO_THETA_LOWER, FOMO_INFLUENCE_THRESHOLD,
    INSTITUTIONAL_INERTIA_THETA_RAISE, OWNERSHIP_BIAS_THETA_RAISE,
    DECISION_STYLE_NOISE_SIGMA, INSTITUTIONAL_INERTIA_MAGNITUDE_DAMPEN,
    MENTAL_ACCOUNTING_NOISE_RANGE, ANCHOR_LOOKBACK_TICKS,
    MAX_CASH_DEPLOY_FRACTION_PER_TICK,
)

# ── Load calibration + profiles from last Iran batch ─────────────────────────
BATCH_DIR = Path("data/runs/batch_477b447b578f")

with open(BATCH_DIR / "calibration.json") as f:
    cal_raw = json.load(f)

# Rebuild CalibrationInput
calibration = CalibrationInput.model_validate(cal_raw)

# Load profiles
with open(BATCH_DIR / "results.json") as f:
    results_raw = json.load(f)

# Use the seed from run 1
run1_seed = results_raw["run_results"][0]["seed_record"]
numpy_seed = run1_seed["numpy_seed"]
persona_seed_str = run1_seed["persona_seed"]

print(f"Batch: {BATCH_DIR.name}")
print(f"Run 1 numpy_seed: {numpy_seed}")
print(f"Shocks: {[(s.shock_type, s.channel, s.magnitude, s.onset_tick, s.duration) for s in calibration.shocks]}")
print()

# ── Rebuild profiles ──────────────────────────────────────────────────────────
# We need ProfileSet — load from stored persona_narrative_md or reconstruct
# from calibration. For diagnostics, build with a fresh profile call if needed.
# Simplest: use enable_narrative=False and skip LLM; profiles from stored data.
# Actually we need the actual profiles used. Reconstruct via profile_generator
# with the stored seed — but that requires LLM. Instead: build_population
# doesn't strictly need profiles to exist; we can use build_population with
# a dummy profile if the stored ProfileSet isn't serialised.
# The results.json doesn't include the full ProfileSet. Use the engine to
# regenerate with enable_narrative=False, or bypass by using fixed characteristics.

# Simpler approach: just rebuild population with build_population, which
# will use the stored profiles_id if available, or we can just call it
# and accept that characteristics may differ slightly from the original run.
# For diagnostics, exact reproduction matters less than the structural patterns.

# Attempt to load stored profiles from data dir
profile_path = BATCH_DIR / "profiles.json"
if profile_path.exists():
    with open(profile_path) as f:
        profiles_raw = json.load(f)
    profiles = ProfileSet.model_validate(profiles_raw)
else:
    # No stored profiles — load from the BatchEngine's stored profiles_id if any,
    # or fall back to the engine's default profile generation path.
    # For diagnostics: use the BatchEngine with enable_narrative=False to get
    # profiles without LLM narrative generation, or use any stored profile JSON.
    # Simplest fallback: look for any profiles JSON in the data directory.
    stored = list(Path("data/profiles").glob("*.json")) if Path("data/profiles").exists() else []
    if stored:
        with open(stored[-1]) as f:
            raw = json.load(f)
        inner = raw.get("profiles", raw)
        # Stored profiles omit agent_type on each persona — inject from dict key
        for atype, persona_list in inner.items():
            for persona in persona_list:
                if "agent_type" not in persona:
                    persona["agent_type"] = atype
        profiles = ProfileSet.model_validate({"personas": inner})
    else:
        raise FileNotFoundError(
            "No profiles.json found in batch dir and no stored profiles available. "
            "Run a batch first or add profiles.json to the batch directory."
        )

agents, persona_seed = build_population(calibration, profiles, numpy_seed)
print(f"Population: {len(agents)} agents across {len(AGENT_TYPES)} types")
print()

# ── Per-tick diagnostic accumulators ─────────────────────────────────────────
# For each tick, capture:
#   s_biased_abs: list of |S_biased| for every agent×sector evaluation that
#                 reached Step 2 (i.e., passed Layer 1 eval probability gate)
#   gate_fired:   count of gate fires
#   gate_total:   count of gate evaluations
#   mag_requested: sum of requested magnitudes
#   mag_actual:    sum of actual magnitudes (after cash cap)
#   buy_count, sell_count
#   cash_weights: list of agent cash weights at tick start

TickStats = dict  # keys defined below

tick_data: dict[int, TickStats] = {}

# ── Instrumented activation function ─────────────────────────────────────────

def _apply_bias_distortions(S, agent, sector_k, env):
    """Reproduce Step 2 from activation.py."""
    p = agent.profile
    if p.confirmation_bias > 0.0:
        has_position = agent.sector_weights[sector_k] > 1e-6
        signal_dir = math.copysign(1.0, S) if S != 0.0 else 0.0
        if has_position and signal_dir < 0:
            S *= (1.0 - p.confirmation_bias * CONFIRMATION_BIAS_ATTENUATION)
    if p.anchoring > 0.0:
        if agent.sector_weights[sector_k] > 1e-6:
            anchor_price = float(agent.purchase_price[sector_k])
        else:
            recent = list(agent.price_memory[sector_k])[-ANCHOR_LOOKBACK_TICKS:]
            if not recent:
                anchor_price = env.prices[sector_k]
            elif S > 0:
                anchor_price = float(min(recent))
            else:
                anchor_price = float(max(recent))
        if anchor_price > 0.0:
            anchor_signal = (anchor_price - env.prices[sector_k]) / anchor_price
            S = S * (1.0 - p.anchoring) + anchor_signal * p.anchoring
    if p.salience_bias > 0.0:
        sigma = env.get_price_history_sigma(sector_k)
        if sigma > 0.0:
            mem = agent.price_memory[sector_k]
            if len(mem) >= 2:
                prices = list(mem)
                last_lr = abs(np.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0.0)
                if last_lr > SALIENCE_SIGMA_THRESHOLD * sigma:
                    S *= (1.0 + p.salience_bias * SALIENCE_BIAS_AMPLIFICATION)
    if p.pattern_matching_bias > 0.0 and S != 0.0:
        prices_list = list(agent.price_memory[sector_k])
        if len(prices_list) >= 3:
            recent = prices_list[-min(5, len(prices_list)):]
            lrs = [np.log(recent[i+1]/recent[i]) for i in range(len(recent)-1) if recent[i]>0]
            mean_ret = float(np.mean(lrs)) if lrs else 0.0
            trend_dir = math.copysign(1.0, mean_ret) if mean_ret != 0.0 else 0.0
            signal_dir = math.copysign(1.0, S)
            if trend_dir * signal_dir > 0:
                S *= (1.0 + p.pattern_matching_bias * PATTERN_MATCHING_AMPLIFICATION)
    return S


def run_instrumented_activation(agent, env, tick, rng, stats):
    """
    Reproduce the 5-step activation from activation.py, capturing diagnostics.
    Returns trades list (same as run_agent_activation).
    """
    agent.check_strategy_adaptability(tick)
    p = agent.profile

    sector_affinities = compute_sector_affinity_weights(
        agent_type=agent.agent_type,
        risk_aversion=p.risk_aversion,
        belief_formation=p.belief_formation,
        time_horizon=p.time_horizon,
        information_quality=p.information_quality,
        price_memory_all=agent.price_memory,
        current_prices=env.prices,
        p0_effective=env.p0_effective,
    )

    trades = []

    for k in range(NUM_SECTORS):
        # Step 1
        S_eff = compute_composite_signal(
            beta_f=agent.beta_f, beta_c=agent.beta_c,
            herding_sensitivity=p.herding_sensitivity,
            information_quality=p.information_quality,
            recency_bias=p.recency_bias, time_horizon=p.time_horizon,
            is_institutional=agent.is_institutional,
            memory_ticks=agent.memory_ticks,
            price_k=env.prices[k],
            price_memory_k=agent.price_memory[k],
            broadcast_direction_k=env.influence_signal.direction[k],
            broadcast_intensity_k=env.influence_signal.intensity[k],
            sector_affinity_k=sector_affinities[k],
            rng=rng, p0_effective=float(env.p0_effective[k]),
        )

        # Step 2
        S_biased = _apply_bias_distortions(S_eff, agent, k, env)
        if S_biased == 0.0:
            continue

        # Capture S_biased magnitude
        stats["s_biased_abs"].append(abs(S_biased))
        stats["gate_total"] += 1

        # Step 3
        sigma = env.get_price_history_sigma(k)
        sigma_sq = max(sigma**2, SIGMA_SQUARED_FLOOR)
        ra = max(p.risk_aversion, 0.05)
        if p.mental_accounting is not None and p.mental_accounting > 0.0:
            noise = float(rng.uniform(-1.0, 1.0)) * MENTAL_ACCOUNTING_NOISE_RANGE * p.mental_accounting
            ra = max(0.05, ra + noise)
        sigma_sq_perceived = max(sigma_sq * (1.0 - p.overconfidence * KAPPA), SIGMA_SQUARED_FLOOR)
        D_i = S_biased / (ra * sigma_sq_perceived)
        if D_i == 0.0:
            continue

        # Step 4 — gate
        direction = 1 if D_i > 0 else -1
        current_price = env.prices[k]
        sector_return = agent.get_current_return_for_sector(k, current_price)
        theta = p.conviction_threshold

        if direction < 0 and sector_return < 0.0:
            theta += p.loss_aversion * LOSS_AVERSION_THETA_RAISE
        if direction < 0:
            wslh = p.winner_selling_loser_holding
            if sector_return >= 0.0:
                theta -= wslh * WINNER_SELLING_THETA_SHIFT
            else:
                theta += wslh * WINNER_SELLING_THETA_SHIFT
        influence_dir = env.influence_signal.direction[k]
        if influence_dir != 0.0 and influence_dir * direction < 0:
            theta += p.regret_aversion * REGRET_AVERSION_THETA_RAISE
        influence_intensity = env.influence_signal.intensity[k]
        if influence_intensity > FOMO_INFLUENCE_THRESHOLD:
            theta -= p.fomo * FOMO_THETA_LOWER
        if p.institutional_inertia is not None:
            theta += p.institutional_inertia * INSTITUTIONAL_INERTIA_THETA_RAISE
        if p.ownership_bias is not None and direction < 0:
            if agent.sector_weights[k] > 1e-6:
                theta += p.ownership_bias * OWNERSHIP_BIAS_THETA_RAISE
        noise_sigma = p.decision_style * DECISION_STYLE_NOISE_SIGMA
        if noise_sigma > 0.0:
            theta += float(rng.normal(0.0, noise_sigma))

        d_gated = min(abs(S_biased), SIGNAL_GATE_CAP)
        x = SIGMOID_STEEPNESS * (d_gated - theta)
        p_act = 1.0 / (1.0 + math.exp(-x))
        fired = bool(rng.random() < p_act)

        if not fired:
            stats["gate_blocked"] += 1
            continue
        stats["gate_fired"] += 1

        # Step 5
        ra2 = max(p.risk_aversion, 0.05)
        magnitude = min(abs(S_biased) / ra2, 1.0) * agent.aum * agent.position_scale
        if p.institutional_inertia is not None and p.institutional_inertia > 0.0:
            magnitude *= (1.0 - p.institutional_inertia * INSTITUTIONAL_INERTIA_MAGNITUDE_DAMPEN)
        magnitude = max(0.0, magnitude)
        stats["mag_requested"] += magnitude

        # Execute (cash cap applied inside)
        actual = agent.execute_trade(k, direction, magnitude, tick)
        stats["mag_actual"] += actual

        if direction > 0:
            stats["buys"] += 1
        else:
            stats["sells"] += 1

        env.accumulate_trade(k, direction * actual)
        trades.append((k, direction, actual))

    return trades


# ── Main run loop ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(numpy_seed)
env = MarketEnvironment(lambdas=list(calibration.market.lambdas))

from backend.simulation.population import agents_by_type

total_ticks = calibration.run.total_ticks
shock_proc = ShockProcessor(calibration.shocks, total_ticks)
nm_intensity = calibration.market.non_market_signal_intensity

PERIODS = {
    "pre_shock":  (20, 39),
    "shock":      (40, 64),
    "post_shock": (65, 89),
}

period_agg: dict[str, dict] = {
    p: {"s_biased_abs": [], "gate_total": 0, "gate_fired": 0, "gate_blocked": 0,
        "mag_requested": 0.0, "mag_actual": 0.0, "buys": 0, "sells": 0,
        "cash_weights": [], "influence_intensities": []}
    for p in PERIODS
}

print("Running instrumented simulation...")

for tick in range(total_ticks):
    env.tick = tick
    env.excess_demand[:] = 0.0
    env.volume[:] = 0.0

    p0_eff = shock_proc.get_p0_effective(tick)
    env.update_p0_effective(p0_eff)

    inf_dirs, inf_ints = compute_background_influence(nm_intensity, rng)
    shock_signal = shock_proc.get_influence_signal(tick)
    if shock_signal is not None:
        covered = shock_signal.intensity > 0
        combined_dirs = np.where(covered, shock_signal.direction, inf_dirs)
        combined_ints = np.where(covered, shock_signal.intensity, inf_ints)
        env.influence_signal = InfluenceSignal(direction=combined_dirs, intensity=combined_ints)
    else:
        env.influence_signal = InfluenceSignal(direction=inf_dirs, intensity=inf_ints)

    # Determine which period this tick belongs to
    period_key = None
    for pk, (t0, t1) in PERIODS.items():
        if t0 <= tick <= t1:
            period_key = pk
            break

    # Tick-level stats
    tick_stats = {
        "s_biased_abs": [], "gate_total": 0, "gate_fired": 0, "gate_blocked": 0,
        "mag_requested": 0.0, "mag_actual": 0.0, "buys": 0, "sells": 0,
    }

    # Capture cash weights at tick start
    if period_key:
        cash_ws = [a.cash_weight for a in agents]
        period_agg[period_key]["cash_weights"].extend(cash_ws)
        # Capture influence intensity (mean across sectors)
        mean_inf = float(np.mean(env.influence_signal.intensity))
        period_agg[period_key]["influence_intensities"].append(mean_inf)

    agent_indices = rng.permutation(len(agents))
    for idx in agent_indices:
        agent = agents[idx]
        if rng.random() >= agent.eval_probability:
            continue
        run_instrumented_activation(agent, env, tick, rng, tick_stats)

    # Merge tick stats into period aggregator
    if period_key:
        agg = period_agg[period_key]
        agg["s_biased_abs"].extend(tick_stats["s_biased_abs"])
        for k in ("gate_total", "gate_fired", "gate_blocked", "buys", "sells"):
            agg[k] += tick_stats[k]
        agg["mag_requested"] += tick_stats["mag_requested"]
        agg["mag_actual"]    += tick_stats["mag_actual"]

    avg_aum = sum(a.aum for a in agents) / len(agents)
    env.apply_price_update(len(agents), aum_scale=avg_aum)

    infl = env.influence_signal
    for agent in agents:
        agent.mark_to_market(env.prices)
        agent.update_memory(env.prices, infl.direction, infl.intensity)

print("Done.\n")

# ── Report ────────────────────────────────────────────────────────────────────
def pct(a, b):
    return f"{100*a/b:.1f}%" if b > 0 else "n/a"

def mean_std(vals):
    if not vals:
        return "n/a", "n/a"
    a = np.array(vals)
    return f"{a.mean():.4f}", f"{a.std():.4f}"

print("=" * 72)
print(f"{'Metric':<38} {'pre_shock':>10} {'shock':>10} {'post_shock':>10}")
print("=" * 72)

rows = [
    ("Signal strength (|S_biased|)",  "s_biased_abs",     "mean"),
    ("Signal strength p25",            "s_biased_abs",     "p25"),
    ("Signal strength p75",            "s_biased_abs",     "p75"),
    ("Gate evaluations (total)",       "gate_total",       "sum"),
    ("Gate fires",                     "gate_fired",       "sum"),
    ("Gate fire rate",                 "gate_fired",       "gate_rate"),
    ("Magnitude requested (total B)",  "mag_requested",    "sum"),
    ("Magnitude actual (total B)",     "mag_actual",       "sum"),
    ("Cash cap ratio (actual/req)",    "mag_actual",       "cash_ratio"),
    ("Buy fraction",                   "buys",             "buy_frac"),
    ("Mean agent cash weight",         "cash_weights",     "mean"),
    ("Mean influence intensity",       "influence_intensities", "mean"),
]

for label, key, stat in rows:
    vals = []
    for pk in ["pre_shock", "shock", "post_shock"]:
        agg = period_agg[pk]
        if stat == "mean":
            data = agg[key]
            v = f"{np.mean(data):.4f}" if data else "n/a"
        elif stat == "p25":
            data = agg[key]
            v = f"{np.percentile(data, 25):.4f}" if data else "n/a"
        elif stat == "p75":
            data = agg[key]
            v = f"{np.percentile(data, 75):.4f}" if data else "n/a"
        elif stat == "sum":
            v = f"{int(agg[key]):,}"
        elif stat == "gate_rate":
            v = pct(agg["gate_fired"], agg["gate_total"])
        elif stat == "cash_ratio":
            v = pct(agg["mag_actual"], agg["mag_requested"]) if agg["mag_requested"] > 0 else "n/a"
        elif stat == "buy_frac":
            total = agg["buys"] + agg["sells"]
            v = pct(agg["buys"], total)
        else:
            v = "?"
        vals.append(v)
    print(f"{label:<38} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

print()
print("=== ENERGY SECTOR SIGNAL DECOMPOSITION (mean across period) ===")
print("(F = fundamentalist, C = chartist, H = influence, using sector 0 = Energy)")
print()

# Re-run a lighter pass just for Energy signal decomposition
# Reset and re-run with tracking of F, C, H components for sector 0
rng2 = np.random.default_rng(numpy_seed)
agents2, _ = build_population(calibration, profiles, numpy_seed)
env2 = MarketEnvironment(lambdas=list(calibration.market.lambdas))
shock_proc2 = ShockProcessor(calibration.shocks, total_ticks)

from backend.simulation.signals import (
    compute_fundamentalist_signal, compute_chartist_signal, compute_influence_signal
)

energy_decomp: dict[str, list] = {pk: {"F": [], "C": [], "H": [], "S_raw": []} for pk in PERIODS}

for tick in range(total_ticks):
    env2.tick = tick
    env2.excess_demand[:] = 0.0
    env2.volume[:] = 0.0
    p0_eff2 = shock_proc2.get_p0_effective(tick)
    env2.update_p0_effective(p0_eff2)
    inf_dirs2, inf_ints2 = compute_background_influence(nm_intensity, rng2)
    shock_signal2 = shock_proc2.get_influence_signal(tick)
    if shock_signal2 is not None:
        covered2 = shock_signal2.intensity > 0
        combined_dirs2 = np.where(covered2, shock_signal2.direction, inf_dirs2)
        combined_ints2 = np.where(covered2, shock_signal2.intensity, inf_ints2)
        env2.influence_signal = InfluenceSignal(direction=combined_dirs2, intensity=combined_ints2)
    else:
        env2.influence_signal = InfluenceSignal(direction=inf_dirs2, intensity=inf_ints2)

    period_key2 = None
    for pk, (t0, t1) in PERIODS.items():
        if t0 <= tick <= t1:
            period_key2 = pk
            break

    agent_indices2 = rng2.permutation(len(agents2))
    for idx in agent_indices2:
        agent = agents2[idx]
        if rng2.random() >= agent.eval_probability:
            continue
        p = agent.profile
        F = compute_fundamentalist_signal(env2.prices[0], float(env2.p0_effective[0]))
        C = compute_chartist_signal(agent.price_memory[0], agent.memory_ticks,
                                    p.recency_bias, p.time_horizon,
                                    agent.is_institutional, agent.beta_c)
        H = compute_influence_signal(env2.influence_signal.direction[0],
                                     env2.influence_signal.intensity[0],
                                     p.herding_sensitivity)
        S_raw = agent.beta_f * F * p.information_quality + agent.beta_c * C + H

        if period_key2:
            energy_decomp[period_key2]["F"].append(F)
            energy_decomp[period_key2]["C"].append(C)
            energy_decomp[period_key2]["H"].append(H)
            energy_decomp[period_key2]["S_raw"].append(S_raw)

        # Still need to run activation to update state properly
        from backend.simulation.activation import run_agent_activation
        run_agent_activation(agent, env2, tick, rng2)

    avg_aum2 = sum(a.aum for a in agents2) / len(agents2)
    env2.apply_price_update(len(agents2), aum_scale=avg_aum2)
    infl2 = env2.influence_signal
    for agent in agents2:
        agent.mark_to_market(env2.prices)
        agent.update_memory(env2.prices, infl2.direction, infl2.intensity)

print(f"{'Component':<12} {'pre_shock':>12} {'shock':>12} {'post_shock':>12}")
print("-" * 52)
for comp in ["F", "C", "H", "S_raw"]:
    row = []
    for pk in ["pre_shock", "shock", "post_shock"]:
        vals3 = energy_decomp[pk][comp]
        if vals3:
            row.append(f"{np.mean(vals3):>+.4f} (sd={np.std(vals3):.4f})")
        else:
            row.append("n/a")
    print(f"{comp:<12} {row[0]:>28} {row[1]:>28} {row[2]:>28}")

print()
print("=== P0_EFFECTIVE FOR ENERGY (tick 38-50) ===")
sp = ShockProcessor(calibration.shocks, total_ticks)
for t in range(38, 51):
    p0e = sp.get_p0_effective(t)[0]
    active = sp.is_active(t)
    print(f"  tick {t}: P0_eff={p0e:.3f}  shock_active={active}")
