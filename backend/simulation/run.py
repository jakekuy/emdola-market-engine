"""
SimulationRun — executes a single simulation run from tick 0 to total_ticks.

Receives pre-built agents (from population.py) and a CalibrationInput.
Returns a RunResult containing tick snapshots and type action summaries.

Spec references: §6 (two-stage activation), §7 (agent behaviour), §11 (output).

Tick loop sequence (per tick):
  1.  Inject market-channel shock excess demand.
  2.  Set influence channel broadcast (shock or ambient background).
  3.  Randomly shuffle agent evaluation order.
  4.  For each agent: Layer 1 gate → if passes, run 5-step activation.
  5.  Apply log-linear price formation.
  6.  Mark all agents to market; update all agent memories.
  7.  Log tick snapshot and type action summary (if tick should be logged).
  8.  Emit WebSocket tick event via tick_callback (if provided).
"""

from __future__ import annotations

import threading
import numpy as np
from collections import defaultdict
from typing import Callable

from backend.config.sector_config import NUM_SECTORS, SECTORS, AGENT_TYPES
from backend.models.calibration import CalibrationInput
from backend.models.results import RunResult, SeedRecord, TickSnapshot, TypeActionSummary, AgentTraceRecord
from backend.simulation.agent import Agent
from backend.simulation.environment import MarketEnvironment, InfluenceSignal
from backend.simulation.shocks import ShockProcessor
from backend.simulation.activation import run_agent_activation
from backend.simulation.signals import compute_background_influence
from backend.config.constants import CROSS_SECTOR_COUPLING, TRACE_REDUNDANCY_DECAY, TRACE_MAX_COUNT
from backend.simulation.population import agents_by_type


def _should_log(tick: int, total_ticks: int, granularity: str) -> bool:
    """
    Return True if this tick should be logged based on granularity setting.

    - full_fidelity: every tick
    - every_5:       every 5 ticks (uniform)
    - end_state:     final tick only
    """
    if granularity == "full_fidelity":
        return True
    if granularity == "end_state":
        return tick == total_ticks - 1
    return tick % 5 == 0 or tick == total_ticks - 1


class SimulationRun:
    """
    Executes one simulation run.

    Parameters
    ----------
    calibration :
        Full setup configuration.
    agents :
        Pre-built agent population for this run (from population.py).
    """

    def __init__(
        self,
        calibration: CalibrationInput,
        agents: list[Agent],
    ) -> None:
        self.calibration = calibration
        self.agents = agents
        self.total_ticks: int = calibration.run.total_ticks
        self.total_agent_count: int = len(agents)
        # Average initial AUM per agent — used to normalise excess demand in the
        # price formula so that λ values are dimensionless (independent of AUM scale).
        self._avg_aum: float = (
            sum(a.aum for a in agents) / len(agents) if agents else 1.0
        )
        self._shock_proc = ShockProcessor(
            calibration.shocks,
            self.total_ticks,
            narrative_half_life=calibration.market.narrative_half_life,
        )
        self._non_market_intensity: float = calibration.market.non_market_signal_intensity
        self._granularity: str = calibration.run.data_granularity
        # Collect shock onset ticks for near-shock logging.

    def run(
        self,
        numpy_seed: int,
        persona_seed: str,
        run_number: int,
        tick_callback: Callable[[dict], None] | None = None,
        pause_event: threading.Event | None = None,
    ) -> RunResult:
        """
        Execute the full tick loop and return the run result.

        Parameters
        ----------
        numpy_seed :
            Seed for the NumPy random generator — controls all stochastic
            elements: agent shuffle order, evaluation probability draws,
            signal noise, gate draws.
        persona_seed :
            8-digit persona selection string — stored in SeedRecord for replay.
        run_number :
            Identifies this run within the batch.
        tick_callback :
            Optional callable invoked after each tick with a dict of tick data.
            Used by the batch engine to stream WebSocket events.
        """
        rng = np.random.default_rng(numpy_seed)
        env = MarketEnvironment(lambdas=list(self.calibration.market.lambdas))

        by_type: dict[str, list[Agent]] = agents_by_type(self.agents)

        tick_snapshots: list[TickSnapshot] = []
        type_summaries: list[TypeActionSummary] = []
        # Raw trade log for traced agents — filtered post-loop.
        _raw_traces: dict[int, list[dict]] = {}  # agent_id → list of trade dicts

        # ── Tick loop ─────────────────────────────────────────────────────────
        for tick in range(self.total_ticks):
            env.tick = tick
            env.excess_demand[:] = 0.0
            env.volume[:] = 0.0

            # Step 1: Update effective fundamental price from shock processor.
            # Market-channel shocks shift P0_effective per sector (what agents
            # believe the fair value is), rather than forcing prices directly.
            # Prices then move EMERGENTLY as agents trade toward the new belief.
            p0_eff = self._shock_proc.get_p0_effective(tick)
            env.update_p0_effective(p0_eff)

            # Step 2: Influence channel broadcast.
            # Background noise is always generated; shock signal overlays it only
            # for sectors the shock explicitly covers.  A partial-sector shock
            # (e.g. Energy/Materials only) must not silence background noise for
            # unaffected sectors — those sectors would otherwise receive zero
            # influence intensity instead of the ambient market noise that was
            # present before the shock, suppressing all signal-driven activity.
            inf_dirs, inf_ints = compute_background_influence(
                self._non_market_intensity, rng
            )
            shock_signal = self._shock_proc.get_influence_signal(tick)
            if shock_signal is not None:
                # For covered sectors: blend shock directional pressure with background
                # noise rather than replacing background entirely.  A pure deterministic
                # shock direction locks all herding-sensitive agents into the same trade
                # every tick → no two-sided flow → smooth price path with no volatility.
                # Blending adds the shock's signed intensity to the background signed
                # intensity; the result is a direction that is BIASED toward the shock
                # but stochastic each tick, producing realistic volatile reversion.
                # For uncovered sectors: background unchanged.
                covered = shock_signal.intensity > 0
                # Convert background to signed form: direction × intensity.
                bg_signed   = inf_dirs * inf_ints
                # Shock directional contribution.
                shk_signed  = shock_signal.direction * shock_signal.intensity
                # Blend: shock biases direction, background adds per-tick noise.
                blend       = np.where(covered, shk_signed + bg_signed, bg_signed)
                combined_dirs = np.where(blend != 0, np.sign(blend), inf_dirs)
                combined_ints = np.clip(np.abs(blend), 0.0, 1.0)
                env.influence_signal = InfluenceSignal(
                    direction=combined_dirs, intensity=combined_ints
                )
            else:
                env.influence_signal = InfluenceSignal(
                    direction=inf_dirs, intensity=inf_ints
                )

            # Cross-sector coupling: derive a market-wide sentiment component from
            # the mean signed influence across all sectors and add it uniformly.
            # Mechanism: large macro shocks create a "risk appetite" dimension —
            # institutional agents reduce broad equity exposure, not just exposure
            # to directly shocked sectors.  This produces realistic cross-sector
            # co-movement without hardcoding correlations.
            # CROSS_SECTOR_COUPLING=0.0 recovers fully independent-sector behaviour.
            if CROSS_SECTOR_COUPLING > 0.0:
                infl = env.influence_signal
                signed = infl.direction * infl.intensity
                mkt_component = CROSS_SECTOR_COUPLING * float(np.mean(signed))
                if mkt_component != 0.0:
                    blended = signed + mkt_component
                    env.influence_signal = InfluenceSignal(
                        direction=np.where(blended != 0.0, np.sign(blended), infl.direction),
                        intensity=np.clip(np.abs(blended), 0.0, 1.0),
                    )

            # Compute shock_active once per tick — used for logging and trace capture.
            shock_active = self._shock_proc.is_active(tick)
            influence_active = self._shock_proc.is_influence_active(tick)

            # Step 3: Per-type tracking accumulators (reset each tick).
            activated: dict[str, int] = {t: 0 for t in AGENT_TYPES}
            agents_with_trades: dict[str, int] = {t: 0 for t in AGENT_TYPES}
            buy_counts: dict[str, int] = {t: 0 for t in AGENT_TYPES}
            sell_counts: dict[str, int] = {t: 0 for t in AGENT_TYPES}
            net_mag: dict[str, np.ndarray] = {
                t: np.zeros(NUM_SECTORS) for t in AGENT_TYPES
            }

            # Step 4: Random sequential activation.
            agent_indices = rng.permutation(self.total_agent_count)
            for idx in agent_indices:
                agent = self.agents[idx]
                atype = agent.agent_type

                # Layer 1: evaluation probability gate (spec §6.3).
                if rng.random() >= agent.eval_probability:
                    continue
                activated[atype] += 1

                # Steps 1–5: composite activation (signals → biases → demand → gate → trade).
                trades = run_agent_activation(agent, env, tick, rng)

                if trades:
                    agents_with_trades[atype] += 1
                    for sector, direction, magnitude in trades:
                        if direction > 0:
                            buy_counts[atype] += 1
                        else:
                            sell_counts[atype] += 1
                        net_mag[atype][sector] += direction * magnitude

                    # Capture trades during shock window or reversals for all agents.
                    # Filtered globally post-loop into a quality-ranked flat list.
                    if shock_active:
                        bucket = _raw_traces.setdefault(agent.agent_id, [])
                        for sector, direction, magnitude in trades:
                            bucket.append({
                                "agent_id": agent.agent_id,
                                "agent_type": agent.agent_type,
                                "archetype_num": agent.archetype_num,
                                "tick": tick,
                                "sector_index": sector,
                                "direction": direction,
                                "magnitude": magnitude,
                                "shock_active": shock_active,
                            })

            # Step 5: Price formation (log-linear excess demand mechanism).
            # aum_scale normalises monetary ED (B USD) to a dimensionless fraction.
            env.apply_price_update(self.total_agent_count, aum_scale=self._avg_aum)

            # Step 6: Mark all agents to market; update all agent memories.
            infl = env.influence_signal
            for agent in self.agents:
                agent.mark_to_market(env.prices)
                agent.update_memory(env.prices, infl.direction, infl.intensity)

            # Step 7: Log this tick if required by granularity setting.
            if _should_log(tick, self.total_ticks, self._granularity):
                tick_snapshots.append(TickSnapshot(
                    tick=tick,
                    prices=env.prices.tolist(),
                    volume=env.volume.tolist(),
                    volatility=env.volatility.tolist(),
                    excess_demand=env.excess_demand.tolist(),
                    shock_active=shock_active,
                    influence_active=influence_active,
                ))
                for atype in AGENT_TYPES:
                    n_active = activated[atype]
                    n_agents = len(by_type[atype])
                    hold = max(0, n_active - agents_with_trades[atype])
                    type_summaries.append(TypeActionSummary(
                        agent_type=atype,
                        tick=tick,
                        activity_ratio=n_active / max(n_agents, 1),
                        net_direction=net_mag[atype].tolist(),
                        buy_count=buy_counts[atype],
                        sell_count=sell_counts[atype],
                        hold_count=hold,
                    ))

            # Step 8: Emit WebSocket tick event.
            if tick_callback is not None:
                tick_callback({
                    "type": "tick",
                    "run": run_number,
                    "tick": tick,
                    "prices": {
                        s: float(p) for s, p in zip(SECTORS, env.prices)
                    },
                    "shock_active": shock_active,
                    "type_activity": {
                        atype: {
                            "activity_ratio": activated[atype] / max(
                                len(by_type[atype]), 1
                            ),
                            "buy": buy_counts[atype],
                            "sell": sell_counts[atype],
                            # Index of sector with highest |net_magnitude| this tick.
                            # -1 if the type made no trades.
                            "top_sector": int(np.argmax(np.abs(net_mag[atype])))
                                if agents_with_trades[atype] > 0 else -1,
                        }
                        for atype in AGENT_TYPES
                    },
                })

            # Pause support: only block if actually paused (avoids overhead per tick).
            if pause_event is not None and not pause_event.is_set():
                pause_event.wait()

        # ── Post-loop: score all captured trades and keep the most interesting ──
        # All trades were captured during shock-active ticks.  Now compute
        # per-trade quality tags and assign a priority score:
        #   COUPLING  (shock tick + unshocked sector)  → 4 pts
        #   DISSENT   (trades against own type's net direction this tick) → 3 pts
        #   REVERSAL  (direction flip vs previous trade in same sector)   → 2 pts
        #   SHOCK     (shock tick, directly shocked sector)               → 1 pt
        # Global top-30 by score then magnitude — no per-agent cap.
        # Compute shocked sector indices from calibration shocks.
        shocked_sector_indices: set[int] = set()
        for shock in self.calibration.shocks:
            for s in shock.affected_sectors:
                shocked_sector_indices.add(s)

        # Build type-net-direction map: (agent_type, tick) → list of net_direction
        type_net: dict[tuple[str, int], np.ndarray] = {}
        for ts in type_summaries:
            type_net[(ts.agent_type, ts.tick)] = np.array(ts.net_direction)

        all_candidates: list[dict] = []
        for agent_id, trades_list in _raw_traces.items():
            if not trades_list:
                continue
            prev_dir: dict[int, int] = {}
            for t in trades_list:
                score = 0
                is_coupling = t["shock_active"] and t["sector_index"] not in shocked_sector_indices
                is_reversal = (
                    t["sector_index"] in prev_dir
                    and prev_dir[t["sector_index"]] != t["direction"]
                )
                prev_dir[t["sector_index"]] = t["direction"]
                net = type_net.get((t["agent_type"], t["tick"]))
                is_dissent = (
                    net is not None
                    and len(net) > t["sector_index"]
                    and net[t["sector_index"]] != 0
                    and int(np.sign(net[t["sector_index"]])) != t["direction"]
                )
                if is_coupling:
                    score += 4
                if is_dissent:
                    score += 3
                if is_reversal:
                    score += 2
                if t["shock_active"] and not is_coupling:
                    score += 1
                if score > 0:
                    all_candidates.append({
                        **t,
                        "_score": score,
                        "_is_coupling": is_coupling,
                        "_is_dissent": is_dissent,
                        "_is_reversal": is_reversal,
                    })

        # Greedy selection with diminishing-returns penalty.
        # Each additional trade sharing the same primary tag costs TRACE_REDUNDANCY_DECAY
        # off its effective score.  No hard per-category caps — if a scenario genuinely
        # has only one type of interesting signal, it still dominates; the penalty just
        # prevents 20 near-identical trades crowding out the few distinct ones.
        def _primary_tag(t: dict) -> str:
            if t["_is_coupling"] and t["_is_dissent"]:
                return "coupling_dissent"
            if t["_is_coupling"]:
                return "coupling"
            if t["_is_dissent"]:
                return "dissent"
            if t["_is_reversal"]:
                return "reversal"
            return "shock"

        tag_counts: dict[str, int] = defaultdict(int)
        remaining = sorted(all_candidates, key=lambda x: (x["_score"], x["magnitude"]), reverse=True)
        top: list[dict] = []
        while remaining and len(top) < TRACE_MAX_COUNT:
            # Effective score = raw score minus accumulated redundancy penalty.
            best = max(remaining, key=lambda t: t["_score"] - TRACE_REDUNDANCY_DECAY * tag_counts[_primary_tag(t)])
            eff_score = best["_score"] - TRACE_REDUNDANCY_DECAY * tag_counts[_primary_tag(best)]
            if eff_score <= 0:
                break
            top.append(best)
            tag_counts[_primary_tag(best)] += 1
            remaining.remove(best)

        top.sort(key=lambda x: x["tick"])
        _drop = {"_score", "_is_coupling", "_is_dissent", "_is_reversal"}
        agent_traces: list[AgentTraceRecord] = [
            AgentTraceRecord(**{k: v for k, v in t.items() if k not in _drop})
            for t in top
        ]

        # ── Build RunResult ───────────────────────────────────────────────────
        from backend.simulation.price import compute_mispricing
        final_prices = env.prices.tolist()
        final_mispricing = compute_mispricing(env.prices).tolist()

        return RunResult(
            run_number=run_number,
            seed_record=SeedRecord(
                run_number=run_number,
                persona_seed=persona_seed,
                numpy_seed=numpy_seed,
            ),
            tick_snapshots=tick_snapshots,
            type_summaries=type_summaries,
            final_prices=final_prices,
            final_mispricing=final_mispricing,
            agent_traces=agent_traces,
        )
