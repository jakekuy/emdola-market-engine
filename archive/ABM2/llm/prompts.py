"""
LLM Prompt Templates
Templates for agent calibration and recalibration
"""

from typing import Dict, Any


class PromptTemplates:
    """
    Prompt templates for LLM-based agent programming

    Two main types:
    1. Calibration: Initial agent persona generation
    2. Recalibration: Update agent based on experience
    """

    @staticmethod
    def get_calibration_system_prompt() -> str:
        """
        System prompt for agent calibration

        Returns:
            System prompt text
        """
        return """You are an expert in behavioral finance and institutional investing.

Your task is to create realistic investor personas that will participate in a financial market simulation. Each persona should have:
- A coherent investment philosophy based on their role and experience
- Realistic decision-making rules that translate beliefs into actions
- Appropriate risk constraints and position sizing logic
- Initial portfolio allocation across asset categories

Be specific and realistic. These personas should behave like real institutional or retail investors, with all their biases, constraints, and decision frameworks.

Provide responses in valid JSON format only."""

    @staticmethod
    def get_calibration_prompt(
        agent_type: str,
        capital: float,
        market_info: Dict[str, Any],
        archetype_description: str
    ) -> str:
        """
        Generate calibration prompt for specific agent

        Args:
            agent_type: Agent archetype (e.g., 'pension_fund')
            capital: Initial capital allocation
            market_info: Current market conditions
            archetype_description: Description of this archetype

        Returns:
            Calibration prompt
        """
        assets = market_info.get('assets', [])
        asset_list = ', '.join([f"{a['ticker']} ({a['name']})" for a in assets])

        prompt = f"""Create a realistic investor persona for: {agent_type.upper().replace('_', ' ')}

**Profile:**
{archetype_description}

**Capital:** ${capital:,.0f}

**Market Context:**
Available assets: {asset_list}

Current market conditions:
- Interest rate environment: {market_info.get('interest_rate', 'neutral')}
- Volatility regime: {market_info.get('volatility_regime', 'normal')}
- Economic cycle: {market_info.get('economic_cycle', 'expansion')}

**Your Task:**
Define this investor's complete decision-making framework as a JSON object with:

1. **investment_philosophy**: Core beliefs about markets (2-3 sentences)

2. **risk_tolerance**: low/moderate/high

3. **time_horizon**: short/medium/long (with specific years)

4. **decision_rules**: How views translate to actions
   - entry_trigger: When to buy (specific criteria)
   - exit_trigger: When to sell (specific criteria)
   - position_sizing: Formula or rule (e.g., "2% of portfolio", "Kelly 50%")
   - max_position: Maximum % per position
   - rebalancing_trigger: When to rebalance

5. **risk_constraints**:
   - max_drawdown_tolerance: % before reducing risk
   - max_leverage: Maximum leverage allowed
   - var_limit: Value at Risk limit (optional)

6. **beliefs**: Confidence levels (high/medium/low) about:
   - market_efficiency
   - mean_reversion
   - momentum_effects
   - correlation_stability
   - volatility_clustering

7. **cognitive_biases**: List 2-3 realistic biases this type exhibits

8. **information_processing**:
   - reaction_speed: fast/moderate/slow
   - contrarian_tendency: 0-100 (0=pure herding, 100=pure contrarian)
   - fundamental_vs_technical: 0-100 (0=pure technical, 100=pure fundamental)

9. **initial_allocation**: Portfolio weights for each asset
   - TECH: 0.0-1.0
   - VALUE: 0.0-1.0
   - SAFE: 0.0-1.0
   (Must sum to 1.0)

10. **market_views**: Current views on each asset
    - TECH: bullish/neutral/bearish (with rationale)
    - VALUE: bullish/neutral/bearish (with rationale)
    - SAFE: bullish/neutral/bearish (with rationale)

**Important:**
- Be realistic for this investor type
- Decision rules must be specific and actionable
- Risk constraints should match their role
- Biases should be realistic for this archetype

Return ONLY valid JSON. No additional text."""

        return prompt

    @staticmethod
    def get_recalibration_prompt(
        agent_type: str,
        original_persona: Dict,
        performance_history: Dict,
        market_history: Dict,
        recent_events: list
    ) -> str:
        """
        Generate recalibration prompt based on agent experience

        Args:
            agent_type: Agent archetype
            original_persona: Original LLM-generated persona
            performance_history: Agent's performance metrics
            market_history: Market evolution
            recent_events: Recent market events

        Returns:
            Recalibration prompt
        """
        prompt = f"""You previously created an investor persona for a {agent_type.upper().replace('_', ' ')}.

**Original Investment Philosophy:**
{original_persona.get('investment_philosophy', 'N/A')}

**Performance Since Calibration:**
- Days active: {performance_history.get('days_active', 0)}
- Total return: {performance_history.get('total_return', 0):.2f}%
- Maximum drawdown: {performance_history.get('max_drawdown', 0):.2f}%
- Current portfolio value: ${performance_history.get('current_value', 0):,.0f}
- Sharpe ratio: {performance_history.get('sharpe_ratio', 'N/A')}

**Current Portfolio Allocation:**
{self._format_allocation(performance_history.get('current_allocation', {}))}

**Market Evolution:**
- Overall market return: {market_history.get('market_return', 0):.2f}%
- Volatility change: {market_history.get('volatility_change', 'stable')}
- Correlation regime: {market_history.get('correlation_regime', 'normal')}

**Recent Significant Events:**
{self._format_events(recent_events)}

**Reflection Task:**
Given this experience, how should this investor's framework evolve?

Consider:
1. Did their original beliefs hold true?
2. Were their risk constraints appropriate?
3. What lessons should they learn from their performance?
4. Should their decision rules adapt?
5. How should recent events affect their market views?

**Provide updated JSON with:**

1. **reflection**: 2-3 sentences on key learnings

2. **updated_beliefs**: Any belief changes (only include if changed)
   - Example: {{"market_efficiency": "medium (was high)"}}

3. **updated_risk_constraints**: Any risk parameter changes

4. **updated_decision_rules**: Any rule modifications

5. **updated_market_views**: Current views on each asset with updated rationale

6. **confidence_adjustment**: Overall confidence (higher/same/lower) with explanation

7. **behavioral_changes**: How biases may have intensified or diminished

8. **recommended_allocation**: Suggested new portfolio weights

**Important:**
- Only include parameters that should change
- Changes should be realistic given experience
- Investors learn but also exhibit path dependence
- Performance affects confidence but not always rationally

Return ONLY valid JSON. No additional text."""

        return prompt

    @staticmethod
    def _format_allocation(allocation: Dict[str, float]) -> str:
        """Format allocation dict for display"""
        if not allocation:
            return "N/A"
        return "\\n".join([f"- {k}: {v*100:.1f}%" for k, v in allocation.items()])

    @staticmethod
    def _format_events(events: list) -> str:
        """Format events list for display"""
        if not events:
            return "None"
        return "\\n".join([f"- Day {e.get('day', '?')}: {e.get('description', 'Event')}" for e in events[-5:]])  # Last 5 events


class AgentArchetypes:
    """
    Descriptions of each agent archetype for calibration
    """

    DESCRIPTIONS = {
        "pension_fund": """Large institutional investor managing retirement assets.
        Characteristics: Very long time horizon (30+ years), high risk aversion, strict regulatory constraints,
        focus on stable income and capital preservation. Typically 60/40 equity/fixed income mandate.
        Experienced professionals, process-driven, slow to react.""",

        "mutual_fund": """Active mutual fund manager benchmarked against index.
        Characteristics: Medium time horizon (3-5 years), moderate risk tolerance, career risk concerns,
        cannot deviate too far from benchmark. Fundamental analysis focus. Faces quarterly pressure.""",

        "sovereign_wealth": """Sovereign wealth fund with multi-generational horizon.
        Characteristics: Very long horizon, can be counter-cyclical, massive capital base,
        lower liquidity needs. Can take contrarian positions during crises.""",

        "insurance": """Insurance company investment portfolio.
        Characteristics: Liability matching focus, need for stable income, regulatory capital requirements,
        conservative risk profile. Asset-liability management driven.""",

        "value_hedge_fund": """Value-focused hedge fund employing fundamental analysis.
        Characteristics: Medium horizon (2-3 years), mean-reversion belief, contrarian tendency,
        moderate leverage (1.5-2x), concentrated positions. Seeks mispriced assets.""",

        "momentum_hedge_fund": """Momentum/trend-following hedge fund.
        Characteristics: Short-medium horizon, technical analysis, trend following,
        quick to exit losing positions, higher turnover. Believes in momentum persistence.""",

        "macro_hedge_fund": """Global macro hedge fund making top-down bets.
        Characteristics: Variable horizon, rates/FX/policy aware, high conviction when thesis develops,
        uses leverage (2-3x), focuses on major regime shifts and macro catalysts.""",

        "activist": """Activist investor targeting specific situations.
        Characteristics: Event-driven, catalyst focus, concentrated positions,
        seeks to influence outcomes. Patient when thesis intact, ruthless when broken.""",

        "retail_momentum": """Retail investor following social media and momentum.
        Characteristics: Short horizon, FOMO-driven, follows trends and influencers,
        limited diversification, higher risk tolerance, behavioral biases prominent.""",

        "retail_value": """Retail buy-and-hold value investor.
        Characteristics: Long horizon, fundamental focus, reads annual reports,
        lower turnover, tax-aware, may have behavioral biases (anchoring, confirmation).""",

        "market_maker": """Algorithmic market maker providing liquidity.
        Characteristics: Very short horizon (seconds to minutes), inventory management focus,
        bid-ask spread capture, delta-neutral target. Not directional, pure liquidity provision.""",

        "hft": """High-frequency trading algorithm.
        Characteristics: Microsecond horizon, statistical arbitrage, mean reversion at micro scale,
        massive turnover, tiny profit per trade. Risk-averse, exits immediately when patterns break."""
    }

    @staticmethod
    def get_description(agent_type: str) -> str:
        """Get archetype description"""
        return AgentArchetypes.DESCRIPTIONS.get(agent_type, "Generic investor")
