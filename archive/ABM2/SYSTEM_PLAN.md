# LLM-Enhanced Agent-Based Market Model - System Plan

## Overview
Building a novel agent-based financial market model where LLM-generated agent personas interact to produce emergent market dynamics. Goal: Create a research-grade, commercially viable tool for understanding sentiment-driven market behavior.

---

## Core Innovation
**Two-Layer Architecture with Dynamic Recalibration**

### Layer 1: LLM as Agent Programmer
- LLM defines heterogeneous agent cognitive frameworks
- Creates realistic behavioral diversity (not just parameter tweaks)
- Periodic recalibration based on market experience

### Layer 2: ABM as Interaction Engine
- Agents trade using LLM-programmed behaviors
- Market mechanics: order matching, price discovery
- Emergent patterns from heterogeneous interactions

---

## System Architecture

```
ABM2/
├── core/                    # Core ABM engine
│   ├── model.py            # Main simulation controller
│   ├── scheduler.py        # Agent activation logic
│   └── datacollector.py   # Metrics tracking & analysis
│
├── agents/                  # Agent system
│   ├── base.py             # Base trader agent class
│   ├── personas.py         # Agent archetype definitions
│   └── behaviors.py        # Decision-making framework
│
├── market/                  # Market infrastructure
│   ├── orderbook.py        # Order matching engine
│   ├── pricing.py          # Price discovery mechanism
│   └── instruments.py      # Asset definitions (multi-category)
│
├── llm/                     # LLM integration layer
│   ├── prompts.py          # Prompt templates (calibration, recalibration)
│   ├── interface.py        # LLM API wrapper (provider-agnostic)
│   └── parser.py           # Extract structured decisions from LLM
│
├── events/                  # Information & shocks
│   ├── news.py             # News generation
│   ├── shocks.py           # Market shocks & crises
│   └── calendar.py         # Scheduled events (earnings, macro data)
│
├── calibration/             # Research-backed parameters
│   ├── market_composition.py  # Agent mix from literature
│   ├── trading_rules.py    # Behavioral parameters (position sizing, risk)
│   └── stylized_facts.py   # Validation metrics (volatility clustering, etc.)
│
├── validation/              # Historical testing
│   ├── scenarios/          # Historical events (2008, COVID, GameStop, Flash)
│   ├── metrics.py          # Comparison tools
│   └── visualization.py    # Results analysis
│
├── config/                  # Configuration
│   └── parameters.yaml     # Centralized parameters
│
└── tests/                   # Unit tests
```

---

## Agent System Design

### Agent Archetypes (10-15 personas based on real financial ecosystem)

**Institutional Players (~60% of capital)**
1. **Pension Fund Manager** - Long-term, risk-averse, regulatory constraints
2. **Mutual Fund Manager** - Active but conservative, benchmark-aware
3. **Sovereign Wealth Fund** - Very long horizon, counter-cyclical
4. **Insurance Portfolio Manager** - Liability-matching, stable income focus

**Active Investors (~25% of capital)**
5. **Value Hedge Fund** - Mean reversion, fundamental analysis
6. **Momentum Hedge Fund** - Trend following, technical analysis
7. **Macro Hedge Fund** - Top-down, rates/currency aware
8. **Activist Investor** - Event-driven, catalyst focus

**Retail & Other (~15% of capital)**
9. **Retail Momentum Trader** - Social media influenced, FOMO-driven
10. **Retail Value Investor** - Buy-and-hold, fundamentals
11. **Algorithmic Market Maker** - Liquidity provision, inventory management
12. **High-Frequency Trader** - Statistical arbitrage, speed-focused

**Optional Additional Archetypes:**
13. **VC/Growth Investor** - Innovation focus, long duration
14. **Distressed Debt Specialist** - Crisis opportunist
15. **Index/Passive Fund** - Rebalancing only, price-insensitive

### Agent Behavior Framework

Each agent has:
- **Persona Profile** (LLM-generated)
  - Background/experience
  - Investment philosophy
  - Risk tolerance
  - Time horizon
  - Cognitive biases

- **Decision Rules** (LLM-generated → structured)
  - Market view interpretation
  - Position sizing logic
  - Entry/exit triggers
  - Risk constraints (VaR, drawdown limits, leverage)

- **State Variables**
  - Portfolio holdings (cash + positions across asset categories)
  - Performance history
  - Recent trades
  - Information exposure

- **Learning/Adaptation**
  - Performance tracking
  - Belief updating based on outcomes
  - Periodic recalibration via LLM

---

## Market Structure

### Asset Universe
**2-3 Asset Categories** (capture key dynamics without full equilibrium)

1. **Growth/Tech Sector**
   - High volatility, sentiment-driven
   - Long duration, growth-focused investors

2. **Value/Financials Sector**
   - Lower volatility, fundamentals-driven
   - Cyclical sensitivity

3. **Safe Haven (Bonds/Gold)**
   - Flight-to-quality dynamics
   - Negative correlation in stress

### Market Mechanics

**Order Book**
- Limit orders (price, quantity, agent ID)
- Market orders
- Order matching algorithm (price-time priority)
- Bid-ask spread tracking

**Price Discovery**
- Continuous double auction
- Price impact from large orders
- Liquidity dynamics (depth, spread)

**Information Flow**
- Public news (all agents receive)
- Delayed information (retail lags institutions)
- Information networks (influence cascade)

---

## LLM Integration Strategy

### Phase 1: Initial Calibration
**When:** Model initialization
**Process:**
1. For each agent archetype, LLM receives:
   - Persona description
   - Market structure info
   - Initial market conditions
2. LLM generates:
   - Investment philosophy & biases
   - Decision framework (views → actions)
   - Risk parameters
   - Initial portfolio allocation
3. Output parsed into structured agent behaviors

**Example Prompt:**
```
You are a Pension Fund Manager with 30 years experience managing $50B.
Profile: Risk-averse, regulatory constraints, 60/40 stock/bond mandate
Given current market conditions [context], define:
1. Your investment philosophy and key beliefs
2. How you interpret different market events (rate changes, volatility, etc.)
3. Position sizing rules (% of portfolio per trade)
4. Risk constraints (max drawdown, VaR limit)
5. Initial portfolio allocation across [Growth, Value, Safe Haven]

Provide structured output...
```

### Phase 2: ABM Execution
**When:** Daily time steps (~500 ticks for 2 years)
**Process:**
1. Market event occurs (news, shock, or normal day)
2. Agents interpret event using LLM-generated decision framework
3. Agents execute trades (buy/sell/hold)
4. Order book matches trades
5. Prices update based on supply/demand
6. Portfolio values update
7. Data collection (prices, volumes, agent states)

**No LLM calls during execution** - agents use pre-programmed logic

### Phase 3: Recalibration Events
**When:** Triggered by significant events
- Major market moves (>5% single-day move)
- Quarterly reflection (every 90 days)
- Significant agent portfolio drawdown (>20%)
- Novel market regime (detected via volatility/correlation shifts)

**Process:**
1. LLM receives updated context:
   - Agent's performance history
   - Market conditions evolution
   - Recent events experienced
   - Current portfolio state
2. LLM updates:
   - Refined beliefs based on experience
   - Adjusted risk parameters
   - Modified decision rules
3. Agent continues with updated framework

**Example Recalibration Prompt:**
```
You are the same Pension Fund Manager from before.
Since initial calibration:
- Your portfolio returned -15% (vs market -20%)
- You experienced a 25% drawdown during [crisis event]
- Interest rates rose 200bps
- Your current allocation: 45% Growth, 35% Value, 20% Safe Haven

Reflect on your experience and update:
1. Has your investment philosophy changed?
2. How do you now interpret [types of events]?
3. Should your risk parameters adjust?
4. What lessons have you learned?

Provide updated structured framework...
```

**Result:** Emergent learning on top of emergent market dynamics!

---

## Calibration Requirements (Research-Backed)

### 1. Realistic Parameters (Views → Behaviors)
**Sources:** Academic literature, regulatory filings, practitioner surveys

- **Position Sizing:**
  - Kelly criterion for optimal sizing
  - Practical constraints (5-10% per position for active, 1-2% for conservative)
  - Leverage limits (1x for long-only, up to 3x for hedge funds)

- **Confidence → Action Mapping:**
  - High confidence (LLM: "strongly bullish") → 8-10% position
  - Moderate confidence → 3-5% position
  - Low confidence → 1-2% position or no action

- **Risk Constraints:**
  - VaR limits (1-day 95% VaR: 1-3% for institutional, 5-10% for hedge funds)
  - Drawdown limits (10-15% for institutional, 25-30% for hedge funds)
  - Sector concentration limits

### 2. Realistic Interactions (Market Microstructure)
**Sources:** Market data, microstructure literature

- **Order Flow:**
  - Institutional: large orders (0.1-1% of daily volume)
  - Retail: small orders (<0.01% of daily volume)
  - HFT: very small, very frequent

- **Price Impact:**
  - Square-root law: impact ∝ √(order size / daily volume)
  - Temporary vs permanent impact
  - Bid-ask bounce

- **Information Networks:**
  - Analyst reports → Institutions → Retail (cascade with lag)
  - Institutional investors cluster (herding)
  - Social media → Retail coordination

### 3. Realistic Mix (Agent Population)
**Sources:** Market composition data (SEC, academic studies)

**Target Distribution:**
- Institutional (passive): 40-50% of capital
- Institutional (active): 20-25% of capital
- Hedge funds: 10-15% of capital
- Retail: 10-15% of capital
- Market makers/HFT: 5-10% of capital

**Number of Agents:**
- Total: 100-200 agents (computational constraint)
- Each agent represents aggregate of similar market participants
- Weighted by capital allocation

---

## Validation Framework

### Stylized Facts to Reproduce
Model should exhibit realistic market patterns:

1. **Returns Distribution**
   - Fat tails (excess kurtosis)
   - Negative skewness (crash risk)
   - ~0 autocorrelation in returns

2. **Volatility Dynamics**
   - Volatility clustering (GARCH effects)
   - Volatility asymmetry (leverage effect)
   - Mean-reverting volatility

3. **Volume Patterns**
   - Volume-volatility correlation
   - Volume clustering

4. **Cross-Asset**
   - Flight-to-quality (negative correlation in stress)
   - Sector rotation

5. **Behavioral Patterns**
   - Momentum (short-term continuation)
   - Reversal (long-term)
   - Herding (correlation spike in stress)

### Historical Scenarios for Backtesting
**After model is functional, test on:**

1. **2008 Financial Crisis**
   - Lehman collapse → liquidity freeze → flight to quality
   - Validate: contagion, correlation breakdown, volatility spike

2. **COVID Crash (March 2020)**
   - Sudden uncertainty → panic selling → V-shaped recovery
   - Validate: speed of crash, policy response impact

3. **GameStop/Meme Stocks (Jan 2021)**
   - Retail coordination → short squeeze → extreme volatility
   - Validate: social dynamics, liquidity crisis

4. **Flash Crash (May 2010)**
   - Algorithmic cascade → liquidity withdrawal → rapid recovery
   - Validate: market microstructure breakdown

---

## Technical Specifications

### Time Scale
- **Granularity:** Daily ticks
- **Simulation Duration:** 2 years (~500 trading days)
- **Fast-forward mode:** Support longer backtests (5-10 years)

### LLM Integration
- **Provider:** Flexible (Claude, GPT-4, local models)
- **API Strategy:**
  - Batch calibration (all agents at once)
  - Async recalibration (parallel queries)
  - Caching common responses

### Performance Targets
- **Agents:** 100-200
- **Assets:** 2-3 categories
- **Simulation Speed:** <1 min for 2-year run (excluding LLM calls)
- **Memory:** <4GB RAM

### Data Collection
**Track at each time step:**
- Price, volume, spread for each asset
- Agent portfolios, cash, PnL
- Order flow (buy/sell imbalance)
- Sentiment measures (aggregate views)
- Volatility, correlation

**Aggregate Metrics:**
- Market returns distribution
- Sharpe ratios by agent type
- Turnover, liquidity measures
- Herding indices

---

## Development Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [x] Planning document
- [ ] Basic ABM framework (model, scheduler, data collector)
- [ ] Simple agent base class
- [ ] Basic order book & price discovery
- [ ] Simple asset definition

### Phase 2: LLM Integration (Week 2)
- [ ] LLM interface (provider-agnostic)
- [ ] Prompt templates (calibration, recalibration)
- [ ] Response parser (structured output extraction)
- [ ] Agent persona system

### Phase 3: Market Mechanics (Week 2-3)
- [ ] Multi-asset implementation
- [ ] Order flow & price impact
- [ ] Information/news system
- [ ] Event triggers (shocks, calendar)

### Phase 4: Calibration (Week 3-4)
- [ ] Literature review → parameter database
- [ ] Agent composition calibration
- [ ] Trading rule calibration
- [ ] Risk parameter calibration

### Phase 5: Validation (Week 4-5)
- [ ] Stylized facts testing
- [ ] Historical scenario implementation
- [ ] Metrics comparison
- [ ] Visualization tools

### Phase 6: Refinement (Week 5-6)
- [ ] Performance optimization
- [ ] Documentation
- [ ] Demo scenarios
- [ ] Publication draft

---

## Open Questions / Future Extensions

1. **Agent Learning Mechanism**
   - Reinforcement learning overlay?
   - Bayesian belief updating?
   - Pure LLM-driven adaptation?

2. **Market Complexity**
   - Add options/derivatives?
   - Short-selling mechanics?
   - Margin calls / liquidations?

3. **Information Structure**
   - Insider trading scenarios?
   - Information quality gradients?
   - Fake news / manipulation?

4. **Validation Depth**
   - Formal econometric tests?
   - Comparison to other models (DSGE, etc.)?
   - Sensitivity analysis?

5. **Commercial Applications**
   - Hedge fund strategy testing?
   - Risk management / stress testing?
   - Regulatory scenario analysis?
   - Market manipulation detection?

---

## Success Criteria

**Minimum Viable Model:**
- ✅ 10+ distinct agent archetypes with LLM-generated behaviors
- ✅ 2-3 asset categories with realistic price discovery
- ✅ Reproduce 3+ stylized facts (fat tails, volatility clustering, etc.)
- ✅ Successfully backtest on 2+ historical scenarios
- ✅ Demonstrate emergent learning via recalibration

**Showcase Quality:**
- ✅ Clean, modular, well-documented code
- ✅ Impressive visualization of agent heterogeneity
- ✅ Clear narrative: "LLM-enhanced ABMs capture realistic market dynamics"
- ✅ Research note quality analysis

**Commercial Potential:**
- ✅ Obvious applications (risk management, strategy testing)
- ✅ Performance benchmarks (faster/more realistic than alternatives?)
- ✅ Extensibility demonstrated (easy to add new scenarios/agents)

---

## References & Literature (To Be Populated)

**Agent-Based Financial Modeling:**
- LeBaron (2006) - Agent-based computational finance
- Farmer & Foley (2009) - The economy needs agent-based modeling
- Hommes & LeBaron (2018) - Handbook of computational economics

**Market Microstructure:**
- Bouchaud et al. (2009) - Price impact
- Cont (2001) - Empirical properties of asset returns (stylized facts)

**Behavioral Finance:**
- Shleifer & Summers (1990) - Noise trader approach
- Barberis & Thaler (2003) - Survey of behavioral finance

**Market Composition:**
- To be researched: institutional ownership data, market participant surveys

---

*Document created: 2026-02-24*
*Last updated: 2026-02-24*
