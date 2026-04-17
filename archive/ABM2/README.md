# LLM-Enhanced Financial Agent-Based Model

A novel agent-based modeling system where Large Language Models (LLMs) "program" heterogeneous agent behaviors, which then interact in a simulated financial market to produce emergent dynamics.

## Core Innovation

**Two-Layer Architecture:**
1. **LLM as Agent Programmer** - Generates cognitive frameworks, decision rules, and risk constraints for diverse investor personas
2. **ABM as Interaction Engine** - Agents trade using LLM-programmed behaviors, producing emergent market dynamics

**Key Features:**
- LLM-generated heterogeneous agent personas (12 archetype types)
- Realistic behavioral finance modeling with cognitive biases
- Market microstructure with order books and price discovery
- Event-driven dynamics (news, shocks, earnings)
- Periodic recalibration - agents learn and adapt based on experience
- Multi-asset market (Tech/Growth, Value/Financials, Safe Haven)

## Architecture

```
┌─────────────────┐
│  LLM Interface  │  ← Anthropic Claude / OpenAI GPT / Mock
└────────┬────────┘
         │ Calibration
         ↓
┌─────────────────┐
│     Agents      │  ← 100-200 heterogeneous traders
│  (12 types)     │     - Pension funds, hedge funds, retail
└────────┬────────┘     - Each with unique persona & rules
         │ Trading
         ↓
┌─────────────────┐
│  Market/Order   │  ← Order matching & price discovery
│     Books       │     - Linear price impact model
└────────┬────────┘     - Daily clearing
         │
         ↓
┌─────────────────┐
│  Data Collector │  ← Time-series metrics
│  & Validator    │     - Returns, volatility, correlations
└─────────────────┘     - Stylized facts validation
```

## Agent Types

The system supports 12 institutional and retail investor archetypes:

**Institutional:**
- Pension Fund (long horizon, conservative)
- Mutual Fund (benchmark-aware, moderate risk)
- Sovereign Wealth Fund (very long horizon, contrarian)
- Insurance (liability matching, stable income)

**Hedge Funds:**
- Value HF (mean reversion, fundamental)
- Momentum HF (trend following, technical)
- Macro HF (top-down, policy-driven)
- Activist (event-driven, concentrated)

**Retail:**
- Momentum Retail (FOMO-driven, social media)
- Value Retail (buy-and-hold, fundamental)

**Algorithmic:**
- Market Maker (liquidity provision, inventory management)
- HFT (microsecond, statistical arbitrage)

## Installation

### Requirements

```bash
python >= 3.8
numpy
pyyaml
anthropic  # For Claude API (optional)
openai     # For GPT API (optional)
tqdm       # For progress bars
```

### Setup

```bash
# Install dependencies
pip install numpy pyyaml anthropic openai tqdm

# Set API key (if using LLM calibration)
export ANTHROPIC_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Quick Start (Mock Mode - No API)

```python
from core.model import MarketModel

# Initialize model
model = MarketModel()

# Setup with mock agents (no LLM calls)
model.setup(calibrate_agents=False)

# Run simulation
model.run(steps=100)  # Run for 100 days

# Get results
summary = model.get_summary()
print(f"Final prices: {summary['market']['prices']}")
print(f"Average agent return: {summary['agents']['avg_return']:.2f}%")
```

### Full Mode (With LLM Calibration)

```python
from core.model import MarketModel

# Initialize model
model = MarketModel()

# Setup with LLM-calibrated agents
model.setup(calibrate_agents=True)  # Requires API key

# Run simulation
model.run(steps=504)  # Run full 2-year simulation

# Get detailed results
summary = model.get_summary()

# Analyze specific agent
agent_perf = model.get_agent_performance('agent_0')
print(f"Agent return: {agent_perf['total_return']:.2f}%")
print(f"Max drawdown: {agent_perf['max_drawdown']:.2f}%")
```

### Test System

```bash
# Run system test
python test_system.py

# Choose mode:
# 1 - Mock mode (fast, no API calls)
# 2 - LLM mode (requires API key)
```

## Configuration

All parameters are centralized in `config/parameters.yaml`:

### Key Settings

```yaml
simulation:
  total_days: 504  # 2 years of trading days
  random_seed: 42

agents:
  total_count: 150  # Number of agents
  total_capital: 10000000000  # $10B total
  distribution:  # Agent type counts
    pension_fund: 15
    mutual_fund: 20
    # ...

market:
  assets:  # Define asset characteristics
    - name: "Growth/Tech"
      ticker: "TECH"
      initial_price: 100.0
      volatility: 0.02  # 2% daily vol

llm:
  provider: "anthropic"  # or "openai" or "mock"
  model: "claude-sonnet-4.5"
  recalibration:
    enabled: true
    triggers:
      quarterly_reflection: 90  # Every 90 days
      major_move_threshold: 0.05  # 5% move
      drawdown_threshold: 0.20  # 20% drawdown
```

## Project Structure

```
ABM2/
├── config/
│   └── parameters.yaml       # All configuration
├── core/
│   ├── model.py              # Main simulation controller
│   ├── scheduler.py          # Agent activation management
│   └── datacollector.py      # Metrics and data collection
├── agents/
│   ├── base.py               # Base agent class
│   └── trader.py             # LLM-enhanced trader implementation
├── market/
│   ├── instruments.py        # Asset classes and universe
│   └── orderbook.py          # Order matching and clearing
├── llm/
│   ├── interface.py          # LLM provider abstraction
│   ├── prompts.py            # Calibration prompt templates
│   ├── parser.py             # Response parsing and validation
│   └── calibrator.py         # Orchestrates calibration
├── events/
│   └── news.py               # Event and news generation
├── test_system.py            # End-to-end test
├── SYSTEM_PLAN.md            # Detailed architecture document
└── README.md                 # This file
```

## How It Works

### Phase 1: Calibration

1. **LLM Generation** - For each agent archetype, the LLM generates:
   - Investment philosophy and beliefs
   - Risk tolerance and time horizon
   - Decision rules (entry/exit triggers)
   - Position sizing logic
   - Risk constraints (max drawdown, leverage)
   - Cognitive biases
   - Initial portfolio allocation

2. **Parsing & Validation** - Structured JSON responses are parsed and validated

3. **Agent Creation** - Agents are instantiated with LLM-generated personas

### Phase 2: Simulation

Each day:

1. **Events** - News generator creates market events
2. **Agent Decisions** - Each agent:
   - Observes market (prices, volumes, volatility)
   - Processes events through their cognitive lens
   - Applies biases (confirmation, anchoring, recency)
   - Calculates signals based on beliefs (momentum vs mean reversion)
   - Evaluates decision rules
   - Submits orders if criteria met
3. **Market Clearing** - Order book matches trades, discovers new prices
4. **Data Collection** - Metrics tracked for analysis
5. **Recalibration Check** - Trigger recalibration if conditions met

### Phase 3: Recalibration (Periodic)

When triggered:

1. **Experience Summary** - Collect agent performance and market evolution
2. **LLM Reflection** - LLM analyzes experience and suggests updates
3. **Persona Update** - Agent beliefs, rules, and constraints updated
4. **Continue Simulation** - Agents trade with updated frameworks

## Output & Analysis

### Collected Metrics

**Model-level:**
- Price time series for each asset
- Daily returns and volatility
- Trading volumes
- Cross-asset correlations

**Agent-level:**
- Portfolio values over time
- Realized returns
- Maximum drawdowns
- Position allocations
- Trade counts

### Stylized Facts Validation

The system calculates stylized facts to validate realism:
- Return distribution (fat tails, skewness, kurtosis)
- Volatility clustering (GARCH effects)
- Volume-volatility correlation
- Correlation dynamics

## Customization

### Add New Agent Type

1. Add to `config/parameters.yaml`:
```yaml
agents:
  distribution:
    my_new_type: 10
  capital_distribution:
    my_new_type: 5
```

2. Add archetype description in `llm/prompts.py`:
```python
DESCRIPTIONS = {
    "my_new_type": """Description of investor..."""
}
```

### Add New Asset

```yaml
market:
  assets:
    - name: "Commodities"
      ticker: "COMM"
      initial_price: 50.0
      volatility: 0.03
```

### Customize LLM Prompts

Edit prompt templates in `llm/prompts.py` to change how agents are calibrated.

## Roadmap

**Core System (Complete):**
- ✅ LLM calibration system
- ✅ Agent-based market simulation
- ✅ Order book and clearing
- ✅ Event generation
- ✅ Recalibration logic
- ✅ Data collection

**Next Steps:**
- Dashboard for calibration/running/visualization
- Historical scenario validation (2008 crisis, COVID, etc.)
- Calibration refinement based on testing
- Advanced analytics and visualization
- Parameter sensitivity analysis
- Multi-run ensemble analysis

## Research Context

This system explores the intersection of:
- **Behavioral Finance** - Heterogeneous beliefs, biases, decision-making
- **Agent-Based Modeling** - Bottom-up emergence from micro interactions
- **LLMs for Simulation** - Using language models to generate realistic agent behaviors
- **Market Microstructure** - Order flow, price discovery, liquidity
- **Emergent Learning** - Adaptive recalibration based on experience

## Citation

```
LLM-Enhanced Financial Agent-Based Model
Built with Claude Sonnet 4.5
February 2026
```

## License

MIT

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
