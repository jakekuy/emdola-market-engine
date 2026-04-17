# Quick Start Guide

## System Status

✅ **Core system build complete and tested**
✅ **Dashboard operational**

## Getting Started

### 1. Install Dependencies

```bash
cd ABM2
python -m pip install -r requirements.txt
python -m pip install dash plotly  # For dashboard
```

### 2. Launch Dashboard

```bash
python dashboard.py
```

Then open your browser to: **http://127.0.0.1:8050**

## Dashboard Walkthrough

### Tab 1: ⚙️ Setup

**Configure your simulation:**

1. **Choose LLM Provider:**
   - **Mock** (Recommended for testing) - Fast, no API key needed, uses default personas
   - **Anthropic Claude** - Requires `ANTHROPIC_API_KEY` environment variable
   - **OpenAI GPT** - Requires `OPENAI_API_KEY` environment variable

2. **Set Parameters:**
   - Total Simulation Days (default: 504 = 2 years)
   - Number of Agents (default: 150)
   - Total Capital in $B (default: 10)
   - Random Seed for reproducibility

3. **Configure Recalibration:**
   - Enable/disable periodic recalibration
   - Set quarterly reflection interval (days)

4. **Click "Initialize Model"**

### Tab 2: 🧠 Calibration

**Generate agent personas:**

1. **Click "Run Calibration"**
   - In Mock mode: Instant, uses default personas
   - In LLM mode: Takes 2-5 minutes, calls LLM API for each agent type

2. **View Results:**
   - Agent distribution chart shows count by type
   - Sample persona displays the generated decision framework for one agent
   - Review investment philosophy, risk tolerance, decision rules, biases

### Tab 3: ▶️ Simulation

**Run the market simulation:**

1. **Set days to run** (can run in batches)

2. **Click "Run"** to start simulation

3. **Monitor Progress:**
   - Progress gauge shows completion
   - Live price chart shows current asset changes
   - Status updates display events and progress

4. **Note:** In current version, runs complete instantly. Future: real-time streaming updates

### Tab 4: 📊 Results

**Analyze simulation outcomes:**

1. **Click "Refresh Results"** to load latest data

2. **View Metrics:**
   - Summary cards: Days completed, active agents, events, avg return
   - Price evolution: Line chart showing how asset prices evolved
   - Agent performance: Histogram of agent returns
   - Events timeline: List of recent market events

3. **Export data** (future feature)

### Tab 5: ✓ Validation

**Validate model realism:**

1. **Click "Calculate Stylized Facts"**

2. **Review Metrics:**
   - Return kurtosis (fat tails)
   - Volatility clustering
   - Skewness
   - Correlations

3. **Examine Charts:**
   - Return distribution with box plots
   - Volatility clustering autocorrelation

## Usage Examples

### Quick Test (Mock Mode)

1. Setup Tab: Leave defaults, click "Initialize Model"
2. Calibration Tab: Click "Run Calibration" (instant)
3. Simulation Tab: Set 100 days, click "Run"
4. Results Tab: Click "Refresh Results" to see outcomes
5. Validation Tab: Click "Calculate Stylized Facts"

**Time: ~30 seconds**

### Full LLM Run (Requires API Key)

**Prerequisites:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

1. Setup Tab: Select "Anthropic Claude", click "Initialize Model"
2. Calibration Tab: Click "Run Calibration" (2-5 minutes for 150 agents)
3. Review generated personas - examine beliefs, biases, decision rules
4. Simulation Tab: Set 504 days (full 2-year run), click "Run"
5. Results Tab: Analyze price dynamics and agent performance
6. Validation Tab: Check if stylized facts match real markets

**Time: 5-10 minutes total**

## CLI Alternative (No Dashboard)

If you prefer command-line:

```python
from core.model import MarketModel

# Initialize
model = MarketModel()

# Setup with mock agents (fast)
model.setup(calibrate_agents=False)

# OR setup with LLM calibration (slow, requires API key)
# model.setup(calibrate_agents=True)

# Run simulation
model.run(steps=100)

# Get results
summary = model.get_summary()
print(f"Days: {summary['current_day']}")
print(f"Prices: {summary['market']['prices']}")
print(f"Avg Return: {summary['agents']['avg_return']:.2f}%")

# Get stylized facts
facts = model.data_collector.get_stylized_facts()
print("\nStylized Facts:")
for key, value in facts.items():
    print(f"  {key}: {value:.4f}")
```

## Next Steps

### After Your First Run

1. **Experiment with parameters** - Change agent counts, capital distribution
2. **Try different scenarios** - Inject shocks, modify event frequencies
3. **Analyze specific agents** - Deep dive into persona → behavior → performance
4. **Compare LLM vs Mock** - How do LLM-generated personas differ from defaults?

### Historical Scenario Validation (Future)

Once comfortable with the system, test against:
- 2008 Financial Crisis
- COVID-19 Market Crash (March 2020)
- GameStop Short Squeeze (Jan 2021)
- Flash Crash (May 2010)

Configure initial conditions and events to match historical scenarios, then validate if emergent dynamics match reality.

### Calibration Refinement (Future)

- Parameter sensitivity analysis
- Prompt engineering for better personas
- Multi-run ensemble analysis
- Optimize agent distribution for realistic dynamics

## Troubleshooting

### "Model not initialized"
- Click "Initialize Model" in Setup tab first

### "No results available"
- Run calibration and simulation first
- Click "Refresh Results" button

### LLM API Errors
- Check API key is set: `echo $ANTHROPIC_API_KEY`
- Verify key is valid and has credits
- Try Mock mode instead for testing

### Slow Performance
- Reduce number of agents (50-100 instead of 150)
- Run shorter simulations (100 days instead of 504)
- Use Mock mode instead of LLM mode

### Dashboard Not Loading
- Check if port 8050 is available
- Try different port: `app.run_server(port=8051)`
- Check console for errors

## Tips

- **Start with Mock mode** to understand the system
- **Save interesting runs** - Take screenshots or export data
- **Iterate quickly** - Run short simulations (50-100 days) while testing
- **Review personas** - Understanding agent decision frameworks is key
- **Watch events** - Market events drive heterogeneous agent responses

## Support

- Documentation: See `README.md` and `SYSTEM_PLAN.md`
- System test: Run `python test_system.py`
- Issues: Check console output for detailed error messages

---

**Ready to explore emergent market dynamics from heterogeneous agent behaviors!** 🚀
