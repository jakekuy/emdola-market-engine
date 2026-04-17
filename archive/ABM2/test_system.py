"""
Test Script for LLM-Enhanced ABM
Verifies end-to-end system functionality
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model import MarketModel


def test_basic_simulation(use_llm: bool = False):
    """
    Test basic simulation functionality

    Args:
        use_llm: Whether to use LLM calibration (requires API key) or mock agents
    """
    print("=" * 70)
    print("LLM-Enhanced Financial ABM - System Test")
    print("=" * 70)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 1. Initialize model
        print("\n[1/5] Initializing model...")
        model = MarketModel()
        print(f"✓ Model initialized with {model.config['agents']['total_count']} agents configured")

        # 2. Setup components (with or without LLM calibration)
        print("\n[2/5] Setting up components...")
        if use_llm:
            print("Note: This will make LLM API calls for agent calibration")
            confirmation = input("Continue with LLM calibration? (y/n): ")
            if confirmation.lower() != 'y':
                print("Switching to mock mode...")
                use_llm = False

        model.setup(calibrate_agents=use_llm)
        print(f"✓ Setup complete: {len(model.agents)} agents, {len(model.asset_universe.assets)} assets")

        # 3. Verify initial state
        print("\n[3/5] Verifying initial state...")
        market_state = model.get_market_state()
        print(f"✓ Market state: Day {market_state['day']}")
        print(f"  Prices: {market_state['prices']}")

        # Show sample agent
        if model.agents:
            sample_agent = model.agents[0]
            print(f"✓ Sample agent: {sample_agent.unique_id} ({sample_agent.agent_type})")
            print(f"  Capital: ${sample_agent.cash:,.0f}")
            print(f"  Positions: {sample_agent.positions}")
            print(f"  Portfolio value: ${sample_agent.portfolio_value:,.0f}")

        # 4. Run short simulation
        print("\n[4/5] Running simulation (10 days)...")
        model.run(steps=10)
        print(f"✓ Simulation completed: {model.current_day} days")

        # 5. Display results
        print("\n[5/5] Displaying results...")
        summary = model.get_summary()

        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY")
        print(f"{'='*70}")
        print(f"Days simulated: {summary['current_day']}")
        print(f"Active agents: {summary.get('agents', {}).get('active_agents', 0)}")
        print(f"Market events: {summary['num_events']}")
        print(f"Recalibrations: {summary['num_recalibrations']}")

        print(f"\nFinal Prices:")
        for ticker, price in summary['market']['prices'].items():
            print(f"  {ticker}: ${price:.2f}")

        if 'agents' in summary:
            print(f"\nAggregate Metrics:")
            print(f"  Total portfolio value: ${summary['agents']['total_portfolio_value']:,.0f}")
            print(f"  Average return: {summary['agents']['avg_return']:.2f}%")

        # Show sample agent final state
        if model.agents:
            sample_agent = model.agents[0]
            perf = sample_agent.get_performance_history()
            print(f"\nSample Agent ({sample_agent.unique_id}):")
            print(f"  Type: {sample_agent.agent_type}")
            print(f"  Total return: {perf['total_return']:.2f}%")
            print(f"  Max drawdown: {perf['max_drawdown']:.2f}%")
            print(f"  Portfolio value: ${perf['current_value']:,.0f}")
            print(f"  Current allocation: {perf['current_allocation']}")

        # Show recent events
        if model.event_history:
            print(f"\nRecent Events:")
            for event in model.event_history[-5:]:
                print(f"  Day {event.day}: {event.description}")

        print(f"\n{'='*70}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")

        return model

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_collection(model: MarketModel):
    """
    Test data collection and metrics

    Args:
        model: Initialized and run model
    """
    print("\n" + "="*70)
    print("DATA COLLECTION TEST")
    print("="*70)

    # Get collected data
    agent_data = model.data_collector.agent_data
    model_data = model.data_collector.model_data

    print(f"\nCollected data:")
    print(f"  Model-level data points: {len(model_data)}")
    print(f"  Agent-level data points: {len(agent_data)}")

    if model_data:
        print(f"\nLatest model metrics:")
        latest = model_data[-1]
        for key, value in latest.items():
            if key != 'day':
                print(f"  {key}: {value}")

    # Get stylized facts
    print(f"\nCalculating stylized facts...")
    facts = model.data_collector.get_stylized_facts()

    print(f"\nStylized Facts:")
    for key, value in list(facts.items())[:10]:  # Show first 10
        print(f"  {key}: {value:.4f}")


def main():
    """Main test execution"""
    print("\nChoose test mode:")
    print("1. Mock mode (no LLM API calls - fast)")
    print("2. LLM mode (requires API key - slow but realistic)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    use_llm = (choice == "2")

    # Run basic test
    model = test_basic_simulation(use_llm=use_llm)

    if model:
        # Run data collection test
        test_data_collection(model)

        print("\n✓ All tests passed successfully!")
        return True
    else:
        print("\n❌ Tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
