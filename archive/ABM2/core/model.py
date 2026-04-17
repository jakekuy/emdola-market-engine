"""
Core ABM Model
Main simulation controller for LLM-enhanced financial market model
"""

import yaml
from pathlib import Path
import numpy as np
import random
import logging
from typing import Dict, List, Optional

from .scheduler import TradingScheduler
from .datacollector import MarketDataCollector
from market.instruments import AssetUniverse
from market.orderbook import Market
from agents.trader import LLMTraderAgent
from events.news import NewsGenerator
from llm.interface import create_llm_interface
from llm.calibrator import AgentCalibrator


class MarketModel:
    """
    Main ABM model coordinating agents, market, and events

    Two-phase execution:
    1. Calibration: LLM generates agent behaviors
    2. Simulation: Agents interact, market evolves
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize market model

        Args:
            config_path: Path to configuration YAML file
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "parameters.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed for reproducibility
        self.random_seed = self.config['simulation']['random_seed']
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Simulation state
        self.current_day = 0
        self.total_days = self.config['simulation']['total_days']
        self.running = True

        # Components (will be initialized)
        self.scheduler = TradingScheduler(self)
        self.data_collector = MarketDataCollector(self)
        self.market = None
        self.asset_universe = None
        self.agents = []
        self.news_generator = None
        self.llm_interface = None
        self.calibrator = None

        # Event tracking
        self.event_history = []
        self.recent_events = []
        self.recalibration_history = []

        self.logger.info(f"MarketModel initialized: {self.total_days} days simulation")

    def setup(self, calibrate_agents: bool = True):
        """
        Setup model components

        Args:
            calibrate_agents: Whether to calibrate agents via LLM (True) or use defaults (False)
        """
        self.logger.info("Setting up market model...")

        # 1. Initialize asset universe
        self.logger.info("Initializing asset universe...")
        asset_configs = self.config['market']['assets']
        correlations = self.config['market']['correlations']
        self.asset_universe = AssetUniverse(asset_configs, correlations)

        # 2. Initialize market (order books)
        self.logger.info("Initializing market...")
        self.market = Market(self.asset_universe)

        # 3. Initialize news/events generator
        self.logger.info("Initializing news generator...")
        self.news_generator = NewsGenerator(self.config)

        # 4. Initialize LLM interface and calibrator
        if calibrate_agents:
            self.logger.info("Initializing LLM interface...")
            self.llm_interface = create_llm_interface(self.config['llm'])
            self.calibrator = AgentCalibrator(self.llm_interface, self.config)

            # 5. Calibrate agents via LLM
            self._calibrate_all_agents()
        else:
            self.logger.info("Skipping LLM calibration, using mock agents...")
            self._create_mock_agents()

        # 6. Register agents with scheduler
        for agent in self.agents:
            self.scheduler.add_agent(agent)

        self.logger.info(f"Setup complete: {len(self.agents)} agents, {len(self.asset_universe.assets)} assets")

    def _calibrate_all_agents(self):
        """Calibrate all agents using LLM"""
        self.logger.info("Starting LLM calibration...")

        # Get agent distribution from config
        distribution = self.config['agents']['distribution']
        capital_dist = self.config['agents']['capital_distribution']
        total_capital = self.config['agents']['total_capital']

        # Build agent specifications
        agent_specs = []
        for agent_type, count in distribution.items():
            capital_weight = capital_dist.get(agent_type, 1.0) / 100.0
            agent_capital = (total_capital * capital_weight) / count

            agent_specs.append({
                'agent_type': agent_type,
                'capital': agent_capital,
                'count': count
            })

        # Get market information for calibration context
        market_info = {
            'assets': [
                {'ticker': a.ticker, 'name': a.name}
                for a in self.asset_universe.get_all_assets()
            ],
            'interest_rate': 'neutral',
            'volatility_regime': 'normal',
            'economic_cycle': 'expansion'
        }

        # Calibrate agents
        personas_by_type = self.calibrator.calibrate_all_agents(agent_specs, market_info)

        # Create agent instances
        agent_id = 0
        for agent_type, personas in personas_by_type.items():
            capital_weight = capital_dist.get(agent_type, 1.0) / 100.0
            agent_capital = (total_capital * capital_weight) / len(personas)

            for persona in personas:
                agent = LLMTraderAgent(
                    model=self,
                    agent_id=f"agent_{agent_id}",
                    agent_type=agent_type,
                    capital=agent_capital,
                    persona=persona
                )
                self.agents.append(agent)
                agent_id += 1

        self.logger.info(f"Created {len(self.agents)} LLM-calibrated agents")

    def _create_mock_agents(self):
        """Create simple mock agents without LLM calibration"""
        distribution = self.config['agents']['distribution']
        capital_dist = self.config['agents']['capital_distribution']
        total_capital = self.config['agents']['total_capital']

        agent_id = 0
        for agent_type, count in distribution.items():
            capital_weight = capital_dist.get(agent_type, 1.0) / 100.0
            agent_capital = (total_capital * capital_weight) / count

            for _ in range(count):
                # Use default persona
                from llm.parser import ResponseParser
                persona = ResponseParser._default_persona()

                agent = LLMTraderAgent(
                    model=self,
                    agent_id=f"agent_{agent_id}",
                    agent_type=agent_type,
                    capital=agent_capital,
                    persona=persona
                )
                self.agents.append(agent)
                agent_id += 1

        self.logger.info(f"Created {len(self.agents)} mock agents")

    def step(self):
        """
        Execute one day of trading

        Process:
        1. Generate/announce any events (news, shocks)
        2. Agents make decisions based on current state
        3. Market processes orders
        4. Update prices
        5. Collect data
        6. Check recalibration triggers
        """
        if not self.running or self.current_day >= self.total_days:
            self.running = False
            return

        self.current_day += 1

        # Step 1: Events (to be implemented)
        self._process_events()

        # Step 2: Agent decisions via scheduler
        self.scheduler.step()

        # Step 3: Market clearing (to be implemented)
        if self.market:
            self.market.clear_orders()

        # Step 4: Data collection
        self.data_collector.collect(self)

        # Step 5: Check recalibration triggers
        self._check_recalibration()

        if self.current_day % 50 == 0:
            print(f"Day {self.current_day}/{self.total_days}")

    def run(self, steps: Optional[int] = None):
        """
        Run simulation for specified number of steps

        Args:
            steps: Number of days to simulate (None = run to completion)
        """
        if steps is None:
            steps = self.total_days - self.current_day

        for _ in range(steps):
            if not self.running:
                break
            self.step()

        print(f"Simulation completed: {self.current_day} days")

    def _process_events(self):
        """Generate and process market events"""
        if not self.news_generator:
            return

        # Generate event for this day
        event = self.news_generator.generate_event(self.current_day)

        if event:
            self.event_history.append(event)
            self.recent_events.append(event)

            # Keep only recent events (last 30 days)
            if len(self.recent_events) > 30:
                self.recent_events = self.recent_events[-30:]

            self.logger.info(
                f"Day {self.current_day}: {event.description} "
                f"(affects: {', '.join(event.affected_assets)})"
            )

    def _check_recalibration(self):
        """
        Check if recalibration should be triggered

        Triggers:
        - Major price move (>5%)
        - Quarterly reflection (every 90 days)
        - Significant drawdown (>20%)
        - Volatility regime shift
        """
        if not self.config['llm']['recalibration']['enabled']:
            return

        if not self.calibrator:
            return

        triggers = self.config['llm']['recalibration']['triggers']

        # Quarterly reflection
        if self.current_day % triggers['quarterly_reflection'] == 0:
            self._trigger_recalibration("quarterly_reflection")
            return

        # Check for major price moves
        for ticker in ['TECH', 'VALUE', 'SAFE']:
            asset = self.asset_universe.get_asset(ticker)
            if asset and len(asset.price_history) >= 2:
                price_change = abs((asset.price / asset.price_history[-2]) - 1)
                if price_change > triggers['major_move_threshold']:
                    self._trigger_recalibration(f"major_move_{ticker}")
                    return

        # Check individual agents for drawdown triggers
        agents_needing_recal = [a for a in self.agents if a.needs_recalibration()]
        if agents_needing_recal:
            self._recalibrate_agents(agents_needing_recal)

    def _trigger_recalibration(self, reason: str):
        """
        Trigger LLM recalibration for all agents

        Args:
            reason: Why recalibration was triggered
        """
        self.logger.info(f"Recalibration triggered: {reason} at day {self.current_day}")

        self.recalibration_history.append({
            'day': self.current_day,
            'reason': reason
        })

        # Recalibrate all agents
        self._recalibrate_agents(self.agents)

    def _recalibrate_agents(self, agents: List):
        """
        Recalibrate specified agents using LLM

        Args:
            agents: List of agents to recalibrate
        """
        if not self.calibrator:
            return

        self.logger.info(f"Recalibrating {len(agents)} agents...")

        # Prepare recalibration data
        agents_to_recalibrate = []
        for agent in agents:
            # Get agent performance history
            performance = agent.get_performance_history()

            # Get market history
            market_history = self._get_market_history()

            # Recent events
            recent_events = [
                {
                    'day': e.day,
                    'description': e.description,
                    'sentiment': e.sentiment.value if hasattr(e.sentiment, 'value') else str(e.sentiment)
                }
                for e in self.recent_events[-10:]
            ]

            agents_to_recalibrate.append((agent, performance, market_history, recent_events))

        # Batch recalibrate
        updates = self.calibrator.batch_recalibrate(agents_to_recalibrate)

        # Apply updates
        for agent_id, updated_persona in updates.items():
            # Find agent
            agent = next((a for a in agents if a.unique_id == agent_id), None)
            if agent:
                agent.apply_recalibration(updated_persona)

        self.logger.info(f"Recalibration complete for {len(updates)} agents")

    def _get_market_history(self) -> Dict:
        """Get market evolution metrics for recalibration"""
        if len(self.data_collector.model_data) < 2:
            return {
                'market_return': 0.0,
                'volatility_change': 'stable',
                'correlation_regime': 'normal'
            }

        # Calculate market-wide metrics from collected data
        # Simplified version
        return {
            'market_return': 0.0,  # Would calculate from price history
            'volatility_change': 'stable',
            'correlation_regime': 'normal'
        }

    def get_market_state(self) -> Dict:
        """
        Get current market state snapshot

        Returns:
            Dictionary with current prices, volumes, volatility, etc.
        """
        state = {
            'day': self.current_day,
            'running': self.running,
            'prices': {},
            'volatility': {},
            'volumes': {}
        }

        if self.asset_universe:
            for ticker in ['TECH', 'VALUE', 'SAFE']:
                asset = self.asset_universe.get_asset(ticker)
                if asset:
                    state['prices'][ticker] = asset.price
                    state['volatility'][ticker] = asset.realized_volatility

        return state

    def get_agent_performance(self, agent_id: str) -> Dict:
        """
        Get performance metrics for specific agent

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with returns, drawdown, positions, etc.
        """
        agent = next((a for a in self.agents if a.unique_id == agent_id), None)

        if not agent:
            return {}

        return agent.get_performance_history()

    def get_summary(self) -> Dict:
        """
        Get comprehensive simulation summary

        Returns:
            Summary statistics and metrics
        """
        summary = {
            'current_day': self.current_day,
            'total_days': self.total_days,
            'num_agents': len(self.agents),
            'num_events': len(self.event_history),
            'num_recalibrations': len(self.recalibration_history)
        }

        # Add market state
        summary['market'] = self.get_market_state()

        # Add aggregate agent statistics
        if self.agents:
            total_portfolio_value = sum(a.portfolio_value for a in self.agents)
            # Calculate average return
            returns = [(a.portfolio_value / a.initial_capital - 1) * 100
                      for a in self.agents if a.initial_capital > 0]
            avg_return = np.mean(returns) if returns else 0.0

            summary['agents'] = {
                'total_portfolio_value': total_portfolio_value,
                'avg_return': avg_return,
                'active_agents': sum(1 for a in self.agents if a.active)
            }

        # Add data collector summary
        if hasattr(self.data_collector, 'get_summary'):
            summary['metrics'] = self.data_collector.get_summary()

        return summary


if __name__ == "__main__":
    # Test basic model initialization
    model = MarketModel()
    print(f"Configuration loaded: {len(model.config)} sections")
    print(f"Total agents configured: {model.config['agents']['total_count']}")
