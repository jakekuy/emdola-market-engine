"""
Base Agent Classes
Foundation for all trader agent types
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import uuid


class BaseTraderAgent(ABC):
    """
    Abstract base class for all trader agents

    Agents have:
    - Unique identity and persona (LLM-generated)
    - Portfolio (cash + positions)
    - Decision framework (LLM-generated behavioral rules)
    - State tracking (performance, recent trades)
    """

    def __init__(
        self,
        model,
        agent_type: str,
        capital: float,
        persona: Optional[Dict] = None
    ):
        """
        Initialize base trader agent

        Args:
            model: Parent market model
            agent_type: Agent archetype (e.g., 'pension_fund')
            capital: Initial capital allocation
            persona: LLM-generated persona dict (set during calibration)
        """
        self.model = model
        self.agent_id = str(uuid.uuid4())[:8]
        self.agent_type = agent_type

        # Portfolio
        self.cash = capital
        self.initial_capital = capital
        self.positions = {}  # {ticker: quantity}

        # Persona & behavior (LLM-generated)
        self.persona = persona or {}
        self.decision_rules = {}  # LLM-generated decision framework
        self.risk_parameters = {}  # LLM-generated risk constraints

        # State tracking
        self.active = True
        self.trade_history = []
        self.portfolio_history = []
        self.last_recalibration_day = 0

        # Performance metrics
        self.total_pnl = 0.0
        self.max_portfolio_value = capital
        self.max_drawdown = 0.0

    @property
    def portfolio_value(self) -> float:
        """
        Calculate current portfolio value (cash + positions)

        Returns:
            Total portfolio value
        """
        position_value = 0.0

        if self.model.market:
            for ticker, quantity in self.positions.items():
                asset = self.model.market.get_asset(ticker)
                if asset:
                    position_value += quantity * asset.price

        return self.cash + position_value

    @property
    def portfolio_return(self) -> float:
        """
        Calculate portfolio return since inception

        Returns:
            Total return (%)
        """
        if self.initial_capital == 0:
            return 0.0
        return (self.portfolio_value / self.initial_capital - 1) * 100

    def step(self):
        """
        Execute one time step - make trading decisions

        Process:
        1. Observe market state
        2. Interpret current situation (using LLM-generated framework)
        3. Generate trading signal
        4. Submit orders to market
        5. Update state
        """
        if not self.active:
            return

        # Get current market state
        market_state = self.model.get_market_state()

        # Make trading decision (implemented by subclasses)
        orders = self.make_decision(market_state)

        # Submit orders to market
        if orders and self.model.market:
            for order in orders:
                self.model.market.submit_order(order)

        # Update state
        self._update_state()

    @abstractmethod
    def make_decision(self, market_state: Dict) -> List[Dict]:
        """
        Make trading decision based on current market state

        This method uses the LLM-generated decision framework
        to determine what trades to make

        Args:
            market_state: Current market conditions

        Returns:
            List of order dictionaries
        """
        pass

    def _update_state(self):
        """Update agent state variables"""
        current_value = self.portfolio_value

        # Update max value & drawdown
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value

        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Track portfolio history
        self.portfolio_history.append({
            'day': self.model.current_day,
            'value': current_value,
            'cash': self.cash,
            'positions': self.positions.copy()
        })

    def execute_trade(self, ticker: str, quantity: int, price: float):
        """
        Execute a trade (called by market when order fills)

        Args:
            ticker: Asset ticker
            quantity: Number of shares (positive = buy, negative = sell)
            price: Execution price
        """
        cost = quantity * price

        # Update cash
        self.cash -= cost

        # Update positions
        if ticker not in self.positions:
            self.positions[ticker] = 0
        self.positions[ticker] += quantity

        # Remove zero positions
        if self.positions[ticker] == 0:
            del self.positions[ticker]

        # Record trade
        self.trade_history.append({
            'day': self.model.current_day,
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'cost': cost
        })

    def can_afford(self, ticker: str, quantity: int, price: float) -> bool:
        """
        Check if agent can afford a trade

        Args:
            ticker: Asset ticker
            quantity: Number of shares
            price: Price per share

        Returns:
            True if affordable, False otherwise
        """
        cost = quantity * price
        return self.cash >= cost

    def get_position_value(self, ticker: str) -> float:
        """
        Get current value of position in specific asset

        Args:
            ticker: Asset ticker

        Returns:
            Position value
        """
        if ticker not in self.positions:
            return 0.0

        if not self.model.market:
            return 0.0

        asset = self.model.market.get_asset(ticker)
        if not asset:
            return 0.0

        return self.positions[ticker] * asset.price

    def get_position_pct(self, ticker: str) -> float:
        """
        Get position as percentage of portfolio

        Args:
            ticker: Asset ticker

        Returns:
            Position percentage
        """
        portfolio_val = self.portfolio_value
        if portfolio_val == 0:
            return 0.0

        position_val = self.get_position_value(ticker)
        return (position_val / portfolio_val) * 100

    def needs_recalibration(self) -> bool:
        """
        Check if agent should be recalibrated

        Returns:
            True if recalibration needed
        """
        # Check drawdown trigger
        triggers = self.model.config['llm']['recalibration']['triggers']
        if self.max_drawdown > triggers['drawdown_threshold']:
            return True

        # Check time since last recalibration
        days_since = self.model.current_day - self.last_recalibration_day
        if days_since >= triggers['quarterly_reflection']:
            return True

        return False

    def recalibrate(self, updated_persona: Dict):
        """
        Update agent's decision framework via LLM recalibration

        Args:
            updated_persona: New persona dict from LLM
        """
        self.persona.update(updated_persona)
        self.last_recalibration_day = self.model.current_day

        print(f"Agent {self.agent_id} ({self.agent_type}) recalibrated at day {self.model.current_day}")

    def __repr__(self):
        return f"{self.agent_type}_{self.agent_id} (${self.portfolio_value:,.0f})"


class SimpleTraderAgent(BaseTraderAgent):
    """
    Simple trader implementation for testing

    Makes random trades with position sizing rules
    """

    def make_decision(self, market_state: Dict) -> List[Dict]:
        """
        Make simple random trading decision

        Returns:
            List of orders
        """
        import random

        # Skip trading sometimes
        if random.random() > 0.3:  # 30% chance to trade
            return []

        # Pick random asset
        if not self.model.market:
            return []

        assets = self.model.market.assets
        if not assets:
            return []

        asset = random.choice(assets)

        # Random direction
        if random.random() > 0.5:
            # Buy
            max_position_size = self.portfolio_value * 0.1  # Max 10%
            quantity = int(max_position_size / asset.price)

            if quantity > 0 and self.can_afford(asset.ticker, quantity, asset.price):
                return [{
                    'agent_id': self.agent_id,
                    'ticker': asset.ticker,
                    'quantity': quantity,
                    'order_type': 'market',
                    'direction': 'buy'
                }]
        else:
            # Sell
            if asset.ticker in self.positions and self.positions[asset.ticker] > 0:
                quantity = int(self.positions[asset.ticker] * 0.5)  # Sell 50%
                if quantity > 0:
                    return [{
                        'agent_id': self.agent_id,
                        'ticker': asset.ticker,
                        'quantity': quantity,
                        'order_type': 'market',
                        'direction': 'sell'
                    }]

        return []
