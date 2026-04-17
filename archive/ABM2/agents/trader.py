"""
LLM-Enhanced Trader Agents
Concrete agent implementations using LLM-generated personas
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from agents.base import BaseTraderAgent
from llm.parser import ResponseParser


class LLMTraderAgent(BaseTraderAgent):
    """
    Trader agent using LLM-generated persona for decision-making

    The LLM 'programs' the agent's behavior through:
    - Investment philosophy and beliefs
    - Decision rules (entry/exit triggers)
    - Risk constraints and position sizing
    - Cognitive biases and information processing

    The agent then executes these programmed behaviors in the ABM
    """

    def __init__(
        self,
        model,
        agent_id: str,
        agent_type: str,
        capital: float,
        persona: Dict
    ):
        """
        Initialize LLM-enhanced trader

        Args:
            model: Reference to main model
            agent_id: Unique agent identifier
            agent_type: Agent archetype (e.g., 'pension_fund')
            capital: Initial capital
            persona: LLM-generated persona dict
        """
        super().__init__(model, agent_type, capital, persona)
        self.unique_id = agent_id
        self.logger = logging.getLogger(f"Agent.{agent_id}")

        # Extract decision parameters from persona
        self._extract_decision_parameters()

        # Initialize agent state
        self._initialize_from_persona()

        # Experience tracking for recalibration
        self.days_active = 0
        self.last_recalibration_day = 0

    def _extract_decision_parameters(self):
        """Extract actionable parameters from LLM persona"""
        # Position sizing
        self.position_size = ResponseParser.extract_position_size(self.persona)
        self.max_position_size = ResponseParser.extract_max_position(self.persona)

        # Risk parameters
        risk_constraints = self.persona.get('risk_constraints', {})
        self.max_drawdown_tolerance = self._parse_percentage(
            risk_constraints.get('max_drawdown_tolerance', '20%')
        )
        self.max_leverage = float(risk_constraints.get('max_leverage', 1.0))

        # Information processing
        info_proc = self.persona.get('information_processing', {})
        self.reaction_speed = info_proc.get('reaction_speed', 'moderate')
        self.contrarian_tendency = info_proc.get('contrarian_tendency', 50) / 100.0
        self.fundamental_weight = info_proc.get('fundamental_vs_technical', 50) / 100.0

        # Cognitive biases
        self.biases = self.persona.get('cognitive_biases', [])

        # Beliefs
        beliefs = self.persona.get('beliefs', {})
        self.mean_reversion_belief = self._belief_to_strength(
            beliefs.get('mean_reversion', 'medium')
        )
        self.momentum_belief = self._belief_to_strength(
            beliefs.get('momentum_effects', 'medium')
        )

    def _initialize_from_persona(self):
        """Initialize portfolio from LLM-recommended allocation"""
        initial_allocation = self.persona.get('initial_allocation', {})

        # Convert allocation percentages to actual positions
        for ticker, weight in initial_allocation.items():
            asset = self.model.market.get_asset(ticker)
            if asset:
                # Calculate target value and shares
                target_value = self.cash * weight
                shares = int(target_value / asset.price)

                if shares > 0:
                    cost = shares * asset.price
                    if cost <= self.cash:
                        self.positions[ticker] = shares
                        self.cash -= cost

        self.logger.info(f"Initialized {self.agent_type} with portfolio: {self.positions}")

    def step(self):
        """Execute one trading day"""
        self.days_active += 1

        # Get market information
        market_state = self._observe_market()

        # Generate trading decisions based on persona
        decisions = self._make_trading_decisions(market_state)

        # Execute trades
        for decision in decisions:
            self._execute_trade(decision)

        # Update performance tracking
        self._update_state()

    def _observe_market(self) -> Dict:
        """
        Observe market state and filter through agent's perspective

        Returns:
            Dict of market observations
        """
        market_state = {
            'prices': {},
            'returns': {},
            'volumes': {},
            'volatility': {}
        }

        for ticker in ['TECH', 'VALUE', 'SAFE']:
            asset = self.model.market.get_asset(ticker)
            if asset and len(asset.price_history) > 0:
                market_state['prices'][ticker] = asset.price

                # Calculate returns if history exists
                if len(asset.price_history) > 1:
                    returns = np.diff(asset.price_history[-20:]) / asset.price_history[-20:-1]
                    market_state['returns'][ticker] = returns[-1] if len(returns) > 0 else 0.0
                    market_state['volatility'][ticker] = asset.realized_volatility
                else:
                    market_state['returns'][ticker] = 0.0
                    market_state['volatility'][ticker] = asset.base_volatility

        # Add recent events
        market_state['recent_events'] = self.model.recent_events[-5:]

        return market_state

    def make_decision(self, market_state: Dict) -> List[Dict]:
        """
        Make trading decisions based on LLM-programmed persona
        (Implements abstract method from BaseTraderAgent)

        Args:
            market_state: Current market observations

        Returns:
            List of trading decisions
        """
        return self._make_trading_decisions(market_state)

    def _make_trading_decisions(self, market_state: Dict) -> List[Dict]:
        """
        Internal method: Make trading decisions based on LLM-programmed persona

        Args:
            market_state: Current market observations

        Returns:
            List of trading decisions
        """
        decisions = []

        # Get market views from persona
        market_views = self.persona.get('market_views', {})

        for ticker in ['TECH', 'VALUE', 'SAFE']:
            # Get agent's view on this asset
            view_data = market_views.get(ticker, {})
            if isinstance(view_data, dict):
                view = view_data.get('view', 'neutral')
            else:
                view = str(view_data) if view_data else 'neutral'

            # Calculate signal strength based on beliefs and market state
            signal = self._calculate_signal(ticker, view, market_state)

            # Apply cognitive biases
            signal = self._apply_biases(signal, ticker, market_state)

            # Check if signal meets decision rules
            decision = self._evaluate_decision_rules(ticker, signal, market_state)

            if decision:
                decisions.append(decision)

        return decisions

    def _calculate_signal(
        self,
        ticker: str,
        base_view: str,
        market_state: Dict
    ) -> float:
        """
        Calculate trading signal strength (-1 to 1)

        Args:
            ticker: Asset ticker
            base_view: Agent's base view (bullish/neutral/bearish)
            market_state: Market observations

        Returns:
            Signal strength
        """
        # Convert view to base signal
        view_signals = {
            'bullish': 0.6,
            'very_bullish': 0.8,
            'neutral': 0.0,
            'bearish': -0.6,
            'very_bearish': -0.8
        }
        signal = view_signals.get(base_view.lower(), 0.0)

        # Adjust based on recent returns (momentum vs mean reversion)
        recent_return = market_state['returns'].get(ticker, 0.0)

        # Momentum component
        momentum_signal = np.sign(recent_return) * min(abs(recent_return) * 10, 0.3)
        signal += momentum_signal * self.momentum_belief

        # Mean reversion component (opposite sign)
        reversion_signal = -np.sign(recent_return) * min(abs(recent_return) * 10, 0.3)
        signal += reversion_signal * self.mean_reversion_belief

        # React to events affecting this asset
        event_signal = self._process_events(ticker, market_state.get('recent_events', []))
        signal += event_signal

        # Clip to valid range
        signal = np.clip(signal, -1.0, 1.0)

        return signal

    def _process_events(self, ticker: str, events: List) -> float:
        """
        Process recent events and generate signal adjustment

        Args:
            ticker: Asset ticker
            events: Recent market events

        Returns:
            Event-based signal adjustment
        """
        if not events:
            return 0.0

        event_signal = 0.0

        for event in events:
            # Check if event affects this asset
            if hasattr(event, 'affected_assets') and ticker in event.affected_assets:
                # Convert sentiment to signal
                sentiment_map = {
                    'very_positive': 0.4,
                    'positive': 0.2,
                    'neutral': 0.0,
                    'negative': -0.2,
                    'very_negative': -0.4
                }

                if hasattr(event, 'sentiment'):
                    sentiment_value = event.sentiment.value if hasattr(event.sentiment, 'value') else str(event.sentiment)
                    base_signal = sentiment_map.get(sentiment_value, 0.0)

                    # Weight by event magnitude
                    magnitude = event.magnitude if hasattr(event, 'magnitude') else 0.5
                    event_signal += base_signal * magnitude

                    # Apply contrarian tendency
                    if random.random() < self.contrarian_tendency:
                        event_signal *= -1

        return np.clip(event_signal, -0.5, 0.5)

    def _apply_biases(self, signal: float, ticker: str, market_state: Dict) -> float:
        """
        Apply cognitive biases to signal

        Args:
            signal: Base signal
            ticker: Asset ticker
            market_state: Market state

        Returns:
            Biased signal
        """
        biased_signal = signal

        # Confirmation bias - strengthen signals that agree with existing position
        if 'confirmation bias' in [b.lower() for b in self.biases]:
            current_position = self.positions.get(ticker, 0)
            if (current_position > 0 and signal > 0) or (current_position < 0 and signal < 0):
                biased_signal *= 1.2

        # Anchoring - stick to original views more strongly
        if 'anchoring' in [b.lower() for b in self.biases]:
            market_views = self.persona.get('market_views', {})
            original_view = market_views.get(ticker, {})
            if isinstance(original_view, dict):
                view_str = original_view.get('view', 'neutral')
                if (view_str == 'bullish' and signal > 0) or (view_str == 'bearish' and signal < 0):
                    biased_signal *= 1.15

        # Recency bias - overweight recent returns
        if 'recency bias' in [b.lower() for b in self.biases]:
            recent_return = market_state['returns'].get(ticker, 0.0)
            biased_signal += np.sign(recent_return) * 0.1

        return np.clip(biased_signal, -1.0, 1.0)

    def _evaluate_decision_rules(
        self,
        ticker: str,
        signal: float,
        market_state: Dict
    ) -> Optional[Dict]:
        """
        Evaluate decision rules and generate trade decision

        Args:
            ticker: Asset ticker
            signal: Trading signal strength
            market_state: Market state

        Returns:
            Trade decision dict or None
        """
        # Extract decision rules
        decision_rules = self.persona.get('decision_rules', {})

        # Check if signal meets entry threshold
        entry_threshold = 0.3  # Can be extracted from decision rules
        if abs(signal) < entry_threshold:
            return None

        # Determine position direction
        direction = 1 if signal > 0 else -1

        # Calculate position size
        current_position = self.positions.get(ticker, 0)
        target_size = self._calculate_position_size(ticker, signal)

        # Calculate change needed
        quantity_change = direction * target_size - current_position

        # Only trade if change is significant
        if abs(quantity_change) < 10:  # Minimum trade size
            return None

        # Check risk constraints
        if not self._check_risk_constraints(ticker, quantity_change, market_state):
            return None

        return {
            'ticker': ticker,
            'quantity': int(quantity_change),
            'order_type': 'market',
            'signal': signal
        }

    def _calculate_position_size(self, ticker: str, signal: float) -> int:
        """
        Calculate target position size

        Args:
            ticker: Asset ticker
            signal: Signal strength

        Returns:
            Target number of shares
        """
        asset = self.model.market.get_asset(ticker)
        if not asset:
            return 0

        # Base position size from persona
        portfolio_value = self.portfolio_value
        base_allocation = portfolio_value * self.position_size

        # Scale by signal strength
        target_value = base_allocation * abs(signal)

        # Apply max position constraint
        max_value = portfolio_value * self.max_position_size
        target_value = min(target_value, max_value)

        # Convert to shares
        shares = int(target_value / asset.price)

        return shares

    def _check_risk_constraints(
        self,
        ticker: str,
        quantity: int,
        market_state: Dict
    ) -> bool:
        """
        Check if trade satisfies risk constraints

        Args:
            ticker: Asset ticker
            quantity: Trade quantity
            market_state: Market state

        Returns:
            True if trade is allowed
        """
        # Check maximum drawdown
        if self.max_drawdown > self.max_drawdown_tolerance:
            self.logger.debug(f"Max drawdown {self.max_drawdown:.2%} exceeded tolerance")
            return False

        # Check leverage constraint (simplified)
        asset = self.model.market.get_asset(ticker)
        if not asset:
            return False

        trade_value = abs(quantity * asset.price)
        if trade_value > self.cash * self.max_leverage:
            self.logger.debug(f"Trade exceeds leverage limit")
            return False

        return True

    def _execute_trade(self, decision: Dict):
        """
        Execute trading decision

        Args:
            decision: Trade decision dict
        """
        order = {
            'agent_id': self.unique_id,
            'ticker': decision['ticker'],
            'quantity': decision['quantity'],
            'order_type': decision.get('order_type', 'market'),
            'timestamp': self.model.current_day
        }

        # Submit to market
        self.model.market.submit_order(order)

        self.logger.debug(
            f"Order submitted: {decision['quantity']} {decision['ticker']} "
            f"(signal: {decision['signal']:.2f})"
        )

    def _parse_percentage(self, value) -> float:
        """Parse percentage string to decimal"""
        if isinstance(value, (int, float)):
            return float(value)

        # Parse string like "20%"
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(value))
        if match:
            val = float(match.group(1))
            return val / 100 if val > 1 else val

        return 0.20  # Default 20%

    def _belief_to_strength(self, belief: str) -> float:
        """Convert belief string to numeric strength"""
        belief_map = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1,
            'none': 0.0
        }
        return belief_map.get(str(belief).lower(), 0.4)

    def get_performance_history(self) -> Dict:
        """
        Get performance history for recalibration

        Returns:
            Performance metrics dict
        """
        if len(self.portfolio_history) < 2:
            return {
                'days_active': self.days_active,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'current_value': self.portfolio_value,
                'sharpe_ratio': 0.0,
                'current_allocation': self.get_allocation()
            }

        # Calculate returns
        initial_value = self.portfolio_history[0]
        current_value = self.portfolio_value
        total_return = ((current_value / initial_value) - 1) * 100

        # Calculate Sharpe ratio
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        return {
            'days_active': self.days_active,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown * 100,
            'current_value': current_value,
            'sharpe_ratio': sharpe,
            'current_allocation': self.get_allocation()
        }

    def get_allocation(self) -> Dict[str, float]:
        """
        Get current portfolio allocation

        Returns:
            Dict of {ticker: weight}
        """
        total_value = self.portfolio_value
        if total_value == 0:
            return {}

        allocation = {}
        for ticker, quantity in self.positions.items():
            asset = self.model.market.get_asset(ticker)
            if asset:
                position_value = quantity * asset.price
                allocation[ticker] = position_value / total_value

        return allocation

    def needs_recalibration(self) -> bool:
        """Check if agent needs recalibration"""
        # Check drawdown trigger
        recal_config = self.model.config['llm'].get('recalibration', {})
        triggers = recal_config.get('triggers', {})

        if self.max_drawdown > triggers.get('drawdown_threshold', 0.20):
            return True

        # Check quarterly reflection
        days_since_recal = self.days_active - self.last_recalibration_day
        if days_since_recal >= triggers.get('quarterly_reflection', 90):
            return True

        return False

    def apply_recalibration(self, updated_persona: Dict):
        """
        Apply recalibration updates to agent

        Args:
            updated_persona: Updated persona components from LLM
        """
        # Update beliefs
        if 'updated_beliefs' in updated_persona:
            beliefs = self.persona.get('beliefs', {})
            beliefs.update(updated_persona['updated_beliefs'])
            self.persona['beliefs'] = beliefs

        # Update risk constraints
        if 'updated_risk_constraints' in updated_persona:
            constraints = self.persona.get('risk_constraints', {})
            constraints.update(updated_persona['updated_risk_constraints'])
            self.persona['risk_constraints'] = constraints

        # Update decision rules
        if 'updated_decision_rules' in updated_persona:
            rules = self.persona.get('decision_rules', {})
            rules.update(updated_persona['updated_decision_rules'])
            self.persona['decision_rules'] = rules

        # Update market views
        if 'updated_market_views' in updated_persona:
            self.persona['market_views'] = updated_persona['updated_market_views']

        # Re-extract decision parameters
        self._extract_decision_parameters()

        self.last_recalibration_day = self.days_active
        self.logger.info(f"Recalibration applied on day {self.days_active}")
