"""
Trading Scheduler
Manages agent activation order and execution
"""

import random
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MarketModel


class TradingScheduler:
    """
    Scheduler for agent trading decisions

    Agents activate in random order each day to avoid systematic bias
    """

    def __init__(self, model: 'MarketModel'):
        """
        Initialize scheduler

        Args:
            model: Parent market model
        """
        self.model = model
        self.agents = []  # Will be populated during setup
        self.steps_executed = 0

    def add_agent(self, agent):
        """
        Add agent to scheduler

        Args:
            agent: Agent instance to schedule
        """
        self.agents.append(agent)

    def remove_agent(self, agent):
        """
        Remove agent from scheduler

        Args:
            agent: Agent instance to remove
        """
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self):
        """
        Execute one time step - activate all agents in random order
        """
        # Shuffle agents to avoid order bias
        agent_order = self.agents.copy()
        random.shuffle(agent_order)

        # Activate each agent
        for agent in agent_order:
            if agent.active:  # Only activate active agents
                agent.step()

        self.steps_executed += 1

    def get_agents_by_type(self, agent_type: str) -> List:
        """
        Get all agents of specific type

        Args:
            agent_type: Type identifier (e.g., 'pension_fund')

        Returns:
            List of agents matching type
        """
        return [a for a in self.agents if a.agent_type == agent_type]

    def get_active_agents(self) -> List:
        """
        Get all currently active agents

        Returns:
            List of active agents
        """
        return [a for a in self.agents if a.active]

    @property
    def agent_count(self) -> int:
        """Total number of agents"""
        return len(self.agents)

    @property
    def active_agent_count(self) -> int:
        """Number of currently active agents"""
        return len(self.get_active_agents())
