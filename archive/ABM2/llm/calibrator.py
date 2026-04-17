"""
LLM Agent Calibrator
Orchestrates calibration and recalibration of agent personas
"""

from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm

from llm.interface import LLMInterface
from llm.prompts import PromptTemplates, AgentArchetypes
from llm.parser import ResponseParser


class AgentCalibrator:
    """
    Orchestrates LLM-based agent calibration

    Two main operations:
    1. Initial calibration: Generate all agent personas
    2. Recalibration: Update agents based on experience
    """

    def __init__(self, llm_interface: LLMInterface, config: Dict):
        """
        Initialize calibrator

        Args:
            llm_interface: LLM interface instance
            config: Configuration dict
        """
        self.llm = llm_interface
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Calibration history
        self.calibration_log = []
        self.recalibration_log = []

    def calibrate_agent(
        self,
        agent_type: str,
        capital: float,
        market_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibrate single agent using LLM

        Args:
            agent_type: Agent archetype (e.g., 'pension_fund')
            capital: Initial capital allocation
            market_info: Current market context

        Returns:
            Agent persona dict
        """
        # Get archetype description
        archetype_desc = AgentArchetypes.get_description(agent_type)

        # Generate calibration prompt
        system_prompt = PromptTemplates.get_calibration_system_prompt()
        user_prompt = PromptTemplates.get_calibration_prompt(
            agent_type=agent_type,
            capital=capital,
            market_info=market_info,
            archetype_description=archetype_desc
        )

        # Call LLM
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )

            # Parse response
            persona = ResponseParser.parse_calibration_response(response)

            # Log calibration
            self.calibration_log.append({
                'agent_type': agent_type,
                'capital': capital,
                'persona': persona,
                'raw_response': response
            })

            self.logger.info(f"Calibrated {agent_type} with capital ${capital:,.0f}")
            return persona

        except Exception as e:
            self.logger.error(f"Calibration failed for {agent_type}: {e}")
            # Return default persona
            return ResponseParser._default_persona()

    def calibrate_all_agents(
        self,
        agent_specs: List[Dict],
        market_info: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """
        Calibrate all agents in batch

        Args:
            agent_specs: List of agent specifications
                Each spec: {'agent_type': str, 'capital': float, 'count': int}
            market_info: Current market context

        Returns:
            Dict mapping agent_type to list of personas
        """
        personas_by_type = {}

        self.logger.info(f"Starting calibration of {len(agent_specs)} agent types...")

        for spec in tqdm(agent_specs, desc="Calibrating agents"):
            agent_type = spec['agent_type']
            capital = spec['capital']
            count = spec['count']

            personas = []

            # Generate personas for this type
            for i in range(count):
                self.logger.debug(f"Calibrating {agent_type} {i+1}/{count}")

                persona = self.calibrate_agent(
                    agent_type=agent_type,
                    capital=capital,
                    market_info=market_info
                )

                personas.append(persona)

            personas_by_type[agent_type] = personas
            self.logger.info(f"Completed {count} {agent_type} agents")

        self.logger.info(f"Calibration complete: {sum(len(p) for p in personas_by_type.values())} agents")
        return personas_by_type

    def recalibrate_agent(
        self,
        agent_type: str,
        original_persona: Dict,
        performance_history: Dict,
        market_history: Dict,
        recent_events: List
    ) -> Dict[str, Any]:
        """
        Recalibrate agent based on experience

        Args:
            agent_type: Agent archetype
            original_persona: Original LLM-generated persona
            performance_history: Agent's performance metrics
            market_history: Market evolution data
            recent_events: Recent market events

        Returns:
            Updated persona components
        """
        # Generate recalibration prompt
        prompt = PromptTemplates.get_recalibration_prompt(
            agent_type=agent_type,
            original_persona=original_persona,
            performance_history=performance_history,
            market_history=market_history,
            recent_events=recent_events
        )

        system_prompt = PromptTemplates.get_calibration_system_prompt()

        # Call LLM
        try:
            response = self.llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )

            # Parse recalibration response
            updates = ResponseParser.parse_recalibration_response(response)

            # Log recalibration
            self.recalibration_log.append({
                'agent_type': agent_type,
                'day': performance_history.get('days_active', 0),
                'updates': updates,
                'raw_response': response
            })

            self.logger.info(f"Recalibrated {agent_type} on day {performance_history.get('days_active', 0)}")
            return updates

        except Exception as e:
            self.logger.error(f"Recalibration failed for {agent_type}: {e}")
            return {}

    def batch_recalibrate(
        self,
        agents_to_recalibrate: List[tuple]
    ) -> Dict[str, Dict]:
        """
        Recalibrate multiple agents in batch

        Args:
            agents_to_recalibrate: List of (agent, performance, market, events) tuples

        Returns:
            Dict mapping agent_id to updated persona
        """
        updates = {}

        self.logger.info(f"Recalibrating {len(agents_to_recalibrate)} agents...")

        for agent, performance, market, events in tqdm(agents_to_recalibrate, desc="Recalibrating"):
            updated_persona = self.recalibrate_agent(
                agent_type=agent.agent_type,
                original_persona=agent.persona,
                performance_history=performance,
                market_history=market,
                recent_events=events
            )

            updates[agent.unique_id] = updated_persona

        return updates

    def get_calibration_summary(self) -> Dict:
        """
        Get summary of calibration activity

        Returns:
            Summary statistics dict
        """
        return {
            'total_calibrations': len(self.calibration_log),
            'total_recalibrations': len(self.recalibration_log),
            'agents_by_type': self._count_by_type(self.calibration_log),
            'recalibrations_by_day': self._group_recalibrations_by_day()
        }

    def _count_by_type(self, log: List[Dict]) -> Dict[str, int]:
        """Count calibrations by agent type"""
        counts = {}
        for entry in log:
            agent_type = entry['agent_type']
            counts[agent_type] = counts.get(agent_type, 0) + 1
        return counts

    def _group_recalibrations_by_day(self) -> Dict[int, int]:
        """Group recalibrations by simulation day"""
        by_day = {}
        for entry in self.recalibration_log:
            day = entry['day']
            by_day[day] = by_day.get(day, 0) + 1
        return by_day
