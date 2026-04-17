"""
LLM Response Parser
Extracts structured data from LLM text responses
"""

import json
import re
from typing import Dict, Any, Optional


class ResponseParser:
    """
    Parse LLM responses into structured data

    Handles:
    - JSON extraction from text
    - Validation
    - Error recovery
    """

    @staticmethod
    def parse_json(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from LLM response

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def parse_calibration_response(response: str) -> Dict[str, Any]:
        """
        Parse agent calibration response

        Args:
            response: LLM response

        Returns:
            Structured persona dict
        """
        parsed = ResponseParser.parse_json(response)

        if parsed is None:
            # Return default persona if parsing fails
            return ResponseParser._default_persona()

        # Validate and fill missing fields
        validated = ResponseParser._validate_calibration(parsed)
        return validated

    @staticmethod
    def parse_recalibration_response(response: str) -> Dict[str, Any]:
        """
        Parse agent recalibration response

        Args:
            response: LLM response

        Returns:
            Structured update dict
        """
        parsed = ResponseParser.parse_json(response)

        if parsed is None:
            return {}

        # Validate recalibration response
        validated = ResponseParser._validate_recalibration(parsed)
        return validated

    @staticmethod
    def _validate_calibration(data: Dict) -> Dict[str, Any]:
        """
        Validate and fill missing calibration fields

        Args:
            data: Parsed JSON data

        Returns:
            Validated persona dict
        """
        # Required fields with defaults
        defaults = {
            'investment_philosophy': 'Balanced approach to risk and return',
            'risk_tolerance': 'moderate',
            'time_horizon': 'medium',
            'decision_rules': {
                'entry_trigger': 'Positive momentum or undervaluation',
                'exit_trigger': 'Target reached or 10% loss',
                'position_sizing': '5% of portfolio',
                'max_position': '10%'
            },
            'risk_constraints': {
                'max_drawdown_tolerance': '20%',
                'max_leverage': '1.0',
            },
            'beliefs': {
                'market_efficiency': 'medium',
                'mean_reversion': 'medium',
                'momentum_effects': 'medium'
            },
            'cognitive_biases': ['confirmation bias', 'anchoring'],
            'information_processing': {
                'reaction_speed': 'moderate',
                'contrarian_tendency': 50,
                'fundamental_vs_technical': 50
            },
            'initial_allocation': {
                'TECH': 0.33,
                'VALUE': 0.33,
                'SAFE': 0.34
            },
            'market_views': {
                'TECH': {'view': 'neutral', 'rationale': 'Awaiting more data'},
                'VALUE': {'view': 'neutral', 'rationale': 'Awaiting more data'},
                'SAFE': {'view': 'neutral', 'rationale': 'Awaiting more data'}
            }
        }

        # Merge with defaults
        result = defaults.copy()
        for key, value in data.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key].update(value)
            else:
                result[key] = value

        # Validate allocation sums to 1.0
        allocation = result['initial_allocation']
        total = sum(allocation.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            for key in allocation:
                allocation[key] /= total

        return result

    @staticmethod
    def _validate_recalibration(data: Dict) -> Dict[str, Any]:
        """
        Validate recalibration response

        Args:
            data: Parsed JSON data

        Returns:
            Validated update dict
        """
        # Recalibration can have partial updates
        # Just ensure types are correct
        validated = {}

        if 'reflection' in data:
            validated['reflection'] = str(data['reflection'])

        if 'updated_beliefs' in data and isinstance(data['updated_beliefs'], dict):
            validated['updated_beliefs'] = data['updated_beliefs']

        if 'updated_risk_constraints' in data and isinstance(data['updated_risk_constraints'], dict):
            validated['updated_risk_constraints'] = data['updated_risk_constraints']

        if 'updated_decision_rules' in data and isinstance(data['updated_decision_rules'], dict):
            validated['updated_decision_rules'] = data['updated_decision_rules']

        if 'updated_market_views' in data and isinstance(data['updated_market_views'], dict):
            validated['updated_market_views'] = data['updated_market_views']

        if 'confidence_adjustment' in data:
            validated['confidence_adjustment'] = data['confidence_adjustment']

        if 'recommended_allocation' in data and isinstance(data['recommended_allocation'], dict):
            allocation = data['recommended_allocation']
            total = sum(allocation.values())
            if abs(total - 1.0) > 0.01 and total > 0:
                # Normalize
                allocation = {k: v/total for k, v in allocation.items()}
            validated['recommended_allocation'] = allocation

        return validated

    @staticmethod
    def _default_persona() -> Dict[str, Any]:
        """
        Return default persona when parsing fails

        Returns:
            Default persona dict
        """
        return {
            'investment_philosophy': 'Balanced approach with moderate risk',
            'risk_tolerance': 'moderate',
            'time_horizon': 'medium',
            'decision_rules': {
                'entry_trigger': 'Positive technical signal',
                'exit_trigger': '10% gain or 5% loss',
                'position_sizing': '5% of portfolio',
                'max_position': '10%'
            },
            'risk_constraints': {
                'max_drawdown_tolerance': '15%',
                'max_leverage': '1.0'
            },
            'beliefs': {
                'market_efficiency': 'medium',
                'mean_reversion': 'medium',
                'momentum_effects': 'medium'
            },
            'cognitive_biases': ['recency bias'],
            'information_processing': {
                'reaction_speed': 'moderate',
                'contrarian_tendency': 50,
                'fundamental_vs_technical': 50
            },
            'initial_allocation': {
                'TECH': 0.33,
                'VALUE': 0.33,
                'SAFE': 0.34
            },
            'market_views': {
                'TECH': {'view': 'neutral', 'rationale': 'Default'},
                'VALUE': {'view': 'neutral', 'rationale': 'Default'},
                'SAFE': {'view': 'neutral', 'rationale': 'Default'}
            }
        }

    @staticmethod
    def extract_market_view(persona: Dict, ticker: str) -> str:
        """
        Extract market view for specific asset

        Args:
            persona: Agent persona dict
            ticker: Asset ticker

        Returns:
            'bullish', 'neutral', or 'bearish'
        """
        market_views = persona.get('market_views', {})
        view_data = market_views.get(ticker, {})

        if isinstance(view_data, dict):
            return view_data.get('view', 'neutral')
        else:
            return str(view_data) if view_data else 'neutral'

    @staticmethod
    def extract_position_size(persona: Dict) -> float:
        """
        Extract position sizing rule as percentage

        Args:
            persona: Agent persona dict

        Returns:
            Position size as decimal (e.g., 0.05 for 5%)
        """
        rules = persona.get('decision_rules', {})
        sizing = rules.get('position_sizing', '5%')

        # Parse percentage or fraction
        if isinstance(sizing, (int, float)):
            return float(sizing)

        # Parse string like "5%", "0.05", "5% of portfolio"
        match = re.search(r'(\d+(?:\.\d+)?)', str(sizing))
        if match:
            value = float(match.group(1))
            # If > 1, assume it's a percentage
            if value > 1:
                return value / 100
            return value

        return 0.05  # Default 5%

    @staticmethod
    def extract_max_position(persona: Dict) -> float:
        """
        Extract max position size

        Args:
            persona: Agent persona dict

        Returns:
            Max position as decimal
        """
        rules = persona.get('decision_rules', {})
        max_pos = rules.get('max_position', '10%')

        # Parse percentage
        match = re.search(r'(\d+(?:\.\d+)?)', str(max_pos))
        if match:
            value = float(match.group(1))
            if value > 1:
                return value / 100
            return value

        return 0.10  # Default 10%
