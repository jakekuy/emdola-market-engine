"""
LLM Interface
Provider-agnostic wrapper for LLM API calls
"""

from typing import Dict, List, Optional, Any
import os
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """
    Abstract interface for LLM providers

    Supports: Anthropic (Claude), OpenAI (GPT), local models
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text from LLM

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum response length

        Returns:
            Generated text
        """
        pass


class ClaudeInterface(LLMInterface):
    """Anthropic Claude interface"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude interface

        Args:
            api_key: Anthropic API key (or from env)
            model: Model identifier
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Anthropic API key required")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text using Claude"""

        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else "",
            messages=messages
        )

        return response.content[0].text


class GPTInterface(LLMInterface):
    """OpenAI GPT interface"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize GPT interface

        Args:
            api_key: OpenAI API key (or from env)
            model: Model identifier
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text using GPT"""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content


class MockLLMInterface(LLMInterface):
    """
    Mock LLM for testing (no API calls)

    Returns structured placeholder responses
    """

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate mock response"""

        # Simple pattern matching for different prompts
        if "calibrat" in prompt.lower():
            return self._mock_calibration_response()
        elif "recalibrat" in prompt.lower():
            return self._mock_recalibration_response()
        else:
            return "Mock LLM response: understood."

    def _mock_calibration_response(self) -> str:
        """Mock initial calibration response"""
        return """
{
  "investment_philosophy": "Value-oriented with focus on fundamental analysis",
  "risk_tolerance": "moderate",
  "time_horizon": "long-term (5+ years)",
  "decision_rules": {
    "entry_trigger": "Price 20% below intrinsic value estimate",
    "exit_trigger": "Price reaches fair value or 15% loss",
    "position_sizing": "Kelly criterion with 50% fraction",
    "max_position": "10% of portfolio"
  },
  "beliefs": {
    "market_efficiency": "semi-strong form",
    "mean_reversion": "high confidence",
    "momentum": "low confidence"
  },
  "biases": ["anchoring", "confirmation bias"],
  "initial_allocation": {
    "TECH": 0.30,
    "VALUE": 0.50,
    "SAFE": 0.20
  }
}
"""

    def _mock_recalibration_response(self) -> str:
        """Mock recalibration response"""
        return """
{
  "updated_beliefs": {
    "market_efficiency": "weak form",
    "volatility_regime": "high"
  },
  "adjusted_risk_tolerance": "conservative",
  "lessons_learned": "Recent drawdown suggests need for better risk management",
  "updated_decision_rules": {
    "position_sizing": "Reduce to 5% max per position",
    "stop_loss": "Tighten to 10%"
  }
}
"""


def create_llm_interface(config: Dict) -> LLMInterface:
    """
    Factory function to create appropriate LLM interface

    Args:
        config: Configuration dict with provider and model info

    Returns:
        LLM interface instance
    """
    provider = config.get('provider', 'mock').lower()
    model = config.get('model', '')

    if provider == 'anthropic' or provider == 'claude':
        return ClaudeInterface(model=model)
    elif provider == 'openai' or provider == 'gpt':
        return GPTInterface(model=model)
    elif provider == 'mock':
        return MockLLMInterface()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Async support for batch processing
class AsyncLLMInterface:
    """
    Async wrapper for LLM calls (for batch calibration)

    Allows parallel LLM calls to calibrate multiple agents simultaneously
    """

    def __init__(self, base_interface: LLMInterface):
        """
        Initialize async interface

        Args:
            base_interface: Underlying LLM interface
        """
        self.base_interface = base_interface

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Async generate (wraps sync call)

        For true async, would need to use provider's async client

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max response length

        Returns:
            Generated text
        """
        # For now, just wrap sync call
        # TODO: Implement true async using provider async clients
        return self.base_interface.generate(prompt, system, temperature, max_tokens)

    async def batch_generate(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> List[str]:
        """
        Generate multiple responses in parallel

        Args:
            prompts: List of prompts
            system: System prompt (same for all)
            temperature: Sampling temperature
            max_tokens: Max response length

        Returns:
            List of generated texts
        """
        import asyncio

        tasks = [
            self.generate_async(p, system, temperature, max_tokens)
            for p in prompts
        ]

        results = await asyncio.gather(*tasks)
        return results
