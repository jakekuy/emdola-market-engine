"""
News and Information Events
Generate market-relevant news and information flows
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of market events"""
    EARNINGS = "earnings"
    MACRO = "macro"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    TECH_INNOVATION = "tech_innovation"
    MARKET_STRUCTURE = "market_structure"


class Sentiment(Enum):
    """Event sentiment"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class MarketEvent:
    """
    Represents a market event/news

    Contains:
    - Type and description
    - Affected assets
    - Sentiment
    - Magnitude
    """
    day: int
    event_type: EventType
    description: str
    affected_assets: List[str]  # Ticker symbols
    sentiment: Sentiment
    magnitude: float  # 0.0-1.0, how significant


class NewsGenerator:
    """
    Generates realistic market events and news flow

    Events affect agent decision-making and create
    heterogeneous interpretations
    """

    def __init__(self, config: Dict):
        """
        Initialize news generator

        Args:
            config: Configuration dict
        """
        self.config = config
        self.event_history = []

        # Event templates
        self.event_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict:
        """Initialize event templates"""
        return {
            EventType.EARNINGS: [
                {"desc": "Tech sector reports {sentiment} earnings", "assets": ["TECH"], "mag": 0.3},
                {"desc": "Financial sector earnings {sentiment} expectations", "assets": ["VALUE"], "mag": 0.3},
                {"desc": "Broad market earnings season shows {sentiment} results", "assets": ["TECH", "VALUE"], "mag": 0.4}
            ],
            EventType.MACRO: [
                {"desc": "Fed signals {sentiment} policy stance", "assets": ["TECH", "VALUE", "SAFE"], "mag": 0.7},
                {"desc": "GDP growth {sentiment} forecasts", "assets": ["TECH", "VALUE"], "mag": 0.5},
                {"desc": "Inflation data shows {sentiment} trend", "assets": ["SAFE", "VALUE"], "mag": 0.6},
                {"desc": "Employment report {sentiment} expectations", "assets": ["TECH", "VALUE"], "mag": 0.4}
            ],
            EventType.GEOPOLITICAL: [
                {"desc": "Geopolitical tensions {sentiment}", "assets": ["SAFE", "TECH"], "mag": 0.8},
                {"desc": "Trade negotiations show {sentiment} progress", "assets": ["TECH", "VALUE"], "mag": 0.5}
            ],
            EventType.REGULATORY: [
                {"desc": "New tech regulations appear {sentiment}", "assets": ["TECH"], "mag": 0.6},
                {"desc": "Financial sector regulation {sentiment}", "assets": ["VALUE"], "mag": 0.5}
            ],
            EventType.TECH_INNOVATION: [
                {"desc": "Major tech breakthrough {sentiment} sector outlook", "assets": ["TECH"], "mag": 0.7},
                {"desc": "AI developments create {sentiment} sentiment", "assets": ["TECH"], "mag": 0.6}
            ]
        }

    def generate_event(self, day: int, force_type: Optional[EventType] = None) -> Optional[MarketEvent]:
        """
        Generate a random market event

        Args:
            day: Current day
            force_type: Force specific event type (optional)

        Returns:
            MarketEvent or None
        """
        # Check if event should occur
        news_freq = self.config['events']['news_frequency']
        if not force_type and random.random() > (1.0 / news_freq):
            return None

        # Choose event type
        if force_type:
            event_type = force_type
        else:
            event_type = random.choice(list(EventType))

        # Get template
        templates = self.event_templates.get(event_type, [])
        if not templates:
            return None

        template = random.choice(templates)

        # Generate sentiment
        sentiment = self._random_sentiment()

        # Format description
        sentiment_word = self._sentiment_to_word(sentiment)
        description = template["desc"].format(sentiment=sentiment_word)

        # Create event
        event = MarketEvent(
            day=day,
            event_type=event_type,
            description=description,
            affected_assets=template["assets"],
            sentiment=sentiment,
            magnitude=template["mag"]
        )

        self.event_history.append(event)
        return event

    def generate_shock(
        self,
        day: int,
        shock_type: str,
        magnitude: float = 0.9
    ) -> MarketEvent:
        """
        Generate a major market shock event

        Args:
            day: Current day
            shock_type: Type of shock
            magnitude: Severity (0.0-1.0)

        Returns:
            MarketEvent
        """
        shock_templates = {
            'flash_crash': {
                'desc': 'Flash crash: Sudden liquidity crisis triggers panic selling',
                'assets': ['TECH', 'VALUE', 'SAFE'],
                'sentiment': Sentiment.VERY_NEGATIVE,
                'type': EventType.MARKET_STRUCTURE
            },
            'credit_crisis': {
                'desc': 'Credit markets freeze: Major financial institution distress',
                'assets': ['VALUE', 'SAFE'],
                'sentiment': Sentiment.VERY_NEGATIVE,
                'type': EventType.MACRO
            },
            'tech_bubble': {
                'desc': 'Tech sector bubble concerns: Valuations questioned',
                'assets': ['TECH'],
                'sentiment': Sentiment.VERY_NEGATIVE,
                'type': EventType.MARKET_STRUCTURE
            },
            'policy_surprise': {
                'desc': 'Unexpected policy intervention: Central bank emergency action',
                'assets': ['TECH', 'VALUE', 'SAFE'],
                'sentiment': Sentiment.VERY_POSITIVE,
                'type': EventType.MACRO
            }
        }

        template = shock_templates.get(shock_type, shock_templates['flash_crash'])

        event = MarketEvent(
            day=day,
            event_type=template['type'],
            description=template['desc'],
            affected_assets=template['assets'],
            sentiment=template['sentiment'],
            magnitude=magnitude
        )

        self.event_history.append(event)
        return event

    def _random_sentiment(self) -> Sentiment:
        """Generate random sentiment with normal distribution bias"""
        # Normal distribution around neutral
        val = random.gauss(0, 1)

        if val > 1.5:
            return Sentiment.VERY_POSITIVE
        elif val > 0.5:
            return Sentiment.POSITIVE
        elif val > -0.5:
            return Sentiment.NEUTRAL
        elif val > -1.5:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.VERY_NEGATIVE

    def _sentiment_to_word(self, sentiment: Sentiment) -> str:
        """Convert sentiment enum to descriptive word"""
        mapping = {
            Sentiment.VERY_POSITIVE: "exceed",
            Sentiment.POSITIVE: "beat",
            Sentiment.NEUTRAL: "meet",
            Sentiment.NEGATIVE: "miss",
            Sentiment.VERY_NEGATIVE: "disappoint"
        }
        return mapping.get(sentiment, "affect")

    def get_recent_events(self, lookback_days: int = 30) -> List[MarketEvent]:
        """
        Get recent events

        Args:
            lookback_days: Number of days to look back

        Returns:
            List of recent events
        """
        if not self.event_history:
            return []

        current_day = self.event_history[-1].day
        cutoff = current_day - lookback_days

        return [e for e in self.event_history if e.day >= cutoff]

    def get_event_summary(self, event: MarketEvent) -> str:
        """
        Get formatted event summary

        Args:
            event: MarketEvent

        Returns:
            Formatted string
        """
        assets_str = ", ".join(event.affected_assets)
        return f"Day {event.day}: {event.description} (affects: {assets_str}, magnitude: {event.magnitude:.1f})"
