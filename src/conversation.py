"""
Conversation Manager for Hebrew Bible Q&A.
Maintains context across turns and resolves references.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class Entity:
    """A tracked entity (person, place, concept)."""
    name: str
    type: str  # "person", "place", "concept"
    aliases: List[str] = field(default_factory=list)
    first_mentioned: Optional[str] = None  # verse_id
    last_mentioned: Optional[str] = None  # verse_id


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)  # verse_ids cited


class ConversationState:
    """Maintains conversation context across turns."""

    def __init__(self):
        self.messages: List[Message] = []
        self.entities: Dict[str, Entity] = {}
        self.recent_sources: List[str] = []  # verse_ids, most recent first
        self.current_topic: Optional[str] = None
        self.current_book: Optional[str] = None
        self.current_chapter: Optional[int] = None

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str, sources: List[str] = None):
        """Add an assistant message to history."""
        sources = sources or []
        self.messages.append(Message(role="assistant", content=content, sources=sources))

        # Update recent sources
        for source in sources:
            if source in self.recent_sources:
                self.recent_sources.remove(source)
            self.recent_sources.insert(0, source)

        # Keep only last 50 sources
        self.recent_sources = self.recent_sources[:50]

    def track_entity(self, name: str, entity_type: str, verse_id: str = None):
        """Track a mentioned entity."""
        if name not in self.entities:
            self.entities[name] = Entity(
                name=name,
                type=entity_type,
                first_mentioned=verse_id
            )
        self.entities[name].last_mentioned = verse_id

    def get_recent_context(self, n_messages: int = 5) -> str:
        """Get summary of recent conversation context."""
        recent = self.messages[-n_messages:] if len(self.messages) > 0 else []

        context_parts = []

        # Recent messages
        if recent:
            context_parts.append("Recent conversation:")
            for msg in recent:
                prefix = "User" if msg.role == "user" else "Assistant"
                # Truncate long messages
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_parts.append(f"  {prefix}: {content}")

        # Active entities
        if self.entities:
            recent_entities = sorted(
                self.entities.values(),
                key=lambda e: e.last_mentioned or "",
                reverse=True
            )[:5]
            context_parts.append("\nRecently discussed entities:")
            for entity in recent_entities:
                context_parts.append(f"  - {entity.name} ({entity.type})")

        # Current focus
        if self.current_book:
            focus = f"\nCurrent focus: {self.current_book}"
            if self.current_chapter:
                focus += f" chapter {self.current_chapter}"
            context_parts.append(focus)

        return "\n".join(context_parts)

    def get_message_history(self) -> List[dict]:
        """Get message history in format suitable for LLM."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def update_focus(self, book: str = None, chapter: int = None, topic: str = None):
        """Update current focus of conversation."""
        if book:
            self.current_book = book
        if chapter:
            self.current_chapter = chapter
        if topic:
            self.current_topic = topic

    def get_sources_for_context(self, n: int = 5) -> List[str]:
        """Get most recent source verse IDs for context."""
        return self.recent_sources[:n]


class QueryEnhancer:
    """Enhances queries with conversation context."""

    # Hebrew pronouns and their possible referents
    PRONOUNS_HE = {
        "הוא": "male_singular",
        "היא": "female_singular",
        "הם": "male_plural",
        "הן": "female_plural",
        "אותו": "male_singular_object",
        "אותה": "female_singular_object",
        "לו": "male_singular_object",
        "לה": "female_singular_object",
        "ממנו": "male_singular",
        "ממנה": "female_singular",
        "עליו": "male_singular",
        "עליה": "female_singular",
    }

    # Reference phrases
    REFERENCE_PHRASES = {
        "הפרק הקודם": "previous_chapter",
        "פרק הקודם": "previous_chapter",
        "בפרק הקודם": "previous_chapter",
        "הפסוק הקודם": "previous_verse",
        "מה שהזכרת": "previous_mention",
        "אחר כך": "following",
        "לפני כן": "before",
    }

    def __init__(self, state: ConversationState):
        self.state = state

    def enhance_query(self, query: str) -> str:
        """
        Enhance a query with context from conversation history.
        Resolves pronouns and references.

        Args:
            query: Original user query

        Returns:
            Enhanced query with resolved references
        """
        enhanced = query

        # Check for pronouns that need resolution
        for pronoun, pronoun_type in self.PRONOUNS_HE.items():
            if pronoun in query:
                replacement = self._resolve_pronoun(pronoun_type)
                if replacement:
                    # Preserve prepositions with pronouns
                    if pronoun in ["לו", "לה"]:
                        enhanced = enhanced.replace(pronoun, f"ל{replacement}")
                    elif pronoun in ["ממנו", "ממנה"]:
                        enhanced = enhanced.replace(pronoun, f"מ{replacement}")
                    elif pronoun in ["עליו", "עליה"]:
                        enhanced = enhanced.replace(pronoun, f"על {replacement}")
                    else:
                        enhanced = enhanced.replace(pronoun, replacement)

        # Check for reference phrases
        for phrase, ref_type in self.REFERENCE_PHRASES.items():
            if phrase in query:
                replacement = self._resolve_reference(ref_type)
                if replacement:
                    enhanced = enhanced.replace(phrase, replacement)

        # Add context about current focus
        if self.state.current_book and self.state.current_book not in enhanced:
            # Only add if query seems to reference current context
            context_indicators = ["שם", "באותו", "בהמשך", "עוד"]
            if any(ind in query for ind in context_indicators):
                enhanced = f"{enhanced} (בהקשר של {self.state.current_book}"
                if self.state.current_chapter:
                    enhanced += f" פרק {self.state.current_chapter}"
                enhanced += ")"

        return enhanced

    def _resolve_pronoun(self, pronoun_type: str) -> Optional[str]:
        """Resolve a pronoun to its referent."""
        # Look for recently mentioned entities matching gender
        if "male" in pronoun_type:
            # Look for male entities
            for entity in self.state.entities.values():
                if entity.type == "person":
                    # Simple heuristic: most recently mentioned person
                    return entity.name

        if "female" in pronoun_type:
            # Look for female entities
            for entity in self.state.entities.values():
                if entity.type == "person":
                    return entity.name

        return None

    def _resolve_reference(self, ref_type: str) -> Optional[str]:
        """Resolve a reference phrase."""
        if ref_type == "previous_chapter":
            if self.state.current_book and self.state.current_chapter:
                if self.state.current_chapter > 1:
                    return f"{self.state.current_book} פרק {self.state.current_chapter - 1}"

        if ref_type == "previous_verse":
            if self.state.recent_sources:
                return f"הפסוק {self.state.recent_sources[0]}"

        if ref_type == "previous_mention":
            if self.state.recent_sources:
                return f"הפסוקים שהוזכרו קודם"

        return None