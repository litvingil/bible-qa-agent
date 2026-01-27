"""
Tests for Hebrew Bible Q&A Agent.
Tests the 5-message conversation scenario from requirements.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search import BibleSearch, SearchResult, Verse
from src.conversation import ConversationState, QueryEnhancer
from src.agent import BibleAgent


class TestConversationState(unittest.TestCase):
    """Test conversation state management."""

    def setUp(self):
        self.state = ConversationState()

    def test_add_messages(self):
        """Test adding messages to history."""
        self.state.add_user_message("שאלה ראשונה")
        self.state.add_assistant_message("תשובה ראשונה", ["bereshit-1-1"])

        self.assertEqual(len(self.state.messages), 2)
        self.assertEqual(self.state.messages[0].role, "user")
        self.assertEqual(self.state.messages[1].role, "assistant")

    def test_recent_sources(self):
        """Test tracking of recent sources."""
        self.state.add_assistant_message("תשובה", ["bereshit-1-1", "bereshit-1-2"])
        self.state.add_assistant_message("תשובה נוספת", ["bereshit-2-1"])

        sources = self.state.get_sources_for_context(3)
        self.assertEqual(sources[0], "bereshit-2-1")  # Most recent
        # Previous batch, last inserted first (stack order)
        self.assertIn("bereshit-1-1", sources)
        self.assertIn("bereshit-1-2", sources)

    def test_entity_tracking(self):
        """Test entity tracking."""
        self.state.track_entity("אברהם", "person", "bereshit-12-1")
        self.state.track_entity("שרה", "person", "bereshit-12-5")

        self.assertIn("אברהם", self.state.entities)
        self.assertEqual(self.state.entities["אברהם"].type, "person")

    def test_focus_update(self):
        """Test focus updates."""
        self.state.update_focus(book="בראשית", chapter=12)

        self.assertEqual(self.state.current_book, "בראשית")
        self.assertEqual(self.state.current_chapter, 12)


class TestQueryEnhancer(unittest.TestCase):
    """Test query enhancement with context."""

    def setUp(self):
        self.state = ConversationState()
        self.enhancer = QueryEnhancer(self.state)

    def test_pronoun_resolution(self):
        """Test pronoun resolution."""
        # Track an entity
        self.state.track_entity("אברהם", "person", "bereshit-12-1")

        query = "מה הוא עשה?"
        enhanced = self.enhancer.enhance_query(query)

        # Should resolve "הוא" to "אברהם"
        self.assertIn("אברהם", enhanced)

    def test_chapter_reference(self):
        """Test chapter reference resolution."""
        self.state.update_focus(book="בראשית", chapter=12)

        query = "מה קרה בפרק הקודם?"
        enhanced = self.enhancer.enhance_query(query)

        # Should mention chapter 11
        self.assertIn("11", enhanced)


class TestBibleSearch(unittest.TestCase):
    """Test Bible search functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up search instance if data exists."""
        data_dir = Path(__file__).parent.parent / "data" / "bible"
        if (data_dir / "index" / "verses.json").exists():
            cls.search = BibleSearch(str(data_dir))
            cls.has_data = True
        else:
            cls.search = None
            cls.has_data = False

    def test_load_data(self):
        """Test that data loads correctly."""
        if not self.has_data:
            self.skipTest("Bible data not extracted")

        self.assertGreater(len(self.search.verses), 0)

    def test_keyword_search(self):
        """Test keyword search."""
        if not self.has_data:
            self.skipTest("Bible data not extracted")

        results = self.search.search_keyword("בראשית ברא", top_k=5)
        self.assertGreater(len(results), 0)
        # First verse should be Genesis 1:1
        self.assertEqual(results[0].chapter, 1)
        self.assertEqual(results[0].verse, 1)

    def test_get_verse(self):
        """Test getting specific verse."""
        if not self.has_data:
            self.skipTest("Bible data not extracted")

        verse = self.search.get_verse("bereshit-1-1")
        if verse:
            self.assertEqual(verse.chapter, 1)
            self.assertEqual(verse.verse, 1)
            self.assertIn("בראשית", verse.text)

    def test_get_chapter(self):
        """Test getting entire chapter."""
        if not self.has_data:
            self.skipTest("Bible data not extracted")

        verses = self.search.get_chapter("bereshit", 1)
        self.assertGreater(len(verses), 0)
        # All verses should be from chapter 1
        for v in verses:
            self.assertEqual(v.chapter, 1)


class TestConversationScenario(unittest.TestCase):
    """
    Test the 5-message conversation scenario from requirements.
    """

    @classmethod
    def setUpClass(cls):
        """Set up agent if data and API key exist."""
        data_dir = Path(__file__).parent.parent / "data" / "bible"
        cls.has_data = (data_dir / "index" / "verses.json").exists()
        cls.has_api_key = os.getenv("OPENAI_API_KEY") is not None

    def test_scenario_setup(self):
        """Verify test can run."""
        if not self.has_data:
            self.skipTest("Bible data not extracted")
        if not self.has_api_key:
            self.skipTest("OPENAI_API_KEY not set")

    def test_full_conversation(self):
        """
        Test full 5-message conversation:
        1. מי זה אברהם לפי הטקסט?
        2. איפה מסופר שהוא יוצא לדרך?
        3. ומה אלוהים מבטיח לו שם?
        4. תצטט משפט קצר שמוכיח את זה
        5. השווה להבטחה אחרת שהזכרת
        """
        if not self.has_data or not self.has_api_key:
            self.skipTest("Missing data or API key")

        agent = BibleAgent()

        # Query 1: Who is Abraham?
        response1 = agent.query("מי זה אברהם לפי הטקסט?")
        self.assertIn("אברהם", response1)
        # Should cite Genesis
        self.assertTrue(
            "בראשית" in response1 or "Genesis" in response1,
            "Response should cite Genesis"
        )

        # Query 2: Where does he leave?
        response2 = agent.query("איפה מסופר שהוא יוצא לדרך?")
        # Should resolve "הוא" to Abraham and find לך לך
        self.assertTrue(
            "לך" in response2 or "12" in response2,
            "Response should mention לך לך or chapter 12"
        )

        # Query 3: What does God promise?
        response3 = agent.query("ומה אלוהים מבטיח לו שם?")
        # Should mention the promises
        self.assertTrue(
            "גוי" in response3 or "ברכה" in response3 or "ארץ" in response3,
            "Response should mention promises"
        )

        # Query 4: Quote a sentence proving this
        response4 = agent.query("תצטט משפט קצר שמוכיח את זה")
        # Should have exact quote with citation
        self.assertTrue(
            ":" in response4,  # Chapter:verse format
            "Response should include verse citation"
        )

        # Query 5: Compare to another promise
        response5 = agent.query("השווה להבטחה אחרת שהזכרת")
        # Should compare promises
        self.assertGreater(len(response5), 50, "Response should be substantive")


class TestHallucinationRejection(unittest.TestCase):
    """Test that agent rejects queries about non-existent content."""

    @classmethod
    def setUpClass(cls):
        """Set up agent if data and API key exist."""
        data_dir = Path(__file__).parent.parent / "data" / "bible"
        cls.has_data = (data_dir / "index" / "verses.json").exists()
        cls.has_api_key = os.getenv("OPENAI_API_KEY") is not None

    def test_hallucination_rejection(self):
        """Test that agent says it can't find non-existent content."""
        if not self.has_data or not self.has_api_key:
            self.skipTest("Missing data or API key")

        agent = BibleAgent()

        # Ask about something not in the Bible
        response = agent.query("מה כתוב על מלחמת העולם הראשונה?")

        # Should indicate no information found
        self.assertTrue(
            "לא מצאתי" in response or "אין" in response or "לא נמצא" in response,
            "Agent should indicate it couldn't find information"
        )


if __name__ == "__main__":
    unittest.main()
