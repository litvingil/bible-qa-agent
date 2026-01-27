"""
Main Agent for Hebrew Bible Q&A.
Orchestrates search, conversation, and response generation.
"""

from typing import List, Optional

from .search import BibleSearch, SearchResult, Verse
from .conversation import (
    ConversationState,
    QueryEnhancer,
)
from .generator import ResponseGenerator


class BibleAgent:
    """
    Agent that answers questions about the Hebrew Bible.
    Uses ReAct pattern with search and read tools.
    """

    def __init__(self, data_dir: str = None, model: str = "gpt-5-mini", log_dir: Optional[str] = None):
        """
        Initialize the Bible agent.

        Args:
            data_dir: Path to Bible data directory
            model: OpenAI model to use
        """
        self.search = BibleSearch(data_dir)
        self.state = ConversationState()
        self.enhancer = QueryEnhancer(self.state)
        self.generator = ResponseGenerator(self.search, model, log_dir=log_dir)

    def reset(self):
        """Reset conversation state."""
        self.state = ConversationState()
        self.enhancer = QueryEnhancer(self.state)

    def query(self, user_query: str) -> str:
        """
        Process a user query and return a response.

        Args:
            user_query: The user's question in Hebrew

        Returns:
            Response text with citations
        """
        # Add to history
        self.state.add_user_message(user_query)

        # Get conversation history for LLM
        history = self.state.get_message_history()[:-1]  # Exclude current message

        # Enhance query with context using LLM-based enhancement
        enhanced_query = self.generator.enhance_query(user_query, conversation_history=history)
        print(f"Enhanced query: {enhanced_query}")

        # Generate response
        response, cited_sources = self.generator.generate(
            enhanced_query,
            conversation_history=history
        )

        # Update conversation state
        self.state.add_assistant_message(response, cited_sources)

        return response

    # Alias for convenience
    def chat(self, user_query: str) -> str:
        """Alias for query method."""
        return self.query(user_query)
    # Direct tool access methods for programmatic use
    def search_bible(
        self,
        query: str,
        book: str = None,
        chapter: int = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for verses matching the query.

        Args:
            query: Search query in Hebrew
            book: Optional book ID to filter by
            chapter: Optional chapter number to filter by
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        return self.search.search(query, book=book, chapter=chapter, top_k=top_k)

    def read_verses(
        self,
        verse_ids: List[str],
        context: int = 2
    ) -> List[Verse]:
        """
        Read specific verses with context.

        Args:
            verse_ids: List of verse IDs
            context: Number of surrounding verses

        Returns:
            List of Verse objects
        """
        return self.search.get_verses(verse_ids, context)

    def read_chapter(self, book: str, chapter: int) -> List[Verse]:
        """
        Read an entire chapter.

        Args:
            book: Book ID
            chapter: Chapter number

        Returns:
            List of Verse objects
        """
        return self.search.get_chapter(book, chapter)

    def list_books(self) -> List[dict]:
        """
        List all available books.

        Returns:
            List of book info dictionaries
        """
        return self.search.list_books()

    def get_context_summary(self) -> str:
        """Get summary of current conversation context."""
        return self.state.get_recent_context()

def create_agent(data_dir: str = None, model: str = "gpt-5-mini", log_dir: Optional[str] = None) -> BibleAgent:
    """
    Factory function to create a Bible agent.

    Args:
        data_dir: Path to Bible data directory
        model: OpenAI model to use

    Returns:
        Configured BibleAgent instance
    """
    return BibleAgent(data_dir, model, log_dir=log_dir)
