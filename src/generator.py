"""
Response Generator for Hebrew Bible Q&A.
Generates grounded answers with citations.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

from .search import SearchResult, Verse


SYSTEM_PROMPT = """You are a scholarly assistant specializing in the Hebrew Bible (Tanakh). Your role is to answer questions about the Bible based solely on the text you retrieve.

IMPORTANT: Always respond in Hebrew.

SEARCH RULES:
1. Use the search_bible tool to find relevant verses
2. Use the read_verses tool to read the full context before answering
3. If your initial search returns no results, try alternative search terms, synonyms, or different search methods (semantic, keyword, hybrid) before concluding
4. For topics clearly outside the Hebrew Bible (e.g., ישו/Jesus, רמב"ם/Maimonides, תלמוד/Talmud, ברית חדשה/New Testament), you may conclude earlier that the text does not cover this topic - but still search at least once to be thorough
5. Only say "לא מצאתי מידע על כך בטקסט" after genuinely trying to find related content

CITATION FORMAT - USE FOOTNOTE STYLE:
6. Use numbered footnotes [1], [2], [3] for citations in the text
7. List all sources at the end under "───" separator
8. Consolidate related verses into single footnotes (e.g., verses 12-14 become one footnote)
9. Keep the main text clean and readable - citations only as small numbers
10. If you don't find supporting information, say: "לא מצאתי מידע על כך בטקסט"

QUOTE INTEGRITY:
11. When quoting verses, copy the EXACT text from the read_verses output
12. Never modify, paraphrase, or insert characters into quotes
13. If unsure of exact wording, re-read the verse before quoting

FACT VERIFICATION:
14. Verify facts against retrieved text before stating them
15. Double-check numbers, names, and directions (e.g., to/from Egypt, north/south)

RESPONSE FORMAT EXAMPLES:

Example 1 - Question: "מה קרה בחלום יעקב בבית אל?"
Answer:
יעקב יצא מבאר שבע בדרכו לחרן, ובלילה חלם חלום מיוחד[1]. בחלום ראה סולם מוצב ארצה וראשו מגיע השמימה, ומלאכי אלוהים עולים ויורדים בו[2]. ה' ניצב עליו והבטיח לו את הארץ ואת ריבוי זרעו[3].

כשהתעורר, יעקב הכריז "אכן יש ה' במקום הזה" והקים מצבה, וקרא למקום "בית אל"[4].

───
[1] בראשית כח:י-יא (עמוד 32)
[2] בראשית כח:יב (עמוד 32)
[3] בראשית כח:יג-יד (עמוד 32)
[4] בראשית כח:טז-יט (עמוד 33)

Example 2 - Question: "ספר לי על עקדת יצחק"
Answer:
האלוהים ניסה את אברהם וציווה עליו לקחת את יצחק בנו יחידו ולהעלותו לעולה בארץ המוריה[1]. אברהם השכים בבוקר, לקח את יצחק ושני נעריו, ויצאו לדרך[2].

ביום השלישי הגיעו למקום. אברהם בנה מזבח, עקד את יצחק והניחו על המזבח[3]. ברגע האחרון קרא מלאך ה' ועצר אותו: "אל תשלח ידך אל הנער"[4]. אברהם ראה איל אחוז בסבך בקרניו והקריב אותו תחת בנו[5].

───
[1] בראשית כב:א-ב (עמוד 25)
[2] בראשית כב:ג (עמוד 25)
[3] בראשית כב:ט (עמוד 25)
[4] בראשית כב:יא-יב (עמוד 26)
[5] בראשית כב:יג (עמוד 26)

Example 3 - Question: "איך דוד הביס את גלית?"
Answer:
גלית היה לוחם פלשתי ענק מגת, גבוה שש אמות וזרת, לבוש שריון כבד ומזוין בחנית וכידון[1]. במשך ארבעים יום חירף את מערכות ישראל ואיש לא העז לצאת נגדו[2].

דוד, שהגיע להביא צידה לאחיו, שמע את חרפות גלית והתנדב להילחם בו[3]. הוא סירב ללבוש את שריון שאול ויצא עם מקלו, חמישה חלוקי אבנים וקלע בידו[4].

בקרב עצמו, דוד רץ לקראת גלית, קלע אבן שפגעה במצחו, וגלית נפל ארצה[5]. דוד לקח את חרב גלית וכרת את ראשו[6].

───
[1] שמואל א יז:ד-ז (עמוד 156)
[2] שמואל א יז:טז (עמוד 157)
[3] שמואל א יז:כג-כו (עמוד 157)
[4] שמואל א יז:לח-מ (עמוד 158)
[5] שמואל א יז:מח-מט (עמוד 159)
[6] שמואל א יז:נא (עמוד 159)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_bible",
            "description": "Search for verses in the Bible by keywords or topic. Returns the most relevant verses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in Hebrew"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "description": "Search method: semantic (FAISS), keyword (BM25), or hybrid"
                    },
                    "book": {
                        "type": "string",
                        "description": "Book ID to filter by (e.g., bereshit, shemot)"
                    },
                    "chapter": {
                        "type": "integer",
                        "description": "Chapter number to filter by"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_verses",
            "description": "Read specific verses with surrounding context. Supports verse ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verse_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of verse IDs. Format: book-chapter-verse (e.g., bereshit-1-1) or range: book-chapter-start-end (e.g., bereshit-17-4-6 for verses 4-6)"
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of verses before/after for context",
                        "default": 2
                    }
                },
                "required": ["verse_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_chapter",
            "description": "Read an entire chapter",
            "parameters": {
                "type": "object",
                "properties": {
                    "book": {
                        "type": "string",
                        "description": "Book ID (e.g., bereshit)"
                    },
                    "chapter": {
                        "type": "integer",
                        "description": "Chapter number"
                    }
                },
                "required": ["book", "chapter"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_books",
            "description": "List all books in the Bible with chapter counts",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


class ResponseGenerator:
    """Generates LLM responses using Bible search tools."""

    def __init__(self, bible_search, model: str = "gpt-5-mini", log_dir: Optional[str] = None):
        """
        Initialize the response generator.

        Args:
            bible_search: BibleSearch instance
            model: OpenAI model to use
        """
        self.search = bible_search
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.search_log_path = self.log_dir / "search_results.jsonl"
            self.response_log_path = self.log_dir / "responses.jsonl"
        else:
            self.search_log_path = None
            self.response_log_path = None

    def _write_jsonl(self, path: Optional[Path], payload: dict) -> None:
        if not path:
            return
        try:
            with path.open("a", encoding="utf-8") as f:
                # JSONL requires a single line per record.
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to write log entry: {e}")
            print(payload)

    def _make_json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except Exception:
                pass
        return obj

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result as string."""
        if name == "search_bible":
            results = self.search.search(
                query=arguments["query"],
                method=arguments.get("method", "semantic"),
                book=arguments.get("book"),
                chapter=arguments.get("chapter"),
                top_k=10
            )
            self._write_jsonl(self.search_log_path, {
                "query": arguments.get("query"),
                "book": arguments.get("book"),
                "chapter": arguments.get("chapter"),
                "method": arguments.get("method", "semantic"),
                "results": [
                    {
                        "verse_id": r.verse_id,
                        "verse_ids": r.verse_ids,
                        "book": r.book,
                        "book_en": r.book_en,
                        "chapter": r.chapter,
                        "verse": r.verse,
                        "page": r.page,
                        "score": r.score,
                    }
                    for r in results
                ],
            })
            return self._format_search_results(results)

        elif name == "read_verses":
            verses = self.search.get_verses(
                verse_ids=arguments["verse_ids"],
                context=arguments.get("context", 2)
            )
            return self._format_verses(verses)

        elif name == "read_chapter":
            verses = self.search.get_chapter(
                book_id=arguments["book"],
                chapter=arguments["chapter"]
            )
            return self._format_verses(verses)

        elif name == "list_books":
            books = self.search.list_books()
            return self._format_books(books)

        return "כלי לא מוכר"

    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM."""
        if not results:
            return "לא נמצאו תוצאות"

        lines = ["תוצאות חיפוש:"]
        for r in results:
            lines.append(f"- {r.verse_id}: {r.book} {r.chapter}:{r.verse}")
            lines.append(f"  טקסט: {r.text}")
            lines.append(f"  עמוד: {r.page}, ציון: {r.score:.2f}")
            if r.verse_ids and r.verse_ids != [r.verse_id]:
                lines.append(f"  מזהי פסוקים: {', '.join(r.verse_ids)}")
            lines.append("")
        return "\n".join(lines)

    def _format_verses(self, verses: List[Verse]) -> str:
        """Format verses for LLM."""
        if not verses:
            return "לא נמצאו פסוקים"

        lines = ["פסוקים:"]
        for v in verses:
            lines.append(f"[{v.id}] {v.book} {v.chapter}:{v.verse} (עמוד {v.page})")
            lines.append(f"  {v.text}")
            lines.append("")
        return "\n".join(lines)

    def _format_books(self, books: List[dict]) -> str:
        """Format book list for LLM."""
        if not books:
            return "לא נמצאו ספרים"

        lines = ["רשימת ספרים:"]
        for b in books:
            lines.append(f"- {b['name_he']} ({b['name_en']}): {b['chapter_count']} פרקים, מזהה: {b['id']}")
        return "\n".join(lines)

    def generate(
        self,
        query: str,
        conversation_history: List[dict] = None,
        max_iterations: int = 10
    ) -> tuple[str, List[str]]:
        """
        Generate a response to the query using tools.

        Args:
            query: User query
            conversation_history: Previous messages
            max_iterations: Maximum tool-use iterations

        Returns:
            Tuple of (response_text, cited_verse_ids)
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": query})

        cited_sources = []
        has_made_tool_calls = False

        for iteration in range(max_iterations):
            is_last_iteration = (iteration == max_iterations - 1)

            # On final iteration, if we've been using tools, force synthesis
            if is_last_iteration and has_made_tool_calls:
                messages.append({
                    "role": "user",
                    "content": "Based on the information you gathered, provide a summary answer now. If you did not find relevant information, say 'לא מצאתי מידע על כך בטקסט'."
                })
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=None  # Disable tools to force text response
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )

            message = response.choices[0].message
            if len(messages) == 2:
                self._write_jsonl(self.response_log_path,{
                    "system": self._make_json_safe(messages[0])})
            self._write_jsonl(self.response_log_path,{
                "message": self._make_json_safe(messages[-1])})
            self._write_jsonl(self.response_log_path,{
                "response": self._make_json_safe(message)})

            # Check if we need to call tools
            if message.tool_calls:
                has_made_tool_calls = True
                messages.append(message)

                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Execute tool
                    result = self._execute_tool(function_name, arguments)

                    # Track sources from read operations
                    if function_name in ["read_verses", "read_chapter"]:
                        if "verse_ids" in arguments:
                            cited_sources.extend(arguments["verse_ids"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No more tool calls, return the response
                final_content = message.content if message.content else "לא הצלחתי לענות על השאלה"
                return final_content, cited_sources

        # Max iterations reached - should rarely happen now due to synthesis forcing
        final_content = message.content if message.content else "לא הצלחתי לענות על השאלה"
        return final_content, cited_sources

    def enhance_query(self, query: str, conversation_history: List[dict] = None) -> str:
        """
        Enhance a query using a small LLM to resolve pronouns and references
        from conversation history.
        
        This function detects Hebrew pronouns (הוא, היא, etc.) and contextual 
        references in the query, then uses conversation history to replace them 
        with specific entities or concepts.
        
        Args:
            query: Original user query in Hebrew
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Enhanced query with resolved pronouns and references
            
        Examples:
            Original: "מה הוא עשה אחר כך?"
            History: [{"role": "user", "content": "ספר לי על משה"}]
            Enhanced: "מה משה עשה אחר כך?"
            
            Original: "מה קרה בפרק הקודם?"
            History: Context shows Genesis chapter 5
            Enhanced: "מה קרה בבראשית פרק 4?"
        """
        # # If no pronouns or reference phrases, return original
        # pronouns = ["הוא", "היא", "הם", "הן", "אותו", "אותה", "לו", "לה", "ממנו", "ממנה", "עליו", "עליה"]
        # # Demonstrative pronouns and phrases that reference previously mentioned things
        # demonstratives = ["הזה", "הזאת", "האלה", "ההבטחה הזאת", "הדבר הזה", "המקום הזה", "הסיפור הזה"]
        # reference_phrases = ["פרק הקודם", "הפרק הקודם", "בפרק הקודם", "הפסוק הקודם", "מה שהזכרת", "אחר כך", "לפני כן", "שם", "באותו", "בהמשך"]
        
        # has_pronoun = any(pronoun in query for pronoun in pronouns)
        # has_demonstrative = any(demo in query for demo in demonstratives)
        # has_reference = any(phrase in query for phrase in reference_phrases)
        
        # if not has_pronoun and not has_demonstrative and not has_reference:
        #     return query
            
        # If no conversation history, can't enhance
        if not conversation_history or len(conversation_history) == 0:
            return query
            
        # Use small model to resolve references
        small_model = os.getenv("GPT_MODEL_SMALL", "gpt-4o-mini")
        
        # Build context from recent conversation
        context_messages = []
        
        for msg in conversation_history:
            context_messages.append(f"{msg['role']}: {msg['content']}")
        
        context_text = "\n".join(context_messages)
        
        enhancement_prompt = f"""You are helping to improve Bible queries by resolving Hebrew pronouns and references from context.

Conversation context:
{context_text}

Original query: {query}

Your task:
1. Identify Hebrew pronouns (הוא, היא, לו, עליו, etc.) in the query
2. Identify reference phrases (פרק הקודם, מה שהזכרת, אחר כך, etc.)
3. Based on the conversation context, replace pronouns with specific names (משה, אברהם, etc.)
4. Replace reference phrases with specific details (book name, chapter number)

Examples:
- "מה הוא עשה?" → "מה משה עשה?" (if Moses was discussed)
- "ומה אלוהים הבטיח לו?" → "ומה אלוהים הבטיח ליעקב?" (if Jacob was discussed)
- "מה קרה בפרק הקודם?" → "מה קרה בבראשית פרק 4?" (if Genesis chapter 5 was discussed)

CRITICAL: 
- You MUST respond in Hebrew only
- Return ONLY the enhanced query, nothing else
- NO explanations, NO English text, NO additional commentary
- If nothing needs enhancement, return the original query exactly as is

Enhanced query (in Hebrew):"""

        try:
            response = self.client.chat.completions.create(
                model=small_model,
                messages=[
                    {"role": "system", "content": "You are an expert in Hebrew language processing and pronoun resolution. You MUST always respond in Hebrew only, never in English."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.3
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # Validate that we got a reasonable response
            # Should be similar length and in Hebrew
            if enhanced and len(enhanced) > 0 and len(enhanced) < len(query) * 3:
                return enhanced
            else:
                return query
                
        except Exception as e:
            # If enhancement fails, return original query
            print(f"Query enhancement failed: {e}")
            return query
        

@dataclass
class ParsedFootnote:
    """A parsed footnote from the response."""
    number: int
    reference: str  # e.g., "בראשית כח:יב"
    page: Optional[str] = None  # e.g., "32"

@dataclass
class ParsedResponse:
    """A parsed response with separated body and footnotes."""
    body: str
    footnotes: List[ParsedFootnote]
    raw: str

    def has_valid_format(self) -> bool:
        """Check if response follows the expected footnote format."""
        # Valid if it has footnotes or explicitly states no info found
        if "לא מצאתי מידע" in self.body:
            return True
        return len(self.footnotes) > 0

    def get_footnote_numbers_in_body(self) -> List[int]:
        """Extract all footnote numbers referenced in the body."""
        return [int(m) for m in re.findall(r'\[(\d+)\]', self.body)]

    def validate_footnotes(self) -> Tuple[bool, List[str]]:
        """
        Validate that all footnotes in body have corresponding definitions.
        Returns (is_valid, list of issues).
        """
        issues = []
        body_refs = set(self.get_footnote_numbers_in_body())
        defined_refs = {fn.number for fn in self.footnotes}

        # Check for undefined references
        undefined = body_refs - defined_refs
        if undefined:
            issues.append(f"הפניות לא מוגדרות: {sorted(undefined)}")

        # Check for unused definitions
        unused = defined_refs - body_refs
        if unused:
            issues.append(f"הגדרות לא בשימוש: {sorted(unused)}")

        return len(issues) == 0, issues

def parse_response(response: str) -> ParsedResponse:
    """
    Parse a response in footnote format into structured parts.

    Args:
        response: Raw response text with footnote citations

    Returns:
        ParsedResponse with separated body and footnotes
    """
    if not response:
        return ParsedResponse(body="", footnotes=[], raw="")

    # Split on the separator (─── or ---)
    separator_pattern = r'\n[─\-]{3,}\n'
    parts = re.split(separator_pattern, response, maxsplit=1)

    body = parts[0].strip()
    footnotes: List[ParsedFootnote] = []

    if len(parts) > 1:
        footnotes_text = parts[1].strip()
        # Parse individual footnotes: [1] reference (page X)
        footnote_pattern = r'\[(\d+)\]\s*([^(\n]+?)(?:\s*\(עמוד\s*(\d+)\))?(?:\n|$)'
        for match in re.finditer(footnote_pattern, footnotes_text):
            footnotes.append(ParsedFootnote(
                number=int(match.group(1)),
                reference=match.group(2).strip(),
                page=match.group(3)
            ))

    return ParsedResponse(body=body, footnotes=footnotes, raw=response)

def format_footnotes_to_legacy(parsed: ParsedResponse) -> str:
    """
    Convert footnote format back to legacy inline citation format.
    Useful for backward compatibility with existing evaluation systems.

    Args:
        parsed: ParsedResponse object

    Returns:
        Text with inline citations in legacy format
    """
    if not parsed.footnotes:
        return parsed.body

    body = parsed.body
    # Build a mapping of footnote number to reference
    fn_map = {fn.number: fn for fn in parsed.footnotes}

    # Replace each [N] with inline citation
    def replace_footnote(match):
        num = int(match.group(1))
        if num in fn_map:
            fn = fn_map[num]
            if fn.page:
                return f"({fn.reference}, עמוד {fn.page})"
            return f"({fn.reference})"
        return match.group(0)

    body_with_inline = re.sub(r'\[(\d+)\]', replace_footnote, body)

    # Add sources section
    sources = "\n\nמקורות:"
    for fn in sorted(parsed.footnotes, key=lambda x: x.number):
        if fn.page:
            sources += f"\n- {fn.reference} (עמוד {fn.page})"
        else:
            sources += f"\n- {fn.reference}"

    return body_with_inline + sources
