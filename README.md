# Hebrew Bible Q&A Agent

A Claude Code-style agent that answers questions about the Hebrew Bible by searching and reading preprocessed text files.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      User Question                              │
├────────────────────────────────────────────────────────────────┤
│                   Conversation Manager                          │
│   - Resolves references ("he", "that chapter")                  │
│   - Enhances query with context from history                    │
│   - Tracks entities and recent sources                          │
├────────────────────────────────────────────────────────────────┤
│                      Agent Loop (ReAct)                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────────┐     │
│   │ Search   │    │ Read     │    │ Answer               │     │
│   │ Tool     │    │ Tool     │    │ (grounded response)  │     │
│   └──────────┘    └──────────┘    └──────────────────────┘     │
├────────────────────────────────────────────────────────────────┤
│              Preprocessed Bible Text (JSON files)               │
│   - Structured by book/chapter                                  │
│   - Verse-level granularity with metadata                       │
│   - Full-text search index (keyword + semantic)                 │
└────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file or export:

```bash
export OPENAI_API_KEY=your-api-key-here
```

### 3. Extract Bible Data

Place your `hebrew_bible.pdf` in the project root, then run:

```bash
python -m scripts.extract_pdf
```

This creates structured JSON files in `data/bible/`.

### 4. (Optional) Build Semantic Search Index

For better conceptual search:

```bash
python -c "from src.search import build_embeddings; build_embeddings()"
```

## Usage

### Interactive Mode

```bash
python -m src.main
```

Example:
```
שאלה> מי זה אברהם לפי הטקסט?

לפי הטקסט בספר בראשית, אברהם הוא בנו של תרח...

מקורות:
- בראשית יא:כו (עמוד 14)
- בראשית יב:א (עמוד 15)
```

### Single Query Mode

```bash
python -m src.main -q "מי ברא את העולם?"
```

### Commands

- `/reset` - Start new conversation
- `/context` - Show current context
- `/books` - List available books
- `/quit` - Exit

## Project Structure

```
bible-qa-agent/
├── hebrew_bible.pdf           # Source PDF
├── data/
│   └── bible/                 # Preprocessed text
│       ├── metadata.json
│       ├── books/
│       └── index/
├── src/
│   ├── main.py               # CLI entry point
│   ├── agent.py              # ReAct agent loop
│   ├── search.py             # Search tools (BM25 + semantic)
│   ├── conversation.py       # Context management
│   └── generator.py          # LLM response generation
├── scripts/
│   └── extract_pdf.py        # One-time preprocessing
├── tests/
│   └── test_conversation.py  # Test scenarios
└── examples/
    └── conversation.md       # Sample conversation
```

## Agent Tools

The agent has access to these tools:

1. **search_bible** - Search for verses by keyword or concept
2. **read_verses** - Read specific verses with surrounding context
3. **read_chapter** - Read an entire chapter
4. **list_books** - List all available books

## Testing

Run tests:

```bash
python -m pytest tests/
```

Test the conversation scenario:

```bash
python -m pytest tests/test_conversation.py::TestConversationScenario -v
```

## Features

- **Grounded Responses**: All answers cite specific verses and page numbers
- **Context Tracking**: Maintains conversation history and resolves pronouns
- **Hybrid Search**: Combines BM25 keyword search with semantic embeddings
- **Hebrew Support**: Properly handles Hebrew text, nikud, and cantillation marks
- **Hallucination Prevention**: Agent refuses to answer about non-existent content

## Benchmark Results

The agent was tested with multi-turn conversation flows covering various biblical topics.

<details>
<summary>Click to expand benchmark results</summary>

### Test Flows

#### Flow 1: Abraham's Story (7 turns)
Multi-turn conversation demonstrating context tracking and follow-up questions.

| # | Question | Context Resolution |
|---|----------|-------------------|
| 1 | מי זה אברהם? מה הסיפור שלו? | Initial question |
| 2 | רגע, אז הוא התחיל בשם אחר? | Follow-up on name |
| 3 | למה השם השתנה? מה הרקע? | Context: Abraham's name |
| 4 | ואיפה כל זה קורה? הוא מסתובב הרבה | Pronoun "הוא" → Abraham |
| 5 | ההבטחה על הארץ - תראה לי את הניסוח המקורי | Reference to previous topic |
| 6 | יש עוד מישהו שמקבל הבטחה דומה? | Comparative question |
| 7 | תשווה את ההבטחות | Reference to multiple mentions |

#### Flow 8: Noah's Ark (10 turns)
Testing minimal context queries with pronoun resolution.

| # | Question | Context Resolution |
|---|----------|-------------------|
| 1 | מי בנה את התיבה? | Initial question |
| 2 | למה? | Implicit: "Why did Noah build the ark?" |
| 3 | מי נכנס? | Implicit: "Who entered the ark?" |
| 4 | כמה זמן? | Implicit: "How long in the ark?" |
| 5 | ואז? | Follow-up: "And then what?" |
| 6 | מה הוא עשה כשיצא? | Pronoun "הוא" → Noah |
| 7 | ומה אלוהים אמר? | Context: after leaving ark |
| 8 | זאת הברית? | Reference to covenant |
| 9 | מה הסימן? | Implicit: sign of covenant |
| 10 | איפה כתוב? תצטט | Request for citation |

#### Flow 9: Ruth, Esther & Deborah (8 turns)
Cross-referencing multiple biblical figures.

| # | Question | Context Resolution |
|---|----------|-------------------|
| 1 | ספרי לי על רות | Initial question |
| 2 | למה היא מיוחדת? היא לא ישראלית | Pronoun "היא" → Ruth |
| 3 | יש עוד נשים לא-ישראליות חשובות? | Comparative question |
| 4 | מעניין. ומה עם אסתר? | New figure introduction |
| 5 | היא גם במצב של 'זרה' בצורה מסוימת? | Pronoun → Esther |
| 6 | מה המשותף ביניהן? | Reference: Ruth & Esther |
| 7 | ודבורה? היא שונה מהן? | New comparison |
| 8 | תביא ציטוט שמאפיין כל אחת מהשלוש | Reference to all three |

### Response Quality

All responses included:
- Source citations with book, chapter, verse, and page numbers
- Hebrew quotations from the biblical text
- Proper context resolution for pronouns and references
- Explicit refusal when information not found in source

</details>
