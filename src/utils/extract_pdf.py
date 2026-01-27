#!/usr/bin/env python3
"""Parse a Hebrew Bible PDF into verse records.

What it does
------------
1) Extracts text from a PDF.
2) Fixes common extraction direction issues (many Hebrew PDFs extract each line
   reversed). We reverse each line and swap bracket directions.
3) Detects chapter headers like: "בראשית - פרק א'".
4) Detects verse starts at the beginning of lines, e.g.:
      א'וַיֹּאמֶר ...
      י"א וַיֹּאמֶר ...
   and groups continuation lines into the same verse.

Output
------
Writes JSONL or CSV with fields:
  book, chapter, verse, text

Notes
-----
- Hebrew PDFs often extract with:
  - Reversed lines (visual RTL order rather than logical order)
  - Excess spaces inserted *inside* words (letters/syllables split apart)
  - Missing glyphs that show up as NUL ("\x00") placeholders

  This script can fix all three:
  - `--direction auto|reverse|none` reverses lines when needed.
  - `--x-tolerance` controls pdfplumber spacing heuristics (higher usually means
    fewer spurious spaces inside words).
  - By default, the script removes niqqud/cantillation and replaces common NUL
    placeholders (see `_fix_null_placeholders`).

- Parashah markers like {פ} and {ס} are removed from verse text.

Example
-------
  python parse_hebrew_bible_pdf.py input.pdf -o verses.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from tqdm import tqdm

from .types import Verse


# Swap bracket directions after reversing a line.
_BRACKET_SWAP = str.maketrans("()[]{}<>", ")(][}{><")

# Hebrew letter ranges (including final forms)
_HEB_LETTERS = "\u05D0-\u05EA\u05DA\u05DD\u05DF\u05E3\u05E5"

# Chapter header: "בראשית - פרק א'"
# _CHAPTER_RE = re.compile(
#     rf"^\s*(?P<book>[\u0590-\u05FF]+)\s*-\s*פרק\s*(?P<chap>[\u0590-\u05FF\"״׳']+)\s*$"
# )
_CHAPTER_RE = re.compile(
    rf"^\s*(?P<book>[\u0590-\u05FF]+(?:\s+[\u0590-\u05FF]+)*)\s*-\s*פרק\s*(?P<chap>[\u0590-\u05FF\"״׳']+)\s*$"
)

# Verse markers:
#   - With geresh:    א'..., י'...
#   - With gershayim: י"א..., ט"ו..., כ"א...
_VERSE_GERESH_RE = re.compile(rf"^\s*(?P<num>[{_HEB_LETTERS}]{{1,8}})[׳']\s*(?P<rest>.*)$")
_VERSE_GERSHAYIM_RE = re.compile(rf"^\s*(?P<num>[{_HEB_LETTERS}]{{1,8}}[\"״][{_HEB_LETTERS}])\s*(?P<rest>.*)$")

# Remove parashah markers and similar curly-brace markers
_CURLY_MARKER_RE = re.compile(r"\{[^}]*\}")

# Collapse whitespace (after we join continuation lines)
_WS_RE = re.compile(r"\s+")

# Page break marker injected by extract_lines
_PAGE_BREAK = "__PAGE_BREAK__"


_GEMATRIA = {
    "א": 1,
    "ב": 2,
    "ג": 3,
    "ד": 4,
    "ה": 5,
    "ו": 6,
    "ז": 7,
    "ח": 8,
    "ט": 9,
    "י": 10,
    "כ": 20,
    "ל": 30,
    "מ": 40,
    "נ": 50,
    "ס": 60,
    "ע": 70,
    "פ": 80,
    "צ": 90,
    "ק": 100,
    "ר": 200,
    "ש": 300,
    "ת": 400,
    # Final forms
    "ך": 20,
    "ם": 40,
    "ן": 50,
    "ף": 80,
    "ץ": 90,
}


# Book name mappings (Hebrew Bible / Tanakh)
BOOK_NAMES = {
    "בראשית": {"en": "Genesis", "id": "bereshit"},
    "שמות": {"en": "Exodus", "id": "shemot"},
    "ויקרא": {"en": "Leviticus", "id": "vayikra"},
    "במדבר": {"en": "Numbers", "id": "bamidbar"},
    "דברים": {"en": "Deuteronomy", "id": "devarim"},
    "יהושע": {"en": "Joshua", "id": "yehoshua"},
    "שופטים": {"en": "Judges", "id": "shoftim"},
    "שמואל א": {"en": "1 Samuel", "id": "shmuel_alef"},
    "שמואל ב": {"en": "2 Samuel", "id": "shmuel_bet"},
    "מלכים א": {"en": "1 Kings", "id": "melachim_alef"},
    "מלכים ב": {"en": "2 Kings", "id": "melachim_bet"},
    "ישעיהו": {"en": "Isaiah", "id": "yeshayahu"},
    "ירמיהו": {"en": "Jeremiah", "id": "yirmiyahu"},
    "יחזקאל": {"en": "Ezekiel", "id": "yechezkel"},
    "הושע": {"en": "Hosea", "id": "hoshea"},
    "יואל": {"en": "Joel", "id": "yoel"},
    "עמוס": {"en": "Amos", "id": "amos"},
    "עובדיה": {"en": "Obadiah", "id": "ovadya"},
    "יונה": {"en": "Jonah", "id": "yona"},
    "מיכה": {"en": "Micah", "id": "micha"},
    "נחום": {"en": "Nahum", "id": "nachum"},
    "חבקוק": {"en": "Habakkuk", "id": "chavakuk"},
    "צפניה": {"en": "Zephaniah", "id": "tzefanya"},
    "חגי": {"en": "Haggai", "id": "chagai"},
    "זכריה": {"en": "Zechariah", "id": "zecharya"},
    "מלאכי": {"en": "Malachi", "id": "malachi"},
    "תהלים": {"en": "Psalms", "id": "tehilim"},
    "משלי": {"en": "Proverbs", "id": "mishlei"},
    "איוב": {"en": "Job", "id": "iyov"},
    "שיר השירים": {"en": "Song of Songs", "id": "shir_hashirim"},
    "רות": {"en": "Ruth", "id": "rut"},
    "איכה": {"en": "Lamentations", "id": "eicha"},
    "קהלת": {"en": "Ecclesiastes", "id": "kohelet"},
    "אסתר": {"en": "Esther", "id": "esther"},
    "דניאל": {"en": "Daniel", "id": "daniel"},
    "עזרא": {"en": "Ezra", "id": "ezra"},
    "נחמיה": {"en": "Nehemiah", "id": "nechemya"},
    "דברי הימים א": {"en": "1 Chronicles", "id": "divrei_hayamim_alef"},
    "דברי הימים ב": {"en": "2 Chronicles", "id": "divrei_hayamim_bet"},
}


@dataclass
class Verse:
    book: str
    chapter: int
    verse: int
    text: str
    page: int


def reverse_rtl_line(line: str) -> str:
    """Reverse a line and swap bracket directions.

    Many Hebrew PDFs extract each line in visual order (right-to-left), which
    appears reversed in plain text. Reversing the string usually restores
    logical order.
    """
    # Keep NULs ("\x00") because in some PDFs they represent *missing glyphs*.
    # We'll fix those later in `_fix_null_placeholders`.
    line = (
        line.replace("\u2009", " ")
        .replace("\u00A0", " ")  # NBSP
        .replace("\u202F", " ")  # narrow NBSP
    )
    # Reverse characters
    line = line[::-1]
    # Fix bracket directions: () [] {} <>
    line = line.translate(_BRACKET_SWAP)
    return line.strip()


def normalize_line(line: str) -> str:
    """Basic cleanup for lines that are already in logical order."""
    return (
        line.replace("\u2009", " ")
        .replace("\u00A0", " ")
        .replace("\u202F", " ")
        .strip()
    )


def _detect_reverse_needed(raw_lines: List[str]) -> bool:
    """Heuristic: decide whether to reverse lines to get logical Hebrew order.

    Many Hebrew PDFs extract each line in visual RTL order, which appears reversed.
    We detect this by checking whether chapter headers match before or after reversing.
    """
    if not raw_lines:
        return True

    # If we already see a chapter header as-is, don't reverse.
    for ln in raw_lines:
        if _CHAPTER_RE.match(normalize_line(ln)):
            return False

    # If we see a chapter header only after reversing, reverse.
    for ln in raw_lines:
        if _CHAPTER_RE.match(reverse_rtl_line(ln)):
            return True

    # Default: most Hebrew PDFs need reversing.
    return True


def extract_lines(
    pdf_path: str,
    direction: str = "auto",
    x_tolerance: float = 8.0,
) -> List[str]:
    """Extract text lines from a PDF.

    Parameters
    ----------
    direction:
        - "auto" (default): detect whether to reverse lines based on chapter headers
        - "reverse": always reverse each extracted line
        - "none": never reverse
    """
    try:
        import pdfplumber  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pdfplumber is required. Install with: pip install pdfplumber"
        ) from e

    out: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        if direction not in {"auto", "reverse", "none"}:
            raise ValueError(f"Invalid direction: {direction!r}")

        reverse_needed: bool
        if direction == "reverse":
            reverse_needed = True
        elif direction == "none":
            reverse_needed = False
        else:
            first_raw = (
                (pdf.pages[0].extract_text(x_tolerance=x_tolerance) or "").splitlines()
                if pdf.pages
                else []
            )
            reverse_needed = _detect_reverse_needed(first_raw)

        for page in tqdm(pdf.pages, desc="Extracting PDF pages"):
            # Raising x_tolerance generally reduces spurious spaces inside Hebrew words.
            raw = page.extract_text(x_tolerance=x_tolerance) or ""
            for raw_line in raw.splitlines():
                fixed = reverse_rtl_line(raw_line) if reverse_needed else normalize_line(raw_line)
                if fixed:
                    out.append(fixed)
            # page boundary marker
            out.append(_PAGE_BREAK)
    return out


def hebrew_numeral_to_int(token: str) -> int:
    """Convert Hebrew numeral notation (gematria) to int.

    Accepts geresh/gershayim and returns the numeric value.
    Example: "א'" -> 1, "י\"א" -> 11.
    """
    cleaned = token
    cleaned = cleaned.replace("'", "").replace("׳", "").replace('"', "").replace("״", "")
    cleaned = cleaned.strip()
    total = 0
    for ch in cleaned:
        if ch in _GEMATRIA:
            total += _GEMATRIA[ch]
    if total <= 0:
        raise ValueError(f"Could not parse Hebrew numeral from: {token!r}")
    return total


def _fix_null_placeholders(text: str) -> str:
    """Replace NUL ("\x00") placeholders with likely missing Hebrew letters.

    In this particular PDF, pdfplumber/pdfminer extracts some glyphs as a NUL.
    Empirically:
      - After an alef (א), the NUL usually represents lamed (ל) as in אלוהים.
      - Otherwise, the NUL usually represents final kaf (ך), e.g. חשך, בתוך.

    This is a heuristic. If you use a different PDF and it behaves differently,
    adjust this mapping.
    """

    if "\x00" not in text:
        return text

    out: List[str] = []
    for i, ch in enumerate(text):
        if ch != "\x00":
            out.append(ch)
            continue

        # Look left for the nearest *base* character (skip whitespace/combining marks)
        j = i - 1
        while j >= 0 and (text[j].isspace() or unicodedata.combining(text[j])):
            j -= 1
        prev = text[j] if j >= 0 else ""

        if prev == "א":
            out.append("ל")
        else:
            out.append("ך")
    return "".join(out)


def _clean_verse_text(text: str, *, remove_niqqud: bool) -> str:
    # Remove {פ} / {ס} / similar curly markers
    text = _CURLY_MARKER_RE.sub("", text)

    # Normalize common spacing artifacts
    text = text.replace("\u2009", " ").replace("\u00A0", " ").replace("\u202F", " ")

    # Decompose presentation forms (e.g., "בּ" -> "ב" + dagesh)
    text = unicodedata.normalize("NFKD", text)

    # Optionally remove niqqud/cantillation (combining marks)
    if remove_niqqud:
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Fix missing-glyph placeholders
    text = _fix_null_placeholders(text)

    # Whitespace + punctuation cleanup
    text = _WS_RE.sub(" ", text).strip()
    # Remove spaces around maqaf (־)
    text = re.sub(r"\s*־\s*", "־", text)
    # Remove spaces before common punctuation (incl. sof pasuq)
    text = re.sub(r"\s+([׃,.;:!?])", r"\1", text)
    #remove "׃"
    text = text.replace("׃", "")
    return text


def parse_verses(lines: Iterable[str], *, remove_niqqud: bool = True) -> List[Verse]:
    verses: List[Verse] = []

    current_book: Optional[str] = None
    current_chapter: Optional[int] = None
    current_verse_num: Optional[int] = None
    current_text_parts: List[str] = []
    current_page = 1

    def flush_current() -> None:
        nonlocal current_book, current_chapter, current_verse_num, current_text_parts
        if current_book is not None and current_chapter is not None and current_verse_num is not None:
            text = _clean_verse_text(" ".join(current_text_parts), remove_niqqud=remove_niqqud)
            verses.append(
                Verse(
                    book=current_book,
                    chapter=current_chapter,
                    verse=current_verse_num,
                    text=text,
                    page=current_page,
                )
            )
        current_verse_num = None
        current_text_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == _PAGE_BREAK:
            current_page += 1
            continue

        # Chapter header?
        m_ch = _CHAPTER_RE.match(line)
        if m_ch:
            flush_current()
            current_book = m_ch.group("book")
            try:
                current_chapter = hebrew_numeral_to_int(m_ch.group("chap"))
            except ValueError:
                # If chapter cannot be parsed, keep previous chapter (rare)
                current_chapter = current_chapter
            continue

        # Book title line alone (e.g., "בראשית")
        if current_book is None and re.fullmatch(r"[\u0590-\u05FF]+", line):
        # if line in BOOK_NAMES:
            current_book = line
            continue
        
        if line in BOOK_NAMES:
            flush_current()
            current_book = line
            continue

        # Verse start?
        m_v = _VERSE_GERSHAYIM_RE.match(line) or _VERSE_GERESH_RE.match(line)
        if m_v and current_book and current_chapter:
            flush_current()
            num_token = m_v.group("num")
            rest = m_v.group("rest")
            try:
                current_verse_num = hebrew_numeral_to_int(num_token)
            except ValueError:
                current_verse_num = None
            current_text_parts = [rest] if rest else []
            continue

        # Continuation line
        if current_verse_num is not None:
            current_text_parts.append(line)

    flush_current()
    return verses


def save_data(verses: list, books: dict, output_dir: Path):
    """Save extracted data to JSON files (same format as extract_pdf.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    books_dir = output_dir / "books"
    books_dir.mkdir(exist_ok=True)
    index_dir = output_dir / "index"
    index_dir.mkdir(exist_ok=True)

    metadata = {
        "total_verses": len(verses),
        "books": [
            {
                "id": info["id"],
                "name_he": name,
                "name_en": info["name_en"],
                "chapter_count": len(info["chapters"]),
            }
            for name, info in books.items()
            if info["chapters"]
        ],
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for book_name, book_data in books.items():
        if book_data["chapters"]:
            book_id = book_data["id"]
            with open(books_dir / f"{book_id}.json", "w", encoding="utf-8") as f:
                json.dump(book_data, f, ensure_ascii=False, indent=2)

    with open(index_dir / "verses.json", "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    print(
        f"Extracted {len(verses)} verses from {len([b for b in books.values() if b['chapters']])} books"
    )
    print(f"Data saved to {output_dir}")


def extract_verses(pdf_path: Path, output_dir: Path) -> int:

    lines = extract_lines(pdf_path, direction="auto")
    verses = parse_verses(lines)

    # Build verse list + book structure to match extract_pdf.py output format
    verses_out = []
    books = {}
    for v in verses:
        book_info = BOOK_NAMES.get(v.book, {"en": v.book, "id": v.book})
        if v.book not in books:
            books[v.book] = {
                "name_he": v.book,
                "name_en": book_info["en"],
                "id": book_info["id"],
                "chapters": {},
            }

        verse_id = f'{book_info["id"]}-{v.chapter}-{v.verse}'
        verse_obj = {
            "id": verse_id,
            "book": v.book,
            "book_en": book_info["en"],
            "book_id": book_info["id"],
            "chapter": v.chapter,
            "verse": v.verse,
            "text": v.text,
            "page": v.page,
        }
        verses_out.append(verse_obj)

        chapter_key = str(v.chapter)
        if chapter_key not in books[v.book]["chapters"]:
            books[v.book]["chapters"][chapter_key] = []
        books[v.book]["chapters"][chapter_key].append(verse_obj)

    save_data(verses_out, books, output_dir)

    # Summary to stderr so piping JSONL works.
    books = sorted({v.book for v in verses})
    chapters = sorted({(v.book, v.chapter) for v in verses})
    print(
        f"Parsed {len(verses)} verses | books={books} | chapters={len(chapters)} | output={output_dir}",
        file=sys.stderr,
    )
    return 0
