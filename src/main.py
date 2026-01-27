#!/usr/bin/env python3
"""
CLI interface for Hebrew Bible Q&A Agent.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .agent import create_agent

def check_environment():
    """Check that required environment variables are set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it in your shell.")
        sys.exit(1)


def check_data():
    """Check that Bible data has been extracted."""
    data_dir = Path(__file__).parent.parent / "data" / "bible"
    verses_file = data_dir / "index" / "verses.json"

    if not verses_file.exists():
        print("Bible data not found. Running extraction...")
        print("Please run: python -m scripts.extract_pdf")
        return False
    return True


def interactive_mode(agent):
    """Run interactive conversation mode."""
    print("=" * 60)
    print("שואל על התנ\"ך - Hebrew Bible Q&A Agent")
    print("=" * 60)
    print()
    print("שאל שאלות על התנ\"ך בעברית.")
    print("פקודות מיוחדות:")
    print("  /reset  - התחל שיחה חדשה")
    print("  /books  - הצג רשימת ספרים")
    print("  /quit   - יציאה")
    print()

    while True:
        try:
            user_input = input("שאלה> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nלהתראות!")
            break

        if not user_input:
            continue

        # Handle special commands
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ["/quit", "/exit", "/q"]:
                print("להתראות!")
                break

            elif cmd == "/reset":
                agent.reset()
                print("השיחה אופסה.")
                continue


            elif cmd == "/books":
                books = agent.list_books()
                print("\nספרי התנ\"ך:")
                for book in books:
                    print(f"  {book['name_he']} ({book['name_en']}): {book['chapter_count']} פרקים")
                print()
                continue

            elif cmd == "/help":
                print("פקודות:")
                print("  /reset  - התחל שיחה חדשה")
                print("  /books  - הצג רשימת ספרים")
                print("  /quit   - יציאה")
                continue

            else:
                print(f"פקודה לא מוכרת: {user_input}")
                continue

        # Process query
        print()
        try:
            response = agent.query(user_input)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()


def single_query_mode(agent, query: str):
    """Process a single query and exit."""
    try:
        response = agent.query(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Hebrew Bible Q&A Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GPT_MODEL", "gpt-5-mini"),
        help="OpenAI model to use (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to Bible data directory"
    )


    args = parser.parse_args()


    # Check environment
    check_environment()

    # Check data
    if not check_data():
        sys.exit(1)

    # Create log directory per run
    logs_root = Path(__file__).parent.parent / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = logs_root / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agent
    try:
        print(f"Creating agent with data_dir: {args.data_dir}, model: {args.model}, log_dir: {run_log_dir}")
        agent = create_agent(data_dir=args.data_dir, model=args.model, log_dir=str(run_log_dir))
    except Exception as e:
        print(f"Error creating agent: {e}", file=sys.stderr)
        sys.exit(1)

    # Run in appropriate mode
    if args.query:
        single_query_mode(agent, args.query)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
