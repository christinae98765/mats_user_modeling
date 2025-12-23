#!/usr/bin/env python3
"""
Fix MCQ Conversations to End with Human Turn

Truncates MCQ conversations that end with an Assistant turn to remove
the final assistant turn, ensuring all conversations end with a Human turn.

This ensures consistency when extracting activations for probes that
capture "what the model thinks before responding."

Usage:
    # Dry run (show what would change)
    python scripts/fix_mcq_endings.py --dry_run

    # Fix in place
    python scripts/fix_mcq_endings.py

    # Fix and save to new directory
    python scripts/fix_mcq_endings.py --output_dir data/mcq_conversations_fixed/txt
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def parse_conversation(text: str) -> list[dict]:
    """Parse conversation into list of turns."""
    turns = []
    pattern = r'###\s*(Human|Assistant):\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    i = 1
    while i < len(parts) - 1:
        role = parts[i].lower()
        content = parts[i + 1].strip()
        turns.append({"role": role, "content": content})
        i += 2

    return turns


def rebuild_conversation(turns: list[dict]) -> str:
    """Rebuild conversation text from turns."""
    lines = []
    for turn in turns:
        role = "Human" if turn["role"] == "human" else "Assistant"
        lines.append(f"### {role}:")
        lines.append(turn["content"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def get_last_role(text: str) -> str:
    """Get the role of the last turn."""
    turns = parse_conversation(text)
    if turns:
        return turns[-1]["role"]
    return "unknown"


def fix_conversation(text: str) -> tuple[str, bool]:
    """
    Fix conversation to end with Human turn.
    Returns (fixed_text, was_modified).
    """
    turns = parse_conversation(text)

    if not turns:
        return text, False

    if turns[-1]["role"] == "human":
        return text, False

    # Remove trailing assistant turns
    while turns and turns[-1]["role"] == "assistant":
        turns.pop()

    if not turns:
        # Edge case: conversation was only assistant turns
        return text, False

    return rebuild_conversation(turns), True


def main():
    parser = argparse.ArgumentParser(description="Fix MCQ conversations to end with Human turn")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/mcq_conversations/txt",
        help="Input directory containing conversation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (if not specified, modifies in place)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original files before modifying",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    files = list(input_dir.glob("*.txt"))
    print(f"Found {len(files)} conversation files")

    # Analyze current state
    ending_human = 0
    ending_assistant = 0
    for f in files:
        with open(f) as fp:
            text = fp.read()
        last_role = get_last_role(text)
        if last_role == "human":
            ending_human += 1
        else:
            ending_assistant += 1

    print(f"\nCurrent state:")
    print(f"  Ending with Human: {ending_human}")
    print(f"  Ending with Assistant: {ending_assistant}")

    if ending_assistant == 0:
        print("\nAll conversations already end with Human. Nothing to fix.")
        return

    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        in_place = False
    else:
        output_dir = input_dir
        in_place = True

    if args.dry_run:
        print(f"\nDRY RUN - showing changes:")
        print("-" * 60)

    modified_count = 0
    for f in files:
        with open(f) as fp:
            original_text = fp.read()

        fixed_text, was_modified = fix_conversation(original_text)

        if was_modified:
            modified_count += 1

            if args.dry_run:
                # Show sample of what changed
                original_turns = parse_conversation(original_text)
                fixed_turns = parse_conversation(fixed_text)
                removed = len(original_turns) - len(fixed_turns)
                print(f"{f.name}: removed {removed} assistant turn(s) "
                      f"({len(original_turns)} -> {len(fixed_turns)} turns)")
            else:
                # Actually write the file
                output_path = output_dir / f.name

                if in_place and args.backup:
                    backup_path = f.with_suffix('.txt.bak')
                    shutil.copy(f, backup_path)

                with open(output_path, 'w') as fp:
                    fp.write(fixed_text)

    print("-" * 60)
    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count} files")

    if not args.dry_run:
        # Verify
        new_ending_human = 0
        new_ending_assistant = 0
        for f in output_dir.glob("*.txt"):
            with open(f) as fp:
                text = fp.read()
            last_role = get_last_role(text)
            if last_role == "human":
                new_ending_human += 1
            else:
                new_ending_assistant += 1

        print(f"\nAfter fix:")
        print(f"  Ending with Human: {new_ending_human}")
        print(f"  Ending with Assistant: {new_ending_assistant}")

        if args.output_dir:
            print(f"\nFixed files saved to: {args.output_dir}")
        else:
            print(f"\nFiles modified in place: {args.input_dir}")
            if args.backup:
                print("  (backups saved with .bak extension)")


if __name__ == "__main__":
    main()