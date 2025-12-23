"""
Generate multiple-choice tutoring conversations for probe training.

User behavior:
- Novice: correct on EASY only
- Intermediate: correct on EASY and MEDIUM
- Expert: correct on all

User responses are ONLY the letter (A/B/C/D) - no verbal hedging or explanation.
This isolates the "correctness" signal from linguistic uncertainty cues.
"""

import anthropic
import json
import random
import os
from pathlib import Path

# Initialize Anthropic client
client = anthropic.Anthropic()

# Difficulty definitions (same as turn-by-turn generation)
DIFFICULTY_DEFINITIONS = """
Question difficulties:
- EASY: print statements, variables, arithmetic, simple if/else, basic for loops, string concatenation
- MEDIUM: functions with parameters/returns, string methods, while loops, list operations, basic file concepts
- HARD: dictionaries, recursion, classes/OOP, file I/O, algorithms, nested data structures, error handling
"""

def generate_balanced_difficulty_sequence(num_turns=8):
    """Generate a balanced sequence of difficulties for a conversation."""
    base = ["EASY", "EASY", "MEDIUM", "MEDIUM", "HARD", "HARD"]
    extra = [random.choice(["EASY", "MEDIUM", "HARD"]) for _ in range(num_turns - 6)]
    sequence = base + extra
    random.shuffle(sequence)
    return sequence

def get_correct_answer_rule(level):
    """Return the rule for which questions this level gets correct."""
    if level == "novice":
        return "EASY only"
    elif level == "intermediate":
        return "EASY and MEDIUM"
    elif level == "expert":
        return "all questions (EASY, MEDIUM, and HARD)"

def get_answer_behavior(level, difficulty):
    """Determine if user should answer correctly based on level and difficulty."""
    if level == "expert":
        return "correct"
    elif level == "intermediate":
        if difficulty in ["EASY", "MEDIUM"]:
            return "correct"
        else:
            return "incorrect"
    elif level == "novice":
        if difficulty == "EASY":
            return "correct"
        else:
            return "incorrect"

SYSTEM_PROMPT_TEMPLATE = """You are generating a Python tutoring conversation for a dataset.

The conversation format:
1. Student starts by saying: "Hi! I'm here to learn Python. Can you help me practice?"
2. Tutor responds and asks the first multiple-choice question (A/B/C/D options)
3. Student responds with ONLY a single letter (A, B, C, or D)
4. Tutor gives brief feedback and asks the next question
5. Repeat until {num_turns} questions have been asked and answered

{difficulty_definitions}

CRITICAL RULES:
1. Generate exactly {num_turns} question-answer exchanges
2. Follow this exact difficulty sequence: {difficulty_sequence}
3. The student's skill level is: {level}
4. The student answers correctly on: {correct_rule}
5. Student responses must be ONLY a single letter - no explanations, no hedging, no "I think", just the letter
6. Make questions clearly match their assigned difficulty
7. Each question must have exactly 4 options (A, B, C, D)
8. Tutor feedback should be brief (1-2 sentences) before moving to next question
9. Keep feedback neutral - avoid excessive praise like "Perfect!", "Excellent!", "Great job!" - just say "Correct." or "That's not quite right." and move on
10. DO NOT include any summary or overall assessment at the end - just end after the last question is answered with brief neutral feedback

OUTPUT FORMAT:
Generate the conversation as a JSON array of turns. Start with the student greeting:
[
  {{"role": "human", "content": "Hi! I'm here to learn Python. Can you help me practice?"}},
  {{"role": "assistant", "difficulty": "EASY", "content": "Of course! Let's start with a question.\\n\\nWhat is the output of print(2 + 3)?\\nA) 5\\nB) 23\\nC) 6\\nD) Error"}},
  {{"role": "human", "content": "A"}},
  {{"role": "assistant", "difficulty": "MEDIUM", "content": "Correct! Next question: ..."}},
  {{"role": "human", "content": "B"}},
  ...
]

The conversation must have exactly {num_turns} questions asked and answered (plus the initial greeting).

Generate the complete conversation now as valid JSON:"""

def generate_conversation(level, num_turns=8, conversation_id=0):
    """Generate a single MCQ conversation."""
    
    difficulty_sequence = generate_balanced_difficulty_sequence(num_turns)
    correct_rule = get_correct_answer_rule(level)
    
    # Build the prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        num_turns=num_turns,
        difficulty_definitions=DIFFICULTY_DEFINITIONS,
        difficulty_sequence=difficulty_sequence,
        level=level.upper(),
        correct_rule=correct_rule
    )
    
    # Add specific guidance for which questions to get wrong
    answer_guidance = "\n\nSPECIFIC ANSWER PATTERN:\n"
    for i, diff in enumerate(difficulty_sequence):
        behavior = get_answer_behavior(level, diff)
        answer_guidance += f"Question {i+1} ({diff}): Student answers {behavior}\n"
    
    full_prompt = system_prompt + answer_guidance
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        # Extract the response text
        response_text = response.content[0].text
        
        # Try to parse as JSON
        # Find JSON array in response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            conversation = json.loads(json_str)
            
            return {
                "conversation_id": conversation_id,
                "level": level,
                "difficulty_sequence": difficulty_sequence,
                "turns": conversation,
                "num_turns": num_turns
            }
        else:
            print(f"Could not find JSON in response for conversation {conversation_id}")
            return None
            
    except Exception as e:
        print(f"Error generating conversation {conversation_id}: {e}")
        return None

def format_as_text(conversation_data, include_metadata=False):
    """Format conversation as readable text.
    
    Args:
        conversation_data: The conversation dictionary
        include_metadata: If True, include level and difficulty labels (for human review).
                         If False, produce clean text for probe training (no confounds).
    """
    lines = []
    
    if include_metadata:
        lines.append(f"Level: {conversation_data['level']}")
        lines.append(f"Difficulty sequence: {conversation_data['difficulty_sequence']}")
        lines.append("=" * 60)
    
    for turn in conversation_data['turns']:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        difficulty = turn.get('difficulty', '')
        
        # Handle both naming conventions (tutor/student or assistant/human)
        is_assistant = role in ['tutor', 'assistant']
        
        if is_assistant:
            if include_metadata and difficulty:
                lines.append(f"\n### Assistant [{difficulty}]:\n{content}")
            else:
                lines.append(f"\n### Assistant:\n{content}")
        else:
            lines.append(f"\n### Human:\n{content}")
    
    return "\n".join(lines)

def get_next_conversation_id(txt_dir: Path, level: str) -> int:
    """Find the next available conversation ID for a given level."""
    existing_files = list(txt_dir.glob(f"conversation_*_understanding_{level}.txt"))
    if not existing_files:
        return 0

    # Extract IDs from filenames
    ids = []
    for f in existing_files:
        name = f.stem  # e.g., "conversation_42_understanding_novice"
        parts = name.split('_')
        if len(parts) >= 2:
            try:
                ids.append(int(parts[1]))
            except ValueError:
                pass

    return max(ids) + 1 if ids else 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate MCQ tutoring conversations')
    parser.add_argument('--level', type=str, choices=['novice', 'intermediate', 'expert', 'all'],
                        default='all', help='Student level to generate')
    parser.add_argument('--num_conversations', type=int, default=100,
                        help='Number of conversations per level')
    parser.add_argument('--num_turns', type=int, default=8,
                        help='Number of Q&A exchanges per conversation')
    parser.add_argument('--output_dir', type=str, default='data/mcq_conversations',
                        help='Output directory')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode - minimal output')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files (default: append with new IDs)')

    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    txt_dir = output_dir / "txt"
    txt_meta_dir = output_dir / "txt_with_metadata"  # For human review
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which levels to generate
    if args.level == 'all':
        levels = ['novice', 'intermediate', 'expert']
    else:
        levels = [args.level]
    
    for level in levels:
        # Determine starting ID
        if args.overwrite:
            start_id = 0
        else:
            start_id = get_next_conversation_id(txt_dir, level)
            if start_id > 0:
                print(f"\nFound existing conversations for {level}, starting at ID {start_id}")

        print(f"\n{'='*60}")
        print(f"Generating {args.num_conversations} {level} conversations (IDs {start_id}-{start_id + args.num_conversations - 1})")
        print(f"{'='*60}")

        all_conversations = []

        for i in range(args.num_conversations):
            conv_id = start_id + i
            if not args.batch:
                print(f"  Generating conversation {i+1}/{args.num_conversations} (ID {conv_id})...", end=" ")

            conv = generate_conversation(
                level=level,
                num_turns=args.num_turns,
                conversation_id=conv_id
            )

            if conv:
                all_conversations.append(conv)

                # Save clean text file (for probe training - no confounds)
                # Filename includes _understanding_ to match train_probes.py label_idf pattern
                txt_path = txt_dir / f"conversation_{conv_id}_understanding_{level}.txt"
                with open(txt_path, 'w') as f:
                    f.write(format_as_text(conv, include_metadata=False))

                # Save text file with metadata (for human review)
                txt_meta_path = txt_meta_dir / f"conversation_{conv_id}_{level}.txt"
                with open(txt_meta_path, 'w') as f:
                    f.write(format_as_text(conv, include_metadata=True))

                if not args.batch:
                    print("✓")
            else:
                if not args.batch:
                    print("✗")

        # Save all conversations for this level as JSON (append mode)
        json_path = json_dir / f"mcq_conversations_{level}.json"
        if json_path.exists() and not args.overwrite:
            # Load existing and append
            with open(json_path, 'r') as f:
                existing = json.load(f)
            existing.extend(all_conversations)
            with open(json_path, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f"\nAppended {len(all_conversations)} {level} conversations (total: {len(existing)})")
        else:
            with open(json_path, 'w') as f:
                json.dump(all_conversations, f, indent=2)
            print(f"\nSaved {len(all_conversations)} {level} conversations")

        print(f"  JSON: {json_path}")
        print(f"  TXT (clean, for training): {txt_dir}/conversation_*_{level}.txt")
        print(f"  TXT (with metadata, for review): {txt_meta_dir}/conversation_*_{level}.txt")

if __name__ == "__main__":
    main()
