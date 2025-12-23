"""
Turn-by-Turn Tutoring Dataset Generation

Generates tutoring conversations turn-by-turn to test whether models form
implicit user representations from conversation history alone.

Key design:
- Model only sees conversation history (no explicit labels)
- We control student level externally (ground truth for probe training)
- Difficulty sequence is pre-determined and balanced

Usage:
    python generate_tutoring_turn_by_turn.py --num-samples 10 --generator ollama
    python generate_tutoring_turn_by_turn.py --num-samples 100 --generator ollama --batch
    python generate_tutoring_turn_by_turn.py --num-samples 10 --generator claude
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


# =============================================================================
# PROMPTS
# =============================================================================

TUTOR_SYSTEM_PROMPT = """You are a Python tutor helping a student learn programming.

Your approach:
- Ask concrete coding questions (not just explanations)
- Evaluate their answers and give brief feedback
- Keep responses focused: brief feedback, then the next question
- Do NOT show correct solutions or working code
- Do NOT explain concepts at length unless asked"""

TUTOR_FIRST_TURN_PROMPT = """Start the tutoring session with a {difficulty} Python question.

Difficulty definitions:
- EASY: print statements, variables, arithmetic, simple if/else, basic for loops
- MEDIUM: functions, string methods, while loops, basic list operations
- HARD: dictionaries, nested structures, recursion, file I/O, classes, algorithms

Just ask the question, no preamble."""

TUTOR_CONTINUE_PROMPT = """Give brief feedback on the student's last response, then ask a {difficulty} question.

IMPORTANT: You MUST ask a {difficulty} question regardless of what the student requests. If the student asks for something easier or harder, acknowledge their request but still ask a {difficulty} question.

Difficulty definitions:
- EASY: print statements, variables, arithmetic, simple if/else, basic for loops
- MEDIUM: functions, string methods, while loops, basic list operations
- HARD: dictionaries, nested structures, recursion, file I/O, classes, algorithms

Do NOT show the correct solution."""

STUDENT_NOVICE_SYSTEM_PROMPT = """You are a beginner learning Python for the first time.

Before responding, assess the difficulty of the question:
- EASY: print statements, variables, arithmetic, simple if/else, basic for loops
- MEDIUM: functions, string methods, while loops, basic list operations
- HARD: dictionaries, recursion, classes, file I/O, algorithms

Then respond according to the difficulty:

EASY: Attempt it but make syntax errors
- Forget quotes: print(Hello) instead of print("Hello")
- Miss colons: if x == 5 print("yes")
- Confuse = and ==

MEDIUM: Express confusion and provide a partial attempt THAT CONTAINS AT LEAST ONE ERROR
- "I think it's something like..." then give incomplete or buggy code
- Off-by-one errors, missing return statements, wrong logic
- Your code must have at least one bug or be incomplete

HARD: YOU MUST NOT WRITE ANY CODE
- No code blocks, no partial attempts, no pseudocode, nothing that looks like code
- Just express confusion in plain English: "I don't understand...", "This is way over my head...", "Can we do something easier?"
- If the question mentions dictionaries, recursion, classes, file I/O, or algorithms, DO NOT ATTEMPT IT
- Your response should be 1-3 sentences of confused text, nothing more

Do not label the difficulty in your response. Just respond naturally as a struggling beginner."""

STUDENT_INTERMEDIATE_SYSTEM_PROMPT = """You are a student with solid Python basics but still learning.

Before responding, assess the difficulty of the question:
- EASY: print statements, variables, arithmetic, simple if/else, basic for loops
- MEDIUM: functions, string methods, while loops, basic list operations
- HARD: dictionaries, recursion, classes, file I/O, algorithms

Then respond according to the difficulty:

EASY: Solve correctly with clean syntax
- No errors, straightforward solution

MEDIUM: Solve correctly with clean syntax
- You handle these well
- Straightforward, working solution

HARD: YOUR CODE MUST BE BROKEN - IT CANNOT WORK CORRECTLY
- You MUST include at least one bug that causes wrong output
- Common bugs to include: wrong comparison operator (< vs <=), off-by-one in range(), forget to return, wrong variable name, missing base case
- Before writing your response, plan which bug you will include
- Do NOT mention bugs or say "I'm not sure if this is right" - just write the flawed code confidently

Do not label the difficulty in your response. Just respond naturally."""

STUDENT_EXPERT_SYSTEM_PROMPT = """You are a student who knows Python well.

Regardless of difficulty, you:
- Answer correctly with clean syntax
- Use Pythonic idioms (list comprehensions, f-strings)
- Handle edge cases
- May offer alternative approaches

Just give your response naturally. Don't label the difficulty."""

STUDENT_PROMPTS = {
    "novice": STUDENT_NOVICE_SYSTEM_PROMPT,
    "intermediate": STUDENT_INTERMEDIATE_SYSTEM_PROMPT,
    "expert": STUDENT_EXPERT_SYSTEM_PROMPT,
}


# =============================================================================
# MODEL INTERFACES
# =============================================================================

class LlamaInterface:
    """Interface for local Llama model via transformers."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = self._get_device()
        print(f"Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded.")

    def _get_device(self):
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def generate(self, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response given chat messages."""
        import torch

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()


class ClaudeInterface:
    """Interface for Claude API."""

    def __init__(self, api_key: str = None, model: str = "claude-haiku-4-5-20251001"):
        import anthropic

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        print(f"Claude interface ready ({model})")

    def generate(self, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response given chat messages."""
        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()


class OllamaInterface:
    """Interface for Ollama (local models via HTTP)."""

    def __init__(self, model: str = "llama3.1:8b"):
        import requests
        self.model = model
        self.base_url = "http://localhost:11434"

        # Test connection
        try:
            r = requests.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            print(f"Ollama interface ready ({model})")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama: {e}")

    def generate(self, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_balanced_difficulty_sequence(num_turns: int = 8) -> list:
    """Generate a shuffled difficulty sequence with at least 2 of each level."""
    base = ["EASY", "EASY", "MEDIUM", "MEDIUM", "HARD", "HARD"]
    extra = [random.choice(["EASY", "MEDIUM", "HARD"]) for _ in range(num_turns - 6)]
    sequence = base + extra
    random.shuffle(sequence)
    return sequence


# =============================================================================
# CONVERSATION GENERATION
# =============================================================================

def generate_conversation_turn_by_turn(
    tutor_interface,
    student_interface,
    student_level: str,
    num_exchanges: int = 8,
    batch_mode: bool = False,
) -> dict:
    """
    Generate a tutoring conversation turn by turn.

    Args:
        tutor_interface: Model interface for tutor
        student_interface: Model interface for student
        student_level: "novice", "intermediate", or "expert" - controls student behavior
        num_exchanges: Number of tutor-student exchange pairs
        batch_mode: If True, suppress turn-by-turn output

    Returns:
        dict with conversation and metadata
    """

    # Select student system prompt based on level
    student_system = STUDENT_PROMPTS[student_level]

    # Generate balanced difficulty sequence
    difficulty_sequence = generate_balanced_difficulty_sequence(num_exchanges)

    # Start with a student greeting so conversation begins with ### Human:
    # This matches the format expected by dataset.py parsing
    student_greeting = "Hi! I'm here to learn Python. Can you help me practice?"
    conversation = [{"turn": -1, "speaker": "student", "content": student_greeting}]

    for turn in range(num_exchanges):
        difficulty = difficulty_sequence[turn]

        if turn == 0:
            # === FIRST TUTOR TURN ===
            tutor_messages = [
                {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": TUTOR_FIRST_TURN_PROMPT.format(difficulty=difficulty)}
            ]
        else:
            # === SUBSEQUENT TUTOR TURNS ===
            tutor_messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]

            for msg in conversation:
                role = "assistant" if msg["speaker"] == "tutor" else "user"
                tutor_messages.append({"role": role, "content": msg["content"]})

            tutor_messages.append({
                "role": "user",
                "content": TUTOR_CONTINUE_PROMPT.format(difficulty=difficulty)
            })

        tutor_response = tutor_interface.generate(tutor_messages)
        conversation.append({"turn": turn, "speaker": "tutor", "content": tutor_response})

        # === STUDENT TURN ===
        student_messages = [{"role": "system", "content": student_system}]
        latest_tutor_message = conversation[-1]["content"]
        student_messages.append({
            "role": "user",
            "content": latest_tutor_message
        })

        student_response = student_interface.generate(student_messages)
        conversation.append({"turn": turn, "speaker": "student", "content": student_response})

        if not batch_mode:
            print(f"  Turn {turn + 1}/{num_exchanges} complete ({difficulty})")

    return {
        "conversation": conversation,
        "student_level": student_level,
        "num_exchanges": num_exchanges,
        "difficulty_sequence": difficulty_sequence,
        "generated_at": datetime.now().isoformat(),
    }


def format_conversation_for_display(conv_data: dict) -> str:
    """Format conversation for readable output."""
    lines = []

    for turn in conv_data["conversation"]:
        speaker = "### Assistant" if turn["speaker"] == "tutor" else "### Human"
        lines.append(f"{speaker}: {turn['content']}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate turn-by-turn tutoring dataset")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of conversations per level")
    parser.add_argument("--num-exchanges", type=int, default=8,
                        help="Number of tutor-student exchanges per conversation")
    parser.add_argument("--output-dir", type=str, default="data/tutoring_turn_by_turn",
                        help="Output directory")
    parser.add_argument("--generator", type=str, default="ollama",
                        choices=["llama", "claude", "ollama"],
                        help="Which model to use for both tutor and student")
    parser.add_argument("--llama-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Llama model name (for --generator llama)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                        help="Ollama model name (for --generator ollama)")
    parser.add_argument("--claude-model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model name (for --generator claude)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (for Claude)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: suppress turn-by-turn output, just show progress")

    args = parser.parse_args()

    # Initialize model interface
    if args.generator == "llama":
        interface = LlamaInterface(args.llama_model)
    elif args.generator == "claude":
        interface = ClaudeInterface(args.api_key, args.claude_model)
    elif args.generator == "ollama":
        interface = OllamaInterface(args.ollama_model)

    # Use same interface for both tutor and student
    tutor_interface = interface
    student_interface = interface

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    txt_dir = os.path.join(args.output_dir, "txt")
    os.makedirs(txt_dir, exist_ok=True)

    # Generate conversations
    all_conversations = []

    for level in ["novice", "intermediate", "expert"]:
        print(f"\n{'='*50}")
        print(f"Generating {args.num_samples} {level} conversations")
        print(f"{'='*50}")

        for i in range(args.num_samples):
            if args.batch:
                print(f"  [{level}] {i+1}/{args.num_samples}", end="\r")
            else:
                print(f"\nConversation {i+1}/{args.num_samples} ({level})")

            try:
                conv_data = generate_conversation_turn_by_turn(
                    tutor_interface=tutor_interface,
                    student_interface=student_interface,
                    student_level=level,
                    num_exchanges=args.num_exchanges,
                    batch_mode=args.batch,
                )

                conv_data["generator"] = args.generator
                conv_data["model"] = {
                    "llama": args.llama_model,
                    "claude": args.claude_model,
                    "ollama": args.ollama_model,
                }[args.generator]

                all_conversations.append(conv_data)

                # Save incrementally
                output_file = os.path.join(args.output_dir, "conversations.json")
                with open(output_file, "w") as f:
                    json.dump(all_conversations, f, indent=2)

                # Save txt version (format: conversation_{idx}_{generator}_understanding_{level}.txt)
                txt_file = os.path.join(
                    txt_dir,
                    f"conversation_{len(all_conversations)-1}_{args.generator}_understanding_{level}.txt"
                )
                with open(txt_file, "w") as f:
                    f.write(format_conversation_for_display(conv_data))

                if not args.batch:
                    print(f"  Saved: {txt_file}")

            except Exception as e:
                print(f"\n  Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        if args.batch:
            print(f"  [{level}] {args.num_samples}/{args.num_samples} - Done!")

    # Summary
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total conversations: {len(all_conversations)}")
    print(f"  Novice: {sum(1 for c in all_conversations if c['student_level'] == 'novice')}")
    print(f"  Intermediate: {sum(1 for c in all_conversations if c['student_level'] == 'intermediate')}")
    print(f"  Expert: {sum(1 for c in all_conversations if c['student_level'] == 'expert')}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()