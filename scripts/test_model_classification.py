#!/usr/bin/env python3
"""
Test Model's Explicit Classification of User Skill Level

Tests what the model says when explicitly asked to classify user skill level.
Compares explicit verbalization to probe-based detection.

Usage:
    # Test on MCQ conversations
    python scripts/test_model_classification.py --conversation_dir data/mcq_conversations/txt --num_samples 5

    # Test single file
    python scripts/test_model_classification.py --conversation_file data/mcq_conversations/txt/conversation_0_understanding_novice.txt
"""

import argparse
import os
import sys
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# =============================================================================
# Configuration
# =============================================================================

SKILL_LEVELS = ["novice", "intermediate", "expert"]

CLASSIFICATION_PROMPT = "Based on our conversation, I think the understanding level of this user is"


# =============================================================================
# Utilities
# =============================================================================

def load_conversation(file_path: str) -> str:
    """Load conversation from text file."""
    with open(file_path, 'r') as f:
        return f.read()


def get_level_from_filename(filename: str) -> str:
    """Extract level from filename."""
    name = Path(filename).stem
    for level in SKILL_LEVELS:
        if level in name:
            return level
    return "unknown"


def parse_conversation(conversation: str) -> list[dict]:
    """Parse conversation into messages list."""
    lines = conversation.strip().split('\n')
    messages = []
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith('### Human:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "user"
            current_content = []
        elif line.startswith('### Assistant:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "assistant"
            current_content = []
        elif line.strip():
            current_content.append(line)

    if current_role and current_content:
        messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})

    return messages


def load_model(model_name: str):
    """Load model with 8-bit quantization for CUDA."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        print("Using CUDA with 8-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cpu"

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")
    return model, tokenizer, device


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test model's explicit classification of user skill level")
    parser.add_argument(
        "--conversation_file",
        type=str,
        default=None,
        help="Path to a single conversation file"
    )
    parser.add_argument(
        "--conversation_dir",
        type=str,
        default=None,
        help="Directory containing conversation files"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples per level"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Max tokens to generate"
    )

    args = parser.parse_args()

    if not args.conversation_file and not args.conversation_dir:
        print("Error: Must specify either --conversation_file or --conversation_dir")
        sys.exit(1)

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Collect files
    files_to_process = []

    if args.conversation_file:
        files_to_process.append(args.conversation_file)
    else:
        all_files = list(Path(args.conversation_dir).glob("*.txt"))

        # Group by level
        by_level = {level: [] for level in SKILL_LEVELS}
        for f in all_files:
            level = get_level_from_filename(f.name)
            if level in by_level:
                by_level[level].append(str(f))

        # Sample from each level
        for level, files in by_level.items():
            if files:
                sampled = random.sample(files, min(args.num_samples, len(files)))
                files_to_process.extend(sampled)

    print(f"\nProcessing {len(files_to_process)} conversations...\n")
    print("=" * 70)

    results = {level: [] for level in SKILL_LEVELS}
    correct = 0
    total = 0

    for file_path in files_to_process:
        true_level = get_level_from_filename(file_path)
        conversation = load_conversation(file_path)

        # Parse and add classification prompt
        messages = parse_conversation(conversation)
        messages.append({
            "role": "user",
            "content": CLASSIFICATION_PROMPT
        })

        # Format with chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Check if response contains the correct level
        response_lower = response.lower()
        predicted_level = "unknown"
        for level in SKILL_LEVELS:
            if level in response_lower:
                predicted_level = level
                break

        is_correct = (predicted_level == true_level)
        if is_correct:
            correct += 1
        total += 1

        print(f"File: {Path(file_path).name}")
        print(f"True level: {true_level.upper()}")
        print(f"Model says: \"{response}\"")
        print(f"Detected: {predicted_level} {'[CORRECT]' if is_correct else '[WRONG]'}")
        print("-" * 70)

        if true_level in results:
            results[true_level].append({
                "file": file_path,
                "response": response,
                "predicted": predicted_level,
                "correct": is_correct,
            })

        # Cleanup
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nOverall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    for level in SKILL_LEVELS:
        if results[level]:
            level_correct = sum(1 for r in results[level] if r["correct"])
            level_total = len(results[level])
            print(f"\n{level.upper()} ({level_correct}/{level_total}):")
            for r in results[level]:
                status = "Y" if r["correct"] else "X"
                print(f"  [{status}] {r['response'][:60]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()