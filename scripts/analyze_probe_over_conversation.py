#!/usr/bin/env python3
"""
Analyze Probe Predictions Over Conversation Turns

Analyzes how probe predictions evolve over the course of a conversation,
showing how the model's representation of user skill develops turn by turn.

Usage:
    # Analyze a single file
    python scripts/analyze_probe_over_conversation.py --file data/mcq_conversations/txt/conversation_0_understanding_novice.txt

    # Analyze multiple files from a directory
    python scripts/analyze_probe_over_conversation.py --dir data/mcq_conversations/txt/ --num-files 10

    # Use specific layer
    python scripts/analyze_probe_over_conversation.py --file <path> --layer 15
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probes import LinearProbeClassification


# =============================================================================
# Configuration
# =============================================================================

LABEL_NAMES = ["novice", "intermediate", "expert"]


# =============================================================================
# Conversation Parsing
# =============================================================================

def parse_conversation_turns(text: str) -> list:
    """Parse a conversation into a list of turns."""
    turns = []
    pattern = r'###\s*(Human|Assistant):\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    i = 1
    while i < len(parts) - 1:
        role = parts[i].lower()
        content = parts[i + 1].strip()
        if content:
            turns.append({"role": role, "content": content})
        i += 2

    return turns


def build_truncated_conversations(turns: list) -> list:
    """
    Build truncated versions of the conversation at each student response.
    Returns list of (turn_number, truncated_text) tuples.
    """
    truncations = []
    current_text = ""
    turn_count = 0

    for turn in turns:
        role_marker = "### Human:" if turn["role"] == "human" else "### Assistant:"
        current_text += f"{role_marker} {turn['content']}\n\n"

        if turn["role"] == "human":
            turn_count += 1
            truncations.append((turn_count, current_text.strip()))

    return truncations


# =============================================================================
# Model and Activation Extraction
# =============================================================================

def load_model(model_name: str):
    """Load model with 8-bit quantization for CUDA."""
    print(f"Loading model: {model_name}")

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
        model.half()
        device = "cpu"

    model.eval()
    return model, tokenizer, device


def extract_activations(text: str, tokenizer, model, device: str, num_layers: int) -> torch.Tensor:
    """Extract last-token activations from each layer."""
    # Parse text and format as chat
    turns = parse_conversation_turns(text)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for turn in turns:
        role = "user" if turn["role"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["content"]})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    encoding = tokenizer(
        formatted,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        output = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            output_hidden_states=True,
            return_dict=True
        )

    activations = []
    for layer_num in range(num_layers + 1):
        layer_acts = output["hidden_states"][layer_num][:, -1].detach().cpu()
        activations.append(layer_acts)

    return torch.cat(activations, dim=0)


# =============================================================================
# Probe Analysis
# =============================================================================

def load_probe(checkpoint_path: str, num_classes: int, hidden_size: int, device: str):
    """Load a trained probe."""
    probe = LinearProbeClassification(
        probe_class=num_classes,
        device=device,
        input_dim=hidden_size,
        logistic=True,
    )
    probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
    probe.eval()
    return probe


def run_probe(probe, activations: torch.Tensor, layer: int, device: str) -> torch.Tensor:
    """Run probe on activations from a specific layer."""
    layer_acts = activations[layer].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = probe(layer_acts)

    probs = logits / logits.sum(dim=-1, keepdim=True)
    return probs.cpu().squeeze()


def analyze_conversation(
    file_path: str,
    tokenizer,
    model,
    probe,
    layer: int,
    device: str,
    num_layers: int,
) -> dict:
    """Analyze how probe predictions evolve over a conversation."""
    with open(file_path, 'r') as f:
        text = f.read()

    turns = parse_conversation_turns(text)
    truncations = build_truncated_conversations(turns)

    results = {
        "file": file_path,
        "num_turns": len(truncations),
        "layer": layer,
        "predictions": [],
    }

    for turn_num, truncated_text in truncations:
        activations = extract_activations(
            truncated_text,
            tokenizer,
            model,
            device,
            num_layers,
        )

        probs = run_probe(probe, activations, layer, device)

        prediction = {
            "turn": turn_num,
            "probabilities": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
            "predicted_class": LABEL_NAMES[probs.argmax().item()],
            "confidence": float(probs.max()),
        }
        results["predictions"].append(prediction)

    return results


def print_results(results: dict):
    """Print analysis results."""
    print(f"\nFile: {results['file']}")
    print(f"Layer: {results['layer']}")
    print("=" * 60)

    for pred in results["predictions"]:
        turn = pred["turn"]
        probs = pred["probabilities"]
        predicted = pred["predicted_class"]

        prob_str = ", ".join([f"{name}: {probs[name]:.2f}" for name in LABEL_NAMES])
        print(f"Turn {turn:2d}: [{prob_str}] -> {predicted}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze probe predictions over conversation turns")
    parser.add_argument("--file", type=str, help="Single conversation file to analyze")
    parser.add_argument("--dir", type=str, help="Directory of conversation files")
    parser.add_argument("--num-files", type=int, default=5, help="Number of files from directory")
    parser.add_argument("--layer", type=int, default=40, help="Layer to probe")
    parser.add_argument("--probe-dir", type=str, default="data/probe_checkpoints/mcq/reading_probe",
                        help="Directory containing probe checkpoints")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model name")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Must specify either --file or --dir")

    # Load model
    model, tokenizer, device = load_model(args.model)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    # Load probe
    probe_path = os.path.join(args.probe_dir, f"understanding_probe_at_layer_{args.layer}.pth")
    if not os.path.exists(probe_path):
        print(f"ERROR: Probe not found at {probe_path}")
        print("Train probes first with: python scripts/train_probes.py")
        sys.exit(1)

    print(f"Loading probe from: {probe_path}")
    probe = load_probe(probe_path, len(LABEL_NAMES), hidden_size, device)

    # Collect files
    files_to_analyze = []
    if args.file:
        files_to_analyze = [args.file]
    else:
        all_files = sorted(Path(args.dir).glob("*_understanding_*.txt"))
        files_to_analyze = [str(f) for f in all_files[:args.num_files]]

    if not files_to_analyze:
        print("No matching files found")
        sys.exit(1)

    print(f"\nAnalyzing {len(files_to_analyze)} file(s)...")

    # Analyze
    all_results = []
    for file_path in files_to_analyze:
        print(f"\nProcessing: {os.path.basename(file_path)}")

        results = analyze_conversation(
            file_path,
            tokenizer,
            model,
            probe,
            args.layer,
            device,
            num_layers,
        )

        # Extract ground truth from filename
        for label in LABEL_NAMES:
            if f"_{label}." in file_path or f"_{label}_" in file_path:
                results["ground_truth"] = label
                break

        print_results(results)
        all_results.append(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.output}")

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for result in all_results:
            ground_truth = result.get("ground_truth", "unknown")
            final_pred = result["predictions"][-1]["predicted_class"]
            final_conf = result["predictions"][-1]["confidence"]

            match_str = "Y" if final_pred == ground_truth else "X"
            print(f"{os.path.basename(result['file'])}: {ground_truth} -> {final_pred} ({final_conf:.2f}) {match_str}")


if __name__ == "__main__":
    main()