#!/usr/bin/env python3
"""
Train probes to detect user skill level from LLM hidden states.

Supports two probe types:
- Reading probes: Trained with appended classification prompt
- Control probes: Trained on raw conversation (for steering)

Supports two datasets:
- mcq: Multiple-choice Q&A conversations
- turn_by_turn: Open-ended tutoring conversations

Usage:
    # Train all probes on MCQ data
    python scripts/train_probes.py --data mcq

    # Train only reading probes on turn-by-turn data
    python scripts/train_probes.py --data turn_by_turn --probe_type reading

    # Extract activations only (no training)
    python scripts/train_probes.py --data mcq --extract_only
"""

import os
import sys
import argparse
import pickle
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sklearn.model_selection
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probes import LinearProbeClassification, TrainerConfig


# =============================================================================
# Configuration
# =============================================================================

LABEL_MAPPINGS = {
    "novice": 0,
    "intermediate": 1,
    "expert": 2,
}

DATASET_DIRS = {
    "mcq": "data/mcq_conversations/txt/",
    "turn_by_turn": "data/tutoring_turn_by_turn/txt/",
}

# Prompt appended for reading probes
READING_PROBE_PROMPT = " I think the understanding level of this user is"


# =============================================================================
# Conversation Parsing
# =============================================================================

def parse_conversation_file(file_path: str) -> list[dict]:
    """Parse a conversation file into messages."""
    with open(file_path, 'r') as f:
        content = f.read()

    messages = []
    lines = content.strip().split('\n')
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


def get_label_from_filename(filename: str) -> int:
    """Extract label from filename."""
    name = Path(filename).stem.lower()
    for level, idx in LABEL_MAPPINGS.items():
        if level in name:
            return idx
    return -1


def format_messages_llama3(messages: list[dict], tokenizer, add_reading_prompt: bool = False) -> str:
    """Format messages in Llama 3 chat format."""
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if add_reading_prompt:
        formatted += READING_PROBE_PROMPT

    return formatted


# =============================================================================
# Model Loading (CUDA with bitsandbytes)
# =============================================================================

def load_model(model_name: str):
    """Load model with 8-bit quantization for CUDA."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    else:
        print("Warning: CUDA not available. Loading in float32 (slow).")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    print("Model loaded!")

    return model, tokenizer


# =============================================================================
# Activation Extraction
# =============================================================================

def get_hidden_states(model, tokenizer, prompt: str, device: str = "cuda"):
    """
    Extract hidden states from all layers for a prompt.

    Returns: numpy array of shape [num_layers, hidden_size] (last token only)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract last token hidden states from each layer
    hidden_states = outputs.hidden_states

    last_token_states = []
    for layer_states in hidden_states:
        last_token = layer_states[0, -1, :].cpu().float().numpy()
        last_token_states.append(last_token)

    result = np.stack(last_token_states, axis=0)

    # Cleanup
    del outputs, hidden_states, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


def extract_activations(
    model,
    tokenizer,
    data_dir: str,
    output_dir: str,
    probe_type: str = "control",
    device: str = "cuda",
):
    """Extract activations from conversations and save to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all conversation files
    files = list(Path(data_dir).glob("*.txt"))
    add_reading_prompt = (probe_type == "reading")

    print(f"Extracting activations from {len(files)} files...")
    print(f"Probe type: {probe_type} (add_reading_prompt={add_reading_prompt})")

    # Get model dimensions
    test_messages = [{"role": "user", "content": "Hello"}]
    test_prompt = format_messages_llama3(test_messages, tokenizer, add_reading_prompt=add_reading_prompt)
    test_states = get_hidden_states(model, tokenizer, test_prompt, device)
    num_layers = test_states.shape[0]
    hidden_size = test_states.shape[1]
    del test_states

    print(f"  Detected {num_layers} layers, hidden size {hidden_size}")

    # Collect valid files
    valid_files = []
    valid_labels = []
    for file_path in files:
        label = get_label_from_filename(file_path.name)
        if label != -1:
            messages = parse_conversation_file(str(file_path))
            if messages:
                valid_files.append(file_path)
                valid_labels.append(label)

    num_samples = len(valid_files)
    print(f"  Found {num_samples} valid conversation files")

    # Create memory-mapped file
    temp_path = os.path.join(output_dir, f"activations_{probe_type}_temp.npy")
    activations_mmap = np.memmap(
        temp_path,
        dtype=np.float32,
        mode='w+',
        shape=(num_samples, num_layers, hidden_size)
    )

    processed = 0
    for i, file_path in enumerate(tqdm(valid_files, desc=f"Extracting ({probe_type})")):
        messages = parse_conversation_file(str(file_path))
        prompt = format_messages_llama3(messages, tokenizer, add_reading_prompt=add_reading_prompt)

        try:
            numpy_acts = get_hidden_states(model, tokenizer, prompt, device)
            activations_mmap[processed] = numpy_acts.astype(np.float32)
            processed += 1
            del numpy_acts
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        # Cleanup
        del messages, prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Log progress
        if processed % 10 == 0:
            activations_mmap.flush()
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                print(f"  [{processed}/{num_samples}] GPU memory: {gpu_mem:.2f} GB")

    activations_mmap.flush()

    if processed == 0:
        print("Error: No activations extracted!")
        del activations_mmap
        os.remove(temp_path)
        return None, None

    # Save as torch tensors
    print("Converting to torch format...")
    activations_tensor = torch.from_numpy(np.array(activations_mmap[:processed]))
    labels_tensor = torch.tensor(valid_labels[:processed])

    del activations_mmap
    gc.collect()

    save_path = os.path.join(output_dir, f"activations_{probe_type}.pt")
    torch.save({
        "activations": activations_tensor,
        "labels": labels_tensor,
        "files": [str(f) for f in valid_files[:processed]],
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "probe_type": probe_type,
    }, save_path)

    os.remove(temp_path)

    print(f"Saved activations to {save_path}")
    print(f"  Shape: {activations_tensor.shape}")
    print(f"  Labels distribution: {torch.bincount(labels_tensor).tolist()}")

    return activations_tensor, labels_tensor


# =============================================================================
# Probe Training
# =============================================================================

def train_probes_from_activations(
    activations_path: str,
    output_dir: str,
    probe_type: str = "control",
    max_epochs: int = 50,
):
    """Train probes from pre-extracted activations."""
    print(f"Loading activations from {activations_path}...")

    data = torch.load(activations_path)
    activations = data["activations"]
    labels = data["labels"]
    hidden_size = data["hidden_size"]
    num_layers = data["num_layers"]

    print(f"  Samples: {len(labels)}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Training device: {device}")

    # Train/test split
    indices = list(range(len(labels)))
    train_idx, test_idx = sklearn.model_selection.train_test_split(
        indices,
        test_size=0.2,
        random_state=12345,
        stratify=labels.numpy(),
    )

    # Output directories
    checkpoint_dir = os.path.join(output_dir, f"{probe_type}_probe")
    plot_dir = os.path.join(output_dir, f"{probe_type}_probe", "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Results
    accuracy_dict = {"best": [], "final": [], "train": []}

    # Train probe for each layer
    for layer_num in tqdm(range(num_layers), desc=f"Training {probe_type} probes"):
        train_acts = activations[train_idx, layer_num, :].to(device)
        train_labels = labels[train_idx].to(device)
        test_acts = activations[test_idx, layer_num, :].to(device)
        test_labels = labels[test_idx].to(device)

        probe = LinearProbeClassification(
            device=device,
            probe_class=3,
            input_dim=hidden_size,
            logistic=True,
        )

        config = TrainerConfig()
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        best_acc = 0

        for epoch in range(max_epochs):
            # Training
            probe.train()
            optimizer.zero_grad()
            logits, loss = probe(train_acts, train_labels)
            loss.backward()
            optimizer.step()

            # Evaluation
            probe.eval()
            with torch.no_grad():
                train_logits, _ = probe(train_acts)
                train_preds = train_logits.argmax(dim=1)
                train_acc = (train_preds == train_labels).float().mean().item()

                test_logits, _ = probe(test_acts)
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == test_labels).float().mean().item()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    probe.state_dict(),
                    os.path.join(checkpoint_dir, f"understanding_probe_at_layer_{layer_num}.pth")
                )

        accuracy_dict["best"].append(best_acc)
        accuracy_dict["final"].append(test_acc)
        accuracy_dict["train"].append(train_acc)

        if layer_num % 10 == 0 or layer_num == num_layers - 1:
            print(f"  Layer {layer_num}: best={best_acc:.4f}, final={test_acc:.4f}")

    # Save results
    with open(os.path.join(checkpoint_dir, "understanding_accuracy.pkl"), "wb") as f:
        pickle.dump(accuracy_dict, f)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_dict["best"], label="Best Test", marker="o", markersize=3)
    plt.plot(accuracy_dict["train"], label="Train", marker="s", markersize=3)
    plt.axhline(y=0.333, color="r", linestyle="--", label="Chance (33.3%)")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"Probe Accuracy by Layer ({probe_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "accuracy_by_layer.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print(f"Best accuracy: {max(accuracy_dict['best']):.4f} at layer {accuracy_dict['best'].index(max(accuracy_dict['best']))}")

    return accuracy_dict


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train skill level probes")
    parser.add_argument(
        "--data",
        type=str,
        choices=["mcq", "turn_by_turn"],
        default="mcq",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/probe_checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=["reading", "control", "both"],
        default="both",
        help="Type of probes to train",
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract activations, don't train probes",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only train probes from existing activations",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Max training epochs per layer",
    )

    args = parser.parse_args()

    print(f"Skill Level Probe Training")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Probe type: {args.probe_type}")
    print(f"  Output: {args.output_dir}")

    # Setup paths
    data_dir = DATASET_DIRS[args.data]
    output_dir = os.path.join(args.output_dir, args.data)
    activations_dir = os.path.join(output_dir, "activations")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine probe types
    if args.probe_type == "both":
        probe_types = ["control", "reading"]
    else:
        probe_types = [args.probe_type]

    # Load model
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.train_only:
        print(f"\n{'='*60}")
        print(f"Loading model")
        print(f"{'='*60}")
        model, tokenizer = load_model(args.model)

    # Process each probe type
    for probe_type in probe_types:
        activations_path = os.path.join(activations_dir, f"activations_{probe_type}.pt")

        # Extract activations
        if not args.train_only:
            print(f"\n{'='*60}")
            print(f"Extracting activations for {probe_type} probes")
            print(f"{'='*60}")

            extract_activations(
                model, tokenizer, data_dir, activations_dir,
                probe_type=probe_type,
                device=device,
            )

        # Train probes
        if not args.extract_only:
            print(f"\n{'='*60}")
            print(f"Training {probe_type} probes")
            print(f"{'='*60}")

            if not os.path.exists(activations_path):
                print(f"Error: Activations not found: {activations_path}")
                print("Run without --train_only first")
                continue

            train_probes_from_activations(
                activations_path,
                output_dir,
                probe_type=probe_type,
                max_epochs=args.max_epochs,
            )

    # Cleanup
    if model is not None:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
