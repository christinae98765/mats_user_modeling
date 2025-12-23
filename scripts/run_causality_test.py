#!/usr/bin/env python3
"""
Causality Test for Skill Level Steering

Tests whether control probes can causally influence model outputs by steering
the model to respond as if the user has a different skill level.

Usage:
    # Use MCQ-trained probes
    python scripts/run_causality_test.py --probe_source mcq

    # Use turn-by-turn trained probes
    python scripts/run_causality_test.py --probe_source turn_by_turn

    # Adjust steering parameters
    python scripts/run_causality_test.py --probe_source mcq --strength 5 --from_layer 40 --to_layer 60
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from baukit import TraceDict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probes import LinearProbeClassification


# =============================================================================
# Configuration
# =============================================================================

SKILL_LEVELS = ["novice", "intermediate", "expert"]

STEERING_TARGETS = {
    "novice": [1.0, 0.0, 0.0],
    "intermediate": [0.0, 1.0, 0.0],
    "expert": [0.0, 0.0, 1.0],
}

PROBE_DIRS = {
    "mcq": "data/probe_checkpoints/mcq/control_probe",
    "turn_by_turn": "data/probe_checkpoints/turn_by_turn/control_probe",
}

QUESTIONS_FILE = "data/causality_test_questions/understanding.txt"

SYSTEM_PROMPT = "You are a helpful assistant."


# =============================================================================
# Model and Probe Loading
# =============================================================================

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
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model loaded!")
    return model, tokenizer, device


def load_probes(probe_dir: str, hidden_size: int, num_layers: int, device: str):
    """Load control probes from directory."""
    probes = {}

    for layer in range(num_layers + 1):
        path = os.path.join(probe_dir, f"understanding_probe_at_layer_{layer}.pth")
        if os.path.exists(path):
            probe = LinearProbeClassification(
                device=device,
                probe_class=3,
                input_dim=hidden_size,
                logistic=True,
            )
            probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            probe.eval()
            probes[layer] = probe

    return probes


def load_questions(questions_path: str) -> list[str]:
    """Load test questions from file."""
    if not os.path.exists(questions_path):
        print(f"Warning: Questions file not found: {questions_path}")
        return []
    with open(questions_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


# =============================================================================
# Causality Tester
# =============================================================================

class CausalityTester:
    def __init__(
        self,
        model,
        tokenizer,
        probes: dict,
        device: str,
        strength: float = 5.0,
        from_layer: int = 40,
        to_layer: int = 60,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probes = probes
        self.device = device
        self.strength = strength
        self.from_layer = from_layer
        self.to_layer = to_layer

        # Build layer names for intervention
        self.layer_names = []
        for name, module in self.model.named_modules():
            if name.startswith("model.layers.") and name.count(".") == 2:
                layer_num = int(name.split(".")[-1])
                if self.from_layer <= layer_num < self.to_layer:
                    self.layer_names.append(name)

        print(f"Steering layers: {self.from_layer} to {self.to_layer - 1}")
        print(f"Steering strength: {self.strength}")

    def _make_edit_function(self, cf_target: torch.Tensor, use_random: bool = False):
        """Create edit function for steering."""
        probes = self.probes
        N = self.strength
        device = self.device

        def edit_output(output, layer_name):
            layer_num = int(layer_name.split(".")[-1])
            probe_idx = layer_num + 1

            if probe_idx not in probes:
                return output

            probe = probes[probe_idx]
            hidden_states = output

            # Extract last token
            last_hidden = hidden_states[:, -1, :].clone().float()

            # Get probe weights and normalize
            weights = probe.proj[0].weight.clone()
            for i in range(weights.shape[0]):
                norm = weights[i].norm()
                if norm > 0:
                    weights[i] = weights[i] / norm

            if use_random:
                random_weights = torch.randn_like(weights)
                for i in range(random_weights.shape[0]):
                    random_norm = random_weights[i].norm()
                    if random_norm > 0:
                        random_weights[i] = random_weights[i] / random_norm
                steering = (cf_target.clone().to(device).float() @ random_weights * N).squeeze(0)
            else:
                steering = (cf_target.clone().to(device).float() @ weights * N).squeeze(0)

            # Apply steering
            last_hidden = last_hidden + steering
            hidden_states[:, -1, :] = last_hidden.half()

            return output

        return edit_output

    def generate_response(
        self,
        question: str,
        steer_to: str = None,
        use_random: bool = False,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a response with optional steering."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Setup steering target
        if steer_to is not None and steer_to in STEERING_TARGETS:
            target_vec = STEERING_TARGETS[steer_to]
            cf_target = torch.tensor([target_vec])
        else:
            cf_target = None

        with torch.no_grad():
            if cf_target is not None:
                edit_fn = self._make_edit_function(cf_target, use_random=use_random)
                with TraceDict(self.model, self.layer_names, edit_output=edit_fn):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def collect_responses(self, questions: list[str], output_dir: str = None) -> dict:
        """Collect responses under all conditions."""
        responses = {"unintervened": [], "gaussian": []}
        for level in SKILL_LEVELS:
            responses[level] = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            comparisons_dir = os.path.join(output_dir, "comparisons")
            os.makedirs(comparisons_dir, exist_ok=True)

        print(f"\nGenerating responses for {len(questions)} questions...")

        for i, q in enumerate(tqdm(questions)):
            # Unintervened
            resp_unintervened = self.generate_response(q, steer_to=None)
            responses["unintervened"].append(resp_unintervened)

            # Gaussian control
            resp_gaussian = self.generate_response(q, steer_to="novice", use_random=True)
            responses["gaussian"].append(resp_gaussian)

            # Each steering target
            for level in SKILL_LEVELS:
                resp = self.generate_response(q, steer_to=level)
                responses[level].append(resp)

            # Save incrementally
            if output_dir:
                text = f"QUESTION: {q}\n\n"
                text += "=" * 60 + "\n"
                text += "UNINTERVENED:\n" + resp_unintervened + "\n\n"

                for level in SKILL_LEVELS:
                    text += "=" * 60 + "\n"
                    text += f"{level.upper()}-STEERED:\n" + responses[level][i] + "\n\n"

                text += "=" * 60 + "\n"
                text += "GAUSSIAN CONTROL:\n" + resp_gaussian + "\n"

                with open(os.path.join(comparisons_dir, f"question_{i+1:02d}.txt"), "w") as f:
                    f.write(text)

        return responses


def save_results(output_dir: str, questions: list[str], responses: dict, config: dict):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save responses
    with open(os.path.join(output_dir, "responses.json"), "w") as f:
        json.dump({
            **config,
            "questions": questions,
            "responses": responses,
        }, f, indent=2)

    # Save summary
    summary = f"""Causality Test Results
==============================

Probe source: {config['probe_source']}
Model: {config['model']}
Steering strength: {config['strength']}
Layers: {config['from_layer']} to {config['to_layer'] - 1}
Number of questions: {len(questions)}

Results saved to: {output_dir}
"""

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(summary)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run causality test for skill level steering")
    parser.add_argument(
        "--probe_source",
        type=str,
        required=True,
        choices=["mcq", "turn_by_turn"],
        help="Which probes to use for steering",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default=None,
        help="Override default probe directory",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=QUESTIONS_FILE,
        help="Path to questions file",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=5.0,
        help="Steering strength",
    )
    parser.add_argument(
        "--from_layer",
        type=int,
        default=40,
        help="Start steering from this layer",
    )
    parser.add_argument(
        "--to_layer",
        type=int,
        default=60,
        help="Stop steering at this layer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit number of questions",
    )

    args = parser.parse_args()

    # Setup paths
    probe_dir = args.probe_dir or PROBE_DIRS[args.probe_source]
    output_dir = args.output_dir or f"data/causality_test_results/{args.probe_source}"

    # Load questions
    questions = load_questions(args.questions_path)
    if not questions:
        print(f"No questions found. Create: {args.questions_path}")
        return

    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Loaded {len(questions)} questions")

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Load probes
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"Loading probes from: {probe_dir}")
    probes = load_probes(probe_dir, hidden_size, num_layers, device)
    print(f"Loaded probes for {len(probes)} layers")

    if not probes:
        print("Error: No probes found. Train probes first.")
        return

    # Create tester
    tester = CausalityTester(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        device=device,
        strength=args.strength,
        from_layer=args.from_layer,
        to_layer=args.to_layer,
    )

    # Collect responses
    responses = tester.collect_responses(questions, output_dir=output_dir)

    # Save results
    config = {
        "probe_source": args.probe_source,
        "model": args.model,
        "strength": args.strength,
        "from_layer": args.from_layer,
        "to_layer": args.to_layer,
    }
    save_results(output_dir, questions, responses, config)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()