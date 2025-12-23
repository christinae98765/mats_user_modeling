()#!/usr/bin/env python3
"""
Causality Test for TalkTuner (Llama 3.x)

This script evaluates whether controlling probes can causally influence model outputs by:
1. Generating responses under different steering conditions
2. Optionally using GPT-4 to evaluate response differences
3. Computing success rate to measure causal effect

Supports multiple attributes: gender, age, education, socioeconomic status, skepticism

Usage:
    # Gender causality test
    python scripts/run_causality_test_llama3.py --attribute gender

    # Skepticism causality test
    python scripts/run_causality_test_llama3.py --attribute skepticism

    # With custom settings
    python scripts/run_causality_test_llama3.py --attribute gender --strength 5 --from_layer 16 --to_layer 24
"""

import os
import sys
import json
import argparse
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probes import LinearProbeClassification

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. GPT-4 evaluation will be skipped.")


# Device selection
if torch.backends.mps.is_available():
    device = "mps"
    torch_device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
    torch_device = "cuda"
else:
    device = "cpu"
    torch_device = "cpu"


# Llama 3.x system prompt
LLAMA3_SYSTEM_PROMPT = """You are a helpful assistant."""


# Attribute configurations
ATTRIBUTE_CONFIG = {
    "gender": {
        "num_classes": 2,
        "labels": ["male", "female"],
        "questions_file": "data/causality_test_questions/gender.txt",
        "steering_targets": {
            "male": [1.0, 0.0],
            "female": [0.0, 1.0],
        },
    },
    "age": {
        "num_classes": 4,
        "labels": ["child", "adolescent", "adult", "older adult"],
        "questions_file": "data/causality_test_questions/age.txt",
        "steering_targets": {
            "child": [1.0, 0.0, 0.0, 0.0],
            "adolescent": [0.0, 1.0, 0.0, 0.0],
            "adult": [0.0, 0.0, 1.0, 0.0],
            "older adult": [0.0, 0.0, 0.0, 1.0],
        },
    },
    "education": {
        "num_classes": 3,
        "labels": ["someschool", "highschool", "collegemore"],
        "questions_file": "data/causality_test_questions/education.txt",
        "steering_targets": {
            "someschool": [1.0, 0.0, 0.0],
            "highschool": [0.0, 1.0, 0.0],
            "collegemore": [0.0, 0.0, 1.0],
        },
    },
    "socioeco": {
        "num_classes": 3,
        "labels": ["low", "middle", "high"],
        "questions_file": "data/causality_test_questions/socioeco.txt",
        "steering_targets": {
            "low": [1.0, 0.0, 0.0],
            "middle": [0.0, 1.0, 0.0],
            "high": [0.0, 0.0, 1.0],
        },
    },
    "skepticism": {
        "num_classes": 2,
        "labels": ["trusting", "skeptical"],
        "questions_file": "data/causality_test_questions/skepticism.txt",
        "steering_targets": {
            "trusting": [1.0, 0.0],
            "skeptical": [0.0, 1.0],
        },
        "default_probe_dir": "data/skepticism_probes/controlling_probe",
        "default_strength": 3.0,
        "default_from_layer": 18,
        "default_to_layer": 22,
    },
    "understanding": {
        "num_classes": 3,
        "labels": ["novice", "intermediate", "expert"],
        "questions_file": "data/causality_test_questions/understanding.txt",
        "steering_targets": {
            "novice": [1.0, 0.0, 0.0],
            "intermediate": [0.0, 1.0, 0.0],
            "expert": [0.0, 0.0, 1.0],
        },
        "default_probe_dir": "data/probe_checkpoints_tutoring/controlling_probe",
    },
    "understanding_mcq": {
        "num_classes": 3,
        "labels": ["novice", "intermediate", "expert"],
        "questions_file": "data/causality_test_questions/understanding.txt",
        "steering_targets": {
            "novice": [1.0, 0.0, 0.0],
            "intermediate": [0.0, 1.0, 0.0],
            "expert": [0.0, 0.0, 1.0],
        },
        "default_probe_dir": "data/probe_checkpoints_mcq/controlling_probe",
    },
}


def load_probes(probe_dir: str, attribute: str, hidden_size: int, num_layers: int):
    """Load controlling probes from directory."""
    config = ATTRIBUTE_CONFIG[attribute]
    probes = {}
    for layer in range(num_layers + 1):
        for suffix in ["", "_final"]:
            path = os.path.join(probe_dir, f"{attribute}_probe_at_layer_{layer}{suffix}.pth")
            if os.path.exists(path):
                probe = LinearProbeClassification(
                    device=device,
                    probe_class=config["num_classes"],
                    input_dim=hidden_size,
                    logistic=True,
                )
                probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                probe.eval()
                probes[layer] = probe
                break
    return probes


def load_questions(questions_path: str) -> list[str]:
    """Load test questions from file."""
    if not os.path.exists(questions_path):
        print(f"Warning: Questions file not found: {questions_path}")
        return []
    with open(questions_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


class CausalityTester:
    def __init__(
        self,
        attribute: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        probe_dir: str = "data/probe_checkpoints/controlling_probe",
        strength: float = 5.0,
        from_layer: int = 16,
        to_layer: int = 24,
    ):
        self.attribute = attribute
        self.config = ATTRIBUTE_CONFIG[attribute]
        self.model_name = model_name
        self.strength = strength
        self.from_layer = from_layer
        self.to_layer = to_layer

        print(f"Attribute: {attribute}")
        print(f"Loading model: {model_name}")
        print(f"Using device: {device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if device == "cuda" or device == "mps":
            self.model.half()
        self.model.to(device)
        self.model.eval()

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Load probes
        print(f"Loading probes from: {probe_dir}")
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        self.probes = load_probes(probe_dir, attribute, hidden_size, num_layers)
        print(f"Loaded probes for {len(self.probes)} layers: {sorted(self.probes.keys())}")

        # Build list of layer names to intervene on
        self.layer_names = []
        for name, module in self.model.named_modules():
            if name.startswith("model.layers.") and name.count(".") == 2:
                layer_num = int(name.split(".")[-1])
                if self.from_layer <= layer_num < self.to_layer:
                    self.layer_names.append(name)

        print(f"Will intervene on layers: {self.from_layer} to {self.to_layer - 1}")

    def _make_edit_function(self, cf_target: torch.Tensor, use_random: bool = False):
        """Create edit function for steering."""
        probes = self.probes
        N = self.strength

        def edit_output(output, layer_name):
            layer_num = int(layer_name.split(".")[-1])
            probe_idx = layer_num + 1
            if probe_idx not in probes:
                return output

            probe = probes[probe_idx]
            hidden_states = output

            # Extract last token, convert to float32 for steering math
            last_hidden = hidden_states[:, -1, :].clone().float()

            # Get probe weights and normalize each direction to unit norm
            # This makes steering strength N interpretable and consistent across layers
            weights = probe.proj[0].weight.clone()
            for i in range(weights.shape[0]):
                norm = weights[i].norm()
                if norm > 0:
                    weights[i] = weights[i] / norm

            # Optionally use random direction (Gaussian control)
            if use_random:
                random_weights = torch.randn_like(weights)
                for i in range(random_weights.shape[0]):
                    random_norm = random_weights[i].norm()
                    if random_norm > 0:
                        random_weights[i] = random_weights[i] / random_norm
                steering = (cf_target.clone().to(torch_device).float() @ random_weights * N).squeeze(0)
            else:
                target_clone = cf_target.clone().to(torch_device).float()
                steering = (target_clone @ weights * N).squeeze(0)

            # Add steering in float32, then convert back to float16
            last_hidden = last_hidden + steering
            hidden_states[:, -1, :] = last_hidden.half()
            return output

        return edit_output

    def generate_response(
        self,
        question: str,
        steer_to: Optional[str] = None,
        use_random: bool = False,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a response with optional steering."""
        # Use apply_chat_template for Llama 3.x formatting with system prompt
        messages = [
            {"role": "system", "content": LLAMA3_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # Set up steering target
        if steer_to is not None and steer_to in self.config["steering_targets"]:
            target_vec = self.config["steering_targets"][steer_to]
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
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Decode only the generated tokens (not the prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def collect_responses(self, questions: list[str], output_dir: str = None) -> dict[str, list[str]]:
        """Collect responses under all conditions, saving incrementally."""
        labels = self.config["labels"]

        # Initialize response dict with all steering conditions
        responses = {"unintervened": [], "gaussian": []}
        for label in labels:
            responses[label] = []

        # Create output directory for incremental saves
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            comparisons_dir = os.path.join(output_dir, "comparisons")
            os.makedirs(comparisons_dir, exist_ok=True)

        print(f"\nGenerating responses for {len(questions)} questions...")
        for i, q in enumerate(tqdm(questions)):
            # Generate unintervened
            resp_unintervened = self.generate_response(q, steer_to=None)
            responses["unintervened"].append(resp_unintervened)

            # Generate gaussian control (using first label's target)
            resp_gaussian = self.generate_response(q, steer_to=labels[0], use_random=True)
            responses["gaussian"].append(resp_gaussian)

            # Generate for each steering target
            for label in labels:
                resp = self.generate_response(q, steer_to=label)
                responses[label].append(resp)

            # Save incrementally
            if output_dir:
                # Save comparison file for this question
                text = f"QUESTION: {q}\n\n"
                text += "=" * 60 + "\n"
                text += "UNINTERVENED:\n"
                text += resp_unintervened + "\n\n"

                for label in labels:
                    text += "=" * 60 + "\n"
                    text += f"{label.upper()}-STEERED:\n"
                    text += responses[label][i] + "\n\n"

                text += "=" * 60 + "\n"
                text += "GAUSSIAN CONTROL:\n"
                text += resp_gaussian + "\n"

                with open(os.path.join(comparisons_dir, f"question_{i+1:02d}.txt"), "w") as f:
                    f.write(text)

                # Save running responses.json
                with open(os.path.join(output_dir, "responses.json"), "w") as f:
                    json.dump({
                        "attribute": self.attribute,
                        "model": self.model_name,
                        "strength": self.strength,
                        "layers": f"{self.from_layer}-{self.to_layer}",
                        "questions": questions[:i+1],
                        "responses": {k: v[:i+1] for k, v in responses.items()},
                    }, f, indent=2)

                print(f"  Saved question {i+1}/{len(questions)}")

        return responses


def save_results(
    output_dir: str,
    attribute: str,
    model_name: str,
    strength: float,
    from_layer: int,
    to_layer: int,
    questions: list[str],
    responses: dict[str, list[str]],
):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    config = ATTRIBUTE_CONFIG[attribute]

    # Save responses
    with open(os.path.join(output_dir, "responses.json"), "w") as f:
        json.dump({
            "attribute": attribute,
            "model": model_name,
            "strength": strength,
            "layers": f"{from_layer}-{to_layer}",
            "questions": questions,
            "responses": responses,
        }, f, indent=2)

    # Save individual comparison files
    comparisons_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)

    for i, q in enumerate(questions):
        text = f"QUESTION: {q}\n\n"
        text += "=" * 60 + "\n"
        text += "UNINTERVENED:\n"
        text += responses["unintervened"][i] + "\n\n"

        for label in config["labels"]:
            text += "=" * 60 + "\n"
            text += f"{label.upper()}-STEERED:\n"
            text += responses[label][i] + "\n\n"

        text += "=" * 60 + "\n"
        text += "GAUSSIAN CONTROL:\n"
        text += responses["gaussian"][i] + "\n"

        with open(os.path.join(comparisons_dir, f"question_{i+1:02d}.txt"), "w") as f:
            f.write(text)

    # Save summary
    summary = f"""Causality Test Results ({attribute})
==============================

Attribute: {attribute}
Model: {model_name}
Steering strength: {strength}
Layers: {from_layer} to {to_layer - 1}
Number of questions: {len(questions)}
Labels: {', '.join(config['labels'])}

Responses saved to: {output_dir}/responses.json
Comparisons saved to: {output_dir}/comparisons/
"""

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Run causality test for Llama 3.x")
    parser.add_argument(
        "--attribute",
        type=str,
        required=True,
        choices=list(ATTRIBUTE_CONFIG.keys()),
        help="Attribute to test (gender, age, education, socioeco, skepticism)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="data/probe_checkpoints/controlling_probe",
        help="Directory containing controlling probes",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=None,
        help="Path to questions file (defaults to attribute-specific file)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Steering strength (default: 5.0, or attribute-specific)",
    )
    parser.add_argument(
        "--from_layer",
        type=int,
        default=None,
        help="Start steering from this layer (default: 16 for Llama 3.1 8B, or attribute-specific)",
    )
    parser.add_argument(
        "--to_layer",
        type=int,
        default=None,
        help="Stop steering at this layer (default: 24 for Llama 3.1 8B, or attribute-specific)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to data/causality_test_results/{attribute}_llama3)",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )

    args = parser.parse_args()

    # Set defaults based on attribute
    config = ATTRIBUTE_CONFIG[args.attribute]
    questions_path = args.questions_path or config["questions_file"]
    output_dir = args.output_dir or f"data/causality_test_results/{args.attribute}_llama3"

    # Apply defaults: use command-line value if provided, else attribute-specific, else global default
    probe_dir = args.probe_dir
    if probe_dir == "data/probe_checkpoints/controlling_probe" and "default_probe_dir" in config:
        probe_dir = config["default_probe_dir"]

    # For strength/layers: None means user didn't specify, so use attribute-specific or global default
    strength = args.strength
    if strength is None:
        strength = config.get("default_strength", 5.0)

    from_layer = args.from_layer
    if from_layer is None:
        from_layer = config.get("default_from_layer", 16)

    to_layer = args.to_layer
    if to_layer is None:
        to_layer = config.get("default_to_layer", 24)

    # Load questions
    questions = load_questions(questions_path)
    if not questions:
        print(f"No questions found. Please create: {questions_path}")
        print("Example format (one question per line):")
        print("  What advice would you give about career development?")
        print("  How should I approach learning a new skill?")
        return

    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Print effective settings
    print(f"Attribute: {args.attribute}")
    print(f"Probe dir: {probe_dir}")
    print(f"Strength: {strength}")
    print(f"Layers: {from_layer}-{to_layer - 1}")

    # Initialize tester
    tester = CausalityTester(
        attribute=args.attribute,
        model_name=args.model,
        probe_dir=probe_dir,
        strength=strength,
        from_layer=from_layer,
        to_layer=to_layer,
    )

    # Collect responses (saves incrementally to output_dir)
    responses = tester.collect_responses(questions, output_dir=output_dir)

    # Save final results
    save_results(
        output_dir,
        args.attribute,
        args.model,
        strength,
        from_layer,
        to_layer,
        questions,
        responses,
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()