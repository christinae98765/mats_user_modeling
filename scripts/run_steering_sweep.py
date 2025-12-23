#!/usr/bin/env python3
"""
Faster Steering Sweep

A focused version of run_steering_sweep.py with:
- Layer ranges: 40-45, 45-50, 50-55, 55-60, 60-65, 65-70, 40-50, 50-60, 60-70
- Strengths: 1, 2, 3
- Both probe sources (mcq, turn_by_turn)

Total: 54 combinations (2 sources × 9 layer ranges × 3 strengths)

Usage:
    python scripts/run_steering_sweep_faster.py

    # Dry run
    python scripts/run_steering_sweep_faster.py --dry_run

    # MCQ only
    python scripts/run_steering_sweep_faster.py --probe_sources mcq

    # Limit questions
    python scripts/run_steering_sweep_faster.py --max_questions 5
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from itertools import product

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probes import LinearProbeClassification

# Import components from run_causality_test
from run_causality_test import (
    CausalityTester,
    load_probes,
    load_questions,
    save_results,
    PROBE_DIRS,
    QUESTIONS_FILE,
    SKILL_LEVELS,
)


# =============================================================================
# Configuration
# =============================================================================

LAYER_RANGES = [
    # 5-layer ranges
    (45, 50),
    (55, 60),
    (65, 70),
    # 10-layer ranges
    (40, 50),
    (50, 60),
    (60, 70),
]

DEFAULT_STRENGTHS = [1.5, 2.5]
DEFAULT_PROBE_SOURCES = ["mcq", "turn_by_turn"]


# =============================================================================
# Model Loading
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


# =============================================================================
# Sweep Runner
# =============================================================================

def generate_combinations(probe_sources: list, layer_ranges: list, strengths: list) -> list:
    """Generate all parameter combinations."""
    combinations = []
    for probe_source, (from_layer, to_layer), strength in product(
        probe_sources, layer_ranges, strengths
    ):
        combinations.append({
            "probe_source": probe_source,
            "from_layer": from_layer,
            "to_layer": to_layer,
            "strength": strength,
        })
    return combinations


def get_output_dir(base_dir: str, probe_source: str, from_layer: int, to_layer: int, strength: int) -> str:
    """Generate output directory path for a combination."""
    return os.path.join(
        base_dir,
        probe_source,
        f"layers_{from_layer}_{to_layer}",
        f"strength_{strength}",
    )


def run_sweep(
    model,
    tokenizer,
    device: str,
    questions: list[str],
    combinations: list[dict],
    base_output_dir: str,
    model_name: str,
) -> list[dict]:
    """Run the parameter sweep."""
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    results_summary = []
    probes_cache = {}  # Cache probes by probe_source

    print(f"\nRunning sweep with {len(combinations)} combinations...")
    print(f"Questions: {len(questions)}")
    print(f"Output directory: {base_output_dir}")
    print("=" * 60)

    for i, combo in enumerate(tqdm(combinations, desc="Sweep progress")):
        probe_source = combo["probe_source"]
        from_layer = combo["from_layer"]
        to_layer = combo["to_layer"]
        strength = combo["strength"]

        print(f"\n[{i+1}/{len(combinations)}] {probe_source} | layers {from_layer}-{to_layer} | strength {strength}")

        # Load probes (with caching)
        if probe_source not in probes_cache:
            probe_dir = PROBE_DIRS[probe_source]
            print(f"  Loading probes from: {probe_dir}")
            probes_cache[probe_source] = load_probes(probe_dir, hidden_size, num_layers, device)
            print(f"  Loaded {len(probes_cache[probe_source])} probes")

        probes = probes_cache[probe_source]

        if not probes:
            print(f"  ERROR: No probes found for {probe_source}")
            results_summary.append({
                "probe_source": probe_source,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "strength": strength,
                "num_questions": len(questions),
                "completed": False,
                "error": "No probes found",
            })
            continue

        # Create output directory
        output_dir = get_output_dir(base_output_dir, probe_source, from_layer, to_layer, strength)

        # Check if already completed
        responses_file = os.path.join(output_dir, "responses.json")
        if os.path.exists(responses_file):
            print(f"  Skipping (already exists): {output_dir}")
            results_summary.append({
                "probe_source": probe_source,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "strength": strength,
                "num_questions": len(questions),
                "completed": True,
                "skipped": True,
            })
            continue

        try:
            # Create tester with current parameters
            tester = CausalityTester(
                model=model,
                tokenizer=tokenizer,
                probes=probes,
                device=device,
                strength=float(strength),
                from_layer=from_layer,
                to_layer=to_layer,
            )

            # Collect responses
            responses = tester.collect_responses(questions, output_dir=output_dir)

            # Save results
            config = {
                "probe_source": probe_source,
                "model": model_name,
                "strength": strength,
                "from_layer": from_layer,
                "to_layer": to_layer,
            }
            save_results(output_dir, questions, responses, config)

            results_summary.append({
                "probe_source": probe_source,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "strength": strength,
                "num_questions": len(questions),
                "completed": True,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                "probe_source": probe_source,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "strength": strength,
                "num_questions": len(questions),
                "completed": False,
                "error": str(e),
            })

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_summary


def save_summary_csv(results: list[dict], output_path: str):
    """Save results summary to CSV."""
    fieldnames = [
        "probe_source", "from_layer", "to_layer", "strength",
        "num_questions", "completed", "skipped", "error"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Summary saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run faster steering parameter sweep")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default=QUESTIONS_FILE,
        help="Path to questions file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/steering_sweep_faster_results",
        help="Base output directory",
    )
    parser.add_argument(
        "--probe_sources",
        type=str,
        nargs="+",
        choices=["mcq", "turn_by_turn"],
        default=DEFAULT_PROBE_SOURCES,
        help="Probe sources to test",
    )
    parser.add_argument(
        "--strengths",
        type=int,
        nargs="+",
        default=DEFAULT_STRENGTHS,
        help="Steering strengths to test",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print combinations without running",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit number of questions",
    )

    args = parser.parse_args()

    # Generate combinations
    combinations = generate_combinations(
        args.probe_sources,
        LAYER_RANGES,
        args.strengths,
    )

    print("=" * 60)
    print("Faster Steering Parameter Sweep")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Probe sources: {args.probe_sources}")
    print(f"Layer ranges: {LAYER_RANGES}")
    print(f"Strengths: {args.strengths}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - All combinations:")
        print("-" * 60)
        for i, combo in enumerate(combinations):
            print(f"{i+1:3d}. {combo['probe_source']:12s} | "
                  f"layers {combo['from_layer']:2d}-{combo['to_layer']:2d} | "
                  f"strength {combo['strength']}")
        print("-" * 60)
        print(f"Total: {len(combinations)} combinations")
        return

    # Load questions
    questions = load_questions(args.questions_path)
    if not questions:
        print(f"No questions found. Create: {args.questions_path}")
        return

    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Loaded {len(questions)} question(s)")

    # Load model ONCE
    model, tokenizer, device = load_model(args.model)

    # Run sweep
    start_time = datetime.now()
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        device=device,
        questions=questions,
        combinations=combinations,
        base_output_dir=args.output_dir,
        model_name=args.model,
    )
    end_time = datetime.now()

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
    save_summary_csv(results, summary_path)

    # Print summary
    completed = sum(1 for r in results if r.get("completed", False))
    skipped = sum(1 for r in results if r.get("skipped", False))
    failed = sum(1 for r in results if not r.get("completed", False))

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Total combinations: {len(combinations)}")
    print(f"Completed: {completed} (skipped: {skipped})")
    print(f"Failed: {failed}")
    print(f"Duration: {end_time - start_time}")
    print(f"Results: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
