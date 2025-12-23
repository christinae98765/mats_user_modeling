#!/usr/bin/env python3
"""
Run causality test for sycophancy questions using understanding probes.

Configuration:
- Model: meta-llama/Llama-3.1-8B-Instruct
- Attribute: understanding (novice/intermediate/expert)
- Layers: 15-18
- Strength: 3
- Questions: data/causality_test_questions/sycophancy.txt
- Output: data/causal_intervention_outputs/tutoring/final_layers/

Usage:
    conda activate talktuner-mps  # or talktuner-gpu
    python scripts/run_sycophancy_causality_test.py
"""

import subprocess
import sys
import os

# Configuration
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ATTRIBUTE = "understanding"
PROBE_DIR = "data/probe_checkpoints_tutoring/controlling_probe"
QUESTIONS_PATH = "data/causality_test_questions/gradschool.txt"
OUTPUT_DIR = "data/causal_intervention_outputs/tutoring/final_layers_grad_school"

# Intervention parameters
FROM_LAYER = 15
TO_LAYER = 18
STRENGTH = 3


def main():
    print("=" * 60)
    print("Sycophancy Causality Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Attribute: {ATTRIBUTE}")
    print(f"Probe dir: {PROBE_DIR}")
    print(f"Questions: {QUESTIONS_PATH}")
    print(f"Layers: {FROM_LAYER}-{TO_LAYER - 1}")
    print(f"Strength: {STRENGTH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_causality_test_llama3.py",
        "--attribute", ATTRIBUTE,
        "--model", MODEL,
        "--probe_dir", PROBE_DIR,
        "--questions_path", QUESTIONS_PATH,
        "--strength", str(STRENGTH),
        "--from_layer", str(FROM_LAYER),
        "--to_layer", str(TO_LAYER),
        "--output_dir", OUTPUT_DIR,
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run the causality test
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: Test failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("Sycophancy causality test completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()