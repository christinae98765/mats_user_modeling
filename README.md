# Skill Representation

Research project investigating how LLMs internally represent user skill/understanding levels, and whether these representations can be detected and steered.

## Overview

This project trains probes to detect user skill level (novice/intermediate/expert) from LLM hidden states, and tests whether these representations causally influence model outputs.

**Key experiments:**
1. Train reading and control probes on two datasets (MCQ conversations, turn-by-turn tutoring)
2. Test causal influence via activation steering
3. Analyze how skill representations evolve over conversation turns
4. Compare explicit model classification vs. implicit probe detection

## Setup

### Lambda Labs (NVIDIA GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for model access)
huggingface-cli login
```

### Local (CPU only - slow)

```bash
pip install -r requirements.txt
```

## Datasets

- `data/mcq_conversations/txt/` - Multiple-choice Q&A conversations (skill encoded via correct/incorrect answers)
- `data/tutoring_turn_by_turn/txt/` - Open-ended tutoring conversations

Both datasets have files named: `conversation_{id}_understanding_{level}.txt` where level is `novice`, `intermediate`, or `expert`.

## Scripts

### 1. Train Probes

Train reading and control probes on hidden state activations:

```bash
# Train on MCQ data (both reading and control probes)
python scripts/train_probes.py --data mcq

# Train on turn-by-turn data
python scripts/train_probes.py --data turn_by_turn

# Train only one probe type
python scripts/train_probes.py --data mcq --probe_type reading
python scripts/train_probes.py --data mcq --probe_type control
```

**Reading probes:** Trained with appended prompt "I think the understanding level of this user is"
**Control probes:** Trained on raw conversation (no appended prompt)

### 2. Run Causality Tests

Test if probes can steer model outputs:

```bash
# Steer using MCQ-trained probes
python scripts/run_causality_test.py --probe_source mcq

# Steer using turn-by-turn trained probes
python scripts/run_causality_test.py --probe_source turn_by_turn

# Adjust steering parameters
python scripts/run_causality_test.py --probe_source mcq --strength 5 --from_layer 16 --to_layer 24
```

### 3. Analyze Probe Over Conversation

See how skill predictions evolve over conversation turns:

```bash
# Analyze a single file
python scripts/analyze_probe_over_conversation.py --file data/mcq_conversations/txt/conversation_0_understanding_novice.txt

# Analyze multiple files
python scripts/analyze_probe_over_conversation.py --dir data/mcq_conversations/txt/ --num-files 10

# Use specific layer
python scripts/analyze_probe_over_conversation.py --file <path> --layer 15
```

### 4. Test Model Classification

Test if the model can explicitly classify user skill when prompted:

```bash
# Test on MCQ conversations
python scripts/test_model_classification.py --conversation_dir data/mcq_conversations/txt --num_samples 5

# Test single file
python scripts/test_model_classification.py --conversation_file data/mcq_conversations/txt/conversation_0_understanding_novice.txt
```

## Project Structure

```
Skill_Representation/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_probes.py           # Train reading/control probes
│   ├── run_causality_test.py     # Steering experiments
│   ├── analyze_probe_over_conversation.py  # Turn-by-turn analysis
│   └── test_model_classification.py        # Explicit classification test
├── src/
│   └── probes.py                 # Probe model classes
└── data/
    ├── mcq_conversations/txt/    # MCQ dataset
    ├── tutoring_turn_by_turn/txt/  # Turn-by-turn dataset
    ├── causality_test_questions/ # Questions for steering tests
    └── probe_checkpoints/        # Saved probe weights
```

## Models

Default model: `meta-llama/Llama-3.3-70B-Instruct` (with 4-bit quantization)

Can also use smaller models for testing:
- `meta-llama/Llama-3.1-8B-Instruct`