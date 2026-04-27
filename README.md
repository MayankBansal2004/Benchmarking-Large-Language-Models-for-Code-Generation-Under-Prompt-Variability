# LLM Benchmark Suite — Code Generation Under Prompt Variability

> **Research Paper:** Benchmarking Large Language Models for Code Generation Under Prompt Variability Using a Composite Evaluation Framework
>
> **Authors:** Mayank Bansal · Dharuv Singla · Dev Bhatia
>
> **Institution:** Chitkara University Institute of Engineering & Technology, Punjab, India

---

## Overview

This repository contains the complete source code, benchmark dataset, and experimental results for our study evaluating **six large language models** on **150 coding tasks** across **three prompt formality levels**.

We introduce a **Composite Evaluation Framework** that scores each model response across four dimensions:

| Dimension | Symbol | Weight | Description |
|-----------|--------|--------|-------------|
| Functional Accuracy | A | **0.40** | Fraction of hidden test cases passed |
| Syntactic Correctness | C | **0.25** | No Python syntax errors (`ast.parse`) |
| Optimisation Quality | R | **0.20** | Complexity tier + Pythonic patterns |
| Response Efficiency | T | **0.15** | Token count (penalises verbosity) |

**Composite Score:** `S = 0.40×A + 0.25×C + 0.20×R + 0.15×T`

---

## Key Results

### Table III — Model Performance by Prompt Type (%)

| Model | Origin | Structured | Semi-Structured | Minimal | **Mean** | Rank |
|-------|--------|-----------|----------------|---------|----------|------|
| Claude 3.7 Sonnet | West | **96.1** | 92.8 | 85.0 | **91.3** | 🥇 1st |
| Kimi K2 Instruct | East | 89.1 | **89.4** | **87.4** | 88.6 | 🥈 2nd |
| Gemini 2.0 Flash | West | 92.0 | 87.5 | 81.5 | 87.0 | 🥉 3rd |
| GLM-4-Plus | East | 87.3 | 85.1 | 80.2 | 84.2 | 4th |
| GPT-4o | West | 89.0 | 82.8 | 76.4 | 82.7 | 5th |
| MiniMax-M2 | East | 84.5 | 82.0 | 78.0 | 81.5 | 6th |

### Table IV — Performance by Task Category (%)

| Model | Algorithm | Debugging | Optimisation | Documentation |
|-------|-----------|-----------|--------------|---------------|
| Claude 3.7 Sonnet | **94.2** | **95.0** | 91.5 | 88.7 |
| Kimi K2 Instruct | 90.1 | 87.5 | 88.3 | **91.2** |
| Gemini 2.0 Flash | 88.4 | 87.0 | **89.5** | 84.3 |
| GLM-4-Plus | 85.2 | 83.6 | 87.5 | 81.0 |
| GPT-4o | 84.0 | 80.5 | 83.2 | 83.4 |
| MiniMax-M2 | 82.7 | 81.3 | 82.1 | 80.6 |

### Table V — Prompt Sensitivity (SP → MP drop)

| Model | SP (%) | MP (%) | Δ (pts) | σ |
|-------|--------|--------|---------|---|
| Kimi K2 Instruct | 89.1 | 87.4 | **1.7** | 0.94 |
| MiniMax-M2 | 84.5 | 78.0 | 6.5 | 2.78 |
| GLM-4-Plus | 87.3 | 80.2 | 7.1 | 2.98 |
| Gemini 2.0 Flash | 92.0 | 81.5 | 10.5 | 4.33 |
| Claude 3.7 Sonnet | 96.1 | 85.0 | 11.1 | 4.59 |
| GPT-4o | 89.0 | 76.4 | **12.6** | 5.14 |

### Table VII — Latency and Token Efficiency

| Model | Avg Latency (s) | Avg Output Tokens |
|-------|-----------------|-------------------|
| MiniMax-M2 | **3.2** | **578** |
| GLM-4-Plus | 3.9 | 610 |
| Gemini 2.0 Flash | 4.8 | 695 |
| Kimi K2 Instruct | 5.1 | 743 |
| Claude 3.7 Sonnet | 6.4 | 812 |
| GPT-4o | 7.2 | 924 |

---

## Repository Structure

```
llm-benchmark-suite/
│
├── benchmark/
│   └── tasks.json              # 150 coding tasks (Algorithm, Debug, Optimise, Docs)
│                               # Each task has 3 prompt variants: structured,
│                               # semi-structured, minimal
│
├── evaluation/
│   ├── scorer.py               # Composite scoring engine (4 dimensions)
│   └── runner.py               # Calls all 6 LLM APIs, runs benchmark, saves results
│
├── results/
│   └── scores.csv              # Full experimental results (matches paper tables)
│
└── README.md
```

---

## Models Evaluated

| Key | Model | Provider | Origin | API |
|-----|-------|----------|--------|-----|
| `claude` | Claude 3.7 Sonnet (`claude-3-7-sonnet-20250219`) | Anthropic | West | Anthropic SDK |
| `gemini` | Gemini 2.0 Flash (`gemini-2.0-flash-001`) | Google DeepMind | West | Google GenAI SDK |
| `gpt4o` | GPT-4o (`gpt-4o-2024-11-20`) | OpenAI | West | OpenAI SDK |
| `glm` | GLM-4-Plus (`glm-4-plus`) | Zhipu AI | East | ZhipuAI SDK |
| `minimax` | MiniMax-M2 (`abab6.5s-chat`) | MiniMax | East | REST API |
| `kimi` | Kimi K2 Instruct (`kimi-k2-0711-preview`) | Moonshot AI | East | OpenAI-compatible |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/llm-benchmark-suite.git
cd llm-benchmark-suite

# 2. Install dependencies
pip install anthropic openai google-generativeai zhipuai requests pandas tqdm
```

---

## Configuration — API Keys

Set your API keys as environment variables before running:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ZHIPU_API_KEY="..."
export MINIMAX_API_KEY="..."
export MOONSHOT_API_KEY="..."
```

> You only need keys for the models you want to run.

---

## Usage

### Run the full benchmark
```bash
cd evaluation
python runner.py
```

### Run a single model
```bash
python runner.py --model claude      # Claude 3.7 Sonnet only
python runner.py --model kimi        # Kimi K2 only
```

### Run a single task
```bash
python runner.py --task ALGO_001
```

### Run a specific prompt type
```bash
python runner.py --prompt structured
python runner.py --prompt minimal
```

### Dry run (no API calls — just prints prompts)
```bash
python runner.py --dry-run
```

### Test the scorer independently
```bash
python scorer.py    # runs built-in smoke test on merge sort
```

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Total tasks | 150 |
| Prompt types | Structured · Semi-Structured · Minimal |
| Task categories | Algorithm · Debugging · Optimisation · Documentation |
| Runs per prompt | 3 (averaged to reduce variance) |
| Temperature | 0.0 (deterministic) |
| Language | Python 3.11 |
| Hardware | Intel Core i9-13900K · 64 GB DDR5 · Ubuntu 22.04 LTS |
| Inter-rater κ | 0.86 |
| Rubric weights | w1=0.40, w2=0.25, w3=0.20, w4=0.15 |

---

## Benchmark Dataset

The 150 tasks are drawn from:
- **HumanEval** (40 tasks) — Chen et al., 2021
- **MBPP** (30 tasks) — Austin et al., 2021
- **LeetCode medium** (50 tasks)
- **Practitioner-authored** (30 tasks) — real-world messy prompts

Each task has **three prompt variants**:

| Type | Description | Example |
|------|-------------|---------|
| **Structured (SP)** | Full specification: task, I/O format, edge cases, complexity | *"Implement merge_sort(arr)... O(n log n)... handle empty arrays..."* |
| **Semi-Structured (SSP)** | Context without requirements | *"Implement merge sort in Python."* |
| **Minimal (MP)** | Raw keywords only | *"merge sort python"* |

---

## Scoring Details

### Dimension 1 — Functional Accuracy (A, weight=0.40)
Each response is executed against hidden test cases in a subprocess with a 10-second timeout. Score = fraction of tests passed.

### Dimension 2 — Syntactic Correctness (C, weight=0.25)
The extracted Python code is parsed with `ast.parse()`. Score = 1.0 if no syntax errors, 0.0 otherwise.

### Dimension 3 — Optimisation Quality (R, weight=0.20)
Heuristic scorer combining:
- Complexity tier detected in comments (O(1) → 1.0, O(n log n) → 0.85, O(n²) → 0.55, etc.)
- Pythonic pattern bonuses: `set()`, `lru_cache`, `Counter`, list comprehensions, `join`, generators
- Anti-pattern penalties: string concat in loops, old-style iteration
- Documentation tasks scored on presence of docstring components

### Dimension 4 — Response Efficiency (T, weight=0.15)
Linear penalty based on token count:
- ≤ 600 tokens → 1.0
- ≥ 1200 tokens → 0.0
- Linear interpolation between

---

## Practical Deployment Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Debugging & Algorithm tasks | **Claude 3.7 Sonnet** | 95.0% debug, 94.2% algorithm |
| Teams with varied prompt styles | **Kimi K2 Instruct** | Lowest sensitivity Δ=1.7, σ=0.94 |
| Code documentation generation | **Kimi K2 Instruct** | Best docs score 91.2% |
| Code optimisation | **Gemini 2.0 Flash** | 89.5% + fast 4.8s latency |
| Latency-sensitive workflows | **MiniMax-M2** | Fastest at 3.2s |
| Budget-conscious optimisation | **GLM-4-Plus** | 87.5% optim at 3.9s latency |

---

## Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@article{bansal2025llmbenchmark,
  title   = {Benchmarking Large Language Models for Code Generation Under
             Prompt Variability Using a Composite Evaluation Framework},
  author  = {Bansal, Mayank and Singla, Dharuv and Bhatia, Dev},
  year    = {2025},
  institution = {Chitkara University Institute of Engineering and Technology}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
