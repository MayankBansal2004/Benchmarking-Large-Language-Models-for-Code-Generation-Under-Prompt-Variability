# Benchmarking Large Language Models for Code Generation Under Prompt Variability Using a Composite Evaluation Framework
### (2210991904, 2210990268, 2210991487)

---

## Team Details

| Roll Number   | Name          | Email                              | Role              |
|---------------|---------------|------------------------------------|-------------------|
| 2210991904    | Mayank Bansal | mayank1904.be22@chitkara.edu.in    | Lead Researcher   |
| 2210990268    | Dharuv Singla | dharuv268.be22@chitkara.edu.in     | Co-Researcher     |
| 2210991487    | Dev Bhatia    | dev1487.be22@chitkara.edu.in       | Co-Researcher     |
| —             | Preeti Saini  | preeti.saini@chitkara.edu.in       | Faculty Supervisor|

**Institution:** Chitkara University Institute of Engineering and Technology, Chitkara University, Punjab, India

---

## Project Title

Benchmarking Large Language Models for Code Generation Under Prompt Variability Using a Composite Evaluation Framework

---

## Type

**Research Paper** (IEEE Conference Format)

---

## Project Description

This project presents a controlled empirical evaluation of six state-of-the-art Large Language Models (LLMs) on 150 coding tasks across three prompt formality levels — Structured, Semi-Structured, and Minimal. A Composite Evaluation Framework scores each model across four dimensions: Functional Accuracy, Syntactic Correctness, Optimisation Quality, and Response Efficiency.

Models evaluated: Claude 3.7 Sonnet, Gemini 2.0 Flash, GPT-4o (Western), GLM-4-Plus, MiniMax-M2, Kimi K2 Instruct (Eastern).

---

## Repository Structure

```
├── IPR_Submission_Proof/       # Research paper PDF + submission screenshot
├── Report_and_PPT/             # Project report and presentation slides
├── Source_Code/                # Python benchmark and scoring source code
│   ├── benchmark/
│   │   └── tasks.json          # 150 coding tasks (3 prompt variants each)
│   ├── evaluation/
│   │   ├── runner.py           # Calls all 6 LLM APIs and runs benchmark
│   │   └── scorer.py           # Composite scoring engine
│   └── results/
│       └── scores.csv          # Full experimental results
└── README.md
```

---

## Current Status

**Submitted** — Research paper compiled in IEEE IEEEtran LaTeX format and submitted for conference review.

---

## Key Results Summary

| Model          | Origin | Mean Score | Rank |
|----------------|--------|------------|------|
| Claude 3.7 Sonnet | West | 91.3%   | 1st  |
| Kimi K2 Instruct  | East | 88.6%   | 2nd  |
| Gemini 2.0 Flash  | West | 87.0%   | 3rd  |
| GLM-4-Plus        | East | 84.2%   | 4th  |
| GPT-4o            | West | 82.7%   | 5th  |
| MiniMax-M2        | East | 81.5%   | 6th  |
