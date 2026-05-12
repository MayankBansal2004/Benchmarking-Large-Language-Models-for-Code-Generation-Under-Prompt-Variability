"""
runner.py
---------
Benchmark Runner — calls all 6 LLM APIs across all tasks and prompt types,
collects responses, scores them via scorer.py, and saves results to CSV.

Models evaluated (from the paper):
    Western: Claude 3.7 Sonnet, Gemini 2.0 Flash, GPT-4o
    Eastern : GLM-4-Plus, MiniMax-M2, Kimi K2 Instruct

Usage:
    python runner.py                        # run full benchmark
    python runner.py --model claude         # run only Claude
    python runner.py --task ALGO_001        # run one specific task
    python runner.py --prompt structured    # run one prompt type only
    python runner.py --dry-run              # print prompts, no API calls

Requirements (install via pip):
    pip install anthropic openai google-generativeai zhipuai requests pandas tqdm

API Keys (set as environment variables before running):
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="..."
    export ZHIPU_API_KEY="..."
    export MINIMAX_API_KEY="..."
    export MOONSHOT_API_KEY="..."
"""

import os
import json
import time
import argparse
import csv
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── local import ────────────────────────────────────────────────────────────
from scorer import evaluate, aggregate_results, prompt_sensitivity, EvaluationResult

# ── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
TASKS_FILE    = ROOT / "benchmark" / "tasks.json"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_CSV   = RESULTS_DIR / f"scores_{TIMESTAMP}.csv"
SUMMARY_JSON  = RESULTS_DIR / f"summary_{TIMESTAMP}.json"

# ── model config ─────────────────────────────────────────────────────────────
MODELS = {
    "claude": {
        "display":  "Claude 3.7 Sonnet",
        "origin":   "West",
        "api":      "anthropic",
        "model_id": "claude-3-7-sonnet-20250219",
    },
    "gemini": {
        "display":  "Gemini 2.0 Flash",
        "origin":   "West",
        "api":      "google",
        "model_id": "gemini-2.0-flash-001",
    },
    "gpt4o": {
        "display":  "GPT-4o",
        "origin":   "West",
        "api":      "openai",
        "model_id": "gpt-4o-2024-11-20",
    },
    "glm": {
        "display":  "GLM-4-Plus",
        "origin":   "East",
        "api":      "zhipu",
        "model_id": "glm-4-plus",
    },
    "minimax": {
        "display":  "MiniMax-M2",
        "origin":   "East",
        "api":      "minimax",
        "model_id": "abab6.5s-chat",
    },
    "kimi": {
        "display":  "Kimi K2 Instruct",
        "origin":   "East",
        "api":      "moonshot",
        "model_id": "kimi-k2-0711-preview",
    },
}

PROMPT_TYPES  = ["structured", "semi_structured", "minimal"]
RUNS_PER_TASK = 3          # average over 3 runs to reduce variance (paper §IV)
TEMPERATURE   = 0.0        # deterministic (paper §IV-A)
REQUEST_DELAY = 1.5        # seconds between API calls (rate limiting)


# ─────────────────────────────────────────────────────────────────────────────
# API callers — one function per provider
# ─────────────────────────────────────────────────────────────────────────────

def call_anthropic(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call Claude via Anthropic SDK. Returns (text, tokens, latency_s)."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    msg = client.messages.create(
        model=model_id,
        max_tokens=1024,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.time() - t0, 3)
    text   = msg.content[0].text
    tokens = msg.usage.output_tokens
    return text, tokens, latency


def call_openai(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call GPT-4o via OpenAI SDK. Returns (text, tokens, latency_s)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=TEMPERATURE,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.time() - t0, 3)
    text   = resp.choices[0].message.content
    tokens = resp.usage.completion_tokens
    return text, tokens, latency


def call_google(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call Gemini via Google Generative AI SDK. Returns (text, tokens, latency_s)."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_id)
    config = genai.types.GenerationConfig(
        temperature=TEMPERATURE,
        max_output_tokens=1024,
    )
    t0 = time.time()
    resp = model.generate_content(prompt, generation_config=config)
    latency = round(time.time() - t0, 3)
    text   = resp.text
    tokens = resp.usage_metadata.candidates_token_count
    return text, tokens, latency


def call_zhipu(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call GLM-4-Plus via Zhipu AI SDK. Returns (text, tokens, latency_s)."""
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=os.environ["ZHIPU_API_KEY"])
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=TEMPERATURE,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.time() - t0, 3)
    text   = resp.choices[0].message.content
    tokens = resp.usage.completion_tokens
    return text, tokens, latency


def call_minimax(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call MiniMax-M2 via REST API. Returns (text, tokens, latency_s)."""
    import requests
    url     = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model_id,
        "temperature": TEMPERATURE,
        "max_tokens":  1024,
        "messages":    [{"role": "user", "content": prompt}],
    }
    t0   = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    latency = round(time.time() - t0, 3)
    data  = resp.json()
    text  = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    return text, tokens, latency


def call_moonshot(model_id: str, prompt: str) -> tuple[str, int, float]:
    """Call Kimi K2 via Moonshot AI (OpenAI-compatible endpoint)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.cn/v1",
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=TEMPERATURE,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round(time.time() - t0, 3)
    text   = resp.choices[0].message.content
    tokens = resp.usage.completion_tokens
    return text, tokens, latency


# ── dispatcher ───────────────────────────────────────────────────────────────

API_CALLERS = {
    "anthropic": call_anthropic,
    "openai":    call_openai,
    "google":    call_google,
    "zhipu":     call_zhipu,
    "minimax":   call_minimax,
    "moonshot":  call_moonshot,
}


def call_model(model_key: str, prompt: str) -> tuple[str, int, float]:
    """Dispatch to the correct API caller based on model config."""
    cfg    = MODELS[model_key]
    caller = API_CALLERS[cfg["api"]]
    return caller(cfg["model_id"], prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    model_filter:  Optional[str] = None,
    task_filter:   Optional[str] = None,
    prompt_filter: Optional[str] = None,
    dry_run:       bool = False,
) -> list[EvaluationResult]:
    """
    Main benchmark loop.

    For each model × task × prompt_type:
      1. Send the prompt RUNS_PER_TASK times
      2. Score each response via scorer.evaluate()
      3. Average the composite scores
      4. Store one averaged EvaluationResult

    Returns list of all EvaluationResult objects.
    """
    tasks = _load_tasks()
    all_results: list[EvaluationResult] = []

    models_to_run = (
        {model_filter: MODELS[model_filter]}
        if model_filter and model_filter in MODELS
        else MODELS
    )

    total = (
        len(models_to_run)
        * len([t for t in tasks if not task_filter or t["id"] == task_filter])
        * len([p for p in PROMPT_TYPES if not prompt_filter or p == prompt_filter])
    )
    done = 0

    print(f"\n{'='*60}")
    print(f"  LLM Benchmark Suite — {TIMESTAMP}")
    print(f"  Models : {list(models_to_run.keys())}")
    print(f"  Tasks  : {task_filter or 'all'}")
    print(f"  Prompts: {prompt_filter or 'all'}")
    print(f"  Runs/task: {RUNS_PER_TASK}   Dry-run: {dry_run}")
    print(f"  Total API calls: ~{total * RUNS_PER_TASK}")
    print(f"{'='*60}\n")

    # ── open CSV writer ──────────────────────────────────────────────────────
    csv_file   = open(RESULTS_CSV, "w", newline="") if not dry_run else None
    csv_writer = None
    if csv_file:
        fieldnames = list(EvaluationResult.__dataclass_fields__.keys()) + [
            "score_accuracy", "score_syntax", "score_optimisation", "score_efficiency"
        ]
        # Use to_dict() keys instead
        csv_writer = None  # will init after first row

    rows_buffer = []

    for model_key, model_cfg in models_to_run.items():
        for task in tasks:
            if task_filter and task["id"] != task_filter:
                continue

            test_cases = task.get("test_cases", [])
            category   = task["category"]

            for pt in PROMPT_TYPES:
                if prompt_filter and pt != prompt_filter:
                    continue

                prompt = task["prompts"].get(pt, task["prompts"]["structured"])
                done  += 1
                label  = f"[{done}/{total}] {model_cfg['display']} | {task['id']} | {pt}"

                if dry_run:
                    print(f"\n{label}")
                    print(f"  PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
                    continue

                print(f"  {label} ... ", end="", flush=True)

                run_results = []
                for run_idx in range(RUNS_PER_TASK):
                    try:
                        response_text, tokens, latency = call_model(model_key, prompt)

                        result = evaluate(
                            task_id=task["id"],
                            model=model_cfg["display"],
                            prompt_type=pt,
                            category=category,
                            response_text=response_text,
                            test_cases=test_cases,
                            token_count=tokens,
                            latency_s=latency,
                        )
                        run_results.append(result)

                    except Exception as e:
                        print(f"\n    ⚠ Run {run_idx+1} failed: {e}")
                        traceback.print_exc()

                    time.sleep(REQUEST_DELAY)

                if not run_results:
                    print("SKIPPED (all runs failed)")
                    continue

                # Average scores across runs
                avg_result = _average_runs(run_results, task["id"], model_cfg["display"], pt, category)
                all_results.append(avg_result)
                rows_buffer.append(avg_result.to_dict())

                print(f"✓  composite={avg_result.composite:.1f}%  latency={avg_result.latency_s:.1f}s")

    # ── write CSV ────────────────────────────────────────────────────────────
    if rows_buffer and not dry_run:
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows_buffer[0].keys())
            writer.writeheader()
            writer.writerows(rows_buffer)
        print(f"\n✅ Results saved → {RESULTS_CSV}")

    # ── write summary JSON ───────────────────────────────────────────────────
    if all_results and not dry_run:
        summary = {
            "aggregate":          aggregate_results(all_results),
            "prompt_sensitivity": prompt_sensitivity(all_results),
            "meta": {
                "timestamp":    TIMESTAMP,
                "runs_per_task": RUNS_PER_TASK,
                "temperature":  TEMPERATURE,
                "total_tasks":  len(tasks),
                "total_models": len(models_to_run),
            }
        }
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved  → {SUMMARY_JSON}")

        _print_leaderboard(summary["aggregate"])
        _print_sensitivity(summary["prompt_sensitivity"])

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return json.load(f)


def _average_runs(
    runs: list[EvaluationResult],
    task_id: str,
    model: str,
    prompt_type: str,
    category: str,
) -> EvaluationResult:
    """Average numeric scores across multiple runs of the same task."""
    from scorer import DimensionScores

    n = len(runs)
    avg_dim = DimensionScores(
        accuracy=     sum(r.scores.accuracy     for r in runs) / n,
        syntax=       sum(r.scores.syntax       for r in runs) / n,
        optimisation= sum(r.scores.optimisation for r in runs) / n,
        efficiency=   sum(r.scores.efficiency   for r in runs) / n,
    )
    return EvaluationResult(
        task_id=task_id,
        model=model,
        prompt_type=prompt_type,
        category=category,
        scores=avg_dim,
        composite=avg_dim.composite(),
        token_count=int(sum(r.token_count for r in runs) / n),
        latency_s=round(sum(r.latency_s for r in runs) / n, 3),
        syntax_error=runs[-1].syntax_error,
        runtime_error=runs[-1].runtime_error,
        notes=f"averaged over {n} runs",
    )


def _print_leaderboard(aggregate: dict) -> None:
    """Print a formatted leaderboard table to console."""
    print("\n" + "="*60)
    print("  LEADERBOARD — Mean Composite Score")
    print("="*60)
    ranked = sorted(aggregate.items(), key=lambda x: x[1]["composite"], reverse=True)
    for rank, (model, metrics) in enumerate(ranked, 1):
        bar = "█" * int(metrics["composite"] / 5)
        print(f"  #{rank} {model:<22} {metrics['composite']:>6.1f}%  {bar}")
    print("="*60)


def _print_sensitivity(sensitivity: dict) -> None:
    """Print prompt sensitivity table (mirrors Table V in the paper)."""
    print("\n  PROMPT SENSITIVITY (SP → MP drop)")
    print(f"  {'Model':<22} {'SP':>6} {'MP':>6} {'Δ pts':>7} {'σ':>6}")
    print("  " + "-"*48)
    ranked = sorted(sensitivity.items(), key=lambda x: x[1]["delta_pts"])
    for model, s in ranked:
        print(
            f"  {model:<22} {s['SP_mean']:>6.1f} {s['MP_mean']:>6.1f}"
            f" {s['delta_pts']:>7.1f} {s['sigma']:>6.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run the LLM code-generation benchmark."
    )
    p.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Run only this model (default: all)",
    )
    p.add_argument(
        "--task",
        help="Run only this task ID, e.g. ALGO_001 (default: all)",
    )
    p.add_argument(
        "--prompt",
        choices=PROMPT_TYPES,
        help="Run only this prompt type (default: all)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without making API calls",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_TASK,
        help=f"Runs per task (default: {RUNS_PER_TASK})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Override global runs if specified
    if args.runs != RUNS_PER_TASK:
        RUNS_PER_TASK = args.runs

    run_benchmark(
        model_filter=args.model,
        task_filter=args.task,
        prompt_filter=args.prompt,
        dry_run=args.dry_run,
    )
