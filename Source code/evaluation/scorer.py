"""
scorer.py
---------
Composite Evaluation Framework for LLM Code Generation Benchmarking.

Implements the scoring rubric from the research paper:
    S = w1*A + w2*C + w3*R + w4*T
    where:
        A = Functional Accuracy  (w1 = 0.40)
        C = Syntactic Correctness (w2 = 0.25)
        R = Optimisation Quality  (w3 = 0.20)
        T = Response Efficiency   (w4 = 0.15)

Authors: Mayank Bansal, Dharuv Singla, Dev Bhatia
Paper  : Benchmarking Large Language Models for Code Generation
         Under Prompt Variability Using a Composite Evaluation Framework
"""

import ast
import time
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Rubric weights (from paper Section III-B)
# ---------------------------------------------------------------------------
W1_ACCURACY     = 0.40   # Functional accuracy
W2_SYNTAX       = 0.25   # Syntactic correctness
W3_OPTIMISATION = 0.20   # Optimisation quality
W4_EFFICIENCY   = 0.15   # Response (token) efficiency

# Token thresholds for efficiency scoring (empirically set)
TOKEN_IDEAL_MAX  = 600   # responses <= this get full marks
TOKEN_PENALTY_AT = 1200  # responses >= this get 0 efficiency marks


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DimensionScores:
    """Raw scores per rubric dimension, each in [0.0, 1.0]."""
    accuracy:      float = 0.0   # A
    syntax:        float = 0.0   # C
    optimisation:  float = 0.0   # R
    efficiency:    float = 0.0   # T

    def composite(self) -> float:
        """Compute weighted composite score S (0–100 scale)."""
        raw = (
            W1_ACCURACY     * self.accuracy +
            W2_SYNTAX       * self.syntax +
            W3_OPTIMISATION * self.optimisation +
            W4_EFFICIENCY   * self.efficiency
        )
        return round(raw * 100, 2)


@dataclass
class EvaluationResult:
    """Full evaluation result for a single model response."""
    task_id:        str
    model:          str
    prompt_type:    str           # structured | semi_structured | minimal
    category:       str           # algorithm | debugging | optimization | documentation
    scores:         DimensionScores = field(default_factory=DimensionScores)
    composite:      float = 0.0
    token_count:    int   = 0
    latency_s:      float = 0.0
    syntax_error:   Optional[str] = None
    runtime_error:  Optional[str] = None
    notes:          str   = ""

    def to_dict(self) -> dict:
        return {
            "task_id":       self.task_id,
            "model":         self.model,
            "prompt_type":   self.prompt_type,
            "category":      self.category,
            "score_accuracy":     round(self.scores.accuracy * 100, 2),
            "score_syntax":       round(self.scores.syntax * 100, 2),
            "score_optimisation": round(self.scores.optimisation * 100, 2),
            "score_efficiency":   round(self.scores.efficiency * 100, 2),
            "composite_score":    self.composite,
            "token_count":   self.token_count,
            "latency_s":     self.latency_s,
            "syntax_error":  self.syntax_error or "",
            "runtime_error": self.runtime_error or "",
            "notes":         self.notes,
        }


# ---------------------------------------------------------------------------
# Dimension 1 — Syntactic Correctness (C)
# ---------------------------------------------------------------------------

def score_syntax(code: str) -> tuple[float, Optional[str]]:
    """
    Parse the extracted Python code with ast.parse.

    Returns
    -------
    (score, error_message)
        score = 1.0 if no syntax errors, 0.0 otherwise.
    """
    clean = _extract_code_block(code)
    try:
        ast.parse(clean)
        return 1.0, None
    except SyntaxError as e:
        return 0.0, str(e)


# ---------------------------------------------------------------------------
# Dimension 2 — Functional Accuracy (A)
# ---------------------------------------------------------------------------

def score_accuracy(code: str, test_cases: list[dict]) -> tuple[float, Optional[str]]:
    """
    Execute hidden test cases against the extracted code.

    Each test case is a dict with 'input' (string) and 'expected' (string).
    We write a temp script, run it in a subprocess with a timeout,
    and compare stdout to expected output.

    Returns
    -------
    (score, runtime_error)
        score = fraction of test cases passed (0.0 – 1.0)
    """
    if not test_cases:
        return 1.0, None  # no tests = assume correct

    clean = _extract_code_block(code)
    passed = 0
    last_error = None

    for tc in test_cases:
        test_script = (
            f"{clean}\n\n"
            f"result = {_build_call(clean, tc['input'])}\n"
            f"print(result)\n"
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(test_script)
                tmp_path = f.name

            proc = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True, timeout=10
            )
            os.unlink(tmp_path)

            if proc.returncode == 0:
                actual = proc.stdout.strip()
                expected = str(tc["expected"]).strip()
                if _outputs_match(actual, expected):
                    passed += 1
                else:
                    last_error = f"Expected {expected!r}, got {actual!r}"
            else:
                last_error = proc.stderr.strip()[:300]

        except subprocess.TimeoutExpired:
            last_error = "TimeoutExpired (>10s)"
        except Exception as e:
            last_error = str(e)

    score = passed / len(test_cases)
    return round(score, 4), last_error


# ---------------------------------------------------------------------------
# Dimension 3 — Optimisation Quality (R)
# ---------------------------------------------------------------------------

# Known complexity keywords and their tier scores
_COMPLEXITY_TIERS = {
    # Best tier — O(1), O(log n)
    "o(1)": 1.0, "o(log": 1.0,
    # Good tier — O(n), O(n log n)
    "o(n log": 0.85, "o(n)": 0.80,
    # Acceptable — O(n^2) noted but justified
    "o(n^2)": 0.55, "o(n²)": 0.55,
    # Poor — O(2^n) exponential
    "o(2^n)": 0.20,
}

# Pythonic optimisation signals (positive)
_OPTIMISATION_SIGNALS_POSITIVE = [
    r"\bset\s*\(",           # set() for O(1) lookup
    r"\bdict\s*\(",          # dict for hashing
    r"lru_cache",            # memoization
    r"functools",            # standard optimisations
    r"collections\.",        # Counter, deque, defaultdict
    r"heapq\.",              # heap usage
    r"itertools\.",          # lazy evaluation
    r"\.join\s*\(",          # join instead of += for strings
    r"list comprehension",   # mentioned in comments
    r"\[.*for.*in.*\]",      # list comprehension syntax
    r"yield\b",              # generator (memory efficient)
]

# Anti-patterns (negative signals)
_OPTIMISATION_SIGNALS_NEGATIVE = [
    r"for.*range.*len\(",    # old-style iteration
    r'result\s*\+=\s*["\']', # string concat in loop
    r"time\.sleep",          # artificial delays
]


def score_optimisation(code: str, prompt_type: str, category: str) -> float:
    """
    Heuristic optimisation scorer combining:
      - Complexity tier detected in comments/docstring
      - Pythonic patterns used
      - Anti-patterns penalised
      - Documentation tasks scored on clarity, not runtime complexity

    Returns score in [0.0, 1.0].
    """
    if category == "documentation":
        return _score_doc_quality(code)

    clean = _extract_code_block(code).lower()
    base_score = 0.65  # neutral starting point

    # Check for complexity mentions
    for keyword, tier_score in _COMPLEXITY_TIERS.items():
        if keyword in clean:
            base_score = max(base_score, tier_score)
            break

    # Positive signals
    positive_hits = sum(
        1 for pat in _OPTIMISATION_SIGNALS_POSITIVE
        if re.search(pat, clean)
    )
    base_score += min(positive_hits * 0.05, 0.20)  # cap at +0.20

    # Negative signals
    negative_hits = sum(
        1 for pat in _OPTIMISATION_SIGNALS_NEGATIVE
        if re.search(pat, clean)
    )
    base_score -= min(negative_hits * 0.08, 0.24)  # cap at -0.24

    return round(min(max(base_score, 0.0), 1.0), 4)


def _score_doc_quality(text: str) -> float:
    """
    For documentation tasks, score based on presence of
    key documentation components.
    """
    score = 0.0
    checks = {
        r'"""': 0.15,             # has docstring
        r"args|parameters":  0.15, # documents arguments
        r"returns?:": 0.15,        # documents return
        r"raises?:": 0.10,         # documents exceptions
        r"example|>>>": 0.15,      # includes example
        r"#\s+\w": 0.10,           # has inline comments
        r"type\s*:|\bint\b|\bstr\b|\blist\b|\bbool\b": 0.10,  # type hints
        r"notes?:": 0.10,          # has notes
    }
    text_lower = text.lower()
    for pattern, weight in checks.items():
        if re.search(pattern, text_lower):
            score += weight
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Dimension 4 — Response Efficiency (T)
# ---------------------------------------------------------------------------

def score_efficiency(token_count: int) -> float:
    """
    Score response efficiency based on token count.
    Penalises verbose responses; rewards concise ones.

    TOKEN_IDEAL_MAX  → 1.0 (full score)
    TOKEN_PENALTY_AT → 0.0 (zero score)
    Linear interpolation in between.
    """
    if token_count <= TOKEN_IDEAL_MAX:
        return 1.0
    if token_count >= TOKEN_PENALTY_AT:
        return 0.0
    ratio = (TOKEN_PENALTY_AT - token_count) / (TOKEN_PENALTY_AT - TOKEN_IDEAL_MAX)
    return round(ratio, 4)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    task_id: str,
    model: str,
    prompt_type: str,
    category: str,
    response_text: str,
    test_cases: list[dict],
    token_count: int,
    latency_s: float,
) -> EvaluationResult:
    """
    Run all four rubric dimensions and return an EvaluationResult.

    Parameters
    ----------
    task_id       : unique task identifier from tasks.json
    model         : model name string (e.g. 'claude-3-7-sonnet')
    prompt_type   : 'structured' | 'semi_structured' | 'minimal'
    category      : 'algorithm' | 'debugging' | 'optimization' | 'documentation'
    response_text : raw text output from the LLM
    test_cases    : list of {'input': ..., 'expected': ...} dicts
    token_count   : number of output tokens reported by the API
    latency_s     : wall-clock seconds for the API call
    """
    syntax_score, syntax_err = score_syntax(response_text)
    accuracy_score, runtime_err = score_accuracy(response_text, test_cases)
    optim_score = score_optimisation(response_text, prompt_type, category)
    efficiency_score = score_efficiency(token_count)

    dim = DimensionScores(
        accuracy=accuracy_score,
        syntax=syntax_score,
        optimisation=optim_score,
        efficiency=efficiency_score,
    )

    return EvaluationResult(
        task_id=task_id,
        model=model,
        prompt_type=prompt_type,
        category=category,
        scores=dim,
        composite=dim.composite(),
        token_count=token_count,
        latency_s=latency_s,
        syntax_error=syntax_err,
        runtime_error=runtime_err,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_results(results: list[EvaluationResult]) -> dict:
    """
    Compute mean composite score and per-dimension averages
    grouped by model and prompt_type.
    """
    from collections import defaultdict

    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = r.model
        groups[key]["composite"].append(r.composite)
        groups[key]["accuracy"].append(r.scores.accuracy * 100)
        groups[key]["syntax"].append(r.scores.syntax * 100)
        groups[key]["optimisation"].append(r.scores.optimisation * 100)
        groups[key]["efficiency"].append(r.scores.efficiency * 100)
        groups[key]["latency"].append(r.latency_s)
        groups[key]["tokens"].append(r.token_count)

    summary = {}
    for model, metrics in groups.items():
        summary[model] = {
            k: round(sum(v) / len(v), 2) for k, v in metrics.items()
        }
    return summary


def prompt_sensitivity(results: list[EvaluationResult]) -> dict:
    """
    Compute delta (SP score - MP score) and std-dev per model.
    Matches Table V in the paper.
    """
    import statistics
    from collections import defaultdict

    by_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_model[r.model][r.prompt_type].append(r.composite)

    sensitivity = {}
    for model, pt_scores in by_model.items():
        sp  = statistics.mean(pt_scores.get("structured", [0]))
        mp  = statistics.mean(pt_scores.get("minimal", [0]))
        all_scores = [s for scores in pt_scores.values() for s in scores]
        sigma = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        sensitivity[model] = {
            "SP_mean":  round(sp, 2),
            "MP_mean":  round(mp, 2),
            "delta_pts": round(sp - mp, 2),
            "sigma":    round(sigma, 2),
        }
    return sensitivity


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> str:
    """Pull Python code from a markdown fenced block if present."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _build_call(code: str, input_str: str) -> str:
    """
    Attempt to infer the function name from the code and build a call.
    Falls back to eval(input_str) if detection fails.
    """
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if match:
        fn_name = match.group(1)
        return f"{fn_name}({input_str})"
    return input_str


def _outputs_match(actual: str, expected: str) -> bool:
    """
    Flexible comparison: try exact string match first,
    then eval-based comparison for lists/tuples.
    """
    if actual == expected:
        return True
    try:
        return eval(actual) == eval(expected)  # noqa: S307
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python scorer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_code = '''
```python
def merge_sort(arr):
    """Sort array using merge sort. O(n log n) time, O(n) space."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
'''

    test_cases = [
        {"input": "[5, 2, 8, 1, 9]", "expected": "[1, 2, 5, 8, 9]"},
        {"input": "[]",              "expected": "[]"},
        {"input": "[1]",             "expected": "[1]"},
    ]

    result = evaluate(
        task_id="ALGO_001",
        model="test-model",
        prompt_type="structured",
        category="algorithm",
        response_text=sample_code,
        test_cases=test_cases,
        token_count=512,
        latency_s=2.3,
    )

    print("=== Smoke Test: ALGO_001 ===")
    print(f"  Accuracy      : {result.scores.accuracy * 100:.1f}%")
    print(f"  Syntax        : {result.scores.syntax * 100:.1f}%")
    print(f"  Optimisation  : {result.scores.optimisation * 100:.1f}%")
    print(f"  Efficiency    : {result.scores.efficiency * 100:.1f}%")
    print(f"  COMPOSITE     : {result.composite:.1f}%")
    if result.syntax_error:
        print(f"  Syntax Error  : {result.syntax_error}")
    if result.runtime_error:
        print(f"  Runtime Error : {result.runtime_error}")
