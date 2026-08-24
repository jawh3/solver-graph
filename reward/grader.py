"""
reward/grader.py — difficulty grading: ground-truth-first, calibrated grader fallback (§9.2, C-6).

Difficulty/category drive the standardization cells (§9.2). Source priority:
  (1) GROUND TRUTH  — picoCTF point value / CTF difficulty tag. Always preferred.
  (2) CALIBRATED GRADER — a *blind* difficulty estimate, used ONLY when no ground
      truth exists, and validated against observed solve rate.

Two hard rules from the design review (C-6):
  - The grader is NOT the solver model. Qwen-grades-Qwen has self-preference bias
    (JudgingTheJudges 2406.07791; CannotSelfCorrect 2310.01798). Use a different
    model on the real path; the mock grader is deliberately a non-Qwen heuristic.
  - The grader is calibrated: maintain grader-bucket → empirical-solve-rate and
    report agreement. An A-02 arm runs a *second, non-Qwen* grader to measure
    family/self-preference bias.

`difficulty_source` is persisted on every RunRecord so the eval can separate
ground-truth cells from grader-estimated cells.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Canonical difficulty buckets (the standardization cell axis).
DIFFICULTY_BUCKETS = ["easy", "medium", "hard", "expert"]


def difficulty_from_ground_truth(challenge: Dict) -> Optional[str]:
    """Map picoCTF points / explicit tags to a difficulty bucket. None if absent."""
    # explicit tag wins
    tag = challenge.get("difficulty")
    if tag and str(tag).strip().lower() in DIFFICULTY_BUCKETS:
        return str(tag).strip().lower()
    # picoCTF point value -> bucket (standard pico tiers)
    pts = challenge.get("points")
    try:
        pts = int(pts)
    except (TypeError, ValueError):
        pts = None
    if pts is not None:
        if pts <= 100:
            return "easy"
        if pts <= 200:
            return "medium"
        if pts <= 400:
            return "hard"
        return "expert"
    return None


@dataclass
class GradeResult:
    difficulty: str
    source: str   # "ground_truth" | "grader"
    grader_model: str = ""


class DifficultyGrader(ABC):
    """Ground-truth-first grader. Subclasses implement only `_estimate`."""

    grader_model: str = "abstract"

    def grade(self, challenge: Dict) -> GradeResult:
        gt = difficulty_from_ground_truth(challenge)
        if gt is not None:
            return GradeResult(difficulty=gt, source="ground_truth")
        est = self._estimate(challenge)
        return GradeResult(difficulty=est, source="grader", grader_model=self.grader_model)

    @abstractmethod
    def _estimate(self, challenge: Dict) -> str:
        ...


class MockDifficultyGrader(DifficultyGrader):
    """GPU-free, deterministic, NON-Qwen heuristic grader.

    Estimates difficulty from cheap surface features (description length, presence
    of high-difficulty keywords, number of artifacts). It is intentionally a
    *different* estimator than the solver model — satisfying the "grader ≠ solver"
    rule without a GPU.
    """

    grader_model = "mock-heuristic-grader"

    _HARD_KW = ("heap", "kernel", "rop", "format string", "race", "blind",
                "side channel", "lattice", "elliptic", "obfuscat", "kernel")

    def _estimate(self, challenge: Dict) -> str:
        desc = (challenge.get("description", "") or "").lower()
        n_files = len(challenge.get("local_files", []) or challenge.get("files", []) or [])
        score = 0
        score += min(len(desc) // 200, 3)          # longer prose ~ harder
        score += sum(1 for kw in self._HARD_KW if kw in desc)
        score += 1 if n_files >= 3 else 0
        idx = min(score, len(DIFFICULTY_BUCKETS) - 1)
        return DIFFICULTY_BUCKETS[idx]


class VLLMDifficultyGrader(DifficultyGrader):
    """Real grader against a NON-solver model endpoint (coordinated GPU runs).

    Must point at a different model than the solver (`grader_model != solver`).
    The prompt asks for a difficulty bucket only; the call is blind to the solve
    attempt (§9.2). Kept thin; the heavy lifting (client) is injected.
    """

    def __init__(self, client, model_name: str):
        self._client = client
        self.grader_model = model_name

    def _estimate(self, challenge: Dict) -> str:
        prompt = (
            "Classify this CTF challenge difficulty as exactly one of "
            f"{DIFFICULTY_BUCKETS}. Respond with the single word only.\n\n"
            f"Category: {challenge.get('category')}\n"
            f"Description: {challenge.get('description','')[:1500]}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.grader_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8,
            )
            out = (resp.choices[0].message.content or "").strip().lower()
            for b in DIFFICULTY_BUCKETS:
                if b in out:
                    return b
        except Exception:
            pass
        return "medium"   # safe default if the grader is unreachable/garbled


class GraderCalibration:
    """Tracks grader-bucket → empirical solve rate, and reports agreement (§9.2).

    Validation logic: an honest difficulty estimate should be monotonically
    *anti*-correlated with solve rate (harder → solved less). We report the
    per-bucket solve rate and the Spearman-sign agreement so the C5 dev log can
    state whether the grader is trustworthy before it is used for standardization.
    """

    def __init__(self):
        self._n: Dict[str, int] = defaultdict(int)
        self._solved: Dict[str, int] = defaultdict(int)

    def observe(self, graded_difficulty: str, solved: bool) -> None:
        self._n[graded_difficulty] += 1
        if solved:
            self._solved[graded_difficulty] += 1

    def solve_rate_by_bucket(self) -> Dict[str, float]:
        return {b: self._solved[b] / self._n[b] for b in self._n if self._n[b] > 0}

    def monotonic_agreement(self) -> float:
        """+1 if solve rate strictly decreases across easy→expert, −1 if increases.

        Fraction of adjacent ordered bucket pairs whose solve-rate ordering is
        consistent with 'harder solves less' (a simple calibration score in
        [−1, 1]).
        """
        rates = self.solve_rate_by_bucket()
        order = [b for b in DIFFICULTY_BUCKETS if b in rates]
        if len(order) < 2:
            return 0.0
        consistent = 0
        total = 0
        for i in range(len(order) - 1):
            total += 1
            if rates[order[i]] >= rates[order[i + 1]]:
                consistent += 1
        # map fraction-consistent in [0,1] to [-1,1]
        return 2.0 * (consistent / total) - 1.0

    def report(self) -> Dict:
        return {
            "solve_rate_by_bucket": self.solve_rate_by_bucket(),
            "counts": dict(self._n),
            "monotonic_agreement": self.monotonic_agreement(),
        }
