"""
Helix — Fitness Evaluator

Computes a composite fitness score (0.0-1.0) representing how capable
and reliable the agent currently is. Used by the SelfImprovementEngine
to decide whether to commit or revert a self-modification.

A change is committed if:  new_fitness >= baseline - 0.05
A change is reverted if:   new_fitness <  baseline - 0.05
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("helix.core.fitness")


class FitnessEvaluator:
    COMMIT_THRESHOLD   =  0.02
    REVERT_THRESHOLD   = -0.05

    def __init__(self, monitor=None):
        self._monitor = monitor
        self._baseline: Optional[float] = None
        self._baseline_ts: Optional[float] = None

    def set_monitor(self, monitor):
        self._monitor = monitor

    def snapshot_baseline(self) -> float:
        score = self._get_current_fitness()
        self._baseline = score
        self._baseline_ts = time.time()
        logger.info(f"[fitness] Baseline: {score:.4f}")
        return score

    def evaluate_delta(self) -> dict:
        if self._baseline is None:
            return {"verdict": "NO_BASELINE", "baseline": None,
                    "current": self._get_current_fitness(), "delta": 0.0,
                    "should_commit": True, "should_revert": False}

        current = self._get_current_fitness()
        delta = current - self._baseline
        should_revert = delta < self.REVERT_THRESHOLD
        verdict = "DEGRADED" if should_revert else ("IMPROVED" if delta >= self.COMMIT_THRESHOLD else "NEUTRAL")

        result = {
            "verdict": verdict,
            "baseline": round(self._baseline, 4),
            "current": round(current, 4),
            "delta": round(delta, 4),
            "should_commit": not should_revert,
            "should_revert": should_revert,
            "age_seconds": round(time.time() - (self._baseline_ts or time.time()), 1),
        }
        logger.info(f"[fitness] delta={delta:+.4f} verdict={verdict}")
        return result

    def _get_current_fitness(self) -> float:
        if self._monitor is not None:
            return self._monitor.get_current_fitness()
        return 0.5

    def get_current(self) -> float:
        return self._get_current_fitness()


_evaluator: Optional[FitnessEvaluator] = None

def get_evaluator() -> Optional[FitnessEvaluator]:
    return _evaluator

def init_evaluator(monitor=None) -> FitnessEvaluator:
    global _evaluator
    _evaluator = FitnessEvaluator(monitor=monitor)
    return _evaluator
