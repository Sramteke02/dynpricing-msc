"""Layer 5: shared evaluation harness, metrics and experiment driver."""

from dynpricing.eval.metrics import EpisodeMetrics, aggregate
from dynpricing.eval.harness import run_episode, evaluate_agent, run_experiment
from dynpricing.eval.stats import (
    summarize,
    paired_comparison,
    bootstrap_mean_ci,
    resolve_agent_name,
)

__all__ = [
    "EpisodeMetrics",
    "aggregate",
    "run_episode",
    "evaluate_agent",
    "run_experiment",
    "summarize",
    "paired_comparison",
    "bootstrap_mean_ci",
    "resolve_agent_name",
]
