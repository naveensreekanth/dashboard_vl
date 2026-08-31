"""Allowlisted relative paths for agent-input data (never recurse the repo)."""

from __future__ import annotations

CORE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "measurements.csv",
        "parts_dim.csv",
        "lots_dim.csv",
        "test_catalog.csv",
        "current_limits.csv",
        "scenario_manifest_public.csv",
        "README_DATA_CONTRACT.md",
        "DATASET_VERSION.json",
        "rules/disposition_rules.json",
        "rules/limit_simulation_config.json",
    }
)

PARAMETRIC_ALLOWLIST: frozenset[str] = frozenset(
    {
        "measurements.csv",
        "parts_dim.csv",
        "lots_dim.csv",
        "conditions_dim.csv",
        "test_catalog.csv",
        "current_limits.csv",
        "scenario_manifest_public.csv",
        "README_DATA_CONTRACT.md",
        "PARAMETRIC_DATASET_VERSION.json",
        "rules/disposition_rules.json",
        "rules/limit_simulation_config.json",
    }
)

# Path fragments / basenames that must never be loaded as agent input.
FORBIDDEN_PATH_FRAGMENTS: frozenset[str] = frozenset(
    {
        "evaluation/",
        "ground_truth_optimal_limits.csv",
        "scenario_ground_truth.csv",
        "split_assignments.csv",
        "latent_die_table_EVAL_ONLY.csv",
        "process_state_bridge_EVAL_ONLY.csv",
        "generator_config.json",
    }
)

FORBIDDEN_COLUMN_NAMES: frozenset[str] = frozenset(
    {
        "true_optimal_limit",
        "true_optimal_lower",
        "true_optimal_upper",
        "true_optimal_lower_limit",
        "true_optimal_upper_limit",
        "acceptable_lower_limit",
        "acceptable_upper_limit",
        "latent_quality",
        "synthetic_quality_score",
        "objective_score",
        "expected_agent_behavior",
        "scenario_ground_truth",
    }
)
