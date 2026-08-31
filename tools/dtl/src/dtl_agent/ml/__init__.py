"""Phase 7 ML candidate ranker package."""

__all__ = ["run_phase7_training"]


def __getattr__(name: str):
    """Lazy-load training entrypoint so API startup does not import Phase 7 training."""
    if name == "run_phase7_training":
        from dtl_agent.ml.pipeline import run_phase7_training

        return run_phase7_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
