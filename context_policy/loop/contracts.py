"""Typed contracts for the experiment loop.

These dataclasses are used by the orchestrator and CLI.
The old RoundState / LoopState contracts are replaced by the
ExperimentConfig / ExperimentState in orchestrator.py and
TuningConfig / TuningState in guidance/tuner.py.
"""
from __future__ import annotations

# Re-export from orchestrator for backward-compatible imports
from context_policy.loop.orchestrator import ExperimentConfig, ExperimentState

__all__ = ["ExperimentConfig", "ExperimentState"]
