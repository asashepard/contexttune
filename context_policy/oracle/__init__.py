"""Oracle evaluator package for LLM-as-judge context tuning.

Public API::

    from context_policy.oracle import run_oracle_loop, OracleConfig
"""
from __future__ import annotations

from context_policy.oracle.loop import run_oracle_loop
from context_policy.oracle.schema import OracleConfig, OracleState

__all__ = ["run_oracle_loop", "OracleConfig", "OracleState"]
