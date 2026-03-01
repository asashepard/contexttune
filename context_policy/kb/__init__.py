"""Knowledge Base package for structured repo context artifacts.

Public API::

    from context_policy.kb import build_kb, RepoKB
"""
from __future__ import annotations

from context_policy.kb.builder import build_kb
from context_policy.kb.schema import RepoKB

__all__ = ["build_kb", "RepoKB"]
