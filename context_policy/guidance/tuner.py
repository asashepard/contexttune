"""Hill-climbing tuner for per-repo guidance.

Outer loop:
  1. Initialize G₀ from repo introspection + LLM.
  2. Score G₀ on N SWE-smith tasks.
  3. For T iterations:
     a. Propose K candidate edits of the current best G*.
     b. Score each candidate on N tasks.
     c. If any candidate beats G*, adopt it as the new G*.
  4. Save the final G* to disk.

All configuration is captured in ``TuningConfig`` so that the CLI
only has to build one object.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from context_policy.guidance.init import initialize_guidance
from context_policy.guidance.propose import propose_candidates
from context_policy.guidance.schema import DEFAULT_CHAR_BUDGET, RepoGuidance
from context_policy.guidance.score import ScoreResult, score_candidate_detailed
from context_policy.git.checkout import checkout_repo


# ── configuration ──────────────────────────────────────────────


MAX_TUNING_ITERATIONS = 20


@dataclass
class TuningConfig:
    """All knobs for one repo's tuning run."""

    repo: str
    commit: str
    tasks_file: str  # path to SWE-smith JSONL for this repo
    model: str

    # Hill-climbing budget
    iterations: int = 10         # T
    candidates_per_iter: int = 6  # K
    tasks_per_score: int = 20    # N

    # Guidance constraints
    char_budget: int = DEFAULT_CHAR_BUDGET

    # Runner settings
    timeout_s: int = 600
    step_limit: int = 30

    # Output paths
    output_dir: str = ""  # set by caller

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if self.iterations > MAX_TUNING_ITERATIONS:
            raise ValueError(
                f"iterations={self.iterations} exceeds cap {MAX_TUNING_ITERATIONS}"
            )
        if self.candidates_per_iter <= 0:
            raise ValueError("candidates_per_iter must be > 0")
        if self.tasks_per_score <= 0:
            raise ValueError("tasks_per_score must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)


# ── tuning state (for resume) ──────────────────────────────────


@dataclass
class TuningState:
    """Persistent state for a single repo tuning run."""

    repo: str
    best_version: int = 0
    best_score: float = 0.0
    history: list[dict] = field(default_factory=list)
    completed_iterations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TuningState:
        return cls(
            repo=d["repo"],
            best_version=d.get("best_version", 0),
            best_score=d.get("best_score", 0.0),
            history=list(d.get("history", [])),
            completed_iterations=d.get("completed_iterations", 0),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> TuningState:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── main tuning loop ───────────────────────────────────────────


def run_tuning_loop(config: TuningConfig) -> RepoGuidance:
    """Run the full hill-climbing tuning loop for one repository.

    Args:
        config: Fully populated ``TuningConfig``.

    Returns:
        The best ``RepoGuidance`` found.
    """
    out = Path(config.output_dir) if config.output_dir else Path("artifacts/guidance") / config.repo.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / "tuning_state.json"
    metrics_path = out / "tuning_metrics.json"
    guidance_dir = out / "versions"
    guidance_dir.mkdir(parents=True, exist_ok=True)

    tasks_file = Path(config.tasks_file)

    # ── resume or init ─────────────────────────────────────────
    if state_path.exists():
        state = TuningState.load(state_path)
        best_path = guidance_dir / f"v{state.best_version}.json"
        if best_path.exists():
            best = RepoGuidance.load(best_path)
            best_score = state.best_score
            print(f"[tune] Resuming {config.repo} from v{state.best_version} (score={best_score:.1%})")
        else:
            # State file exists but guidance is missing — re-init
            state = TuningState(repo=config.repo)
            best, best_score, init_metrics = _init_and_score(config, tasks_file, guidance_dir)
            state.best_version = best.version
            state.best_score = best_score
            state.history.append({
                "version": 0,
                "score": best_score,
                "type": "init",
                "resolved": init_metrics.resolved,
                "total": init_metrics.total,
                "instance_metrics_path": init_metrics.instance_metrics_path,
            })
            state.save(state_path)
    else:
        state = TuningState(repo=config.repo)
        best, best_score, init_metrics = _init_and_score(config, tasks_file, guidance_dir)
        state.best_version = best.version
        state.best_score = best_score
        state.history.append({
            "version": 0,
            "score": best_score,
            "type": "init",
            "resolved": init_metrics.resolved,
            "total": init_metrics.total,
            "instance_metrics_path": init_metrics.instance_metrics_path,
        })
        state.save(state_path)

    tuning_metrics: dict = {
        "repo": config.repo,
        "model": config.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "iterations": [],
    }

    # ── hill-climbing iterations ───────────────────────────────
    start_iter = state.completed_iterations + 1
    for t in range(start_iter, config.iterations + 1):
        print(f"\n[tune] {config.repo} iteration {t}/{config.iterations}")
        print(f"  Current best: v{best.version} score={best_score:.1%}")

        # Propose K candidates
        score_history = [(h["version"], h["score"]) for h in state.history]
        candidates = propose_candidates(
            guidance=best,
            score=best_score,
            model=config.model,
            k=config.candidates_per_iter,
            history=score_history,
            timeout_s=config.timeout_s,
        )

        if not candidates:
            print(f"  No candidates generated, skipping iteration {t}")
            state.completed_iterations = t
            state.save(state_path)
            continue

        print(f"  Scoring {len(candidates)} candidates...")

        for ci, candidate in enumerate(candidates):
            # Assign unique version numbers
            candidate_version = best.version + ci + 1
            candidate = candidate.copy(version=candidate_version)

            preds_dir = out / "preds" / f"iter{t:02d}" / f"c{ci}"
            c_result = score_candidate_detailed(
                guidance=candidate,
                tasks_file=tasks_file,
                model=config.model,
                n_tasks=config.tasks_per_score,
                timeout_s=config.timeout_s,
                step_limit=config.step_limit,
                preds_dir=preds_dir,
            )
            c_score = c_result.rate

            state.history.append({
                "version": candidate_version,
                "score": c_score,
                "type": "candidate",
                "iteration": t,
                "candidate_index": ci,
                "resolved": c_result.resolved,
                "total": c_result.total,
                "non_empty_patches": c_result.non_empty_patches,
                "elapsed_s": c_result.total_elapsed_s,
                "token_usage": c_result.token_usage,
                "instance_metrics_path": c_result.instance_metrics_path,
            })

            # Save candidate guidance
            candidate.save(guidance_dir / f"v{candidate_version}.json")

            is_improved = c_score > best_score
            if is_improved:
                print(f"  ✓ Candidate {ci} (v{candidate_version}) improves: {best_score:.1%} -> {c_score:.1%}")
                best = candidate
                best_score = c_score
                state.best_version = candidate_version
                state.best_score = best_score
            else:
                print(f"  ✗ Candidate {ci} (v{candidate_version}): {c_score:.1%} <= {best_score:.1%}")

            tuning_metrics["iterations"].append({
                "iteration": t,
                "candidate_index": ci,
                "version": candidate_version,
                "score": c_score,
                "resolved": c_result.resolved,
                "total": c_result.total,
                "non_empty_patch_rate": (c_result.non_empty_patches / c_result.total) if c_result.total else 0.0,
                "elapsed_s": c_result.total_elapsed_s,
                "token_usage": c_result.token_usage,
                "improved_best": is_improved,
                "instance_metrics_path": c_result.instance_metrics_path,
            })

        state.completed_iterations = t
        state.save(state_path)
        metrics_path.write_text(json.dumps(tuning_metrics, indent=2) + "\n", encoding="utf-8")

    # ── save final best ────────────────────────────────────────
    final_path = out / "best_guidance.json"
    best.save(final_path)
    print(f"\n[tune] Done. Best for {config.repo}: v{best.version} score={best_score:.1%}")
    print(f"  Saved to {final_path}")

    # Save config for reproducibility
    config_path = out / "tuning_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    return best


def _init_and_score(
    config: TuningConfig,
    tasks_file: Path,
    guidance_dir: Path,
) -> tuple[RepoGuidance, float, ScoreResult]:
    """Initialize G₀ and score it."""
    print(f"[tune] Initializing guidance for {config.repo}...")
    repo_dir = checkout_repo(config.repo, config.commit)

    g0 = initialize_guidance(
        repo=config.repo,
        commit=config.commit,
        repo_dir=repo_dir,
        model=config.model,
        char_budget=config.char_budget,
        timeout_s=config.timeout_s,
    )
    g0.save(guidance_dir / "v0.json")

    print(f"[tune] Scoring G₀ ({g0.char_count()} chars, {len(g0.lines)} lines)...")
    g0_result = score_candidate_detailed(
        guidance=g0,
        tasks_file=tasks_file,
        model=config.model,
        n_tasks=config.tasks_per_score,
        timeout_s=config.timeout_s,
        step_limit=config.step_limit,
    )
    g0_score = g0_result.rate
    print(f"[tune] G₀ score: {g0_score:.1%}")
    return g0, g0_score, g0_result
