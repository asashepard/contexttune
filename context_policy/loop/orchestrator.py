"""Per-repo guidance tuning orchestrator.

Supports three experimental conditions:
1. ``no_context``: agent sees only the issue + tree (baseline).
2. ``static_kb``: agent sees tree-sitter KB rendered as AGENTS.md (no LLM tuning).
3. ``oracle_tuned``: agent sees AGENTS.md refined by the LLM-as-judge oracle loop.

Evaluation on SWE-bench Verified is unchanged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.guidance.schema import RepoGuidance
from context_policy.oracle.loop import run_oracle_loop
from context_policy.oracle.schema import OracleConfig
from context_policy.runner.mini_swe_agent_swebench import generate_patch_with_mini_swebench_result
from context_policy.utils.jsonl import read_jsonl
from context_policy.utils.paths import PREDS_DIR, PROJECT_ROOT, RESULTS_DIR
from context_policy.utils.subproc import run as subproc_run


# Valid experimental conditions
VALID_CONDITIONS = ("no_context", "static_kb", "oracle_tuned")


# ── data classes ───────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_id: str
    model: str
    repos: list[dict]  # each: {"repo": str, "commit": str}

    # Conditions to evaluate
    conditions: list[str] = field(default_factory=lambda: list(VALID_CONDITIONS))

    # Oracle tuning hyperparams
    oracle_iterations: int = 5

    # Runner settings
    timeout_s: int = 600
    step_limit: int = 30

    # Eval settings
    eval_dataset: str = "princeton-nlp/SWE-bench_Verified"
    eval_split: str = "test"
    eval_instance_ids_file: str | None = None
    max_workers_eval: int = 4

    def __post_init__(self) -> None:
        for c in self.conditions:
            if c not in VALID_CONDITIONS:
                raise ValueError(
                    f"Unknown condition '{c}'. Valid: {VALID_CONDITIONS}"
                )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentState:
    """Persistent experiment state for resume support."""

    experiment_id: str
    created_at: str = ""
    tuning_completed: list[str] = field(default_factory=list)
    eval_completed: list[str] = field(default_factory=list)  # "<repo>__<condition>"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentState:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)


# ── run experiment ─────────────────────────────────────────────


def run_experiment(config: ExperimentConfig, *, dry_run: bool = False) -> Path:
    """Run the full tuning + evaluation experiment.

    Phase 1: For each repo, build KB and run oracle loop (if conditions require).
    Phase 2: Evaluate on SWE-bench Verified under selected conditions.

    Args:
        config: Experiment configuration.
        dry_run: If True, skip inference (produce empty patches).

    Returns:
        Path to the experiment results directory.
    """
    exp_root = RESULTS_DIR / config.experiment_id
    exp_root.mkdir(parents=True, exist_ok=True)
    state_path = exp_root / "experiment_state.json"

    if state_path.exists():
        state = ExperimentState.load(state_path)
    else:
        state = ExperimentState(
            experiment_id=config.experiment_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        state.save(state_path)

    # Save config
    config_path = exp_root / "experiment_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    # Optional repo restriction: if eval instance IDs are provided, tune/prep only
    # repos that appear in those eval instances.
    eval_instances_prefiltered: list[dict] | None = None
    restricted_repos: set[str] | None = None
    if config.eval_instance_ids_file:
        prefilter_ids = read_instance_ids(config.eval_instance_ids_file)
        eval_instances_prefiltered = load_instances(
            dataset_name=config.eval_dataset,
            split=config.eval_split,
            instance_ids=prefilter_ids,
        )
        restricted_repos = {inst["repo"] for inst in eval_instances_prefiltered}

        repo_order = [r["repo"] for r in config.repos]
        restricted_in_config = [r for r in repo_order if r in restricted_repos]
        skipped_count = len(repo_order) - len(restricted_in_config)
        print(
            "[experiment] eval_instance_ids provided: restricting tuning/prep "
            f"to {len(restricted_in_config)}/{len(repo_order)} repos "
            f"(skipped {skipped_count}): {', '.join(restricted_in_config)}"
        , flush=True)

    # ── Phase 1: Build KB + Oracle tuning ──────────────────────
    # Determine which repos need oracle tuning
    needs_kb = "static_kb" in config.conditions or "oracle_tuned" in config.conditions
    needs_oracle = "oracle_tuned" in config.conditions

    guidance_map: dict[str, dict[str, RepoGuidance]] = {}
    #  guidance_map[repo][condition] = RepoGuidance

    for repo_info in config.repos:
        repo = repo_info["repo"]
        commit = repo_info["commit"]

        if restricted_repos is not None and repo not in restricted_repos:
            continue

        guidance_map.setdefault(repo, {})

        if not needs_kb:
            if repo in state.tuning_completed:
                print(f"[experiment] Skipping KB for {repo} (no KB conditions)")
            continue

        if repo in state.tuning_completed:
            # Load previously computed artifacts
            g_dir = exp_root / "guidance" / repo.replace("/", "__")
            static_path = g_dir / "versions" / "v0.json"
            best_path = g_dir / "best_guidance.json"

            if static_path.exists():
                guidance_map[repo]["static_kb"] = RepoGuidance.load(static_path)
            if best_path.exists():
                guidance_map[repo]["oracle_tuned"] = RepoGuidance.load(best_path)
            print(f"[experiment] Skipping {repo} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[experiment] Processing {repo}")
        print(f"{'='*60}")

        oracle_iters = config.oracle_iterations if needs_oracle else 0
        out_dir = str(exp_root / "guidance" / repo.replace("/", "__"))

        if dry_run:
            g = RepoGuidance(repo=repo, commit=commit, lines=["- (dry run)"], version=0)
            g_out = Path(out_dir)
            g_out.mkdir(parents=True, exist_ok=True)
            g.save(g_out / "best_guidance.json")
            (g_out / "versions").mkdir(parents=True, exist_ok=True)
            g.save(g_out / "versions" / "v0.json")
            guidance_map[repo]["static_kb"] = g
            guidance_map[repo]["oracle_tuned"] = g
        else:
            oc = OracleConfig(
                repo=repo,
                commit=commit,
                model=config.model,
                iterations=oracle_iters,
                timeout_s=config.timeout_s,
                output_dir=out_dir,
            )
            kb, best = run_oracle_loop(oc)

            # v0 is always the static_kb version
            v0_path = Path(out_dir) / "versions" / "v0.json"
            if v0_path.exists():
                guidance_map[repo]["static_kb"] = RepoGuidance.load(v0_path)
            else:
                guidance_map[repo]["static_kb"] = best

            guidance_map[repo]["oracle_tuned"] = best

        state.tuning_completed.append(repo)
        state.save(state_path)

    # ── Phase 2: Verified evaluation ───────────────────────────
    print(f"\n{'='*60}")
    print(f"[experiment] Phase 2: SWE-bench Verified evaluation")
    print(f"{'='*60}")

    instance_ids = None
    if eval_instances_prefiltered is not None:
        eval_instances = eval_instances_prefiltered
    else:
        if config.eval_instance_ids_file:
            instance_ids = read_instance_ids(config.eval_instance_ids_file)

        eval_instances = load_instances(
            dataset_name=config.eval_dataset,
            split=config.eval_split,
            instance_ids=instance_ids,
        )
    print(f"  Loaded {len(eval_instances)} eval instances")

    # Group instances by repo
    instances_by_repo: dict[str, list[dict]] = {}
    for inst in eval_instances:
        instances_by_repo.setdefault(inst["repo"], []).append(inst)

    eval_results: dict[str, dict] = {}

    for condition in config.conditions:
        cond_preds_path = PREDS_DIR / config.experiment_id / condition / "preds.jsonl"
        cond_preds_path.parent.mkdir(parents=True, exist_ok=True)
        cond_metrics_path = exp_root / "metrics" / f"{condition}_instances.jsonl"
        cond_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume support
        completed_ids: set[str] = set()
        completed_metrics: dict[str, dict] = {}
        if cond_preds_path.exists():
            completed_ids = {r["instance_id"] for r in read_jsonl(cond_preds_path)}
        if cond_metrics_path.exists():
            for rec in read_jsonl(cond_metrics_path):
                completed_metrics[rec["instance_id"]] = rec

        condition_elapsed_s = 0.0
        condition_non_empty = 0
        condition_total = 0
        condition_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        total_instances = sum(len(v) for v in instances_by_repo.values())
        done_count = len(completed_ids)

        for repo, instances in instances_by_repo.items():
            key = f"{repo}__{condition}"
            if key in state.eval_completed:
                continue

            guidance_text = None
            if condition in ("static_kb", "oracle_tuned"):
                repo_guidance = guidance_map.get(repo, {}).get(condition)
                if repo_guidance is not None:
                    guidance_text = repo_guidance.render()

            for i, instance in enumerate(instances):
                iid = instance["instance_id"]
                if iid in completed_ids:
                    prev = completed_metrics.get(iid)
                    if prev:
                        condition_elapsed_s += float(prev.get("elapsed_s", 0.0) or 0.0)
                        if prev.get("patch_non_empty"):
                            condition_non_empty += 1
                        usage = prev.get("token_usage", {})
                        condition_tokens["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                        condition_tokens["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                        condition_tokens["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                        condition_total += 1
                    continue

                try:
                    if dry_run:
                        patch = ""
                        run_meta = {
                            "elapsed_s": 0.0,
                            "token_usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "status": "dry_run",
                            "error": None,
                        }
                    else:
                        run_meta = generate_patch_with_mini_swebench_result(
                            instance=instance,
                            model=config.model,
                            context_md=guidance_text,
                            timeout_s=config.timeout_s,
                            step_limit=config.step_limit,
                            traj_dir=PREDS_DIR / config.experiment_id / condition / "trajectories",
                        )
                        patch = run_meta.get("patch", "")
                except Exception as exc:
                    print(f"  Eval error {iid}: {exc}", file=sys.stderr)
                    patch = ""
                    run_meta = {
                        "elapsed_s": 0.0,
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "status": "error",
                        "error": str(exc),
                    }

                record = {
                    "instance_id": iid,
                    "model_name_or_path": config.model,
                    "model_patch": patch,
                }
                with cond_preds_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

                usage = run_meta.get("token_usage", {}) if isinstance(run_meta, dict) else {}
                metrics_record = {
                    "instance_id": iid,
                    "repo": repo,
                    "condition": condition,
                    "elapsed_s": float(run_meta.get("elapsed_s", 0.0) or 0.0),
                    "patch_non_empty": bool(patch and patch.strip()),
                    "status": run_meta.get("status", "ok"),
                    "error": run_meta.get("error"),
                    "token_usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    },
                }
                with cond_metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_record, sort_keys=True, ensure_ascii=False) + "\n")

                condition_elapsed_s += metrics_record["elapsed_s"]
                if metrics_record["patch_non_empty"]:
                    condition_non_empty += 1
                condition_total += 1
                condition_tokens["prompt_tokens"] += metrics_record["token_usage"]["prompt_tokens"]
                condition_tokens["completion_tokens"] += metrics_record["token_usage"]["completion_tokens"]
                condition_tokens["total_tokens"] += metrics_record["token_usage"]["total_tokens"]

                done_count += 1
                status = "OK" if patch else "EMPTY"
                print(f"  [{condition}] [{done_count}/{total_instances}] {iid} -> {status}")

            state.eval_completed.append(key)
            state.save(state_path)

        # Run SWE-bench evaluation harness
        run_id = f"{config.experiment_id}__{condition}"
        eval_cmd = [
            "bash", "scripts/run_swebench_eval.sh",
            config.eval_dataset,
            str(cond_preds_path),
            run_id,
            str(config.max_workers_eval),
        ]
        logs_dir = exp_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        rc = subproc_run(
            eval_cmd,
            cwd=PROJECT_ROOT,
            stdout_path=logs_dir / f"eval_{condition}.stdout.log",
            stderr_path=logs_dir / f"eval_{condition}.stderr.log",
            timeout_s=3600,
        )
        if rc != 0:
            print(f"  WARNING: eval harness failed for {condition}")

        # Load results
        try:
            from context_policy.report.summarize import compute_rate, load_results
            resolved, total = load_results(RESULTS_DIR / run_id)
            eval_results[condition] = {
                "run_id": run_id,
                "resolved": resolved,
                "total": total,
                "rate": compute_rate(resolved, total),
                "preds_path": str(cond_preds_path),
                "instance_metrics_path": str(cond_metrics_path),
                "generation_metrics": {
                    "instances_processed": condition_total,
                    "patch_non_empty": condition_non_empty,
                    "patch_non_empty_rate": (condition_non_empty / condition_total) if condition_total else 0.0,
                    "elapsed_s": condition_elapsed_s,
                    "mean_elapsed_s": (condition_elapsed_s / condition_total) if condition_total else 0.0,
                    "token_usage": condition_tokens,
                },
            }
        except Exception as exc:
            print(f"  WARNING: could not load results for {condition}: {exc}")
            eval_results[condition] = {"error": str(exc)}

    # ── Summary ────────────────────────────────────────────────
    summary = {
        "experiment_id": config.experiment_id,
        "model": config.model,
        "repos": [r["repo"] for r in config.repos],
        "conditions": config.conditions,
        "oracle_iterations": config.oracle_iterations,
        "eval_results": eval_results,
    }

    # Compute deltas between conditions
    nc = eval_results.get("no_context", {})
    sk = eval_results.get("static_kb", {})
    ot = eval_results.get("oracle_tuned", {})
    deltas: dict[str, dict] = {}
    if "rate" in nc and "rate" in sk:
        deltas["static_kb_vs_no_context"] = {
            "absolute": sk["rate"] - nc["rate"],
            "no_context_rate": nc["rate"],
            "static_kb_rate": sk["rate"],
        }
    if "rate" in nc and "rate" in ot:
        deltas["oracle_tuned_vs_no_context"] = {
            "absolute": ot["rate"] - nc["rate"],
            "no_context_rate": nc["rate"],
            "oracle_tuned_rate": ot["rate"],
        }
    if "rate" in sk and "rate" in ot:
        deltas["oracle_tuned_vs_static_kb"] = {
            "absolute": ot["rate"] - sk["rate"],
            "static_kb_rate": sk["rate"],
            "oracle_tuned_rate": ot["rate"],
        }
    if deltas:
        summary["deltas"] = deltas

    summary_path = exp_root / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[experiment] Summary written to {summary_path}")
    print(json.dumps(summary, indent=2))

    return exp_root
