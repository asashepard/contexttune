#!/usr/bin/env python3
"""Hard-trim repository by deleting generated run bloat.

Default behavior is aggressive deletion:
- Delete all generated outputs under `artifacts/` and `results/`
- Delete Python bytecode caches and root Slurm logs

Optional: keep summary snapshots with `--keep_summaries`.
"""
from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        def _onerror(func, failed_path, exc_info):
            failed = Path(failed_path)
            try:
                failed.chmod(stat.S_IWRITE)
            except Exception:
                pass
            func(failed_path)

        shutil.rmtree(path, onerror=_onerror)
    else:
        try:
            path.chmod(stat.S_IWRITE)
        except Exception:
            pass
        path.unlink()


def copy_summary_snapshots(results_dir: Path) -> int:
    retained_root = results_dir / "_retained_summaries"
    ensure_dir(retained_root)

    copied = 0
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "_retained_summaries":
            continue
        summary_src = run_dir / "summary.json"
        if not summary_src.exists():
            continue
        summary_dst = retained_root / run_dir.name / "summary.json"
        ensure_dir(summary_dst.parent)
        shutil.copy2(summary_src, summary_dst)
        copied += 1
    return copied


def trim_results(results_dir: Path) -> int:
    if not results_dir.exists():
        return 0

    deleted = 0
    for child in sorted(results_dir.iterdir()):
        remove_path(child)
        deleted += 1
    return deleted


def trim_artifacts(artifacts_dir: Path) -> int:
    generated_subdirs = [
        "logs",
        "preds",
        "repos_cache",
        "worktrees",
        "signals",
        "contexts",
    ]
    deleted = 0
    for name in generated_subdirs:
        target = artifacts_dir / name
        if target.exists():
            remove_path(target)
            deleted += 1
        ensure_dir(target)
    return deleted


def trim_python_caches(repo_root: Path) -> tuple[int, int]:
    pycache_deleted = 0
    pyc_deleted = 0

    for pycache_dir in repo_root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            remove_path(pycache_dir)
            pycache_deleted += 1

    for pyc_file in repo_root.rglob("*.pyc"):
        if pyc_file.is_file():
            remove_path(pyc_file)
            pyc_deleted += 1

    return pycache_deleted, pyc_deleted


def trim_root_slurm_logs(repo_root: Path) -> int:
    deleted = 0
    for pattern in ("slurm-*.out", "slurm-*.err"):
        for log_path in repo_root.glob(pattern):
            if log_path.is_file():
                remove_path(log_path)
                deleted += 1
    return deleted


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-delete generated run outputs.")
    parser.add_argument(
        "--keep_summaries",
        action="store_true",
        help="Retain summary snapshots under results/_retained_summaries/.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "artifacts"
    results_dir = repo_root / "results"

    ensure_dir(artifacts_dir)
    ensure_dir(results_dir)

    before_artifacts_files = count_files(artifacts_dir)
    before_results_files = count_files(results_dir)

    copied_summaries = 0
    if args.keep_summaries:
        copied_summaries = copy_summary_snapshots(results_dir)

    deleted_results_entries = trim_results(results_dir)

    if args.keep_summaries:
        copied_summaries = copy_summary_snapshots(results_dir)

    deleted_artifact_dirs = trim_artifacts(artifacts_dir)
    pycache_deleted, pyc_deleted = trim_python_caches(repo_root)
    deleted_slurm_logs = trim_root_slurm_logs(repo_root)

    after_artifacts_files = count_files(artifacts_dir)
    after_results_files = count_files(results_dir)

    print("Hard trim complete")
    print(f"- Keep summaries: {args.keep_summaries}")
    print(f"- Before files: artifacts={before_artifacts_files}, results={before_results_files}")
    print(f"- After files: artifacts={after_artifacts_files}, results={after_results_files}")
    print(f"- Summary snapshots retained: {copied_summaries}")
    print(f"- Results entries deleted: {deleted_results_entries}")
    print(f"- Artifact subdirs reset: {deleted_artifact_dirs}")
    print(f"- __pycache__ dirs deleted: {pycache_deleted}")
    print(f"- .pyc files deleted: {pyc_deleted}")
    print(f"- Root slurm logs deleted: {deleted_slurm_logs}")


if __name__ == "__main__":
    main()
