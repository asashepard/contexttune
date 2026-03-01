#!/usr/bin/env python3
"""Generate SWE-smith tasks for a single repository.

Usage:
    python scripts/generate_swesmith_tasks.py \\
        --repo django/django \\
        --commit <sha> \\
        --n-train 200 --n-holdout 50 \\
        --output-dir artifacts/tasks/django__django

Requires ``swesmith`` to be installed (``pip install swesmith``).
If swesmith is not available, the script generates stub task files
from the repo's git history as a fallback.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


# 12 repos used in the experiment
REPOS = [
    "django/django",
    "astropy/astropy",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "pallets/flask",
    "sphinx-doc/sphinx",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "psf/requests",
    "pydata/xarray",
    "mwaskom/seaborn",
]


def generate_tasks_swesmith(
    repo: str,
    commit: str,
    n: int,
    output_path: Path,
) -> int:
    """Generate tasks using the swesmith library."""
    try:
        from swesmith.bug_gen import generate_bugs  # type: ignore
    except Exception as exc:
        print(
            "  [swesmith] generate_bugs API unavailable "
            f"({type(exc).__name__}: {exc}). Falling back to HF dataset..."
        )
        return generate_tasks_from_hf(repo, commit, n, output_path)

    bugs = generate_bugs(repo=repo, commit=commit, n=n)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for bug in bugs:
            record = {
                "instance_id": bug.get("instance_id", f"{repo.replace('/', '__')}__{bug.get('id', '')}"),
                "repo": repo,
                "base_commit": bug.get("base_commit", commit),
                "problem_statement": bug.get("problem_statement", bug.get("issue", "")),
                "source": "swe_smith",
            }
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    return len(bugs)


def _row_to_task(row: dict, repo: str, default_commit: str) -> dict | None:
    """Normalize a SWE-smith row into our task contract."""
    instance_id = row.get("instance_id") or row.get("id")
    row_repo = row.get("repo") or row.get("repository")
    base_commit = row.get("base_commit") or row.get("commit") or default_commit
    problem_statement = (
        row.get("problem_statement")
        or row.get("issue")
        or row.get("problem")
        or row.get("prompt")
        or ""
    )
    if not instance_id or not row_repo or not base_commit:
        return None
    if str(row_repo) != repo:
        return None

    return {
        "instance_id": str(instance_id),
        "repo": str(row_repo),
        "base_commit": str(base_commit),
        "problem_statement": str(problem_statement),
        "source": "swe_smith_hf",
    }


def generate_tasks_from_hf(repo: str, commit: str, n: int, output_path: Path) -> int:
    """Fallback task generator using HF SWE-smith dataset.

    This path is used when local swesmith generation APIs are unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for SWE-smith HF fallback. Install with: pip install datasets"
        ) from exc

    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    rows = []
    for row in ds:
        task = _row_to_task(dict(row), repo=repo, default_commit=commit)
        if task is not None:
            rows.append(task)

    if not rows:
        raise RuntimeError(
            f"No SWE-smith HF rows found for repo={repo}."
        )

    # Prefer matching commit when caller specified one; otherwise keep repo-wide set.
    if commit != "HEAD":
        commit_rows = [r for r in rows if r.get("base_commit") == commit]
        if commit_rows:
            rows = commit_rows

    # Stable pseudo-random sample for reproducibility across runs.
    seed = abs(hash(f"{repo}:{commit}")) % (2 ** 32)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in rows:
            f.write(json.dumps(task, sort_keys=True, ensure_ascii=False) + "\n")

    return len(rows)


def generate_tasks_fallback(
    repo: str,
    commit: str,
    n: int,
    output_path: Path,
) -> int:
    """Generate stub tasks from git history (fallback when swesmith is unavailable).

    Creates synthetic task records using recent commits as pseudo-tasks.
    These are only useful for testing the pipeline — real tuning requires
    swesmith-generated tasks with ground-truth patches.
    """
    import subprocess
    from context_policy.git.checkout import checkout_repo

    repo_dir = checkout_repo(repo, commit)

    result = subprocess.run(
        ["git", "log", f"--max-count={n}", "--format=%H %s"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  git log failed: {result.stderr[:200]}", file=sys.stderr)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    repo_slug = repo.replace("/", "__")

    with output_path.open("w", encoding="utf-8") as f:
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            sha = parts[0]
            msg = parts[1] if len(parts) > 1 else "Fix issue"
            record = {
                "instance_id": f"{repo_slug}__{sha[:8]}",
                "repo": repo,
                "base_commit": sha,
                "problem_statement": f"Issue: {msg}",
                "source": "fallback",
            }
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SWE-smith tasks for a repo")
    parser.add_argument("--repo", required=True, help="Repository slug (e.g. django/django)")
    parser.add_argument("--commit", required=True, help="Base commit SHA")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training tasks")
    parser.add_argument("--n-holdout", type=int, default=50, help="Number of holdout tasks")
    parser.add_argument("--output-dir", required=True, help="Output directory for task files")
    parser.add_argument("--fallback", action="store_true", help="Use git-history fallback instead of swesmith")

    args = parser.parse_args()
    out = Path(args.output_dir)

    gen_fn = generate_tasks_fallback if args.fallback else generate_tasks_swesmith

    print(f"Generating {args.n_train} train tasks for {args.repo}...")
    train_count = gen_fn(args.repo, args.commit, args.n_train, out / "train.jsonl")
    print(f"  -> {train_count} tasks written to {out / 'train.jsonl'}")

    print(f"Generating {args.n_holdout} holdout tasks for {args.repo}...")
    holdout_count = gen_fn(args.repo, args.commit, args.n_holdout, out / "holdout.jsonl")
    print(f"  -> {holdout_count} tasks written to {out / 'holdout.jsonl'}")


if __name__ == "__main__":
    main()
