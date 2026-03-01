"""Oracle evaluator loop — replaces the SWE-Smith hill-climbing tuner.

Flow:
1. Checkout repo, run probes → build RepoKB (deterministic).
2. Render initial AGENTS.md from KB → RepoGuidance v0 (``static_kb``).
3. Generate micro-test probes from KB.
4. For T iterations:
   a. Evaluate all probes against current AGENTS.md (simulate + judge).
   b. Collect failures (probes with any FAIL verdict).
   c. If no failures, stop early.
   d. diagnose_failures() → structured edits.
   e. apply_edits() → updated AGENTS.md.
   f. Wrap into RepoGuidance v(N+1), save version.
5. Save final best AGENTS.md + KB.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from context_policy.git.checkout import checkout_repo
from context_policy.guidance.schema import RepoGuidance
from context_policy.kb.agents_md import render_agents_md
from context_policy.kb.builder import build_kb
from context_policy.kb.schema import RepoKB
from context_policy.oracle.apply import apply_edits
from context_policy.oracle.diagnose import diagnose_failures
from context_policy.oracle.judge import evaluate_probe
from context_policy.oracle.probes import generate_probes
from context_policy.oracle.schema import OracleConfig, OracleState
from context_policy.probes import run_all_probes


def run_oracle_loop(config: OracleConfig) -> tuple[RepoKB, RepoGuidance]:
    """Run the full oracle evaluator loop for one repository.

    Args:
        config: Fully populated ``OracleConfig``.

    Returns:
        Tuple of (RepoKB, best RepoGuidance found).
    """
    out = (
        Path(config.output_dir) if config.output_dir
        else Path("artifacts/guidance") / config.repo.replace("/", "__")
    )
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / "tuning_state.json"
    kb_dir = out / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    guidance_dir = out / "versions"
    guidance_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Checkout + probes + KB ─────────────────────────
    print(f"[oracle] Checking out {config.repo} @ {config.commit}...")
    repo_dir = checkout_repo(config.repo, config.commit)

    print(f"[oracle] Running probes on {config.repo}...")
    probe_results = run_all_probes(repo_dir)

    # Save raw probe results
    _save_probe_results_summary(probe_results, kb_dir / "probes_summary.json")

    kb = build_kb(config.repo, config.commit, probe_results)
    kb.save(kb_dir / "kb.json")
    print(
        f"[oracle] KB built: arch={len(kb.architecture)} chars, "
        f"map={len(kb.symbol_map)} chars, "
        f"ctx={len(kb.context)} chars, "
        f"conv={len(kb.conventions)} chars"
    )

    # ── Step 2: Initial AGENTS.md (static_kb) ─────────────────
    agents_md = render_agents_md(kb)
    v0_guidance = RepoGuidance(
        repo=config.repo,
        commit=config.commit,
        lines=agents_md.splitlines(),
        version=0,
    )
    v0_guidance.save(guidance_dir / "v0.json")
    # Also save as standalone AGENTS.md
    (kb_dir / "agents_md_v0.md").write_text(agents_md, encoding="utf-8")
    print(f"[oracle] Static AGENTS.md: {len(agents_md)} chars, {len(v0_guidance.lines)} lines")

    # If iterations == 0, return the static KB result immediately
    if config.iterations <= 0:
        final_path = out / "best_guidance.json"
        v0_guidance.save(final_path)
        return kb, v0_guidance

    # ── Step 3: Generate micro-test probes ─────────────────────
    probes = generate_probes(kb)
    print(f"[oracle] Generated {len(probes)} probes: "
          f"{', '.join(p.category for p in probes)}")

    # ── Resume support ─────────────────────────────────────────
    if state_path.exists():
        state = OracleState.load(state_path)
        if state.completed_iterations > 0:
            best_path = guidance_dir / f"v{state.best_version}.json"
            if best_path.exists():
                best = RepoGuidance.load(best_path)
                agents_md = best.render()
                print(f"[oracle] Resuming from v{state.best_version} "
                      f"(pass_rate={state.best_pass_rate:.1%})")
            else:
                state = OracleState(repo=config.repo)
        else:
            state = OracleState(repo=config.repo)
    else:
        state = OracleState(repo=config.repo)

    # Score initial v0 if not already done
    if not state.history:
        v0_pass_rate = _evaluate_all_probes(agents_md, probes, config)
        state.best_pass_rate = v0_pass_rate
        state.history.append({
            "version": 0,
            "pass_rate": v0_pass_rate,
            "type": "init",
        })
        state.save(state_path)
        print(f"[oracle] v0 pass rate: {v0_pass_rate:.1%}")

    best = v0_guidance
    best_pass_rate = state.best_pass_rate

    # ── Step 4: Oracle iterations ──────────────────────────────
    start_iter = state.completed_iterations + 1
    for t in range(start_iter, config.iterations + 1):
        print(f"\n[oracle] {config.repo} iteration {t}/{config.iterations}")
        print(f"  Current best: v{best.version} pass_rate={best_pass_rate:.1%}")

        # 4a: Evaluate all probes
        results = _evaluate_all_probes_detailed(agents_md, probes, config)

        # 4b: Collect failures
        failures = [r for r in results if r.pass_rate < 1.0]
        if not failures:
            print(f"  All probes pass — stopping early at iteration {t}")
            state.completed_iterations = t
            state.save(state_path)
            break

        fail_count = sum(
            1 for r in results for v in r.verdicts if not v.passed
        )
        total_behaviors = sum(len(r.verdicts) for r in results)
        print(f"  Failures: {fail_count}/{total_behaviors} behaviors failed")

        # 4c: Diagnose failures
        edits = diagnose_failures(agents_md, failures, config.model, timeout_s=config.timeout_s)
        if not edits:
            print(f"  No edits proposed — skipping iteration {t}")
            state.completed_iterations = t
            state.save(state_path)
            continue
        print(f"  Proposed {len(edits)} edits: "
              f"{', '.join(f'{e.action}@{e.section}' for e in edits)}")

        # 4d: Apply edits
        new_agents_md = apply_edits(agents_md, edits, config.model, timeout_s=config.timeout_s)

        # 4e: Score new version
        new_pass_rate = _evaluate_all_probes(new_agents_md, probes, config)
        new_version = best.version + 1

        new_guidance = RepoGuidance(
            repo=config.repo,
            commit=config.commit,
            lines=new_agents_md.splitlines(),
            version=new_version,
        )
        new_guidance.save(guidance_dir / f"v{new_version}.json")

        state.history.append({
            "version": new_version,
            "pass_rate": new_pass_rate,
            "type": "oracle_iteration",
            "iteration": t,
            "edits_count": len(edits),
            "edits": [{"section": e.section, "action": e.action, "content": e.content} for e in edits],
        })

        if new_pass_rate >= best_pass_rate:
            print(f"  ✓ v{new_version} improves: {best_pass_rate:.1%} → {new_pass_rate:.1%}")
            best = new_guidance
            best_pass_rate = new_pass_rate
            agents_md = new_agents_md
            state.best_version = new_version
            state.best_pass_rate = new_pass_rate
        else:
            print(f"  ✗ v{new_version}: {new_pass_rate:.1%} < {best_pass_rate:.1%} (keeping current)")

        state.completed_iterations = t
        state.save(state_path)

    # ── Save final best ───────────────────────────────────────
    final_path = out / "best_guidance.json"
    best.save(final_path)
    print(f"\n[oracle] Done. Best for {config.repo}: "
          f"v{best.version} pass_rate={best_pass_rate:.1%}")
    print(f"  Saved to {final_path}")

    # Save config for reproducibility
    config_path = out / "oracle_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8",
    )

    return kb, best


def _evaluate_all_probes(
    agents_md: str,
    probes: list,
    config: OracleConfig,
) -> float:
    """Evaluate all probes and return aggregate pass rate."""
    results = _evaluate_all_probes_detailed(agents_md, probes, config)
    if not results:
        return 0.0
    total_passed = sum(
        1 for r in results for v in r.verdicts if v.passed
    )
    total_behaviors = sum(len(r.verdicts) for r in results)
    return total_passed / total_behaviors if total_behaviors > 0 else 0.0


def _evaluate_all_probes_detailed(
    agents_md: str,
    probes: list,
    config: OracleConfig,
) -> list:
    """Evaluate all probes and return detailed results."""
    from context_policy.oracle.schema import Probe as ProbeType
    from context_policy.oracle.schema import ProbeResult

    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        print(f"    Evaluating probe {i+1}/{len(probes)}: [{probe.category}] {probe.id}...")
        try:
            result = evaluate_probe(
                agents_md, probe, config.model, timeout_s=config.timeout_s,
            )
            results.append(result)
            status = f"{result.pass_rate:.0%}"
            print(f"      → {status}")
        except Exception as exc:
            print(f"      → ERROR: {exc}")
            results.append(ProbeResult(
                probe_id=probe.id,
                category=probe.category,
                verdicts=[],
                pass_rate=0.0,
            ))
    return results


def _save_probe_results_summary(probe_results, path: Path) -> None:
    """Save a JSON summary of probe results (not the full trees)."""
    summary = {
        "repo_dir": probe_results.repo_dir,
        "hub_count": len(probe_results.imports.hubs),
        "hub_files": [h.file for h in probe_results.imports.hubs],
        "symbol_count": len(probe_results.symbols.entries),
        "entry_point_count": len(probe_results.entry_points.entries),
        "cluster_count": len(probe_results.clusters.clusters),
        "chain_count": len(probe_results.clusters.chains),
        "integration_count": len(probe_results.clusters.integrations),
        "test_command": probe_results.tests.test_command,
        "test_dir_count": len(probe_results.tests.test_dirs),
        "conventions": probe_results.conventions.detected_patterns,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
