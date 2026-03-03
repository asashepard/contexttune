"""Continuous oracle tuning loop for AGENTS.md.

Flow:
1. Checkout repo, run deterministic probes, build RepoKB.
2. Render initial AGENTS.md from KB -> RepoGuidance v0.
3. For each iteration (fixed count):
   a. Generate new probes via LLM (no fixed categories).
   b. Evaluate probes for diagnostics and edit opportunities.
   c. Aggregate/propose edits.
   d. Apply edits to AGENTS.md.
   e. Save new guidance version.
4. Save latest guidance as best_guidance.json.
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
from context_policy.oracle.schema import Edit, OracleConfig, OracleState, Probe, ProbeResult
from context_policy.probes import run_all_probes


def run_oracle_loop(config: OracleConfig) -> tuple[RepoKB, RepoGuidance]:
    """Run the continuous LLM-driven oracle loop for one repository."""
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

    print(f"[oracle] Checking out {config.repo} @ {config.commit}...")
    repo_dir = checkout_repo(config.repo, config.commit)

    print(f"[oracle] Running probes on {config.repo}...")
    probe_results = run_all_probes(repo_dir)

    _save_probe_results_summary(probe_results, kb_dir / "probes_summary.json")

    kb = build_kb(config.repo, config.commit, probe_results)
    kb.save(kb_dir / "kb.json")
    print(
        f"[oracle] KB built: arch={len(kb.architecture)} chars, "
        f"map={len(kb.symbol_map)} chars, "
        f"ctx={len(kb.context)} chars, "
        f"conv={len(kb.conventions)} chars"
    )

    agents_md = render_agents_md(kb)
    v0_guidance = RepoGuidance(
        repo=config.repo,
        commit=config.commit,
        lines=agents_md.splitlines(),
        version=0,
    )
    v0_guidance.save(guidance_dir / "v0.json")
    (kb_dir / "agents_md_v0.md").write_text(agents_md, encoding="utf-8")
    print(f"[oracle] Static AGENTS.md: {len(agents_md)} chars, {len(v0_guidance.lines)} lines")

    if config.iterations <= 0:
        final_path = out / "best_guidance.json"
        v0_guidance.save(final_path)
        return kb, v0_guidance

    if state_path.exists():
        state = OracleState.load(state_path)
        if state.completed_iterations > 0:
            current_path = guidance_dir / f"v{state.current_version}.json"
            if current_path.exists():
                current = RepoGuidance.load(current_path)
                agents_md = current.render()
                print(
                    f"[oracle] Resuming from v{state.current_version} "
                    f"(completed_iterations={state.completed_iterations})"
                )
            else:
                state = OracleState(repo=config.repo)
        else:
            state = OracleState(repo=config.repo)
    else:
        state = OracleState(repo=config.repo)

    current = RepoGuidance(
        repo=config.repo,
        commit=config.commit,
        lines=agents_md.splitlines(),
        version=state.current_version,
    )

    probes: list[Probe] = []
    start_iter = state.completed_iterations + 1
    for t in range(start_iter, config.iterations + 1):
        print(f"\n[oracle] {config.repo} iteration {t}/{config.iterations}")

        new_probes = generate_probes(
            kb,
            config.model,
            agents_md,
            prior_probes=probes,
            timeout_s=config.timeout_s,
        )
        probes.extend(new_probes)
        print(
            f"  Generated {len(new_probes)} new probes; "
            f"total probe pool={len(probes)}"
        )

        results = _evaluate_all_probes_detailed(agents_md, probes, config)

        llm_edits = diagnose_failures(agents_md, results, config.model, timeout_s=config.timeout_s)
        direct_edits = _collect_edits_from_results(results)
        edits = _dedupe_edits([*direct_edits, *llm_edits])

        print(
            f"  Proposed edits: direct={len(direct_edits)}, "
            f"diagnose={len(llm_edits)}, merged={len(edits)}"
        )

        if edits:
            agents_md = apply_edits(agents_md, edits, config.model, timeout_s=config.timeout_s)
        else:
            print("  No valid edits returned; preserving AGENTS.md for this iteration")

        new_version = current.version + 1
        current = RepoGuidance(
            repo=config.repo,
            commit=config.commit,
            lines=agents_md.splitlines(),
            version=new_version,
        )
        current.save(guidance_dir / f"v{new_version}.json")

        state.current_version = new_version
        state.completed_iterations = t
        state.history.append(
            {
                "version": new_version,
                "type": "oracle_iteration",
                "iteration": t,
                "probe_pool_size": len(probes),
                "new_probes": len(new_probes),
                "edits_count": len(edits),
                "edits": [
                    {"section": e.section, "action": e.action, "content": e.content}
                    for e in edits
                ],
            }
        )
        state.save(state_path)

        print(f"  Saved v{new_version} ({len(agents_md)} chars)")

    final_path = out / "best_guidance.json"
    current.save(final_path)
    print(f"\n[oracle] Done. Final for {config.repo}: v{current.version}")
    print(f"  Saved to {final_path}")

    config_path = out / "oracle_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8",
    )

    return kb, current


def _evaluate_all_probes_detailed(
    agents_md: str,
    probes: list[Probe],
    config: OracleConfig,
) -> list[ProbeResult]:
    """Evaluate all probes and return detailed results."""
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        print(f"    Evaluating probe {i+1}/{len(probes)}: {probe.id}...")
        try:
            result = evaluate_probe(
                agents_md, probe, config.model, timeout_s=config.timeout_s,
            )
            results.append(result)
            missing = sum(1 for r in result.behavior_reviews if r.assessment == "missing")
            partial = sum(1 for r in result.behavior_reviews if r.assessment == "partial")
            print(
                f"      → reviews={len(result.behavior_reviews)} "
                f"(missing={missing}, partial={partial}), "
                f"edits={len(result.proposed_edits)}"
            )
        except Exception as exc:
            print(f"      → ERROR: {exc}")
            results.append(
                ProbeResult(
                    probe_id=probe.id,
                    task=probe.task,
                    response="",
                    behavior_reviews=[],
                    proposed_edits=[],
                    overall_notes=f"evaluation_error: {exc}",
                )
            )
    return results


def _collect_edits_from_results(results: list[ProbeResult]) -> list[Edit]:
    edits: list[Edit] = []
    for result in results:
        edits.extend(result.proposed_edits)
    return edits


def _dedupe_edits(edits: list[Edit]) -> list[Edit]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Edit] = []
    for edit in edits:
        key = (edit.section.strip(), edit.action.strip().lower(), edit.content.strip())
        if not key[2]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Edit(
                section=key[0] or "General",
                action=key[1] or "add",
                content=key[2],
            )
        )
    return out


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
