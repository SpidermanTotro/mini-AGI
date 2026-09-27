# Greenlight Recur: Project History

> **Status:** living project history, maintained from repository evidence.
>
> Greenlight Recur is derived from Alexey Borsky's mini-AGI project. This
> document deliberately separates inherited mini-AGI history from work carried
> out as Greenlight Recur. It is not a claim that Greenlight originated the
> upstream architecture.

## Origins: mini-AGI (19–25 September 2026)

The repository history begins on **19 September 2026** with commits by Alexey
Borsky. During 19–25 September the upstream project added its initial
implementation, scaling and dashboard material, synthetic corpus generation,
the published main code, licensing, training updates, continual-learning
evidence, and further training-status updates.

These commits are part of Greenlight's ancestry, not Greenlight-authored work.
They remain in Git history so the project's provenance is preserved.

## Greenlight begins (26 September 2026)

Work that became Greenlight Recur accelerated on **26 September 2026**. Early
fork work added a 16 GB setup and rebuild tooling, GitHub Actions testing,
updated Actions dependencies, 16 GB training dependency instructions, a larger
16 GB configuration, coding/debugging-assistant work, checkpoint lifecycle
testing, and a reproducible benchmark harness.

A subsequent hardening pass addressed benchmark configuration, GradSNR
active-set handling, held-out evaluation bounds, paged-layout detection,
Hugging Face cache isolation, a dedicated self-knowledge corpus lane, serving
cache rollover, and protection against dry reads writing source expert files.
Upstream changes were also reconciled rather than silently discarded.

Two commits then made the fork's identity explicit:

- **1355453217** — `Document Greenlight fork goals and upstream relationship`
- **0de1603150** — `Establish Greenlight Recur project branding`

Those commits are the clearest repository boundary for the named **Greenlight
Recur** project.

## Integration and pre-training hardening

Greenlight's release-review and local-integration work was merged through PRs
#2 and #3. The local integration brought recovered Greenlight lab and
validation work onto main. PR #4 then performed debug cleanup before Greenlight
training, including bounded-round validation, detaching training loss before
scalar logging, and closing paged-storage test handles.

PR #5 cleaned checkpoint ignore rules so generated local training artifacts
would not accidentally become source-controlled project state.

## Regression coverage and extracted training policy

On 26 September, Greenlight moved from individual fixes toward smaller,
test-backed components:

- **PR #6** hardened local Python artifact ignores.
- **PR #7** added context-ramp regression coverage.
- **PR #8** extracted pure training-policy helpers and tested their edge cases.
- **PR #9** extracted and tested JSONL history truncation.
- **PR #10** extracted a shared JSONL append helper for telemetry.
- **PR #11** introduced an explicitly closable, line-buffered JSONL recorder
  and used it for stream history.

These changes did not yet make the training implementation independent. Much
of training still lived in the inherited `train.py`, but behavior was being
isolated behind tests so it could later be replaced safely.

## Independence transition (27 September 2026)

The next phase changed the goal from merely hardening inherited code to
progressively owning the Greenlight training implementation.

**PR #12 — `Start Greenlight-owned training entry point`** created the
`minagi.training` package, added a tested Greenlight command-dispatch
boundary, and routed the legacy CLI through it. The PR passed the repository's
Python 3.11 and 3.14 unit-test and source-compilation jobs and was merged as
**57b172d0d831fcaf9da95674881d68a943c89959**.

At that checkpoint, the inherited `train.py` was still approximately 124 KB
and 2,362 lines. The migration strategy became: move or replace one
responsibility, add regression coverage, switch callers, run CI, and then
delete the superseded legacy implementation.

The intended destination is a small training entry point backed by
Greenlight-owned modules for runner lifecycle, streaming, continual reading,
optimisation, scheduling, recovery, growth, checkpointing, and telemetry.

## Work in progress

At the time of this history update, **PR #13** is the next independence step.
It moves the `ponder-probe` parser definition out of legacy `train.py` and
into `minagi.training.cli`, with parser-level regression tests. It should be
described as work in progress until it is merged.

## What has not been established

This history records repository development; it does **not** establish
independent media coverage, academic validation, Wikipedia notability, or a
completed independent Greenlight training implementation. Historical upstream
dashboards and results should not be presented as Greenlight model results.

Likewise, Greenlight's technical independence does not erase its origin.
mini-AGI remains the project's documented foundation, and its authorship,
license, and Git history should continue to be preserved.

## Milestone timeline

| Date | Milestone |
| --- | --- |
| 19 Sep 2026 | Earliest mini-AGI commits in inherited repository history |
| 21 Sep 2026 | Upstream main code/licensing era visible in Git history |
| 26 Sep 2026 | Greenlight fork engineering, 16 GB tooling and CI work |
| 26 Sep 2026 | Greenlight Recur name and upstream relationship documented |
| 26 Sep 2026 | Release review/local integration and pre-training hardening |
| 26 Sep 2026 | PRs #6–#11 build regression coverage and extracted helpers |
| 27 Sep 2026 | PR #12 establishes Greenlight-owned training package/dispatch |
| 27 Sep 2026 | Training-core independence migration begins |

---

This document is intentionally conservative. New milestones should be added
only after they are represented by repository evidence; planned work belongs
under **Work in progress**, not in the completed history.
