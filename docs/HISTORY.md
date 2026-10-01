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

## Upstream integration and Training Doctor v2 (1 October 2026)

Upstream advanced four commits past the merge base (`5a99d7d`..`7361e7a`),
adding a text-wide expert-selection rule, a writing-degeneration fix and
`halt_freeze`. Those were brought into `integration/upstream-7361e7a` first, so
the project would not be improving a model while carrying bugs upstream had
already fixed. The integration is recorded here because what it surfaced
mattered more than the integration itself.

### Five defects, three classes

None of these were found by reading the diff or by a failing unit test. Every
one was found by running the code the project actually runs.

| # | Defect | Class | Origin |
| --- | --- | --- | --- |
| 1 | `train.py` lost `import math` while still calling `math.` four times; every run that reached evaluation died with `NameError` | pre-existing, untested path | Greenlight |
| 2 | `_load_optim` restored moments with no step counter, so the first optimiser step after a restart died with `KeyError: 'step'` | pre-existing, untested path | Greenlight |
| 3 | `build_paged` sized router rows from the manifest's expert count, so a pool that had never grown could not be reloaded by `read` at all | unit/reality mismatch | upstream |
| 4 | `--segment-chars` deleted; `read` raised `AttributeError` on its first line of real work | silent merge deletion | merge of upstream |
| 5 | `FolderEvaluator`'s signature taken from upstream, body kept from Greenlight | silent merge deletion | merge of upstream |

Defect 2 deserves its own line in the history: **no restart of a Greenlight
model had ever worked.** The project advertises continual learning, and the
save-and-continue path raised before training a batch.

Defects 4 and 5 are one shape. Upstream removes things Greenlight had added,
and git merges *deletions* cleanly, because removing an option is not a
conflict with a file that still reads it. `--dwell-chars` was the same, found
and repaired earlier in the same integration.

### Why 129 tests were green throughout

`cmd_read` and `cmd_stream` had no test that constructed them. Not weak
coverage — none. The suite tested components; the defects lived in the
commands, and two of them made `read`, described in the README as the command
the project runs, unable to start.

Two guards were added at the two levels that matter:

- **Always on.** Every argument `train.py` reads must be one it defines. That
  class of deletion is invisible to every other kind of test; this catches it
  in milliseconds.
- **Opt in, `GREENLIGHT_LIFECYCLE=1`.** `stream` -> train -> evaluate -> save ->
  resume -> keep learning as a real subprocess. Measured at 17.8 s on CPU with
  a toy corpus against a 4.8 s suite.

### Training Doctor v2

Doctor v1 read history rows tagged `event`; `train.py`'s recorder writes rows
tagged `kind`. It therefore found zero samples in every real history, found
nothing wrong, and reported HEALTHY — on a good run and on a run that had
crashed. Both spellings are read now, and a start record with no done record is
reported as a crash, which no loss series can show.

The Doctor gained a `--preflight` mode that replays a restart and asks whether
the optimiser can take a step, plus `--experts` for routing distribution and
`--samples` for generation quality. Results from the project's own artefacts:

- **Restart, on a real checkpoint.** 223 moments restored, none without a step
  counter, first step succeeded. With the repair deliberately removed: 192
  without a counter and `KeyError: 'step'`. The contrast is kept as a test,
  because a preflight that cannot fail is an opinion.
- **Routing, on the real 128-expert history.** Preference 0.006 where 0 is
  perfectly even and 1 is a single expert taking everything; gates spanning
  0.9988 to 1.0038. The router was not separating experts, and no loss figure
  said so.
- **Generation, on the real 432k-step samples log.** Held-out loss fell from
  1.1437 to 0.7167 (37%) while repeated 8-grams rose from 0.0911 to 0.1158
  (27%), with a character repeated 130 times in one sample and runs of six
  consonants surviving the repetition guard. The two numbers moved in opposite
  directions across 431k steps, and the loss alone reported nothing.

### Commits

| Commit | Contents |
| --- | --- |
| `e4d7562` | Upstream integration, four conflicts resolved by hand, defects 1-3 repaired |
| `024be19` | Training Doctor v1 imported from PR #19, with its schema defect recorded |
| `859c578` | Doctor v2: real histories, dead-run detection, restart preflight |
| `5ffe46d` | Doctor v2: routing distribution and per-domain retention |
| `b09b03e` | Defects 4-5 repaired, CLI contract test and opt-in lifecycle test |
| `5dd1e93` | Doctor v2: generation quality measured against falling loss |

### Milestone timeline addition

| Date | Milestone |
| --- | --- |
| 1 Oct 2026 | Upstream `7361e7a` integrated; five defects found and repaired |
| 1 Oct 2026 | Training Doctor v2: restart preflight, routing, retention, generation |
| 1 Oct 2026 | Lifecycle coverage gap closed: contract test plus opt-in subprocess test |

---

This document is intentionally conservative. New milestones should be added
only after they are represented by repository evidence; planned work belongs
under **Work in progress**, not in the completed history.
