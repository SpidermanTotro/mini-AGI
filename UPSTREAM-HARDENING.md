# Upstream hardening status

This branch audits open upstream mini-AGI defects against Greenlight and keeps
each fix behind regression coverage before it is considered complete.

| Upstream issue | Greenlight status | Fix / protection |
| --- | --- | --- |
| #19 changing gradient set crashes GradSNR | Fixed | Reset telemetry EMA when the active parameter signature changes; regression tests cover different-size and equal-size/different-parameter sets. |
| #18 held-out evaluation can exceed RoPE context | Fixed | FileReader caps each measured chunk to the model context; oversized-chunk regression test. |
| #17 serving/paged checkpoint startup failures | Fixed | SharedPool-safe startup reporting and paged-layout inference from experts/; regression coverage for pre-manifest paged layout. |
| #12 corpus fetch deletes unrelated HF caches | Fixed | Default fetch owns a private temporary cache and deletes only that cache; tests cover default and --keep-cache behavior. |
| #10 self-knowledge lane never built | Already fixed in Greenlight | Dedicated data_self_chat_char source expands into data/train/self-knowledge; regression test protects separation from generic chat. |
| #8 serving reuses RoPE positions after KV trimming | Fixed | Serving releases the full cache before chunked half-window prefill; tests cover repeated rollover, rotary positions, retained tokens, and cached/full-forward equivalence. |
| #6 dry reads can write expert updates into source checkpoint | Fixed | Dry training uses a temporary writable overlay; trained expert weights and Adam moments survive eviction while source checkpoint contents and metadata stay unchanged. Fresh dry models exist only in temporary storage. |

## Validation policy

A production change is not marked complete merely because it imports or looks
correct. GitHub Actions runs the repository unittest suite on every pushed
commit. Keep main untouched until this hardening branch is reviewed and merged
deliberately.

## Branch

`greenlight-upstream-hardening`

## 2026-09-26 release review

- Reconciled upstream through `5a99d7d`, including training reports, plotting,
  expert auditions, and resume-history handling. Kept Greenlight branding,
  credits, hardening, and the existing 8,192 default context ceiling; the
  separate 16 GB profile remains at 4,096.
- Removed the duplicate audition admission loop that incorrectly refreshed
  pruning clocks. Added regression coverage for audition versus earned admission.
- Remapped audition state and last-try timestamps after pruning, and invalidated
  cached candidate scores after growth/pruning to prevent stale expert indices.
- Adapted upstream PRs #7 and #9 (GroveMinting), retaining original authorship
  and commit references, for dry-read overlays and bounded serving rollover.
- Added Flask to CI's test dependencies and included entry points and plotting
  tools in compilation checks.
- Validation: 45 CPU tests passed locally with Python 3.12 and PyTorch
  2.14.0+cpu; compilation and `git diff --check` passed. GPU performance and
  long-running training were not evaluated in this review.

The release-review branch is `greenlight-release-review`; promotion to `main`
uses a reviewed pull request and successful CI.

## 2026-09-27 local-work integration

Recovered Henric's code from bundle commit `00f0b1e81ea6b83e3d0d985207ac5a7c480311f3`
and combined it with the reviewed `341ed6c` main branch. Model weight files
were not included in the bundle or this integration.

- Preserved the lab runner, Greenlight config alias and R1 naming, checkpoint
  completeness checks, chat API validation, corpus failure propagation,
  context ramp endpoint, growth reporting cadence, and local search containment.
- Kept the previous paging, dry-read, serving rollover and upstream fixes.
- Resolved explicit benchmark config precedence with GREENLIGHT_CONFIG.
- Routed lab history and samples into its output directory and disabled shared
  plotting, so lab runs do not rewrite the main run's history or plots.
- Preserved README.md.clean as supplied; README.md is the maintained entry point.
- Local validation: 61 CPU tests passed, plus compilation and diff whitespace
  checks (excluding whitespace in the preserved README.md.clean). CI now checks Python 3.11 and 3.14. GPU training was not run here.
