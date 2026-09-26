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
| #8 serving reuses RoPE positions after KV trimming | Fixed | Serving rebuilds a recent half-window into fresh caches before rotary-position exhaustion, matching RecurCoder.generate behavior. |
| #6 dry reads can write expert updates into source checkpoint | Fixed | train.py opens paged storage read-only unless --save is requested; storage-level regression proves eviction/flush cannot write source experts. |

## Validation policy

A production change is not marked complete merely because it imports or looks
correct. GitHub Actions runs the repository unittest suite on every pushed
commit. Keep main untouched until this hardening branch is reviewed and merged
deliberately.

## Branch

`greenlight-upstream-hardening`
