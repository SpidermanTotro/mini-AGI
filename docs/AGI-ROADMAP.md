# Greenlight General-Intelligence Research Roadmap

Greenlight Recur does **not** currently claim AGI. This roadmap defines a sequence of falsifiable research gates for testing whether the same continually-learning system can become progressively more general.

The purpose is to replace "looks intelligent" with measurements that can fail.

## Ground rules

1. Reliability comes before capability claims.
2. Evaluation data must be held out from training.
3. The same model must face all capability gates; separate hand-built models do not count as generality.
4. Improvements must be compared against frozen baselines under matched conditions.
5. A failed gate is a research result, not something to hide.
6. Tool access, retrieval, or a large context window do not by themselves demonstrate intelligence.
7. Greenlight remains experimental even if every internal gate passes; strong external evaluation would still be required before an AGI claim.

## R5 — Reliable continual-training lifecycle

**Question:** Can learning state survive real interruption?

Required evidence:

- train and create non-trivial AdamW state
- save model, optimiser, expert state and training metadata
- terminate the original process
- reconstruct the model and optimiser from disk
- resume without silent optimiser reinitialisation
- verify `step`, `exp_avg`, `exp_avg_sq`, expert/router state and global progress
- continue training and show finite, plausible loss
- evaluate and generate after restart
- repeat for paged experts

**Gate:** no known cold-restart corruption across the supported training paths.

## R6 — Generalization

**Question:** Can the model solve material it was not trained to reproduce?

Build contamination-controlled held-out suites for:

- arithmetic
- code
- language
- factual reasoning
- compositional tasks
- unfamiliar synthetic tasks

Measure accuracy/loss before learning, immediately after learning, and after unrelated subsequent training.

**Gate:** statistically credible improvement on unseen examples without unacceptable regression on established domains.

## R7 — Continual learning and transfer

**Question:** Can Greenlight acquire multiple new skills sequentially?

Teach a sequence of previously unseen task families. After each learning phase, retest every earlier family.

Measure:

- forward transfer
- backward transfer
- retention
- catastrophic forgetting
- expert utilization
- router entropy/starvation
- compute and storage growth

**Gate:** repeated acquisition produces useful retained capability rather than a cycle of learn-and-forget.

## R8 — Reasoning, planning and correction

**Question:** Can the same model carry out multi-step work and recover from mistakes?

Use hidden tasks requiring decomposition, intermediate state, verification and revision. Score final correctness separately from plausible explanation.

Include adversarial cases where the first apparent strategy is wrong.

**Gate:** performance improves materially when the model is allowed to plan/check, and corrections reliably increase final-task success.

## R9 — Memory and tool use

**Question:** Can Greenlight use external capabilities without confusing access with knowledge?

Test:

- working memory
- durable learned knowledge
- explicit retrieval
- tool selection
- tool-result verification
- recovery from failed tool calls
- resistance to irrelevant or contradictory retrieved information

**Gate:** tools and memory increase task success while the model can still identify uncertainty and bad evidence.

## R10 — General-agent evaluation

**Question:** Does one system transfer competence across broad, unfamiliar tasks?

Use a locked evaluation harness containing domains and task formats withheld from development. Include long-horizon tasks where success requires learning, planning, memory, tools and self-correction together.

Compare against:

- the previous Greenlight release
- ablations with learning/memory/tools disabled
- suitable public baselines where reproducible

**Gate:** broad performance must come from the same system and survive independent reproduction.

## AGI evidence threshold

R10 passing would **not automatically prove AGI**. It would justify stronger external evaluation.

An eventual AGI claim would need evidence that the system displays broad, transferable competence across genuinely novel tasks, learns new capabilities efficiently, retains them, plans and corrects errors, and does so without benchmark-specific engineering.

Independent researchers should be able to reproduce or challenge the evidence.

## Immediate priority

Finish R5.

A system intended to learn continually must first demonstrate that its learned state survives a shutdown and cold restart. Capability experiments should not be allowed to obscure that reliability requirement.
