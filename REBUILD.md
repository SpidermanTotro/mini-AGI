# Teardown and Rebuild Guide

This guide reconstructs the **source environment** from a clean checkout or
the source ZIP. Training a fresh model is a separate, long-running step. The
ZIP intentionally does not contain trained weights, downloaded corpora, or
run outputs.

## Teardown: What Is Here

The execution path is:

1. `minagi/tokenizer.py` maps bytes and structural markers to the model's 265
   token IDs.
2. `minagi/model.py` defines the dense transformer pieces: RMSNorm, RoPE,
   causal attention, SwiGLU, and residual blocks.
3. `minagi/recur.py` assembles embeddings, prelude blocks, recurrent passes,
   the adapter, halting head, final norm, and output head. It owns the public
   forward and generation APIs.
4. `minagi/pool.py` replaces recurrent feed-forward layers with a shared,
   top-k expert pool. `minagi/paged.py` manages its resident set and optimizer
   state across device, RAM, and disk.
5. `minagi/store.py` saves a manifest and model data. For paged models, expert
   weights and optimizer moments live in separate files under the weights
   directory; the manifest is the index.
6. `minagi/stream.py` reads text in chunks and carries the KV cache between
   them. `minagi/precision.py`, `minagi/optim.py`, and
   `minagi/plasticity.py` supply compute precision, optimizer helpers, and
   learning-rate control.
7. `train.py` is the training CLI and orchestration layer. `serve.py` is the
   Flask chat server. Serving can update weights unless run in read-only mode.
8. `corpora/` builds training and held-out text. `replication/` contains
   analysis scripts and recorded experiment results.
9. `mini_agent/` is a separate local assistant. It calls Ollama over localhost,
   keeps conversation history in SQLite, and exposes bounded workspace tools.
   It is not the model trained by `train.py`.
10. `config-16gb.yaml` and `train-16gb.sh` provide an approximately 924M
   parameter starting profile and monitored training helper for a 16-GB NVIDIA
   GPU / 32-GB RAM host. It uses a wider architecture than `config.yaml` and
   requires its own fresh weights directory. The helper records a prompt-
   evaluation report after each successful run.
11. `minagi/evaluate.py` records repeatable generation probes from saved
   checkpoints so training changes can be compared on the same prompts.

`config.yaml` is the main settings file. `README.md` gives the short project
overview and original run instructions. The model is trained from scratch;
there is no pretrained checkpoint in this repository.

## Clean Setup

The documented reference environment is Python 3.10 or newer and an
8-GB-or-larger CUDA GPU. The README's reference machine used PyTorch
2.6.0+cu124. Install the PyTorch build that matches your operating system and
CUDA driver using the selector at <https://pytorch.org/get-started/locally/>.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

The generic `torch` install above is only a placeholder: choose the correct
PyTorch command for the machine before training on CUDA. Install optional
dependencies only for the features you need:

```bash
python -m pip install -r requirements-optional.txt
```

The workspace assistant itself uses only Python's standard library. Its local
inference server is a separate install: install Ollama for your operating
system, start its server, and pull a tool-capable instruct model such as
`qwen3:8b`.

The core checks need no corpus or trained weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q minagi corpora mini_agent tests
```

## Rebuild From Zero

1. **Build a small corpus first.** This downloads data, so it needs network
   access and enough disk space. A limited run is the quick setup check:

   ```bash
   python -m corpora all --limit 5000
   ```

   Full data builds can be several gigabytes or more. Review each upstream
   dataset's terms before using or redistributing it. You can instead point
   `config.yaml` at your own text files.

2. **Start training.** This creates the weights directory if it does not
   exist. Start with the limited corpus and a short run before committing to a
   long training job:

   ```bash
   python train.py read data/train --save --weights-dir weights \
       --held-out data/val --sample-every 10
   ```

   Use `python train.py --help` for the full set of training controls. Check
   the device, context, chunk size, and available disk before a long run.

      For the larger 16-GB VRAM / 32-GB RAM profile, use its separate weights
      directory. Its 768-wide trunk and 3072-wide experts are not compatible
      with default or older 16-GB-profile checkpoints:

      ```bash
      MINI_AGI_CONFIG=config-16gb.yaml python train.py read data/train \
         --weights-dir agi-16-large --held-out data/val --save --minutes 1
      ```

      The profile starts near 924M parameters, with 64 resident experts, a
      128-expert RAM cache, and a 4096-character context. `SMOKE_MINUTES=0
      PASSES=2 ./train-16gb.sh` removes the time limit and runs two full passes.
      Watch GPU/RAM/disk usage; tune resident experts in increments of 8. The
      memory numbers are estimates, not hardware validation.

3. **Serve the trained model.** The server loads from `weights` by default.

   ```bash
   python serve.py --port 8080
   ```

   Open `http://127.0.0.1:8080`. Use the server's read-only option if a chat
   session must not learn into the saved model.

4. **Keep experiment outputs out of the source archive.** The default data,
   weights, and run directories are local artifacts, not source files. Back
   them up separately if you need to resume a trained run.

## Run the Local Assistant

With Ollama installed and running:

```bash
ollama pull qwen3:8b
python -m mini_agent --workspace . --model qwen3:8b
```

`qwen3:8b` is an example tool-calling model, not a state-of-the-art guarantee.
Select a stronger coding/instruct model that fits the host with `--model` or
`OLLAMA_MODEL`. For coding, the assistant can inspect files, request approval
for edits, check Python syntax, and run all or a selected `test_*.py` file after
approval.
For image generation, start a local AUTOMATIC1111/Forge server with its API
enabled, then set `MINI_AGENT_IMAGE_API=http://127.0.0.1:7860` or pass
`--image-api-url http://127.0.0.1:7860`. The assistant saves an approved PNG
inside the workspace through that separate diffusion model; the byte-level
model trained by this repository is text-only.

Use `/exit` to leave the CLI and `/reset` to clear the active conversation.
History stays in `~/.local/share/mini-agi/agent.sqlite3` unless
`--database` or `MINI_AGENT_DB` selects another path. Workspace reads, search,
and Git status/diff are confined to the chosen root. The assistant can
statically check Python syntax; running tests requires interactive approval
because tests execute repository code. Each file write also requires approval.
There is no general shell-execution tool. This is a bounded assistant, not AGI.

## Compare Your Checkpoints

After the first saved checkpoint, run the same prompt set again after further
training. This saves the model step and exact generated continuations:

```bash
python -m minagi.evaluate --weights agi-16-large \
   --output runs/agi-16-large-baseline.json
```

Use a new output filename for later snapshots, then compare the responses.
The open-ended items are qualitative; four arithmetic prompts receive exact
match scores. This is still only a smoke probe, not a standardized benchmark
or evidence of general intelligence. Use separate held-out scored tasks for
capabilities you want to improve reliably.

## Make and Restore the ZIP

Install the `zip` utility if it is missing. Run this from the repository root
to create a source-only archive:

```bash
zip -r -FS mini-AGI-rebuild.zip \
   README.md REBUILD.md LICENSE requirements.txt requirements-optional.txt \
   .gitignore .github config.yaml config-16gb.yaml train-16gb.sh train.py serve.py mini_agent minagi \
   corpora replication assets tests \
  -x '*/__pycache__/*' '*.pyc'
```

The archive includes the source, config, docs, assets, tests, and replication
results. It excludes `.git`, `data/`, `weights/`, and `runs/`; those may be
large, machine-specific, or derived. The command uses an explicit allowlist,
so adding a new top-level source directory means adding it to the command.

To restore it into a separate directory:

```bash
mkdir mini-AGI-restored
unzip mini-AGI-rebuild.zip -d mini-AGI-restored
cd mini-AGI-restored
```

Then repeat **Clean Setup** and **Rebuild From Zero**. The ZIP recreates the
code checkout, not the downloaded data or trained model.

## Remaining Gaps

- `requirements.txt` gives compatible version ranges, not a lockfile. PyTorch
  must be selected for the target CUDA/runtime; corpus builders also depend on
  remote dataset revisions that are not pinned here.
- CPU tests now cover paged save/resume, expert and trunk optimizer restoration,
  and a small expert growth/pruning lifecycle. They do not yet cover context
  ramping, growth-brake decisions over a run, corpus downloads, or HTTP serving
  end to end.
- `.github/workflows/tests.yml` runs the unit suite and syntax compilation on
   pushes and pull requests, using CPU-only PyTorch. There is still no automated
   CUDA/memory stress job or declared package/build metadata; the project is
   run from its repository root.
- `train.py` is a large orchestration module. Splitting its CLI, training
  loops, and evaluation into independently tested components would make
  future changes easier to isolate.
- Loading legacy `.pt` checkpoints uses PyTorch deserialization with
  `weights_only=False`; only load checkpoint files you trust.
- The language model is text-only. Image generation is delegated to an optional
   local diffusion server; no vision, audio, or image decoder is trained here.
- The prompt evaluator is a smoke probe, not a broad standardized coding,
   reasoning, or agent benchmark and does not establish AGI-level capability.
- Local and CI validation use CPU-only PyTorch. They do not establish that the
   expanded 16-GB profile fits a particular GPU or that long-run training is
   stable.