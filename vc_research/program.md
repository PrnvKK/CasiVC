# program.md — Execution Contract

> **Owner: @director (Claude). Executed by @executor (opencode).**
> This is the *only* project file the executor needs beyond its assigned brief,
> one experiment folder, and the code. Do not read `Research.md`, `ledger.md`,
> or history; `STATUS.md` is the one-line summary if you need orientation.

---

## 0. Your role

You are the **executor**: implement the change described in your assigned
`claude_briefs/BRIEF-XXX.md`, run the experiment, and report **structured
evidence**. You do not decide research direction and you do not edit the
strategy docs. Keep your context small: read only what the brief, one
`EXP-XXX/` folder, and the `src/vc/**` interfaces require.

If the brief is ambiguous, write the ambiguity into `EXP-XXX/audit.md` and stop;
do not invent scope.

## 1. Hard invariants (violating these invalidates everything)

1. **Training validity.** Any training path MUST pass per-sample
   `target_lengths` and `content_lengths` to the model and use a **masked** mel
   loss. The batch-max resampler length + unmasked loss is a known, catastrophic
   bug. Never reintroduce it. `tests/` guards this.
2. **Parameter budget.** Trainable parameters must stay **< 500,000**. The model
   asserts this at construction.
3. **Evaluation protocol.** Benchmark = ≥16 held-out targets, n≥240 conversions,
   **paired bootstrap by target speaker**. Always report `sim_src`, `sim_tgt`,
   `content_L1` with the margin.
4. **Frozen evidence.** `eval/`, `eval/PROTOCOL.md`, `eval/splits/`, and
   `eval/INDEX.md` are controller-owned and read-only. Write your results only
   under `eval/EXP-XXX/`.
5. **Never edit** `Research.md`, `kill_log.md`, `claude_briefs/`,
   `claude_decisions/`. You may append a status to `ledger.md` only.
6. **Provenance.** Every result JSON must carry the manifest header
   (`git_commit`, `config_hash`, `checkpoint`, `ckpt_sha256`, `split_id`,
   `protocol`, `seed`, `created_utc`) and MUST retain the `rows` field.

## 2. Experiment lifecycle (mandatory order)

1. **IDEA** — the brief references an `IDEA-XXX` (or `OBS-XXX`). If one does not
   exist, create `changes/ideas/IDEA-XXX-<slug>.md` from the template and link
   it.
2. **EXP** — create `changes/experiments/EXP-XXX-<slug>/` with:
   - `config.md` — frozen knobs + the exact command + `config_hash`.
   - `predictions.md` — the **falsifiable numeric gate**, written BEFORE any run
     (e.g. "paired margin Δ ≥ +0.01, CI excludes 0, `sim_src` not increased").
   - `brief.md` — link to the brief that spawned it.
   - `audit.md` — filled AFTER the run (see §4).
3. **Run** — via `tools/` only; use the venv explicitly.
4. **Report** — write `audit.md`, copy results to `eval/EXP-XXX/`, link the
   result back (pointer/symlink), update the `ledger.md` status row.
5. **Verdict** — pass → note it; fail → record, and the director will add it to
   `kill_log.md` (you may propose the entry in `audit.md`).

No hypothesis/gate → no run. This gate is enforced by `tools/run_exp.py`.

## 3. Running things (local CPU)

Always use the venv:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py tools\content_factorial.py --stage eval `
    --content vq --init checkpoints/cf_km100_idP2.ckpt `
    --n_content 2 --n_ref 2 --results eval/EXP-XXX/results.json
```

- Long runs must show a tqdm ETA and use `--save_every` for resumability.
- Prefer foreground runs; `Start-Process` background runs can die on sleep.
- Colab path: `--device cuda`; the upload bundle is code-only.

## 4. Reporting format (`audit.md`)

Report numbers, not narrative. Use this skeleton:

```
# EXP-XXX audit
status: pass | fail | inconclusive
brief: BRIEF-XXX   idea: IDEA-XXX
command: <exact command>
artifacts: eval/EXP-XXX/  (pointer)
keys:
  margin: +0.0xxx  CI [...]
  sim_src / sim_tgt / content_l1: ...
  paired Δ vs <baseline>: +0.0xxx [CI], wins k/N
prediction: <restate the gate>
verdict: <agent's assessment vs gate>
notes: <only decision-relevant; never paste raw logs>
```

Raw stdout/tqdm logs go to `eval/EXP-XXX/logs/`; never into `ledger.md`.

## 5. Budget & escalation

- Default executor budget: 30k tokens / 60 min per task (`roles.yaml`).
- Any experiment estimated over the escalation threshold, or any **scale-up**,
  needs a `claude_decisions/DECISION-XXX.md` from the director first.
- Stop early if the gate is already clearly failed; report the failure.

## 6. Do-not-repeat

Before proposing any mechanism, check `kill_log.md` is *not* already listing it.
Anything in the kill log is falsified with evidence; do not re-run it.

## 7. Definition of done

A task is done when: code is committed, `tests/` pass, `EXP-XXX/audit.md` is
written, results live in `eval/EXP-XXX/`, the result is linked back, and the
`ledger.md` status row is updated. Anything less is partial — say so explicitly.
