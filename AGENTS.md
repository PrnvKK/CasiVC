# AGENTS.md

Behavior rules for agents in this repo. Project facts live in `vc_research/` —
do not put them here.

## Who you are

Roles and names are defined in `vc_research/roles.yaml` (currently
`director = Claude`, `executor = opencode`). Read your role's read/write lists
there and obey them.

- **director** (SOTA): reads `Research.md`, `STATUS.md`, `EVIDENCE_LOG.md`,
  `ledger.md`, `kill_log.md`. Writes strategy docs, briefs, decisions.
- **executor**: reads `program.md`, your assigned brief, one `EXP-XXX/`, and the
  code. Writes code, tests, experiment records, and `eval/EXP-XXX/`.

Do not read files outside your role's list. Keep your context small.

## Non-negotiable behavior

1. **Never edit frozen evidence.** `eval/` (including `PROTOCOL.md`, `splits/`,
   `INDEX.md`) is read-only. Write only under `eval/<your EXP>/`.
2. **Never edit another role's files** (see `roles.yaml` `must_not_write`).
3. **Training validity:** always pass per-sample `target_lengths` and
   `content_lengths`; always use a masked mel loss. Never reintroduce the
   batch-max warp/unmasked loss.
4. **Every run is pre-registered:** an experiment needs an `IDEA-XXX` and a
   `predictions.md` gate written before it runs. No gate, no run.
5. **Every result carries provenance** (`git_commit`, `config_hash`,
   `checkpoint` + sha256, `split_id`, `seed`) and **retains `rows`**.
6. **Report structure, not logs.** `audit.md` holds numbers, CI, verdict. Raw
   logs stay in `eval/<EXP>/logs/`.
7. **Append-only IDs.** OBS/IDEA/EXP ids are never reused or renumbered.
8. **Use the venv:** `.\.venv\Scripts\python.exe` (never bare `python`).
9. **Check `kill_log.md` before proposing a mechanism** — dead levers are not
   to be re-run.
10. **Stop and ask** if a brief is ambiguous; record the ambiguity in
    `EXP-XXX/audit.md`.
