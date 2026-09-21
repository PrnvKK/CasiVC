# CURRENT_STATE.md — read me first

> The **minimum entry point** for a fresh agent session. If you have 2 minutes,
> read this. Everything else is linked from here by group.

## Where we are

- **Best arm:** `km100_idP2` — margin **+0.1173** [+0.0925, +0.1395], 40 held-out
  targets, n=624. Content = K-means(256) quantized HuBERT L9; identity = Phase-2
  full-length HiFi-GAN→ECAPA supervision.
- **Established:** content K-means suppresses source identity (dominant lever);
  true vocoded-ECAPA identity supervision works; loss-shaping is audibly
  saturated; remaining gap is structural (conditioning capacity + decoder
  rendering), not a missing signal.
- **Next action (open):** a conditioning-capacity mechanism that lowers `sim_src`
  locally before any scale-up, OR the decoder scale-up gated by a listening test.
  See `Research.md` §4 and `ledger.md`.

## Reading order (hot → cold)

| Reader | Read, in order |
|---|---|
| @director (Claude) | `Research.md` → `STATUS.md` → `EVIDENCE_LOG.md` → `ledger.md` → `kill_log.md` → `claude_briefs/` |
| @executor (opencode) | `program.md` → your `claude_briefs/BRIEF-XXX.md` → your `changes/experiments/EXP-XXX/` → `src/vc/**` |

## Groups

- **Strategy** (director + human): `Research.md`, `EVIDENCE_LOG.md`
- **Contract** (executor): `program.md`, `roles.yaml`
- **Records**: `ledger.md` (all hypotheses), `kill_log.md` (dead levers),
  `changes/` (OBS / IDEA / EXP)
- **Evidence** (frozen): `eval/` — never edited
- **Code**: `src/vc/` (library), `tools/` (CLIs), `probes/`, `baselines/`, `tests/`
