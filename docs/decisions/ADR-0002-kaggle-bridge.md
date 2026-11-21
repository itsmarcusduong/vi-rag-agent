# ADR-0002: GPU experiments on Kaggle from GitHub Actions

Status: Accepted (recorded retrospectively on 2026-11-22; in force since #11, 2026-11-18)

## Context
The development sandbox cannot reach Hugging Face; the owner's laptop has a 4 GB GPU; Kaggle offers a
free weekly T4 quota. The solo playbook says automating Kaggle is worth it only for repeated jobs.

## Decision
GPU jobs are registered in `scripts/kaggle_job.py` and run by pushing `kaggle/<job>`; the `kaggle`
workflow pins the kernel to the commit, waits, downloads outputs and commits `results/*.json`.
CPU jobs use the `eval` workflow. Docs-only commits on a `kaggle/*` branch carry the skip marker.

## Consequences
- Justified: by M3.1 the bridge had run four jobs, and it made cross-environment reproduction checks
  cheap (M2 identical on Kaggle; M3 numbers identical in the M3.1 run).
- Results are committed by `github-actions[bot]` with the run ID; the kernel log is a workflow artifact.
- Depends on the Kaggle CLI (pinned to 2.2.5 plus `six`) and on string-matching its status output.
