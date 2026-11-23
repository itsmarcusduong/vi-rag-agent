# Running experiments on Kaggle from GitHub Actions

Heavy jobs (GPU encoders, fine-tuning) run on Kaggle; nobody creates notebooks by hand.

```text
push branch kaggle/<job> ──► workflow `kaggle` ──► builds a kernel pinned to the commit SHA
                                                 ──► kaggle kernels push (T4 GPU if the job needs it)
                                                 ──► polls status, downloads output
                                                 ──► commits results/<file>.json to the branch
```

## One-time setup (repository owner)
1. Kaggle account must be phone-verified (needed for internet access inside kernels).
2. Kaggle -> Settings -> API -> **Create New Token**. Copy the token. Do not paste it in chats,
   issues or code.
3. GitHub repo -> Settings -> Secrets and variables -> Actions -> **New repository secret**:
   name `KAGGLE_API_TOKEN`, value = the token.
4. Optional: repository variable `KAGGLE_USERNAME` if it is not `minhquangdng`.

## Adding a job
Register it in `JOBS` in `scripts/kaggle_job.py` (script, extras, output files, `gpu`). The script
must write its results under `results/` and record the commit and environment.

## Adding docs to a kaggle branch without re-running
Put the marker `[skip kaggle]` in the commit message of docs-only commits. The workflow checks the
message of the pushed head commit only, so never quote the marker in a commit that should run.

## Kernel stuck in QUEUED, or finished after the workflow gave up
The workflow fails after 30 polls in `QUEUED` (no GPU quota or capacity). The kernel may still run later
on Kaggle. To collect its output without starting a new version, push a commit whose message contains
`[kaggle fetch]` to the same `kaggle/<job>` branch.

## What runs where
- Kernels are private. They clone this public repository at the exact commit, install the package,
  run the script, and copy only the declared result files to the Kaggle output.
- The kernel log is attached to the workflow run as an artifact (`kaggle-<job>`).
- GPU quota is the account's weekly free Kaggle quota; CPU jobs do not use it.

## Verification (2026-11-18)
The M2 BM25 job was re-run on Kaggle through this bridge (workflow run 44174860471, code `dae128c`,
Python 3.13 on Kaggle vs 3.12 on GitHub Actions). Every metric, confidence interval and gold-rank
bucket in [`results/repro/m2_bm25_kaggle.json`](../results/repro/m2_bm25_kaggle.json) is identical to
[`results/m2_bm25.json`](../results/m2_bm25.json); only timings and environment differ. This checks
both the bridge and the reproducibility of M2.
