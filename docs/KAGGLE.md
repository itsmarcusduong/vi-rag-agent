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

## What runs where
- Kernels are private. They clone this public repository at the exact commit, install the package,
  run the script, and copy only the declared result files to the Kaggle output.
- The kernel log is attached to the workflow run as an artifact (`kaggle-<job>`).
- GPU quota is the account's weekly free Kaggle quota; CPU jobs do not use it.
