## Summary
<!-- What changes and why, in 2-4 sentences. -->

Closes #

## Changes
-

## Evidence / Checks
<!-- Commands run and their real output; results files with commit and run ID. -->
- [ ] `ruff check .` and `ruff format --check .`
- [ ] `pytest`
- [ ] Numbers come from `results/*.json` of an actual run (link it)
- [ ] For experiments: protocol in `docs/protocols/` was merged before the run; reproduction gate passed

## Self-review and adversarial review
<!-- Leakage, fair baseline, sample size, and three ways the conclusion could be wrong. -->

## Limitations
<!-- What was not verified; what the numbers do not show. -->

## Checklist
- [ ] No data, secrets, `.env`, checkpoints or indexes committed
- [ ] Docs / CHANGELOG updated (ADR if a rule or architecture changed)
- [ ] README still accurate: status, results table, extras, commands, layout (or "README: no change needed" with the reason)
- [ ] Conventional Commit title with `(#n)`
