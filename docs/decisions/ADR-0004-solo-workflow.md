# ADR-0004: Solo-project workflow

Status: Accepted (2026-11-22)

## Context
The project has one author and no reviewer. The main risk is adjusting rules after seeing results. The
owner's playbook for solo AI/ML projects (written after a previous two-person project) replaces the
reviewer with time-stamped traces.

## Decision
1. `PLAN.md` holds milestones and done criteria, written before the work from M4 on.
2. Each experiment has `docs/protocols/<experiment>.md` (candidates, fixed settings, selection rule,
   speed constraint, decision rule) **merged to `main` before the run**. The merge commit is the timestamp.
3. Reproduction gate: an experiment script first reproduces the previous milestone's reference number
   from its results file and stops if it differs.
4. Rules are never changed after results are seen; a later analysis is labelled exploratory.
5. Branches `issue-<n>-<topic>` (GPU jobs keep `kaggle/<job>`); commits `<type>(scope): ... (#n)`; PR
   description: Summary, Changes, Evidence/Checks, Limitations; PRs with a new ADR or protocol get a
   self-review checklist plus an adversarial review ("three ways this conclusion could be wrong").
6. Commits are authored by the owner without Co-Authored-By trailers, and PR/issue texts carry no
   generated-by footers (applies from 2026-11-22; earlier commits keep their trailers since history is not rewritten).

## Consequences
- Earlier milestones' protocols lived in issue bodies (time-stamped by GitHub, but not in git); from M4
  they are files in the repository.
- Weekly evidence folders from the playbook are not used: there is no weekly schedule; evidence is the
  results JSON, the milestone report and the Release.
