# ADR-0001: Data, splits and use of the test set

Status: Accepted (recorded retrospectively on 2026-11-22; in force since M2, 2026-11-18)

## Context
The benchmark is built from UIT-ViQuAD 2.0 (Hugging Face mirror `taidng/UIT-ViQuAD2.0`). Questions are
written about passages; passages are grouped by Wikipedia article. 582 of 5,317 passages are used by
questions in more than one split. The mirror marks every test question as answerable.

## Decision
- Use the original train / validation / test splits. Gold document = the passage the question was written on.
- All model and configuration choices are made on validation (answerable questions).
- Test is used only when closing a milestone, once, for the configurations selected on validation,
  and always also on the subset of test questions whose passage no train question uses.
- Data is downloaded at run time and never committed. License: not stated on the mirror; v1.0 listed as
  CC BY-NC-SA 4.0 elsewhere (NEEDS_REVIEW against the UIT release page); use is non-commercial.

## Consequences
- Test has been used at M2, M3 and M3.1, each time only for pre-selected configurations. Repeated use
  still allows mild adaptive overfitting to test; conclusions with small margins (e.g. RRF k = 10) are
  labelled minor.
- Any model trained on train questions must exclude passages shared with test (not done so far).
