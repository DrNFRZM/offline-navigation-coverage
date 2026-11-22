# Audit evidence

- **This folder (current):** evidence from the final run of the checks after the corrections made in
  response to the independent review (2026-11-22): `scripts/final_check.ps1 -Full` on the final code
  (81 tests; `offnav verify` for main, pilot and smoke; independent recomputation with 26 documented
  claims, including the post hoc failure-aware times; reanalysis reproduction; re-evaluation of six
  policies; table regeneration; smoke rerun with the current code; CPU job). Files:
  `final_check_full_summary.txt`, `pytest_results.txt`, `integrity_checks.json`,
  `reproduction_checks.json`, `reanalysis_check.json`, `independent_recompute_summary.json`,
  `claims_check.csv`, `recomputed_*.csv`, `run_records_check.csv`, `checkpoint_check.csv`,
  `h3_*.csv`, `composition_effects.csv`. The clean-environment validation of the public release is
  described in `../REVIEW_RESPONSE.md`; its record with the archive's SHA-256 and git tree hash is kept
  next to the archive (`release/validation_record.json`, not part of the release), because a file
  cannot contain the hash of the archive it is in.
- **`first_review/`:** the evidence of the first final review (same day, before the independent
  review; 55 tests, 14 claims), kept unchanged for the record. Superseded where it differs.
