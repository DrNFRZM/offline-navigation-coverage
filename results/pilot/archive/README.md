# Archived pilot record

`seed_level_first_evaluation_in_training_process.csv` is the seed-level table of the pilot's
first evaluation, which ran inside each training process. The pilot evaluations were later
regenerated in fresh processes after the secondary metric (success within H/3) was added; those
regenerated files are the ones in `results/pilot/runs/` and `results/pilot/tables/`.

The two evaluations disagree on one medium-maze BC episode outcome (success 0.99 here, 1.00 in the
regenerated table) and on first-success times (for example U-maze BC 68.1 here vs 66.1). This file
is the source of the "first evaluation" numbers quoted in `docs/protocol.md` (amendment 1) and
`docs/decisions.md` (D13). It lacks the `fast_success_*` columns, which did not exist yet.
