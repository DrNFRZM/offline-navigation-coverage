# Development scripts (ad hoc, kept for traceability)

Small scripts used during development to inspect data, measure throughput and check
reproducibility. They are not part of the study pipeline (`offnav` CLI) and were not needed to
produce any result in `results/`; they are kept because numbers they printed are cited in the
documentation. Run them from the repository root with the package installed.

| Script | Used for | Cited in |
|---|---|---|
| `inspect1.py` | First look at the raw Minari episodes, metadata and specs before writing the audit | `docs/data.md` (audit design) |
| `bench.py` | Eager update throughput on GPU and CPU (6 threads) | `docs/decisions.md` D7 |
| `bench2.py` | Single-thread CPU throughput and CUDA-graph replay throughput | D7, `docs/reproducibility.md` |
| `thread_check.py`, `thread_check2.py` | Evaluation repeatability across thread counts, deterministic mode and CUDA initialisation | D13, protocol amendment 1 |
| `peek.py`, `core_look.py` | Read-only looks at completed main-study runs while the grid was running | none (no decision depended on them) |
| `reeval.py` | Post-study re-evaluation of two main runs in fresh processes, compared episode by episode | D13, research note section 8 |
| `compare_copy.py` | SHA-256 comparison of the original project folder and this copy | `local_records/` (local only) |
| `checkpoint_index.py` | Builds `artifacts/checkpoint_index.csv` | `artifacts/README.md` |

The main study's training started at 2026-11-21T20:59:07Z (recorded when the grid was launched).
