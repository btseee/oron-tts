# Progress: publication and logging

Plan: docs/superpowers/plans/2026-09-05-publication-and-logging.md
Branch: docs/publication-and-logging

Task 1: complete (commits 98495e6..c8553fc, review clean)
  Minor, for final triage:
  - re-running tb_report leaves the previous run's written tfevents in place; eval points accumulate across files rather than replace
  - tb_report docstring forward-references patch_trainer_logging.py (Task 5 creates it)
  - the copy+write combo test guards a TensorBoard-library behaviour, not a local code path
  Resolved during review: real eval.json has complete gender coverage; Task 6's publish omits --stages, which is why the hparams guard was blocking
