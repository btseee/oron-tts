# Progress: publication and logging

Plan: docs/superpowers/plans/2026-09-05-publication-and-logging.md
Branch: docs/publication-and-logging

Task 1: complete (commits 98495e6..c8553fc, review clean)
  Minor, for final triage:
  - re-running tb_report leaves the previous run's written tfevents in place; eval points accumulate across files rather than replace
  - tb_report docstring forward-references patch_trainer_logging.py (Task 5 creates it)
  - the copy+write combo test guards a TensorBoard-library behaviour, not a local code path
  Resolved during review: real eval.json has complete gender coverage; Task 6's publish omits --stages, which is why the hparams guard was blocking
Task 2: complete (commits 32ddff1..cf3677a, review clean)
  Minor, for final triage:
  - `import numpy as np` sits inside the per-clip loop in write_summary_run; hoist it
  - no test covers mel_image on silence, nor write_summary_run with both dicts empty (both verified correct by the reviewer running them directly)
Task 1 fix round 2: CI hygiene regression (b71fb39) -- test_tb_report.py imported EventAccumulator at module scope; CI installs neither torch nor tensorboard as test deps. Guarded with pytest.importorskip, the repo's existing pattern. Both Task 1 reviews missed it because they ran only the task's own file; every later reviewer prompt requires the full suite.
Task 3: complete (commits 474faee..6a63f10, review clean after one fix round)
  Fixed: seven hand-typed numbers in the card body (plan's own example violated the plan's Global Constraints -- constraint governed); round(...,8) magic number making frontmatter and body table look like different measurements; SystemExit in a pure function; dead helper; duplicated selection.
  Minor, for final triage:
  - inference config numerals (nfe_step=32, cfg_strength=2.0, seed=0) remain literal in the usage sample -- configuration, not measurements
  - frontmatter() still self-computes best_checkpoint when called without best=, to stay independently testable
Task 4: complete (commits 495ca15..cf56364, review clean)
  cf56364 also cleared the ruff gate branch-wide: 7 errors, 2 of which pre-dated the branch (they arrived on main with the speaker_similarity commit). CI lint is now green.
  Minor, for final triage:
  - dataset_card_meta.main() raises a bare KeyError if HF_TOKEN is unset
  - `import yaml` duplicated at function scope in split_card and enrich (inherited from the plan's code)
  - Task 4 report claimed the files matched the brief "verbatim"; one unused `import yaml` was dropped, correctly
Task 5: complete (commits caae373..00945d2, review clean)
  Reviewer independently re-verified all three anchors against the real upstream trainer.py, derived the add_image tensor shape (1, n_mel, gen_time = valid CHW), and confirmed the grad-norm read is guarded on every path.
  Minor, for final triage:
  - no automated test asserts against the real upstream trainer.py (deliberate: CI installs neither torch nor F5-TTS; the loud failure fires at patch time on the pod)
