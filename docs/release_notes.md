# Release correspondence

The source reference is `paper/IEEE-latex/IEEE-DiffTR-revised.tex`, including its SR appendix and ablation tables, as present when this release was assembled. `source_manifest.json` records original source hashes. `release_manifest.json` records packaged file hashes, including modifications and new files.

## Packaging changes

- Keep the DiffTR networks, diffusion losses, corruption models, data normalization, and observation solvers; retain their existing task APIs.
- Add a dependency file, task launcher, fresh SR manifest preparation, complete-cell aggregation, data checks, a tiny synthetic training test, and standalone inference profiling.
- Replace dependencies on archived result plans with `scripts/sr_protocol.py`, which generates all 50 scene/degradation combinations from raw inputs.
- Use the existing time-budget baseline solvers and replace the old DC-10 inference in their training driver with the manuscript's Soft1000 reconstruction. The primary baseline budget remains setup plus 10,000 DiffTR updates.
- Point sampling and terminal ablations at newly generated checkpoints and verify observation/checkpoint hashes before inference.
- Separate completion checkpoints by task, rank/scene, sampling mode/rate, seed, and fold; create parent directories. Add JSON completion summaries.
- Import NumPy before PyTorch in tabular entry points to avoid an MKL/OpenMP import-order conflict in the local Conda environment.
- Honor launcher GPU visibility. Use the active Python interpreter and relative repository paths.
- Set both color-video completion YAML training budgets to 40,000 updates, as stated in the manuscript's implementation table. Original YAMLs used shorter budgets; exact archived scores may therefore differ.
- Include the manuscript's Fig. 1 as vector PDF and PNG.
- Exclude data cubes, trained weights, result arrays, caches, paper-editing utilities, automatic publication tools, and the estimated/unvalidated ScoreTR table-filling scripts.

## Reproduction boundaries

- ACC's available code trains once against a fixed training set and evaluates 50 test subsets. It does not perform the five-fold training stated in the manuscript. The source split is preserved, with no synthesized alternative split.
- Tabular/visual training retains the original evaluation and checkpoint-selection behavior. The denoising main table mixes reference-selected BCD checkpoints and final-EMA likelihood runs, as recorded in the paper.
- The main CAVE denoising comparison has different spatial resolutions for DiffTR and baselines. Crop ablations are separate protocols.
- Four SR methods have complete executable adapters. ScoreTR HSI training is available, but the recorded full HSI reproduction was stopped; no estimates are included as measurements.
- Other comparison-table baselines have no matching runnable reproduction in the workspace. CATTE and generic diffusion demo repositories are not paper baseline reproductions and are excluded from the task launchers.
- The paper's baseline primary time budget depends on measured hardware performance. Repeating it on another device can change the number of eligible baseline updates.
- The WDC response retains TenFuse's 191-band ordering and weights; the source provides no independent wavelength table. Its mapping follows the paper's stated band-order assumption.
- Released SR outputs are newly sampled from newly trained checkpoints. Their metrics are averaged over three reconstructions, not over an averaged image.

## Validation

See `validation.md` for the checks actually run while preparing this release. Full training matrices have not been rerun as part of packaging.
