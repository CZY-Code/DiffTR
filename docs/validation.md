# Release validation

Validation performed on 2026-09-29 using the existing Python 3.10 / PyTorch 2.4.1 environment.

| Check | Result |
|---|---|
| Core and release integration tests (`unittest discover -s tests`) | 33 passed |
| SR baseline numerical tests | 6 passed |
| Time-budget tests | 2 passed |
| Ablation intervention tests | 3 passed |
| Synthetic CPU training smoke test | Two training updates, saved checkpoint, finite Soft1000 reconstruction, converged terminal solver |
| Exported-directory test | Code copied outside the research tree; imports resolved inside the export; all 33 tests passed again |
| Entry-point checks | 21 training/analysis entry points accepted `--help`; the additional SR plot entry point also passed |
| Fresh observation/configuration checks | WDC patch 0, PaviaU patch 0, and CAVE Beers at DS4/sigma0.5 matched archived GT, LR-HSI, HR-MSI and response hashes; model/diffusion/trainer settings matched, excluding output paths |
| Source syntax | Python compilation passed; shell syntax check passed |

The exported-directory data check used temporary read-only links to the existing raw datasets. Those links and generated verification outputs are outside the release and are not packaged. The release directory contains no dataset cubes or pretrained weights.

These checks validate packaging, experiment coverage, observation preparation, numerical operators, model training/inference on small data, and baseline adapters. Full-budget GPU experiments and published metric tables have not been rerun during release preparation. Baseline tests exercise their numerical components rather than full model training.
