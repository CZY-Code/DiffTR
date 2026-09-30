# Third-party source and attribution

The SR adapter sources are accompanied by snapshots of the author repositories. `SOURCE.json` records the original commit and file hashes. Only source, documentation, and available license files are packaged; datasets, ZIP archives, trained weights, and encrypted MATLAB functions are omitted.

| Method | Repository | Pinned commit |
|---|---|---|
| CS2DIPs | [Ambitionfy/CS2DIPs](https://github.com/Ambitionfy/CS2DIPs) | `943a282f058cb8787dcbcec97348bedc45ce85cf` |
| GTNN | [renweidian/GTNN](https://github.com/renweidian/GTNN) | `997132b1db2a8db01bb9f8e23c25374af790fbe1` |
| GNLTN | [xuhonghui96/GNLTN](https://github.com/xuhonghui96/GNLTN) | `fed96ddc97c530e9f2a5e91d48670dfab24b0ef6` |
| EDIP-Net | [JiaxinLiCAS/EDIP-Net_TGRS](https://github.com/JiaxinLiCAS/EDIP-Net_TGRS) | `89413738cb8bfa6cec2693d40c6120ed2e532a04` |
| CTDF | [tingxu113/CTDF](https://github.com/tingxu113/CTDF) | `62a7cd7df870f39e3455081f23653355ab26ae55` |

CTDF is retained as source provenance/reference material; no CTDF executable adapter is supplied. SSLRDN has no implementation in the available workspace.

## Adaptations

- **GTNN/GNLTN:** nonlocal grouping and three transformed SVD/shrinkage updates follow the author implementations. The data-consistency step uses the actual replicate-boundary blur/block-mean adjoints and CG. Initialization uses PyTorch bicubic; clustering uses Lloyd iterations with a float64 refinement when needed. WDC uses the Pavia configuration. Selected SVD operations run through CPU LAPACK.
- **CS2DIPs:** author CGDIP architecture, rank 35, width 256, learning rate 0.001, seed 766. The adapter supplies torchvision deformable convolution for the MMCV interface. Output selection uses observed-data Huber loss.
- **EDIP-Net:** author spectral networks and width-240 dual U-Nets. Known operators replace blind degradation estimation. Stage 2 receives 2/9 of the optimization budget; stage 3 receives the remainder. GT-based best-PSNR/SAM selection is replaced by eligible fixed-budget outputs. Learning rate starts at 0.004.
- **Time budgets:** GTNN/GNLTN/CS2DIPs caps are 100/50/5000 updates; EDIP-Net caps are 4000/14000 for stages 2/3. Primary-budget reconstructions are used for the comparison.

## Other bundled code

- `scoretr_study/vendor/` preserves the local ScoreTR EBM and random Fourier feature snapshot. The workspace supplied no pinned upstream commit for this snapshot; original file hashes are in `source_manifest.json`.
- `rff/` contains the local Fourier-feature implementation (`__version__ = 1.0.1`).
- The diffusion/training code builds on the local diffusion implementation associated with denoising-diffusion-pytorch; it has been extended for coordinate-conditioned tensor recovery. Local source hashes are recorded rather than inventing an upstream revision.
- `data/responses/TenFuse/provenance.json` records the source URL and hashes of the WDC response. The two root-level response MAT files are copied from the research workspace.

## License information

Existing third-party license notices are retained in their source directories. Several source snapshots provide no top-level license file; `SOURCE.json` establishes provenance, not a replacement license. No new blanket license has been assigned to the original research code or third-party files. Applicable upstream terms remain with each component.
