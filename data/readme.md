# Data layout

Place the original datasets under this directory. Large datasets and checkpoints are excluded from Git. CAVE directories must contain the 31 spectral PNG files directly, in band order; remove RGB previews from these directories. The loader divides CAVE uint16 values by 65535.

```text
data/
  alog/train-fold-1.txt ... train-fold-5.txt
  alog/test-fold-1.txt  ... test-fold-5.txt
  acc/ibm-large-tensor.txt
  acc/ibm.mat
  beijing.npy
  ctr_50k.npy
  misc/{Plane,Peppers,House,Sailboat}.tiff
  Videos/{foreman,news,tempete,paris}_cif.y4m
  MSIs/{beers,clay,flowers,balloons,beads,fruits}/<31 spectral PNGs>
  PaviaU.mat
  dc.tif
  responses/TenFuse/SRF_matrix.mat
```

- **Alog:** one coordinate triple and value per row; the loader retains the existing split and normalization.
- **ACC:** fixed training file and the 50 test subsets stored in MATLAB variable `data`. See the ACC protocol note in the main README.
- **Air / Click:** the preprocessed NumPy objects consumed by `utils/ob_dataV2.py:get_air_data` and `get_click_data`; raw source CSVs are not drop-in replacements. NumPy object loading uses pickle, so use trusted dataset files.
- **RGB:** USC-SIPI TIFFs, dimensions matching the image configuration.
- **Video:** Xiph CIF Y4M files, 352 × 288, first ten RGB frames; `--type y4m` is the paper's color-video path. Conversion follows the existing approximate YUV-to-RGB implementation.
- **CAVE:** Beers, Clay, Flowers, Balloons for completion/SR; Balloons, Beads, Flowers, Fruits for denoising. Completion/denoising resize to 256 × 256. SR uses full 512 × 512 cubes.
- **PaviaU:** `PaviaU.mat`, variable `paviaU`, H × W × 103.
- **WDC:** `dc.tif`, stored as 191 × H × W. Full-scene normalization precedes crop extraction.
- **Responses:** the release includes the small response matrices. PaviaU uses the root-level `IKONOS_Spectral_Responses.mat`; CAVE uses `response_coefficient.mat`. WDC provenance and SHA256 are recorded beside `SRF_matrix.mat`.

Run `python scripts/check_data.py all` from the release root to check paths. This checks layout; actual loaders validate arrays during execution. No dataset or trained model download is performed automatically.

Dataset source addresses recorded in the manuscript: CAVE — https://cave.cs.columbia.edu/repository/Multispectral ; USC-SIPI — https://sipi.usc.edu/database/database.php ; Xiph — https://media.xiph.org/video/derf/ . The tabular experiments require the original preprocessed splits supplied with this research workspace.
