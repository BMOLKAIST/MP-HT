# Supplements for _Morphology-Preserving Holotomography: Quantitative Analysis of 3D Organoid Dynamics

**Supplements for _Morphology-Preserving Holotomography: Quantitative Analysis of 3D Organoid Dynamics_**  
MATLAB reference implementation for the MP-HT (morphology-preserving holotomography) representation and the Fig. 3 demonstration pipeline.

> **Paper (preprint):**  
> _Morphology-Preserving Holotomography: Quantitative Analysis of 3D Organoid Dynamics_ (arXiv:2512.22486)  
> DOI: 10.48550/arXiv.2512.22486

---

## Overview

Holotomography (HT) enables label-free 3D refractive-index (RI) imaging but often suffers from missing-cone–induced axial distortion.  
This repository provides MATLAB code to reproduce the **MP-HT representation** and the **Fig. 3 demo** (MP-HT visualization, epithelial/lumen segmentation, and dry-mass quantification) described in the accompanying manuscript.

---

## Repository contents

- `demo.m` — **Main entry point** (Fig. 3 demo).  
  Loads `dataset_fig3.mat`, runs MP-HT, segmentation, and prints dry-mass metrics.
- `MP_HT.m` — Core MP-HT transform (frequency-domain filtering + nonlinearity + smoothing).

> Additional `.m` files (visualization helpers, utilities, etc.) should be placed in the same folder (or added to MATLAB path).

---

## Requirements

- MATLAB **R2022b** or later (recommended)
- Toolboxes:
  - Image Processing Toolbox (required)
  - Parallel Computing Toolbox (optional; for `gpuArray` acceleration)

---

## Dataset (MAT files)

The `.mat` dataset files are hosted separately on IEEE DataPort:

- https://ieee-dataport.org/documents/supplementary-dataset-morphology-preserving-holotomography-quantitative-analysis-3d

### Dataset citation (IEEE DataPort)

ChulMin Oh, Jimin Cho, Juyeon Park, Hoyeon Lee, and YongKeun Park, “Supplementary Dataset for Morphology-Preserving Holotomography: Quantitative Analysis of 3D Organoid Dynamics.” IEEE DataPort.

### Expected file name / variables

`demo.m` assumes a MAT file named:

- `dataset_fig3.mat`

and expects variables such as:
- `RI`, `NAc`, `NAo`, `ResolutionX`, `ResolutionZ`, `RImedium`, `wavelength`  
(see `demo.m` for the exact list)

---

## Quick start

1. Clone this repository and place all `.m` files in the repo folder.
2. Download the dataset MAT file from IEEE DataPort.
3. Put `dataset_fig3.mat` in the same folder as `demo.m` (or edit the `load(...)` line in `demo.m`).
4. In MATLAB:
   ```matlab
   cd <repo_root>
   demo
