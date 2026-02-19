# DeepACSON SEM Dataset

## Overview
This dataset contains scanning electron microscopy (SEM) volumes of spinal cord white matter
with corresponding segmentation masks for myelin and myelinated axons.

## Acquisition
- Modality: Serial Block-Face Scanning Electron Microscopy (SBF-SEM)
- Instrument: Zeiss MultiSEM
- Voxel size HM: 8 × 8 × 8 nm³ (isotropic)
- Voxel size LM: 30 × 30 × 30 nm³ (isotropic)
- Tissue preparation: OsO4 + Uranyl Acetate + Lead staining

## Subjects
Multiple subjects from sham and TBI experimental groups with contralateral and ipsilateral samples.

## Derivatives
Segmentation masks generated using AxonDeepSeg v4.0:
- `desc-myelin_seg`: Binary mask of myelin sheaths
- `desc-axonInstances_dseg`: Instance segmentation of individual myelinated axons
- `desc-nucleus_seg`: Binary mask of cell nuclei (LM only)

## Contact
For questions, contact: your.email@institution.edu
