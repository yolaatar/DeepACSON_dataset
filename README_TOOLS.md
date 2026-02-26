# DeepACSON Dataset - BIDS Conversion Tools

Tools for converting DeepACSON SEM microscopy datasets to BIDS (Brain Imaging Data Structure) format with OME-Zarr storage.

## Overview

This repository contains scripts to convert DeepACSON electron microscopy datasets from HDF5/MATLAB formats to **BIDS-compliant OME-Zarr** following the [Microscopy-BIDS specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/microscopy.html).

## Features

- ✅ Converts HDF5 raw volumes to OME-Zarr multi-resolution format
- ✅ Converts MATLAB segmentation masks to OME-Zarr
- ✅ Generates BIDS-compliant directory structure and metadata
- ✅ Creates all required BIDS sidecar JSON files
- ✅ Visualization tools (matplotlib and napari)
- ✅ Interactive Jupyter notebook for data exploration

## Requirements

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install h5py zarr ome-zarr numpy scipy matplotlib napari PyQt5 jupyter ipywidgets
```

## Input Data Format

Your source data should include:
- `HM_25_contra.h5` - Raw SEM volume (HDF5 format, dataset at key `/raw`)
- `HM_25_contra_myelin.mat` - Myelin binary mask (MATLAB format)
- `HM_25_contra_myelinated_axons.mat` - Axon instance segmentation (MATLAB format)

## Usage

### 1. Convert Dataset to BIDS

```bash
python code/curate_em_to_bids.py
```

This will create:
```
DeepACSONdataset/
├── dataset_description.json      # Root dataset metadata
├── README.md                      # Dataset documentation
├── participants.tsv               # Subject metadata
├── samples.tsv                    # Sample metadata
├── code/
│   └── curate_em_to_bids.py      # This conversion script
├── sub-25/
│   └── micr/
│       ├── sub-25_sample-contra_acq-HM_SEM.ome.zarr/
│       └── sub-25_sample-contra_acq-HM_SEM.json
└── derivatives/
    └── labels/
        └── sub-25/
            └── micr/
                ├── sub-25_sample-contra_acq-HM_desc-myelin_seg.ome.zarr/
                ├── sub-25_sample-contra_acq-HM_desc-myelin_seg.json
                ├── sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.ome.zarr/
                └── sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.json
```

### 2. Visualize Data

**Quick 2D slice view:**
```bash
python code/visualize_bids_data.py --mode matplotlib
```

**Interactive 3D viewer:**
```bash
python code/visualize_bids_data.py --mode napari
```

**Jupyter notebook exploration:**
```bash
jupyter notebook explore_bids_data.ipynb
```

### 3. Validate BIDS Compliance

```bash
# Install BIDS validator
npm install -g bids-validator

# Validate your dataset
bids-validator ./
```

## BIDS Naming Convention

The conversion uses the following BIDS entities:

- `sub-25`: Subject/animal ID
- `sample-contra`: Hemisphere as sample (contralateral)
- `acq-HM`: Acquisition type (High Magnification, 8nm voxel size)
- `desc-myelin`: Binary myelin mask
- `desc-axonInstances`: Instance segmentation (discrete labels)
- `_seg`: Binary segmentation suffix
- `_dseg`: Discrete segmentation suffix
- `_SEM`: Scanning Electron Microscopy modality

## Customization

To convert your own dataset, modify `code/curate_em_to_bids.py`:

1. **Change subject ID**: Update `sub-25` to your subject identifier
2. **Change sample name**: Update `sample-contra` to your sample label
3. **Adjust voxel size**: Modify `voxel_size_hm` (currently 8nm isotropic)
4. **Update metadata**: Edit JSON sidecar content (manufacturer, staining protocol, etc.)

For multiple acquisitions (e.g., LM - Low Magnification):
- Use `acq-LM` entity
- Adjust voxel size accordingly

## File Formats

### Input
- **HDF5** (`.h5`): Raw microscopy volumes
- **MATLAB** (`.mat`): Segmentation masks

### Output
- **OME-Zarr** (`.ome.zarr/`): Multi-resolution pyramid format
  - Chunked storage for efficient access
  - Multi-scale for visualization at different zoom levels
  - OME-NGFF metadata for interoperability

## References

- **BIDS Specification**: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/microscopy.html
- **BIDS Examples**: https://github.com/bids-standard/bids-examples (see `micr_SEMzarr`)
- **OME-Zarr Spec**: https://ngff.openmicroscopy.org/latest/
- **NeuroPoly Curation Guidelines**: https://intranet.neuro.polymtl.ca/data/dataset-curation.html



