# BIDS Conversion Summary

## ✅ Conversion Completed Successfully

Your DeepACSON dataset has been converted to **BIDS Microscopy format** with **OME-Zarr** storage.

## 📁 Generated Structure

```
DeepACSONdataset/
├── dataset_description.json      # Root dataset metadata
├── README.md                      # Dataset documentation
├── participants.tsv               # Subject/animal metadata
├── samples.tsv                    # Sample metadata
├── code/
│   └── curate_em_to_bids.py      # This conversion script
├── sub-25/
│   └── micr/
│       ├── sub-25_sample-contra_acq-HM_SEM.ome.zarr/    # Raw SEM volume
│       └── sub-25_sample-contra_acq-HM_SEM.json         # Raw metadata
└── derivatives/
    └── labels/
        ├── dataset_description.json
        └── sub-25/
            └── micr/
                ├── sub-25_sample-contra_acq-HM_desc-myelin_seg.ome.zarr/          # Myelin mask
                ├── sub-25_sample-contra_acq-HM_desc-myelin_seg.json
                ├── sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.ome.zarr/  # Axon instances
                └── sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.json
```

## 📊 Data Processed

### Raw Volume
- **File**: `HM_25_contra.h5` → `sub-25_sample-contra_acq-HM_SEM.ome.zarr`
- **Shape**: (285, 1048, 1042) [Z, Y, X]
- **Dtype**: uint8
- **Voxel size**: 8 × 8 × 8 nm³ (isotropic)

### Segmentations (Derivatives)
1. **Myelin Binary Mask** (`desc-myelin_seg`)
   - Source: `HM_25_contra_myelin.mat`
   - Type: Binary segmentation (0/1)
   - Shape: (1042, 1048, 285) → converted to match raw

2. **Axon Instances** (`desc-axonInstances_dseg`)
   - Source: `HM_25_contra_myelinated_axons.mat`
   - Type: Discrete segmentation (instance IDs)
   - **367 unique axon instances** detected
   - Dtype: uint16

## 🎯 BIDS Entities Used

- `sub-25`: Subject/animal ID
- `sample-contra`: Hemisphere (contralateral)
- `acq-HM`: Acquisition type (High Magnification, 8nm)
- `desc-myelin`: Description for myelin masks
- `desc-axonInstances`: Description for instance segmentation
- `_seg`: Binary segmentation suffix
- `_dseg`: Discrete segmentation suffix
- `_SEM`: Scanning Electron Microscopy modality

## 📝 Next Steps

### 1. Validate BIDS Compliance
```bash
# Install BIDS validator (if not already installed)
npm install -g bids-validator

# Validate your dataset
bids-validator .
```

### 2. Update Metadata
Edit the following files with accurate information:

- **`dataset_description.json`**: Add proper authors, license, DOI
- **`participants.tsv`**: Update age, sex, experimental group
- **`samples.tsv`**: Update sample type and pathology info
- **`README.md`**: Add detailed acquisition protocol, contact info

### 3. Review JSON Sidecars
Check the `.json` files in:
- `sub-25/micr/` (raw data metadata)
- `derivatives/labels/sub-25/micr/` (segmentation metadata)

Update fields like:
- `Manufacturer`, `ManufacturersModelName`
- `ImageAcquisitionProtocol`
- `SampleStaining`

### 4. Verify OME-Zarr Integrity
```python
# Quick check to load and inspect a zarr volume
import zarr
raw = zarr.open('sub-25/micr/sub-25_sample-contra_acq-HM_SEM.ome.zarr', mode='r')
print(f"Zarr group: {raw}")
print(f"Arrays: {list(raw.arrays())}")
```

### 5. Add More Subjects
To convert additional subjects (e.g., `LM_25_contra.h5` for low magnification):
- Duplicate the conversion logic in `code/curate_em_to_bids.py`
- Change `acq-HM` → `acq-LM`
- Update voxel size accordingly

## 🔗 References

- **BIDS Specification**: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/microscopy.html
- **BIDS Examples**: https://github.com/bids-standard/bids-examples (micr_SEMzarr template)
- **OME-Zarr Spec**: https://ngff.openmicroscopy.org/latest/
- **NeuroPoly Curation**: https://intranet.neuro.polymtl.ca/data/dataset-curation.html

## ✨ What Makes This BIDS-Compliant

✅ Follows **Microscopy-BIDS** naming conventions  
✅ Uses **OME-Zarr** (`.ome.zarr`) format for multi-resolution storage  
✅ Includes `sample-<label>` entity (required for microscopy)  
✅ Separates raw data and derivatives properly  
✅ Binary masks use `_seg`, instance segmentations use `_dseg`  
✅ Includes all required metadata files  
✅ Follows NeuroPoly lab curation standards  

---

**Script**: `code/curate_em_to_bids.py`  
**Date**: 2026-02-11  
**Conversion tool**: Python + h5py + scipy + ome-zarr-py
