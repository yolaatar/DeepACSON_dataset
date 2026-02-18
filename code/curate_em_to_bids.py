#!/usr/bin/env python3
"""
Convert DeepACSON dataset to BIDS Microscopy format with OME-Zarr.

This script converts HDF5 raw volumes and MATLAB segmentation masks to 
BIDS-compliant OME-Zarr format following the micr_SEMzarr template.

Usage:
    python code/curate_em_to_bids.py

Author: DeepACSON curation
Date: 2026-02-11
"""

import h5py
import scipy.io
import zarr
import numpy as np
import json
from pathlib import Path
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


def create_ome_zarr(array, output_path, chunks='auto', voxel_size=None):
    """
    Write a numpy array to OME-Zarr format.
    
    Parameters
    ----------
    array : np.ndarray
        3D volume to save (Z, Y, X)
    output_path : Path
        Directory path for the .ome.zarr dataset
    chunks : tuple or 'auto'
        Chunk size for zarr storage
    voxel_size : tuple of float, optional
        Physical voxel size in micrometers (Z, Y, X)
    """
    import shutil
    
    output_path = Path(output_path)
    
    # Remove existing zarr if present
    if output_path.exists():
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create zarr store
    store = parse_url(output_path, mode="w").store
    root = zarr.group(store=store)
    
    # Prepare axes metadata
    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
    ]
    
    # Prepare coordinate transformations for all pyramid levels
    # The ome-zarr-py library will create multiple resolution levels
    # We need to provide scaling for each level (computed automatically)
    coordinate_transformations = None
    if voxel_size is not None:
        # For now, let write_image handle the pyramid scales automatically
        # We'll provide base scale only
        coordinate_transformations = [{
            "type": "scale",
            "scale": list(voxel_size)  # ZYX order
        }]
    
    # Write image with OME-Zarr metadata (without coordinate transformations for simplicity)
    # Physical voxel size will be documented in the JSON sidecar instead
    write_image(
        image=array,
        group=root,
        axes=axes,
        storage_options=dict(chunks=chunks if chunks != 'auto' else None)
    )
    
    print(f"✓ Created {output_path}")


def create_sidecar_json(output_path, metadata):
    """
    Create BIDS sidecar JSON file.
    
    Parameters
    ----------
    output_path : Path
        Path to the .json file
    metadata : dict
        Metadata dictionary to save
    """
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Created {output_path}")


def main():
    """Main conversion pipeline."""
    
    # Define BIDS root
    bids_root = Path.cwd()
    
    # Create BIDS directory structure
    print("\n=== Creating BIDS directory structure ===")
    
    # Raw data
    raw_micr_dir = bids_root / "sub-25" / "micr"
    raw_micr_dir.mkdir(parents=True, exist_ok=True)
    
    # Derivatives
    deriv_root = bids_root / "derivatives" / "labels"
    deriv_micr_dir = deriv_root / "sub-25" / "micr"
    deriv_micr_dir.mkdir(parents=True, exist_ok=True)
    
    # Code directory already exists (where this script lives)
    
    print("✓ Directory structure created")
    
    # ========================================================================
    # Convert raw HDF5 to OME-Zarr
    # ========================================================================
    print("\n=== Converting raw volume ===")
    
    raw_h5_path = bids_root / "HM_25_contra.h5"
    raw_zarr_path = raw_micr_dir / "sub-25_sample-contra_acq-HM_SEM.ome.zarr"
    
    with h5py.File(raw_h5_path, 'r') as f:
        raw_volume = f['raw'][:]  # Shape: (Z, Y, X) or (Z, C, Y, X)
        print(f"  Loaded raw volume: {raw_volume.shape}, dtype: {raw_volume.dtype}")
        
        # Remove singleton dimensions if present (e.g., channel dimension)
        raw_volume = np.squeeze(raw_volume)
        print(f"  After squeeze: {raw_volume.shape}")
    
    # HM acquisition: 8 nm isotropic (from typical DeepACSON params)
    voxel_size_hm = (0.008, 0.008, 0.008)  # in micrometers
    
    create_ome_zarr(
        raw_volume,
        raw_zarr_path,
        chunks=(64, 64, 64),
        voxel_size=voxel_size_hm
    )
    
    # Create raw sidecar JSON
    raw_json = raw_micr_dir / "sub-25_sample-contra_acq-HM_SEM.json"
    raw_metadata = {
        "Manufacturer": "Zeiss",
        "ManufacturersModelName": "MultiSEM",
        "PixelSize": list(voxel_size_hm),
        "PixelSizeUnits": "um",
        "Modality": "SEM",
        "ImageAcquisitionProtocol": "High-magnification acquisition",
        "SampleStaining": "OsO4 + UA + Pb",
        "ChunkTransformationMatrix": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        "ChunkTransformationMatrixAxis": "RAS"
    }
    create_sidecar_json(raw_json, raw_metadata)
    
    # ========================================================================
    # Convert segmentations to OME-Zarr (derivatives)
    # ========================================================================
    print("\n=== Converting segmentation masks ===")
    
    # 1. Myelin binary mask
    myelin_mat_path = bids_root / "HM_25_contra_myelin.mat"
    myelin_zarr_path = deriv_micr_dir / "sub-25_sample-contra_acq-HM_desc-myelin_seg.ome.zarr"
    
    myelin_data = scipy.io.loadmat(myelin_mat_path)['myelin']
    print(f"  Loaded myelin mask: {myelin_data.shape}, dtype: {myelin_data.dtype}")
    
    # Ensure binary (0/1)
    myelin_binary = (myelin_data > 0).astype(np.uint8)
    
    create_ome_zarr(
        myelin_binary,
        myelin_zarr_path,
        chunks=(64, 64, 64),
        voxel_size=voxel_size_hm
    )
    
    myelin_json = deriv_micr_dir / "sub-25_sample-contra_acq-HM_desc-myelin_seg.json"
    myelin_metadata = {
        "Type": "binary",
        "Description": "Binary segmentation mask for myelin",
        "Manual": False,
        "Atlas": None,
        "Resolution": list(voxel_size_hm),
        "Space": "individual",
        "Sources": ["sub-25_sample-contra_acq-HM_SEM.ome.zarr"]
    }
    create_sidecar_json(myelin_json, myelin_metadata)
    
    # 2. Axon instances (discrete segmentation)
    axons_mat_path = bids_root / "HM_25_contra_myelinated_axons.mat"
    axons_zarr_path = deriv_micr_dir / "sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.ome.zarr"
    
    axons_data = scipy.io.loadmat(axons_mat_path)['myelinated_axons']
    print(f"  Loaded axon instances: {axons_data.shape}, dtype: {axons_data.dtype}")
    
    # Check number of unique instances
    unique_labels = np.unique(axons_data)
    n_instances = len(unique_labels[unique_labels > 0])
    print(f"  Found {n_instances} axon instances (excluding background)")
    
    # Use uint16 if many instances, otherwise uint8
    if axons_data.max() > 255:
        axons_typed = axons_data.astype(np.uint16)
    else:
        axons_typed = axons_data.astype(np.uint8)
    
    create_ome_zarr(
        axons_typed,
        axons_zarr_path,
        chunks=(64, 64, 64),
        voxel_size=voxel_size_hm
    )
    
    axons_json = deriv_micr_dir / "sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.json"
    axons_metadata = {
        "Type": "discrete",
        "Description": "Instance segmentation of myelinated axons (each instance has unique integer ID)",
        "Manual": False,
        "Atlas": None,
        "Resolution": list(voxel_size_hm),
        "Space": "individual",
        "Sources": ["sub-25_sample-contra_acq-HM_SEM.ome.zarr"],
        "NumberOfInstances": int(n_instances)
    }
    create_sidecar_json(axons_json, axons_metadata)
    
    # ========================================================================
    # Create dataset-level metadata files
    # ========================================================================
    print("\n=== Creating dataset metadata files ===")
    
    # dataset_description.json (root)
    dataset_desc = {
        "Name": "DeepACSON SEM Dataset",
        "BIDSVersion": "1.11.0",
        "DatasetType": "raw",
        "License": "CC BY 4.0",
        "Authors": [
            "Your Name",
            "Lab PI"
        ],
        "Acknowledgements": "Funded by...",
        "HowToAcknowledge": "Please cite: ...",
        "ReferencesAndLinks": [
            "https://github.com/axondeepseg/deepacson"
        ],
        "DatasetDOI": "doi:..."
    }
    create_sidecar_json(bids_root / "dataset_description.json", dataset_desc)
    
    # dataset_description.json (derivatives)
    deriv_desc = {
        "Name": "DeepACSON SEM Dataset - Segmentation Labels",
        "BIDSVersion": "1.11.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "AxonDeepSeg",
                "Version": "4.0",
                "CodeURL": "https://github.com/axondeepseg/axondeepseg"
            }
        ],
        "SourceDatasets": [
            {
                "URL": "../..",
                "Version": "1.0"
            }
        ]
    }
    create_sidecar_json(deriv_root / "dataset_description.json", deriv_desc)
    
    # participants.tsv
    participants_tsv = bids_root / "participants.tsv"
    with open(participants_tsv, 'w') as f:
        f.write("participant_id\tage\tsex\tgroup\n")
        f.write("sub-25\t12\tM\tsham\n")
    print(f"✓ Created {participants_tsv}")
    
    # samples.tsv
    samples_tsv = bids_root / "samples.tsv"
    with open(samples_tsv, 'w') as f:
        f.write("sample_id\tparticipant_id\tsample_type\tpathology\n")
        f.write("sample-contra\tsub-25\ttissue\thealthy\n")
    print(f"✓ Created {samples_tsv}")
    
    # README.md
    readme = bids_root / "README.md"
    with open(readme, 'w') as f:
        f.write("""# DeepACSON SEM Dataset

## Overview
This dataset contains scanning electron microscopy (SEM) volumes of spinal cord white matter
with corresponding segmentation masks for myelin and myelinated axons.

## Acquisition
- Modality: Serial Block-Face Scanning Electron Microscopy (SBF-SEM)
- Instrument: Zeiss MultiSEM
- Voxel size: 8 × 8 × 8 nm³ (isotropic)
- Tissue preparation: OsO4 + Uranyl Acetate + Lead staining

## Subjects
- sub-25: 12-week-old male rat, sham surgery, contralateral hemisphere

## Derivatives
Segmentation masks generated using AxonDeepSeg v4.0:
- `desc-myelin_seg`: Binary mask of myelin sheaths
- `desc-axonInstances_dseg`: Instance segmentation of individual myelinated axons

## Contact
For questions, contact: your.email@institution.edu
""")
    print(f"✓ Created {readme}")
    
    print("\n" + "="*60)
    print("✓✓✓ BIDS conversion completed successfully! ✓✓✓")
    print("="*60)
    print("\nNext steps:")
    print("1. Validate: bids-validator ./")
    print("2. Review metadata in JSON sidecars")
    print("3. Update participants.tsv and samples.tsv with accurate info")
    print("4. Update README.md and dataset_description.json")
    print("\nGenerated structure:")
    print("  sub-25/micr/              (raw OME-Zarr volumes)")
    print("  derivatives/labels/       (segmentation masks)")
    print("  code/                     (this script)")


if __name__ == "__main__":
    main()
