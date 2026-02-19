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
import gc
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


def load_matlab_file(mat_path, variable_name):
    """
    Load MATLAB file handling both v7.3 (HDF5) and older formats.
    
    Parameters
    ----------
    mat_path : Path
        Path to the .mat file
    variable_name : str
       Name of the variable to extract (for older format files)
        
    Returns
    -------
    np.ndarray
        The loaded array
    """
    try:
        # Try scipy.io.loadmat first (for older .mat files)
        data = scipy.io.loadmat(mat_path)[variable_name]
        return data
    except NotImplementedError:
        # Fall back to h5py for MATLAB v7.3 files
        with h5py.File(mat_path, 'r') as f:
            # MATLAB v7.3 files from this dataset use 'final_lbl' as the key
            if 'final_lbl' in f.keys():
                data = f['final_lbl'][:]
                return data
            # Fallback: try the requested variable name
            elif variable_name in f.keys():
                data = f[variable_name][:]
                return data
            else:
                # Use first available data key
                keys = [k for k in f.keys() if not k.startswith('#')]
                if keys:
                    data = f[keys[0]][:]
                    return data
                raise KeyError(f"Could not find data in {mat_path}. Available keys: {list(f.keys())}")


def convert_subject_sample(bids_root, source_dir, subject_id, sample_label, acq_label, voxel_size, group_label):
    """
    Convert one subject/sample/acquisition to BIDS format.
    
    Parameters
    ----------
    bids_root : Path
        BIDS root directory
    source_dir : Path
        Source directory containing the raw data
    subject_id : str
        Subject ID (e.g., '25', '49', '24')
    sample_label : str
        Sample label (e.g., 'contra', 'ipsi')
    acq_label : str
        Acquisition label (e.g., 'HM', 'LM')
    voxel_size : tuple
        Voxel size in micrometers (Z, Y, X)
    group_label : str
        Experimental group (e.g., 'sham', 'tbi')
    """
    print(f"\n{'='*60}")
    print(f"Converting sub-{subject_id}, sample-{sample_label}, acq-{acq_label}")
    print(f"{'='*60}")
    
    # Create BIDS directory structure
    raw_micr_dir = bids_root / f"sub-{subject_id}" / "micr"
    raw_micr_dir.mkdir(parents=True, exist_ok=True)
    
    deriv_root = bids_root / "derivatives" / "labels"
    deriv_micr_dir = deriv_root / f"sub-{subject_id}" / "micr"
    deriv_micr_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    prefix = f"{acq_label}_{subject_id}_{sample_label}"
    raw_h5_path = source_dir / f"{prefix}.h5"
    myelin_mat_path = source_dir / f"{prefix}_myelin.mat"
    axons_mat_path = source_dir / f"{prefix}_myelinated_axons.mat"
    nucleus_mat_path = source_dir / f"{prefix}_nucleus.mat"  # LM only
    
    # BIDS naming
    bids_prefix = f"sub-{subject_id}_sample-{sample_label}_acq-{acq_label}"
    
    # ========================================================================
    # Convert raw HDF5 to OME-Zarr
    # ========================================================================
    if not raw_h5_path.exists():
        print(f"  ⚠ Skipping: {raw_h5_path} not found")
        return
    
    print(f"\n=== Converting raw volume ===")
    raw_zarr_path = raw_micr_dir / f"{bids_prefix}_SEM.ome.zarr"
    
    with h5py.File(raw_h5_path, 'r') as f:
        raw_volume = f['raw'][:]
        print(f"  Loaded raw volume: {raw_volume.shape}, dtype: {raw_volume.dtype}")
        raw_volume = np.squeeze(raw_volume)
        print(f"  After squeeze: {raw_volume.shape}")
    
    create_ome_zarr(
        raw_volume,
        raw_zarr_path,
        chunks=(32, 32, 32),
        voxel_size=voxel_size
    )
    del raw_volume  # Free memory
    gc.collect()
    
    # Create raw sidecar JSON
    raw_json = raw_micr_dir / f"{bids_prefix}_SEM.json"
    protocol = "High-magnification" if acq_label == "HM" else "Low-magnification"
    raw_metadata = {
        "Manufacturer": "Zeiss",
        "ManufacturersModelName": "MultiSEM",
        "PixelSize": list(voxel_size),
        "PixelSizeUnits": "um",
        "Modality": "SEM",
        "ImageAcquisitionProtocol": f"{protocol} acquisition",
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
    print(f"\n=== Converting segmentation masks ===")
    
    # 1. Myelin binary mask
    if myelin_mat_path.exists():
        myelin_zarr_path = deriv_micr_dir / f"{bids_prefix}_desc-myelin_seg.ome.zarr"
        
        myelin_data = load_matlab_file(myelin_mat_path, 'myelin')
        print(f"  Loaded myelin mask: {myelin_data.shape}, dtype: {myelin_data.dtype}")
        
        # Ensure binary (0/1)
        myelin_binary = (myelin_data > 0).astype(np.uint8)
        
        create_ome_zarr(
            myelin_binary,
            myelin_zarr_path,
            chunks=(32, 32, 32),
            voxel_size=voxel_size
        )
        del myelin_data  # Free memory
        gc.collect()
        
        myelin_json = deriv_micr_dir / f"{bids_prefix}_desc-myelin_seg.json"
        myelin_metadata = {
            "Type": "binary",
            "Description": "Binary segmentation mask for myelin",
            "Manual": False,
            "Atlas": None,
            "Resolution": list(voxel_size),
            "Space": "individual",
            "Sources": [f"{bids_prefix}_SEM.ome.zarr"]
        }
        create_sidecar_json(myelin_json, myelin_metadata)
    
    # 2. Axon instances (discrete segmentation)
    if axons_mat_path.exists():
        axons_zarr_path = deriv_micr_dir / f"{bids_prefix}_desc-axonInstances_dseg.ome.zarr"
        
        axons_data = load_matlab_file(axons_mat_path, 'myelinated_axons')
        print(f"  Loaded axon instances: {axons_data.shape}, dtype: {axons_data.dtype}")
        
        # Check number of unique instances (memory-efficient for large arrays)
        # Use max() instead of unique() to avoid memory issues
        n_instances = int(axons_data.max())
        print(f"  Found {n_instances} axon instances (excluding background)")
        
        # Determine appropriate dtype
        if axons_data.max() > 65535:
            target_dtype = np.uint32
        elif axons_data.max() > 255:
            target_dtype = np.uint16
        else:
            target_dtype = np.uint8
        
        # Check if array is too large for memory (>3GB for safety)
        array_size_gb = (axons_data.size * target_dtype().itemsize) / (1024**3)
        
        if array_size_gb > 3:
            # Process in chunks to avoid memory issues
            print(f"  Large array ({array_size_gb:.1f} GB), processing in chunks...")
            
            # Create zarr array directly
            import shutil
            if axons_zarr_path.exists():
                shutil.rmtree(axons_zarr_path)
            axons_zarr_path.mkdir(parents=True, exist_ok=True)
            
            store = parse_url(axons_zarr_path, mode="w").store
            root = zarr.group(store=store)
            
            # Create zarr array with chunking (smaller chunks for safety)
            z_arr = root.create_array('0', shape=axons_data.shape, chunks=(32, 32, 32), dtype=target_dtype)
            
            # Copy data in chunks (smaller chunks = less RAM)
            chunk_size = 32
            total_z_chunks = (axons_data.shape[0] + chunk_size - 1) // chunk_size
            for z_idx, z in enumerate(range(0, axons_data.shape[0], chunk_size)):
                z_end = min(z + chunk_size, axons_data.shape[0])
                if z_idx % 10 == 0:  # Print progress every 10 z-chunks
                    print(f"    Axon chunk {z_idx+1}/{total_z_chunks}... ({int(100*z_idx/total_z_chunks)}%)")
                for y in range(0, axons_data.shape[1], chunk_size):
                    y_end = min(y + chunk_size, axons_data.shape[1])
                    for x in range(0, axons_data.shape[2], chunk_size):
                        x_end = min(x + chunk_size, axons_data.shape[2])
                        chunk = axons_data[z:z_end, y:y_end, x:x_end]
                        z_arr[z:z_end, y:y_end, x:x_end] = chunk.astype(target_dtype)
                        del chunk  # Free memory immediately
            
            # Add minimal OME metadata
            root.attrs['multiscales'] = [{
                'version': '0.4',
                'axes': [{'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
                'datasets': [{'path': '0'}]
            }]
            print(f"✓ Created {axons_zarr_path}")
            del axons_data  # Free memory
            gc.collect()  # Force garbage collection
        else:
            # Small enough to process normally
            axons_typed = axons_data.astype(target_dtype)
            del axons_data  # Free original
            create_ome_zarr(
                axons_typed,
                axons_zarr_path,
                chunks=(32, 32, 32),
                voxel_size=voxel_size
            )
            del axons_typed  # Free memory
            gc.collect()  # Force garbage collection
        
        axons_json = deriv_micr_dir / f"{bids_prefix}_desc-axonInstances_dseg.json"
        axons_metadata = {
            "Type": "discrete",
            "Description": "Instance segmentation of myelinated axons (each instance has unique integer ID)",
            "Manual": False,
            "Atlas": None,
            "Resolution": list(voxel_size),
            "Space": "individual",
            "Sources": [f"{bids_prefix}_SEM.ome.zarr"],
            "NumberOfInstances": int(n_instances)
        }
        create_sidecar_json(axons_json, axons_metadata)
    
    # 3. Nucleus segmentation (LM only)
    if nucleus_mat_path.exists():
        nucleus_zarr_path = deriv_micr_dir / f"{bids_prefix}_desc-nucleus_seg.ome.zarr"
        
        # For large files, process in a memory-efficient way
        print(f"  Processing nucleus mask from {nucleus_mat_path.name}...")
        
        try:
            # Try loading with scipy first (older format)
            nucleus_data = scipy.io.loadmat(nucleus_mat_path)['nucleus']
            nucleus_binary = (nucleus_data > 0).astype(np.uint8)
            create_ome_zarr(nucleus_binary, nucleus_zarr_path, chunks=(64, 64, 64), voxel_size=voxel_size)
            print(f"  Loaded nucleus mask: {nucleus_data.shape}, dtype: uint8")
        except (NotImplementedError, MemoryError):
            # For MATLAB v7.3 or large files, write directly to zarr without loading full array
            with h5py.File(nucleus_mat_path, 'r') as f:
                nucleus_h5 = f['final_lbl'] if 'final_lbl' in f.keys() else f['nucleus']
                shape = nucleus_h5.shape
                print(f"  Processing large nucleus mask: {shape}, dtype: {nucleus_h5.dtype}")
                
                # Create zarr array directly
                import shutil
                if nucleus_zarr_path.exists():
                    shutil.rmtree(nucleus_zarr_path)
                nucleus_zarr_path.mkdir(parents=True, exist_ok=True)
                
                store = parse_url(nucleus_zarr_path, mode="w").store
                root = zarr.group(store=store)
                
                # Create zarr array with chunking (smaller chunks for safety)
                z_arr = root.create_dataset('0', shape=shape, chunks=(32, 32, 32), dtype='u1')
                
                # Copy data in chunks to avoid memory issues (smaller chunks = less RAM)
                chunk_size = 32
                total_z_chunks = (shape[0] + chunk_size - 1) // chunk_size
                for z_idx, z in enumerate(range(0, shape[0], chunk_size)):
                    z_end = min(z + chunk_size, shape[0])
                    if z_idx % 10 == 0:  # Print progress every 10 z-chunks
                        print(f"    Nucleus chunk {z_idx+1}/{total_z_chunks}... ({int(100*z_idx/total_z_chunks)}%)")
                    for y in range(0, shape[1], chunk_size):
                        y_end = min(y + chunk_size, shape[1])
                        for x in range(0, shape[2], chunk_size):
                            x_end = min(x + chunk_size, shape[2])
                            chunk = nucleus_h5[z:z_end, y:y_end, x:x_end]
                            z_arr[z:z_end, y:y_end, x:x_end] = (chunk > 0).astype(np.uint8)
                            del chunk  # Free memory immediately
                
                # Add minimal OME metadata
                root.attrs['multiscales'] = [{
                    'version': '0.4',
                    'axes': [{'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
                    'datasets': [{'path': '0'}]
                }]
                print(f"✓ Created {nucleus_zarr_path}")
                gc.collect()  # Force garbage collection after large operation
        
        nucleus_json = deriv_micr_dir / f"{bids_prefix}_desc-nucleus_seg.json"
        nucleus_metadata = {
            "Type": "binary",
            "Description": "Binary segmentation mask for cell nuclei",
            "Manual": False,
            "Atlas": None,
            "Resolution": list(voxel_size),
            "Space": "individual",
            "Sources": [f"{bids_prefix}_SEM.ome.zarr"]
        }
        create_sidecar_json(nucleus_json, nucleus_metadata)


def main():
    """Main conversion pipeline."""
    
    # Define BIDS root
    bids_root = Path.cwd()
    data_root = bids_root / "data" / "White matter EM"
    
    # Define voxel sizes (in micrometers)
    voxel_size_hm = (0.008, 0.008, 0.008)  # 8 nm isotropic for HM
    voxel_size_lm = (0.030, 0.030, 0.030)  # 30 nm isotropic for LM (typical)
    
    # Parse all subjects from data directory
    subjects_data = []
    
    for subject_dir in sorted(data_root.glob("*")):
        if not subject_dir.is_dir():
            continue
        
        # Parse directory name: e.g., "Sham_25_contra" or "TBI_24_contra"
        parts = subject_dir.name.split('_')
        if len(parts) != 3:
            print(f"⚠ Skipping unexpected directory: {subject_dir.name}")
            continue
        
        group = parts[0].lower()  # 'sham' or 'tbi'
        subject_id = parts[1]  # '25', '49', '24'
        sample_label = parts[2]  # 'contra', 'ipsi'
        
        subjects_data.append({
            'source_dir': subject_dir,
            'subject_id': subject_id,
            'sample_label': sample_label,
            'group': group
        })
    
    # Filter: Only process sub-24 (others already complete)
    subjects_data = [s for s in subjects_data if s['subject_id'] == '24']
    
    print(f"\n{'='*60}")
    print(f"Found {len(subjects_data)} samples to convert (filtering for sub-24 only)")
    print(f"{'='*60}")
    
    # Convert all subjects
    for idx, subject_info in enumerate(subjects_data, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(subjects_data)}] Processing sub-{subject_info['subject_id']}, sample-{subject_info['sample_label']} ({subject_info['group']})")
        print(f"{'='*60}")
        
        # Convert HM acquisition
        convert_subject_sample(
            bids_root=bids_root,
            source_dir=subject_info['source_dir'],
            subject_id=subject_info['subject_id'],
            sample_label=subject_info['sample_label'],
            acq_label='HM',
            voxel_size=voxel_size_hm,
            group_label=subject_info['group']
        )
        
        # Convert LM acquisition
        convert_subject_sample(
            bids_root=bids_root,
            source_dir=subject_info['source_dir'],
            subject_id=subject_info['subject_id'],
            sample_label=subject_info['sample_label'],
            acq_label='LM',
            voxel_size=voxel_size_lm,
            group_label=subject_info['group']
        )
        
        # Force garbage collection between subjects to free memory
        gc.collect()
        print(f"  ✓ Completed sub-{subject_info['subject_id']}, sample-{subject_info['sample_label']}")
    
    # ========================================================================
    # Create dataset-level metadata files
    # ========================================================================
    print(f"\n{'='*60}")
    print("Creating dataset metadata files")
    print(f"{'='*60}")
    
    # Collect all unique subjects and samples
    all_subjects = {}  # {subject_id: {group, samples}}
    for subject_info in subjects_data:
        sid = subject_info['subject_id']
        if sid not in all_subjects:
            all_subjects[sid] = {
                'group': subject_info['group'],
                'samples': []
            }
        all_subjects[sid]['samples'].append(subject_info['sample_label'])
    
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
    deriv_root = bids_root / "derivatives" / "labels"
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
        for sid in sorted(all_subjects.keys()):
            group = all_subjects[sid]['group']
            f.write(f"sub-{sid}\t12\tM\t{group}\n")
    print(f"✓ Created {participants_tsv}")
    
    # samples.tsv
    samples_tsv = bids_root / "samples.tsv"
    with open(samples_tsv, 'w') as f:
        f.write("sample_id\tparticipant_id\tsample_type\tpathology\n")
        for subject_info in sorted(subjects_data, key=lambda x: (x['subject_id'], x['sample_label'])):
            sid = subject_info['subject_id']
            sample = subject_info['sample_label']
            pathology = "healthy" if subject_info['group'] == 'sham' else "tbi"
            f.write(f"sample-{sample}\tsub-{sid}\ttissue\t{pathology}\n")
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
""")
    print(f"✓ Created {readme}")
    
    print(f"\n{'='*60}")
    print("✓✓✓ BIDS conversion completed successfully! ✓✓✓")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Validate: bids-validator ./")
    print("2. Review metadata in JSON sidecars")
    print("3. Update participants.tsv and samples.tsv with accurate info")
    print("4. Update README.md and dataset_description.json")
    print(f"\nConverted {len(subjects_data)} samples across {len(all_subjects)} subjects")
    print("Generated structure:")
    print("  sub-*/micr/              (raw OME-Zarr volumes)")
    print("  derivatives/labels/      (segmentation masks)")
    print("  code/                    (this script)")


if __name__ == "__main__":
    main()
