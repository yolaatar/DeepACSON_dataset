#!/usr/bin/env python3
"""
Visualize BIDS microscopy data using matplotlib or napari.

Usage:
    # Quick 2D slice view with matplotlib
    python code/visualize_bids_data.py --mode matplotlib
    
    # Interactive 3D volume viewer with napari
    python code/visualize_bids_data.py --mode napari
"""

import argparse
import numpy as np
import zarr
from pathlib import Path


def load_zarr_volume(zarr_path, resolution_level=0):
    """Load a volume from OME-Zarr format."""
    z = zarr.open_group(str(zarr_path), mode='r')
    data = z[str(resolution_level)][:]
    return data


def visualize_matplotlib(raw_data, myelin_data, axon_data):
    """Create a matplotlib visualization with multiple views."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Get middle slices
    z_mid = raw_data.shape[0] // 2
    y_mid = raw_data.shape[1] // 2
    x_mid = raw_data.shape[2] // 2
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Raw data views
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(raw_data[z_mid, :, :], cmap='gray')
    ax1.set_title(f'Raw SEM - XY plane (Z={z_mid})')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(raw_data[:, y_mid, :], cmap='gray', aspect='auto')
    ax2.set_title(f'Raw SEM - XZ plane (Y={y_mid})')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(raw_data[:, :, x_mid], cmap='gray', aspect='auto')
    ax3.set_title(f'Raw SEM - YZ plane (X={x_mid})')
    ax3.axis('off')
    
    # Myelin segmentation (need to match dimensions)
    # Note: myelin_data has different axis order (Y, X, Z) vs raw (Z, Y, X)
    z_mid_m = myelin_data.shape[2] // 2
    y_mid_m = myelin_data.shape[0] // 2
    x_mid_m = myelin_data.shape[1] // 2
    
    ax4 = plt.subplot(3, 3, 4)
    myelin_slice = myelin_data[:, :, z_mid_m]
    ax4.imshow(myelin_slice, cmap='Reds', alpha=1.0)
    ax4.set_title(f'Myelin Mask - XY plane (Z={z_mid_m})')
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(myelin_data[y_mid_m, :, :], cmap='Reds', aspect='auto')
    ax5.set_title(f'Myelin Mask - XZ plane (Y={y_mid_m})')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(myelin_data[:, x_mid_m, :], cmap='Reds', aspect='auto')
    ax6.set_title(f'Myelin Mask - YZ plane (X={x_mid_m})')
    ax6.axis('off')
    
    # Axon instances
    ax7 = plt.subplot(3, 3, 7)
    axon_slice = axon_data[:, :, z_mid_m]
    im = ax7.imshow(axon_slice, cmap='tab20', interpolation='nearest')
    ax7.set_title(f'Axon Instances - XY plane (Z={z_mid_m})')
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.imshow(axon_data[y_mid_m, :, :], cmap='tab20', interpolation='nearest', aspect='auto')
    ax8.set_title(f'Axon Instances - XZ plane (Y={y_mid_m})')
    ax8.axis('off')
    
    # Overlay view
    ax9 = plt.subplot(3, 3, 9)
    # Match the slice from raw data
    raw_slice = raw_data[z_mid_m, :, :]
    # Transpose myelin to match
    myelin_overlay = myelin_data[:, :, z_mid_m].T
    
    ax9.imshow(raw_slice, cmap='gray')
    ax9.imshow(myelin_overlay, cmap='Reds', alpha=0.3)
    ax9.set_title('Overlay: Raw + Myelin')
    ax9.axis('off')
    
    plt.suptitle('DeepACSON BIDS Dataset Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print("\n✓ Matplotlib visualization created")
    print("  Close the window to continue...")
    plt.show()


def visualize_napari(raw_data, myelin_data, axon_data):
    """Launch napari for interactive 3D visualization."""
    import napari
    
    print("\n✓ Launching napari 3D viewer...")
    print("  Controls:")
    print("    - Left click + drag: rotate")
    print("    - Right click + drag: zoom")
    print("    - Scroll: move through slices")
    print("    - Toggle layers on/off in the left panel")
    
    # Create viewer
    viewer = napari.Viewer(title="DeepACSON BIDS Dataset")
    
    # Add raw data (Z, Y, X order)
    viewer.add_image(
        raw_data,
        name='Raw SEM',
        colormap='gray',
        scale=[0.008, 0.008, 0.008],  # 8nm isotropic
        blending='translucent'
    )
    
    # Add myelin mask (Y, X, Z order - need to transpose)
    # Transpose to (Z, Y, X)
    myelin_reordered = np.transpose(myelin_data, (2, 0, 1))
    viewer.add_labels(
        myelin_reordered,
        name='Myelin Mask',
        scale=[0.008, 0.008, 0.008],
        opacity=0.5,
        blending='translucent'
    )
    
    # Add axon instances (Y, X, Z order - transpose)
    axon_reordered = np.transpose(axon_data, (2, 0, 1))
    viewer.add_labels(
        axon_reordered,
        name='Axon Instances',
        scale=[0.008, 0.008, 0.008],
        opacity=0.7,
        blending='translucent'
    )
    
    # Set initial view
    viewer.dims.ndisplay = 3  # 3D view
    
    print("✓ Napari viewer launched")
    
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize BIDS microscopy dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 2D slice view
  python code/visualize_bids_data.py --mode matplotlib
  
  # Interactive 3D viewer (recommended)
  python code/visualize_bids_data.py --mode napari
  
  # Use lower resolution for faster loading
  python code/visualize_bids_data.py --mode napari --resolution 1
        """
    )
    parser.add_argument(
        '--mode',
        choices=['matplotlib', 'napari'],
        default='matplotlib',
        help='Visualization mode (default: matplotlib)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=0,
        help='OME-Zarr resolution level (0=highest, higher=lower res, default: 0)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("BIDS Dataset Visualization")
    print("="*70)
    
    # Define paths
    base_path = Path.cwd()
    raw_path = base_path / "sub-25/micr/sub-25_sample-contra_acq-HM_SEM.ome.zarr"
    myelin_path = base_path / "derivatives/labels/sub-25/micr/sub-25_sample-contra_acq-HM_desc-myelin_seg.ome.zarr"
    axon_path = base_path / "derivatives/labels/sub-25/micr/sub-25_sample-contra_acq-HM_desc-axonInstances_dseg.ome.zarr"
    
    # Check files exist
    for path in [raw_path, myelin_path, axon_path]:
        if not path.exists():
            print(f"✗ Error: {path} not found")
            return 1
    
    print(f"\n✓ Loading data (resolution level {args.resolution})...")
    
    # Load volumes
    raw_data = load_zarr_volume(raw_path, args.resolution)
    myelin_data = load_zarr_volume(myelin_path, args.resolution)
    axon_data = load_zarr_volume(axon_path, args.resolution)
    
    print(f"  Raw SEM: {raw_data.shape} ({raw_data.dtype})")
    print(f"  Myelin: {myelin_data.shape} ({myelin_data.dtype})")
    print(f"  Axons: {axon_data.shape} ({axon_data.dtype})")
    print(f"  Memory: ~{(raw_data.nbytes + myelin_data.nbytes + axon_data.nbytes) / 1024**2:.1f} MB")
    
    # Visualize
    if args.mode == 'matplotlib':
        visualize_matplotlib(raw_data, myelin_data, axon_data)
    else:
        visualize_napari(raw_data, myelin_data, axon_data)
    
    print("\n✓ Visualization complete")
    return 0


if __name__ == "__main__":
    exit(main())
