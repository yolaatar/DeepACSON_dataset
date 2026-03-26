"""
Napari viewer for DeepACSON EM volumes + segmentation masks.

Usage:
    python code/view_napari.py                          # opens HM_25_contra (default)
    python code/view_napari.py Sham_25_contra HM        # pick sample + resolution
    python code/view_napari.py TBI_24_contra LM

Available samples: Sham_25_contra, Sham_25_ipsi, Sham_49_contra, Sham_49_ipsi,
                   TBI_2_ipsi, TBI_24_contra
Available resolutions: HM (high-mag), LM (low-mag)
"""

import sys
import os
import numpy as np
import h5py
import scipy.io as sio
import napari


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "White matter EM")


def load_mat(path):
    """Load a .mat file regardless of version (v5 or v7.3/HDF5)."""
    try:
        mat = sio.loadmat(path)
        key = next(k for k in mat if not k.startswith("_"))
        return mat[key]
    except NotImplementedError:
        # v7.3 HDF5-based .mat
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]
            return f[key][:]


def load_sample(sample_name: str, resolution: str):
    """
    Returns:
        raw   : np.ndarray (z, y, x) uint8
        masks : dict[str -> np.ndarray (z, y, x)]
    """
    folder = os.path.join(DATA_ROOT, sample_name)
    prefix = f"{resolution}_{sample_name.split('_', 1)[1]}"  # e.g. HM_25_contra
    h5_path = os.path.join(folder, f"{prefix}.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Not found: {h5_path}\nCheck sample name and resolution.")

    # --- Raw EM ---
    with h5py.File(h5_path, "r") as f:
        raw = f["raw"][:]
    raw = np.squeeze(raw)  # remove channel dim if present: (z,1,y,x) -> (z,y,x)
    print(f"Raw EM: {raw.shape} {raw.dtype}")

    # --- Masks ---
    masks = {}
    for suffix, label in [
        ("myelin",           "myelin"),
        ("myelinated_axons", "axons"),
        ("nucleus",          "nucleus"),
    ]:
        mat_path = os.path.join(folder, f"{prefix}_{suffix}.mat")
        if not os.path.exists(mat_path):
            continue
        arr = load_mat(mat_path)
        # MATLAB v5 uses (x,y,z) → transpose to (z,y,x) to match raw
        # HDF5-backed v7.3 is already read as (z,y,x) by h5py
        if arr.shape == raw.shape:
            pass  # already aligned
        elif arr.T.shape == raw.shape:
            arr = arr.T
        else:
            print(f"  WARNING: {suffix} shape {arr.shape} doesn't match raw {raw.shape}, skipping")
            continue
        masks[label] = arr
        print(f"Mask '{label}': {arr.shape} {arr.dtype}")

    return raw, masks


def main():
    sample = sys.argv[1] if len(sys.argv) > 1 else "Sham_25_contra"
    resolution = sys.argv[2].upper() if len(sys.argv) > 2 else "HM"

    print(f"\nLoading {resolution}_{sample}...")
    raw, masks = load_sample(sample, resolution)

    viewer = napari.Viewer(title=f"DeepACSON — {resolution}_{sample}")

    # Raw EM in grayscale
    viewer.add_image(raw, name="EM raw", colormap="gray", contrast_limits=[raw.min(), raw.max()])

    # Segmentation masks as label layers
    label_colormaps = {
        "myelin":  "cyan",
        "axons":   "magenta",
        "nucleus": "yellow",
    }
    for name, arr in masks.items():
        if name == "axons":
            # Instance labels (uint16): use as Labels layer for per-axon coloring
            viewer.add_labels(arr.astype(np.int32), name=name, opacity=0.5)
        else:
            # Binary masks: show as image overlay
            viewer.add_image(
                arr.astype(np.float32),
                name=name,
                colormap=label_colormaps.get(name, "green"),
                blending="additive",
                opacity=0.4,
            )

    print("\nNapari viewer open.")
    print("  - Use scroll wheel to move through z-slices")
    print("  - Toggle layers with the eye icon in the layer list")
    print("  - Press '3' in the viewer for 3D rendering (may be slow on large volumes)")
    napari.run()


if __name__ == "__main__":
    main()
