"""
Neuroglancer viewer for DeepACSON EM volumes + segmentation masks.

Opens a local neuroglancer server and prints a URL to open in your browser.
WebGL-accelerated 3D rendering — no GUI window needed.

Usage:
    python code/view_neuroglancer.py                          # HM_25_contra (default)
    python code/view_neuroglancer.py Sham_25_contra HM
    python code/view_neuroglancer.py TBI_24_contra LM

Available samples: Sham_25_contra, Sham_25_ipsi, Sham_49_contra, Sham_49_ipsi,
                   TBI_2_ipsi, TBI_24_contra
Available resolutions: HM (high-mag), LM (low-mag)

Controls (in browser):
    Scroll          → z-slice navigation
    Click + drag    → pan
    Right-click drag → rotate 3D
    Shift+drag      → zoom
    Press '3'       → toggle 3D view
"""

import sys
import os
import time
import numpy as np
import h5py
import scipy.io as sio
import neuroglancer


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "White matter EM")

# Physical voxel size in nanometers (from paper: ~50nm isotropic for HM, larger for LM)
VOXEL_SIZE_HM_NM = (50, 50, 50)
VOXEL_SIZE_LM_NM = (130, 130, 130)


def load_mat(path):
    try:
        mat = sio.loadmat(path)
        key = next(k for k in mat if not k.startswith("_"))
        return mat[key]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]
            return f[key][:]


def load_sample(sample_name, resolution):
    folder = os.path.join(DATA_ROOT, sample_name)
    prefix = f"{resolution}_{sample_name.split('_', 1)[1]}"
    h5_path = os.path.join(folder, f"{prefix}.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        raw = f["raw"][:]
    raw = np.squeeze(raw)
    print(f"Raw EM: {raw.shape} {raw.dtype}")

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
        if arr.shape == raw.shape:
            pass
        elif arr.T.shape == raw.shape:
            arr = arr.T
        else:
            print(f"  WARNING: {suffix} shape mismatch, skipping")
            continue
        masks[label] = arr
        print(f"Mask '{label}': {arr.shape} {arr.dtype}")

    return raw, masks


def make_viewer(sample_name, resolution):
    raw, masks = load_sample(sample_name, resolution)

    voxel_size = VOXEL_SIZE_HM_NM if resolution == "HM" else VOXEL_SIZE_LM_NM
    # neuroglancer expects (x, y, z) voxel size in nm
    res = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=voxel_size,
    )

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        # Raw EM
        s.layers["EM raw"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=raw,
                dimensions=res,
            ),
            shader="""
void main() {
  emitGrayscale(toNormalized(getDataValue()));
}
""",
        )

        # Masks
        layer_colors = {
            "myelin":  "#00ffff",
            "axons":   "#ff00ff",
            "nucleus": "#ffff00",
        }
        for name, arr in masks.items():
            if name == "axons":
                # Instance segmentation
                s.layers[name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=arr.astype(np.uint32),
                        dimensions=res,
                    ),
                )
            else:
                # Binary mask as image with color shader
                color = layer_colors.get(name, "#00ff00")
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                s.layers[name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(
                        data=arr.astype(np.float32),
                        dimensions=res,
                    ),
                    shader=f"""
void main() {{
  float v = toNormalized(getDataValue());
  emitRGBA(vec4({r:.3f}, {g:.3f}, {b:.3f}, v * 0.5));
}}
""",
                )

        # Center the view
        z, y, x = [s // 2 for s in raw.shape]
        s.position = [x, y, z]
        s.cross_section_scale = 1e-6

    return viewer


def main():
    sample = sys.argv[1] if len(sys.argv) > 1 else "Sham_25_contra"
    resolution = sys.argv[2].upper() if len(sys.argv) > 2 else "HM"

    print(f"\nLoading {resolution}_{sample}...")

    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = make_viewer(sample, resolution)

    url = viewer.get_viewer_url()
    print(f"\nNeuroglancer ready. Open this URL in Chrome/Firefox:\n\n  {url}\n")
    print("Press Ctrl+C to stop the server.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
