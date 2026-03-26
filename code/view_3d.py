"""
3D mesh viewer for DeepACSON segmentation masks using PyVista.

Extracts surfaces from segmentation masks via marching cubes, then renders
them as interactive 3D meshes. Much faster than volume rendering for inspecting
individual axon shapes.

Usage:
    python code/view_3d.py                              # HM_25_contra, all masks
    python code/view_3d.py Sham_25_contra HM            # pick sample
    python code/view_3d.py TBI_24_contra LM axons       # specific mask only
    python code/view_3d.py Sham_25_contra HM myelin axons nucleus

Available masks: myelin, axons, nucleus
"""

import sys
import os
import numpy as np
import h5py
import scipy.io as sio
from skimage.measure import marching_cubes
import pyvista as pv


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "White matter EM")

# Physical voxel spacing in nm → used to scale the mesh correctly
VOXEL_SPACING = {
    "HM": (50.0, 50.0, 50.0),   # z, y, x in nm
    "LM": (130.0, 130.0, 130.0),
}

MASK_COLORS = {
    "myelin":  "#00d4ff",   # cyan
    "axons":   "#ff4dc4",   # magenta
    "nucleus": "#ffe033",   # yellow
}

MASK_OPACITY = {
    "myelin":  0.35,
    "axons":   0.85,
    "nucleus": 0.70,
}


def load_mat(path):
    try:
        mat = sio.loadmat(path)
        key = next(k for k in mat if not k.startswith("_"))
        return mat[key]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]
            return f[key][:]


def load_masks(sample_name, resolution, wanted):
    folder = os.path.join(DATA_ROOT, sample_name)
    prefix = f"{resolution}_{sample_name.split('_', 1)[1]}"

    # Need raw shape to align masks
    h5_path = os.path.join(folder, f"{prefix}.h5")
    with h5py.File(h5_path, "r") as f:
        raw_shape = np.squeeze(f["raw"][:]).shape  # (z, y, x)

    suffix_map = {
        "myelin":  "myelin",
        "axons":   "myelinated_axons",
        "nucleus": "nucleus",
    }

    masks = {}
    for name in wanted:
        mat_path = os.path.join(folder, f"{prefix}_{suffix_map[name]}.mat")
        if not os.path.exists(mat_path):
            print(f"  Skipping '{name}': file not found")
            continue
        arr = load_mat(mat_path)
        if arr.T.shape == raw_shape:
            arr = arr.T
        if arr.shape != raw_shape:
            print(f"  Skipping '{name}': shape mismatch {arr.shape} vs {raw_shape}")
            continue
        masks[name] = arr
        print(f"  Loaded '{name}': {arr.shape} {arr.dtype}")

    return masks, raw_shape


def mask_to_mesh(arr, spacing, step_size=2):
    """Run marching cubes on a binary or instance mask, return a PyVista mesh."""
    binary = (arr > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    verts, faces, normals, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=spacing,   # physical scale (z, y, x) in nm
        step_size=step_size,
        allow_degenerate=False,
    )
    # PyVista PolyData from marching cubes output
    face_arr = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts, face_arr)
    mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)  # light smoothing
    return mesh


def axons_to_meshes(arr, spacing, step_size=2, max_axons=200):
    """
    For instance-labelled axon masks: build one mesh per axon ID.
    Returns a single merged mesh with per-cell scalars for coloring.
    """
    ids = np.unique(arr)
    ids = ids[ids > 0]
    if len(ids) > max_axons:
        print(f"  Limiting to {max_axons} largest axons (out of {len(ids)} total)")
        sizes = [(arr == i).sum() for i in ids]
        ids = ids[np.argsort(sizes)[::-1][:max_axons]]

    meshes = []
    for axon_id in ids:
        binary = (arr == axon_id).astype(np.uint8)
        if binary.sum() < 50:  # skip tiny fragments
            continue
        try:
            verts, faces, _, _ = marching_cubes(
                binary, level=0.5, spacing=spacing, step_size=step_size,
                allow_degenerate=False,
            )
        except (ValueError, RuntimeError):
            continue
        face_arr = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        m = pv.PolyData(verts, face_arr)
        m["axon_id"] = np.full(m.n_points, float(axon_id))
        meshes.append(m)

    if not meshes:
        return None
    combined = meshes[0]
    for m in meshes[1:]:
        combined = combined.merge(m)
    return combined


def main():
    sample    = sys.argv[1] if len(sys.argv) > 1 else "Sham_25_contra"
    resolution = sys.argv[2].upper() if len(sys.argv) > 2 else "HM"
    wanted    = [a.lower() for a in sys.argv[3:]] if len(sys.argv) > 3 else ["myelin", "axons"]

    spacing = VOXEL_SPACING[resolution]  # (z, y, x) nm

    print(f"\nLoading {resolution}_{sample} — masks: {wanted}")
    masks, raw_shape = load_masks(sample, resolution, wanted)

    if not masks:
        print("No masks loaded. Exiting.")
        sys.exit(1)

    print("\nExtracting surfaces (marching cubes)...")
    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("#1a1a2e")
    plotter.add_text(
        f"DeepACSON 3D — {resolution}_{sample}",
        position="upper_left", font_size=12, color="white",
    )

    for name, arr in masks.items():
        print(f"  {name}...", end=" ", flush=True)
        if name == "axons":
            mesh = axons_to_meshes(arr, spacing)
            if mesh is not None:
                plotter.add_mesh(
                    mesh,
                    scalars="axon_id",
                    cmap="tab20",
                    show_scalar_bar=False,
                    opacity=MASK_OPACITY[name],
                    smooth_shading=True,
                    label=name,
                )
                print(f"OK — {mesh.n_cells} faces")
            else:
                print("empty")
        else:
            mesh = mask_to_mesh(arr, spacing)
            if mesh is not None:
                plotter.add_mesh(
                    mesh,
                    color=MASK_COLORS[name],
                    opacity=MASK_OPACITY[name],
                    smooth_shading=True,
                    label=name,
                )
                print(f"OK — {mesh.n_cells} faces")
            else:
                print("empty")

    plotter.add_legend(bcolor="#00000088", border=True, size=(0.15, 0.15))
    plotter.add_axes()

    # Scale bar annotation (1 µm)
    z, y, x = [s * sp for s, sp in zip(raw_shape, spacing)]
    print(f"\nVolume physical size: {x/1000:.1f} × {y/1000:.1f} × {z/1000:.1f} µm")

    print("\nControls:")
    print("  Left-drag  → rotate   |  Right-drag → zoom")
    print("  Middle-drag → pan     |  'r' → reset camera")
    print("  'p'        → pick point info")

    plotter.show()


if __name__ == "__main__":
    main()
