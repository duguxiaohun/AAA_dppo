import argparse
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STYLE_BY_NAME = {
    "ego": ("#D62728", "#FF6347", "#FFD700", 3.0, 10),
    "v_2": ("#1F77B4", "#4FC3F7", "#E1F5FE", 2.0, 5),
    "v_3": ("#9467BD", "#E0B0FF", "#F3E5F5", 2.0, 5),
    "v_7": ("#2CA02C", "#90EE90", "#F1F8E9", 2.0, 5),
}
DEFAULT_STYLE = ("#7F7F7F", "#D3D3D3", "#FFFFFF", 1.5, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot vehicle trajectories from a collected CARLA HDF5 experiment."
    )
    parser.add_argument(
        "input",
        help="Experiment folder, town folder, data root, or a concrete .hdf5 file.",
    )
    parser.add_argument(
        "--episode",
        "--epoch",
        dest="episode",
        type=int,
        required=True,
        help="Episode id in the HDF5 file, for example 19 for episodes/00019.",
    )
    parser.add_argument(
        "--vehicles",
        nargs="+",
        default=["ego"],
        help="Vehicle ids to plot, for example: ego 2 3 v_7. Use 'all' for all ego/v_* keys.",
    )
    parser.add_argument("--step", type=int, default=10, help="Marker interval.")
    parser.add_argument(
        "--origin",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="Subtract this origin from x/y before plotting.",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show-axis", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to <experiment folder>/plots.",
    )
    return parser.parse_args()


def find_hdf5(input_path):
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path

    candidates = sorted(path.rglob("offline_dataset_*.hdf5"))
    if not candidates:
        candidates = sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No HDF5 file found under {path}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def experiment_dir_for(h5_path):
    return h5_path.parent


def normalize_vehicle_name(name):
    text = str(name)
    if text == "ego" or text.startswith("v_"):
        return text
    if text.isdigit():
        return f"v_{text}"
    return text


def read_xy(group, name, origin):
    ds = group[name]
    if ds.dtype.kind == "O":
        mat = np.vstack([
            np.asarray(ds[i], dtype=np.float32).reshape(-1)
            for i in range(len(ds))
        ])
    else:
        mat = np.asarray(ds[...], dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[:, None]

    if mat.shape[1] < 2:
        raise ValueError(f"Dataset {name} does not contain x/y columns")

    x = mat[:, 0] - origin[0]
    y = mat[:, 1] - origin[1]
    return x, y


def plot_gradient_traj(ax, x, y, name, step):
    if len(x) < 2:
        return

    base_color, glow_color, core_color, lw, zorder = STYLE_BY_NAME.get(name, DEFAULT_STYLE)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    line_alphas = np.linspace(0.1, 1.0, len(segments))
    base_rgb = to_rgba(base_color)[:3]
    colors = np.zeros((len(segments), 4))
    colors[:, :3] = base_rgb
    colors[:, 3] = line_alphas
    ax.add_collection(LineCollection(segments, colors=colors, linewidth=lw, zorder=zorder))

    idx_points = np.arange(0, len(x), max(1, step))
    progress = idx_points / max(1, len(x) - 1)
    dot_alphas = 0.2 + 0.8 * progress

    core_rgba = np.zeros((len(idx_points), 4))
    core_rgba[:, :3] = to_rgba(core_color)[:3]
    core_rgba[:, 3] = dot_alphas

    glow_rgba = np.zeros((len(idx_points), 4))
    glow_rgba[:, :3] = to_rgba(glow_color)[:3]
    glow_rgba[:, 3] = dot_alphas * 0.5

    ax.scatter(x[idx_points], y[idx_points], s=120, c=glow_rgba, edgecolors="none", zorder=zorder + 1)
    ax.scatter(
        x[idx_points],
        y[idx_points],
        s=25,
        c=core_rgba,
        edgecolors="white",
        linewidth=0.5,
        zorder=zorder + 2,
    )
    ax.scatter([x[0]], [y[0]], marker="o", s=150, color=base_color, alpha=0.3,
               edgecolors="white", linewidth=1.5, zorder=zorder + 3)
    ax.scatter([x[-1]], [y[-1]], marker="x", s=150, color=base_color, alpha=1.0,
               linewidth=3.0, zorder=zorder + 3)


def auto_limits(values, explicit):
    if explicit is not None:
        return explicit
    arr = np.concatenate(values)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    pad = max((hi - lo) * 0.08, 1.0)
    return lo - pad, hi + pad


def main():
    args = parse_args()
    h5_path = find_hdf5(args.input)
    ep_name = f"{args.episode:05d}"

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_dir_for(h5_path) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        if "episodes" not in h5:
            raise KeyError(f"{h5_path} does not contain an 'episodes' group")
        if ep_name not in h5["episodes"]:
            raise KeyError(f"Episode {ep_name} not found in {h5_path}")

        group = h5["episodes"][ep_name]
        if args.vehicles == ["all"]:
            vehicles = sorted(k for k in group.keys() if k == "ego" or k.startswith("v_"))
        else:
            vehicles = [normalize_vehicle_name(v) for v in args.vehicles]

        trajectories = {}
        for vehicle in vehicles:
            if vehicle not in group:
                print(f"[WARN] vehicle {vehicle} not found in episode {ep_name}")
                continue
            trajectories[vehicle] = read_xy(group, vehicle, args.origin)

    if not trajectories:
        raise RuntimeError("No matching vehicle trajectories were found.")

    xlim = auto_limits([xy[0] for xy in trajectories.values()], args.xlim)
    ylim = auto_limits([xy[1] for xy in trajectories.values()], args.ylim)
    ratio = max((xlim[1] - xlim[0]) / max(ylim[1] - ylim[0], 1e-6), 1e-6)

    fig = plt.figure(figsize=(6 * ratio, 6), dpi=args.dpi)
    ax = plt.gca()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for vehicle, (x, y) in trajectories.items():
        plot_gradient_traj(ax, x, y, vehicle, args.step)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if args.show_axis:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{h5_path.name} / episode {ep_name}")
    else:
        ax.axis("off")

    plt.tight_layout()
    vehicle_tag = "_".join(trajectories.keys()).replace("/", "_")
    save_path = output_dir / f"traj_episode_{ep_name}_{vehicle_tag}.png"
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[INFO] saved trajectory plot: {save_path}")


if __name__ == "__main__":
    main()
