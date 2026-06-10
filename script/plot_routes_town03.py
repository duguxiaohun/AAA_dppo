"""
Generate and save straight-through routes for the Town03 crossroads scenario.

Scenario: Ego approaches from south (y≈-150), drives NORTH (+y direction),
and goes STRAIGHT THROUGH the 4-way junction at (~-82, -138), exiting north
at (~y=-115).

Two northbound lanes are generated:
  Route 1 — inner lane (x≈-84.5)
  Route 2 — outer lane (x≈-88.0)

Usage (CARLA must already be running on the configured port):
    PYTHONPATH=$(pwd) python script/plot_routes_town03.py [--port 2000]

Output files:
    env/map/town03/town03_straight_route1.npy
    env/map/town03/town03_straight_route2.npy
    env/map/town03/routes_town03.png  (visualisation)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

sys.path.append('/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg')
sys.path.append('/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla/')
CARLA_PYAPI = '/home/codon/CARLA/CARLA_0.9.12/PythonAPI/carla'
sys.path.insert(0, CARLA_PYAPI)
sys.path.insert(0, os.path.join(CARLA_PYAPI, 'agents'))

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ── Coordinates (must match carla_env_town03.py constants) ───────────────────
SPAWN_X,  SPAWN_Y  = -84.5, -158.0   # inner lane spawn (~20 m south of junction)
SPAWN2_X           = -88.0            # outer lane spawn x
END_X,    END_Y    = -84.5, -112.0    # inner lane exit (~26 m north of junction)
END2_X,   END2_Y   = -88.0, -112.0    # outer lane exit
# ─────────────────────────────────────────────────────────────────────────────


def smooth_route(pts, spacing=0.5, spline_s=1.0):
    """Deduplicate, spline-fit, and uniformly resample a route."""
    mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts = pts[mask]
    if len(pts) < 4:
        return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=spline_s, k=min(3, len(pts)-1))
    u_dense = np.linspace(0, 1, 8000)
    xd, yd = splev(u_dense, tck)
    seg_len = np.sqrt(np.diff(xd) ** 2 + np.diff(yd) ** 2)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum_len[-1]
    n_pts = max(int(total / spacing) + 1, 4)
    u_new = np.interp(np.linspace(0, total, n_pts), cum_len, u_dense)
    xn, yn = splev(u_new, tck)
    return np.column_stack([xn, yn])


def make_parallel_route(inner_pts, x_offset, spacing=0.5):
    """
    Build the outer-lane route by shifting the inner-lane route by x_offset
    along the x axis.  For a straight north-south road the perpendicular
    direction is purely x, so a constant shift is exact.
    The route is then resampled at uniform `spacing`.
    """
    outer_pts = inner_pts.copy()
    outer_pts[:, 0] += x_offset

    seg_len = np.linalg.norm(np.diff(outer_pts, axis=0), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    n_pts   = max(int(total / spacing) + 1, 4)
    t_new   = np.linspace(0, total, n_pts)
    x_new   = np.interp(t_new, cum_len, outer_pts[:, 0])
    y_new   = np.interp(t_new, cum_len, outer_pts[:, 1])
    return np.column_stack([x_new, y_new])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client('localhost', args.port)
    client.set_timeout(15.0)
    world  = client.load_world('Town03')
    cmap   = world.get_map()
    grp    = GlobalRoutePlanner(cmap, 0.5)

    print(f"Connected to CARLA on port {args.port}, map: {cmap.name}")

    # ── Plan inner-lane route (Lane 1) via GlobalRoutePlanner ────────────────
    start1 = carla.Location(x=SPAWN_X,  y=SPAWN_Y,  z=0)
    end1   = carla.Location(x=END_X,    y=END_Y,    z=0)
    raw1   = np.array([[w.transform.location.x, w.transform.location.y]
                       for w, _ in grp.trace_route(start1, end1)])
    wp1    = smooth_route(raw1, spacing=0.5, spline_s=1.0)
    print(f"Route 1 (inner): raw={len(raw1)} pts → smoothed={len(wp1)} pts")

    # ── Build outer-lane route (Lane 2) as x-shifted parallel ────────────────
    x_offset = SPAWN2_X - SPAWN_X   # typically ≈ -3.5 m
    wp2 = make_parallel_route(wp1, x_offset=x_offset, spacing=0.5)
    print(f"Route 2 (outer): x_offset={x_offset:.1f} m → {len(wp2)} pts")

    # Verify that the outer route snaps to a valid CARLA road waypoint
    mid_idx = len(wp2) // 2
    mid_pt  = carla.Location(x=float(wp2[mid_idx, 0]),
                              y=float(wp2[mid_idx, 1]), z=0)
    snap    = cmap.get_waypoint(mid_pt, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if snap is not None:
        snap_x = snap.transform.location.x
        lateral_err = abs(snap_x - wp2[mid_idx, 0])
        print(f"  outer-lane road snap: x={snap_x:.2f}  "
              f"(lateral error = {lateral_err:.2f} m)")
        if lateral_err > 2.0:
            print("  WARNING: lateral error > 2 m — outer lane may not exist at "
                  "this x offset.  Check in CARLA and adjust SPAWN2_X / END2_X "
                  "in carla_env_town03.py, or set _SINGLE_LANE = True.")
    else:
        print("  WARNING: outer-lane mid-point did not snap to any road waypoint.")

    # ── Save ─────────────────────────────────────────────────────────────────
    map_dir = os.path.join(os.path.dirname(__file__), '..', 'env', 'map', 'town03')
    os.makedirs(map_dir, exist_ok=True)
    np.save(os.path.join(map_dir, 'town03_straight_route1.npy'), wp1)
    np.save(os.path.join(map_dir, 'town03_straight_route2.npy'), wp2)
    print(f"Saved routes to {os.path.abspath(map_dir)}/")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(raw1[:, 0], raw1[:, 1], 'g.-', ms=3, label='Route1 raw (GRP)')
    ax.plot(SPAWN_X, SPAWN_Y, 'y*', ms=14, zorder=5, label='START (inner)')
    ax.plot(END_X,   END_Y,   'g^', ms=10, zorder=5, label='END   (inner)')
    ax.set_title('GRP raw — inner lane')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.4)
    ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[1]
    ax.plot(wp1[:, 0], wp1[:, 1], 'g.-',  ms=3, label='Route1 smoothed (inner)')
    ax.plot(wp2[:, 0], wp2[:, 1], 'c.-',  ms=3, label='Route2 offset   (outer)')
    ax.plot(SPAWN_X,  SPAWN_Y,  'y*',  ms=14, zorder=5, label='START inner')
    ax.plot(SPAWN2_X, SPAWN_Y,  'y^',  ms=14, zorder=5, label='START outer')
    ax.plot(END_X,    END_Y,    'g^',  ms=10, zorder=5, label='END1  inner')
    ax.plot(END2_X,   END2_Y,   'c^',  ms=10, zorder=5, label='END2  outer')
    ax.set_title('Smoothed routes (saved)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.4)
    ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    png_path = os.path.join(map_dir, 'routes_town03.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")
    plt.show()


if __name__ == '__main__':
    main()
