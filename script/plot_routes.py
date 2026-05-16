"""
Plan, clean, smooth, and save routes for Town05 left-turn scenario.

Usage:
    PYTHONPATH=$(pwd) python script/plot_routes.py
CARLA must already be running on port 2000.
"""
import sys, os
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

# ── coordinates (must match carla_env_town05.py constants) ───────────────────
SPAWN_X, SPAWN_Y = -47.01,  30.0
END1_X,  END1_Y  = -78.0,   -0.7
END2_X,  END2_Y  = -78.0,   -4.2
# ─────────────────────────────────────────────────────────────────────────────


def smooth_route(pts, spacing=0.5, spline_s=2.0):
    """Deduplicate, spline-fit and uniformly resample a GRP route."""
    # deduplicate
    mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts = pts[mask]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=spline_s, k=3)
    u_dense = np.linspace(0, 1, 8000)
    xd, yd  = splev(u_dense, tck)
    seg_len = np.sqrt(np.diff(xd)**2 + np.diff(yd)**2)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    n_pts   = max(int(total / spacing) + 1, 4)
    u_new   = np.interp(np.linspace(0, total, n_pts), cum_len, u_dense)
    xn, yn  = splev(u_new, tck)
    return np.column_stack([xn, yn])


def make_outer_lane(inner_pts, lane_width, ramp_start_y=10.0, spacing=0.5):
    """
    Synthesise the outer-lane route from the clean inner route.

    Strategy: offset each point of inner_pts by `lane_width` along the
    right-perpendicular of the local tangent.  The offset is linearly ramped
    from 0 (at ramp_start_y) to full lane_width (at the last point's y),
    so the two routes share the same approach and diverge smoothly into the
    turn.  This completely avoids the GRP lane-switch artefact.
    """
    pts = inner_pts.copy()

    # tangent via central differences
    tangents = np.gradient(pts, axis=0)
    norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-10)

    # right perpendicular (CW 90°): (dy, -dx) → nope; correct: [-t_y, t_x]
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # linear ramp from 0 → 1 as y goes from ramp_start_y → y_end
    y_vals = pts[:, 1]
    y_end  = y_vals[-1]
    ramp   = np.clip((ramp_start_y - y_vals) / (ramp_start_y - y_end), 0.0, 1.0)

    outer_pts = pts + ramp[:, np.newaxis] * lane_width * perps

    # resample at uniform spacing
    seg_len = np.linalg.norm(np.diff(outer_pts, axis=0), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    n_pts   = max(int(total / spacing) + 1, 4)
    t_new   = np.linspace(0, total, n_pts)
    x_new   = np.interp(t_new, cum_len, outer_pts[:, 0])
    y_new   = np.interp(t_new, cum_len, outer_pts[:, 1])
    return np.column_stack([x_new, y_new])


# ── connect & plan ────────────────────────────────────────────────────────────
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world  = client.load_world('Town05')
cmap   = world.get_map()
grp    = GlobalRoutePlanner(cmap, 0.5)

raw1 = np.array([[w.transform.location.x, w.transform.location.y]
                 for w, _ in grp.trace_route(carla.Location(x=SPAWN_X, y=SPAWN_Y, z=0),
                                              carla.Location(x=END1_X,  y=END1_Y,  z=0))])

# Route1: clean & smooth via GRP
wp1 = smooth_route(raw1)

# Route2: synthesised from Route1 by gradual perpendicular offset.
# GRP's Route2 has a hard lane-switch kink at the intersection;
# offset from Route1 starting at y=10 gives a smooth outer-lane arc.
lane_width = abs(END2_Y - END1_Y)   # 3.5 m
wp2 = make_outer_lane(wp1, lane_width, ramp_start_y=10.0)

print(f"Route1: raw={len(raw1)}  smoothed={len(wp1)} pts")
print(f"Route2: offset from Route1, lane_width={lane_width:.1f} m, {len(wp2)} pts")

# ── save ──────────────────────────────────────────────────────────────────────
map_dir = os.path.join(os.path.dirname(__file__), '..', 'env', 'map')
os.makedirs(map_dir, exist_ok=True)
np.save(os.path.join(map_dir, 'route1.npy'), wp1)
np.save(os.path.join(map_dir, 'route2.npy'), wp2)

# ── plot ──────────────────────────────────────────────────────────────────────
ref_wp  = np.load(os.path.join(map_dir, 'wp.npy'))
ref_wp2 = np.load(os.path.join(map_dir, 'wp2.npy'))

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

ax = axes[0]
ax.plot(ref_wp[:,0],  ref_wp[:,1],  'g.-', ms=3, label='wp (ref inner)')
ax.plot(ref_wp2[:,0], ref_wp2[:,1], 'b.-', ms=3, label='wp2 (ref outer)')
ax.set_title('Reference curves'); ax.set_aspect('equal')
ax.grid(True, alpha=0.4); ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[1]
ax.plot(raw1[:,0], raw1[:,1], 'g.-', ms=3, label='Route1 raw (GRP)')
ax.set_title('GRP raw'); ax.set_aspect('equal')
ax.grid(True, alpha=0.4); ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[2]
ax.plot(wp1[:,0], wp1[:,1], 'g.-', ms=3, label='Route1 smoothed')
ax.plot(wp2[:,0], wp2[:,1], 'c.-', ms=3, label='Route2 smoothed')
ax.plot(SPAWN_X, SPAWN_Y, 'y*', ms=16, zorder=5, label='START')
ax.plot(END1_X,  END1_Y,  'g^', ms=12, zorder=5, label='END1')
ax.plot(END2_X,  END2_Y,  'c^', ms=12, zorder=5, label='END2')
ax.set_title('Smoothed (saved)'); ax.set_aspect('equal')
ax.grid(True, alpha=0.4); ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
png_path = os.path.join(map_dir, 'routes.png')
plt.savefig(png_path, dpi=150)
print(f"Saved: {png_path}")
plt.show()
