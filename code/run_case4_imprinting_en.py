"""
English-label counterpart of run_case4_imprinting.py -- Case 4: effect of
the 12 pentagonal singularities of the icosahedral mesh ("grid imprinting")
on the accuracy of the discrete operator. See the original script's
docstring for the full diagnostic rationale.
"""
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV, real_spherical_harmonic

plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.dpi": 300})

l, m = 2, 1
eig = l * (l + 1)

levels = [2, 3, 4, 5, 6, 7]
err_pent_max, err_hex_median, err_hex_max, hs = [], [], [], []

for level in levels:
    solver = HeatSphereFV(level=level)
    verts = solver.mesh.vertices
    Y = real_spherical_harmonic(l, m, verts)

    Lu = solver.L @ Y
    approx_lap = Lu / solver.area
    exact_lap = eig * Y
    e = approx_lap - exact_lap

    e_pent = np.abs(e[:12])
    e_hex = np.abs(e[12:])

    err_pent_max.append(e_pent.max())
    err_hex_median.append(np.median(e_hex))
    err_hex_max.append(e_hex.max())
    hs.append(np.sqrt(4 * np.pi / solver.V))
    print(f"level {level}: V={solver.V:7d}  max_trunc_err_pentagon={e_pent.max():.4e}  "
          f"median_trunc_err_hexagon={np.median(e_hex):.4e}  "
          f"max_trunc_err_hexagon={e_hex.max():.4e}")

order_pent = [np.log2(err_pent_max[i] / err_pent_max[i + 1]) for i in range(len(levels) - 1)]
order_hex = [np.log2(err_hex_median[i] / err_hex_median[i + 1]) for i in range(len(levels) - 1)]
print("estimated local order (max over the 12 pentagons):", [f"{o:.2f}" for o in order_pent])
print("estimated local order (median over hexagons):", [f"{o:.2f}" for o in order_hex])

fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.loglog(hs, err_pent_max, "o-", color="#c05621", label="max. over the 12 pentagons")
ax.loglog(hs, err_hex_median, "s-", color="#2b6cb0", label="median over hexagons")
ref = np.array(err_hex_median[0]) * (np.array(hs) / hs[0]) ** 2
ax.loglog(hs, ref, "--", color="gray", label=r"$O(h^2)$")
ax.set_xlabel(r"$h \sim \sqrt{4\pi/V_n}$")
ax.set_ylabel(r"local truncation error $|e_i|$")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("../figures_en/fig_caso4_imprinting_nivel.pdf", bbox_inches="tight")
print("saved ../figures_en/fig_caso4_imprinting_nivel.pdf")

import os
os.makedirs("../notes/data_en", exist_ok=True)
with open("../notes/data_en/caso4_imprinting_nivel.csv", "w") as fh:
    fh.write("level,V,max_trunc_err_pentagon,median_trunc_err_hexagon,max_trunc_err_hexagon\n")
    for lv, V, a, b, c in zip(levels, [round(4*np.pi/h**2) for h in hs], err_pent_max, err_hex_median, err_hex_max):
        fh.write(f"{lv},{V},{a:.6e},{b:.6e},{c:.6e}\n")
print("saved ../notes/data_en/caso4_imprinting_nivel.csv")

# --- spatial error map (fixed level), to visualize where the error concentrates ---
level_map = 4
solver = HeatSphereFV(level=level_map)
verts = solver.mesh.vertices
faces = solver.mesh.faces
Y = real_spherical_harmonic(l, m, verts)
e = np.abs(solver.L @ Y / solver.area - eig * Y)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
pent_verts = verts[:12] * 1.02


def _camera_direction(elev, azim):
    er, ar = np.radians(elev), np.radians(azim)
    return np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])


views = [(20, 35), (-20, 35 + 180)]

fig = plt.figure(figsize=(9.5, 5))
for k, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.computed_zorder = False

    face_vals = e[faces].mean(axis=1)
    norm = plt.Normalize(0, np.percentile(e, 99))
    colors = plt.colormaps["magma"](norm(face_vals))
    tris = verts[faces]
    poly = Poly3DCollection(tris, facecolor=colors, edgecolor="none")
    poly.set_zorder(1)
    ax.add_collection3d(poly)

    cam_dir = _camera_direction(elev, azim)
    visible = (pent_verts / np.linalg.norm(pent_verts, axis=1, keepdims=True)) @ cam_dir > 0.0
    sc = ax.scatter(*pent_verts[visible].T, color="#00e5ff", s=110, edgecolor="white",
                     linewidth=1.2, depthshade=False, label="pentagonal vertices")
    sc.set_zorder(10)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("front" if k == 0 else "back", fontsize=11, y=0.95)
    if k == 0:
        ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("../figures_en/fig_caso4_imprinting_mapa.pdf", bbox_inches="tight")
print("saved ../figures_en/fig_caso4_imprinting_mapa.pdf")
