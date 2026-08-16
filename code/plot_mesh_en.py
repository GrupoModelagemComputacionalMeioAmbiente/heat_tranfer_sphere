"""Generation of icosahedral mesh and dual mesh figures for the article (English labels)."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from icosphere import build_icosphere, build_dual_mesh

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
})

PRIMAL_EDGE_COLOR = "#2b6cb0"
DUAL_EDGE_COLOR = "#c05621"
PRIMAL_FACE_COLOR = "#bee3f8"
PENT_COLOR = "#fbd38d"
HEX_COLOR = "#c6f6d5"


def _setup_3d_axis(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def plot_primal_mesh(level, ax=None, show_faces=True, edge_lw=0.6):
    mesh = build_icosphere(level)
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

    if show_faces:
        tris = mesh.vertices[mesh.faces]
        poly = Poly3DCollection(tris, facecolor=PRIMAL_FACE_COLOR,
                                 edgecolor=PRIMAL_EDGE_COLOR,
                                 linewidths=edge_lw, alpha=0.95)
        ax.add_collection3d(poly)
    else:
        segs = mesh.vertices[mesh.edges]
        lc = Line3DCollection(segs, colors=PRIMAL_EDGE_COLOR, linewidths=edge_lw)
        ax.add_collection3d(lc)

    _setup_3d_axis(ax)
    if own_fig:
        return fig, ax, mesh
    return mesh


def plot_dual_mesh(level, ax=None, edge_lw=0.9, color_by_type=True):
    mesh = build_icosphere(level)
    dm = build_dual_mesh(mesh)
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

    polys = []
    colors = []
    for i, cf in enumerate(dm.cell_faces):
        poly = dm.dual_vertices[cf]
        polys.append(poly)
        colors.append(PENT_COLOR if len(cf) == 5 else HEX_COLOR)

    coll = Poly3DCollection(polys, facecolor=colors if color_by_type else DUAL_EDGE_COLOR,
                             edgecolor=DUAL_EDGE_COLOR, linewidths=edge_lw, alpha=0.95)
    ax.add_collection3d(coll)
    _setup_3d_axis(ax)
    if own_fig:
        return fig, ax, mesh, dm
    return mesh, dm


def plot_primal_and_dual_overlay(level, ax=None):
    mesh = build_icosphere(level)
    dm = build_dual_mesh(mesh)
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection="3d")

    segs_p = mesh.vertices[mesh.edges]
    lc_p = Line3DCollection(segs_p, colors=PRIMAL_EDGE_COLOR, linewidths=0.5, alpha=0.6)
    ax.add_collection3d(lc_p)

    polys = [dm.dual_vertices[cf] for cf in dm.cell_faces]
    coll = Poly3DCollection(polys, facecolor=(0, 0, 0, 0), edgecolor=DUAL_EDGE_COLOR, linewidths=0.9)
    ax.add_collection3d(coll)

    _setup_3d_axis(ax)
    if own_fig:
        return fig, ax, mesh, dm
    return mesh, dm


def fig_refinement_sequence(levels=(0, 1, 2, 3, 4, 5), outfile="../figures_en/fig_malha_refinamento.pdf"):
    lw_by_level = {0: 0.7, 1: 0.6, 2: 0.6, 3: 0.5, 4: 0.3, 5: 0.15, 6: 0.08}
    fig = plt.figure(figsize=(4 * len(levels), 4.3))
    for k, n in enumerate(levels):
        ax = fig.add_subplot(1, len(levels), k + 1, projection="3d")
        plot_primal_mesh(n, ax=ax, edge_lw=lw_by_level.get(n, 0.6))
        ax.set_title(f"level $n={n}$", fontsize=12, y=0.95)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("saved", outfile)


def fig_primal_vs_dual(level=2, outfile="../figures_en/fig_malha_primal_dual.pdf"):
    fig = plt.figure(figsize=(12, 4.3))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plot_primal_mesh(level, ax=ax1)
    ax1.set_title("Icosahedral mesh (primal)", fontsize=11, y=0.95)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_dual_mesh(level, ax=ax2)
    ax2.set_title("Dual mesh (pentagons/hexagons)", fontsize=11, y=0.95)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_primal_and_dual_overlay(level, ax=ax3)
    ax3.set_title("Primal/dual overlay", fontsize=11, y=0.95)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("saved", outfile)


def fig_dual_cell_detail(level=2, outfile="../figures_en/fig_celula_dual_detalhe.pdf"):
    """Zoom on a hexagonal and a pentagonal dual cell, with labels
    consistent with the thesis notation (u0, u_j, d_j, h_j)."""
    mesh = build_icosphere(level)
    dm = build_dual_mesh(mesh)

    pent_i = 0
    hex_i = next(i for i in range(mesh.n_vertices) if len(dm.cell_faces[i]) == 6)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, i, label in ((axes[0], pent_i, "pentagonal (m=5)"),
                          (axes[1], hex_i, "hexagonal (m=6)")):
        center = mesh.vertices[i]
        normal = center / np.linalg.norm(center)
        ref = dm.dual_vertices[dm.cell_faces[i][0]] - center
        ref = ref - np.dot(ref, normal) * normal
        t1 = ref / np.linalg.norm(ref)
        t2 = np.cross(normal, t1)

        def to2d(p):
            v = p - center
            return np.dot(v, t1), np.dot(v, t2)

        poly = [to2d(dm.dual_vertices[f]) for f in dm.cell_faces[i]]
        poly_arr = np.array(poly + [poly[0]])
        ax.plot(poly_arr[:, 0], poly_arr[:, 1], "-", color=DUAL_EDGE_COLOR, lw=1.4)
        ax.fill(poly_arr[:, 0], poly_arr[:, 1], color=PENT_COLOR if label.startswith("pent") else HEX_COLOR, alpha=0.5)

        ax.plot(0, 0, "o", color="black", ms=5)
        ax.annotate(r"$u_0$", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=11)

        m = len(dm.cell_faces[i])
        for k in range(m):
            j = dm.neighbors[i][k]
            xj, yj = to2d(mesh.vertices[j])
            xm, ym = xj / 2, yj / 2
            ax.plot([0, xj], [0, yj], "--", color="gray", lw=0.6)
            ax.plot(xm, ym, "s", color="#c53030", ms=4)
            if k == 0:
                ax.annotate(r"$u_j,\ d_j$", (xm, ym), textcoords="offset points",
                            xytext=(4, 4), fontsize=9, color="#c53030")

        bk = poly[0]
        bk1 = poly[1]
        ax.annotate(r"$h_j$", ((bk[0] + bk1[0]) / 2, (bk[1] + bk1[1]) / 2),
                    textcoords="offset points", xytext=(0, -12), fontsize=9, color=DUAL_EDGE_COLOR)

        ax.set_aspect("equal")
        kind, paren = label.split(" ", 1)
        ax.set_title(f"{kind.capitalize()} cell {paren}", fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("saved", outfile)


if __name__ == "__main__":
    import os
    os.makedirs("../figures_en", exist_ok=True)
    fig_refinement_sequence()
    fig_primal_vs_dual()
    fig_dual_cell_detail()
