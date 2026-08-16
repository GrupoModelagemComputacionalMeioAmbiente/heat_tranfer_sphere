"""
Reproducao (em espirito) dos Casos 1 e 2 do Cap. 12 da tese / Sec. 5 do
artigo de congresso: condicao inicial concentrada em um hemisferio,
evoluida sob a equacao do calor (i) sem fonte (Caso 1) e (ii) com fonte
de calor constante na mesma regiao inicialmente quente (Caso 2).

A tese nao fornece a forma fechada exata da condicao inicial usada
(apenas os valores medios ~1.8/~0.2 lidos do grafico da Figura 12.1),
de modo que aqui adotamos uma condicao inicial suave e reprodutivel,
com a mesma interpretacao fisica (hemisferio "quente" vs. "frio"),
documentada explicitamente para uso no artigo:

    u0(x) = 1.0 + 0.8 * tanh(k * z(x))      (z = altura sobre o eixo polar)

com k=4, que varia suavemente entre ~0.2 (polo sul) e ~1.8 (polo norte),
evitando oscilacoes de Gibbs de uma funcao degrau descontinua na malha.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV

plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.dpi": 300})


def initial_condition(mesh_vertices, k=4.0):
    z = mesh_vertices[:, 2]
    return 1.0 + 0.8 * np.tanh(k * z)


def source_case2(mesh_vertices):
    """Fonte de calor constante igual a 1 no hemisferio norte (mesma
    regiao inicialmente mais quente), nula no hemisferio sul -- analogo
    a Figura 12.10 da tese."""
    z = mesh_vertices[:, 2]
    return np.where(z > 0, 1.0, 0.0)


def run_case(levels, dts, t_final, source_fn=None, label=""):
    results = {}
    for n in levels:
        solver = HeatSphereFV(level=n)
        u0 = initial_condition(solver.mesh.vertices)
        f = source_fn(solver.mesh.vertices) if source_fn is not None else 0.0
        for dt in dts:
            n_steps = round(t_final / dt)
            hist = solver.solve(u0, dt, n_steps, f=f)
            t_arr = [h[0] for h in hist]
            mean_arr = [solver.mean(h[1]) for h in hist]
            results[(n, dt)] = dict(t=t_arr, mean=mean_arr, u_final=hist[-1][1],
                                     solver=solver, u0=u0)
        print(f"[{label}] nivel {n} concluido")
    return results


def table_mean_conservation(results, levels, dt_ref, t_final, outfile):
    rows = []
    for n in levels:
        r = results[(n, dt_ref)]
        rows.append((n, r["mean"][0], r["mean"][-1]))
    with open(outfile, "w") as fh:
        fh.write("nivel,V,media_t0,media_tfinal\n")
        for n, m0, mf in rows:
            V = results[(n, dt_ref)]["solver"].V
            fh.write(f"{n},{V},{m0:.6f},{mf:.6f}\n")
    print("salvo", outfile)
    return rows


def table_dt_sensitivity(results, level_ref, dts, t_final, outfile):
    rows = []
    for dt in dts:
        r = results[(level_ref, dt)]
        rows.append((dt, r["mean"][-1]))
    with open(outfile, "w") as fh:
        fh.write("dt,media_tfinal\n")
        for dt, mf in rows:
            fh.write(f"{dt},{mf:.6f}\n")
    print("salvo", outfile)
    return rows


def fig_mesh_level_convergence(results, levels, dt_ref, outfile):
    """Norma 1 da diferenca entre solucoes em niveis consecutivos de
    malha (mesmo t final), interpolando a malha mais grossa nos
    vertices da mais fina por proximidade angular -- analogo a Fig.
    12.8/12.16 da tese."""
    from scipy.spatial import cKDTree
    diffs = []
    labels = []
    for a, b in zip(levels[:-1], levels[1:]):
        ra = results[(a, dt_ref)]
        rb = results[(b, dt_ref)]
        tree = cKDTree(ra["solver"].mesh.vertices)
        _, idx = tree.query(rb["solver"].mesh.vertices)
        ua_interp = ra["u_final"][idx]
        diff = np.mean(np.abs(ua_interp - rb["u_final"]))
        diffs.append(diff)
        labels.append(f"{a}–{b}")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(labels, diffs, "o-", color="#2b6cb0")
    ax.set_xlabel("níveis de malha consecutivos")
    ax.set_ylabel(r"$\|u_a - u_b\|_1$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("salvo", outfile)
    return diffs


def fig_dt_convergence(results, level_ref, dts, outfile):
    diffs = []
    labels = []
    for a, b in zip(dts[:-1], dts[1:]):
        ua = results[(level_ref, a)]["u_final"]
        ub = results[(level_ref, b)]["u_final"]
        diff = np.mean(np.abs(ua - ub))
        diffs.append(diff)
        labels.append(f"{a}–{b}")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(labels, diffs, "s-", color="#c05621")
    ax.set_xlabel(r"pares de $\Delta t$ consecutivos")
    ax.set_ylabel(r"$\|u_{\Delta t_a} - u_{\Delta t_b}\|_1$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("salvo", outfile)
    return diffs


def fig_snapshots(results, level_ref, dt_ref, times_idx, outfile, cmap="inferno", source_fn=None):
    r = results[(level_ref, dt_ref)]
    solver = r["solver"]
    verts = solver.mesh.vertices
    faces = solver.mesh.faces

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(4 * len(times_idx), 4.2))

    # rebuild history to get intermediate snapshots (already computed once;
    # recompute cheaply since level_ref mesh assembly is cached-free but fast)
    u0 = r["u0"]
    solver2 = r["solver"]
    dt = dt_ref
    n_steps = round(max(r["t"]) / dt)
    f = source_fn(verts) if source_fn is not None else np.zeros(solver2.V)
    hist = solver2.solve(u0, dt, n_steps, f=f, save_every=1)

    vmin = min(h[1].min() for h in hist)
    vmax = max(h[1].max() for h in hist)

    for k, tidx in enumerate(times_idx):
        ax = fig.add_subplot(1, len(times_idx), k + 1, projection="3d")
        t, u = hist[tidx]
        face_vals = u[faces].mean(axis=1)
        norm = plt.Normalize(vmin, vmax)
        colors = plt.colormaps[cmap](norm(face_vals))
        tris = verts[faces]
        poly = Poly3DCollection(tris, facecolor=colors, edgecolor="none")
        ax.add_collection3d(poly)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=35)
        ax.set_title(f"$t={t:.2f}$", fontsize=11, y=0.95)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("salvo", outfile)


def fig_source_case2(level=3, outfile="../figures/fig_fonte_caso2.pdf", cmap="inferno"):
    """Visualizacao do termo-fonte f (Caso 2) sobre a esfera, lado a
    lado com a condicao inicial u0 -- ajuda o leitor a visualizar a
    configuracao do problema antes de ver a evolucao temporal."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    solver = HeatSphereFV(level=level)
    verts, faces = solver.mesh.vertices, solver.mesh.faces
    u0 = initial_condition(verts)
    f = source_case2(verts)

    fig = plt.figure(figsize=(8.4, 4.2))
    for k, (vals, title) in enumerate([(u0, r"condição inicial $u_0$"), (f, r"termo-fonte $f$")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        face_vals = vals[faces].mean(axis=1)
        norm = plt.Normalize(vals.min(), vals.max())
        colors = plt.colormaps[cmap](norm(face_vals))
        tris = verts[faces]
        poly = Poly3DCollection(tris, facecolor=colors, edgecolor="none")
        ax.add_collection3d(poly)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=35)
        ax.set_title(title, fontsize=12, y=0.95)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print("salvo", outfile)


if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    os.makedirs("../notes/data", exist_ok=True)

    levels = [0, 1, 2, 3, 4, 5, 6]
    dts = [1.0, 0.1, 0.01]
    t_final = 4.0

    print("=== Caso 1 (f=0) ===")
    res1 = run_case(levels, dts, t_final, source_fn=None, label="Caso1")
    table_mean_conservation(res1, levels, dt_ref=0.1, t_final=t_final,
                             outfile="../notes/data/caso1_conservacao_malha.csv")
    table_dt_sensitivity(res1, level_ref=3, dts=dts, t_final=t_final,
                          outfile="../notes/data/caso1_conservacao_dt.csv")
    fig_mesh_level_convergence(res1, levels, dt_ref=0.1,
                                outfile="../figures/fig_caso1_convergencia_malha.pdf")
    fig_dt_convergence(res1, level_ref=3, dts=dts,
                        outfile="../figures/fig_caso1_convergencia_dt.pdf")

    print("=== Caso 2 (fonte constante) ===")
    res2 = run_case(levels, dts, t_final, source_fn=source_case2, label="Caso2")
    table_mean_conservation(res2, levels, dt_ref=0.1, t_final=t_final,
                             outfile="../notes/data/caso2_media_malha.csv")
    table_dt_sensitivity(res2, level_ref=3, dts=dts, t_final=t_final,
                          outfile="../notes/data/caso2_media_dt.csv")
    fig_mesh_level_convergence(res2, levels, dt_ref=0.1,
                                outfile="../figures/fig_caso2_convergencia_malha.pdf")
    fig_dt_convergence(res2, level_ref=3, dts=dts,
                        outfile="../figures/fig_caso2_convergencia_dt.pdf")

    print("=== Snapshots (Caso 1, nivel 3) ===")
    fig_snapshots(res1, level_ref=3, dt_ref=0.1, times_idx=[0, 10, 20, 40],
                  outfile="../figures/fig_caso1_snapshots.pdf")

    print("=== Snapshots (Caso 2, nivel 3) ===")
    fig_snapshots(res2, level_ref=3, dt_ref=0.1, times_idx=[0, 10, 20, 40],
                  outfile="../figures/fig_caso2_snapshots.pdf", source_fn=source_case2)

    print("=== Fontes (Caso 2) ===")
    fig_source_case2(outfile="../figures/fig_fonte_caso2.pdf")

    print("done")
