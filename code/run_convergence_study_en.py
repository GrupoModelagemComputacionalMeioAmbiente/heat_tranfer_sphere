"""
English-label counterpart of run_convergence_study.py -- convergence
study against the exact spherical-harmonic solution. See the original
script's docstring for the mathematical rationale.
"""

import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV, real_spherical_harmonic

plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.dpi": 300})

L, M = 3, 2


def spatial_convergence(levels, dt, t_final, outfile_csv, outfile_fig):
    rows = []
    for level in levels:
        solver = HeatSphereFV(level=level)
        u0 = real_spherical_harmonic(L, M, solver.mesh.vertices)
        n_steps = round(t_final / dt)
        hist = solver.solve(u0, dt, n_steps, f=0.0)
        t_f, u_f = hist[-1]
        u_exact = np.exp(-L * (L + 1) * t_f) * u0
        err_l2 = np.sqrt(np.dot(solver.area, (u_f - u_exact) ** 2) / solver.total_area)
        err_inf = np.max(np.abs(u_f - u_exact))
        h_char = np.sqrt(solver.total_area / solver.V)
        rows.append((level, solver.V, h_char, err_l2, err_inf))

    with open(outfile_csv, "w") as fh:
        fh.write("level,V,h_char,err_L2,err_Linf\n")
        for level, V, h, e2, ei in rows:
            fh.write(f"{level},{V},{h:.6f},{e2:.8e},{ei:.8e}\n")
    print("saved", outfile_csv)

    orders = []
    for k in range(1, len(rows)):
        h0, e0 = rows[k - 1][2], rows[k - 1][3]
        h1, e1 = rows[k][2], rows[k][3]
        p = np.log(e1 / e0) / np.log(h1 / h0)
        orders.append(p)
    print("estimated spatial orders:", [f"{p:.2f}" for p in orders])

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    hs = [r[2] for r in rows]
    e2s = [r[3] for r in rows]
    ax.loglog(hs, e2s, "o-", color="#2b6cb0", label=r"$L^2$ error (numerical)")
    ref = e2s[0] * (np.array(hs) / hs[0]) ** 2
    ax.loglog(hs, ref, "--", color="gray", label=r"$O(h^2)$ reference")
    ax.set_xlabel(r"characteristic mesh size $h$")
    ax.set_ylabel(r"$L^2$ error at $t=%.2f$" % t_final)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outfile_fig, bbox_inches="tight")
    print("saved", outfile_fig)
    return rows, orders


def temporal_convergence(level, dts, t_final, outfile_csv, outfile_fig):
    solver = HeatSphereFV(level=level)
    u0 = real_spherical_harmonic(L, M, solver.mesh.vertices)
    rows = []
    for dt in dts:
        n_steps = round(t_final / dt)
        hist = solver.solve(u0, dt, n_steps, f=0.0)
        t_f, u_f = hist[-1]
        u_exact = np.exp(-L * (L + 1) * t_f) * u0
        err_l2 = np.sqrt(np.dot(solver.area, (u_f - u_exact) ** 2) / solver.total_area)
        rows.append((dt, n_steps, err_l2))

    with open(outfile_csv, "w") as fh:
        fh.write("dt,n_steps,err_L2\n")
        for dt, n, e in rows:
            fh.write(f"{dt},{n},{e:.8e}\n")
    print("saved", outfile_csv)

    orders = []
    for k in range(1, len(rows)):
        dt0, e0 = rows[k - 1][0], rows[k - 1][2]
        dt1, e1 = rows[k][0], rows[k][2]
        p = np.log(e1 / e0) / np.log(dt1 / dt0)
        orders.append(p)
    print("estimated temporal orders:", [f"{p:.2f}" for p in orders])

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    dts_arr = [r[0] for r in rows]
    e2s = [r[2] for r in rows]
    ax.loglog(dts_arr, e2s, "s-", color="#c05621", label=r"$L^2$ error (numerical)")
    ref = e2s[0] * (np.array(dts_arr) / dts_arr[0]) ** 2
    ax.loglog(dts_arr, ref, "--", color="gray", label=r"$O(\Delta t^2)$ reference")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$L^2$ error at $t=%.2f$ (level %d)" % (t_final, level))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outfile_fig, bbox_inches="tight")
    print("saved", outfile_fig)
    return rows, orders


if __name__ == "__main__":
    import os
    os.makedirs("../figures_en", exist_ok=True)
    os.makedirs("../notes/data_en", exist_ok=True)

    print("=== Spatial convergence ===")
    rows_s, orders_s = spatial_convergence(
        levels=[1, 2, 3, 4, 5], dt=1e-3, t_final=0.1,
        outfile_csv="../notes/data_en/convergencia_espacial.csv",
        outfile_fig="../figures_en/fig_convergencia_espacial.pdf")

    print("=== Temporal convergence ===")
    rows_t, orders_t = temporal_convergence(
        level=6, dts=[0.1, 0.05, 0.025, 0.0125], t_final=0.4,
        outfile_csv="../notes/data_en/convergencia_temporal.csv",
        outfile_fig="../figures_en/fig_convergencia_temporal.pdf")

    with open("../notes/data_en/ordens_convergencia.txt", "w") as fh:
        fh.write("Estimated spatial convergence orders (between consecutive levels):\n")
        fh.write(", ".join(f"{p:.3f}" for p in orders_s) + "\n")
        fh.write(f"mean = {np.mean(orders_s):.3f}\n\n")
        fh.write("Estimated temporal convergence orders (between consecutive dt):\n")
        fh.write(", ".join(f"{p:.3f}" for p in orders_t) + "\n")
        fh.write(f"mean = {np.mean(orders_t):.3f}\n")
    print("saved convergence orders")
