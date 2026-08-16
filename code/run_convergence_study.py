"""
Estudo de convergencia com solucao analitica exata (harmonicos
esfericos) -- resultado NOVO em relacao a tese/artigo de congresso
originais, que so compararam solucoes numericas entre si (niveis de
malha e passos de tempo consecutivos), sem uma solucao exata de
referencia para o problema transiente.

Solucao exata usada: para f=0 e u0 = Y_l^m (harmonico esferico real),
a solucao exata da equacao do calor homogenea sobre S^2 e

    u(x,t) = exp(-l(l+1) t) Y_l^m(x)

pois Y_l^m e autofuncao do operador de Laplace-Beltrami com autovalor
-l(l+1). Isso permite calcular o erro numerico exato (nao apenas a
diferenca entre duas solucoes numericas) e estimar rigorosamente a
ordem de convergencia espacial (refinamento de malha, dt fixo pequeno)
e temporal (refinamento de dt, malha fixa fina).
"""

import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV, real_spherical_harmonic

plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.dpi": 300})

L, M = 3, 2  # harmonico esferico usado como condicao inicial exata


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
        # comprimento de aresta caracteristico h ~ sqrt(area media da celula)
        h_char = np.sqrt(solver.total_area / solver.V)
        rows.append((level, solver.V, h_char, err_l2, err_inf))

    with open(outfile_csv, "w") as fh:
        fh.write("nivel,V,h_caracteristico,erro_L2,erro_Linf\n")
        for level, V, h, e2, ei in rows:
            fh.write(f"{level},{V},{h:.6f},{e2:.8e},{ei:.8e}\n")
    print("salvo", outfile_csv)

    # ordem estimada por ajuste log-log entre niveis consecutivos
    orders = []
    for k in range(1, len(rows)):
        h0, e0 = rows[k - 1][2], rows[k - 1][3]
        h1, e1 = rows[k][2], rows[k][3]
        p = np.log(e1 / e0) / np.log(h1 / h0)
        orders.append(p)
    print("ordens espaciais estimadas:", [f"{p:.2f}" for p in orders])

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    hs = [r[2] for r in rows]
    e2s = [r[3] for r in rows]
    ax.loglog(hs, e2s, "o-", color="#2b6cb0", label=r"erro $L^2$ (numérico)")
    # linha de referencia O(h^2)
    ref = e2s[0] * (np.array(hs) / hs[0]) ** 2
    ax.loglog(hs, ref, "--", color="gray", label=r"referência $O(h^2)$")
    ax.set_xlabel(r"$h$ característico da malha")
    ax.set_ylabel(r"erro $L^2$ em $t=%.2f$" % t_final)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outfile_fig, bbox_inches="tight")
    print("salvo", outfile_fig)
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
        fh.write("dt,n_passos,erro_L2\n")
        for dt, n, e in rows:
            fh.write(f"{dt},{n},{e:.8e}\n")
    print("salvo", outfile_csv)

    orders = []
    for k in range(1, len(rows)):
        dt0, e0 = rows[k - 1][0], rows[k - 1][2]
        dt1, e1 = rows[k][0], rows[k][2]
        p = np.log(e1 / e0) / np.log(dt1 / dt0)
        orders.append(p)
    print("ordens temporais estimadas:", [f"{p:.2f}" for p in orders])

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    dts_arr = [r[0] for r in rows]
    e2s = [r[2] for r in rows]
    ax.loglog(dts_arr, e2s, "s-", color="#c05621", label=r"erro $L^2$ (numérico)")
    ref = e2s[0] * (np.array(dts_arr) / dts_arr[0]) ** 2
    ax.loglog(dts_arr, ref, "--", color="gray", label=r"referência $O(\Delta t^2)$")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"erro $L^2$ em $t=%.2f$ (nível %d)" % (t_final, level))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outfile_fig, bbox_inches="tight")
    print("salvo", outfile_fig)
    return rows, orders


if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    os.makedirs("../notes/data", exist_ok=True)

    print("=== Convergencia espacial ===")
    rows_s, orders_s = spatial_convergence(
        levels=[1, 2, 3, 4, 5], dt=1e-3, t_final=0.1,
        outfile_csv="../notes/data/convergencia_espacial.csv",
        outfile_fig="../figures/fig_convergencia_espacial.pdf")

    print("=== Convergencia temporal ===")
    rows_t, orders_t = temporal_convergence(
        level=6, dts=[0.1, 0.05, 0.025, 0.0125], t_final=0.4,
        outfile_csv="../notes/data/convergencia_temporal.csv",
        outfile_fig="../figures/fig_convergencia_temporal.pdf")

    with open("../notes/data/ordens_convergencia.txt", "w") as fh:
        fh.write("Ordens de convergencia espacial (entre niveis consecutivos):\n")
        fh.write(", ".join(f"{p:.3f}" for p in orders_s) + "\n")
        fh.write(f"media = {np.mean(orders_s):.3f}\n\n")
        fh.write("Ordens de convergencia temporal (entre dt consecutivos):\n")
        fh.write(", ".join(f"{p:.3f}" for p in orders_t) + "\n")
        fh.write(f"media = {np.mean(orders_t):.3f}\n")
    print("salvo ordens de convergencia")
