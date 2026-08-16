"""
English-label counterpart of run_beta_sweep_heat.py -- beta sweep for the
HEAT equation's local iterative algorithm (thesis eqs. 10.18-10.20).
See the original script's docstring for the numerical rationale.
"""
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV, real_spherical_harmonic
from heat_iterative import HeatIterativeSolver

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

levels = [0, 1, 2, 3, 4]
dt = 0.05
l, m = 3, 2
tol = 1e-6
max_iter = 4000

betas = np.geomspace(0.5, 5000, 60)

beta_opts, iters_opts, diffs = [], [], []

for level in levels:
    it = HeatIterativeSolver(level=level)
    u0 = real_spherical_harmonic(l, m, it.mesh.vertices)
    w0 = it._initial_flux(u0, dt)

    direct = HeatSphereFV(level=level)
    lu = direct.make_lu(dt)
    u1_direct = direct.step(u0, dt, np.zeros(direct.V), lu=lu)

    best = None
    for beta in betas:
        u1, w1, n_iter = it._step(u0, w0, dt, np.zeros(it.V), beta=beta, tol=tol, max_iter=max_iter)
        if best is None or n_iter < best[1]:
            best = (beta, n_iter, u1)
    beta_opt, iters_opt, u1_best = best
    diff = np.max(np.abs(u1_best - u1_direct)) / np.max(np.abs(u1_direct))
    beta_opts.append(beta_opt)
    iters_opts.append(iters_opt)
    diffs.append(diff)
    print(f"level {level}: V={it.V:6d}  beta_opt={beta_opt:.4f}  iter={iters_opt:6d}  "
          f"rel_diff_vs_direct={diff:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(levels, beta_opts, "o-", color="#2b6cb0")
axes[0].set_xlabel("mesh level $n$")
axes[0].set_ylabel(r"$\beta$ (fewest iterations on tested grid)")
axes[0].grid(alpha=0.3)

axes[1].plot(levels, iters_opts, "s-", color="#c05621")
axes[1].set_xlabel("mesh level $n$")
axes[1].set_ylabel(r"iterations (1 CN step, with $\beta$ from left column)")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
OUT_FIG = "../figures_en/fig_beta_otimo_vs_nivel_calor.pdf"
OUT_CSV = "../notes/data_en/beta_otimo_vs_nivel_calor.csv"
fig.savefig(OUT_FIG, bbox_inches="tight")
print("saved", OUT_FIG)

np.savetxt(OUT_CSV,
           np.column_stack([levels, beta_opts, iters_opts, diffs]),
           delimiter=",", header="level,beta_fewest_iter,iterations,rel_diff_vs_direct", comments="")
print("saved", OUT_CSV)
