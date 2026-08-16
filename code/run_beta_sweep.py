import numpy as np
import matplotlib.pyplot as plt
from poisson_iterative import PoissonIterativeSolver, analytic_solution, source_term

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

levels = [0, 1, 2, 3, 4]
beta_opts, iters_opts, l2s, linfs = [], [], [], []

for level in levels:
    solver = PoissonIterativeSolver(level=level)
    f = source_term(solver.mesh.vertices)
    u_exact = analytic_solution(solver.mesh.vertices)
    betas = np.concatenate([np.linspace(0.02, 1, 40), np.linspace(1.1, 10, 20)])
    best = None
    for beta in betas:
        uc, n_iter = solver.solve(f, beta, tol=1e-6, max_iter=3000, impose_zero_mean=True)
        if best is None or n_iter < best[1]:
            best = (beta, n_iter, uc)
    beta_opt, iters_opt, uc = best
    err = np.abs(uc - u_exact)
    beta_opts.append(beta_opt)
    iters_opts.append(iters_opt)
    l2s.append(np.sqrt(np.mean(err ** 2)))
    linfs.append(err.max())
    print(f"nivel {level}: V={solver.V:5d}  beta_opt={beta_opt:.4f}  iter={iters_opt:5d}  "
          f"erro_l2={l2s[-1]:.5f}  erro_inf={linfs[-1]:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(levels, beta_opts, "o-", color="#2b6cb0")
axes[0].set_xlabel("nível de malha $n$")
axes[0].set_ylabel(r"$\beta$ ótimo")
axes[0].grid(alpha=0.3)

axes[1].plot(levels, iters_opts, "s-", color="#c05621")
axes[1].set_xlabel("nível de malha $n$")
axes[1].set_ylabel(r"iterações (com $\beta$ ótimo)")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
OUT_FIG = "../figures/fig_beta_otimo_vs_nivel.pdf"
OUT_CSV = "../notes/data/beta_otimo_vs_nivel.csv"
fig.savefig(OUT_FIG, bbox_inches="tight")
print("salvo", OUT_FIG)

np.savetxt(OUT_CSV,
           np.column_stack([levels, beta_opts, iters_opts, l2s, linfs]),
           delimiter=",", header="nivel,beta_otimo,iteracoes,erro_l2,erro_inf", comments="")
print("salvo", OUT_CSV)
