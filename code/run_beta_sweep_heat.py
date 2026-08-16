"""
Varredura de beta para o algoritmo iterativo local da equacao do CALOR
(eqs. 10.18-10.20 da tese), analoga a run_beta_sweep.py (feita para
Poisson) -- gera beta_otimo x nivel de malha e o numero de iteracoes
correspondente, para um UNICO passo de Crank-Nicolson a partir de uma
condicao inicial suave (harmonico esferico), com o fluxo inicial
corretamente inicializado (ver HeatIterativeSolver._initial_flux).

Existe um beta_otimo interior (assim como em Poisson), mas em uma escala
MUITO maior (dezenas a centenas, nao ~0.5): para beta pequeno a
convergencia e' lenta (como em Poisson); para beta muito grande, xi_j =
c_j/(1+c_j*beta) -> 0, a acoplagem entre celulas vizinhas enfraquece e a
iteracao "descola" da equacao correta (o numero de iteracoes volta a
crescer E a diferenca em relacao ao solver direto tambem cresce). A
grade de varredura precisa cobrir varias ordens de grandeza (0.1 a
~2e4) para capturar esse minimo.
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
    print(f"nivel {level}: V={it.V:6d}  beta_opt={beta_opt:.4f}  iter={iters_opt:6d}  "
          f"diff_rel_vs_direto={diff:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(levels, beta_opts, "o-", color="#2b6cb0")
axes[0].set_xlabel("nível de malha $n$")
axes[0].set_ylabel(r"$\beta$ (menor iteração na grade testada)")
axes[0].grid(alpha=0.3)

axes[1].plot(levels, iters_opts, "s-", color="#c05621")
axes[1].set_xlabel("nível de malha $n$")
axes[1].set_ylabel(r"iterações (1 passo de CN, com $\beta$ da coluna esquerda)")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
OUT_FIG = "../figures/fig_beta_otimo_vs_nivel_calor.pdf"
OUT_CSV = "../notes/data/beta_otimo_vs_nivel_calor.csv"
fig.savefig(OUT_FIG, bbox_inches="tight")
print("salvo", OUT_FIG)

np.savetxt(OUT_CSV,
           np.column_stack([levels, beta_opts, iters_opts, diffs]),
           delimiter=",", header="nivel,beta_menor_iter,iteracoes,diff_rel_vs_direto", comments="")
print("salvo", OUT_CSV)
