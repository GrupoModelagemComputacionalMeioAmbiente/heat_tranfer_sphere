"""
Caso 3: resposta a uma fonte de calor harmonica -- verificacao com
solucao analitica NAO-homogenea (f != 0, estendendo a Secao 6.1, que so
testa f=0) e convergencia ao estado estacionario.

Fonte f = Y_l^m (harmonico esferico real, l>=1 => media nula sobre a
esfera, condicao necessaria para existir estado estacionario: como
d/dt integral(u dS) = integral(f dS), um estado estacionario so existe
se a fonte tiver media nula).

Como Y_l^m e autofuncao do Laplace-Beltrami com autovalor -l(l+1), o
problema u_t = Delta_S u + Y_l^m, u(x,0)=0, tem solucao FECHADA exata:

    u(x,t) = [Y_l^m(x) / (l(l+1))] * (1 - exp(-l(l+1) t))

que converge, quando t -> infinito, ao estado estacionario exato do
problema de Poisson associado, -Delta_S u_inf = Y_l^m:

    u_inf(x) = Y_l^m(x) / (l(l+1))

Isso permite (i) estender a verificacao com solucao analitica da Secao
6.1 para o caso nao-homogeneo, com erro esperado O(h^2 + Delta t^2), e
(ii) demonstrar numericamente a convergencia ao estado estacionario,
cruzando com a formulacao de Poisson (Secoes 4.3 e 6.4-6.6).
"""
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatSphereFV, real_spherical_harmonic

plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.dpi": 300})

l, m = 2, 1
eig = l * (l + 1)


def u_exact(Y, t):
    return (Y / eig) * (1.0 - np.exp(-eig * t))


# --- (i) verificacao com solucao analitica nao-homogenea, ordem de convergencia ---
levels = [2, 3, 4, 5, 6]
dt = 0.02
t_test = 1.0
n_steps = round(t_test / dt)

errs_l2 = []
Vs = []
for level in levels:
    solver = HeatSphereFV(level=level)
    Y = real_spherical_harmonic(l, m, solver.mesh.vertices)
    hist = solver.solve(np.zeros(solver.V), dt, n_steps, f=Y)
    u_num = hist[-1][1]
    u_ex = u_exact(Y, t_test)
    err = u_num - u_ex
    errl2 = np.sqrt(np.dot(solver.area, err ** 2) / solver.total_area)
    errs_l2.append(errl2)
    Vs.append(solver.V)
    print(f"nivel {level}: V={solver.V:7d}  erro_l2={errl2:.3e}")

orders = [np.log2(errs_l2[i] / errs_l2[i + 1]) for i in range(len(errs_l2) - 1)]
print("ordens espaciais estimadas (caso nao-homogeneo):", [f"{o:.2f}" for o in orders])

fig, ax = plt.subplots(figsize=(4.3, 3.6))
h_approx = [np.sqrt(4 * np.pi / V) for V in Vs]
ax.loglog(h_approx, errs_l2, "o-", color="#2b6cb0", label="erro numérico")
ref = errs_l2[0] * (np.array(h_approx) / h_approx[0]) ** 2
ax.loglog(h_approx, ref, "--", color="gray", label=r"$O(h^2)$")
ax.set_xlabel(r"$h \sim \sqrt{4\pi/V_n}$")
ax.set_ylabel(r"erro $L^2$ (fonte harmônica, $t=1{,}0$)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("../figures/fig_caso3_convergencia_espacial.pdf", bbox_inches="tight")
print("salvo ../figures/fig_caso3_convergencia_espacial.pdf")

np.savetxt("../notes/data/caso3_convergencia_espacial.csv",
           np.column_stack([levels, Vs, errs_l2]),
           delimiter=",", header="nivel,V,erro_l2", comments="")

# --- (ii) convergencia ao estado estacionario ---
level_ss = 4
solver = HeatSphereFV(level=level_ss)
Y = real_spherical_harmonic(l, m, solver.mesh.vertices)
u_inf = Y / eig

dt = 0.05
t_final = 3.0
n_steps = round(t_final / dt)
hist = solver.solve(np.zeros(solver.V), dt, n_steps, f=Y, save_every=1)

t_arr = np.array([h[0] for h in hist])
err_ss = np.array([
    np.sqrt(np.dot(solver.area, (h[1] - u_inf) ** 2) / solver.total_area)
    for h in hist
])

fig, ax = plt.subplots(figsize=(4.3, 3.6))
ax.semilogy(t_arr, err_ss, "-", color="#c05621", label="numérico")
ax.semilogy(t_arr, err_ss[0] * np.exp(-eig * t_arr), "--", color="gray",
            label=r"$e^{-l(l+1)t}$ (taxa exata)")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\|u(\cdot,t) - u_\infty\|_{L^2}$")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("../figures/fig_caso3_estado_estacionario.pdf", bbox_inches="tight")
print("salvo ../figures/fig_caso3_estado_estacionario.pdf")

np.savetxt("../notes/data/caso3_estado_estacionario.csv",
           np.column_stack([t_arr, err_ss]),
           delimiter=",", header="t,erro_l2_vs_estacionario", comments="")

print(f"erro final (t={t_final}) vs estado estacionario: {err_ss[-1]:.3e}")

# --- (iii) convergencia temporal (refinamento de Delta t, nivel de malha fixo) ---
# Complementa (i): la a malha variava com dt=0.02 fixo; aqui o nivel de
# malha e fixado suficientemente fino para que o erro espacial seja
# desprezivel frente ao erro temporal, e Delta t e refinado, sempre
# comparando contra a solucao analitica exata (13) em t_test fixo --
# mais rigoroso do que comparar solucoes numericas sucessivas (usado
# nos Casos 1-2), pois isola o erro de fato, sem depender da suposicao
# de convergencia monotona.
level_dt = 7
solver_dt = HeatSphereFV(level=level_dt)
Y_dt = real_spherical_harmonic(l, m, solver_dt.mesh.vertices)

# nivel de malha fino o bastante para que o piso de erro espacial
# (~1.5e-6, medido separadamente com dt->0) fique bem abaixo do erro
# temporal em todos os Delta t exceto o menor, onde a saturacao e
# discutida explicitamente no texto.
dts_conv = [0.4, 0.2, 0.1, 0.05, 0.025]
t_test_dt = 1.0
errs_dt = []
for dt_c in dts_conv:
    n_steps_c = round(t_test_dt / dt_c)
    hist_c = solver_dt.solve(np.zeros(solver_dt.V), dt_c, n_steps_c, f=Y_dt)
    u_num_c = hist_c[-1][1]
    u_ex_c = u_exact(Y_dt, t_test_dt)
    err_c = u_num_c - u_ex_c
    errl2_c = np.sqrt(np.dot(solver_dt.area, err_c ** 2) / solver_dt.total_area)
    errs_dt.append(errl2_c)
    print(f"dt={dt_c}: erro_l2={errl2_c:.3e}")

orders_dt = [np.log2(errs_dt[i] / errs_dt[i + 1]) for i in range(len(errs_dt) - 1)]
print("ordens temporais estimadas (caso nao-homogeneo):", [f"{o:.2f}" for o in orders_dt])

fig, ax = plt.subplots(figsize=(4.3, 3.6))
ax.loglog(dts_conv, errs_dt, "s-", color="#c05621", label="erro numérico")
ref_dt = errs_dt[0] * (np.array(dts_conv) / dts_conv[0]) ** 2
ax.loglog(dts_conv, ref_dt, "--", color="gray", label=r"$O(\Delta t^2)$")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"erro $L^2$ (fonte harmônica, $t=1{,}0$, nível $6$)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("../figures/fig_caso3_convergencia_dt.pdf", bbox_inches="tight")
print("salvo ../figures/fig_caso3_convergencia_dt.pdf")

np.savetxt("../notes/data/caso3_convergencia_dt.csv",
           np.column_stack([dts_conv, errs_dt]),
           delimiter=",", header="dt,erro_l2", comments="")

# --- (iv) "retrato" bonito da evolucao temporal sobre a esfera, no
# mesmo estilo das figuras de snapshots dos Casos 1-2, para dar ao
# leitor uma visualizacao direta do transiente ate o estado
# estacionario (soma de um modo l=2,m=1 crescendo suavemente a partir
# de zero) ---
def fig_caso3_snapshots(level=3, dt=0.05, times_idx=(0, 20, 40, 80),
                         outfile="../figures/fig_caso3_snapshots.pdf", cmap="RdBu_r"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    solver_s = HeatSphereFV(level=level)
    verts, faces = solver_s.mesh.vertices, solver_s.mesh.faces
    Ys = real_spherical_harmonic(l, m, verts)
    n_steps_s = max(times_idx)
    hist_s = solver_s.solve(np.zeros(solver_s.V), dt, n_steps_s, f=Ys, save_every=1)

    vmax_s = max(np.abs(hist_s[t][1]).max() for t in times_idx)
    vmin_s = -vmax_s

    fig = plt.figure(figsize=(4 * len(times_idx), 4.2))
    for k, tidx in enumerate(times_idx):
        ax = fig.add_subplot(1, len(times_idx), k + 1, projection="3d")
        t, u = hist_s[tidx]
        face_vals = u[faces].mean(axis=1)
        norm = plt.Normalize(vmin_s, vmax_s)
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


fig_caso3_snapshots()
