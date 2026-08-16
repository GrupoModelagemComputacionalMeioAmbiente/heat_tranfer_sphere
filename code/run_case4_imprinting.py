"""
Caso 4: efeito das 12 singularidades pentagonais da malha icosaedral
("grid imprinting") sobre a precisao do operador discreto -- conecta
diretamente com Peixoto e Barros (2013), ja citados no artigo, que
estudam o mesmo efeito para operadores miméticos em malhas geodesicas.

Diagnostico: erro de truncamento LOCAL do laplaciano discreto L (o
mesmo L de heat_solver.py, Secao 4.3) em cada vertice, comparado
separadamente nos 12 vertices pentagonais (indices 0-11, defeitos
topologicos permanentes: 5 vizinhos em vez de 6) e nos vertices
hexagonais "genericos" (6 vizinhos, longe de qualquer pentagono).

Usamos como funcao teste o harmonico esferico Y_2^1 (suave, sem
simetria acidental com a malha), cujo laplaciano exato e' conhecido:
Delta_S Y_l^m = -l(l+1) Y_l^m. O erro de truncamento em cada vertice i
e'

    e_i = (L u)_i / A_i - l(l+1) Y_l^m(x_i)

(sinal: L u aproxima -Delta_S u vezes a area da celula, ver Secao 4.3).
Comparamos max_i |e_i| restrito aos 12 vertices pentagonais contra a
mediana de |e_i| nos vertices hexagonais, para varios niveis de malha,
para ver se a ordem de convergencia LOCAL perto dos pentagonos e'
inferior a' ordem global O(h^2) confirmada na Secao 6.1.
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
    approx_lap = Lu / solver.area          # aproxima -Delta_S Y
    exact_lap = eig * Y                     # -Delta_S Y = l(l+1) Y
    e = approx_lap - exact_lap

    e_pent = np.abs(e[:12])
    e_hex = np.abs(e[12:])

    err_pent_max.append(e_pent.max())
    err_hex_median.append(np.median(e_hex))
    err_hex_max.append(e_hex.max())
    hs.append(np.sqrt(4 * np.pi / solver.V))
    print(f"nivel {level}: V={solver.V:7d}  erro_trunc_max_pentagono={e_pent.max():.4e}  "
          f"erro_trunc_mediano_hexagono={np.median(e_hex):.4e}  "
          f"erro_trunc_max_hexagono={e_hex.max():.4e}")

order_pent = [np.log2(err_pent_max[i] / err_pent_max[i + 1]) for i in range(len(levels) - 1)]
order_hex = [np.log2(err_hex_median[i] / err_hex_median[i + 1]) for i in range(len(levels) - 1)]
print("ordem local estimada (max nos 12 pentagonos):", [f"{o:.2f}" for o in order_pent])
print("ordem local estimada (mediana nos hexagonos):", [f"{o:.2f}" for o in order_hex])

fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.loglog(hs, err_pent_max, "o-", color="#c05621", label="máx. nos 12 pentágonos")
ax.loglog(hs, err_hex_median, "s-", color="#2b6cb0", label="mediana nos hexágonos")
ref = np.array(err_hex_median[0]) * (np.array(hs) / hs[0]) ** 2
ax.loglog(hs, ref, "--", color="gray", label=r"$O(h^2)$")
ax.set_xlabel(r"$h \sim \sqrt{4\pi/V_n}$")
ax.set_ylabel(r"erro de truncamento local $|e_i|$")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("../figures/fig_caso4_imprinting_nivel.pdf", bbox_inches="tight")
print("salvo ../figures/fig_caso4_imprinting_nivel.pdf")

with open("../notes/data/caso4_imprinting_nivel.csv", "w") as fh:
    fh.write("nivel,V,erro_trunc_max_pentagono,erro_trunc_mediano_hexagono,erro_trunc_max_hexagono\n")
    for lv, V, a, b, c in zip(levels, [round(4*np.pi/h**2) for h in hs], err_pent_max, err_hex_median, err_hex_max):
        fh.write(f"{lv},{V},{a:.6e},{b:.6e},{c:.6e}\n")
print("salvo ../notes/data/caso4_imprinting_nivel.csv")

# --- mapa espacial do erro (nivel fixo), para visualizar onde o erro se concentra ---
# Duas vistas (frente/verso, 180 graus de diferenca em azimute) para que
# os 12 vertices pentagonais -- distribuidos por toda a esfera -- fiquem
# todos visiveis em pelo menos uma das duas; sem isso, cerca da metade
# fica sempre do lado oculto de qualquer vista unica.
#
# O Poly3DCollection do matplotlib nao faz um z-buffer real: por padrao,
# marcadores de dispersao (scatter) sobrepostos a uma malha 3D sao
# desenhados por baixo dos poligonos da superficie exceto exatamente na
# silhueta, tornando a maioria dos marcadores invisiveis. A correcao usa
# `ax.computed_zorder = False` (disponivel desde o matplotlib 3.5), que
# permite controlar manualmente a ordem de desenho por artista; em
# conjunto, filtramos para desenhar em cada vista apenas os vertices do
# hemisferio voltado para a camera (produto escalar entre a normal do
# vertice e a direcao da camera positivo), evitando marcar pontos que
# deveriam estar ocultos atras da esfera.
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


# Os 12 vertices pentagonais do icosaedro formam 6 pares antipodais
# (v, -v); a segunda vista usa a camera exatamente antipodal a primeira
# (eleva-se -elev e some-se 180 graus ao azimute) para garantir que os 6
# vertices "traseiros" da primeira vista sejam exatamente os 6
# "dianteiros" da segunda -- cobrindo os 12 vertices sem sobreposicao.
views = [(20, 35), (-20, 35 + 180)]

fig = plt.figure(figsize=(9.5, 5))
for k, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.computed_zorder = False  # controle manual de profundidade (ver nota acima)

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
                     linewidth=1.2, depthshade=False, label="vértices pentagonais")
    sc.set_zorder(10)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("frente" if k == 0 else "verso", fontsize=11, y=0.95)
    if k == 0:
        ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("../figures/fig_caso4_imprinting_mapa.pdf", bbox_inches="tight")
print("salvo ../figures/fig_caso4_imprinting_mapa.pdf")
