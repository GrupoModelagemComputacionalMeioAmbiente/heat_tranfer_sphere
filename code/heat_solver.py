"""
Solver da equacao do calor sobre a esfera, discretizada por volumes
finitos mistos (potencial-fluxo) na malha dual icosaedral, com
semi-discretizacao temporal de Crank-Nicolson -- Caps. 10-11 da tese de
Fabio Freitas Ferreira (UERJ/IPRJ, 2008).

Equivalencia com o metodo iterativo local da tese
--------------------------------------------------
A tese resolve, a cada passo de tempo, o sistema misto (potencial u nos
vertices, fluxo w nas arestas, eqs. 10.9-10.10) por um algoritmo
iterativo local celula-a-celula (tipo Gauss-Seidel, eqs. 10.18-10.20),
necessario na epoca por limitacoes de memoria/paralelizacao. A propria
tese demonstra (Sec. 10.3, eq. 10.14) que a eliminacao algebrica exata
do fluxo w leva a um UNICO sistema linear simetrico positivo-definido

        (D + A^T G A) u^{r+1} = f_bar^r                         (10.14)

que e precisamente o LAPLACIANO DE GRAFO PONDERADO da malha icosaedral
(D = areas das celulas duais, pesos de aresta g_ij = h_ij/d_ij). Ou
seja, o ponto fixo do algoritmo iterativo local da tese e, por
construcao, a solucao deste sistema linear esparso.

Nesta reimplementacao usamos diretamente a montagem e solucao esparsa
desse sistema global (via scipy.sparse), o que e matematicamente
equivalente ao algoritmo original e muito mais eficiente/robusto em
hardware atual -- eliminando a necessidade da iteracao local em k e do
parametro de relaxacao beta, que na tese existiam apenas para tornar o
metodo executavel com os recursos computacionais de 2008 (Acer Aspire
2992, Intel Celeron M 440, 1 GB RAM).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from icosphere import build_icosphere, build_dual_mesh, build_dual_geometry_fast


def _great_circle_distance(p, q, R):
    cos_ang = np.clip(np.dot(p, q) / (R * R), -1.0, 1.0)
    return R * np.arccos(cos_ang)


class HeatSphereFV:
    """Discretizacao em volumes finitos mistos da equacao do calor sobre
    a esfera, em malha icosaedral dual de nivel `level`.

        u_t = Delta_S u + f,   x em S^2,  t>0
        u(x,0) = u0(x)

    Semi-discretizacao espacial: M du/dt = -L u + M f
      M = diag(A_i)           (areas das celulas duais)
      L = laplaciano de grafo ponderado, peso w_ij = h_ij/d_ij
          (h_ij = comprimento da aresta dual entre os circuncentros
           que flanqueiam a aresta primal i-j; d_ij = distancia
           geodesica entre os vertices primais i e j)

    Semi-discretizacao temporal (Crank-Nicolson, eq. 10.3 da tese):
      (M + dt/2 L) u^{r+1} = (M - dt/2 L) u^r + dt M f

    A montagem usa `build_dual_geometry_fast` (icosphere.py), totalmente
    vetorizada em NumPy (sem laco Python por vertice/aresta), o que
    viabiliza niveis de refinamento altos (testado ate n=9, ~2,6 milhoes
    de vertices) -- a versao anterior, com um laco Python explicito por
    vertice, tornava-se impraticavel acima de n~7.
    """

    def __init__(self, level: int, R: float = 1.0):
        self.level = level
        self.R = R
        self.mesh = build_icosphere(level, R=R)
        self.geom = build_dual_geometry_fast(self.mesh)
        self.V = self.mesh.n_vertices
        self._assemble()

    def _assemble(self):
        mesh, geom = self.mesh, self.geom
        V = self.V
        edges = mesh.edges  # (E,2)

        w = geom.edge_h / geom.edge_d  # (E,) peso de cada aresta primal

        i_idx = edges[:, 0]
        j_idx = edges[:, 1]
        rows = np.concatenate([i_idx, j_idx, np.arange(V)])
        cols = np.concatenate([j_idx, i_idx, np.arange(V)])
        diag = np.zeros(V)
        np.add.at(diag, i_idx, w)
        np.add.at(diag, j_idx, w)
        vals = np.concatenate([-w, -w, diag])

        self.L = sp.csr_matrix((vals, (rows, cols)), shape=(V, V))
        self.M = sp.diags(geom.cell_area)
        self.area = geom.cell_area.copy()
        self.total_area = self.area.sum()

    def mean(self, u):
        """Aproximacao de integral_{S^2} u dS, por soma ponderada pelas
        areas das celulas duais (quadratura de primeira ordem)."""
        return float(np.dot(self.area, u))

    def step(self, u, dt, f, A_lhs=None, lu=None):
        """Um passo de Crank-Nicolson: retorna u^{r+1}."""
        rhs = (self.M - 0.5 * dt * self.L) @ u + dt * (self.M @ f)
        if lu is not None:
            return lu.solve(rhs)
        A = self.M + 0.5 * dt * self.L
        return spla.spsolve(A.tocsc(), rhs)

    def make_lu(self, dt):
        """Fatoracao LU da matriz do sistema (constante entre passos se
        dt for fixo) -- acelera integracoes com muitos passos de tempo."""
        A = (self.M + 0.5 * dt * self.L).tocsc()
        return spla.splu(A)

    def solve(self, u0, dt, n_steps, f=None, save_every=1):
        """Integra de t=0 ate t=n_steps*dt, retornando lista de (t, u)."""
        V = self.V
        if f is None:
            f = np.zeros(V)
        elif np.isscalar(f):
            f = np.full(V, float(f))

        lu = self.make_lu(dt)
        u = u0.copy()
        history = [(0.0, u.copy())]
        for r in range(1, n_steps + 1):
            u = self.step(u, dt, f, lu=lu)
            if r % save_every == 0:
                history.append((r * dt, u.copy()))
        return history


# -----------------------------------------------------------------
# Solucao analitica via harmonicos esfericos (para validacao/erro)
# -----------------------------------------------------------------

def real_spherical_harmonic(l, m, xyz):
    """Harmonico esferico real Y_l^m avaliado nos pontos xyz (N,3) sobre
    a esfera unitaria. Autofuncao do Laplace-Beltrami:
        Delta_S Y_l^m = -l(l+1) Y_l^m
    Usado para construir uma solucao exata da equacao do calor
    homogenea (f=0): u(x,t) = exp(-l(l+1) t) Y_l^m(x).
    """
    from scipy.special import sph_harm_y
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(z / r, -1, 1))   # colatitude
    phi = np.arctan2(y, x)                      # longitude

    Y = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2) * (-1) ** m * Y.real
    elif m < 0:
        return np.sqrt(2) * (-1) ** m * Y.imag
    else:
        return Y.real


if __name__ == "__main__":
    solver = HeatSphereFV(level=3)
    print(f"nivel 3: V={solver.V}, area total = {solver.total_area:.6f} (esperado {4*np.pi:.6f})")

    # teste rapido: condicao inicial = harmonico esferico l=3,m=2; f=0
    l, m = 3, 2
    u0 = real_spherical_harmonic(l, m, solver.mesh.vertices)
    dt = 0.01
    hist = solver.solve(u0, dt, n_steps=50, f=0.0)
    t_final, u_final = hist[-1]
    u_exact = np.exp(-l * (l + 1) * t_final) * u0
    err = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
    print(f"t={t_final:.3f}  erro relativo (l2) vs solucao exata = {err:.6e}")
