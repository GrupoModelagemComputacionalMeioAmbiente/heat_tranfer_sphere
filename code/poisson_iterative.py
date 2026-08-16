"""
Reimplementacao em Python do algoritmo iterativo ORIGINAL (Scilab,
'CODIGO_DA_ESFERA', F. F. Ferreira, 2011) que resolve a equacao de
POISSON sobre a esfera com o metodo local celula-a-celula parametrizado
por beta (condicao de Robin) -- Caps. 4-7 da tese.

As formulas replicam exatamente PRINCIPAL.sce / CALCULAR_ALPHA.sci /
CALCULAR_GAMA.sci / CALCULAR_U_CENTRAL.sci / CALCULAR_V.sci /
CALCULAR_U.sci / CALCULAR_CRITERIO.sci / CALCULAR_MEDIA.sci:

    alpha(i)   = 1 / sum_j [ h(i,j) / (d(i,j)+beta) ]
    gama(i,j)  = h(i,j) / (d(i,j)+beta)
    uc(i)      = alpha(i) * ( f(i)*area(i) + sum_j gama(i,j)*(uv(fv,acv) + beta*vv(fv,acv)) )
    v(i,j)     = (1/(d(i,j)+beta)) * ( uc(i) - uv(fv,acv) - beta*vv(fv,acv) )
    u(i,j)     = uv(fv,acv) + beta*( v(i,j) + vv(fv,acv) )
    criterio   = max_i |uc(i) - ucv(i)|      (norma L-infinito, igual ao codigo original)

onde (fv,acv) = (vizinho global de i pela aresta j, indice LOCAL dessa
mesma aresta visto do vizinho) -- a reciprocidade de indices locais e
calculada uma vez em `_build_geometry_arrays` (equivalente a 'acviz' do
codigo Scilab original). E' uma iteracao de JACOBI (nao Gauss-Seidel):
uc, u, v de TODAS as celulas sao calculados a partir dos valores antigos
(uv,vv) e so trocados ao final de cada sweep -- mesma ordem do
PRINCIPAL.sce original (TROCAR_VELHOR_NOVO chamado uma vez por laco).

DIFERENCA em relacao ao codigo original: aqui a geometria (h, d, area,
vizinhos) vem da malha gerada em icosphere.py, com circuncentros
projetados exatamente sobre a esfera (geometria esferica genuina). Ao
comparar os arquivos precomputados do codigo original (pasta
DADOS_ENTRADA) com esta geometria, encontramos duas diferencas:

  1. Os pontos duais do codigo original (COORDENADA_DUAL/cdual*.txt) tem
     norma ~0.7946 (nao 1.0): sao os circuncentros EUCLIDIANOS (planos)
     de cada face triangular, sem projetar de volta sobre a esfera --
     "d" e "h" no codigo original sao distancias em CORDA (reta 3D), nao
     arcos geodesicos. Conferimos que o valor de d no nivel 0 bate
     exatamente (0.546533) com a corda vertice-ate-ponto-medio-do-arco:
     aproximacao de primeira ordem, razoavel para malhas finas, mas com
     erro visivel no nivel 0 (mais grosseiro).
  2. A AREA precomputada (AREA/area*.txt) NAO respeita a simetria icosaedral
     esperada: os 12 pentagonos do nivel 0 (congruentes por simetria)
     tem areas variando ate 40% entre si, e a soma total das areas em
     varios niveis testados (n=0,2,4) fica entre 22 e 29 -- nunca fecha
     em 4*pi=12.566 (area real da esfera unitaria), e o erro NAO diminui
     com o refinamento. Isso aponta para um erro no calculo de area do
     codigo original, nao apenas uma aproximacao numerica.

Por isso este modulo usa a geometria de icosphere.py (validada: soma de
areas = 4*pi exatamente, ate a precisao de maquina) em vez de reler os
arquivos DADOS_ENTRADA -- preservando o ALGORITMO original (que e
matematicamente correto e e o que realmente importa para o estudo do
parametro beta), mas com uma geometria de malha sem os problemas acima.
"""

import numpy as np
from icosphere import build_icosphere, build_dual_mesh


def _build_geometry_arrays(dm, V):
    """Converte as listas por vertice (h,d,neighbors) da malha dual
    (formato ordenado, build_dual_mesh) em arrays densos (V,6), com
    padding para os 12 vertices pentagonais (m=5), e calcula o indice
    local reciproco (equivalente a 'acviz' do codigo Scilab original)."""
    m_of = np.array([len(dm.neighbors[i]) for i in range(V)])
    nbr = np.zeros((V, 6), dtype=int)
    h = np.zeros((V, 6))
    d = np.zeros((V, 6))
    valid = np.zeros((V, 6), dtype=bool)
    for i in range(V):
        m = m_of[i]
        nbr[i, :m] = dm.neighbors[i]
        h[i, :m] = dm.h[i]
        # dm.d[i][k] e' a distancia LOCAL do centro da celula i ate o
        # bissetor da aresta i-j (formulacao mista/hibrida: v(i,j) usa a
        # distancia PROPRIA de cada lado, nao a distancia completa i-j
        # -- confirmado empiricamente: usar a distancia completa aqui
        # quebra a correspondencia com uma resolucao direta esparsa do
        # mesmo sistema, tentativa de "correcao" descartada).
        d[i, :m] = dm.d[i]
        valid[i, :m] = True
        nbr[i, m:] = i  # padding: aponta para si mesmo (peso sera zero)

    pos_lookup = {}
    for i in range(V):
        for k in range(m_of[i]):
            pos_lookup[(i, nbr[i, k])] = k

    recip = np.zeros((V, 6), dtype=int)
    for i in range(V):
        for k in range(m_of[i]):
            j = nbr[i, k]
            recip[i, k] = pos_lookup[(j, i)]

    return nbr, h, d, valid, recip, m_of


class PoissonIterativeSolver:
    """Algoritmo iterativo local (Robin/beta) para -Delta_S u = f, com
    normalizacao de media nula, replicando exatamente o codigo Scilab
    original de F. F. Ferreira (PRINCIPAL.sce, 2011), com geometria
    esferica corrigida (ver docstring do modulo)."""

    def __init__(self, level: int, R: float = 1.0):
        self.level = level
        self.mesh = build_icosphere(level, R=R)
        self.dm = build_dual_mesh(self.mesh)
        self.V = self.mesh.n_vertices
        self.nbr, self.h, self.d, self.valid, self.recip, self.m_of = \
            _build_geometry_arrays(self.dm, self.V)
        self.area = self.dm.cell_area.copy()
        self.A_total = self.area.sum()

    def solve(self, f, beta, tol=1e-3, max_iter=10000, impose_zero_mean=True, verbose=False):
        V = self.V
        nbr, h, d, valid, recip = self.nbr, self.h, self.d, self.valid, self.recip

        gama = np.where(valid, h / (d + beta), 0.0)   # (V,6)
        alpha = 1.0 / gama.sum(axis=1)                 # (V,)

        uc = np.zeros(V)
        ucv = np.zeros(V)
        uv = np.zeros((V, 6))
        vv = np.zeros((V, 6))

        criterio = 1.0
        n_iter = 0
        while criterio > tol and n_iter < max_iter:
            uv_r = uv[nbr, recip]   # (V,6): u na celula vizinha, indice local reciproco
            vv_r = vv[nbr, recip]

            termo = np.sum(gama * (uv_r + beta * vv_r), axis=1)
            uc = alpha * (f * self.area + termo)

            uc_col = uc[:, None]
            v_new = (1.0 / (d + beta)) * (uc_col - uv_r - beta * vv_r)
            u_new = uv_r + beta * (v_new + vv_r)

            criterio = np.max(np.abs(uc - ucv))

            ucv = uc.copy()
            uv = u_new
            vv = v_new
            n_iter += 1

        if verbose:
            print(f"beta={beta:.4g}  iteracoes={n_iter}  criterio_final={criterio:.3e}")

        if impose_zero_mean:
            media = np.dot(uc, self.area) / self.A_total
            uc = uc - media

        return uc, n_iter


# ---------------------------------------------------------------------
# Caso de teste da tese (Cap. 7, eqs. 7.2-7.3): solucao analitica
# u(theta,phi) = sen^3(phi) cos(theta),
# termo fonte f = 4 sen(phi) cos(theta) (1 - 3 cos^2(phi))
# ---------------------------------------------------------------------

def analytic_solution(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.linalg.norm(vertices, axis=1)
    phi = np.arccos(np.clip(z / r, -1, 1))
    theta = np.arctan2(y, x)
    return np.sin(phi) ** 3 * np.cos(theta)


def source_term(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.linalg.norm(vertices, axis=1)
    phi = np.arccos(np.clip(z / r, -1, 1))
    theta = np.arctan2(y, x)
    return 4.0 * np.sin(phi) * np.cos(theta) * (1.0 - 3.0 * np.cos(phi) ** 2)


if __name__ == "__main__":
    solver = PoissonIterativeSolver(level=2)
    f = source_term(solver.mesh.vertices)
    u_exact = analytic_solution(solver.mesh.vertices)

    uc, n_iter = solver.solve(f, beta=2.4, tol=1e-6, impose_zero_mean=True, verbose=True)
    err = np.abs(uc - u_exact)
    print(f"nivel 2: erro_max={err.max():.5f}  erro_l2={np.sqrt(np.mean(err**2)):.5f}")
