"""
Algoritmo iterativo local (celula-a-celula, parametrizado por beta) para
a equacao do CALOR sobre a esfera -- Caps. 10-11 da tese de Fabio
Freitas Ferreira (UERJ/IPRJ, 2008), eqs. (10.18)-(10.20).

Diferente do problema de Poisson (poisson_iterative.py), aqui NAO
dispomos do codigo-fonte original (o codigo Scilab encontrado resolve
apenas o problema estacionario). Este modulo implementa a formula
exatamente como extraida do texto da tese, sem um ground-truth de
codigo para conferir -- ver ressalva na Secao correspondente do artigo.

Formulas (thesis eqs. 10.18-10.20), por passo de tempo r->r+1, indice
de iteracao k:

    c_j    = h_j * dt / (2*d_j)
    xi_j   = c_j / (1 + c_j*beta)

    w_j^{k+1} = xi_j * ( u0^{k+1} - u_til_j^{k} - beta*w_til_j^{k} )
    u0^{k+1}  = [ f_bar + sum_l xi_l*(u_til_l^k + beta*w_til_l^k) ]
                / [ A_i + sum_l xi_l ]
    u_j^{k+1} = beta*w_j^{k+1} + u_til_j^{k} + beta*w_til_j^{k}

    f_bar = A_i*(u0^r + dt*f0) - sum_l w_l^r      (constante dentro do passo de tempo)

onde (u_til_j, w_til_j) sao os valores de u,w na celula VIZINHA, no
indice LOCAL reciproco (mesma logica de 'acviz' usada em
poisson_iterative.py).

Cada passo de tempo faz sua propria iteracao em k ate convergencia
(norma L-infinito de u0), usando como chute inicial os valores
convergidos do passo de tempo anterior.

Este modulo tambem serve para VALIDAR numericamente a equivalencia
alegada na Secao 4.3 do artigo entre este algoritmo iterativo local e a
solucao direta do sistema esparso global (heat_solver.HeatSphereFV):
ambos devem convergir para a mesma solucao a cada passo de tempo.
"""

import numpy as np
from icosphere import build_icosphere, build_dual_mesh


def _build_geometry_arrays(dm, V):
    m_of = np.array([len(dm.neighbors[i]) for i in range(V)])
    nbr = np.zeros((V, 6), dtype=int)
    h = np.zeros((V, 6))
    d = np.zeros((V, 6))
    valid = np.zeros((V, 6), dtype=bool)
    for i in range(V):
        m = m_of[i]
        nbr[i, :m] = dm.neighbors[i]
        h[i, :m] = dm.h[i]
        # dm.d[i][k] e' a distancia LOCAL do centro i ate o bissetor da
        # aresta i-j (formulacao mista/hibrida -- ver mesma ressalva em
        # poisson_iterative.py). E' o valor correto para c_j dentro da
        # iteracao local (confirmado: reproduz o solver direto quando
        # usado assim, ver _initial_flux() para a UNICA excecao onde a
        # distancia completa 2*d e' necessaria).
        d[i, :m] = dm.d[i]
        valid[i, :m] = True
        nbr[i, m:] = i

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


class HeatIterativeSolver:
    """Solver da equacao do calor por algoritmo iterativo local com beta
    (eqs. 10.18-10.20 da tese), usando a geometria de icosphere.py."""

    def __init__(self, level: int, R: float = 1.0):
        self.level = level
        self.mesh = build_icosphere(level, R=R)
        self.dm = build_dual_mesh(self.mesh)
        self.V = self.mesh.n_vertices
        self.nbr, self.h, self.d, self.valid, self.recip, self.m_of = \
            _build_geometry_arrays(self.dm, self.V)
        self.area = self.dm.cell_area.copy()

    def _step(self, u0_r, w_r, dt, f, beta, tol=1e-8, max_iter=2000):
        """Um passo de Crank-Nicolson (r -> r+1), resolvido pela
        iteracao local em k. w_r tem shape (V,6): fluxo (variavel w) na
        celula i, na aresta local k, ao final do passo anterior."""
        V = self.V
        nbr, h, d, valid, recip = self.nbr, self.h, self.d, self.valid, self.recip

        d_safe = np.where(valid, d, 1.0)                       # evita 0/0 nas posicoes de padding
        c = np.where(valid, h * dt / (2.0 * d_safe), 0.0)      # (V,6)
        xi = np.where(valid, c / (1.0 + c * beta), 0.0)        # (V,6)

        f_bar = self.area * (u0_r + dt * f) - np.sum(w_r, axis=1)  # (V,)

        u0 = u0_r.copy()
        u_edge = np.zeros((V, 6))   # u nos bissetores (chute inicial: 0, ou herdado)
        w_edge = w_r.copy()

        criterio = 1.0
        n_iter = 0
        while criterio > tol and n_iter < max_iter:
            u_til = u_edge[nbr, recip]
            w_til = w_edge[nbr, recip]

            denom = self.area + xi.sum(axis=1)
            numer = f_bar + np.sum(xi * (u_til + beta * w_til), axis=1)
            u0_new = numer / denom

            w_new = xi * (u0_new[:, None] - u_til - beta * w_til)
            u_new = beta * w_new + u_til + beta * w_til

            criterio = np.max(np.abs(u0_new - u0))

            u0 = u0_new
            u_edge = u_new
            w_edge = w_new
            n_iter += 1

        return u0, w_edge, n_iter

    def _initial_flux(self, u0_init, dt):
        """Fluxo w^0 consistente com a condicao inicial v^0=-grad_S(u0)
        (eq. 10.4 da tese), NAO zero em geral.

        Diferente de c_j (usado DENTRO da iteracao local, que opera
        sobre o valor de traco de aresta u_j -- corretamente aproximado
        pela distancia LOCAL d_j do centro i ate o bissetor), aqui em
        r=0 o campo u0 e' conhecido explicitamente em toda a malha, sem
        precisar de nenhum valor de traco intermediario. O fluxo fisico
        correto atraves da aresta i-j e' dado pela formula padrao de
        diferencas centradas com a distancia COMPLETA entre os centros
        (d_i+d~_i~ = 2*d_j, ja' que a malha e' quase-regular e o
        bissetor fica exatamente no meio): v_j^0=(u0_i-u0_vizinho)/(2 d_j).
        Usar a distancia LOCAL d_j sozinha (ou u0 do vizinho como se
        fosse o valor no bissetor) superestima o fluxo em 2x -- bug
        corrigido nesta versao, confirmado numericamente contra o
        laplaciano de grafo do solver direto (heat_solver.py) ao nivel
        da precisao de maquina.
        Usar w^0=0 (estado de repouso) e' so' correto se u0 for
        constante; para uma condicao inicial generica, zerar o fluxo
        inicial introduz um erro O(1) no primeiro passo de Crank-Nicolson
        (bug tambem corrigido nesta versao)."""
        nbr, h, d, valid = self.nbr, self.h, self.d, self.valid
        d_full_safe = np.where(valid, 2.0 * d, 1.0)
        c_full = np.where(valid, h * dt / (2.0 * d_full_safe), 0.0)
        u0_neighbor = np.where(valid, u0_init[nbr], 0.0)
        return c_full * (u0_init[:, None] - u0_neighbor)

    def solve(self, u0_init, dt, n_steps, f=None, beta=1.0, tol=1e-8, max_iter=2000,
              track_iters=False):
        V = self.V
        if f is None:
            f = np.zeros(V)
        elif np.isscalar(f):
            f = np.full(V, float(f))

        u0 = u0_init.copy()
        w = self._initial_flux(u0_init, dt)
        history = [(0.0, u0.copy())]
        iters_per_step = []
        for r in range(1, n_steps + 1):
            u0, w, n_iter = self._step(u0, w, dt, f, beta, tol=tol, max_iter=max_iter)
            history.append((r * dt, u0.copy()))
            iters_per_step.append(n_iter)

        if track_iters:
            return history, iters_per_step
        return history


if __name__ == "__main__":
    # validacao cruzada contra o solver direto (sistema esparso global)
    from heat_solver import HeatSphereFV, real_spherical_harmonic

    level = 3
    dt = 0.05
    n_steps = 10
    l, m = 3, 2

    direct = HeatSphereFV(level=level)
    u0 = real_spherical_harmonic(l, m, direct.mesh.vertices)
    hist_direct = direct.solve(u0, dt, n_steps, f=0.0)
    u_direct_final = hist_direct[-1][1]

    iterativo = HeatIterativeSolver(level=level)
    hist_iter, iters = iterativo.solve(u0, dt, n_steps, f=0.0, beta=1.0, tol=1e-10, track_iters=True)
    u_iter_final = hist_iter[-1][1]

    diff = np.max(np.abs(u_direct_final - u_iter_final))
    rel = diff / np.max(np.abs(u_direct_final))
    print(f"nivel {level}, dt={dt}, {n_steps} passos:")
    print(f"  diferenca maxima entre solver direto e iterativo local: {diff:.3e} (relativa: {rel:.3e})")
    print(f"  iteracoes por passo de tempo (beta=1.0): {iters}")
