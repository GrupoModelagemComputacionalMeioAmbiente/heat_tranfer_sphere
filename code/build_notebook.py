"""Monta um notebook .ipynb autocontido (pronto para Google Colab) a partir
dos modulos .py do projeto, com celulas de markdown explicando cada parte."""

import json
import re

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def read(fname, strip_main=True, strip_imports_from=()):
    with open(fname) as fh:
        txt = fh.read()
    if strip_main:
        txt = re.split(r'\nif __name__ == "__main__":', txt)[0].rstrip() + "\n"
    for mod in strip_imports_from:
        txt = re.sub(rf'^from {mod} import .*$\n?', '', txt, flags=re.M)
    return txt


cells = []

cells.append(md_cell(
"""# Solução da equação do calor sobre a esfera — malha icosaedral

Notebook autocontido com todo o código do projeto: geração da malha icosaedral e sua malha dual, visualização, solver da equação do calor (volumes finitos mistos + Crank–Nicolson), validação com solução analítica (harmônicos esféricos) e reprodução dos Casos 1 e 2.

**Como usar no Google Colab:** File → Save a copy in Drive (opcional, para salvar suas edições), depois Runtime → Run all. Todas as dependências (NumPy, SciPy, Matplotlib) já vêm instaladas no Colab por padrão.

Autor: Fábio Freitas Ferreira. Código gerado a partir da formulação matemática da tese *Problemas inversos sobre a esfera* (UERJ/IPRJ, 2008) e do artigo de congresso associado."""
))

cells.append(code_cell(
"""import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from dataclasses import dataclass, field
import os

plt.rcParams.update({"font.size": 11, "font.family": "serif"})
os.makedirs("figures", exist_ok=True)
print("Ambiente pronto.")"""
))

# ---------------------------------------------------------------
cells.append(md_cell(
"""## Parte 1 — Malha icosaedral e malha dual

Gera a malha icosaedral por refinamento diádico sucessivo de um icosaedro regular (nível 0: 12 vértices, 30 arestas, 20 faces), e a malha dual correspondente (um vértice dual por face triangular, no circuncentro; células pentagonais/hexagonais em torno de cada vértice primal)."""
))
cells.append(code_cell(read("icosphere.py")))

cells.append(code_cell(
"""# Teste rápido: reproduz a Tabela 3.1 da tese e confere a área total da malha dual (deve ser 4*pi)
for n in range(0, 5):
    m = build_icosphere(n)
    tc = m.theoretical_counts()
    assert m.n_vertices == tc["V"] and m.n_faces == tc["F"] and m.n_edges == tc["A"]
    dm = build_dual_mesh(m)
    print(f"nivel {n}: V={m.n_vertices:6d} F={m.n_faces:6d} A={m.n_edges:6d}  "
          f"area_total_dual={dm.cell_area.sum():.6f} (esperado {4*np.pi:.6f})")"""
))

# ---------------------------------------------------------------
cells.append(md_cell("## Parte 2 — Visualização da malha"))
cells.append(code_cell(read("plot_mesh.py", strip_imports_from=("icosphere",))))

cells.append(code_cell(
"""fig = plt.figure(figsize=(16, 4.3))
for k, n in enumerate([0, 1, 2, 3]):
    ax = fig.add_subplot(1, 4, k + 1, projection="3d")
    plot_primal_mesh(n, ax=ax)
    ax.set_title(f"nível $n={n}$")
fig.tight_layout()
plt.show()"""
))

cells.append(code_cell(
"""fig = plt.figure(figsize=(12, 4.3))
ax1 = fig.add_subplot(1, 3, 1, projection="3d"); plot_primal_mesh(2, ax=ax1); ax1.set_title("Malha primal")
ax2 = fig.add_subplot(1, 3, 2, projection="3d"); plot_dual_mesh(2, ax=ax2); ax2.set_title("Malha dual")
ax3 = fig.add_subplot(1, 3, 3, projection="3d"); plot_primal_and_dual_overlay(2, ax=ax3); ax3.set_title("Sobreposição")
fig.tight_layout()
plt.show()"""
))

# ---------------------------------------------------------------
cells.append(md_cell(
"""## Parte 3 — Solver da equação do calor

Monta e resolve, a cada passo de tempo, o sistema linear esparso $(M + \\tfrac{\\Delta t}{2}L)\\,u^{r+1} = (M - \\tfrac{\\Delta t}{2}L)\\,u^r + \\Delta t\\, M f$, equivalente ao esquema de volumes finitos mistos com Crank–Nicolson da tese (ver artigo, Seção 4)."""
))
cells.append(code_cell(read("heat_solver.py", strip_imports_from=("icosphere",))))

cells.append(code_cell(
"""# Teste rápido: condição inicial = harmônico esférico, comparação com a solução exata
solver = HeatSphereFV(level=3)
l, m = 3, 2
u0 = real_spherical_harmonic(l, m, solver.mesh.vertices)
dt = 0.01
hist = solver.solve(u0, dt, n_steps=50, f=0.0)
t_final, u_final = hist[-1]
u_exact = np.exp(-l * (l + 1) * t_final) * u0
err = np.linalg.norm(u_final - u_exact) / np.linalg.norm(u_exact)
print(f"nivel 3, V={solver.V}: t={t_final:.3f}  erro relativo (l2) vs solucao exata = {err:.6e}")"""
))

# ---------------------------------------------------------------
cells.append(md_cell(
"""## Parte 4 — Estudo de convergência (espacial e temporal)

Usa a solução exata via harmônicos esféricos, $u(x,t)=e^{-l(l+1)t}Y_l^m(x)$, para calcular o erro numérico real e estimar a ordem de convergência."""
))
cells.append(code_cell(read("run_convergence_study.py", strip_imports_from=("heat_solver",))
                        .replace('outfile_csv="../notes/data/convergencia_espacial.csv",\n        outfile_fig="../figures/fig_convergencia_espacial.pdf")',
                                  'outfile_csv="figures/convergencia_espacial.csv",\n        outfile_fig="figures/fig_convergencia_espacial.pdf")')
                        .replace('outfile_csv="../notes/data/convergencia_temporal.csv",\n        outfile_fig="../figures/fig_convergencia_temporal.pdf")',
                                  'outfile_csv="figures/convergencia_temporal.csv",\n        outfile_fig="figures/fig_convergencia_temporal.pdf")')
))

cells.append(code_cell(
"""print("=== Convergencia espacial (pode levar ~1 min) ===")
rows_s, orders_s = spatial_convergence(
    levels=[1, 2, 3, 4, 5], dt=1e-3, t_final=0.1,
    outfile_csv="figures/convergencia_espacial.csv",
    outfile_fig="figures/fig_convergencia_espacial.pdf")
plt.show()

print("\\n=== Convergencia temporal (pode levar alguns minutos, malha nivel 6) ===")
rows_t, orders_t = temporal_convergence(
    level=6, dts=[0.1, 0.05, 0.025, 0.0125], t_final=0.4,
    outfile_csv="figures/convergencia_temporal.csv",
    outfile_fig="figures/fig_convergencia_temporal.pdf")
plt.show()

print("\\nordem espacial media:", np.mean(orders_s))
print("ordem temporal media:", np.mean(orders_t))"""
))

# ---------------------------------------------------------------
cells.append(md_cell(
"""## Parte 5 — Casos 1 e 2 (reprodução dos testes da tese)

Caso 1: difusão sem fonte a partir de uma condição inicial concentrada num hemisfério. Caso 2: mesma condição inicial, com fonte de calor constante no hemisfério inicialmente mais quente."""
))
cells.append(code_cell(read("run_case_studies.py", strip_imports_from=("heat_solver",))))

cells.append(code_cell(
"""levels = [0, 1, 2, 3, 4, 5, 6]
dts = [1.0, 0.1, 0.01]
t_final = 4.0

print("=== Caso 1 (f=0) ===")
res1 = run_case(levels, dts, t_final, source_fn=None, label="Caso1")
table_mean_conservation(res1, levels, dt_ref=0.1, t_final=t_final,
                         outfile="figures/caso1_conservacao_malha.csv")
fig_mesh_level_convergence(res1, levels, dt_ref=0.1,
                            outfile="figures/fig_caso1_convergencia_malha.pdf")
plt.show()"""
))

cells.append(code_cell(
"""fig_snapshots(res1, level_ref=3, dt_ref=0.1, times_idx=[0, 10, 20, 40],
              outfile="figures/fig_caso1_snapshots.pdf")
plt.show()"""
))

cells.append(code_cell(
"""print("=== Caso 2 (fonte constante) ===")
res2 = run_case(levels, dts, t_final, source_fn=source_case2, label="Caso2")
table_mean_conservation(res2, levels, dt_ref=0.1, t_final=t_final,
                         outfile="figures/caso2_media_malha.csv")
fig_mesh_level_convergence(res2, levels, dt_ref=0.1,
                            outfile="figures/fig_caso2_convergencia_malha.pdf")
plt.show()"""
))

cells.append(code_cell(
"""fig_snapshots(res2, level_ref=3, dt_ref=0.1, times_idx=[0, 10, 20, 40],
              outfile="figures/fig_caso2_snapshots.pdf", source_fn=source_case2)
plt.show()

fig_source_case2(outfile="figures/fig_fonte_caso2.pdf")
plt.show()"""
))

cells.append(md_cell(
"""## Parte 5.1 — Caso 3: fonte harmônica, verificação não-homogênea e estado estacionário

Fonte de calor fixa $f=Y_l^m$ (harmônico esférico, média nula). Como $Y_l^m$ é autofunção do laplaciano, $u_t=\\Delta u + Y_l^m$ com $u(x,0)=0$ tem solução FECHADA: $u(x,t)=\\frac{Y_l^m}{l(l+1)}(1-e^{-l(l+1)t})$, que converge ao estado estacionário $u_\\infty=Y_l^m/l(l+1)$ (a solução do problema de Poisson associado). Isso estende a verificação da Parte 3 (que só cobre $f=0$) ao caso não-homogêneo, e mostra a convergência ao equilíbrio. Também inclui convergência espacial, convergência temporal (refinamento de $\\Delta t$ com malha fixa, comparando sempre com a solução exata) e um retrato bonito da evolução sobre a esfera."""
))
cells.append(code_cell(read("run_case3_steady.py", strip_imports_from=("heat_solver",))
                        .replace("../figures/", "figures/")
                        .replace("../notes/data/", "figures/")
                        + "\nplt.show()\n"))

cells.append(md_cell(
"""## Parte 5.2 — Caso 4: efeito das singularidades pentagonais (*grid imprinting*)

Os 12 vértices pentagonais da malha (vértices 0-11, os do icosaedro original) são defeitos topológicos permanentes — mesmo no nível de refinamento mais fino, eles continuam tendo 5 vizinhos em vez de 6. Medimos o erro de truncamento LOCAL do operador discreto (comparando com o laplaciano exato de um harmônico esférico) nos pentágonos vs. nos hexágonos "genéricos", por nível de malha. Achado: o erro típico converge normalmente (~O(h²)) até bem perto dos pentágonos, mas o erro MÁXIMO entre os vértices hexagonais — sempre localizado exatamente nos vizinhos imediatos de um pentágono — não diminui com o refinamento. Conecta com a literatura de *grid imprinting* (Peixoto & Barros, 2013), já citada no artigo."""
))
cells.append(code_cell(read("run_case4_imprinting.py", strip_imports_from=("icosphere", "heat_solver"))
                        .replace("../figures/", "figures/")
                        .replace("../notes/data/", "figures/")
                        + "\nplt.show()\n"))

cells.append(md_cell(
"""## Parte 6 — Algoritmo iterativo original (com β) e a otimização de β por nível

Esta parte reimplementa em Python o algoritmo **exato** do seu código Scilab original (`CODIGO_DA_ESFERA/PRINCIPAL.sce` e módulos `.sci`), que resolve a equação de **Poisson** (não a do calor) pelo método iterativo local com o parâmetro β da condição de Robin — o mesmo β que você provou ser preciso ser maior que zero.

**Importante — o que encontramos comparando com os dados originais (pasta `DADOS_ENTRADA`):**

1. Os pontos duais do código original (`COORDENADA_DUAL/cdual*.txt`) têm norma ≈0,7946 (não 1,0) — ou seja, são os circuncentros **euclidianos** (planos) de cada triângulo, sem projetar de volta na esfera. As distâncias `h` e `d` do código original são **cordas retas em 3D**, não arcos geodésicos. Conferimos: o valor de `d` no nível 0 bate exatamente (0,546533) com a corda do vértice até o ponto médio do arco — uma aproximação razoável para malha fina, mas com erro visível no nível 0.
2. A **área** pré-computada (`AREA/area*.txt`) **não respeita a simetria icosaedral** esperada: os 12 pentágonos do nível 0 (que são congruentes por simetria) têm áreas variando até 40% entre si, e a soma total das áreas (em vários níveis testados) nunca fecha em $4\\pi$ — um erro que não diminui com o refinamento, indicando um **bug no cálculo de área** do código original, não apenas uma aproximação numérica.

Por isso, abaixo usamos o **algoritmo exato** do seu código (as mesmas fórmulas de α, γ, u central, v, u, critério de parada) mas com a **geometria correta** (gerada por `icosphere.py`, cuja soma de áreas fecha em $4\\pi$ à precisão de máquina)."""
))
cells.append(code_cell(read("poisson_iterative.py", strip_imports_from=("icosphere",))))

cells.append(code_cell(
"""# Teste rápido: caso de teste do Cap. 7 da tese (solução exata sen^3(phi)cos(theta))
solver = PoissonIterativeSolver(level=2)
f = source_term(solver.mesh.vertices)
u_exact = analytic_solution(solver.mesh.vertices)
uc, n_iter = solver.solve(f, beta=2.4, tol=1e-6, impose_zero_mean=True, verbose=True)
err = np.abs(uc - u_exact)
print(f"nivel 2: erro_max={err.max():.5f}  erro_l2={np.sqrt(np.mean(err**2)):.5f}")"""
))

cells.append(code_cell(read("run_beta_sweep.py", strip_imports_from=("poisson_iterative",))
                        .replace('OUT_FIG = "../figures/fig_beta_otimo_vs_nivel.pdf"', 'OUT_FIG = "figures/fig_beta_otimo_vs_nivel.pdf"')
                        .replace('OUT_CSV = "../notes/data/beta_otimo_vs_nivel.csv"', 'OUT_CSV = "figures/beta_otimo_vs_nivel.csv"')
                        + "\nplt.show()\n"
))

cells.append(md_cell(
"""**O que encontramos na varredura de β:** diferente da Tabela 7.1/Figura 7.2 da tese (onde β ótimo crescia com o nível — 0,73; 1,2; 2,4; 4,9; 9,9; 18,6), aqui, com a geometria corrigida, β ótimo **converge para um platô** (~0,52) a partir do nível 2, em vez de continuar crescendo. Isso provavelmente reflete a diferença de geometria (área/distâncias corrigidas mudam a relação entre β e a taxa de convergência). Vale investigar mais a fundo antes de basear o problema inverso nisso — posso aprofundar se você quiser."""
))

cells.append(md_cell(
"""## Parte 7 — Algoritmo iterativo original (com β) para a equação do CALOR

Diferente do problema de Poisson (Parte 6), aqui **não existe código-fonte original** para conferir: o Scilab encontrado resolve só o problema estacionário. Reimplementamos o algoritmo iterativo local da equação do calor (Caps. 10-11 da tese, com a variável de fluxo auxiliar `w` nas arestas) diretamente das fórmulas do texto.

**Bug encontrado e corrigido durante a validação:** a primeira versão zerava o fluxo inicial `w⁰` a cada simulação, o que só é correto se a condição inicial `u₀` for constante — para uma condição inicial genérica, isso introduz um erro de ordem 1 já no primeiro passo de Crank-Nicolson (a tese exige `v⁰ = -∇u₀`, não repouso). Corrigido, o algoritmo iterativo passou a reproduzir a solução do sistema linear direto (`HeatSphereFV`, Parte 3) em precisão numérica — ver validação cruzada abaixo."""
))
cells.append(code_cell(read("heat_iterative.py", strip_imports_from=("icosphere", "heat_solver"))))

cells.append(code_cell(
"""# Validacao cruzada: algoritmo iterativo local vs. solver direto esparso,
# condicao inicial = harmonico esferico exato, varios niveis de malha
# (HeatSphereFV e real_spherical_harmonic ja foram definidos na Parte 3 acima)

for level in [1, 2, 3, 4]:
    dt = 0.05
    l, m = 3, 2
    direct = HeatSphereFV(level=level)
    u0 = real_spherical_harmonic(l, m, direct.mesh.vertices)
    hist_direct = direct.solve(u0, dt, n_steps=10, f=0.0)
    u_direct_final = hist_direct[-1][1]

    it = HeatIterativeSolver(level=level)
    hist_iter, iters = it.solve(u0, dt, n_steps=10, f=0.0, beta=3.0, tol=1e-10, track_iters=True)
    u_iter_final = hist_iter[-1][1]
    diff = np.max(np.abs(u_direct_final - u_iter_final))
    rel = diff / np.max(np.abs(u_direct_final))
    print(f"nivel {level}: V={direct.V:5d}  diff={diff:.3e}  rel={rel:.3e}  max_iters_por_passo={max(iters)}")"""
))

cells.append(code_cell(read("run_beta_sweep_heat.py", strip_imports_from=("heat_solver", "heat_iterative"))
                        .replace('OUT_FIG = "../figures/fig_beta_otimo_vs_nivel_calor.pdf"', 'OUT_FIG = "figures/fig_beta_otimo_vs_nivel_calor.pdf"')
                        .replace('OUT_CSV = "../notes/data/beta_otimo_vs_nivel_calor.csv"', 'OUT_CSV = "figures/beta_otimo_vs_nivel_calor.csv"')
                        + "\nplt.show()\n"
))

cells.append(md_cell(
"""**O que encontramos:** diferente de Poisson (onde β ótimo era da ordem de 0,5-1), aqui o β que minimiza o número de iterações **cresce** com o nível de malha e fica numa escala bem maior (dezenas a centenas). Isso é consistente com a fórmula ξⱼ=cⱼ/(1+cⱼβ): para β muito grande, ξⱼ→0, a célula "descola" das vizinhas e a convergência piora de novo (por isso existe um mínimo interior, não um crescimento sem limite). Este é um resultado preliminar (um único passo de tempo, condição inicial específica) — bom ponto de partida caso você queira aprofundar antes de usar β no problema inverso."""
))

cells.append(md_cell(
"""---
Todas as figuras geradas ficam salvas na pasta `figures/` deste ambiente (no Colab: ícone de pasta na barra lateral esquerda → `figures/`)."""
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "name": "heat_sphere_icosahedral.ipynb"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("../heat_sphere_icosahedral.ipynb", "w") as fh:
    json.dump(notebook, fh, indent=1)

print("notebook criado")
