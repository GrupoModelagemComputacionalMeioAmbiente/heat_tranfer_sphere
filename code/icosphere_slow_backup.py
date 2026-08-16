"""
Gerador de malha icosaedral sobre a esfera e sua malha dual, seguindo a
construcao descrita no Capitulo 3 da tese de doutorado de Fabio Freitas
Ferreira (UERJ/IPRJ, 2008), "Problemas inversos sobre a esfera":

  - Nivel 0: icosaedro regular (12 vertices, 30 arestas, 20 faces).
  - Refinamento diadico: cada face triangular e subdividida em 4,
    ligando os pontos medios das arestas, projetados de volta na esfera
    de raio R:   P_novo = (P_i + P_j) / ||P_i + P_j||          (eq. 3.18)
  - Malha dual: um vertice dual por face triangular, no circuncentro
    dessa face (projetado na esfera). As celulas duais (uma por vertice
    da malha icosaedral original) sao poligonos de 5 lados (pentagonos,
    em torno dos 12 vertices originais do icosaedro) ou 6 lados
    (hexagonos, nos demais vertices), tipicos de uma malha geodesica.

Este modulo substitui o gerador original em C (nao mais executavel por
causa de bibliotecas obsoletas) por uma implementacao Python autonoma,
usando apenas NumPy, e calcula tambem os pesos geometricos (areas de
celula, comprimentos de aresta dual h_j, distancias d_j) necessarios
para a discretizacao em volumes finitos do Capitulo 4/10 da tese.
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Icosaedro regular de nivel 0
# ---------------------------------------------------------------------

def _icosahedron_level0(R=1.0):
    """Vertices e faces do icosaedro regular inscrito na esfera de raio R.

    Construcao classica via retangulo aureo (equivalente, a menos de
    rotacao rigida, a parametrizacao em coordenadas esfericas com duas
    coroas pentagonais da tese, Sec. 3.1, eqs. 3.2-3.17).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # razao aurea

    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * R

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)

    return verts, faces


@dataclass
class IcoMesh:
    """Malha icosaedral de nivel n sobre a esfera de raio R."""
    level: int
    R: float
    vertices: np.ndarray                     # (V, 3)
    faces: np.ndarray                        # (F, 3) indices em vertices
    edges: np.ndarray = None                 # (E, 2) indices em vertices

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    @property
    def n_edges(self):
        return len(self.edges)

    def theoretical_counts(self):
        """Contagens teoricas F_n=20p^2, A_n=30p^2, V_n=10p^2+2 (p=2^n)."""
        p = 2 ** self.level
        return dict(F=20 * p * p, A=30 * p * p, V=10 * p * p + 2)


def _build_edges(faces):
    edge_set = set()
    for (a, b, c) in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edge_set.add((min(i, j), max(i, j)))
    return np.array(sorted(edge_set), dtype=int)


def build_icosphere(level: int, R: float = 1.0) -> IcoMesh:
    """Constroi a malha icosaedral de nivel `level` por refinamento diadico
    (eq. 3.18 da tese): cada face e subdividida em 4, inserindo um novo
    vertice no ponto medio normalizado de cada aresta.
    """
    verts, faces = _icosahedron_level0(R=R)
    verts = [v for v in verts]
    faces = [tuple(f) for f in faces]

    for _ in range(level):
        midpoint_cache = {}

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            p = np.array(verts[i]) + np.array(verts[j])
            p = p / np.linalg.norm(p) * R
            idx = len(verts)
            verts.append(p)
            midpoint_cache[key] = idx
            return idx

        new_faces = []
        for (a, b, c) in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ])
        faces = new_faces

    vertices = np.array(verts, dtype=float)
    faces = np.array(faces, dtype=int)
    edges = _build_edges(faces)

    return IcoMesh(level=level, R=R, vertices=vertices, faces=faces, edges=edges)


# ---------------------------------------------------------------------
# Geometria esferica auxiliar
# ---------------------------------------------------------------------

def _great_circle_distance(p, q, R):
    """Distancia geodesica (arco de circulo maximo) entre dois pontos da
    esfera de raio R."""
    cos_ang = np.clip(np.dot(p, q) / (R * R), -1.0, 1.0)
    return R * np.arccos(cos_ang)


def _spherical_triangle_area(a, b, c, R):
    """Area do triangulo esferico de vertices a,b,c (na esfera de raio R),
    via formula vetorial de Van Oosterom & Strackee (robusta e sem casos
    especiais), equivalente ao excesso esferico E = area / R^2:

        tan(E/2) = |a . (b x c)| / (R^3 + R^2(a.b + b.c + c.a))
    """
    au, bu, cu = a / R, b / R, c / R
    numer = np.dot(au, np.cross(bu, cu))
    denom = 1.0 + np.dot(au, bu) + np.dot(bu, cu) + np.dot(cu, au)
    E = 2.0 * np.arctan2(numer, denom)
    return abs(E) * R * R


def _circumcenter_on_sphere(p0, p1, p2, R):
    """Circuncentro do triangulo (p0,p1,p2), projetado na esfera de raio R
    (ponto equidistante geodesicamente dos tres vertices, equivalente a
    formula fechada eqs. 3.19-3.22 da tese via mediatrizes no plano
    tangente): direcao normal ao plano euclidiano definido pelos 3
    vertices, com orientacao consistente com a normal externa.
    """
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n)
    if np.dot(n, p0 + p1 + p2) < 0:
        n = -n
    return n * R


# ---------------------------------------------------------------------
# Malha dual
# ---------------------------------------------------------------------

@dataclass
class DualMesh:
    """Malha dual de uma IcoMesh: um vertice dual por face triangular do
    primal; uma celula poligonal (pentagono/hexagono) por vertice primal.
    """
    R: float
    dual_vertices: np.ndarray      # (F, 3) circuncentros das faces primais
    primal: IcoMesh
    cell_faces: list               # cell_faces[i] = indices (em dual_vertices) da celula i, em ordem ciclica
    neighbors: list                # neighbors[i] = indices dos vertices primais vizinhos de i, mesma ordem ciclica
    cell_area: np.ndarray          # (V,) area de cada celula dual
    h: list                        # h[i][k] = comprimento geodesico da aresta dual entre cell_faces[i][k] e [k+1]
    d: list                        # d[i][k] = distancia geodesica de i ao "bissetor" (ponto medio) da aresta primal i-neighbors[i][k]


def build_dual_mesh(mesh: IcoMesh) -> DualMesh:
    V = mesh.n_vertices
    verts = mesh.vertices
    faces = mesh.faces
    R = mesh.R

    dual_vertices = np.array([
        _circumcenter_on_sphere(verts[a], verts[b], verts[c], R)
        for (a, b, c) in faces
    ])

    vert_to_faces = [[] for _ in range(V)]
    for fidx, (a, b, c) in enumerate(faces):
        vert_to_faces[a].append(fidx)
        vert_to_faces[b].append(fidx)
        vert_to_faces[c].append(fidx)

    cell_faces = []
    neighbors = []
    cell_area = np.zeros(V)
    h_list = []
    d_list = []

    for i in range(V):
        incident = vert_to_faces[i]
        center = verts[i]
        normal = center / np.linalg.norm(center)

        # base ortonormal do plano tangente em i, para ordenar ciclicamente
        ref = dual_vertices[incident[0]] - center
        ref = ref - np.dot(ref, normal) * normal
        t1 = ref / np.linalg.norm(ref)
        t2 = np.cross(normal, t1)

        def angle(fidx):
            v = dual_vertices[fidx] - center
            v = v - np.dot(v, normal) * normal
            return np.arctan2(np.dot(v, t2), np.dot(v, t1))

        ordered = sorted(incident, key=angle)
        m = len(ordered)

        # vizinho primal associado a cada aresta dual consecutiva:
        # o vertice (!= i) comum as duas faces triangulares consecutivas
        ordered_neighbors = []
        for k in range(m):
            f_a = set(faces[ordered[k]])
            f_b = set(faces[ordered[(k + 1) % m]])
            common = (f_a & f_b) - {i}
            j = common.pop()
            ordered_neighbors.append(j)

        # area da celula: soma de triangulos esfericos (i, b_k, b_{k+1})
        area = 0.0
        h_i = np.zeros(m)
        d_i = np.zeros(m)
        for k in range(m):
            bk = dual_vertices[ordered[k]]
            bk1 = dual_vertices[ordered[(k + 1) % m]]
            area += _spherical_triangle_area(center, bk, bk1, R)
            h_i[k] = _great_circle_distance(bk, bk1, R)
            # distancia do centro i ao "bissetor" (ponto medio geodesico)
            # da aresta primal i-j; para a malha icosaedral quase-regular
            # aproxima-se muito bem pela metade da distancia geodesica i-j
            j = ordered_neighbors[k]
            d_i[k] = 0.5 * _great_circle_distance(center, verts[j], R)

        cell_faces.append(ordered)
        neighbors.append(ordered_neighbors)
        cell_area[i] = area
        h_list.append(h_i)
        d_list.append(d_i)

    return DualMesh(R=R, dual_vertices=dual_vertices, primal=mesh,
                     cell_faces=cell_faces, neighbors=neighbors,
                     cell_area=cell_area, h=h_list, d=d_list)


if __name__ == "__main__":
    for n in range(0, 5):
        m = build_icosphere(n)
        tc = m.theoretical_counts()
        assert m.n_vertices == tc["V"], (m.n_vertices, tc["V"])
        assert m.n_faces == tc["F"], (m.n_faces, tc["F"])
        assert m.n_edges == tc["A"], (m.n_edges, tc["A"])
        dm = build_dual_mesh(m)
        total_area = dm.cell_area.sum()
        print(f"nivel {n}: V={m.n_vertices:6d} F={m.n_faces:6d} A={m.n_edges:6d} "
              f"  area_total_dual={total_area:.6f} (esperado {4*np.pi:.6f})")
