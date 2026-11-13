# -*- coding: utf-8 -*-
"""
fem.py  --  Завдання №3: матриці МСЕ на лінійних трикутних елементах.

Задача (слабка форма (4) з методички): знайти u з W = W_2^1(Omega) такою, що

  INT_Omega ( a11 u_x1 v_x1 + a22 u_x2 v_x2 ) dOmega + d INT_Omega u v dOmega
      + INT_Gamma (sigma/beta) u v dGamma
      = INT_Omega f v dOmega + INT_Gamma (psi/beta) v dGamma      для всіх v з W.

Дискретизація u_h = N_e q_e, де N_e = (phi_i, phi_j, phi_m) -- лінійні функції форми.
Формули (аналітичні, а не чисельне інтегрування) беруться з інтегралів (*) методички:

   INT phi_i^a phi_j^b phi_m^c dOmega = a! b! c! / (a+b+c+2)! * delta ,  delta = 2*S
   INT phi_i^a phi_j^b dGamma         = a! b!   / (a+b+1)!   * L

Локальні матриці
   K_e = (1/(2*delta)) * ( a11 * b b^T + a22 * c c^T )         (жорсткість)
   M_e = d * delta/24 * [[2,1,1],[1,2,1],[1,1,2]]               (маса / реакція)
   Q_e = delta/24 * [[2,1,1],[1,2,1],[1,1,2]] @ f_e            (права частина, f_e = значення f у вузлах)
   R_e = (sigma/beta) * L/6 * [[2,1],[1,2]]                     (границя, ліва частина)
   P_e = (psi/beta)   * L/2 * [1,1]^T                           (границя, права частина)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .triangulation import Mesh

# Значення, якими за методичкою "імітуються" типи граничних умов
EPS, BIG = 1e-5, 1e5


@dataclass
class BoundaryCondition:
    """Умова beta*Nu + sigma*u = psi на ділянці межі з певним маркером."""
    beta: float
    sigma: float
    psi: float = 0.0

    @staticmethod
    def dirichlet(value: float = 0.0) -> "BoundaryCondition":
        # u = value:  beta -> 0, sigma -> inf, psi = sigma*value  (методичка: 1e-5, 1e5)
        return BoundaryCondition(beta=EPS, sigma=BIG, psi=BIG * value)

    @staticmethod
    def neumann(flux: float = 0.0) -> "BoundaryCondition":
        # Nu = flux:  sigma -> 0, beta -> inf  (методичка: sigma=1e-5, beta=1e5)
        return BoundaryCondition(beta=BIG, sigma=EPS, psi=BIG * flux)


@dataclass
class Problem:
    """-div(A grad u) + d u = f  в Omega,  BC за маркерами ребер межі."""
    a11: float
    a22: float
    d: float
    f: object                      # число або функція f(x1, x2)
    bcs: dict = field(default_factory=dict)   # {marker: BoundaryCondition}

    def f_at(self, pts: np.ndarray) -> np.ndarray:
        if callable(self.f):
            return np.array([self.f(x, y) for x, y in pts], dtype=float)
        return np.full(len(pts), float(self.f))


# ----------------------------------------------------------------------------
# Локальні матриці
# ----------------------------------------------------------------------------
def triangle_geometry(xy: np.ndarray):
    """
    xy -- (3,2) координати вершин i, j, m.  Повертає (b, c, delta):
      b_i = x2^(j) - x2^(m),  c_i = x1^(m) - x1^(j)  (і циклічно j, m)
      delta = 2*S  (за модулем).
    phi_k = (a_k + b_k x1 + c_k x2)/delta  =>  grad phi_k = (b_k, c_k)/delta.
    """
    (xi, yi), (xj, yj), (xm, ym) = xy
    b = np.array([yj - ym, ym - yi, yi - yj])
    c = np.array([xm - xj, xi - xm, xj - xi])
    delta_signed = (xj - xi) * (ym - yi) - (xm - xi) * (yj - yi)   # = 2*S зі знаком
    return b, c, abs(delta_signed)


_MASS = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)


def local_matrices(xy: np.ndarray, prob: Problem, f_nodes: np.ndarray):
    """K_e, M_e, Q_e на одному трикутнику."""
    b, c, delta = triangle_geometry(xy)
    K = (prob.a11 * np.outer(b, b) + prob.a22 * np.outer(c, c)) / (2.0 * delta)
    M = prob.d * delta / 24.0 * _MASS
    Q = delta / 24.0 * _MASS @ f_nodes
    return K, M, Q


def local_boundary_matrices(xy: np.ndarray, bc: BoundaryCondition):
    """R_e (2x2) та P_e (2) на граничному ребрі з вершинами xy (2,2)."""
    L = float(np.hypot(*(xy[1] - xy[0])))
    R = bc.sigma / bc.beta * L / 6.0 * np.array([[2, 1], [1, 2]], dtype=float)
    P = bc.psi / bc.beta * L / 2.0 * np.array([1.0, 1.0])
    return R, P


# ----------------------------------------------------------------------------
# Збірка глобальної системи A q = F
# ----------------------------------------------------------------------------
def assemble(mesh: Mesh, prob: Problem):
    """
    Глобальна матриця A = sum(K_e + M_e) + sum(R_e),  F = sum(Q_e) + sum(P_e).
    Локальний індекс k (0..2) трикутника t відповідає глобальному вузлу triangles[t,k]
    -- саме масив зв'язності з завдання №1 задає, КУДИ додається локальна матриця.
    """
    n = mesh.n_nodes
    A = np.zeros((n, n))
    F = np.zeros(n)
    f_all = prob.f_at(mesh.nodes)

    for tri in mesh.triangles:                       # внутрішні елементи
        K, M, Q = local_matrices(mesh.nodes[tri], prob, f_all[tri])
        A[np.ix_(tri, tri)] += K + M
        F[tri] += Q

    if mesh.edges is not None:                       # граничні елементи
        for edge, mk in zip(mesh.edges, mesh.edge_markers):
            if mk == 0 or mk not in prob.bcs:
                continue                             # внутрішня лінія / природна умова без внеску
            R, P = local_boundary_matrices(mesh.nodes[edge], prob.bcs[mk])
            A[np.ix_(edge, edge)] += R
            F[edge] += P
    return A, F


def solve(mesh: Mesh, prob: Problem):
    A, F = assemble(mesh, prob)
    return np.linalg.solve(A, F), A, F
