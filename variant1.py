# -*- coding: utf-8 -*-
"""
variant1.py -- розв'язання індивідуального завдання (варіант 1) методом скінченних елементів.

  -u_x1x1 - u_x2x2 = 2   в Omega = трикутник (0,0),(1,0),(0,1)
  u = 0       на катетах   (тип 1: Діріхле, маркер 1)
  du/dnu = 0  на гіпотенузі (тип 2: Нейман,  маркер 2)

Перевірка: відбиваючи трикутник відносно гіпотенузи, отримуємо одиничний квадрат з u=0
на всіх сторонах, тому точний розв'язок -- ряд для -Delta u = 2 у квадраті.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from mesh2d import (Domain, DelaunayMesher, renumber_with_rcm, build_connectivity,
                    compute_bandwidth, plot_mesh)
from mesh2d.fem import Problem, BoundaryCondition, solve, assemble, local_matrices


def build_domain():
    dom = Domain()
    pts = [[0, 0], [1, 0], [0, 1]]
    # ребра: (0,0)->(1,0) низ-катет [1], (1,0)->(0,1) гіпотенуза [2], (0,1)->(0,0) лівий катет [1]
    dom.add_boundary(pts, markers=[1, 2, 1])
    return dom


def exact_square(x, y, nmax=61):
    """-Delta u = 2 у (0,1)^2, u=0 на межі (подвійний синусний ряд)."""
    u = np.zeros_like(x, dtype=float)
    for m in range(1, nmax, 2):
        for n in range(1, nmax, 2):
            u += 2 * 16 / (np.pi ** 2 * m * n) / (np.pi ** 2 * (m * m + n * n)) \
                 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return u


def run(max_area, min_angle=30.0, rcm=True):
    mesh = DelaunayMesher(min_angle_deg=min_angle, max_area=max_area).triangulate(build_domain())
    bw0 = compute_bandwidth(build_connectivity(mesh).adjacency)
    bw1 = bw0
    if rcm:
        mesh, bw0, bw1 = renumber_with_rcm(mesh)
    prob = Problem(a11=1.0, a22=1.0, d=0.0, f=2.0,
                   bcs={1: BoundaryCondition.dirichlet(0.0), 2: BoundaryCondition.neumann(0.0)})
    q, A, F = solve(mesh, prob)
    ex = exact_square(mesh.nodes[:, 0], mesh.nodes[:, 1])
    err_max = np.abs(q - ex).max()
    return mesh, q, ex, err_max, bw0, bw1


if __name__ == "__main__":
    print("Збіжність (макс. похибка у вузлах відносно точного розв'язку):")
    prev = None
    for h in (0.02, 0.005, 0.00125, 0.0003):
        mesh, q, ex, err, bw0, bw1 = run(h)
        rate = "" if prev is None else f"   (відношення {prev/err:.2f})"
        print(f"  max_area={h:<8} вузлів={mesh.n_nodes:<6} трикутників={mesh.n_triangles:<6} "
              f"bw {bw0}->{bw1}  err={err:.2e}{rate}")
        prev = err

    mesh, q, ex, err, bw0, bw1 = run(0.002)
    print(f"\nu_max FEM = {q.max():.5f}, u_max точне = {ex.max():.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)
    plot_mesh(mesh, ax=axes[0], show_nodes=False, title=f"Сітка ({mesh.n_triangles} трик.)")
    cs = axes[1].tricontourf(tri, q, 20, cmap="viridis"); plt.colorbar(cs, ax=axes[1])
    axes[1].set_aspect("equal"); axes[1].set_title("Розв'язок МСЕ u_h")
    cs = axes[2].tricontourf(tri, np.abs(q - ex), 20, cmap="magma"); plt.colorbar(cs, ax=axes[2])
    axes[2].set_aspect("equal"); axes[2].set_title("|u_h - u_точне|")
    plt.tight_layout(); fig.savefig("variant1_result.png", dpi=110)
    print("збережено variant1_result.png")
