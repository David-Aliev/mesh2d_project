# -*- coding: utf-8 -*-
"""
visualization.py
=================
Допоміжні функції побудови графіків matplotlib:
    * сама сітка (трикутники + вузли, за бажанням із нумерацією);
    * вузли межі, розфарбовані за маркером граничної умови;
    * "spy"-візуалізація портрету (профілю) матриці зв'язності --
      наочно демонструє ефект перенумерації Катхілла-Маккі.
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection

from .triangulation import Mesh
from .connectivity import sparsity_pattern

_MARKER_COLORS = ["#999999", "#e74c3c", "#3498db", "#2ecc71",
                  "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


def plot_mesh(mesh: Mesh, ax=None, show_nodes=True, show_node_ids=False,
              show_tri_ids=False, title="Сітка скінченних елементів",
              color_by_attribute=False):
    """Малює тріангуляцію: ребра елементів + (опційно) вузли й нумерацію."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)

    if color_by_attribute and mesh.triangle_attributes is not None:
        ax.tripcolor(tri, facecolors=mesh.triangle_attributes, cmap="Pastel1",
                     edgecolors="k", linewidth=0.4, alpha=0.9)
    else:
        ax.triplot(tri, color="#34495e", linewidth=0.5)

    if show_nodes:
        ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], "o", color="#2c3e50",
                 markersize=2.5, zorder=3)

    if show_node_ids and mesh.n_nodes <= 300:
        for i, (x, y) in enumerate(mesh.nodes):
            ax.annotate(str(i), (x, y), fontsize=6, color="#c0392b",
                        xytext=(2, 2), textcoords="offset points")

    if show_tri_ids and mesh.n_triangles <= 300:
        centroids = mesh.nodes[mesh.triangles].mean(axis=1)
        for i, (x, y) in enumerate(centroids):
            ax.annotate(str(i), (x, y), fontsize=6, color="#2980b9", ha="center")

    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def plot_boundary_markers(mesh: Mesh, ax=None, title="Маркування граничних умов"):
    """Розфарбовує вузли межі за їхнім маркером граничної умови."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)
    ax.triplot(tri, color="#dcdcdc", linewidth=0.4, zorder=1)

    markers = np.unique(mesh.node_markers)
    for m in markers:
        idx = np.where(mesh.node_markers == m)[0]
        if m == 0:
            continue  # внутрішні вузли не малюємо окремо
        color = _MARKER_COLORS[int(m) % len(_MARKER_COLORS)]
        ax.plot(mesh.nodes[idx, 0], mesh.nodes[idx, 1], "o", color=color,
                markersize=5, label=f"маркер {int(m)}", zorder=3)

    ax.set_aspect("equal")
    ax.set_title(title)
    if len(markers) > 1:
        ax.legend(loc="best", fontsize=8)
    return ax


def plot_sparsity(adjacency, ax=None, title="Портрет матриці"):
    """
    "spy"-графік ненульового портрету матриці зв'язності: кожна точка (i,j)
    -- потенційно ненульовий елемент матриці жорсткості. Наочно показує,
    наскільки далеко від діагоналі розташовані ненульові елементи
    (тобто ширину стрічки) до/після перенумерації RCM.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    rows, cols = sparsity_pattern(adjacency)
    ax.scatter(cols, rows, s=1.2, color="#2c3e50")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("номер стовпця (вузол j)")
    ax.set_ylabel("номер рядка (вузол i)")
    return ax


def fig_to_none():
    """Заглушка -- не використовується; залишено для сумісності імпорту."""
    return None
