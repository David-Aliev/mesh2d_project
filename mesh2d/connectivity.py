# -*- coding: utf-8 -*-
"""
connectivity.py
================
1. Формування масиву зв'язності "трикутники <-> вузли" та побудова
   графа сітки (список суміжності вузлів через спільні ребра трикутників).
2. Обчислення ширини стрічки (bandwidth) матриці жорсткості/мас,
   яка виникає при МСЕ-дискретизації на даній нумерації вузлів.
3. Алгоритм Катхілла-Маккі (Cuthill-McKee) та його "зворотна" версія
   (Reverse Cuthill-McKee, RCM) для ОПТИМАЛЬНОЇ перенумерації вузлів,
   що мінімізує ширину стрічки -- реалізовано "з нуля" на чистому
   Python/NumPy (без використання готових функцій scipy).

Матрична ширина стрічки (bandwidth) для симетричної розрідженої
матриці A, що виникає з МСЕ, визначається як

        B = max_{(i,j): A_ij != 0} |i - j|

Ненульові позначки A_ij відповідають парам вузлів i, j, які входять
в один скінченний елемент (тобто є "сусідами" в графі сітки). Отже,
щоб мінімізувати B, потрібно перенумерувати вузли графа так, щоб
номери суміжних вузлів були якомога ближчими один до одного.
Саме це й робить алгоритм Катхілла-Маккі.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

from .triangulation import Mesh


@dataclass
class Connectivity:
    """
    node_to_triangles : list[list[int]]  -- для кожного вузла список номерів
                                             трикутників, які його містять
    triangle_to_nodes : (T,3) ndarray    -- те саме, що mesh.triangles
                                             (явно зберігаємо тут як частину
                                             "масиву зв'язності")
    adjacency : list[set[int]]           -- граф сітки: список суміжних вузлів
                                             (вузли, що входять в один трикутник)
    """
    node_to_triangles: list
    triangle_to_nodes: np.ndarray
    adjacency: list


def build_connectivity(mesh: Mesh) -> Connectivity:
    """
    Формує масив зв'язності між трикутниками і вузлами (вимога п.2):
        - node_to_triangles[k]  -> список індексів трикутників, що містять вузол k
        - adjacency[k]          -> множина вузлів, суміжних із вузлом k
                                    (мають з ним спільне ребро хоча б одного трикутника)
    """
    n = mesh.n_nodes
    node_to_triangles = [[] for _ in range(n)]
    adjacency = [set() for _ in range(n)]

    for t_idx, tri in enumerate(mesh.triangles):
        for local_node in tri:
            node_to_triangles[local_node].append(t_idx)
        i, j, k = tri
        adjacency[i].update((j, k))
        adjacency[j].update((i, k))
        adjacency[k].update((i, j))

    return Connectivity(node_to_triangles=node_to_triangles,
                         triangle_to_nodes=mesh.triangles.copy(),
                         adjacency=adjacency)


def compute_bandwidth(adjacency: list) -> int:
    """
    Обчислює ширину стрічки матриці для поточної нумерації вузлів:
        B = max по всіх ребрах графа |i - j|
    """
    bw = 0
    for i, neigh in enumerate(adjacency):
        if neigh:
            bw = max(bw, max(abs(i - j) for j in neigh))
    return bw


def compute_profile(adjacency: list) -> int:
    """
    Додаткова, тісно пов'язана метрика -- "профіль" матриці
    (сума відстаней до найдальшого лівого сусіда в кожному рядку),
    яка також суттєво зменшується після RCM-перенумерації і напряму
    впливає на обсяг пам'яті при skyline/profile-зберіганні матриці.
    """
    profile = 0
    for i, neigh in enumerate(adjacency):
        if neigh:
            profile += (i - min(neigh)) if min(neigh) < i else 0
    return profile


# ========================================================================
# Алгоритм Катхілла-Маккі (Cuthill-McKee) / Reverse Cuthill-McKee (RCM)
# ========================================================================

def _find_pseudo_peripheral_node(adjacency: list, start: int) -> int:
    """
    Пошук "псевдо-периферійного" вузла (спрощений алгоритм Джорджа-Лю,
    George-Liu). Хороший стартовий вузол суттєво покращує результат
    Катхілла-Маккі: алгоритм BFS з поточного вузла будує рівні
    (level structure), бере вузол з останнього (найдальшого) рівня
    з МІНІМАЛЬНИМ степенем як нового кандидата, і повторює процес,
    доки глибина (ексцентриситет) перестає зростати.
    """
    current = start
    best_depth = -1

    while True:
        levels = _bfs_levels(adjacency, current)
        depth = len(levels) - 1
        last_level = levels[-1]
        # кандидат -- вузол останнього рівня з мінімальним степенем
        candidate = min(last_level, key=lambda v: len(adjacency[v]))

        if depth <= best_depth:
            return current
        best_depth = depth
        current = candidate


def _bfs_levels(adjacency: list, start: int) -> list:
    """Повертає список рівнів BFS (список списків вузлів) від вершини start."""
    n = len(adjacency)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    levels = [[start]]
    while True:
        next_level = []
        for u in levels[-1]:
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    next_level.append(v)
        if not next_level:
            break
        levels.append(next_level)
    return levels


def cuthill_mckee_order(adjacency: list) -> np.ndarray:
    """
    Класичний алгоритм Катхілла-Маккі.

    Ідея алгоритму:
        1. Обрати стартовий вузол з мінімальним степенем (або
           псевдо-периферійний вузол -- дає кращий результат).
        2. Виконати обхід графа У ШИРИНУ (BFS), але на кожному кроці
           сусідів поточного вузла додавати до черги, ВІДСОРТОВАНИМИ
           за зростанням їхнього степеня (вузли з малим степенем -
           раніше). Це "стягує" сусідні вузли ближче один до одного
           в підсумковій нумерації.
        3. Порядок, у якому вузли вилучаються з черги, і є новою
           нумерацією (0, 1, 2, ...).

    Компонент зв'язності графа може бути декілька (якщо область
    складається з непоєднаних частин) -- алгоритм обробляє кожну
    компоненту окремо, у порядку зростання найменшого степеня вузла.

    Повертає масив order, довжини N: order[new_position] = old_index.
    """
    n = len(adjacency)
    degree = np.array([len(adjacency[i]) for i in range(n)])
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    pos = 0

    # Вузли можуть утворювати декілька незв'язних компонент
    remaining = set(range(n))
    while remaining:
        # стартуємо з вузла мінімального степеня серед необроблених
        start = min(remaining, key=lambda v: degree[v])
        start = _find_pseudo_peripheral_node(adjacency, start)

        visited[start] = True
        queue = deque([start])
        order[pos] = start
        pos += 1
        remaining.discard(start)

        while queue:
            u = queue.popleft()
            neighbors = [v for v in adjacency[u] if not visited[v]]
            neighbors.sort(key=lambda v: degree[v])  # зростання степеня
            for v in neighbors:
                visited[v] = True
                order[pos] = v
                pos += 1
                remaining.discard(v)
                queue.append(v)

    return order


def reverse_cuthill_mckee_order(adjacency: list) -> np.ndarray:
    """
    Reverse Cuthill-McKee (RCM): та сама нумерація, що й Cuthill-McKee,
    але в оберненому порядку. Доведено (Джордж, 1971), що обернений
    порядок ніколи не гірший за прямий і на практиці майже завжди дає
    меншу або таку саму ширину стрічки, тому саме RCM є стандартом
    де-факто в скінченно-елементних пакетах.
    """
    order = cuthill_mckee_order(adjacency)
    return order[::-1]


def permutation_to_mapping(order: np.ndarray) -> np.ndarray:
    """
    order[new_pos] = old_index  ->  old_to_new[old_index] = new_pos
    """
    n = len(order)
    old_to_new = np.empty(n, dtype=int)
    old_to_new[order] = np.arange(n)
    return old_to_new


def apply_renumbering(mesh: Mesh, order: np.ndarray) -> Mesh:
    """
    Застосовує перестановку вузлів `order` (order[new] = old) до сітки:
    перенумеровує вузли (координати, маркери) і переписує масив
    трикутників у нових індексах вузлів.
    """
    old_to_new = permutation_to_mapping(order)
    new_nodes = mesh.nodes[order]
    new_markers = mesh.node_markers[order]
    new_triangles = old_to_new[mesh.triangles]
    new_edges = None if mesh.edges is None else old_to_new[mesh.edges]
    return Mesh(nodes=new_nodes, triangles=new_triangles,
                node_markers=new_markers,
                triangle_attributes=None if mesh.triangle_attributes is None
                else mesh.triangle_attributes.copy(),
                edges=new_edges,
                edge_markers=None if mesh.edge_markers is None else mesh.edge_markers.copy(),
                triangle_coordinates=new_nodes[new_triangles].copy())


def renumber_with_rcm(mesh: Mesh) -> tuple[Mesh, int, int]:
    """
    Зручна функція "все в одному": будує граф сітки, обчислює вихідну
    ширину стрічки, застосовує RCM і повертає (нова_сітка, bw_до, bw_після).
    """
    conn = build_connectivity(mesh)
    bw_before = compute_bandwidth(conn.adjacency)
    order = reverse_cuthill_mckee_order(conn.adjacency)
    new_mesh = apply_renumbering(mesh, order)
    new_conn = build_connectivity(new_mesh)
    bw_after = compute_bandwidth(new_conn.adjacency)
    return new_mesh, bw_before, bw_after


def sparsity_pattern(adjacency: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Повертає координати (rows, cols) ненульових елементів матриці
    зв'язності (включно з діагоналлю) -- зручно для візуалізації
    "профілю" матриці (spy-plot) до/після RCM.
    """
    rows, cols = [], []
    for i, neigh in enumerate(adjacency):
        rows.append(i); cols.append(i)
        for j in neigh:
            rows.append(i); cols.append(j)
    return np.array(rows), np.array(cols)
