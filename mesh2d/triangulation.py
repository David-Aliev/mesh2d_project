# -*- coding: utf-8 -*-
"""
triangulation.py
=================
Реалізація тріангуляції Делоне з контролем якості (мінімального кута)
та контролем густини сітки (глобальним і локальним).

Використовується бібліотека `triangle` -- Python-обгортка класичного
генератора Дж. Шюvarchuka (Jonathan Shewchuk, Triangle, 1996), який
реалізує:
    * побудову тріангуляції Делоне (умова Делоне: жодна вершина сітки
      не потрапляє всередину описаного кола жодного трикутника);
    * "Constrained Delaunay Triangulation" -- тріангуляцію, що
      обов'язково містить задані відрізки межі області (щоб межа
      області збігалася зі сторонами трикутників);
    * ітеративне згущення сітки (алгоритм Рупперта, Ruppert's Delaunay
      refinement) із гарантованим мінімальним кутом трикутників та/або
      максимальною площею трикутника.

Ключ -q<angle> вмикає власне алгоритм Рупперта: доки в сітці є
трикутник з кутом меншим за заданий (за замовчуванням гарантовано
до ~34-34.9 градуса теоретично, а на практиці стабільно до ~33-34),
генератор вставляє додаткові ("Штейнерові") вузли й перебудовує
сітку так, щоб УМОВА ДЕЛОНЕ виконувалась на кожному кроці.
Ключ -a<area> обмежує максимальну площу трикутника (густина сітки),
а регіональні атрибути (-A разом із `regions`) дозволяють задавати
РІЗНУ максимальну площу в різних підобластях (локальне згущення).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import triangle as tr

from .geometry import Domain


@dataclass
class Mesh:
    """
    Результат тріангуляції: вузли, елементи (трикутники) та
    маркери граничних умов на вузлах.

    nodes : (N,2) ndarray            -- координати вузлів
    triangles : (T,3) ndarray(int)   -- індекси вузлів кожного трикутника
                                          (нумерація трикутників = індекс рядка)
    node_markers : (N,) ndarray(int) -- маркер граничної умови вузла (0 = внутрішній)
    triangle_attributes : (T,) ndarray -- атрибут підобласті трикутника (для локальної густини)
    """
    nodes: np.ndarray
    triangles: np.ndarray
    node_markers: np.ndarray
    triangle_attributes: np.ndarray = field(default=None)

    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_triangles(self) -> int:
        return self.triangles.shape[0]

    def boundary_nodes(self, marker: int | None = None) -> np.ndarray:
        """Індекси вузлів межі; якщо marker задано -- лише вузли з цим маркером."""
        if marker is None:
            return np.where(self.node_markers != 0)[0]
        return np.where(self.node_markers == marker)[0]

    def triangle_areas(self) -> np.ndarray:
        p0 = self.nodes[self.triangles[:, 0]]
        p1 = self.nodes[self.triangles[:, 1]]
        p2 = self.nodes[self.triangles[:, 2]]
        cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
                (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        return 0.5 * np.abs(cross)

    def triangle_min_angles_deg(self) -> np.ndarray:
        """Мінімальний внутрішній кут (у градусах) кожного трикутника."""
        p0 = self.nodes[self.triangles[:, 0]]
        p1 = self.nodes[self.triangles[:, 1]]
        p2 = self.nodes[self.triangles[:, 2]]

        def angle_at(a, b, c):
            # кут у вершині a, утворений відрізками a-b та a-c
            v1 = b - a
            v2 = c - a
            cos_a = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-300
            )
            cos_a = np.clip(cos_a, -1.0, 1.0)
            return np.degrees(np.arccos(cos_a))

        a0 = angle_at(p0, p1, p2)
        a1 = angle_at(p1, p2, p0)
        a2 = angle_at(p2, p0, p1)
        return np.minimum(np.minimum(a0, a1), a2)

    def copy(self) -> "Mesh":
        return Mesh(
            nodes=self.nodes.copy(),
            triangles=self.triangles.copy(),
            node_markers=self.node_markers.copy(),
            triangle_attributes=None if self.triangle_attributes is None
            else self.triangle_attributes.copy(),
        )


class DelaunayMesher:
    """
    Обгортка над бібліотекою `triangle`, що будує тріангуляцію Делоне
    довільної області із заданими параметрами якості та густини сітки.
    """

    def __init__(self, min_angle_deg: float = 20.0, max_area: float | None = None,
                 enforce_conforming_delaunay: bool = True):
        """
        min_angle_deg : гарантований мінімальний кут трикутників сітки
                        (контроль мінімального кута, п.1 вимог).
                        Практична стеля надійної збіжності алгоритму
                        Рупперта -- приблизно 33-34 градуси.
        max_area      : глобальне обмеження площі трикутника
                        (глобальний контроль густини сітки). None -- без обмеження
                        (лише якісна тріангуляція без згущення).
        enforce_conforming_delaunay : якщо True, використовується ключ 'D'
                        (conforming Delaunay), який гарантує виконання умови
                        Делоне не лише для трикутників, а й build "справжню"
                        (не лише constrained) тріангуляцію Делоне.
        """
        self.min_angle_deg = min_angle_deg
        self.max_area = max_area
        self.enforce_conforming_delaunay = enforce_conforming_delaunay

    # ------------------------------------------------------------------ #
    def _build_switches(self, has_regions: bool) -> str:
        # p  -- тріангулювати PSLG (задану межу як constrained-відрізки)
        # q  -- забезпечити якість (мінімальний кут), алгоритм Рупперта
        # a  -- обмеження площі трикутника (глобальне і/або регіональне)
        # A  -- зберегти регіональні атрибути (для локального контролю густини)
        # D  -- conforming Delaunay: гарантує "справжню" умову Делоне
        # z  -- нумерація вузлів з 0 (зручно для Python)
        switches = "pz"
        if self.enforce_conforming_delaunay:
            switches += "D"
        if self.min_angle_deg is not None:
            switches += f"q{self.min_angle_deg:.4f}"
        if self.max_area is not None:
            switches += f"a{self.max_area:.8f}"
        if has_regions:
            switches += "A"
        return switches

    def triangulate(self, domain: Domain) -> Mesh:
        tri_input = domain.to_triangle_input()
        has_regions = "regions" in tri_input
        switches = self._build_switches(has_regions)

        result = tr.triangulate(tri_input, switches)

        nodes = np.asarray(result["vertices"], dtype=float)
        triangles = np.asarray(result["triangles"], dtype=int)
        node_markers = np.asarray(result.get("vertex_markers",
                                              np.zeros((nodes.shape[0], 1), dtype=int))).reshape(-1)
        tri_attrs = None
        if "triangle_attributes" in result:
            tri_attrs = np.asarray(result["triangle_attributes"]).reshape(-1)

        mesh = Mesh(nodes=nodes, triangles=triangles,
                    node_markers=node_markers, triangle_attributes=tri_attrs)
        return mesh

    def refine(self, mesh: Mesh, new_min_angle_deg: float | None = None,
               new_max_area: float | None = None) -> Mesh:
        """
        Додаткове (повторне) згущення вже побудованої сітки: приймає
        готову тріангуляцію "як є" (-r) і застосовує до неї нові,
        жорсткіші параметри якості/густини, зберігаючи умову Делоне.
        """
        min_angle = self.min_angle_deg if new_min_angle_deg is None else new_min_angle_deg
        max_area = self.max_area if new_max_area is None else new_max_area

        tri_input = {
            "vertices": mesh.nodes,
            "triangles": mesh.triangles,
            "vertex_markers": mesh.node_markers.reshape(-1, 1),
        }
        if mesh.triangle_attributes is not None:
            tri_input["triangle_attributes"] = mesh.triangle_attributes.reshape(-1, 1)

        switches = "rpz"
        if self.enforce_conforming_delaunay:
            switches += "D"
        if min_angle is not None:
            switches += f"q{min_angle:.4f}"
        if max_area is not None:
            switches += f"a{max_area:.8f}"

        result = tr.triangulate(tri_input, switches)
        nodes = np.asarray(result["vertices"], dtype=float)
        triangles = np.asarray(result["triangles"], dtype=int)
        node_markers = np.asarray(result.get("vertex_markers",
                                              np.zeros((nodes.shape[0], 1), dtype=int))).reshape(-1)
        tri_attrs = None
        if "triangle_attributes" in result:
            tri_attrs = np.asarray(result["triangle_attributes"]).reshape(-1)
        return Mesh(nodes=nodes, triangles=triangles,
                    node_markers=node_markers, triangle_attributes=tri_attrs)


def verify_delaunay(mesh: Mesh, tol: float = 1e-9) -> tuple[bool, int]:
    """
    Незалежна перевірка умови Делоне "з нуля" (без довіри до бібліотеки):
    для кожного трикутника перевіряє, чи не потрапляє жодна ІНША вершина
    сітки всередину його описаного кола. Повертає (чи_виконана, к-сть_порушень).
    Використовується лише для невеликих сіток (діагностика/демонстрація),
    бо має складність O(T*N).
    """
    nodes = mesh.nodes
    violations = 0
    for tri in mesh.triangles:
        a, b, c = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        # центр і радіус описаного кола трикутника abc
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-14:
            continue  # вироджений трикутник (не повинно траплятись)
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) +
              (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) +
              (cx ** 2 + cy ** 2) * (bx - ax)) / d
        r2 = (ax - ux) ** 2 + (ay - uy) ** 2
        dist2 = (nodes[:, 0] - ux) ** 2 + (nodes[:, 1] - uy) ** 2
        inside = dist2 < (r2 - tol)
        inside[tri] = False  # власні вершини не рахуємо
        if np.any(inside):
            violations += int(np.sum(inside))
    return violations == 0, violations
