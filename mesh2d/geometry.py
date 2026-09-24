# -*- coding: utf-8 -*-
"""
geometry.py
===========
Опис геометрії довільної двовимірної області у форматі PSLG
(Planar Straight Line Graph) — набору вершин та відрізків, що
з'єднують їх у замкнені контури. Формат PSLG є "мовою", яку
розуміє бібліотека `triangle` (обгортка навколо класичного
генератора Триангуляції Делоне Дж. Шюvarchuka).

Домен (Domain) може містити:
    * зовнішній контур межі області;
    * довільну кількість отворів (внутрішніх вирізаних областей);
    * довільну кількість "зон згущення" — замкнених контурів
      усередині області, які НЕ є фізичною межею (маркер = 0),
      а лише розмежовують підобласті з різною густиною сітки;
    * маркери граничних умов, що присвоюються відрізкам межі.

Завдяки тому, що область описується не жорстко "зашитими"
координатами, а довільним набором точок/відрізків, побудований
на цьому модулі тріангулятор працює з ДОВІЛЬНОЮ областю, а не
лише з фігурою з індивідуального завдання.
"""

from __future__ import annotations
import numpy as np


class Domain:
    """
    Контейнер геометрії області у форматі, зрозумілому triangle.triangulate().

    Атрибути
    --------
    points : (N,2) ndarray  -- координати всіх опорних вузлів межі/зон
    segments : (M,2) ndarray(int) -- пари індексів точок, що утворюють відрізок межі
    segment_markers : (M,) ndarray(int) -- маркер граничної умови для кожного відрізка
                       (0 означає "не фізична межа", а внутрішня технічна лінія)
    holes : (K,2) ndarray -- точки-мітки всередині "дірок", які потрібно вирізати
    regions : (R,4) ndarray -- рядки [x, y, attribute, max_area] для регіонального
                       контролю густини сітки (за замовчуванням attribute=0)
    """

    def __init__(self):
        self._points: list[list[float]] = []
        self._segments: list[list[int]] = []
        self._segment_markers: list[int] = []
        self._holes: list[list[float]] = []
        self._regions: list[list[float]] = []

    # ------------------------------------------------------------------ #
    # Побудова геометрії
    # ------------------------------------------------------------------ #
    def add_polyline(self, points_xy, marker: int = 1, closed: bool = True) -> list[int]:
        """
        Додає ламану (контур) із точок points_xy у список вершин домену
        та відповідні відрізки з граничним маркером `marker`.

        marker = 0  ->  внутрішня технічна лінія (не є фізичною межею,
                        використовується лише для розмежування зон густини).
        marker > 0  ->  ділянка зовнішньої/внутрішньої межі з типом
                        граничної умови `marker`.

        Повертає список глобальних індексів доданих точок.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        start_idx = len(self._points)
        idx = list(range(start_idx, start_idx + len(points_xy)))
        self._points.extend(points_xy.tolist())

        n = len(points_xy)
        rng = range(n) if closed else range(n - 1)
        for k in rng:
            i = idx[k]
            j = idx[(k + 1) % n]
            self._segments.append([i, j])
            self._segment_markers.append(marker)
        return idx

    def add_boundary(self, points_xy, markers) -> list[int]:
        """
        Додає ОДИН замкнений контур межі, точки якого вводяться РІВНО ОДИН
        РАЗ (на відміну від повторних викликів add_polyline для кожної
        сторони окремо, що створювало б дублікати вершин у спільних кутах
        і могло призводити до збою тріангулятора на виродженому вводі).

        points_xy : (n,2) -- вершини контуру в порядку обходу (без повтору
                    першої точки в кінці -- контур вважається замкненим).
        markers   : або одне ціле число (весь контур -- одна гранична умова),
                    або список довжини n (окремий маркер для кожного
                    ребра points[k] -> points[(k+1) % n]) -- це і є спосіб
                    промаркувати різні частини межі різними типами
                    граничних умов на одному спільному наборі вершин.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        n = len(points_xy)
        if np.isscalar(markers):
            markers = [markers] * n
        if len(markers) != n:
            raise ValueError("довжина markers має дорівнювати кількості вершин контуру")

        start_idx = len(self._points)
        idx = list(range(start_idx, start_idx + n))
        self._points.extend(points_xy.tolist())
        for k in range(n):
            i = idx[k]
            j = idx[(k + 1) % n]
            self._segments.append([i, j])
            self._segment_markers.append(int(markers[k]))
        return idx

    def add_hole(self, xy) -> None:
        """Позначає точку всередині отвору, який потрібно вилучити з сітки."""
        self._holes.append([float(xy[0]), float(xy[1])])

    def add_region(self, xy, attribute: int, max_area: float) -> None:
        """
        Задає підобласть (для регіонального контролю густини сітки):
        усі трикутники всередині замкненого контуру, що містить точку xy,
        отримають атрибут `attribute` і матимуть площу <= max_area.
        """
        self._regions.append([float(xy[0]), float(xy[1]), float(attribute), float(max_area)])

    def add_refinement_zone_circle(self, cx, cy, r, max_area_inside,
                                    n_segments: int = 24, background_attr: int = 0) -> None:
        """
        Додає кругову "зону згущення" сітки всередині області.
        Це замкнений контур-маркер (не фізична межа, marker=0), що ділить
        область на дві підобласті: усередині кола (дрібна сітка,
        max_area_inside) та решту (густина контролюється глобальним
        параметром a або окремим регіоном).
        """
        ang = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
        pts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
        self.add_polyline(pts, marker=0, closed=True)
        self.add_region((cx, cy), attribute=background_attr + 1, max_area=max_area_inside)

    def add_refinement_zone_rect(self, x0, y0, x1, y1, max_area_inside,
                                  background_attr: int = 0) -> None:
        """Прямокутна зона згущення (аналогічно до кругової)."""
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        self.add_polyline(pts, marker=0, closed=True)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.add_region((cx, cy), attribute=background_attr + 1, max_area=max_area_inside)

    def add_background_region(self, xy, attribute: int, max_area: float) -> None:
        """Явно задає густину сітки для "фонової" (решти) підобласті."""
        self.add_region(xy, attribute=attribute, max_area=max_area)

    # ------------------------------------------------------------------ #
    # Експорт у формат triangle
    # ------------------------------------------------------------------ #
    def to_triangle_input(self) -> dict:
        data = {
            "vertices": np.asarray(self._points, dtype=float),
            "segments": np.asarray(self._segments, dtype=int),
            "segment_markers": np.asarray(self._segment_markers, dtype=int),
        }
        if self._holes:
            data["holes"] = np.asarray(self._holes, dtype=float)
        if self._regions:
            data["regions"] = np.asarray(self._regions, dtype=float)
        return data

    @property
    def n_points(self) -> int:
        return len(self._points)

    @property
    def boundary_markers(self) -> list[int]:
        """Список унікальних маркерів фізичної межі (без внутрішнього 0)."""
        return sorted({m for m in self._segment_markers if m != 0})


# ========================================================================
# Фабричні функції для типових областей + приклад довільної області
# ========================================================================

def rectangle_domain(width: float, height: float, x0: float = 0.0, y0: float = 0.0,
                      markers=(1, 2, 3, 4)) -> Domain:
    """
    Прямокутна область.  markers = (низ, право, верх, ліво) — 4 різні
    типи граничних умов на 4 сторонах (демонструє маркування частин межі).
    """
    dom = Domain()
    pts = np.array([[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]])
    # чотири вершини вводяться РІВНО ОДИН РАЗ; кожному ребру (низ/право/
    # верх/ліво) присвоюється свій маркер граничної умови -- без дублювання
    # вершин у спільних кутах (що раніше спричиняло збій тріангулятора)
    edge_markers = [markers[0], markers[1], markers[2], markers[3]]
    dom.add_boundary(pts, edge_markers)
    return dom


def circle_domain(cx: float, cy: float, r: float, n_segments: int = 48,
                   markers=(1, 2, 3, 4)) -> Domain:
    """
    Кругова область. Межа поділяється на 4 дуги (верх/право/низ/ліво),
    кожна зі своїм маркером граничної умови -- ще один приклад
    маркування частин межі з однаковим типом граничної умови.
    """
    dom = Domain()
    ang = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
    quarter = n_segments // 4
    # кожному РЕБРУ (points[k] -> points[k+1]) присвоюємо маркер залежно
    # від того, у яку з 4 дуг воно потрапляє; самі вершини вводяться
    # один раз через add_boundary (без дублікатів у місцях стику дуг)
    edge_markers = [markers[min(k // quarter, len(markers) - 1)] for k in range(n_segments)]
    dom.add_boundary(pts, edge_markers)
    return dom


def polygon_domain(points_xy, marker: int = 1) -> Domain:
    """Довільний простий (не самоперетинний) багатокутник з одним типом межі."""
    dom = Domain()
    dom.add_polyline(points_xy, marker=marker, closed=True)
    return dom


def polygon_domain_multi_marker(edges_with_markers) -> Domain:
    """
    Довільний багатокутник, у якого РІЗНІ ділянки межі мають РІЗНІ маркери.

    edges_with_markers: список кортежів (run_points, marker), де run_points --
    послідовні вершини одного "прогону" межі (БЕЗ останньої точки прогону,
    вона ж перша точка наступного прогону). Прогони йдуть послідовно й
    разом утворюють один замкнений контур. Наприклад для трикутника з
    трьома різними гран.умовами: [([P0], m1), ([P1], m2), ([P2], m3)]
    описує ребра P0->P1 (маркер m1), P1->P2 (m2), P2->P0 (m3).

    Усі вершини контуру вводяться РІВНО ОДИН РАЗ (через Domain.add_boundary),
    що виключає дублікати точок у місцях стику прогонів.
    """
    all_points = []
    all_markers = []
    for run_points, marker in edges_with_markers:
        run_points = np.atleast_2d(np.asarray(run_points, dtype=float))
        for p in run_points:
            all_points.append(p)
            all_markers.append(marker)

    dom = Domain()
    dom.add_boundary(np.array(all_points), all_markers)
    return dom


def rectangle_with_hole_domain(width, height, hole_cx, hole_cy, hole_r,
                                outer_markers=(1, 2, 3, 4), hole_marker=5,
                                n_hole_segments=32) -> Domain:
    """
    Приклад БАГАТОЗВ'ЯЗНОЇ (multiply-connected) довільної області:
    прямокутна пластина з круглим отвором. Демонструє, що алгоритм
    не обмежений однозв'язними контурами з індивідуального завдання.
    """
    dom = rectangle_domain(width, height, markers=outer_markers)
    ang = np.linspace(0.0, 2.0 * np.pi, n_hole_segments, endpoint=False)
    hole_pts = np.column_stack([hole_cx + hole_r * np.cos(ang), hole_cy + hole_r * np.sin(ang)])
    dom.add_polyline(hole_pts, marker=hole_marker, closed=True)
    dom.add_hole((hole_cx, hole_cy))
    return dom
