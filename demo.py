# -*- coding: utf-8 -*-
"""
demo.py
=======
Невеликий консольний скрипт-демонстрація (НЕ є заміною GUI, лише
швидка перевірка бібліотеки mesh2d "з командного рядка"):
будує пластину з отвором і локальною зоною згущення, перевіряє
умову Делоне, виконує RCM-перенумерацію та зберігає картинку
mesh_demo.png з 4 панелями (сітка, граничні умови, портрет матриці
до/після RCM).

Запуск:
    python demo.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh2d import (
    rectangle_with_hole_domain, DelaunayMesher, verify_delaunay,
    build_connectivity, compute_bandwidth, renumber_with_rcm,
    plot_mesh, plot_boundary_markers, plot_sparsity,
)


def main():
    print("1. Побудова геометрії: пластина 2x1 з круглим отвором у центрі...")
    dom = rectangle_with_hole_domain(width=2.0, height=1.0,
                                      hole_cx=1.0, hole_cy=0.5, hole_r=0.2,
                                      outer_markers=(1, 2, 3, 4), hole_marker=5)
    dom.add_refinement_zone_circle(cx=0.3, cy=0.8, r=0.12, max_area_inside=0.001)
    dom.add_background_region((0.05, 0.05), attribute=0, max_area=0.02)

    print("2. Тріангуляція Делоне (мін.кут=28°, глобальна площа<=0.02)...")
    mesher = DelaunayMesher(min_angle_deg=28.0, max_area=0.02)
    mesh = mesher.triangulate(dom)
    print(f"   -> {mesh.n_nodes} вузлів, {mesh.n_triangles} трикутників")
    print(f"   -> мінімальний кут у сітці: {mesh.triangle_min_angles_deg().min():.2f} град.")

    print("3. Незалежна перевірка умови Делоне...")
    ok, violations = verify_delaunay(mesh)
    print(f"   -> умова Делоне виконується: {ok} (порушень: {violations})")

    print("4. Побудова масиву зв'язності та ширини стрічки матриці...")
    conn = build_connectivity(mesh)
    bw_before = compute_bandwidth(conn.adjacency)
    print(f"   -> ширина стрічки до перенумерації: {bw_before}")

    print("5. Оптимальна нумерація вузлів (Reverse Cuthill-McKee)...")
    new_mesh, _, bw_after = renumber_with_rcm(mesh)
    print(f"   -> ширина стрічки після RCM: {bw_after} "
          f"(зменшення у {bw_before/max(bw_after,1):.2f} рази)")

    print("6. Збереження графіків у mesh_demo.png...")
    new_conn = build_connectivity(new_mesh)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_mesh(mesh, ax=axes[0, 0], color_by_attribute=True,
              title=f"Сітка: {mesh.n_nodes} вузлів, {mesh.n_triangles} трикутників")
    plot_boundary_markers(mesh, ax=axes[0, 1])
    plot_sparsity(conn.adjacency, ax=axes[1, 0], title=f"До RCM (bw={bw_before})")
    plot_sparsity(new_conn.adjacency, ax=axes[1, 1], title=f"Після RCM (bw={bw_after})")
    plt.tight_layout()
    fig.savefig("mesh_demo.png", dpi=120)
    print("   -> готово: mesh_demo.png")


if __name__ == "__main__":
    main()
