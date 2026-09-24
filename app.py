# -*- coding: utf-8 -*-
"""
app.py -- Графічний інтерфейс користувача (Streamlit) для тріангуляції
Делоне довільних 2D областей.

Запуск:
    streamlit run app.py

Можливості інтерфейсу:
    * вибір типу області (прямокутник, прямокутник з отвором, круг,
      довільний багатокутник за координатами користувача);
    * керування параметрами сітки: мінімальний кут (якість) і
      максимальна площа трикутника (густина), включно з локальною
      зоною згущення;
    * візуалізація сітки, маркування граничних умов, портрету матриці
      зв'язності до/після оптимальної нумерації (RCM) з чисельним
      порівнянням ширини стрічки;
    * режим ПОРІВНЯННЯ двох наборів параметрів (густина/геометрія)
      "пліч-о-пліч".
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh2d import (
    rectangle_domain, circle_domain, polygon_domain, rectangle_with_hole_domain,
    DelaunayMesher, verify_delaunay,
    build_connectivity, compute_bandwidth, renumber_with_rcm,
    plot_mesh, plot_boundary_markers, plot_sparsity,
)

st.set_page_config(page_title="Тріангуляція Делоне 2D", layout="wide")
st.title("🔺 Розбиття двовимірної області на трикутні скінченні елементи")
st.caption("Тріангуляція Делоне з контролем якості, локальним згущенням сітки, "
           "маркуванням граничних умов та оптимальною (RCM) нумерацією вузлів.")


# ============================================================================
# Побудова геометрії та сітки за параметрами, обраними в інтерфейсі
# ============================================================================

def _point_in_polygon(pt, poly) -> bool:
    """Перевірка \"точка строго всередині багатокутника\" методом променя (ray casting)."""
    x, y = pt
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
            if x < x_int:
                inside = not inside
    return inside


def _outer_polygon_points(kind, params):
    """Повертає полігон зовнішньої межі області (для перевірки коректності зони згущення)."""
    if kind == "Прямокутник" or kind == "Прямокутник з отвором":
        w, h = params["w"], params["h"]
        return [(0, 0), (w, 0), (w, h), (0, h)]
    if kind == "Круг":
        r = params["radius"]
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        return list(zip(r * np.cos(ang), r * np.sin(ang)))
    return [tuple(p) for p in params["poly_points"]]


def _validate_refine_zone(kind, params):
    """
    Перевіряє, що зона згущення повністю лежить УСЕРЕДИНІ області (не
    перетинає і не торкається зовнішньої межі чи отвору). Без цієї
    перевірки вироджена вхідна геометрія (наприклад, зона, що торкається
    кута прямокутника) може призвести до аварійного завершення C-бібліотеки
    тріангуляції -- тому помилку краще показати користувачу заздалегідь.
    """
    outer = _outer_polygon_points(kind, params)

    if params["refine_shape"] == "коло":
        cx, cy, r = params["rz_cx"], params["rz_cy"], params["rz_r"]
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        zone_pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]
    else:
        x0, y0, x1, y1 = params["rz_x0"], params["rz_y0"], params["rz_x1"], params["rz_y1"]
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Прямокутна зона згущення: X0<X1 та Y0<Y1 мають виконуватись.")
        zone_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    for p in zone_pts:
        if not _point_in_polygon(p, outer):
            raise ValueError(
                "Зона згущення виходить за межі області (або торкається її межі). "
                "Зменшіть розмір зони чи перемістіть її ближче до центру області."
            )

    if kind == "Прямокутник з отвором":
        hcx, hcy, hr = params["hole_cx"], params["hole_cy"], params["hole_r"]
        for p in zone_pts:
            if (p[0] - hcx) ** 2 + (p[1] - hcy) ** 2 <= hr ** 2:
                raise ValueError("Зона згущення перетинає отвір у пластині.")


def _pick_background_point(kind, params, outer_poly):
    """
    Автоматично підбирає точку "фонової" (решти) підобласті для регіонального
    контролю густини: точка має бути ВСЕРЕДИНІ області, ПОЗА зоною згущення
    і ПОЗА отвором (інакше регіональний атрибут буде присвоєно неправильній
    підобласті -- див. пояснення в README.md).
    """
    if params["refine_shape"] == "коло":
        rz_check = lambda p: (p[0] - params["rz_cx"]) ** 2 + (p[1] - params["rz_cy"]) ** 2 <= params["rz_r"] ** 2
    else:
        x0, y0, x1, y1 = params["rz_x0"], params["rz_y0"], params["rz_x1"], params["rz_y1"]
        rz_check = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    if kind == "Прямокутник з отвором":
        hcx, hcy, hr = params["hole_cx"], params["hole_cy"], params["hole_r"]
        hole_check = lambda p: (p[0] - hcx) ** 2 + (p[1] - hcy) ** 2 <= hr ** 2
    else:
        hole_check = lambda p: False

    outer_arr = np.asarray(outer_poly)
    centroid = outer_arr.mean(axis=0)
    candidates = [tuple(centroid)]
    for t in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95):
        for v in outer_arr:
            candidates.append(tuple(v * (1 - t) + centroid * t))

    for pt in candidates:
        if _point_in_polygon(pt, outer_poly) and not rz_check(pt) and not hole_check(pt):
            return pt
    raise ValueError("Не вдалося підібрати точку для фонової підобласті густини сітки "
                      "-- спробуйте зменшити зону згущення.")


def build_domain(params: dict):
    """Створює об'єкт Domain за словником параметрів з бічної панелі."""
    kind = params["kind"]

    if kind == "Прямокутник":
        dom = rectangle_domain(params["w"], params["h"], markers=(1, 2, 3, 4))

    elif kind == "Прямокутник з отвором":
        dom = rectangle_with_hole_domain(
            params["w"], params["h"], params["hole_cx"], params["hole_cy"],
            params["hole_r"], outer_markers=(1, 2, 3, 4), hole_marker=5,
        )

    elif kind == "Круг":
        dom = circle_domain(0.0, 0.0, params["radius"], n_segments=64,
                             markers=(1, 2, 3, 4))

    else:  # "Довільний багатокутник"
        pts = params["poly_points"]
        dom = polygon_domain(pts, marker=1)

    if params.get("use_refine_zone"):
        _validate_refine_zone(kind, params)
        if params["refine_shape"] == "коло":
            dom.add_refinement_zone_circle(
                params["rz_cx"], params["rz_cy"], params["rz_r"],
                max_area_inside=params["rz_area"], n_segments=24,
            )
        else:
            dom.add_refinement_zone_rect(
                params["rz_x0"], params["rz_y0"], params["rz_x1"], params["rz_y1"],
                max_area_inside=params["rz_area"],
            )
        # фонова (решта області) густина -- задається через глобальний max_area;
        # точку для неї підбираємо автоматично, щоб вона не потрапила в зону згущення
        outer_poly = _outer_polygon_points(kind, params)
        background_pt = _pick_background_point(kind, params, outer_poly)
        dom.add_background_region(background_pt, attribute=0, max_area=params["max_area"])

    return dom


def parse_polygon_text(text: str) -> np.ndarray:
    pts = []
    for line in text.strip().splitlines():
        line = line.strip().replace(";", ",")
        if not line:
            continue
        x_str, y_str = line.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=float)


def run_pipeline(params: dict) -> dict:
    """Будує область -> тріангулює -> перевіряє Делоне -> перенумеровує (RCM)."""
    dom = build_domain(params)
    mesher = DelaunayMesher(min_angle_deg=params["min_angle"], max_area=params["max_area"])
    mesh = mesher.triangulate(dom)

    delaunay_ok, delaunay_violations = verify_delaunay(mesh) if mesh.n_nodes <= 2500 else (None, None)

    conn = build_connectivity(mesh)
    bw_before = compute_bandwidth(conn.adjacency)
    new_mesh, _, bw_after = renumber_with_rcm(mesh)

    return dict(
        mesh=mesh, new_mesh=new_mesh, conn=conn,
        bw_before=bw_before, bw_after=bw_after,
        delaunay_ok=delaunay_ok, delaunay_violations=delaunay_violations,
        min_angle_achieved=float(mesh.triangle_min_angles_deg().min()),
        max_area_achieved=float(mesh.triangle_areas().max()),
    )


# ============================================================================
# Бічна панель: параметри
# ============================================================================

def sidebar_params(label_suffix: str = "", defaults: dict | None = None) -> dict:
    """
    Малює блок віджетів параметрів у бічній панелі. label_suffix потрібен,
    щоб мати ДВА незалежні набори віджетів у режимі порівняння (набір А/Б)
    без конфлікту ключів Streamlit.
    """
    d = defaults or {}
    p = {}
    st.markdown(f"##### Геометрія області {label_suffix}")
    p["kind"] = st.selectbox(
        "Тип області", ["Прямокутник", "Прямокутник з отвором", "Круг", "Довільний багатокутник"],
        index=["Прямокутник", "Прямокутник з отвором", "Круг", "Довільний багатокутник"].index(
            d.get("kind", "Прямокутник")),
        key=f"kind{label_suffix}",
    )

    if p["kind"] == "Прямокутник":
        c1, c2 = st.columns(2)
        p["w"] = c1.number_input("Ширина", 0.1, 20.0, d.get("w", 2.0), key=f"w{label_suffix}")
        p["h"] = c2.number_input("Висота", 0.1, 20.0, d.get("h", 1.0), key=f"h{label_suffix}")

    elif p["kind"] == "Прямокутник з отвором":
        c1, c2 = st.columns(2)
        p["w"] = c1.number_input("Ширина", 0.1, 20.0, d.get("w", 2.0), key=f"w{label_suffix}")
        p["h"] = c2.number_input("Висота", 0.1, 20.0, d.get("h", 1.0), key=f"h{label_suffix}")
        c3, c4, c5 = st.columns(3)
        p["hole_cx"] = c3.number_input("Отвір: X центру", 0.0, p["w"], d.get("hole_cx", p["w"] / 2), key=f"hx{label_suffix}")
        p["hole_cy"] = c4.number_input("Отвір: Y центру", 0.0, p["h"], d.get("hole_cy", p["h"] / 2), key=f"hy{label_suffix}")
        p["hole_r"] = c5.number_input("Отвір: радіус", 0.01, min(p["w"], p["h"]) / 2.5, d.get("hole_r", 0.2), key=f"hr{label_suffix}")

    elif p["kind"] == "Круг":
        p["radius"] = st.number_input("Радіус", 0.1, 20.0, d.get("radius", 1.0), key=f"r{label_suffix}")

    else:
        default_txt = d.get("poly_text", "0,0\n2,0\n2.5,1\n1,2\n-0.5,1")
        txt = st.text_area("Вершини багатокутника (x,y на кожному рядку, за годинниковою "
                            "стрілкою або проти неї)", value=default_txt, height=110,
                            key=f"poly{label_suffix}")
        try:
            pts = parse_polygon_text(txt)
            if len(pts) < 3:
                st.error("Потрібно щонайменше 3 точки.")
                pts = parse_polygon_text(default_txt)
        except Exception as e:
            st.error(f"Помилка розбору точок: {e}")
            pts = parse_polygon_text(default_txt)
        p["poly_points"] = pts

    st.markdown(f"##### Параметри сітки {label_suffix}")
    p["min_angle"] = st.slider("Мінімальний кут трикутників, °", 10.0, 33.0,
                                d.get("min_angle", 25.0), 0.5, key=f"ang{label_suffix}",
                                help="Контроль якості сітки (алгоритм Рупперта). "
                                     "Надійна верхня межа ~33-34°.")
    area_guess = d.get("max_area", 0.05)
    p["max_area"] = st.number_input("Максимальна площа трикутника (густина сітки)",
                                     min_value=1e-5, max_value=10.0, value=area_guess,
                                     step=0.001, format="%.5f", key=f"area{label_suffix}",
                                     help="Менше значення -> дрібніша (густіша) сітка.")

    st.markdown(f"##### Локальне згущення {label_suffix} (опційно)")
    p["use_refine_zone"] = st.checkbox("Додати зону локального згущення", key=f"urz{label_suffix}",
                                        value=d.get("use_refine_zone", False))
    if p["use_refine_zone"]:
        # безпечний центр за замовчуванням -- геометричний центр самої області,
        # щоб зона згущення завідомо не торкалась межі (яка призводить до
        # виродженої вхідної геометрії для тріангулятора)
        if p["kind"] == "Прямокутник":
            def_cx, def_cy, def_r = p["w"] / 2, p["h"] / 2, min(p["w"], p["h"]) * 0.12
        elif p["kind"] == "Прямокутник з отвором":
            # зміщуємо зону згущення в куток області, щоб вона за замовчуванням
            # не перетиналась з отвором (який за промовчанням у центрі)
            def_cx, def_cy, def_r = p["w"] * 0.2, p["h"] * 0.8, min(p["w"], p["h"]) * 0.1
        elif p["kind"] == "Круг":
            def_cx, def_cy, def_r = 0.0, 0.0, p["radius"] * 0.25
        else:
            centroid = p["poly_points"].mean(axis=0)
            span = p["poly_points"].max(axis=0) - p["poly_points"].min(axis=0)
            def_cx, def_cy, def_r = centroid[0], centroid[1], float(min(span)) * 0.15

        p["refine_shape"] = st.radio("Форма зони", ["коло", "прямокутник"], horizontal=True,
                                      key=f"rzshape{label_suffix}")
        if p["refine_shape"] == "коло":
            c1, c2, c3 = st.columns(3)
            p["rz_cx"] = c1.number_input("Xц", value=d.get("rz_cx", def_cx), key=f"rzcx{label_suffix}")
            p["rz_cy"] = c2.number_input("Yц", value=d.get("rz_cy", def_cy), key=f"rzcy{label_suffix}")
            p["rz_r"] = c3.number_input("Радіус зони", 0.001, 1000.0, d.get("rz_r", def_r), key=f"rzr{label_suffix}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            p["rz_x0"] = c1.number_input("X0", value=d.get("rz_x0", def_cx - def_r), key=f"rzx0{label_suffix}")
            p["rz_y0"] = c2.number_input("Y0", value=d.get("rz_y0", def_cy - def_r), key=f"rzy0{label_suffix}")
            p["rz_x1"] = c3.number_input("X1", value=d.get("rz_x1", def_cx + def_r), key=f"rzx1{label_suffix}")
            p["rz_y1"] = c4.number_input("Y1", value=d.get("rz_y1", def_cy + def_r), key=f"rzy1{label_suffix}")
        p["rz_area"] = st.number_input("Макс. площа трикутника в зоні (дрібніша сітка)",
                                        1e-6, 1.0, d.get("rz_area", 0.002), step=0.0005,
                                        format="%.5f", key=f"rzarea{label_suffix}")
    return p


with st.sidebar:
    st.header("⚙️ Налаштування")
    compare_mode = st.checkbox("Режим порівняння двох варіантів", value=False)
    st.divider()
    st.subheader("Варіант А" if compare_mode else "Параметри")
    params_a = sidebar_params(" (А)" if compare_mode else "")
    params_b = None
    if compare_mode:
        st.divider()
        st.subheader("Варіант Б")
        params_b = sidebar_params(" (Б)", defaults=params_a)
    st.divider()
    go = st.button("🚀 Побудувати сітку", type="primary", use_container_width=True)


# ============================================================================
# Обчислення (лише за натисканням кнопки -- зберігається в session_state)
# ============================================================================

if go:
    with st.spinner("Тріангуляція..."):
        try:
            st.session_state["result_a"] = run_pipeline(params_a)
            st.session_state["error"] = None
        except Exception as e:
            st.session_state["error"] = str(e)
        if compare_mode and st.session_state.get("error") is None:
            try:
                st.session_state["result_b"] = run_pipeline(params_b)
            except Exception as e:
                st.session_state["error"] = str(e)

if st.session_state.get("error"):
    st.error(f"Помилка побудови сітки: {st.session_state['error']}\n\n"
             "Спробуйте зменшити мінімальний кут або збільшити максимальну площу.")

if "result_a" not in st.session_state:
    st.info("⬅️ Налаштуйте параметри області та сітки на бічній панелі й натисніть "
            "**«Побудувати сітку»**.")
    st.stop()


# ============================================================================
# Виведення результатів
# ============================================================================

def render_result(res: dict, label: str):
    mesh, new_mesh = res["mesh"], res["new_mesh"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Вузлів", mesh.n_nodes)
    c2.metric("Трикутників", mesh.n_triangles)
    c3.metric("Мін. кут, °", f"{res['min_angle_achieved']:.1f}")
    c4.metric("Ширина стрічки до RCM", res["bw_before"])
    c5.metric("Ширина стрічки після RCM", res["bw_after"],
              delta=f"-{100*(1 - res['bw_after']/max(res['bw_before'],1)):.0f}%",
              delta_color="inverse")

    if res["delaunay_ok"] is not None:
        if res["delaunay_ok"]:
            st.success("✅ Умова Делоне виконується для всіх трикутників "
                       "(незалежна перевірка описаних кіл).")
        else:
            st.warning(f"⚠️ Знайдено {res['delaunay_violations']} порушень умови Делоне.")

    tab1, tab2, tab3 = st.tabs(["Сітка", "Граничні умови", "Нумерація вузлів (RCM)"])

    with tab1:
        show_ids = mesh.n_nodes <= 150
        show_ids_ui = st.checkbox("Показати номери вузлів", value=False, key=f"showid_{label}") if show_ids else False
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_mesh(mesh, ax=ax, show_nodes=True, show_node_ids=show_ids_ui,
                  color_by_attribute=mesh.triangle_attributes is not None,
                  title=f"Сітка ({label}): {mesh.n_nodes} вузлів, {mesh.n_triangles} трикутників")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_boundary_markers(mesh, ax=ax, title=f"Граничні умови ({label})")
        st.pyplot(fig)
        plt.close(fig)
        markers = sorted(set(int(m) for m in mesh.node_markers if m != 0))
        st.caption(f"Виявлені маркери граничних умов: {markers}. "
                   "Кожен маркер відповідає окремій ділянці межі (наприклад, "
                   "різним типам умов Діріхле/Неймана в задачі МСЕ).")

    with tab3:
        conn = res["conn"]
        new_conn = build_connectivity(new_mesh)
        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_sparsity(conn.adjacency, ax=ax, title=f"До RCM (bw={res['bw_before']})")
            st.pyplot(fig)
            plt.close(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_sparsity(new_conn.adjacency, ax=ax, title=f"Після RCM (bw={res['bw_after']})")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Кожна точка -- потенційно ненульовий елемент матриці жорсткості "
                   "(пара вузлів одного трикутника). Після RCM ненульові елементи "
                   "стягуються до діагоналі -> менша ширина стрічки -> менше пам'яті "
                   "й швидше розв'язання СЛАР методами для стрічкових матриць.")

        with st.expander("Фрагмент масиву зв'язності (перші 10 трикутників)"):
            st.write("Трикутник -> вузли (оригінальна нумерація):")
            st.dataframe({"triangle_id": list(range(min(10, mesh.n_triangles))),
                          "nodes": [list(t) for t in mesh.triangles[:10]]})

    csv_lines = ["# nodes: id,x,y,boundary_marker"]
    for i, (xy, m) in enumerate(zip(new_mesh.nodes, new_mesh.node_markers)):
        csv_lines.append(f"{i},{xy[0]:.6f},{xy[1]:.6f},{int(m)}")
    csv_lines.append("# triangles: id,n0,n1,n2")
    for i, t in enumerate(new_mesh.triangles):
        csv_lines.append(f"{i},{t[0]},{t[1]},{t[2]}")
    st.download_button(f"⬇️ Завантажити сітку ({label}) як .txt (RCM-нумерація)",
                        data="\n".join(csv_lines), file_name=f"mesh_{label}.txt",
                        mime="text/plain", key=f"dl_{label}")


if compare_mode and "result_b" in st.session_state:
    colL, colR = st.columns(2)
    with colL:
        st.markdown("### Варіант А")
        render_result(st.session_state["result_a"], "А")
    with colR:
        st.markdown("### Варіант Б")
        render_result(st.session_state["result_b"], "Б")

    st.divider()
    st.markdown("### 📊 Пряме порівняння")
    ra, rb = st.session_state["result_a"], st.session_state["result_b"]
    st.table({
        "Показник": ["Кількість вузлів", "Кількість трикутників", "Мінімальний кут, °",
                     "Ширина стрічки (до RCM)", "Ширина стрічки (після RCM)",
                     "Коефіцієнт зменшення стрічки"],
        "Варіант А": [str(ra["mesh"].n_nodes), str(ra["mesh"].n_triangles),
                      f"{ra['min_angle_achieved']:.2f}", str(ra["bw_before"]), str(ra["bw_after"]),
                      f"{ra['bw_before']/max(ra['bw_after'],1):.2f}x"],
        "Варіант Б": [str(rb["mesh"].n_nodes), str(rb["mesh"].n_triangles),
                      f"{rb['min_angle_achieved']:.2f}", str(rb["bw_before"]), str(rb["bw_after"]),
                      f"{rb['bw_before']/max(rb['bw_after'],1):.2f}x"],
    })
else:
    render_result(st.session_state["result_a"], "A")

st.divider()
st.caption("Мінімізація ширини стрічки виконується алгоритмом Reverse Cuthill-McKee, "
           "реалізованим вручну в mesh2d/connectivity.py (без scipy) -- див. README.md.")
