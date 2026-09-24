# -*- coding: utf-8 -*-
"""
mesh2d -- пакет для тріангуляції Делоне довільних двовимірних областей
з контролем якості сітки, оптимальною нумерацією вузлів (RCM) та
маркуванням граничних умов.
"""

from .geometry import (
    Domain, rectangle_domain, circle_domain, polygon_domain,
    polygon_domain_multi_marker, rectangle_with_hole_domain,
)
from .triangulation import DelaunayMesher, Mesh, verify_delaunay
from .connectivity import (
    build_connectivity, compute_bandwidth, compute_profile,
    cuthill_mckee_order, reverse_cuthill_mckee_order,
    apply_renumbering, renumber_with_rcm, sparsity_pattern,
)
from .visualization import plot_mesh, plot_boundary_markers, plot_sparsity

__all__ = [
    "Domain", "rectangle_domain", "circle_domain", "polygon_domain",
    "polygon_domain_multi_marker", "rectangle_with_hole_domain",
    "DelaunayMesher", "Mesh", "verify_delaunay",
    "build_connectivity", "compute_bandwidth", "compute_profile",
    "cuthill_mckee_order", "reverse_cuthill_mckee_order",
    "apply_renumbering", "renumber_with_rcm", "sparsity_pattern",
    "plot_mesh", "plot_boundary_markers", "plot_sparsity",
]
