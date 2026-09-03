"""
vmlib — измерения, диагностика и визуализация численных методов.

Библиотека предоставляет замеры времени, подсчёт вызовов, нормы ошибки,
таблицы, графики, анимацию, сверку с эталоном и строгие самопроверки.
Численные алгоритмы остаются в коде пользователя.

Быстрый справочник
------------------
    import vmlib as vm

    # результат метода — единый формат для всех заданий
    res = vm.Result(x=u, success=True, elapsed_time=t, n_fevals=f.count)

    # честный замер времени и числа вызовов
    counted_rhs = vm.CallCounter(rhs)
    with vm.timed() as t:
        u = my_solver(counted_rhs, problem)
    vm.benchmark(my_solver, counted_rhs, problem,
                 repeats=5, reset=counted_rhs)  # прогрев и разброс

    # ошибки и порядки
    vm.max_error(u, u_exact)
    ct = vm.convergence(hs, errs, expected=2)
    print(ct.verdict())

    # таблицы и графики
    vm.compare_methods({"РК4": r1, "Верле": r2})
    vm.convergence_plot({"РК4": (hs, errs)}, expected_orders={"РК4": 4})
    vm.cost_vs_error({"РК4": rs1, "РК45": rs2})
    vm.invariant_drift({"РК4": (t, energy)})

    # сверка с эталоном
    bl = vm.load_baseline()
    vm.compare(my_u, bl["u_ref"], name="решение")

    # самопроверки
    vm.check_order(solver, exact, [10, 20, 40, 80], expected=2)
    vm.check_exact_on(my_quad, {"полином 2-й степени": ((f2,), 1/3)})
    vm.check_invariant(t, energy)
    vm.report({"порядок": ok1, "эталон": ok2})

    # анимация: несколько реализаций бок о бок
    vm.render([scene_ref, scene_mine], n_frames=120, dt=0.05, out="cmp.gif")
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._version import __version__
from .result import Result, COST_UNITS
from .bench import Timing, benchmark, CallCounter, timed
from .errors import (max_error, l2_error, rel_error, empirical_orders,
                     ConvergenceTable, convergence, ORDER_TOL,
                     MIN_POINTS_FOR_ORDER, NOISE_FACTOR, order_slack,
                     order_matches, noise_floor, count_nonfinite)
from .tables import Table, compare_methods, convergence_table
from .plots import (convergence_plot, cost_vs_error, invariant_drift,
                    residual_history, solution_vs_reference, PALETTE)
from .baseline import load_baseline, save_baseline, compare, Comparison
from .checks import (check_order, check_exact_on, check_invariant,
                     check_not_a_stub, report)
from .anim import Scene, Meter, render, render_frames


def source_fingerprint() -> str:
    """Вернуть короткий SHA-256 исходников для различения копий пакета."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]

__all__ = [
    "Result", "COST_UNITS",
    "Timing", "benchmark", "CallCounter", "timed",
    "max_error", "l2_error", "rel_error", "empirical_orders",
    "ConvergenceTable", "convergence", "ORDER_TOL",
    "MIN_POINTS_FOR_ORDER", "NOISE_FACTOR", "order_slack",
    "order_matches", "noise_floor", "count_nonfinite",
    "Table", "compare_methods", "convergence_table",
    "convergence_plot", "cost_vs_error", "invariant_drift",
    "residual_history", "solution_vs_reference", "PALETTE",
    "load_baseline", "save_baseline", "compare", "Comparison",
    "check_order", "check_exact_on", "check_invariant", "check_not_a_stub",
    "report",
    "Scene", "Meter", "render", "render_frames",
    "__version__", "source_fingerprint",
]
