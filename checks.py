"""Самопроверки с отказом на неполных и невалидных данных."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .errors import (
    MIN_POINTS_FOR_ORDER,
    ORDER_TOL,
    convergence,
    max_error,
    noise_floor,
    order_matches,
    order_slack,
)

__all__ = [
    "check_order", "check_exact_on", "check_invariant",
    "check_not_a_stub", "report",
]

_GREEN, _RED, _YELLOW = "✓", "✗", "⚠"


def _say(ok: bool, text: str, warn: bool = False) -> bool:
    mark = _YELLOW if warn else (_GREEN if ok else _RED)
    print(f"{mark} {text}")
    return bool(ok)


def check_order(solver: Callable, exact, params: Sequence, expected: float,
                param_to_h: Callable[[float], float] | None = None,
                name: str = "метод", tol: float = ORDER_TOL,
                norm: Callable = max_error) -> bool:
    """Проверить наблюдаемый порядок минимум по четырём сеткам.

    Проверка не проходит, если хотя бы один прогон дал ``NaN``/``Inf``.
    Таблица исходных ошибок печатается всегда, чтобы вердикт был проверяемым.
    """
    if param_to_h is None:
        param_to_h = lambda p: 1.0 / p if p > 1 else float(p)
    parameters = list(params)
    if len(parameters) < MIN_POINTS_FOR_ORDER:
        return _say(
            False,
            f"порядок {name}: сеток {len(parameters)}, нужно не менее "
            f"{MIN_POINTS_FOR_ORDER}",
        )

    hs: list[float] = []
    errors: list[float] = []
    scale = 0.0
    for parameter in parameters:
        approximation = np.asarray(solver(parameter))
        reference = np.asarray(exact(parameter) if callable(exact) else exact)
        hs.append(float(param_to_h(parameter)))
        errors.append(float(norm(approximation, reference)))
        if reference.size and np.all(np.isfinite(reference)):
            scale = max(scale, float(np.max(np.abs(reference))))

    hs_array = np.asarray(hs, dtype=float)
    errors_array = np.asarray(errors, dtype=float)
    bad = ~np.isfinite(errors_array)
    if np.any(bad):
        failed_params = ", ".join(
            str(parameter) for parameter, is_bad in zip(parameters, bad) if is_bad
        )
        _say(
            False,
            f"порядок {name}: NaN/Inf на параметрах {failed_params}; "
            "о порядке говорить нельзя",
        )
        _print_series(hs_array, errors_array)
        return False

    table = convergence(
        hs_array, errors_array, expected=expected, label=name,
        tol=tol, scale=scale or None,
    )
    observed = table.asymptotic_order
    passed = order_matches(observed, expected, tol)
    _say(
        passed,
        f"порядок {name}: наблюдаемый {observed:.2f}, ожидался {expected:g} "
        f"(нижняя граница {expected - order_slack(expected, tol):.2f})",
    )
    _print_series(hs_array, errors_array, table.orders)
    if table.reason():
        _say(True, table.reason(), warn=True)
    return passed


def _print_series(hs: np.ndarray, errors: np.ndarray,
                  orders: np.ndarray | None = None) -> None:
    print("  шаги:  ", "  ".join(f"{h:.3g}" for h in hs))
    print("  ошибки:", "  ".join(f"{error:.2e}" for error in errors))
    if orders is not None:
        print("  порядки:", "  ".join(f"{order:.2f}" for order in orders))


def check_exact_on(approx_fn: Callable, cases: Mapping,
                   tol: float = 1e-12,
                   name: str = "точность на контрольных функциях") -> bool:
    """Проверить точность на обязательных контрольных случаях.

    Используется смешанный допуск ``tol + tol*abs(expected)``. Пустой,
    нефинитный результат или несовпадение форм не проходит проверку.
    """
    if not cases:
        return _say(False, f"{name}: контрольные случаи не заданы")
    all_ok = True
    print(f"{name}:")
    for description, (args, expected) in cases.items():
        try:
            got = np.asarray(approx_fn(*args), dtype=float)
            reference = np.asarray(expected, dtype=float)
            valid = (
                got.shape == reference.shape and got.size > 0
                and np.all(np.isfinite(got)) and np.all(np.isfinite(reference))
            )
            error = float(np.max(np.abs(got - reference))) if valid else float("inf")
            limit = tol + tol * float(np.max(np.abs(reference))) if valid else tol
            passed = valid and error <= limit
        except Exception as exc:
            error, passed = float("inf"), False
            print(f"  {_RED} {description}: {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        all_ok = all_ok and passed
        print(
            f"  {_GREEN if passed else _RED} {description}: "
            f"макс. отклонение {error:.2e}, допуск {limit:.2e}"
        )
    return bool(all_ok)


def check_invariant(t, values, tol_drift: float = 1e-3,
                    name: str = "инвариант") -> bool:
    """Проверить дрейф инварианта на согласованной временной сетке."""
    times = np.asarray(t, dtype=float).ravel()
    invariant = np.asarray(values, dtype=float).ravel()
    if times.size == 0 or invariant.size == 0:
        return _say(False, f"{name}: нет данных")
    if times.size != invariant.size:
        return _say(
            False,
            f"{name}: времён {times.size}, значений {invariant.size}",
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(invariant)):
        return _say(False, f"{name}: в данных есть NaN/Inf")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        return _say(False, f"{name}: время должно строго возрастать")

    data_scale = max(1.0, float(np.max(np.abs(invariant))))
    denominator = max(abs(float(invariant[0])), noise_floor(data_scale))
    drift_values = np.abs(invariant - invariant[0]) / denominator
    drift = float(np.max(drift_values))
    passed = drift <= tol_drift
    _say(
        passed,
        f"{name}: максимальный относительный дрейф {drift:.3e} "
        f"(допуск {tol_drift:.1e})",
    )
    if passed and times.size >= 6:
        slope = float(np.polyfit(times - times[0], drift_values, 1)[0])
        total_trend = max(0.0, slope * (times[-1] - times[0]))
        if total_trend > 0.5 * tol_drift:
            _say(True, f"{name}: заметен систематический рост дрейфа", warn=True)
    return passed


def check_not_a_stub(func: Callable, *args, not_equal_to=None, **kwargs) -> bool:
    """Быстрая smoke-проверка реализации.

    Она ловит ``None``, нули, нефинитные числа, отказ ``Result`` и, если
    передан ``not_equal_to``, точное совпадение с известной заглушкой. Это не
    заменяет предметные тесты корректности.
    """
    name = getattr(func, "__name__", "функция")
    try:
        output = func(*args, **kwargs)
    except NotImplementedError:
        return _say(False, f"{name}: не реализована (NotImplementedError)")
    except Exception as exc:
        return _say(False, f"{name}: падает с {type(exc).__name__}: {exc}")
    if output is None:
        return _say(False, f"{name}: возвращает None — TODO не заполнен")
    if hasattr(output, "success") and not bool(output.success):
        message = getattr(output, "message", "")
        return _say(False, f"{name}: метод сообщил об отказе — {message}")
    try:
        array = np.asarray(output.x if hasattr(output, "x") else output, dtype=float)
    except (TypeError, ValueError) as exc:
        return _say(False, f"{name}: результат не является числовым: {exc}")
    if array.size == 0:
        return _say(False, f"{name}: возвращает пустой результат")
    if not np.all(np.isfinite(array)):
        return _say(False, f"{name}: в результате есть NaN или Inf")
    if np.all(array == 0):
        return _say(False, f"{name}: возвращает одни нули — похоже на заглушку")
    if not_equal_to is not None:
        try:
            stub = np.asarray(not_equal_to, dtype=float)
            if array.shape == stub.shape and np.array_equal(array, stub):
                return _say(False, f"{name}: результат совпал с известной заглушкой")
        except (TypeError, ValueError):
            pass
    return _say(True, f"{name}: возвращает непустые конечные значения")


def report(checks: Mapping[str, object], strict: bool = True) -> bool:
    """Свести проверки в итоговый вердикт.

    В строгом режиме принимаются только ``bool`` и объекты с полем
    ``passed`` (например ``Comparison``), чтобы произвольный непустой объект
    не превратился в ложную галочку.
    """
    if not checks:
        _say(False, "нет ни одной проверки")
        return False
    normalized: dict[str, bool] = {}
    for name, value in checks.items():
        if isinstance(value, (bool, np.bool_)):
            normalized[name] = bool(value)
        elif hasattr(value, "passed"):
            normalized[name] = bool(value.passed)
        elif strict:
            raise TypeError(
                f"report: проверка {name!r} вернула {type(value).__name__}, "
                "ожидался bool или Comparison"
            )
        else:
            normalized[name] = bool(value)

    total = len(normalized)
    passed = sum(normalized.values())
    print("\n" + "─" * 56)
    for name, ok in normalized.items():
        print(f"  {_GREEN if ok else _RED} {name}")
    print("─" * 56)
    print(
        f"Пройдено {passed} из {total}."
        + ("  Всё зелёное." if passed == total else "  Есть что доделать.")
    )
    return passed == total
