"""Нормы ошибок и строгая диагностика сходимости.

Модуль намеренно не скрывает ``NaN``/``Inf`` и не разрешает NumPy
broadcasting при сравнении. Иначе разошедшийся метод или случайная форма
``(n, 1)`` вместо ``(n,)`` легко превращаются в правдоподобный зелёный отчёт.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ORDER_TOL", "MIN_POINTS_FOR_ORDER", "NOISE_FACTOR",
    "order_slack", "order_matches", "noise_floor", "count_nonfinite",
    "max_error", "l2_error", "rel_error", "empirical_orders",
    "ConvergenceTable", "convergence",
]

ORDER_TOL = 0.3
MIN_POINTS_FOR_ORDER = 4
NOISE_FACTOR = 64.0
ORDER_EXCESS_WARN = 1.0


def order_slack(expected: float, tol: float = ORDER_TOL) -> float:
    """Абсолютный допуск, не меньше 10% от ожидаемого порядка."""
    return max(float(tol), 0.1 * abs(float(expected)))


def order_matches(observed: float, expected: float,
                  tol: float = ORDER_TOL) -> bool:
    """Проверить порядок односторонним критерием снизу."""
    if not np.isfinite(observed):
        return False
    return bool(float(observed) >= float(expected) - order_slack(expected, tol))


def noise_floor(scale: float | None = None) -> float:
    """Нижняя оценка уровня шума округления с учётом масштаба решения."""
    value = 1.0 if scale is None else abs(float(scale))
    if not np.isfinite(value):
        raise ValueError("noise_floor: scale должен быть конечным")
    return NOISE_FACTOR * float(np.finfo(float).eps) * max(value, 1.0)


def _as_real_array(value, what: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise TypeError(f"{what}: комплексные значения не поддерживаются")
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what}: ожидался числовой массив") from exc


def count_nonfinite(values) -> int:
    """Число ``NaN`` и ``Inf`` в массиве."""
    return int(np.count_nonzero(~np.isfinite(_as_real_array(values, "values"))))


def _as_diff(approx, exact, what: str) -> np.ndarray:
    a = _as_real_array(approx, what)
    e = _as_real_array(exact, what)
    if a.shape != e.shape:
        raise ValueError(
            f"{what}: формы не совпадают — ваша {a.shape}, эталонная {e.shape}"
        )
    if a.size == 0:
        raise ValueError(f"{what}: пустой массив, сравнивать нечего")
    return a - e


def max_error(approx, exact) -> float:
    """Равномерная норма ``max(abs(approx - exact))``."""
    return float(np.max(np.abs(_as_diff(approx, exact, "max_error"))))


def l2_error(approx, exact, h=None, ndim: int | None = None) -> float:
    """Дискретная L2-норма ошибки.

    Без ``h`` возвращается RMS. Для скалярного шага используется множитель
    ``h**(ndim/2)``; ``ndim`` по умолчанию берётся из формы поля. Если поле
    многомерной сетки развёрнуто в вектор, размерность нужно указать явно.
    Последовательность шагов задаёт анизотропную меру ячейки.
    """
    diff = _as_diff(approx, exact, "l2_error")
    norm = float(np.linalg.norm(diff.ravel()))
    if h is None:
        return norm / float(np.sqrt(diff.size))
    hs = _as_real_array(h, "l2_error").ravel()
    if hs.size == 0 or not np.all(np.isfinite(hs)) or np.any(hs <= 0):
        raise ValueError(
            f"l2_error: шаг сетки должен быть положительным и конечным, получено {h!r}"
        )
    if hs.size > 1:
        if ndim is not None and int(ndim) != hs.size:
            raise ValueError(
                f"l2_error: передано {hs.size} шагов по осям, а ndim={ndim}"
            )
        cell_measure = float(np.prod(hs))
    else:
        dim = diff.ndim if ndim is None else int(ndim)
        if dim < 1:
            raise ValueError("l2_error: размерность сетки должна быть >= 1")
        cell_measure = float(hs[0]) ** dim
    return norm * float(np.sqrt(cell_measure))


def rel_error(approx, exact, ord=None) -> float:
    """Относительная ошибка; для нулевого эталона возвращается ``nan``."""
    diff = _as_diff(approx, exact, "rel_error")
    reference = _as_real_array(exact, "rel_error")
    denominator = float(np.linalg.norm(reference, ord=ord))
    if denominator == 0:
        return float("nan")
    return float(np.linalg.norm(diff, ord=ord) / denominator)


def empirical_orders(hs, errors) -> np.ndarray:
    """Эмпирические порядки между соседними сетками."""
    h = _as_real_array(hs, "empirical_orders: hs").ravel()
    e = _as_real_array(errors, "empirical_orders: errors").ravel()
    if h.size != e.size:
        raise ValueError(
            f"empirical_orders: {h.size} шагов и {e.size} ошибок — длины должны совпадать"
        )
    if h.size < 2:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("empirical_orders: шаги должны быть положительными и конечными")
    ratios = h[:-1] / h[1:]
    if np.any(np.isclose(ratios, 1.0)):
        raise ValueError("empirical_orders: соседние шаги должны различаться")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e[:-1] / e[1:]) / np.log(ratios)


@dataclass
class ConvergenceTable:
    """Серия шагов, ошибок и наблюдаемых порядков с диагностикой."""

    hs: np.ndarray
    errors: np.ndarray
    orders: np.ndarray
    expected: float | None = None
    label: str = ""
    tol: float = ORDER_TOL
    scale: float | None = None

    _STALL_RATIO = 0.7

    def __post_init__(self) -> None:
        self.hs = _as_real_array(self.hs, "таблица сходимости: hs").ravel()
        self.errors = _as_real_array(self.errors, "таблица сходимости: errors").ravel()
        self.orders = _as_real_array(self.orders, "таблица сходимости: orders").ravel()
        if self.errors.size != self.hs.size:
            raise ValueError(
                f"таблица сходимости: {self.hs.size} шагов и "
                f"{self.errors.size} ошибок — длины должны совпадать"
            )
        expected_orders = max(self.hs.size - 1, 0)
        if self.orders.size != expected_orders:
            raise ValueError(
                f"таблица сходимости: для {self.hs.size} шагов нужно "
                f"{expected_orders} порядков, получено {self.orders.size}"
            )

    @property
    def n_nonfinite(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.errors)))

    def _tail_plateau(self) -> float:
        if self.errors.size < 2:
            return 0.0
        plateau: list[float] = []
        for index in range(self.errors.size - 1, 0, -1):
            previous, current = self.errors[index - 1], self.errors[index]
            if not (np.isfinite(previous) and np.isfinite(current)):
                break
            if not (self.hs[index] < self.hs[index - 1] and previous > 0):
                break
            if current / previous <= self._STALL_RATIO:
                break
            plateau.append(float(current))
        return 1.25 * float(np.median(plateau)) if plateau else 0.0

    def noise_level(self) -> float:
        return max(noise_floor(self.scale), self._tail_plateau())

    def _reliable(self) -> np.ndarray:
        return np.isfinite(self.errors) & (self.errors > self.noise_level())

    @property
    def discarded(self) -> np.ndarray:
        """Индексы конечных точек, отброшенных как численный шум."""
        reliable = self._reliable()
        return np.flatnonzero(np.isfinite(self.errors) & ~reliable)

    @property
    def stagnating(self) -> bool:
        return self._tail_plateau() > 0

    @property
    def hit_machine_precision(self) -> bool:
        """Совместимое имя: достигнут ли фактический предел точности."""
        return bool(self.discarded.size or self.stagnating)

    @property
    def asymptotic_order(self) -> float:
        if self.n_nonfinite or self.hs.size < MIN_POINTS_FOR_ORDER:
            return float("nan")
        reliable = self._reliable()
        valid_pairs = reliable[:-1] & reliable[1:] & np.isfinite(self.orders)
        values = self.orders[valid_pairs]
        if values.size < 3:
            return float("nan")
        return float(np.median(values[-3:]))

    def reason(self) -> str:
        messages: list[str] = []
        if self.n_nonfinite:
            bad = np.flatnonzero(~np.isfinite(self.errors))
            bad_h = ", ".join(f"{x:.3g}" for x in self.hs[bad])
            messages.append(
                f"метод не досчитал на {bad.size} из {self.errors.size} сеток "
                f"(h = {bad_h}): ошибка NaN/Inf, серия невалидна"
            )
        if self.hs.size < MIN_POINTS_FOR_ORDER:
            messages.append(
                f"точек в серии {self.hs.size}, нужно не менее {MIN_POINTS_FOR_ORDER}"
            )
        if self.discarded.size:
            bad_h = ", ".join(f"{x:.3g}" for x in self.hs[self.discarded])
            messages.append(
                f"точки h = {bad_h} ниже уровня шума {self.noise_level():.2e} "
                "и исключены из оценки"
            )
        if self.stagnating:
            messages.append("на хвосте ошибка почти не убывает — достигнут предел точности")
        if self.hs.size > 1 and np.any(np.diff(self.hs) >= 0):
            messages.append("шаги должны строго убывать для оценки асимптотики")
        if self.orders.size and np.all(np.isfinite(self.orders)) and np.all(self.orders < 0):
            messages.append("вместо шагов h, вероятно, переданы числа узлов N")
        observed = self.asymptotic_order
        if (self.expected is not None and np.isfinite(observed)
                and observed > self.expected + ORDER_EXCESS_WARN):
            messages.append(
                f"порядок {observed:.2f} заметно выше теоретического "
                f"{self.expected:g}: проверьте сверхсходимость и длину серии"
            )
        reliable = self._reliable()
        if (self.hs.size >= MIN_POINTS_FOR_ORDER and not self.n_nonfinite
                and np.count_nonzero(reliable[:-1] & reliable[1:]) < 3):
            messages.append("после фильтра шума осталось меньше трёх надёжных порядков")
        return "; ".join(dict.fromkeys(messages))

    def verdict(self) -> str:
        observed = self.asymptotic_order
        why = self.reason()
        if self.expected is None:
            main = (f"наблюдаемый порядок ≈ {observed:.2f}"
                    if np.isfinite(observed) else "наблюдаемый порядок не определён")
        else:
            passed = order_matches(observed, self.expected, self.tol)
            state = "совпадает с теорией" if passed else "НЕ совпадает с теорией"
            main = (f"наблюдаемый порядок ≈ {observed:.2f}, "
                    f"ожидался {self.expected:g} — {state}")
        return main + (f"\n⚠ {why}" if why else "")

    def to_rows(self) -> list[dict]:
        return [
            {
                "h": float(h),
                "ошибка": float(error),
                "порядок": float(self.orders[i - 1]) if i else None,
            }
            for i, (h, error) in enumerate(zip(self.hs, self.errors))
        ]


def convergence(hs, errors, expected: float | None = None, label: str = "",
                tol: float = ORDER_TOL,
                scale: float | None = None) -> ConvergenceTable:
    """Собрать и проверить таблицу сходимости."""
    hs_array = _as_real_array(hs, "convergence: hs").ravel()
    errors_array = _as_real_array(errors, "convergence: errors").ravel()
    return ConvergenceTable(
        hs_array, errors_array, empirical_orders(hs_array, errors_array),
        expected, label, tol, scale,
    )
