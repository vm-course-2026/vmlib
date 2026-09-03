"""Повторяемые замеры времени и явный подсчёт вызовов."""

from __future__ import annotations

import functools
import statistics
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import numpy as np

__all__ = ["Timing", "benchmark", "CallCounter", "timed"]


@dataclass
class Timing:
    """Результат серии замеров одной функции."""

    best: float
    median: float
    mean: float
    spread: float
    runs: list[float]
    value: Any = None

    def __repr__(self) -> str:
        spread = f"{self.spread:.1%}" if np.isfinite(self.spread) else "н/д"
        return (
            f"Timing(best={self.best:.4g}s, median={self.median:.4g}s, "
            f"разброс={spread}, повторов={len(self.runs)})"
        )


def _reset(reset) -> None:
    if reset is None:
        return
    callback = reset.reset if hasattr(reset, "reset") else reset
    if not callable(callback):
        raise TypeError("benchmark: reset должен быть функцией или иметь метод reset()")
    callback()


def benchmark(func: Callable, *args, repeats: int = 5, warmup: int = 1,
              reset: Callable | object | None = None, **kwargs) -> Timing:
    """Замерить чистое время с прогревом и повторами.

    ``reset`` вызывается перед каждым прогоном. Передайте туда
    ``CallCounter`` (или его ``reset``), чтобы итоговый счётчик относился
    только к последнему замеру, а не суммировал прогрев и повторы.
    """
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
        raise ValueError("benchmark: repeats должен быть целым числом >= 1")
    if isinstance(warmup, bool) or int(warmup) != warmup or warmup < 0:
        raise ValueError("benchmark: warmup должен быть целым числом >= 0")
    for _ in range(int(warmup)):
        _reset(reset)
        func(*args, **kwargs)

    runs: list[float] = []
    value = None
    for _ in range(int(repeats)):
        _reset(reset)
        started = time.perf_counter()
        value = func(*args, **kwargs)
        runs.append(time.perf_counter() - started)

    median = statistics.median(runs)
    spread = ((median - min(runs)) / median
              if len(runs) > 1 and median > 0 else float("nan"))
    return Timing(
        best=min(runs), median=median, mean=statistics.fmean(runs),
        spread=spread, runs=runs, value=value,
    )


class CallCounter:
    """Прозрачная обёртка, считающая вызовы функции."""

    def __init__(self, func: Callable, name: str = ""):
        self._func = func
        self.name = name or getattr(func, "__name__", "func")
        self.count = 0
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._func(*args, **kwargs)

    def __get__(self, instance, owner):
        return self if instance is None else MethodType(self, instance)

    def reset(self) -> None:
        self.count = 0

    def __repr__(self) -> str:
        return f"CallCounter({self.name}, count={self.count})"


class timed:
    """Одноразовый контекстный менеджер для чистого времени счёта."""

    def __init__(self):
        self.elapsed = float("nan")
        self._active = False
        self._used = False
        self._started = float("nan")

    def __enter__(self):
        if self._active:
            raise RuntimeError("timed: один объект нельзя вкладывать сам в себя")
        if self._used:
            raise RuntimeError("timed: создайте новый объект для нового замера")
        self._active = True
        self._used = True
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._started
        self._active = False
        return False
