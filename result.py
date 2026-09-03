"""Единый, сериализуемый контракт результата численного метода."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Result", "COST_UNITS"]

COST_UNITS = {
    "matvec_count": "умножений на матрицу",
    "n_fevals": "вычислений правой части",
    "iterations": "итераций",
    "n_steps": "шагов",
}


def _positive_finite(value: Any) -> bool:
    """Проверить числовую стоимость без утечки ``TypeError`` наружу."""
    try:
        return bool(np.isfinite(value) and value > 0)
    except (TypeError, ValueError):
        return False


@dataclass
class Result:
    """Результат численного метода и измеренная цена решения."""

    x: Any = None
    success: bool = False
    message: str = ""
    elapsed_time: float = float("nan")

    residual_norms: list[float] | None = None
    step_norms: list[float] | None = None
    snapshots: list[Any] | None = None

    newton_iters: int | None = None
    gmres_iters: int | None = None
    iterations: int | None = None
    matvec_count: int | None = None
    n_fevals: int | None = None
    n_steps: int | None = None
    n_rejected: int | None = None

    max_error: float | None = None
    l2_error: float | None = None
    cond: float | None = None

    # Если график использует ``cost``, автор явно может выбрать единицу.
    # Без выбора остаётся совместимый приоритет, но resolved_cost_field
    # позволяет графику проверить, что разные методы измерены одинаково.
    cost_field: str | None = None
    cost_unit: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def resolved_cost_field(self) -> str | None:
        if self.cost_field is not None:
            if self.cost_field not in COST_UNITS:
                raise ValueError(
                    f"неизвестное cost_field={self.cost_field!r}; "
                    f"допустимы {', '.join(COST_UNITS)}"
                )
            return self.cost_field
        for name in COST_UNITS:
            value = getattr(self, name)
            if value is not None and _positive_finite(value):
                return name
        return None

    @property
    def resolved_cost_unit(self) -> str | None:
        field_name = self.resolved_cost_field
        return self.cost_unit or (COST_UNITS[field_name] if field_name else None)

    @property
    def cost(self) -> int | float | None:
        """Положительная стоимость в одной явно определимой единице."""
        field_name = self.resolved_cost_field
        if field_name is None:
            return None
        value = getattr(self, field_name)
        if value is None or not _positive_finite(value):
            return None
        return value

    def as_row(self, name: str = "") -> dict:
        """Одна строка сводной таблицы."""
        row: dict[str, Any] = {"метод": name} if name else {}
        row["успех"] = "да" if self.success else "НЕТ"
        for key, label in (
            ("max_error", "ошибка ∞"),
            ("l2_error", "ошибка L2"),
            ("iterations", "итераций"),
            ("newton_iters", "Ньютон"),
            ("gmres_iters", "GMRES"),
            ("matvec_count", "matvec"),
            ("n_fevals", "вызовов f"),
            ("n_steps", "шагов"),
            ("n_rejected", "отклонено"),
            ("cond", "cond"),
        ):
            value = getattr(self, key)
            if value is not None:
                row[label] = value
        row["время, с"] = self.elapsed_time
        if not self.success and self.message:
            row["причина"] = self.message
        return row

    def to_dict(self) -> dict:
        """Вернуть структуру, которую можно напрямую передать в JSON."""
        return _jsonable(asdict(self))

    def __repr__(self) -> str:
        bits = [f"success={bool(self.success)}"]
        if self.max_error is not None:
            bits.append(f"max_error={_number(self.max_error, '.3e')}")
        try:
            cost = self.cost
        except (TypeError, ValueError):
            cost = None
        if cost is not None:
            bits.append(f"cost={cost}")
        bits.append(f"time={_number(self.elapsed_time, '.4f')}s")
        suffix = f" — {self.message}" if self.message else ""
        return f"Result({', '.join(bits)}){suffix}"


def _number(value: Any, spec: str) -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError, OverflowError):
        return "н/д"


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
