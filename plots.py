"""Стандартные диагностические графики численных экспериментов."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .errors import noise_floor

__all__ = [
    "convergence_plot", "cost_vs_error", "invariant_drift",
    "residual_history", "solution_vs_reference", "PALETTE",
]

PALETTE = ["#2166ac", "#b2182b", "#1a7f37", "#d97706", "#6a3d9a", "#00838f"]


def _style(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, framealpha=0.9)


def _save(ax, save: str | None, close: bool, own: bool) -> None:
    if save:
        destination = Path(save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(destination, bbox_inches="tight", facecolor="white")
    if close and own:
        plt.close(ax.figure)


def _positive_xy(x, y, label: str) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()
    if x_array.size != y_array.size:
        raise ValueError(
            f"{label}: по x {x_array.size} точек, по y {y_array.size}"
        )
    keep = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0) & (y_array > 0)
    dropped = int(keep.size - np.count_nonzero(keep))
    if dropped:
        warnings.warn(
            f"{label}: отброшено {dropped} точек с NaN/Inf или неположительными "
            "координатами для логарифмической оси",
            RuntimeWarning,
            stacklevel=3,
        )
    return x_array[keep], y_array[keep]


def convergence_plot(series: Mapping[str, tuple[Sequence, Sequence]],
                     expected_orders: Mapping[str, float] | None = None,
                     xlabel: str = "шаг $h$",
                     ylabel: str = r"ошибка $\|u-u_h\|_\infty$",
                     title: str = "Сходимость", ax=None,
                     save: str | None = None, close: bool = False):
    """Лог-лог график ошибки; возвращает ``Axes`` для совместимости."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110)
    all_errors: list[float] = []
    for index, (name, (hs, errors)) in enumerate(series.items()):
        x, y = _positive_xy(hs, errors, name)
        if x.size == 0:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        all_errors.extend(y.tolist())
        color = PALETTE[index % len(PALETTE)]
        ax.plot(x, y, "o-", color=color, lw=1.6, ms=5, label=name)
        if expected_orders and name in expected_orders:
            expected = float(expected_orders[name])
            anchor = float(np.median(y / x ** expected))
            ax.plot(
                x, anchor * x ** expected, "--", color=color, lw=1,
                alpha=0.6, label=f"{name}: $h^{{{expected:g}}}$",
            )
    if all_errors:
        smallest = min(all_errors)
        floor = noise_floor(max(all_errors))
        if smallest <= 20 * floor:
            ax.axhline(floor, color="grey", lw=0.8, ls=":", alpha=0.7,
                       label="уровень округления")
    _style(ax, xlabel, ylabel, title, logx=True, logy=True)
    _save(ax, save, close, own)
    return ax


def cost_vs_error(results: Mapping[str, Sequence],
                  title: str = "Цена точности", ax=None,
                  xlabel: str = "число вычислений", save: str | None = None,
                  close: bool = False):
    """Ошибка против положительной стоимости в согласованных единицах."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110)
    units: set[str] = set()
    for index, (name, values) in enumerate(results.items()):
        points: list[tuple[float, float]] = []
        for result in values:
            cost = result.cost
            error = result.max_error
            if cost is None or error is None:
                continue
            if not (np.isfinite(cost) and np.isfinite(error) and cost > 0 and error > 0):
                warnings.warn(
                    f"{name}: пропущена точка с неположительной/нефинитной "
                    "стоимостью или ошибкой",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            unit = getattr(result, "resolved_cost_unit", None)
            if unit:
                units.add(str(unit))
            points.append((float(cost), float(error)))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        x, y = zip(*points)
        ax.plot(
            x, y, "o-", color=PALETTE[index % len(PALETTE)],
            lw=1.6, ms=5, label=name,
        )
    if len(units) > 1:
        if own:
            plt.close(ax.figure)
        raise ValueError(
            "cost_vs_error: смешаны несопоставимые единицы стоимости: "
            + ", ".join(sorted(units))
        )
    actual_xlabel = xlabel
    if xlabel == "число вычислений" and len(units) == 1:
        actual_xlabel = next(iter(units))
    _style(ax, actual_xlabel, "ошибка", title, logx=True, logy=True)
    _save(ax, save, close, own)
    return ax


def invariant_drift(series: Mapping[str, tuple[Sequence, Sequence]],
                    xlabel: str = "время",
                    ylabel: str = "относительный дрейф",
                    title: str = "Дрейф инварианта", ax=None,
                    save: str | None = None, close: bool = False):
    """Нормированный дрейф инварианта во времени."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.8, 4.4), dpi=110)
    for index, (name, (times, values)) in enumerate(series.items()):
        t = np.asarray(times, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if t.size == 0 or v.size == 0:
            warnings.warn(f"{name}: пустая серия инварианта", RuntimeWarning, stacklevel=2)
            continue
        if t.size != v.size:
            raise ValueError(f"{name}: времён {t.size}, значений {v.size}")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
            raise ValueError(f"{name}: в серии инварианта есть NaN/Inf")
        scale = max(abs(float(v[0])), noise_floor(float(np.max(np.abs(v)))))
        ax.plot(
            t, (v - v[0]) / scale, lw=1.4,
            color=PALETTE[index % len(PALETTE)], label=name,
        )
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    _style(ax, xlabel, ylabel, title)
    _save(ax, save, close, own)
    return ax


def residual_history(results: Mapping[str, object],
                     title: str = "История невязки", ax=None,
                     save: str | None = None, close: bool = False):
    """Истории положительной нормы невязки по итерациям."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.4, 4.4), dpi=110)
    for index, (name, result) in enumerate(results.items()):
        raw = getattr(result, "residual_norms", None)
        if raw is None:
            continue
        residuals = np.asarray(raw, dtype=float).ravel()
        if residuals.size == 0:
            continue
        keep = np.isfinite(residuals) & (residuals > 0)
        if not np.all(keep):
            warnings.warn(
                f"{name}: неположительные или нефинитные невязки не показаны",
                RuntimeWarning,
                stacklevel=2,
            )
        indices = np.arange(residuals.size)[keep]
        if indices.size:
            ax.semilogy(
                indices, residuals[keep], "o-", ms=4, lw=1.5,
                color=PALETTE[index % len(PALETTE)], label=name,
            )
    _style(ax, "номер итерации", r"$\|F(u_s)\|$", title)
    _save(ax, save, close, own)
    return ax


def solution_vs_reference(x, mine, reference, title: str = "Решение и эталон",
                          labels=("ваша реализация", "эталон"),
                          save: str | None = None, close: bool = False):
    """Решение и разность на двух вертикальных панелях; возвращает Figure."""
    coordinates = np.asarray(x, dtype=float).ravel()
    approximation = np.asarray(mine, dtype=float).ravel()
    exact = np.asarray(reference, dtype=float).ravel()
    if not (coordinates.size == approximation.size == exact.size) or coordinates.size == 0:
        raise ValueError("solution_vs_reference: x, mine и reference должны иметь одну непустую длину")
    if not (np.all(np.isfinite(coordinates)) and np.all(np.isfinite(approximation))
            and np.all(np.isfinite(exact))):
        raise ValueError("solution_vs_reference: данные содержат NaN/Inf")
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.8, 6.0), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    top.plot(coordinates, exact, lw=2.6, color="#bbbbbb", label=labels[1])
    top.plot(coordinates, approximation, lw=1.4, color=PALETTE[0], label=labels[0])
    top.set_ylabel("значение")
    top.set_title(title, fontsize=11)
    top.grid(alpha=0.25)
    top.legend(fontsize=9)
    bottom.plot(coordinates, approximation - exact, lw=1.2, color=PALETTE[1])
    bottom.axhline(0, color="grey", lw=0.8, ls=":")
    bottom.set_xlabel("x")
    bottom.set_ylabel("разность")
    bottom.grid(alpha=0.25)
    fig.tight_layout()
    if save:
        destination = Path(save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return fig
