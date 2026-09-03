"""Стенд для сравнительных GIF и статических раскадровок.

Импорт модуля не меняет backend Matplotlib. В headless-среде backend нужно
выбрать до импорта pyplot, например ``MPLBACKEND=Agg pytest``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

__all__ = ["Scene", "render", "render_frames", "Meter"]


@dataclass
class Meter:
    """Раздельные метрики численного счёта и служебного рендера."""

    errors: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    render_times: list[float] = field(default_factory=list)
    extra: dict[str, float | int | str] = field(default_factory=dict)

    def push(self, err: float | None = None,
             dt_compute: float | None = None) -> None:
        """Добавить ошибку и/или время только численного вычисления."""
        if err is not None:
            self.errors.append(float(err))
        if dt_compute is not None:
            value = float(dt_compute)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Meter: время счёта должно быть конечным и >= 0")
            self.times.append(value)

    def push_render(self, value: float) -> None:
        duration = float(value)
        if np.isfinite(duration) and duration >= 0:
            self.render_times.append(duration)

    @property
    def last_error(self) -> float:
        return self.errors[-1] if self.errors else float("nan")

    @property
    def max_error(self) -> float:
        finite = np.asarray(self.errors, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.max(finite)) if finite.size else float("nan")

    @property
    def ms_per_frame(self) -> float:
        """Среднее время численного счёта, не отрисовки."""
        return 1e3 * float(np.mean(self.times)) if self.times else float("nan")

    @property
    def render_ms_per_frame(self) -> float:
        return (1e3 * float(np.mean(self.render_times))
                if self.render_times else float("nan"))

    def sparkline(self, width: int = 24) -> str:
        if width < 1:
            raise ValueError("Meter.sparkline: width должен быть >= 1")
        finite = np.asarray(self.errors, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        indices = np.linspace(0, finite.size - 1, min(width, finite.size)).astype(int)
        values = np.log10(np.maximum(np.abs(finite[indices]), 1e-300))
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-12:
            return blocks[0] * values.size
        levels = ((values - low) / (high - low) * (len(blocks) - 1)).astype(int)
        return "".join(blocks[level] for level in levels)


class Scene:
    """Одна панель: наследник реализует ``setup`` и ``draw``."""

    def __init__(self, title: str, meter: Meter | None = None):
        self.title = str(title)
        self.meter = meter if meter is not None else Meter()

    def setup(self, ax) -> None:  # pragma: no cover
        raise NotImplementedError

    def draw(self, ax, k: int, t: float) -> Sequence:  # pragma: no cover
        raise NotImplementedError

    def status(self, k: int, t: float) -> str:
        meter = self.meter
        parts: list[str] = []
        if meter.errors:
            parts.append(
                f"ошибка {meter.last_error:.2e} (макс {meter.max_error:.2e})"
            )
        if meter.times:
            parts.append(f"счёт {meter.ms_per_frame:.2f} мс/кадр")
        if meter.render_times:
            parts.append(f"рендер {meter.render_ms_per_frame:.2f} мс/кадр")
        parts.extend(f"{key}: {value}" for key, value in meter.extra.items())
        return " · ".join(parts)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} должен быть целым числом >= 1")
    return int(value)


def render(scenes: Sequence[Scene], n_frames: int, dt: float = 1.0,
           out: str = "animation.gif", fps: int = 20,
           suptitle: str | None = None,
           hud: Callable[[int, float], str] | None = None,
           figsize: tuple[float, float] | None = None, dpi: int = 90,
           facecolor: str = "white", textcolor: str = "#222222") -> str:
    """Отрендерить непустой набор сцен бок о бок в GIF."""
    scenes = list(scenes)
    if not scenes:
        raise ValueError("render: нужен хотя бы один Scene")
    if any(not isinstance(scene, Scene) for scene in scenes):
        raise TypeError("render: все элементы scenes должны наследовать Scene")
    frame_count = _positive_integer(n_frames, "n_frames")
    fps_value = _positive_integer(fps, "fps")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("render: dt должен быть положительным и конечным")
    destination = Path(out)
    if destination.suffix.lower() != ".gif":
        raise ValueError("render: выходной файл должен иметь расширение .gif")
    destination.parent.mkdir(parents=True, exist_ok=True)

    count = len(scenes)
    if figsize is None:
        figsize = (4.6 * count, max(5.0, 4.55 + 0.32 * count))
    fig, axes_value = plt.subplots(1, count, figsize=figsize, dpi=dpi)
    axes = [axes_value] if count == 1 else list(np.ravel(axes_value))
    fig.patch.set_facecolor(facecolor)

    try:
        for ax, scene in zip(axes, scenes):
            scene.setup(ax)
            ax.set_title(scene.title, fontsize=11, pad=8, color=textcolor)

        bottom = min(0.42, 0.12 + 0.035 * (count + 1))
        hud_text = fig.text(
            0.5, 0.035, "", ha="center", va="bottom", fontsize=9,
            family="monospace", color=textcolor,
        )
        if suptitle:
            fig.suptitle(suptitle, fontsize=13, y=0.985, color=textcolor)
        fig.subplots_adjust(
            bottom=bottom, top=0.84 if suptitle else 0.92,
            left=0.06, right=0.98, wspace=0.22,
        )

        def initialize():
            hud_text.set_text("")
            return [hud_text]

        def draw_frame(index: int):
            model_time = index * dt_value
            artists: list = []
            for ax, scene in zip(axes, scenes):
                started = time.perf_counter()
                drawn = scene.draw(ax, index, model_time)
                scene.meter.push_render(time.perf_counter() - started)
                if drawn is not None:
                    artists.extend(list(drawn))
            if hud is not None:
                line = str(hud(index, model_time))
            else:
                statuses = [
                    f"{scene.title}: {scene.status(index, model_time)}"
                    for scene in scenes if scene.status(index, model_time)
                ]
                line = (
                    f"кадр {index + 1}/{frame_count} · t = {model_time:.3f}"
                    + (("\n" + "\n".join(statuses)) if statuses else "")
                )
            hud_text.set_text(line)
            artists.append(hud_text)
            return artists

        animation = FuncAnimation(
            fig, draw_frame, frames=frame_count, init_func=initialize,
            blit=False, interval=1000 / fps_value,
        )
        animation.save(
            destination, writer=PillowWriter(fps=fps_value),
            savefig_kwargs={"facecolor": facecolor},
        )
    finally:
        plt.close(fig)
    return str(destination)


def render_frames(scene_factories: Sequence[Callable[[], Scene]],
                  frame_indices: Sequence[int], dt: float = 1.0,
                  out: str = "frames.png", suptitle: str | None = None,
                  dpi: int = 110) -> str:
    """Статическая раскадровка: строки — кадры, столбцы — сцены."""
    factories = list(scene_factories)
    if not factories:
        raise ValueError("render_frames: нужна хотя бы одна фабрика сцен")
    indices = list(frame_indices)
    if not indices:
        raise ValueError("render_frames: нужен хотя бы один номер кадра")
    if any(isinstance(k, bool) or int(k) != k or k < 0 for k in indices):
        raise ValueError("render_frames: номера кадров должны быть целыми и >= 0")
    indices = [int(k) for k in indices]
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("render_frames: dt должен быть положительным и конечным")
    destination = Path(out)
    if not destination.suffix:
        raise ValueError("render_frames: у выходного файла должно быть расширение")
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows, columns = len(indices), len(factories)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.8 * rows),
        dpi=dpi, squeeze=False,
    )
    fig.patch.set_facecolor("white")
    try:
        for column, factory in enumerate(factories):
            for row, frame_index in enumerate(indices):
                scene = factory()
                if not isinstance(scene, Scene):
                    raise TypeError("render_frames: фабрика должна возвращать Scene")
                ax = axes[row][column]
                scene.setup(ax)
                for step in range(frame_index + 1):
                    scene.draw(ax, step, step * dt_value)
                if row == 0:
                    ax.set_title(scene.title, fontsize=11)
                if column == 0:
                    ax.set_ylabel(f"кадр {frame_index}", fontsize=9)
        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96) if suptitle else None)
        fig.savefig(destination, facecolor="white")
    finally:
        plt.close(fig)
    return str(destination)
