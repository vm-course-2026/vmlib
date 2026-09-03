"""Загрузка эталонов и строгая сверка результатов."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import noise_floor

__all__ = ["load_baseline", "compare", "Comparison", "save_baseline"]


def _find_baseline_dir(start: str | os.PathLike | None = None) -> Path:
    here = Path(start or os.getcwd()).resolve()
    for candidate in [here, *here.parents]:
        directory = candidate / "baseline"
        if directory.is_dir():
            return directory
    raise FileNotFoundError(
        "Не найдена папка baseline/. Запускайте ноутбук из папки задания."
    )


def load_baseline(path: str | os.PathLike | None = None) -> dict:
    """Загрузить ``baseline.npz`` и ``baseline.json`` в один словарь."""
    directory = Path(path) if path is not None else _find_baseline_dir()
    if directory.is_file():
        directory = directory.parent
    output: dict = {}
    npz = directory / "baseline.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as archive:
            output.update({key: archive[key] for key in archive.files})
    metadata = directory / "baseline.json"
    if metadata.exists():
        output.update(json.loads(metadata.read_text(encoding="utf-8")))
    if not output:
        raise FileNotFoundError(
            f"В {directory} нет ни baseline.npz, ни baseline.json"
        )
    return output


def save_baseline(path: str | os.PathLike, arrays: dict | None = None,
                  meta: dict | None = None) -> None:
    """Сохранить эталон (служебная функция для авторов задания)."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(
            directory / "baseline.npz",
            **{key: np.asarray(value) for key, value in arrays.items()},
        )
    if meta:
        (directory / "baseline.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@dataclass
class Comparison:
    """Итог сверки одной величины с эталоном."""

    name: str
    max_abs: float
    max_rel: float
    tol: float
    passed: bool
    note: str = ""

    def __bool__(self) -> bool:
        return bool(self.passed)

    def __repr__(self) -> str:
        mark = "✓ СОВПАЛО" if self.passed else "✗ РАСХОЖДЕНИЕ"
        absolute = _metric(self.max_abs)
        relative = _metric(self.max_rel)
        text = (
            f"{mark}  {self.name}: макс. абс. {absolute}, "
            f"макс. отн. {relative} (допуск {self.tol:.1e})"
        )
        return text + (f"\n   {self.note}" if self.note else "")

    def _repr_html_(self) -> str:
        color = "#1a7f37" if self.passed else "#b42318"
        mark = "✓ совпало" if self.passed else "✗ расхождение"
        return (
            f'<div style="font-family:monospace;color:{color}">'
            f"<b>{html.escape(mark)}</b> &nbsp; {html.escape(self.name)}: "
            f"макс. абс. {_metric(self.max_abs)}, "
            f"макс. отн. {_metric(self.max_rel)} "
            f"(допуск {self.tol:.1e})"
            + (f"<br>{html.escape(self.note)}" if self.note else "")
            + "</div>"
        )


def _metric(value: float) -> str:
    return f"{value:.3e}" if np.isfinite(value) else str(value)


def _failed(name: str, tol: float, note: str, verbose: bool) -> Comparison:
    result = Comparison(name, float("inf"), float("inf"), tol, False, note)
    if verbose:
        print(result)
    return result


def compare(mine, reference, name: str = "", tol: float = 1e-6,
            verbose: bool = True) -> Comparison:
    """Сравнить массивы, скаляры или поля ``Result.x``.

    Каждая координата проверяется смешанным условием
    ``abs(a-b) <= atol + tol*abs(reference)``. ``atol`` равен максимуму
    ``tol`` и уровня машинного шума масштаба эталона. Любой ``NaN``/``Inf``,
    пустой результат, несовпадение форм или ``Result(success=False)`` — отказ.
    """
    label = name or "результат"
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("compare: tol должен быть неотрицательным и конечным")
    if hasattr(mine, "success") and not bool(mine.success):
        message = getattr(mine, "message", "") or "численный метод сообщил об отказе"
        return _failed(label, tol, message, verbose)

    try:
        raw_a = np.asarray(getattr(mine, "x", mine))
        raw_b = np.asarray(getattr(reference, "x", reference))
        if np.iscomplexobj(raw_a) or np.iscomplexobj(raw_b):
            return _failed(label, tol, "комплексные значения не поддерживаются", verbose)
        a = np.asarray(raw_a, dtype=float)
        b = np.asarray(raw_b, dtype=float)
    except (TypeError, ValueError) as exc:
        return _failed(label, tol, f"нечисловой результат: {exc}", verbose)

    if a.shape != b.shape:
        return _failed(
            label, tol, f"разная форма: ваша {a.shape}, эталон {b.shape}", verbose
        )
    if a.size == 0:
        return _failed(label, tol, "пустой результат, сравнивать нечего", verbose)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        mine_bad = int(np.count_nonzero(~np.isfinite(a)))
        ref_bad = int(np.count_nonzero(~np.isfinite(b)))
        return _failed(
            label, tol,
            f"NaN/Inf: в вашем результате {mine_bad}, в эталоне {ref_bad}",
            verbose,
        )

    diff = np.abs(a - b)
    scale = max(1.0, float(np.max(np.abs(b))))
    atol = max(float(tol), noise_floor(scale))
    limits = atol + float(tol) * np.abs(b)
    passed = bool(np.all(diff <= limits))
    max_abs = float(np.max(diff))
    significant = np.abs(b) > atol
    max_rel = (
        float(np.max(diff[significant] / np.abs(b[significant])))
        if np.any(significant) else 0.0
    )

    note = f"абсолютный допуск около нуля {atol:.2e}"
    if not passed:
        normalized = diff / np.maximum(limits, np.finfo(float).tiny)
        flat_index = int(np.argmax(normalized))
        index = np.unravel_index(flat_index, a.shape)
        note = (
            f"худшая точка: индекс {index}, ваше {a.flat[flat_index]:.6g}, "
            f"эталон {b.flat[flat_index]:.6g}; {note}"
        )
    result = Comparison(label, max_abs, max_rel, tol, passed, note)
    if verbose:
        print(result)
    return result
