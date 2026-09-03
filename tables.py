"""Безопасные HTML-, Markdown- и текстовые таблицы без pandas."""

from __future__ import annotations

import html
from typing import Any, Sequence

import numpy as np

__all__ = ["Table", "compare_methods", "convergence_table"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _special_number(value: float) -> str | None:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+∞"
    if np.isneginf(value):
        return "−∞"
    return None


def _formatter(values: Sequence[Any]):
    finite_floats = [
        abs(float(value)) for value in values
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value != 0
    ]
    scientific = bool(finite_floats) and (
        min(finite_floats) < 1e-4 or max(finite_floats) >= 1e5
    )

    def format_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "да" if bool(value) else "нет"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            number = float(value)
            special = _special_number(number)
            if special is not None:
                return special
            if number == 0:
                return "0.000e+00" if scientific else "0"
            return f"{number:.3e}" if scientific else f"{number:.4g}"
        return str(value)

    return format_value


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


class Table:
    """Таблица из последовательности словарей с устойчивым форматированием."""

    def __init__(self, rows: Sequence[dict], title: str = "",
                 caption: str = ""):
        raw_rows = list(rows)
        if any(not isinstance(row, dict) for row in raw_rows):
            raise TypeError("Table: каждая строка должна быть словарём")
        # Строковые имена столбцов — часть публичного контракта. Нормализуем
        # сами строки, а не только список columns, иначе ключ ``1`` потеряется
        # при последующем ``row.get("1")``.
        self.rows = [
            {str(key): value for key, value in row.items()}
            for row in raw_rows
        ]
        self.title = str(title)
        self.caption = str(caption)
        self.columns: list[str] = []
        for row in self.rows:
            for key in row:
                column = str(key)
                if column not in self.columns:
                    self.columns.append(column)
        self._formats = {
            column: _formatter([row.get(column) for row in self.rows])
            for column in self.columns
        }
        self._numeric = {
            column: any(_is_number(row.get(column)) for row in self.rows)
            for column in self.columns
        }

    def _cell(self, row: dict, column: str) -> str:
        return self._formats[column](row.get(column))

    def to_markdown(self) -> str:
        if not self.rows:
            return "_(пусто)_"
        head = "| " + " | ".join(_escape_markdown(c) for c in self.columns) + " |"
        align = "|" + "|".join(
            "---:" if self._numeric[c] else "---" for c in self.columns
        ) + "|"
        body = [
            "| " + " | ".join(
                _escape_markdown(self._cell(row, column))
                for column in self.columns
            ) + " |"
            for row in self.rows
        ]
        parts: list[str] = []
        if self.title:
            parts.append(f"**{_escape_markdown(self.title)}**\n")
        parts.extend([head, align, *body])
        if self.caption:
            parts.append(f"\n_{_escape_markdown(self.caption)}_")
        return "\n".join(parts)

    def to_html(self) -> str:
        """Экранированный HTML для интерактивного отображения."""
        if not self.rows:
            return "<i>(пусто)</i>"
        css_th = (
            "padding:4px 10px;border-bottom:2px solid #bbb;"
            "font-weight:600"
        )
        css_td = "padding:3px 10px;border-bottom:1px solid #eee"
        output = ["<div style='font-family:system-ui,sans-serif'>"]
        if self.title:
            output.append(
                "<div style='font-weight:600;margin-bottom:4px'>"
                f"{html.escape(self.title)}</div>"
            )
        output.append("<table style='border-collapse:collapse;font-size:13px'>")
        output.append(
            "<tr>" + "".join(
                f"<th style='{css_th};text-align:"
                f"{'right' if self._numeric[column] else 'left'}'>"
                f"{html.escape(column)}</th>"
                for column in self.columns
            ) + "</tr>"
        )
        for row in self.rows:
            failed = str(row.get("успех", "")).upper() == "НЕТ"
            row_style = " style='color:#b42318'" if failed else ""
            output.append(
                f"<tr{row_style}>" + "".join(
                    f"<td style='{css_td};text-align:"
                    f"{'right' if self._numeric[column] else 'left'}'>"
                    f"{html.escape(self._cell(row, column))}</td>"
                    for column in self.columns
                ) + "</tr>"
            )
        output.append("</table>")
        if self.caption:
            output.append(
                "<div style='font-size:12px;color:#666;margin-top:4px'>"
                f"{html.escape(self.caption)}</div>"
            )
        output.append("</div>")
        return "".join(output)

    def _repr_html_(self) -> str:
        return self.to_html()

    def __repr__(self) -> str:
        if not self.rows:
            return "(пусто)"
        widths = {
            column: max(
                len(column), *(len(self._cell(row, column)) for row in self.rows)
            )
            for column in self.columns
        }
        header = "  ".join(
            column.rjust(widths[column]) if self._numeric[column]
            else column.ljust(widths[column])
            for column in self.columns
        )
        output = ([self.title] if self.title else []) + [header, "-" * len(header)]
        for row in self.rows:
            output.append("  ".join(
                self._cell(row, column).rjust(widths[column])
                if self._numeric[column]
                else self._cell(row, column).ljust(widths[column])
                for column in self.columns
            ))
        if self.caption:
            output.append(self.caption)
        return "\n".join(output)


def compare_methods(results: dict, title: str = "Сравнение методов") -> Table:
    """Сводная таблица, включая причину отказа неуспешного метода."""
    return Table(
        [result.as_row(name) for name, result in results.items()],
        title=title,
        caption="время — чистый счёт, без отрисовки",
    )


def convergence_table(ct, title: str = "") -> Table:
    return Table(
        ct.to_rows(),
        title=title or (f"Сходимость: {ct.label}" if ct.label else "Сходимость"),
        caption=ct.verdict(),
    )
