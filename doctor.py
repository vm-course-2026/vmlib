"""Минимальная диагностика окружения vmlib."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib
import numpy
import PIL

import vmlib


def main() -> int:
    print(f"OS:         {platform.platform()}")
    # Полный sys.executable часто содержит имя пользователя и структуру
    # домашнего каталога. Для диагностики достаточно имени бинарного файла.
    print(f"Python:     {sys.version.split()[0]} ({Path(sys.executable).name})")
    print(f"NumPy:      {numpy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__} ({matplotlib.get_backend()})")
    print(f"Pillow:     {PIL.__version__}")
    print(f"vmlib:      {vmlib.__version__} ({vmlib.source_fingerprint()})")
    ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    # Keep the machine-readable diagnostic output ASCII-only.  In particular,
    # redirected stdout on an English Windows installation can use cp1252;
    # printing Cyrillic there raises UnicodeEncodeError before the exit status
    # can describe the actual environment health.
    print("Status:     " + ("ready" if ok else "unsupported Python"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
