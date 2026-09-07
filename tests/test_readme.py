"""Keep every public Python example executable, without an installed notebook."""
from pathlib import Path
import os
import re
import subprocess
import sys

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
def examples(text):
    return re.findall(r"^```python\r?\n(.*?)^```", text, re.M | re.S)


EXAMPLES = examples(README.read_text("utf-8"))


def test_readme_has_runnable_examples():
    assert len(EXAMPLES) >= 2


def test_example_extraction_accepts_crlf():
    assert examples('```python\r\nprint("ok")\r\n```') == ['print("ok")\r\n']


@pytest.mark.parametrize("source", EXAMPLES, ids=[f"example-{i + 1}" for i in range(len(EXAMPLES))])
def test_readme_example(source, tmp_path):
    # Fresh interpreter: examples cannot inherit Matplotlib state from other tests.
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUTF8="1")
    completed = subprocess.run(
        [sys.executable, "-c", source], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    if 'save="convergence.png"' in source:
        assert (tmp_path / "convergence.png").stat().st_size > 1000
