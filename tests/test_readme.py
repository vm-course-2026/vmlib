"""Keep every public Python example executable, without an installed notebook."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
EXAMPLES = re.findall(r"^```python\n(.*?)^```", README.read_text("utf-8"), re.M | re.S)


def test_readme_has_runnable_examples():
    assert len(EXAMPLES) >= 2


@pytest.mark.parametrize("source", EXAMPLES, ids=[f"example-{i + 1}" for i in range(len(EXAMPLES))])
def test_readme_example(source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        exec(compile(source, "README.md", "exec"), {"__name__": "__main__"})
        if 'save="convergence.png"' in source:
            assert (tmp_path / "convergence.png").stat().st_size > 1000
    finally:
        plt.close("all")
