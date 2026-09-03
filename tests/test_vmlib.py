from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vmlib as vm


def test_norms_reject_shape_broadcast_and_empty():
    with pytest.raises(ValueError, match="формы не совпадают"):
        vm.max_error(np.ones(3), np.ones((3, 1)))
    with pytest.raises(ValueError, match="пустой"):
        vm.l2_error([], [])


def test_norms_propagate_nonfinite_and_reject_complex():
    assert np.isnan(vm.max_error([np.nan], [0.0]))
    with pytest.raises(TypeError, match="комплексные"):
        vm.rel_error([1j], [1j])


def test_l2_scaling_in_two_dimensions_and_anisotropic_grid():
    error = np.ones((2, 2))
    assert vm.l2_error(error, np.zeros_like(error), h=0.5) == pytest.approx(1.0)
    assert vm.l2_error(error.ravel(), np.zeros(4), h=0.5, ndim=2) == pytest.approx(1.0)
    assert vm.l2_error(error, np.zeros_like(error), h=(0.5, 0.25)) == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("h", [0, -1, np.nan, []])
def test_l2_rejects_invalid_steps(h):
    with pytest.raises(ValueError):
        vm.l2_error([1], [0], h=h)


def test_empirical_orders_validate_input():
    with pytest.raises(ValueError, match="длины"):
        vm.empirical_orders([1, 0.5], [1])
    with pytest.raises(ValueError, match="различаться"):
        vm.empirical_orders([1, 1], [1, 0.5])


def test_convergence_needs_four_reliable_points():
    short = vm.convergence([0.2, 0.1, 0.05], [0.04, 0.01, 0.0025], expected=2)
    assert np.isnan(short.asymptotic_order)
    assert "не менее" in short.reason()


def test_convergence_detects_order_and_nan_failure():
    good = vm.convergence(
        [0.2, 0.1, 0.05, 0.025], [0.04, 0.01, 0.0025, 0.000625], expected=2
    )
    assert good.asymptotic_order == pytest.approx(2)
    assert vm.order_matches(good.asymptotic_order, 2)
    bad = vm.convergence(
        [0.2, 0.1, 0.05, 0.025], [0.04, 0.01, np.nan, 0.000625], expected=2
    )
    assert np.isnan(bad.asymptotic_order)
    assert "NaN/Inf" in bad.reason()


def test_convergence_detects_tail_plateau():
    table = vm.convergence(
        [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        [1e-8, 2.5e-9, 6.25e-10, 2e-10, 2e-10, 2e-10],
        expected=2,
    )
    assert table.stagnating
    assert table.discarded.size >= 2
    assert "предел точности" in table.reason()


def test_comparison_is_boolean_and_checks_each_coordinate():
    passed = vm.compare([1.0, 0.0], [1.0, 0.0], verbose=False)
    failed = vm.compare([1.0, 1e-3], [1.0, 0.0], tol=1e-6, verbose=False)
    assert bool(passed)
    assert not bool(failed)


def test_comparison_rejects_invalid_inputs():
    assert not vm.compare([], [], verbose=False)
    assert not vm.compare([1, 2], [[1, 2]], verbose=False)
    assert not vm.compare([np.nan], [np.nan], verbose=False)
    assert not vm.compare(vm.Result(x=[1], success=False, message="нет сходимости"), [1], verbose=False)


def test_comparison_html_escapes_user_text():
    comparison = vm.compare([1], [2], name="<script>x</script>", verbose=False)
    rendered = comparison._repr_html_()
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_check_order_fails_on_short_and_nonfinite_series(capsys):
    assert not vm.check_order(lambda n: np.array([n ** -2]), np.array([0.0]), [2, 4, 8], 2)
    assert not vm.check_order(
        lambda n: np.array([np.nan if n == 16 else n ** -2]),
        np.array([0.0]), [2, 4, 8, 16], 2,
    )
    assert "NaN/Inf" in capsys.readouterr().out


def test_stub_check_and_report_are_strict():
    assert not vm.check_not_a_stub(lambda: np.ones(2), not_equal_to=np.ones(2))
    assert not vm.report({})
    with pytest.raises(TypeError):
        vm.report({"не bool": object()})
    assert not vm.report({"сверка": vm.compare([1], [2], verbose=False)})


def test_check_exact_on_is_scale_aware_and_nonempty():
    assert vm.check_exact_on(lambda: np.array([1e8 + 1e-5]), {"large": ((), np.array([1e8]))}, tol=1e-12)
    assert not vm.check_exact_on(lambda: 1, {})


def test_check_invariant_validates_time():
    assert not vm.check_invariant([0, 1], [1])
    assert not vm.check_invariant([0, 0], [1, 1])
    assert vm.check_invariant([0, 1, 2], [1, 1 + 1e-5, 1 - 1e-5])


def test_benchmark_counter_reset_and_single_spread():
    counter = vm.CallCounter(lambda x: x + 1)
    timing = vm.benchmark(counter, 1, repeats=3, warmup=2, reset=counter)
    assert timing.value == 2
    assert counter.count == 1
    assert timing.spread >= 0
    assert np.isnan(vm.benchmark(lambda: None, repeats=1).spread)


def test_call_counter_preserves_metadata_and_descriptor():
    @vm.CallCounter
    def named(value):
        "doc"
        return value

    assert named.__name__ == "named"
    assert named(2) == 2 and named.count == 1

    class Demo:
        @vm.CallCounter
        def method(self, value):
            return value + 1

    demo = Demo()
    assert demo.method(2) == 3
    assert Demo.method.count == 1


def test_timed_is_initialized_and_one_shot():
    timer = vm.timed()
    assert np.isnan(timer.elapsed)
    with timer:
        pass
    assert timer.elapsed >= 0
    with pytest.raises(RuntimeError):
        with timer:
            pass


def test_result_cost_is_positive_explicit_and_json_serializable(tmp_path):
    result = vm.Result(
        x=np.array([1, 2]), success=True, n_fevals=4,
        cost_field="n_fevals", extra={"nested": [np.array([3]), np.float64(2.0)]},
    )
    assert result.cost == 4
    assert result.resolved_cost_unit == "вычислений правой части"
    json.dumps(result.to_dict())
    assert vm.Result(x=1, success=True, n_fevals=0).cost is None
    with pytest.raises(ValueError):
        _ = vm.Result(cost_field="unknown").cost


def test_result_repr_survives_bad_elapsed_time():
    assert "н/д" in repr(vm.Result(elapsed_time="bad"))
    assert vm.Result(n_fevals="not a number").cost is None


def test_table_escapes_html_markdown_and_distinguishes_nonfinite():
    table = vm.Table([{"name": "<b>x|y</b>", "v": np.nan}, {"name": "z", "v": np.inf}])
    assert "<b>" not in table.to_html()
    assert "&lt;b&gt;" in table.to_html()
    assert "\\|" in table.to_markdown()
    assert "NaN" in table.to_markdown() and "+∞" in table.to_markdown()
    assert "value" in vm.Table([{1: "value"}]).to_markdown()


def test_compare_methods_keeps_failure_message():
    table = vm.compare_methods({"bad": vm.Result(success=False, message="breakdown")})
    assert "breakdown" in table.to_markdown()


def test_plot_helpers_handle_arrays_filter_log_data_and_close(tmp_path):
    before = set(plt.get_fignums())
    result = vm.Result(success=True, residual_norms=np.array([1.0, 0.1, 0.01]))
    ax = vm.residual_history({"method": result})
    assert len(ax.lines) == 1
    plt.close(ax.figure)
    with pytest.warns(RuntimeWarning):
        ax = vm.convergence_plot({"method": ([1, 0.5, 0.25], [1, 0, 0.1])})
    plt.close(ax.figure)
    path = tmp_path / "plot.png"
    ax = vm.convergence_plot({"method": ([1, 0.5], [1, 0.25])}, save=path, close=True)
    assert path.exists()
    assert set(plt.get_fignums()) == before


def test_cost_plot_rejects_mixed_units():
    first = vm.Result(success=True, max_error=1e-2, n_fevals=10)
    second = vm.Result(success=True, max_error=1e-3, matvec_count=20)
    with pytest.raises(ValueError, match="единицы"):
        vm.cost_vs_error({"a": [first], "b": [second]})


def test_import_does_not_change_matplotlib_backend():
    code = (
        "import matplotlib; matplotlib.use('svg'); before=matplotlib.get_backend(); "
        "import vmlib; after=matplotlib.get_backend(); print(before); print(after)"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    proc = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True,
        env=env, check=True,
    )
    lines = proc.stdout.strip().lower().splitlines()
    assert lines[-2:] == ["svg", "svg"]


class DotScene(vm.Scene):
    def setup(self, ax):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        self.dot, = ax.plot([], [], "o")

    def draw(self, ax, k, t):
        started = __import__("time").perf_counter()
        self.dot.set_data([min(t, 1)], [0.5])
        self.meter.push(err=1 / (k + 1), dt_compute=__import__("time").perf_counter() - started)
        return [self.dot]


def test_animation_validates_and_renders_exact_frames(tmp_path):
    with pytest.raises(ValueError):
        vm.render([], 2, out=tmp_path / "x.gif")
    scene = DotScene("dot")
    output = vm.render([scene], 3, dt=0.1, out=tmp_path / "nested" / "x.gif", fps=5)
    assert Path(output).exists()
    assert len(scene.meter.times) == 3
    assert len(scene.meter.render_times) == 3
    frames = vm.render_frames([lambda: DotScene("dot")], [0, 2], out=tmp_path / "frames.png")
    assert Path(frames).exists()


def test_version_and_fingerprint_are_public():
    assert vm.__version__ == "1.1.0"
    assert len(vm.source_fingerprint()) == 12
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
    )
    assert metadata["project"]["version"] == vm.__version__
    assert metadata["project"]["name"] == "vmlib-course"


def test_doctor_is_path_safe_and_ascii_compatible():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    # A redirected stream on an English Windows runner may use a legacy
    # codepage.  ASCII is the strictest useful common denominator and catches
    # accidental non-ASCII status messages on every development platform.
    env.update({"MPLBACKEND": "Agg", "PYTHONIOENCODING": "ascii:strict"})
    proc = subprocess.run(
        [sys.executable, "-m", "vmlib.doctor"],
        capture_output=True,
        encoding="ascii",
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"vmlib.doctor exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "Status:     ready" in proc.stdout
    assert proc.stdout.isascii()
    assert sys.executable not in proc.stdout
    assert str(Path.home()) not in proc.stdout
