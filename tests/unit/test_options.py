# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Bad parameter values stop at the door with one line naming them: the override checks (btb/options.py), the
device name against this machine, the CLI's parse-time checks, and `btb.load` refusing before it loads."""

import sys
from collections.abc import Callable

import pytest
from pytest import CaptureFixture

from btb.options import (
    BadDevice,
    BadValue,
    Device,
    OptionError,
    TooManyLayers,
    UnknownOption,
    check,
    check_device,
    check_layers,
    check_sampling,
)
from tests.helpers import checkout, fixture


def _bad(fn: Callable[..., object], *args: object) -> OptionError:
    with pytest.raises(OptionError) as e:
        fn(*args)
    return e.value


def test_every_kind_of_value_is_checked() -> None:
    """flags 0/1, counts whole and floored, rates in 0..1, positives, non-negatives, choices; strings and bools
    converted, NaN and text refused, an unknown option named with the list"""
    assert check({"fp32": True, "context": "4096", "top_p": 0.5, "seed": 7, "temperature": 0}) == {
        "fp32": 1,
        "context": 4096,
        "top_p": 0.5,
        "seed": 7,
        "temperature": 0.0,
    }
    assert check({"tree_budget": None}) == {}
    for kw in (
        {"fp32": 2},
        {"context": -1},
        {"context": 1.5},
        {"cold_slots": 0},
        {"tree_min_prob": 7},
        {"top_p": -0.1},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": "hot"},
        {"draft_temp_ratio": 0},
        {"expert_cache_gb": 0},
        {"ram_reserve_gb": -3},
        {"ram_reserve_gb": "abc"},
        {"ram_reserve_gb": "150%"},
        {"ram_reserve_gb": "-5%"},
        {"draft_bits": 5},
        {"kv_bits": 4},
        {"top_k": -3},
        {"seed": 1.5},
        {"seed": True},
        {"temperature": True},
    ):
        ((name, v),) = kw.items()
        e = _bad(check, kw)
        assert isinstance(e, BadValue) and e.name == name and (e.value is v or e.value == v), (kw, e)
    e = _bad(check, {"tree_bugdet": 3})
    assert isinstance(e, UnknownOption) and e.name == "tree_bugdet" and "tree_budget" in e.known
    assert check({"top_p": 0}) == {"top_p": 0.0} and check({"top_p": 1}) == {"top_p": 1.0}


def test_the_layer_counts_are_held_to_the_model() -> None:
    check_layers({"cpu_layers": 3, "resident_last": 5}, 8)
    for name in ("cpu_layers", "resident_last"):
        e = _bad(check_layers, {name: 9}, 8)
        assert isinstance(e, BadValue) and (e.name, e.value) == (name, 9)
    e = _bad(check_layers, {"cpu_layers": 4, "resident_last": 5}, 8)
    assert isinstance(e, TooManyLayers) and (e.cpu, e.last, e.layers) == (4, 5, 8)


def test_the_sampling_fields_of_a_request_are_checked() -> None:
    assert check_sampling({"temperature": "0.7", "top_p": 0.9, "top_k": 40, "seed": 3, "other": 1}) == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 3,
    }
    assert check_sampling({"temperature": None}) == {}
    for f in ({"temperature": "hot"}, {"top_p": 5}, {"top_k": -1}, {"seed": "x"}, {"temperature": float("inf")}):
        _bad(check_sampling, f)
    from btb.sampling import Sampling

    with pytest.raises(OptionError):
        Sampling(temperature=-1)
    with pytest.raises(OptionError):
        Sampling(temperature=0.8, top_p=2)


def test_the_device_name_is_checked_before_any_import() -> None:
    assert check_device(None) is None and check_device("auto") is None and check_device("") is None
    from btb.options import Device, DeviceName

    assert check_device(" CUDA:1 ") == DeviceName(Device.CUDA, 1) and str(check_device("cpu")) == "cpu"
    assert check_device(Device.MLX if sys.platform == "darwin" else Device.CPU) is not None
    for bad in ("foo", "mlx:0", "cuda:x", "gpu", "cuda:"):
        e = _bad(check_device, bad)
        assert isinstance(e, BadDevice) and e.device == bad, bad
    if sys.platform != "darwin":
        e = _bad(check_device, "mlx")
        assert isinstance(e, BadDevice) and e.device == "mlx"
    else:
        assert check_device("mlx") == DeviceName(Device.MLX)


def test_resolve_device_refuses_a_device_that_is_not_here() -> None:
    """a named device must exist: 'cuda' without a card, 'mlx' off Apple silicon, an index past the cards"""
    import torch

    from btb.engine.device import mlx_available, resolve_device

    if not torch.cuda.is_available():
        e = _bad(resolve_device, "cuda")
        assert isinstance(e, BadDevice) and str(e.device) == "cuda"
    else:
        n = torch.cuda.device_count()
        e = _bad(resolve_device, f"cuda:{n}")
        assert isinstance(e, BadDevice) and str(e.device) == f"cuda:{n}"
    if not mlx_available():
        e = _bad(resolve_device, "mlx")
        assert isinstance(e, BadDevice) and str(e.device) == "mlx"
    else:
        assert resolve_device("mlx").kind is Device.MLX
    assert resolve_device("cpu").kind is Device.CPU and str(resolve_device("cpu")) == "cpu"
    assert resolve_device(None).kind in Device


def test_load_refuses_a_bad_override_before_it_loads() -> None:
    """the check runs before the pool, torch and the weights: a bad value on the fixture fails at once"""
    import btb

    fx = fixture("tiny_q35")
    for kw in ({"context": -1}, {"nonsense": 1}, {"cpu_layers": 999}, {"temperature": float("nan")}):
        with pytest.raises(OptionError):
            btb.load(fx, device="cpu", **kw)
    with pytest.raises(OptionError):
        btb.load(fx, device="mlx:0")


def test_the_cli_stops_on_a_bad_value_with_one_line(capsys: CaptureFixture[str]) -> None:
    """argparse's own line for a flag it checks at parse time, btb's line (by the flag's name) for a value the
    load checks; exit 2 both ways, never a traceback"""
    from btb.cli import main

    fx = fixture("tiny_q35")
    with pytest.raises(SystemExit) as e:
        main(["run", fx, "-d", "mlx:0", "-p", "hi"])
    assert e.value.code == 2
    assert "cpu, mlx, cuda or cuda:N" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["run", fx, "--new", "0", "-p", "hi"])
    assert e.value.code == 2 and "a whole number above 0" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["serve", "--port", "70000"])
    assert e.value.code == 2 and "a port" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["serve", "--models-filter", "("])
    assert e.value.code == 2 and "not a regular expression" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["bench", fx, "--prompts", checkout("bench", "questions.jsonl"), "--rows", "a"])
    assert e.value.code == 2
    capsys.readouterr()
    rc = main(["run", fx, "-d", "cpu", "-q", "-p", "hi", "--new", "4", "--ram-reserve", "-3"])
    out = capsys.readouterr()
    assert rc == 2 and "[btb] --ram-reserve -3.0: 0 or above" in out.err and "Traceback" not in out.err


def test_a_reserve_is_gb_or_a_percent() -> None:
    from btb.options import Percent, check_value, resolve_reserve

    # GB: a plain number, any of the input forms
    assert check_value("ram_reserve_gb", "12") == 12.0
    assert check_value("ram_reserve_gb", 2) == 2.0 and check_value("vram_reserve_gb", 0.5) == 0.5
    # a percent, from the CLI string or the Python API
    assert check_value("ram_reserve_gb", "10%") == Percent(0.1)
    assert check_value("vram_reserve_gb", "8%") == Percent(0.08)
    assert check_value("ram_reserve_gb", "10 %") == Percent(0.1)  # a space before the % is fine
    assert check_value("ram_reserve_gb", Percent(0.25)) == Percent(0.25)
    # resolved to GB against the total, a plain number passed through
    assert resolve_reserve(Percent(0.1), 64 * 2**30) == 6.4
    assert resolve_reserve(Percent(0.08), 24 * 2**30) == 24 * 0.08
    assert resolve_reserve(3.0, 64 * 2**30) == 3.0
    for bad in ("10%%", "%", "%50", "nan%", "50 percent", "1.2.3", "", "  "):
        assert isinstance(_bad(check_value, "ram_reserve_gb", bad), BadValue), bad
