# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
# The package metadata is in pyproject.toml; this file exists only to tag the wheel.
#
# btb ships a compiled native library (btb/native/<platform>/libbtb_native.{dylib,so,dll}) that the datapath
# loads with ctypes. So the wheel is platform-specific, not the pure `py3-none-any` setuptools would build
# for a package with no C extension - that tag claims the wheel runs anywhere, and pip would install a macOS
# arm64 build on a Linux box. The library is loaded by ctypes, not linked to the Python ABI, so the tag is
# `py3-none-<platform>`: any Python 3, that platform. `build.py` sets BTB_WHEEL_PLAT to the target it built
# the library for (accurate across a cross-build); a bare `pip install .` falls back to the host platform.
import datetime
import glob
import os

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools that has not vendored wheel yet
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# btb ships prebuilt wheels for these platforms; other platforms have no wheel and no in-tree compile step, so an
# install falls back to the sdist and would produce a library-less, unimportable package. The wheel build below
# refuses that case with a clear message instead; the guard passes whenever build.py has placed
# the library.
SUPPORTED_WHEELS = "linux x86_64 and aarch64 (manylinux_2_28), windows x86_64 and arm64, macos arm64 (Apple Silicon)"
_NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btb", "native")
_LIB_NAMES = ("btb_native.dll", "libbtb_native.so", "libbtb_native.dylib")


class bdist_wheel(_bdist_wheel):  # type: ignore[misc]  # wheel ships no stubs
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False  # not py3-none-any: the wheel carries a compiled library
        if not any(glob.glob(os.path.join(_NATIVE, "*", n)) for n in _LIB_NAMES):
            raise SystemExit(
                "btb has no native library to package: it ships wheels for "
                f"{SUPPORTED_WHEELS}. On those, pip installs the wheel. On other platforms build the library "
                "first with `python build.py`; btb cannot be built from the sdist alone."
            )

    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, plat = super().get_tag()
        return "py3", "none", os.environ.get("BTB_WHEEL_PLAT", plat)


class build_py(_build_py):  # type: ignore[misc]  # setuptools ships no stubs for its commands
    def run(self) -> None:
        super().run()
        # bake the build date into the package so the FSL notice's two-year window keys off the actual build,
        # not a constant someone must remember to bump. Written into build_lib (never the source tree), and
        # SOURCE_DATE_EPOCH honored so a reproducible build stays reproducible.
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        day = datetime.datetime.fromtimestamp(int(epoch), datetime.UTC).date() if epoch else datetime.date.today()
        target = os.path.join(self.build_lib, "btb", "_build.py")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(
                f'# generated at build time by setup.py; the FSL notice keys its window off this\nBUILD_DATE = "{day.isoformat()}"\n'
            )


setup(cmdclass={"bdist_wheel": bdist_wheel, "build_py": build_py})
