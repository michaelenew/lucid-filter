"""Build the kernel in place: `python3 -m lucid_kernel.build`.

There is no separate build system.  The extension is a single translation
unit against NumPy's C API, so this drives the C compiler directly and puts
the shared object next to this file, which is where `__init__` looks for it.

`-ffp-contract=off` is not an optimisation setting, it is a correctness one:
with contraction on, the compiler may fuse `a * b + c` into a single rounded
operation, which gives a different -- and better -- number than the NumPy
expression this file is required to reproduce exactly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent

FLAGS = [
    "-O3",
    "-std=c11",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",        # see the docstring; not negotiable
    "-fno-fast-math",
    "-fvisibility=hidden",
    "-Wall",
]


def target() -> Path:
    return HERE / ("_kernel" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"))


def build(cc: str | None = None, extra: list[str] | None = None) -> Path:
    import numpy

    cc = cc or os.environ.get("CC") or "cc"
    out = target()
    cmd = [cc, *FLAGS, *(extra or []),
           "-I" + numpy.get_include(),
           "-I" + sysconfig.get_paths()["include"],
           "-I" + str(HERE),
           str(HERE / "_kernel.c"),
           "-o", str(out),
           "-lm"]
    subprocess.run(cmd, check=True)
    return out


def main(argv: list[str]) -> int:
    try:
        out = build()
    except FileNotFoundError as exc:
        print(f"no C compiler: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"compile failed ({exc.returncode})", file=sys.stderr)
        return exc.returncode
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
