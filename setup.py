"""Build the optional C kernel alongside the pure-Python package.

Everything else about the build is declared in `pyproject.toml`; this file
exists only because the kernel is a C extension and an *optional* one.  If
there is no compiler, or no NumPy headers, the extension is skipped and the
package installs and works exactly as it did before -- the filters fall back
to their NumPy path, which the kernel is checked against anyway.

`-ffp-contract=off` is a correctness flag, not an optimisation one: with
contraction on, the compiler may fuse `a * b + c` into a single rounded
operation, which gives a different -- and better -- number than the NumPy
expression the kernel is required to reproduce exactly.
"""
from setuptools import setup, Extension

try:
    import numpy
    include_dirs = [numpy.get_include(), "lucid/lucid_kernel"]
except ImportError:                                 # pragma: no cover
    include_dirs = ["lucid/lucid_kernel"]

setup(
    ext_modules=[
        Extension(
            "lucid.lucid_kernel._kernel",
            sources=["lucid/lucid_kernel/_kernel.c"],
            include_dirs=include_dirs,
            extra_compile_args=["-O3", "-std=c11", "-ffp-contract=off",
                                "-fno-fast-math"],
            libraries=["m"],
            optional=True,              # no compiler is not an error
        ),
    ],
)
