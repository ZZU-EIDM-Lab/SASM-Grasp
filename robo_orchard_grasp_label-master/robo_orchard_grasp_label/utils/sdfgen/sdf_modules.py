# Licensed under the MIT License. To view a copy of this license, visit:
#       https://opensource.org/licenses/MIT

import os
from pathlib import Path


def load_sdf():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    sources = [root / "sdfgen_interface.cpp", root / "makelevelset3.cpp"]

    func = load(
        name="sdfgen",
        sources=sources,
        extra_cflags=["-O3", "-msse4.2", "-mfpmath=sse", "-mtune=native"],
        extra_include_paths=["."],
    )
    return func


try:
    import sdfgen
except ImportError:
    sdfgen = load_sdf()


def sdf(path, dim=100, padding=5):
    sdfgen.run_sdf(path, dim, padding)
