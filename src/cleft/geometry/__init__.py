"""Geometry: staging, the two crop arms, patch generators.

Importing this package registers all three generators, so ``patches.get(name)``
works without the caller having to know which module each one lives in.

Contents: pad-square-resize staging (white pad, 224), the trapezium mask and the
G2 unwarp with its asymmetry gate, three patch generators behind one interface,
and the per-image mapping that places patches in each image's own pixels.
"""

from . import generators, mapping, patches, staging, trapezium  # noqa: F401

__all__ = ["generators", "mapping", "patches", "staging", "trapezium"]
