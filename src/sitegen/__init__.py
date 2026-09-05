"""Synthetic construction-site MCAP generation with held-out ground truth."""

from typing import Any

__all__ = ["Scene", "generate"]


def __getattr__(name: str) -> Any:
    """Resolve the two public names on demand.

    Importing them eagerly would drag the MCAP writer -- and with it mcap,
    protobuf and Pillow -- into every consumer of a leaf module. The Blender
    subprocess that renders the camera imports `sitegen.meshes` and
    `sitegen.cycles` inside an interpreter that has bpy and nothing else, and
    it has no business needing a protobuf schema package to do it.
    """
    if name == "Scene":
        from .scene import Scene

        return Scene
    if name == "generate":
        from .writer import generate

        return generate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
