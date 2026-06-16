"""Public package exports for pKasso."""

from __future__ import annotations

from typing import Any

__all__ = ["batch_protonate", "protonate", "scan_pH"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from pkasso import py_interface

        return getattr(py_interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
