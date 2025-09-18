# qroute/__init__.py
"""
Top-level API re-exports.
"""

from .routing.allocator import QuantumRoutingAllocator as Allocator

__all__ = ["Allocator"]
