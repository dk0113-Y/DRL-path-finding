"""Training package marker.

Keep this file import-light so utility scripts can access training submodules
without importing torch-heavy components through package side effects.
"""

__all__: list[str] = []
