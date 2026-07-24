"""Backward-compatible import path for I-ROCKET model selection.

The canonical public module is :mod:`irocket_model_selection`. This compatibility
module preserves analyses written against the earlier 0.7 development path.
"""

from irocket_model_selection import *  # noqa: F401,F403
