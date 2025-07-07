from __future__ import annotations

"""Common package init: provides compatibility helpers available to all sub-modules."""

import sys
import typing

# ---------------------------------------------------------------------------
# Python <3.11 compatibility – add typing.Self so that annotations like
#   def clone(self) -> Self
# work under older interpreters (3.8 / 3.9 / 3.10).
# ---------------------------------------------------------------------------
if not hasattr(typing, "Self"):
    from typing import TypeVar  # pylint: disable=import-error

    typing.Self = TypeVar("Self")  # type: ignore[attr-defined]
