import typing

if not hasattr(typing, "Self"):
    from typing import TypeVar

    typing.Self = TypeVar("Self") 