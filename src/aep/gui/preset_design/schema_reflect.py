"""Introspection helpers for mapping Pydantic field types to GUI editors."""

from __future__ import annotations

import types
import typing
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for Union[..., None]."""
    if annotation is Any:
        return annotation, False

    if sys_union_optional(annotation):
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return non_none[0], True
        return annotation, False

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return non_none[0], True

    return annotation, False


def sys_union_optional(annotation: Any) -> bool:
    """True if annotation is PEP604 X | None style."""
    u_type = getattr(types, "UnionType", None)
    if u_type is None:
        return False
    return isinstance(annotation, u_type)


def literal_choices(annotation: Any) -> tuple[Any, ...] | None:
    o = get_origin(annotation)
    if o is typing.Literal:
        return get_args(annotation)
    return None


def is_basemodel_subtype(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def is_list_of_str(annotation: Any) -> bool:
    if get_origin(annotation) not in (list, list):
        return False
    args = get_args(annotation)
    return bool(args) and args[0] is str
