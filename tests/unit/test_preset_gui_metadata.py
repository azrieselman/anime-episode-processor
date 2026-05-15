"""Ensure every Preset schema leaf used by the GUI carries a Field description."""

from __future__ import annotations

from pydantic import BaseModel

from aep.gui.preset_design.schema_reflect import is_basemodel_subtype, strip_optional
from aep.persist.presets import Preset

# Rare exceptions must be reviewed explicitly when adding entries.
KNOWN_UNDOCUMENTED_GUI_FIELDS: frozenset[str] = frozenset()


def _leaf_field_paths(model_cls: type[BaseModel], prefix: tuple[str, ...]) -> None:
    for fname, finfo in model_cls.model_fields.items():
        path = prefix + (fname,)
        inner, _ = strip_optional(finfo.annotation)

        if is_basemodel_subtype(inner):
            assert isinstance(inner, type) and issubclass(inner, BaseModel)
            _leaf_field_paths(inner, path)
            continue

        dotted = ".".join(path)
        if dotted in KNOWN_UNDOCUMENTED_GUI_FIELDS:
            continue

        desc = (finfo.description or "").strip()
        assert desc, f"missing Field(description=…) for preset leaf: {dotted}"


def test_preset_leaf_fields_are_documented_for_gui() -> None:
    _leaf_field_paths(Preset, ())
