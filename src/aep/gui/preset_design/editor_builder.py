"""Build Qt editors from Preset Pydantic models with bindings to nested dicts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, get_args, get_origin

import annotated_types as at
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from aep.gui.preset_design.schema_reflect import (
    is_basemodel_subtype,
    is_list_of_str,
    literal_choices,
    strip_optional,
)
from aep.persist.presets import Preset, TargetResolution

_CATEGORY_ORDER = [
    "meta",
    "container",
    "resolution",
    "upscaler",
    "interpolation",
    "encoder",
    "decode",
    "streams",
    "postprocess",
    "batching",
]

_CATEGORY_TITLE = {
    "meta": "Preset info",
    "container": "Container",
    "resolution": "Resolution",
    "upscaler": "Upscaler",
    "interpolation": "Interpolation",
    "encoder": "Encoder",
    "decode": "Decode",
    "streams": "Streams",
    "postprocess": "Post-process",
    "batching": "Batching",
}

_FALLBACK_CATEGORY = {
    "PresetMeta": "meta",
    "UpscalerCfg": "upscaler",
    "TargetResolution": "resolution",
    "InterpolationCfg": "interpolation",
    "EncoderCfg": "encoder",
    "StreamMappingCfg": "streams",
    "PostprocessCfg": "postprocess",
    "DecodeCfg": "decode",
    "BatchingCfg": "batching",
}


class EditorBinding:
    __slots__ = ("apply", "commit")

    def __init__(
        self,
        apply: Callable[[dict[str, Any]], None],
        commit: Callable[[dict[str, Any]], None],
    ) -> None:
        self.apply = apply
        self.commit = commit


def deep_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for k in path:
        cur = cur[k]
    return cur


def deep_set(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = data
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def _gui_meta(field_info: FieldInfo, model_cls_name: str) -> tuple[str, str]:
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        gui = extra.get("gui") or {}
        cat = gui.get("category")
        tier = gui.get("tier")
        if isinstance(cat, str) and tier in ("simple", "advanced"):
            return cat, tier
    return _FALLBACK_CATEGORY.get(model_cls_name, model_cls_name.removesuffix("Cfg").lower()), "advanced"


def _float_spin_decimals(field_info: FieldInfo) -> int:
    """QDoubleSpinBox decimal places; field may override via json_schema_extra['gui']."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        gui = extra.get("gui")
        if isinstance(gui, dict):
            raw = gui.get("spin_decimals")
            if isinstance(raw, int) and 0 <= raw <= 14:
                return raw
    return 4


def _num_bounds(field_info: FieldInfo) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in field_info.metadata:
        if isinstance(m, at.Ge):
            lo = float(m.ge) if lo is None else max(lo, float(m.ge))
        elif isinstance(m, at.Gt):
            v = float(m.gt) + 1.0
            lo = v if lo is None else max(lo, v)
        elif isinstance(m, at.Le):
            hi = float(m.le) if hi is None else min(hi, float(m.le))
        elif isinstance(m, at.Lt):
            v = float(m.lt) - 1.0
            hi = v if hi is None else min(hi, v)
    return lo, hi


def _attach_tooltip(widget: QWidget, description: str) -> None:
    if description:
        widget.setToolTip(description)


def build_preset_editor(
    *,
    on_changed: Callable[[], None] | None = None,
) -> tuple[QWidget, list[EditorBinding]]:
    """Return a widget with category tabs and bindings for apply/commit to a preset dict."""
    bindings: list[EditorBinding] = []

    RowList = list[tuple[QLabel, QWidget]]
    section_rows: dict[str, dict[str, RowList]] = defaultdict(
        lambda: {"simple": [], "advanced": []},
    )

    def trigger() -> None:
        if on_changed:
            on_changed()

    _walk_model_fields(
        Preset,
        (),
        bindings,
        section_rows,
        Preset.__name__,
        trigger,
    )

    _append_target_resolution_rows(bindings, section_rows, trigger)

    tabs_parent = QWidget()
    outer = QVBoxLayout(tabs_parent)
    outer.setContentsMargins(0, 0, 0, 0)

    tab_widget = QTabWidget()

    for cat in _CATEGORY_ORDER:
        rows_bucket = section_rows.get(cat)
        if not rows_bucket:
            continue
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(8, 8, 8, 8)

        simple_box = QGroupBox("Simple")
        sf = QFormLayout(simple_box)
        for lbl, w in rows_bucket["simple"]:
            sf.addRow(lbl, w)

        adv_toggle = QToolButton()
        adv_toggle.setCheckable(True)
        adv_toggle.setText("Show advanced")
        adv_wrap = QWidget()
        adv_v = QVBoxLayout(adv_wrap)
        adv_v.setContentsMargins(0, 0, 0, 0)
        adv_v.addWidget(adv_toggle)
        adv_box = QGroupBox()
        af = QFormLayout(adv_box)
        for lbl, w in rows_bucket["advanced"]:
            af.addRow(lbl, w)
        adv_box.setVisible(False)
        adv_v.addWidget(adv_box)

        def _toggle_adv(on: bool, box=adv_box, btn=adv_toggle) -> None:
            box.setVisible(on)
            btn.setText("Hide advanced" if on else "Show advanced")

        adv_toggle.toggled.connect(_toggle_adv)
        vl.addWidget(simple_box)
        vl.addWidget(adv_wrap)
        vl.addStretch(1)
        scroll.setWidget(inner)
        tab_widget.addTab(scroll, _CATEGORY_TITLE.get(cat, cat.title()))

    outer.addWidget(tab_widget)
    return tabs_parent, bindings


def _walk_model_fields(
    model_cls: type[BaseModel],
    path: tuple[str, ...],
    bindings: list[EditorBinding],
    section_rows: dict[str, dict[str, list[tuple[QLabel, QWidget]]]],
    model_cls_name: str,
    trigger: Callable[[], None],
) -> None:
    for fname, finfo in model_cls.model_fields.items():
        if path == () and fname == "target_resolution":
            continue
        ann = finfo.annotation
        inner, optional = strip_optional(ann)

        if inner is TargetResolution:
            continue

        if is_basemodel_subtype(inner):
            sub = inner
            assert issubclass(sub, BaseModel)
            _walk_model_fields(sub, path + (fname,), bindings, section_rows, sub.__name__, trigger)
            continue

        cat, tier = _gui_meta(finfo, model_cls_name)
        title = fname.replace("_", " ").title()
        label = QLabel(title)
        desc = finfo.description or ""
        if desc:
            label.setToolTip(desc)

        leaf = _make_leaf_widget(
            fname,
            finfo,
            inner,
            optional,
            path + (fname,),
            bindings,
            trigger,
        )
        if leaf is None:
            continue
        row_widget = leaf
        _attach_tooltip(row_widget, desc)
        section_rows[cat][tier].append((label, row_widget))


def _make_leaf_widget(
    fname: str,
    finfo: FieldInfo,
    inner: Any,
    optional: bool,
    leaf_path: tuple[str, ...],
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget | None:
    if fname == "suitable_for" and get_origin(inner) is list:
        args = get_args(inner)
        if args:
            lit = literal_choices(args[0])
            if lit is not None:
                return _build_literal_list_checkbox(leaf_path, lit, bindings, trigger)

    if is_list_of_str(inner):
        return _build_string_list_editor(leaf_path, bindings, trigger)

    lit = literal_choices(inner)
    if lit is not None:
        return _build_literal_combo(leaf_path, lit, optional, bindings, trigger)

    if inner is bool:
        return _build_bool(leaf_path, bindings, trigger)

    if inner is int:
        lo, hi = _num_bounds(finfo)
        lo_i = int(lo) if lo is not None else -(10**9)
        hi_i = int(hi) if hi is not None else 10**9
        return _build_int_spin(leaf_path, lo_i, hi_i, optional, bindings, trigger)

    if inner is float:
        lo, hi = _num_bounds(finfo)
        return _build_float_spin(
            leaf_path,
            lo,
            hi,
            optional,
            bindings,
            trigger,
            decimals=_float_spin_decimals(finfo),
        )

    if inner is str:
        return _build_line_edit(leaf_path, optional, bindings, trigger)

    return _build_line_edit(leaf_path, optional, bindings, trigger)


def _build_bool(
    leaf_path: tuple[str, ...],
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    cb = QCheckBox()

    def apply(data: dict[str, Any]) -> None:
        v = bool(deep_get(data, leaf_path))
        with QSignalBlocker(cb):
            cb.setChecked(v)

    def commit(data: dict[str, Any]) -> None:
        deep_set(data, leaf_path, cb.isChecked())

    cb.toggled.connect(lambda _v=False: trigger())
    bindings.append(EditorBinding(apply, commit))
    return cb


def _build_literal_combo(
    leaf_path: tuple[str, ...],
    choices: tuple[Any, ...],
    optional: bool,
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    combo = QComboBox()
    if optional:
        combo.addItem("(none)", None)
    for c in choices:
        combo.addItem(str(c), c)

    def apply(data: dict[str, Any]) -> None:
        v = deep_get(data, leaf_path)
        with QSignalBlocker(combo):
            idx = combo.findData(v)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

    def commit(data: dict[str, Any]) -> None:
        deep_set(data, leaf_path, combo.currentData())

    combo.currentIndexChanged.connect(lambda _i=0: trigger())
    bindings.append(EditorBinding(apply, commit))
    return combo


def _build_int_spin(
    leaf_path: tuple[str, ...],
    lo: int,
    hi: int,
    optional: bool,
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    if optional:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        en = QCheckBox("Set")
        sp = QSpinBox()
        sp.setRange(lo, hi)

        def apply(data: dict[str, Any]) -> None:
            v = deep_get(data, leaf_path)
            with QSignalBlocker(en), QSignalBlocker(sp):
                if v is None:
                    en.setChecked(False)
                    sp.setEnabled(False)
                else:
                    en.setChecked(True)
                    sp.setEnabled(True)
                    sp.setValue(int(v))

        def commit(data: dict[str, Any]) -> None:
            if en.isChecked():
                deep_set(data, leaf_path, int(sp.value()))
            else:
                deep_set(data, leaf_path, None)

        def on_toggle(c: bool) -> None:
            sp.setEnabled(c)
            trigger()

        en.toggled.connect(on_toggle)
        sp.valueChanged.connect(lambda _v=0: trigger())
        bindings.append(EditorBinding(apply, commit))
        return row

    sp = QSpinBox()
    sp.setRange(lo, hi)

    def apply(data: dict[str, Any]) -> None:
        v = deep_get(data, leaf_path)
        with QSignalBlocker(sp):
            sp.setValue(int(v) if v is not None else 0)

    def commit(data: dict[str, Any]) -> None:
        deep_set(data, leaf_path, int(sp.value()))

    sp.valueChanged.connect(lambda _v=0: trigger())
    bindings.append(EditorBinding(apply, commit))
    return sp


def _build_float_spin(
    leaf_path: tuple[str, ...],
    lo: float | None,
    hi: float | None,
    optional: bool,
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
    *,
    decimals: int = 4,
) -> QWidget:
    lo_v = -1e9 if lo is None else float(lo)
    hi_v = 1e9 if hi is None else float(hi)

    if optional:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        en = QCheckBox("Set")
        sp = QDoubleSpinBox()
        sp.setRange(lo_v, hi_v)
        sp.setDecimals(decimals)

        def apply(data: dict[str, Any]) -> None:
            v = deep_get(data, leaf_path)
            with QSignalBlocker(en), QSignalBlocker(sp):
                if v is None:
                    en.setChecked(False)
                    sp.setEnabled(False)
                else:
                    en.setChecked(True)
                    sp.setEnabled(True)
                    sp.setValue(float(v))

        def commit(data: dict[str, Any]) -> None:
            if en.isChecked():
                deep_set(data, leaf_path, float(sp.value()))
            else:
                deep_set(data, leaf_path, None)

        en.toggled.connect(lambda c: (sp.setEnabled(c), trigger()))
        sp.valueChanged.connect(lambda _v=0.0: trigger())
        bindings.append(EditorBinding(apply, commit))
        return row

    sp = QDoubleSpinBox()
    sp.setRange(lo_v, hi_v)
    sp.setDecimals(decimals)

    def apply(data: dict[str, Any]) -> None:
        v = deep_get(data, leaf_path)
        with QSignalBlocker(sp):
            sp.setValue(float(v) if v is not None else 0.0)

    def commit(data: dict[str, Any]) -> None:
        deep_set(data, leaf_path, float(sp.value()))

    sp.valueChanged.connect(lambda _v=0.0: trigger())
    bindings.append(EditorBinding(apply, commit))
    return sp


def _build_line_edit(
    leaf_path: tuple[str, ...],
    optional: bool,
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    if optional:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        en = QCheckBox("Set")
        ed = QLineEdit()

        def apply(data: dict[str, Any]) -> None:
            v = deep_get(data, leaf_path)
            with QSignalBlocker(en), QSignalBlocker(ed):
                if v is None:
                    en.setChecked(False)
                    ed.clear()
                    ed.setEnabled(False)
                else:
                    en.setChecked(True)
                    ed.setEnabled(True)
                    ed.setText(str(v))

        def commit(data: dict[str, Any]) -> None:
            if en.isChecked():
                txt = ed.text().strip()
                deep_set(data, leaf_path, txt if txt else None)
            else:
                deep_set(data, leaf_path, None)

        en.toggled.connect(lambda c: (ed.setEnabled(c), trigger()))
        ed.textChanged.connect(lambda _t="": trigger())
        bindings.append(EditorBinding(apply, commit))
        return row

    ed = QLineEdit()

    def apply(data: dict[str, Any]) -> None:
        v = deep_get(data, leaf_path)
        with QSignalBlocker(ed):
            ed.setText("" if v is None else str(v))

    def commit(data: dict[str, Any]) -> None:
        deep_set(data, leaf_path, ed.text())

    ed.textChanged.connect(lambda _t="": trigger())
    bindings.append(EditorBinding(apply, commit))
    return ed


def _build_string_list_editor(
    leaf_path: tuple[str, ...],
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    te = QPlainTextEdit()
    te.setPlaceholderText("One ffmpeg argument per line.")
    te.setFixedHeight(96)

    def apply(data: dict[str, Any]) -> None:
        rows = deep_get(data, leaf_path) or []
        with QSignalBlocker(te):
            te.setPlainText("\n".join(str(x) for x in rows))

    def commit(data: dict[str, Any]) -> None:
        lines = [ln.strip() for ln in te.toPlainText().splitlines()]
        deep_set(data, leaf_path, [ln for ln in lines if ln])

    te.textChanged.connect(lambda: trigger())
    bindings.append(EditorBinding(apply, commit))
    return te


def _build_literal_list_checkbox(
    leaf_path: tuple[str, ...],
    choices: tuple[Any, ...],
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
) -> QWidget:
    box = QWidget()
    v = QVBoxLayout(box)
    v.setContentsMargins(0, 0, 0, 0)
    cbs: dict[Any, QCheckBox] = {}
    for c in choices:
        cb = QCheckBox(str(c))
        cbs[c] = cb
        v.addWidget(cb)
        cb.toggled.connect(lambda _checked=False: trigger())

    def apply(data: dict[str, Any]) -> None:
        cur = set(deep_get(data, leaf_path) or [])
        for val, cb in cbs.items():
            with QSignalBlocker(cb):
                cb.setChecked(val in cur)

    def commit(data: dict[str, Any]) -> None:
        out = [k for k, cb in cbs.items() if cb.isChecked()]
        deep_set(data, leaf_path, out)

    bindings.append(EditorBinding(apply, commit))
    return box


def _append_target_resolution_rows(
    bindings: list[EditorBinding],
    section_rows: dict[str, dict[str, list[tuple[QLabel, QWidget]]]],
    trigger: Callable[[], None],
) -> None:
    fi_mode = TargetResolution.model_fields["mode"]
    fi_named = TargetResolution.model_fields["named"]

    mode_combo = QComboBox()
    for v in ("named", "explicit", "scale_only"):
        mode_combo.addItem(v, v)

    named_combo = QComboBox()
    named_combo.addItem("(none)", None)
    for v in ("720p", "1080p", "1440p", "2160p"):
        named_combo.addItem(v, v)

    row_named = QWidget()
    hn = QHBoxLayout(row_named)
    hn.setContentsMargins(0, 0, 0, 0)
    hn.addWidget(named_combo)

    wh_row = QWidget()
    hw_l = QHBoxLayout(wh_row)
    hw_l.setContentsMargins(0, 0, 0, 0)
    w_cb = QCheckBox("Width")
    w_sp = QSpinBox()
    w_sp.setRange(1, 16384)
    h_cb = QCheckBox("Height")
    h_sp = QSpinBox()
    h_sp.setRange(1, 16384)
    hw_l.addWidget(w_cb)
    hw_l.addWidget(w_sp)
    hw_l.addWidget(h_cb)
    hw_l.addWidget(h_sp)

    def sync_visibility() -> None:
        mode = mode_combo.currentData()
        row_named.setVisible(mode == "named")
        wh_row.setVisible(mode == "explicit")

    def apply_all(data: dict[str, Any]) -> None:
        sub = data.get("target_resolution") or {}
        with QSignalBlocker(mode_combo):
            mode = sub.get("mode", "named")
            idx = mode_combo.findData(mode)
            mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        named_v = sub.get("named")
        with QSignalBlocker(named_combo):
            ix = named_combo.findData(named_v)
            named_combo.setCurrentIndex(ix if ix >= 0 else 0)
        wv, hv = sub.get("width"), sub.get("height")
        with QSignalBlocker(w_cb), QSignalBlocker(w_sp), QSignalBlocker(h_cb), QSignalBlocker(h_sp):
            w_cb.setChecked(wv is not None)
            w_sp.setEnabled(wv is not None)
            if wv is not None:
                w_sp.setValue(int(wv))
            h_cb.setChecked(hv is not None)
            h_sp.setEnabled(hv is not None)
            if hv is not None:
                h_sp.setValue(int(hv))
        sync_visibility()

    def commit_all(data: dict[str, Any]) -> None:
        tr = data.setdefault("target_resolution", {})
        tr["mode"] = mode_combo.currentData()
        tr["named"] = named_combo.currentData()
        if w_cb.isChecked():
            tr["width"] = int(w_sp.value())
        else:
            tr["width"] = None
        if h_cb.isChecked():
            tr["height"] = int(h_sp.value())
        else:
            tr["height"] = None

    def _on_mode_changed(_i: int = 0) -> None:
        sync_visibility()
        trigger()

    mode_combo.currentIndexChanged.connect(_on_mode_changed)
    named_combo.currentIndexChanged.connect(lambda _i=0: trigger())

    def _wh_toggle(_: bool = False) -> None:
        w_sp.setEnabled(w_cb.isChecked())
        h_sp.setEnabled(h_cb.isChecked())
        trigger()

    w_cb.toggled.connect(_wh_toggle)
    h_cb.toggled.connect(_wh_toggle)
    w_sp.valueChanged.connect(lambda _v=0: trigger())
    h_sp.valueChanged.connect(lambda _v=0: trigger())

    bindings.append(EditorBinding(apply_all, commit_all))

    lm = QLabel("Mode")
    lm.setToolTip(fi_mode.description or "")
    mode_combo.setToolTip(fi_mode.description or "")
    section_rows["resolution"]["simple"].append((lm, mode_combo))

    ln = QLabel("Named target")
    ln.setToolTip(fi_named.description or "")
    named_combo.setToolTip(fi_named.description or "")
    section_rows["resolution"]["simple"].append((ln, row_named))

    fi_w = TargetResolution.model_fields["width"]
    fi_h = TargetResolution.model_fields["height"]
    lh = QLabel("Explicit size")
    lh.setToolTip(((fi_w.description or "") + " " + (fi_h.description or "")).strip())
    section_rows["resolution"]["advanced"].append((lh, wh_row))

    sync_visibility()
