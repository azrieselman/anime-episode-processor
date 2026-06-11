"""Build Qt editors from Preset Pydantic models with bindings to nested dicts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
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

from aep.bench.hardware import probe_hardware
from aep.encode.encoder_family import encode_name_for, encoder_family, software_name_for
from aep.gui.preset_design.schema_reflect import (
    is_basemodel_subtype,
    is_list_of_str,
    literal_choices,
    strip_optional,
)
from aep.persist.presets import Preset, TargetResolution, encoder_rc_matches_when

_CATEGORY_ORDER = [
    "meta",
    "container",
    "resolution",
    "upscaler",
    "interpolation",
    "encoding",
    "streams",
    "batching",
]

_CATEGORY_TITLE = {
    "meta": "Preset info",
    "container": "Container",
    "resolution": "Resolution",
    "upscaler": "Upscaler",
    "interpolation": "Interpolation",
    "encoding": "Encoding",
    "streams": "Streams",
    "batching": "Batching",
}

_FALLBACK_CATEGORY = {
    "PresetMeta": "meta",
    "UpscalerCfg": "upscaler",
    "TargetResolution": "resolution",
    "InterpolationCfg": "interpolation",
    "EncoderCfg": "encoding",
    "StreamMappingCfg": "streams",
    "PostprocessCfg": "encoding",
    "DecodeCfg": "encoding",
    "FrameDedupeCfg": "encoding",
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


@dataclass
class FieldRow:
    category: str
    tier: str
    group: str
    when_family: str
    when_rc: str
    path: tuple[str, ...]
    label: QLabel
    widget: QWidget


def _codec_caps_from_hardware() -> dict[str, set[str]]:
    caps: dict[str, set[str]] = {
        "nvenc": {"h264", "hevc", "av1"},
        "qsv": {"h264", "hevc", "av1"},
        "amf": {"h264", "hevc", "av1"},
        "d3d12": {"h264", "av1"},
        "vulkan": {"h264", "hevc", "av1"},
        "software": {"h264", "hevc"},
    }
    try:
        hw = probe_hardware()
    except Exception:
        return caps

    caps["nvenc"] = {
        c for c, ok in {
            "h264": hw.gpu.nvenc_h264 and hw.has_encoder("h264_nvenc"),
            "hevc": hw.gpu.nvenc_hevc and hw.has_encoder("hevc_nvenc"),
            "av1": hw.gpu.nvenc_av1 and hw.has_encoder("av1_nvenc"),
        }.items() if ok
    } or {"h264", "hevc", "av1"}
    caps["qsv"] = {
        c for c, ok in {
            "h264": hw.gpu.qsv_h264 and hw.has_encoder("h264_qsv"),
            "hevc": hw.gpu.qsv_hevc and hw.has_encoder("hevc_qsv"),
            "av1": hw.gpu.qsv_av1 and hw.has_encoder("av1_qsv"),
        }.items() if ok
    } or {"h264", "hevc", "av1"}
    caps["amf"] = {
        c for c, ok in {
            "h264": hw.gpu.amf_h264 and hw.has_encoder("h264_amf"),
            "hevc": hw.gpu.amf_hevc and hw.has_encoder("hevc_amf"),
            "av1": hw.gpu.amf_av1 and hw.has_encoder("av1_amf"),
        }.items() if ok
    } or {"h264", "hevc", "av1"}
    caps["d3d12"] = {
        c for c, ok in {
            "h264": hw.gpu.d3d12_h264 and hw.has_encoder("h264_d3d12"),
            "av1": hw.gpu.d3d12_av1 and hw.has_encoder("av1_d3d12"),
        }.items() if ok
    } or {"h264", "av1"}
    caps["vulkan"] = {
        c for c, ok in {
            "h264": hw.gpu.vulkan_h264 and hw.has_encoder("h264_vulkan"),
            "hevc": hw.gpu.vulkan_hevc and hw.has_encoder("hevc_vulkan"),
            "av1": hw.gpu.vulkan_av1 and hw.has_encoder("av1_vulkan"),
        }.items() if ok
    } or {"h264", "hevc", "av1"}
    return caps


def _hardware_hint_text() -> str:
    try:
        hw = probe_hardware()
    except Exception:
        return "Hardware hint unavailable until probe completes."

    labels: list[str] = []
    if hw.gpu.nvenc_h264 or hw.gpu.nvenc_hevc or hw.gpu.nvenc_av1:
        nv = [c.upper() for c, ok in {
            "h264": hw.gpu.nvenc_h264 and hw.has_encoder("h264_nvenc"),
            "hevc": hw.gpu.nvenc_hevc and hw.has_encoder("hevc_nvenc"),
            "av1": hw.gpu.nvenc_av1 and hw.has_encoder("av1_nvenc"),
        }.items() if ok]
        if nv:
            labels.append(f"NVIDIA NVENC ({', '.join(nv)})")
    if hw.gpu.qsv_h264 or hw.gpu.qsv_hevc or hw.gpu.qsv_av1:
        qsv = [c.upper() for c, ok in {
            "h264": hw.gpu.qsv_h264 and hw.has_encoder("h264_qsv"),
            "hevc": hw.gpu.qsv_hevc and hw.has_encoder("hevc_qsv"),
            "av1": hw.gpu.qsv_av1 and hw.has_encoder("av1_qsv"),
        }.items() if ok]
        if qsv:
            labels.append(f"Intel QSV ({', '.join(qsv)})")
    if hw.gpu.amf_h264 or hw.gpu.amf_hevc or hw.gpu.amf_av1:
        amf = [c.upper() for c, ok in {
            "h264": hw.gpu.amf_h264 and hw.has_encoder("h264_amf"),
            "hevc": hw.gpu.amf_hevc and hw.has_encoder("hevc_amf"),
            "av1": hw.gpu.amf_av1 and hw.has_encoder("av1_amf"),
        }.items() if ok]
        if amf:
            labels.append(f"AMD AMF ({', '.join(amf)})")
    if hw.gpu.d3d12_h264 or hw.gpu.d3d12_av1:
        d3d12 = [c.upper() for c, ok in {
            "h264": hw.gpu.d3d12_h264 and hw.has_encoder("h264_d3d12"),
            "av1": hw.gpu.d3d12_av1 and hw.has_encoder("av1_d3d12"),
        }.items() if ok]
        if d3d12:
            labels.append(f"D3D12 ({', '.join(d3d12)})")
    if hw.gpu.vulkan_h264 or hw.gpu.vulkan_hevc or hw.gpu.vulkan_av1:
        vk = [c.upper() for c, ok in {
            "h264": hw.gpu.vulkan_h264 and hw.has_encoder("h264_vulkan"),
            "hevc": hw.gpu.vulkan_hevc and hw.has_encoder("hevc_vulkan"),
            "av1": hw.gpu.vulkan_av1 and hw.has_encoder("av1_vulkan"),
        }.items() if ok]
        if vk:
            labels.append(f"Vulkan ({', '.join(vk)})")
    if not labels:
        return "No hardware encoder capabilities detected from the last probe."
    return "Detected: " + "; ".join(labels)


class EncoderNamePicker(QWidget):
    def __init__(self, *, trigger: Callable[[], None]) -> None:
        super().__init__()
        self._caps = _codec_caps_from_hardware()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)

        self._backend = QComboBox()
        self._backend.addItem("NVIDIA (NVENC)", "nvenc")
        self._backend.addItem("Intel (QSV)", "qsv")
        self._backend.addItem("AMD (AMF)", "amf")
        self._backend.addItem("DirectX D3D12", "d3d12")
        self._backend.addItem("Vulkan", "vulkan")
        self._backend.addItem("CPU (libx264/libx265)", "software")
        self._codec = QComboBox()
        h.addWidget(self._backend, 1)
        h.addWidget(self._codec, 1)

        self._backend.currentIndexChanged.connect(self._on_backend_changed)
        self._codec.currentIndexChanged.connect(lambda _i=0: trigger())
        self._backend.currentIndexChanged.connect(lambda _i=0: trigger())
        self._rebuild_codecs(preferred="hevc")

    def _on_backend_changed(self, _i: int = 0) -> None:
        self._rebuild_codecs(preferred=None)

    def _rebuild_codecs(self, *, preferred: str | None) -> None:
        backend = str(self._backend.currentData() or "software")
        available = sorted(self._caps.get(backend, {"h264", "hevc"}))
        if backend == "software":
            available = [c for c in ("h264", "hevc") if c in available]
            if not available:
                available = ["h264", "hevc"]
        elif backend == "d3d12":
            available = [c for c in ("h264", "av1") if c in available]
            if not available:
                available = ["h264", "av1"]
        elif not available:
            available = ["h264", "hevc", "av1"]

        with QSignalBlocker(self._codec):
            self._codec.clear()
            for codec in available:
                label = {"h264": "H.264", "hevc": "HEVC", "av1": "AV1"}[codec]
                self._codec.addItem(label, codec)
            if preferred is not None:
                idx = self._codec.findData(preferred)
                if idx >= 0:
                    self._codec.setCurrentIndex(idx)
            elif self._codec.count() > 0:
                self._codec.setCurrentIndex(0)

        if backend != "software" and "av1" not in available:
            self._codec.setToolTip(
                "AV1 is hidden because the last hardware probe did not report support."
            )
        else:
            self._codec.setToolTip("")

    def active_family(self) -> str:
        return str(self._backend.currentData() or "software")

    def encoder_name(self) -> str:
        backend = str(self._backend.currentData() or "software")
        codec = str(self._codec.currentData() or "hevc")
        if backend == "software":
            return software_name_for("h264" if codec == "h264" else "hevc")
        return encode_name_for(backend, codec)  # type: ignore[arg-type]

    def set_encoder_name(self, name: str) -> None:
        try:
            fam = encoder_family(name)
        except ValueError:
            fam = "x265"
        backend = fam if fam in {"nvenc", "qsv", "amf", "d3d12", "vulkan"} else "software"
        codec = "hevc"
        if fam in {"nvenc", "qsv", "amf", "d3d12", "vulkan"}:
            codec = name.split("_", 1)[0]
        elif fam == "x264":
            codec = "h264"
        with QSignalBlocker(self._backend):
            idx = self._backend.findData(backend)
            self._backend.setCurrentIndex(idx if idx >= 0 else 0)
        self._rebuild_codecs(preferred=codec)


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


def _gui_meta(field_info: FieldInfo, model_cls_name: str) -> tuple[str, str, str, str, str]:
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        gui = extra.get("gui") or {}
        cat = gui.get("category")
        tier = gui.get("tier")
        group = gui.get("group")
        when_family = gui.get("when_family")
        when_rc = gui.get("when_rc")
        if isinstance(cat, str) and tier in ("simple", "advanced"):
            return (
                cat,
                tier,
                str(group) if isinstance(group, str) else "general",
                str(when_family) if isinstance(when_family, str) else "all",
                str(when_rc) if isinstance(when_rc, str) else "",
            )
    return (
        _FALLBACK_CATEGORY.get(model_cls_name, model_cls_name.removesuffix("Cfg").lower()),
        "advanced",
        "general",
        "all",
        "",
    )


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
) -> tuple[QWidget, list[EditorBinding], Callable[[], None]]:
    """Return a widget with category tabs and bindings for apply/commit to a preset dict."""
    bindings: list[EditorBinding] = []
    rows: list[FieldRow] = []
    on_encoder_family_changed: list[Callable[[], None]] = []
    visibility_refreshers: list[Callable[[], None]] = []

    def trigger() -> None:
        for fn in on_encoder_family_changed:
            fn()
        if on_changed:
            on_changed()

    _walk_model_fields(
        Preset,
        (),
        bindings,
        rows,
        Preset.__name__,
        trigger,
        on_encoder_family_changed=on_encoder_family_changed,
    )

    _append_target_resolution_rows(bindings, rows, trigger)

    tabs_parent = QWidget()
    outer = QVBoxLayout(tabs_parent)
    outer.setContentsMargins(0, 0, 0, 0)

    tab_widget = QTabWidget()

    for cat in _CATEGORY_ORDER:
        cat_rows = [r for r in rows if r.category == cat]
        if not cat_rows:
            continue
        if cat == "encoding":
            scroll = _build_encoding_tab(
                cat_rows,
                on_encoder_family_changed,
                trigger,
                visibility_refreshers,
            )
        else:
            scroll = _build_standard_tab(cat_rows)
        tab_widget.addTab(scroll, _CATEGORY_TITLE.get(cat, cat.title()))

    outer.addWidget(tab_widget)

    def refresh_dynamic_visibility() -> None:
        for fn in visibility_refreshers:
            fn()

    return tabs_parent, bindings, refresh_dynamic_visibility


def _build_standard_tab(rows: list[FieldRow]) -> QScrollArea:
    simple_rows = [r for r in rows if r.tier == "simple"]
    advanced_rows = [r for r in rows if r.tier == "advanced"]

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    vl = QVBoxLayout(inner)
    vl.setContentsMargins(8, 8, 8, 8)

    simple_box = QGroupBox("Simple")
    sf = QFormLayout(simple_box)
    for row in simple_rows:
        sf.addRow(row.label, row.widget)
    vl.addWidget(simple_box)

    if advanced_rows:
        adv_toggle = QToolButton()
        adv_toggle.setCheckable(True)
        adv_toggle.setText("Show advanced")
        adv_wrap = QWidget()
        adv_v = QVBoxLayout(adv_wrap)
        adv_v.setContentsMargins(0, 0, 0, 0)
        adv_v.addWidget(adv_toggle)
        adv_box = QGroupBox()
        af = QFormLayout(adv_box)
        for row in advanced_rows:
            af.addRow(row.label, row.widget)
        adv_box.setVisible(False)
        adv_v.addWidget(adv_box)

        def _toggle_adv(on: bool, box=adv_box, btn=adv_toggle) -> None:
            box.setVisible(on)
            btn.setText("Hide advanced" if on else "Show advanced")

        adv_toggle.toggled.connect(_toggle_adv)
        vl.addWidget(adv_wrap)

    vl.addStretch(1)
    scroll.setWidget(inner)
    return scroll


def _build_encoding_tab(
    rows: list[FieldRow],
    on_encoder_family_changed: list[Callable[[], None]],
    trigger: Callable[[], None],
    visibility_refreshers: list[Callable[[], None]],
) -> QScrollArea:
    group_order = [
        ("selection", "Selection"),
        ("quality", "Quality"),
        ("nvenc", "NVIDIA NVENC Tuning"),
        ("qsv", "Intel QSV Tuning"),
        ("amf", "AMD AMF Tuning"),
        ("d3d12", "D3D12 Tuning"),
        ("vulkan", "Vulkan Tuning"),
        ("software", "Software Encoder Tuning"),
        ("decode", "Source and Decode"),
        ("polish", "Output Polish"),
        ("expert", "Expert"),
    ]

    by_group: dict[str, list[FieldRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    vl = QVBoxLayout(inner)
    vl.setContentsMargins(8, 8, 8, 8)

    visibility_rows: list[FieldRow] = []
    family_picker: EncoderNamePicker | None = None
    amf_rc_combo: QComboBox | None = None
    nvenc_rc_combo: QComboBox | None = None

    for row in rows:
        if row.path == ("encoder", "name") and isinstance(row.widget, EncoderNamePicker):
            family_picker = row.widget
        if row.path == ("encoder", "amf_rc") and isinstance(row.widget, QComboBox):
            amf_rc_combo = row.widget
        if row.path == ("encoder", "nvenc_rc") and isinstance(row.widget, QComboBox):
            nvenc_rc_combo = row.widget

    if family_picker is None:
        for row in rows:
            if row.path == ("encoder", "name"):
                row.widget = EncoderNamePicker(trigger=trigger)
                family_picker = row.widget if isinstance(row.widget, EncoderNamePicker) else None
                break

    for group_key, group_title in group_order:
        grouped = by_group.get(group_key, [])
        if not grouped:
            continue
        simple_rows = [r for r in grouped if r.tier == "simple"]
        advanced_rows = [r for r in grouped if r.tier == "advanced"]
        if not simple_rows and not advanced_rows:
            continue

        box = QGroupBox(group_title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        for row in simple_rows:
            form.addRow(row.label, row.widget)
            visibility_rows.append(row)
        box_layout.addLayout(form)

        if advanced_rows:
            adv_toggle = QToolButton()
            adv_toggle.setCheckable(True)
            adv_toggle.setText("Show advanced")
            box_layout.addWidget(adv_toggle)

            adv_box = QGroupBox()
            adv_form = QFormLayout(adv_box)
            for row in advanced_rows:
                adv_form.addRow(row.label, row.widget)
                visibility_rows.append(row)
            adv_box.setVisible(False)
            box_layout.addWidget(adv_box)

            def _toggle_adv(on: bool, btn=adv_toggle, adv=adv_box) -> None:
                adv.setVisible(on)
                btn.setText("Hide advanced" if on else "Show advanced")

            adv_toggle.toggled.connect(_toggle_adv)

        if group_key == "selection":
            hint = QLabel(_hardware_hint_text())
            hint.setWordWrap(True)
            box_layout.addWidget(hint)

        vl.addWidget(box)

    def _sync_visibility() -> None:
        active = family_picker.active_family() if family_picker is not None else "all"
        current_rc = ""
        if active == "amf" and amf_rc_combo is not None:
            current_rc = str(amf_rc_combo.currentData() or "")
        elif active == "nvenc" and nvenc_rc_combo is not None:
            current_rc = str(nvenc_rc_combo.currentData() or "")
        amf_codec = ""
        if family_picker is not None and active == "amf":
            amf_codec = str(family_picker._codec.currentData() or "")
        for row in visibility_rows:
            target = row.when_family
            visible = target in {"all", active}
            if target == "software":
                visible = active == "software"
            if visible and row.when_rc:
                visible = encoder_rc_matches_when(row.when_rc, active, current_rc)
            if row.path == ("encoder", "amf_bit_depth"):
                visible = visible and active == "amf" and amf_codec in {"hevc", "av1"}
            row.label.setVisible(visible)
            row.widget.setVisible(visible)

    visibility_refreshers.append(_sync_visibility)
    if family_picker is not None:
        on_encoder_family_changed.append(_sync_visibility)
    if amf_rc_combo is not None:
        amf_rc_combo.currentIndexChanged.connect(lambda _i=0: _sync_visibility())
    if nvenc_rc_combo is not None:
        nvenc_rc_combo.currentIndexChanged.connect(lambda _i=0: _sync_visibility())
    _sync_visibility()

    vl.addStretch(1)
    scroll.setWidget(inner)
    return scroll


def _walk_model_fields(
    model_cls: type[BaseModel],
    path: tuple[str, ...],
    bindings: list[EditorBinding],
    rows: list[FieldRow],
    model_cls_name: str,
    trigger: Callable[[], None],
    *,
    on_encoder_family_changed: list[Callable[[], None]],
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
            _walk_model_fields(
                sub,
                path + (fname,),
                bindings,
                rows,
                sub.__name__,
                trigger,
                on_encoder_family_changed=on_encoder_family_changed,
            )
            continue

        cat, tier, group, when_family, when_rc = _gui_meta(finfo, model_cls_name)
        title = fname.replace("_", " ").title()
        if path == ("encoder",) and fname == "name":
            title = "Encoder"
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
            on_encoder_family_changed=on_encoder_family_changed,
        )
        if leaf is None:
            continue
        row_widget = leaf
        _attach_tooltip(row_widget, desc)
        rows.append(
            FieldRow(
                category=cat,
                tier=tier,
                group=group,
                when_family=when_family,
                when_rc=when_rc,
                path=path + (fname,),
                label=label,
                widget=row_widget,
            ),
        )


def _make_leaf_widget(
    fname: str,
    finfo: FieldInfo,
    inner: Any,
    optional: bool,
    leaf_path: tuple[str, ...],
    bindings: list[EditorBinding],
    trigger: Callable[[], None],
    *,
    on_encoder_family_changed: list[Callable[[], None]],
) -> QWidget | None:
    if leaf_path == ("encoder", "name"):
        picker = EncoderNamePicker(trigger=trigger)

        def apply(data: dict[str, Any]) -> None:
            picker.set_encoder_name(str(deep_get(data, leaf_path)))
            for fn in on_encoder_family_changed:
                fn()

        def commit(data: dict[str, Any]) -> None:
            deep_set(data, leaf_path, picker.encoder_name())

        bindings.append(EditorBinding(apply, commit))
        return picker

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
    rows: list[FieldRow],
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
    rows.append(
        FieldRow(
            category="resolution",
            tier="simple",
            group="general",
            when_family="all",
            when_rc="",
            path=("target_resolution", "mode"),
            label=lm,
            widget=mode_combo,
        ),
    )

    ln = QLabel("Named target")
    ln.setToolTip(fi_named.description or "")
    named_combo.setToolTip(fi_named.description or "")
    rows.append(
        FieldRow(
            category="resolution",
            tier="simple",
            group="general",
            when_family="all",
            when_rc="",
            path=("target_resolution", "named"),
            label=ln,
            widget=row_named,
        ),
    )

    fi_w = TargetResolution.model_fields["width"]
    fi_h = TargetResolution.model_fields["height"]
    lh = QLabel("Explicit size")
    lh.setToolTip(((fi_w.description or "") + " " + (fi_h.description or "")).strip())
    rows.append(
        FieldRow(
            category="resolution",
            tier="advanced",
            group="general",
            when_family="all",
            when_rc="",
            path=("target_resolution", "explicit_size"),
            label=lh,
            widget=wh_row,
        ),
    )

    sync_visibility()
