"""Stream inspector — shows MediaInfo for the selected job.

This is the view a user opens to confirm "yes, my MKV has 2 audio tracks (jpn + eng), 5
sub tracks, 47 attachments, 24 chapters" before committing to processing. We display
language, default/forced flags, codecs, dimensions, and any ffprobe notes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.gui import theme
from aep.media.models import MediaInfo

log = logging.getLogger(__name__)


def _format_bytes(n: int) -> str:
    """Render a byte count with one of B / KB / MB / GB suffixes.

    We pick a binary base (1024) because attachments inside MKV containers
    are tracked in raw byte counts and humans reading the inspector are
    almost always sanity-checking against `dir`/`ls -l` style output.
    """
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _yn(b: bool) -> str:
    return "✔" if b else ""


class StreamInspectorView(QWidget):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._job_id: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        head = QHBoxLayout()
        head.addWidget(theme.make_page_title_label("Stream Inspector", self))
        head.addStretch(1)
        self._reanalyze_btn = QPushButton("Re-analyze")
        self._reanalyze_btn.clicked.connect(self._on_reanalyze)
        head.addWidget(self._reanalyze_btn)
        root.addLayout(head)

        self._summary = QLabel("Select a job to inspect.")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        self._tabs = QTabWidget()
        self._video_table = self._make_table(
            ["#", "Codec", "Resolution", "FPS", "PixFmt", "Lang", "Title", "Default", "Forced"]
        )
        self._audio_table = self._make_table(
            ["#", "Codec", "Channels", "Layout", "Sample Rate", "Lang", "Title", "Default", "Forced"]
        )
        self._sub_table = self._make_table(
            ["#", "Codec", "Lang", "Title", "Default", "Forced"]
        )
        self._att_table = self._make_table(["#", "Filename", "MIME", "Size"])
        self._chap_table = self._make_table(["#", "Start", "End", "Title"])
        self._notes_label = QLabel("")
        self._notes_label.setWordWrap(True)
        theme.style_inspector_note_label(self._notes_label)

        self._tabs.addTab(self._video_table, "Video")
        self._tabs.addTab(self._audio_table, "Audio")
        self._tabs.addTab(self._sub_table, "Subtitles")
        self._tabs.addTab(self._att_table, "Attachments")
        self._tabs.addTab(self._chap_table, "Chapters")
        root.addWidget(self._tabs, 1)
        root.addWidget(self._notes_label)

    def _make_table(self, headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def set_job(self, job_id: str | None) -> None:
        self._job_id = job_id
        # Cached attachment-size lookup (filename -> bytes), populated lazily
        # from `mkvmerge -J` so we only pay the subprocess cost once per job.
        self._att_sizes: dict[str, int] = {}
        if not job_id:
            self._clear()
            return
        # Find job and either use cached probe or run fresh.
        for j in self._services.jobs.list_jobs():
            if j.id != job_id:
                continue
            if j.probe:
                info = MediaInfo.model_validate(j.probe)
                self._att_sizes = self._fetch_attachment_sizes(Path(j.source_path), info)
                self._render(info)
            else:
                self._summary.setText("No probe data yet — click Re-analyze.")
                self._clear_tables()
            return

    def _on_reanalyze(self) -> None:
        if not self._job_id:
            return
        for j in self._services.jobs.list_jobs():
            if j.id == self._job_id:
                try:
                    info = self._services.media.analyze(Path(j.source_path))
                    self._att_sizes = self._fetch_attachment_sizes(
                        Path(j.source_path), info,
                    )
                    self._render(info)
                except Exception as exc:
                    self._summary.setText(f"Analysis failed: {exc}")
                return

    def _clear(self) -> None:
        self._summary.setText("Select a job to inspect.")
        self._clear_tables()

    def _clear_tables(self) -> None:
        for t in (self._video_table, self._audio_table, self._sub_table,
                  self._att_table, self._chap_table):
            t.setRowCount(0)
        self._notes_label.setText("")

    def _render(self, info: MediaInfo) -> None:
        dur = info.fmt.duration_s or 0
        size_mb = (info.fmt.size_bytes or 0) / (1024 * 1024)
        self._summary.setText(
            f"{info.fmt.format_name}  ·  {dur:.1f}s  ·  {size_mb:.1f} MB  ·  "
            f"video={len(info.video_streams)}  audio={len(info.audio_streams)}  "
            f"subs={len(info.subtitle_streams)}  chapters={len(info.chapters)}  "
            f"attachments={len(info.attachments)}  mkv={'yes' if info.is_matroska else 'no'}"
        )

        self._fill_video(info)
        self._fill_audio(info)
        self._fill_subs(info)
        self._fill_attachments(info)
        self._fill_chapters(info)

        if info.notes:
            self._notes_label.setText("Notes:\n  • " + "\n  • ".join(info.notes))
        else:
            self._notes_label.setText("")

    def _fill_video(self, info: MediaInfo) -> None:
        rows = info.video_streams
        self._video_table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            self._set_row(self._video_table, r, [
                str(s.index), s.codec_name or "?",
                f"{s.width or '?'}x{s.height or '?'}",
                s.avg_frame_rate or "?", s.pix_fmt or "",
                s.language or "", s.title or "",
                _yn(s.disposition.default), _yn(s.disposition.forced),
            ])

    def _fill_audio(self, info: MediaInfo) -> None:
        rows = info.audio_streams
        self._audio_table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            self._set_row(self._audio_table, r, [
                str(s.index), s.codec_name or "?",
                str(s.channels or "?"), s.channel_layout or "",
                f"{s.sample_rate}" if s.sample_rate else "",
                s.language or "", s.title or "",
                _yn(s.disposition.default), _yn(s.disposition.forced),
            ])

    def _fill_subs(self, info: MediaInfo) -> None:
        rows = info.subtitle_streams
        self._sub_table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            self._set_row(self._sub_table, r, [
                str(s.index), s.codec_name or "?",
                s.language or "", s.title or "",
                _yn(s.disposition.default), _yn(s.disposition.forced),
            ])

    def _fill_attachments(self, info: MediaInfo) -> None:
        rows = info.attachments
        self._att_table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            filename = s.filename or s.tags.get("filename", "")
            size_bytes = self._att_sizes.get(filename, 0)
            size_str = _format_bytes(size_bytes) if size_bytes else ""
            self._set_row(self._att_table, r, [
                str(s.index),
                filename,
                s.mimetype or s.tags.get("mimetype", ""),
                size_str,
            ])

    def _fill_chapters(self, info: MediaInfo) -> None:
        rows = info.chapters
        self._chap_table.setRowCount(len(rows))
        for r, c in enumerate(rows):
            self._set_row(self._chap_table, r, [
                str(c.id),
                f"{c.start_time_s:.3f}",
                f"{c.end_time_s:.3f}",
                c.title or "",
            ])

    @staticmethod
    def _fetch_attachment_sizes(source: Path, info: MediaInfo) -> dict[str, int]:
        """Look up attachment byte counts via `mkvmerge -J`.

        Returns an empty dict (no sizes shown) if anything goes wrong --
        mkvmerge missing, source isn't MKV, identify call raised, etc. The
        inspector should not refuse to render just because we couldn't get
        sizes; sizes are advisory.

        The join key is the attachment filename (mkvmerge.file_name vs
        ffprobe.filename). Filenames inside an MKV container are unique
        in practice so we don't need to disambiguate by index.
        """
        if not info.is_matroska or not info.attachments:
            return {}
        try:
            from aep.adapters.mkvtoolnix import MkvmergeAdapter
            ident = MkvmergeAdapter().identify(source)
        except Exception as exc:
            log.debug("attachment size lookup skipped: %s", exc)
            return {}
        return {a.file_name: a.size for a in ident.attachments if a.file_name}

    @staticmethod
    def _set_row(table: QTableWidget, row: int, values: list[str]) -> None:
        for col, v in enumerate(values):
            item = QTableWidgetItem(str(v))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, col, item)
