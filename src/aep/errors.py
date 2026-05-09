"""Typed error hierarchy.

Every error the user can see in the GUI MUST be one of these; the GUI maps error class
to a human-readable hint string. This is the single source of truth for "what went wrong"
so we never surface a raw stack trace to a user.
"""

from __future__ import annotations

from typing import Any


class AEPError(Exception):
    """Base for all application errors. Carries an optional structured context."""

    user_hint: str = "An unexpected error occurred. Please check the logs."

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


class ConfigError(AEPError):
    user_hint = "Configuration is invalid. Reset to defaults from Settings or fix the file."


class PresetError(AEPError):
    user_hint = "A preset failed to load or validate. Open Presets to inspect."


class ToolNotFoundError(AEPError):
    user_hint = "A required tool is missing. Run the tool fetcher from Settings → Tools."


class ToolVersionMismatchError(AEPError):
    user_hint = "A tool version doesn't match the pinned version. Re-run the tool fetcher."


class ProbeError(AEPError):
    user_hint = "Could not analyze the file. It may be corrupt or in an unsupported format."


class PipelineError(AEPError):
    user_hint = "The processing pipeline failed. See the job log for details."


class StageError(PipelineError):
    user_hint = "A processing stage failed. The pipeline can sometimes resume after a fix."


class EncodeError(PipelineError):
    user_hint = "Encoding failed. Try a different encoder preset or check VRAM."


class MuxError(PipelineError):
    user_hint = "Muxing the final file failed. Subtitles or attachments may be incompatible."


class ValidationError(PipelineError):
    user_hint = "The output failed post-run validation. The file may be unusable."


class OOMError(PipelineError):
    user_hint = "Out of GPU memory. Lower tile size, disable interpolation, or use Low-VRAM Safe."


class CancelledError(AEPError):
    user_hint = "The job was cancelled."


class PausedError(AEPError):
    user_hint = "The job was paused."
