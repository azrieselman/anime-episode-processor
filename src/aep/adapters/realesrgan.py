"""Real-ESRGAN ncnn-vulkan adapter.

Upstream binary: github.com/xinntao/Real-ESRGAN-ncnn-vulkan, pinned at 0.2.5.0.

Argv surface we use:
    realesrgan-ncnn-vulkan.exe
        -i <in_dir> -o <out_dir>
        -n <model_id>       e.g. realesr-animevideov3, realesrgan-x4plus-anime
        -s <2|3|4>          scale (the model embeds its native scale; -s lets you
                            request a smaller scale that the tool achieves by
                            re-downscaling. We always pass the model's native scale.)
        -t <tile_size>
        -g <gpu_id>
        -j <load:proc:save>
        -f png|webp
        -x                  TTA

Quirks:
* The model name (``-n``) doubles as the model file basename in the
  ``models/`` subdirectory; e.g. ``realesr-animevideov3.bin``/``.param``.
* This binary refuses to run if ``-m`` doesn't point at the dir containing
  those files. We pass it explicitly anyway because the implicit lookup is
  Windows-PATH-sensitive.
* The ``realesr-animevideov3`` model was specifically trained on video frames
  with temporal consistency; it's our default for anime episodes. The
  ``realesrgan-x4plus-anime`` model produces sharper stills but flickers in
  motion. We emit a warning if the user picks the still model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.ncnn_base import NcnnVulkanAdapter
from aep.constants import BIN_REALESRGAN

log = logging.getLogger(__name__)


# Models we know about. Native scale is what the model was trained for; we
# always pass that exact scale to ``-s`` to keep the model honest.
_KNOWN_MODELS: dict[str, dict[str, object]] = {
    "realesr-animevideov3": {
        "native_scale": 4,
        "video_trained": True,
        "description": "anime video model, temporally stable",
    },
    "realesrgan-x4plus-anime": {
        "native_scale": 4,
        "video_trained": False,
        "description": "anime stills model, sharp but flickers in motion",
    },
    "realesrgan-x4plus": {
        "native_scale": 4,
        "video_trained": False,
        "description": "general photo model, NOT recommended for anime",
    },
}


@dataclass(frozen=True)
class EsrganJob:
    input_dir: Path
    output_dir: Path
    model_id: str = "realesr-animevideov3"
    scale: int = 4                     # the *requested* scale; binary will downscale if < native
    tile_size: int = 192               # x4 from 1080p with tile=192 fits ~8.5GB on RTX 3080 10GB
    gpu_id: int = 0
    tta: bool = False
    frame_format: str = "png"


class RealesrganAdapter(NcnnVulkanAdapter):
    tool_id = "realesrgan-ncnn-vulkan"
    bin_name = BIN_REALESRGAN
    tools_subdir = "realesrgan-ncnn-vulkan"

    default_tile_size = 192            # ESRGAN is hungrier than CUGAN
    tile_size_floor = 64
    supports_format_flag = True

    # --------------------------------------------------------- model dir

    def resolve_model_dir(self, model_id: str) -> Path:
        """ESRGAN ships all models in a single ``models/`` dir alongside the binary.

        We don't subdir per-model — passing ``-n <model_id>`` selects the file.
        Override the base-class behavior accordingly.
        """
        bin_dir = self.path.parent
        candidates = [bin_dir / "models", bin_dir]
        for c in candidates:
            if c.is_dir() and (c / f"{model_id}.bin").is_file():
                return c.resolve()
        # Fall back to base behavior (will likely raise) so the error is consistent.
        return super().resolve_model_dir(model_id)

    # --------------------------------------------------------- argv

    def build_esrgan_argv(self, job: EsrganJob, *, tile_size_override: int | None = None) -> list[str | Path]:
        tile = tile_size_override if tile_size_override is not None else job.tile_size
        model_dir = self.resolve_model_dir(job.model_id)
        # ESRGAN takes the model name with `-n`, NOT `-m` (which points at the dir).
        argv = self.build_argv(
            input_dir=job.input_dir,
            output_dir=job.output_dir,
            model_dir=model_dir,
            tile_size=tile,
            gpu_id=job.gpu_id,
            scale=job.scale,
            tta=job.tta,
            frame_format=job.frame_format,
            extra=["-n", job.model_id],
        )
        return argv

    # --------------------------------------------------------- planner help

    @staticmethod
    def validate_combination(model_id: str, scale: int) -> list[str]:
        warnings: list[str] = []
        info = _KNOWN_MODELS.get(model_id)
        if info is None:
            warnings.append(
                f"Real-ESRGAN model {model_id!r} is not in our catalog; assuming x4 native scale."
            )
            return warnings
        native = int(info["native_scale"])  # type: ignore[arg-type]
        if scale > native:
            warnings.append(
                f"Real-ESRGAN model {model_id!r} has native scale x{native}; "
                f"requested x{scale} would extrapolate."
            )
        if scale < native:
            warnings.append(
                f"Real-ESRGAN model {model_id!r} runs at x{native} natively; the binary will "
                f"downscale the result to x{scale}, which wastes ~{(native ** 2 - scale ** 2)} "
                f"GPU-frames worth of compute. Consider Real-CUGAN at native x{scale}."
            )
        if not info["video_trained"]:
            warnings.append(
                f"Real-ESRGAN model {model_id!r} is a stills model and will flicker in motion. "
                f"For anime episodes prefer realesr-animevideov3."
            )
        return warnings
