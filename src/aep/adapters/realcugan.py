"""Real-CUGAN ncnn-vulkan adapter.

Upstream binary: github.com/nihui/realcugan-ncnn-vulkan, pinned at 20220728.

Argv surface we use:
    realcugan-ncnn-vulkan.exe
        -i <in_dir> -o <out_dir>
        -s <2|3|4>          scale factor
        -n <-1|0|1|2|3>     denoise level (-1 = no denoise; 3 = strongest)
        -m <model_dir>      e.g. .../models-pro
        -t <tile_size>
        -g <gpu_id>
        -j <load:proc:save>
        -f png|webp
        -x                  TTA mode (4x slower, marginal quality bump)

Quirks:
* CUGAN's model dirs are siblings of the binary, named ``models-<flavor>``
  (e.g. ``models-pro`` and ``models-se``). They contain per-(scale, denoise)
  pairs as separate ``.param``/``.bin`` files. Not every (scale, denoise)
  combination exists — for example the ``pro`` model only ships the 2x and 3x
  scales with denoise 3; selecting (4x, denoise=3) silently falls back upstream
  to (4x, denoise=0). We mirror this in the adapter and emit a warning.
* The ``models-nose`` legacy flavor is intentionally not exposed because its
  output is visibly worse on cel anime (the entire reason CUGAN exists is its
  noise-aware path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.ncnn_base import NcnnVulkanAdapter
from aep.constants import BIN_REALCUGAN

log = logging.getLogger(__name__)


# (scale, denoise) combinations that ship in stock CUGAN model dirs.
# Anything outside this set should produce a planner warning.
_SUPPORTED_PRO: frozenset[tuple[int, int]] = frozenset({
    (2, -1), (2, 0), (2, 3),
    (3, -1), (3, 0), (3, 3),
})
_SUPPORTED_SE: frozenset[tuple[int, int]] = frozenset({
    (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
    (3, -1), (3, 0), (3, 3),
    (4, -1), (4, 0), (4, 3),
})


@dataclass(frozen=True)
class CuganJob:
    """All parameters needed to run one CUGAN pass."""
    input_dir: Path
    output_dir: Path
    model_id: str = "models-pro"   # or "models-se"
    scale: int = 2
    denoise: int = 3
    tile_size: int = 256
    gpu_id: int = 0
    fp16: bool = True              # currently CUGAN binary is fp16 by default; flag is forward-looking
    tta: bool = False
    frame_format: str = "png"


class RealCuganAdapter(NcnnVulkanAdapter):
    tool_id = "realcugan-ncnn-vulkan"
    bin_name = BIN_REALCUGAN
    tools_subdir = "realcugan-ncnn-vulkan"

    default_tile_size = 256
    tile_size_floor = 64
    supports_format_flag = True

    # --------------------------------------------------------- argv

    def build_cugan_argv(self, job: CuganJob, *, tile_size_override: int | None = None) -> list[str | Path]:
        """Compose the CUGAN argv for one pass.

        ``tile_size_override`` exists so the OOM-fallback wrapper can shrink
        without rebuilding the dataclass.
        """
        tile = tile_size_override if tile_size_override is not None else job.tile_size
        model_dir = self.resolve_model_dir(job.model_id)
        # CUGAN denoise=-1 is encoded as `-n -1` (literal -1, not omitted).
        return self.build_argv(
            input_dir=job.input_dir,
            output_dir=job.output_dir,
            model_dir=model_dir,
            tile_size=tile,
            gpu_id=job.gpu_id,
            scale=job.scale,
            denoise=job.denoise,
            tta=job.tta,
            frame_format=job.frame_format,
        )

    # --------------------------------------------------------- planner help

    @staticmethod
    def validate_combination(model_id: str, scale: int, denoise: int) -> list[str]:
        """Return warnings for unusual (model, scale, denoise) tuples.

        We never refuse to run — the binary will fall back internally — but we
        surface to the user that they're not getting what they asked for.
        """
        warnings: list[str] = []
        supported = _SUPPORTED_PRO if model_id == "models-pro" else _SUPPORTED_SE
        if (scale, denoise) not in supported:
            warnings.append(
                f"Real-CUGAN {model_id} does not ship a (scale={scale}, denoise={denoise}) "
                f"checkpoint; the binary will silently fall back. "
                f"Consider scale=2/3 with denoise=3 for cel anime."
            )
        if scale == 4 and model_id == "models-pro":
            warnings.append(
                "Real-CUGAN pro lacks a native 4x checkpoint; the binary cascades 2x→2x. "
                "Real-ESRGAN x4 is usually preferable for 4x."
            )
        return warnings
