"""Waifu2x ncnn-vulkan adapter.

Upstream binary: github.com/nihui/waifu2x-ncnn-vulkan, pinned at 20220728.

Argv surface we use:
    waifu2x-ncnn-vulkan.exe
        -i <in_dir> -o <out_dir>
        -n <-1|0|1|2|3>     denoise level (-1 = no denoise; 3 = strongest)
        -s <1|2>            scale factor (cunet supports 1 and 2; upconv_7 only 2)
        -m <model_dir>      e.g. .../models-cunet
        -t <tile_size>
        -g <gpu_id>
        -j <load:proc:save>
        -f png|webp
        -x                  TTA mode (4x slower, marginal quality bump)

Quirks:
* Waifu2x model dirs are siblings of the binary, named ``models-<flavor>``,
  matching CUGAN's convention. Each directory contains separate ``.param``/
  ``.bin`` files keyed by (scale, denoise). Not every combination ships:
    - ``models-cunet``: scale {1, 2} × denoise {-1, 0, 1, 2, 3}  (anime, default)
    - ``models-upconv_7_anime_style_art_rgb``: scale {2} × denoise {-1, 0, 1, 2, 3}
    - ``models-upconv_7_photo``: scale {2} × denoise {-1, 0, 1, 2, 3}  (photo)
* The legacy ``models-cunet`` is the right default for cel anime: it has the
  noise-aware path and supports 1x denoise-only mode (handy for compression
  cleanup without resolution change). The ``upconv_7`` variants are older
  and visibly softer on cel art; the ``photo`` flavor specifically is wrong
  for anime and we warn loudly.
* Waifu2x has no native 4x — selecting scale=4 here is invalid; the planner
  should validate before this adapter ever sees the job. We surface a hard
  warning if asked for anything outside the matrix.
* Unlike CUGAN which accepts only specific (scale, denoise) checkpoints,
  waifu2x ships every combination in the supported matrix; off-grid values
  are denoise levels outside {-1..3} or scale outside the model's set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.ncnn_base import NcnnVulkanAdapter
from aep.constants import BIN_WAIFU2X

log = logging.getLogger(__name__)


# Model catalog. ``scales`` and ``denoise`` enumerate the checkpoints that
# actually ship in stock model dirs; anything outside earns a planner warning.
_KNOWN_MODELS: dict[str, dict[str, object]] = {
    "models-cunet": {
        "scales": frozenset({1, 2}),
        "denoise": frozenset({-1, 0, 1, 2, 3}),
        "anime_trained": True,
        "description": "anime cel model, noise-aware, supports 1x denoise-only",
    },
    "models-upconv_7_anime_style_art_rgb": {
        "scales": frozenset({2}),
        "denoise": frozenset({-1, 0, 1, 2, 3}),
        "anime_trained": True,
        "description": "older anime model, softer than cunet; 2x only",
    },
    "models-upconv_7_photo": {
        "scales": frozenset({2}),
        "denoise": frozenset({-1, 0, 1, 2, 3}),
        "anime_trained": False,
        "description": "photo model, NOT recommended for anime",
    },
}


@dataclass(frozen=True)
class Waifu2xJob:
    """All parameters needed to run one waifu2x pass."""
    input_dir: Path
    output_dir: Path
    model_id: str = "models-cunet"     # cunet is anime default
    scale: int = 2
    denoise: int = 3
    tile_size: int = 256
    gpu_id: int = 0
    fp16: bool = True                  # waifu2x-ncnn-vulkan is fp16 by default; flag is forward-looking
    tta: bool = False
    frame_format: str = "png"


class Waifu2xAdapter(NcnnVulkanAdapter):
    tool_id = "waifu2x-ncnn-vulkan"
    bin_name = BIN_WAIFU2X
    tools_subdir = "waifu2x-ncnn-vulkan"

    default_tile_size = 256
    tile_size_floor = 64
    supports_format_flag = True

    # --------------------------------------------------------- argv

    def build_waifu2x_argv(
        self,
        job: Waifu2xJob,
        *,
        tile_size_override: int | None = None,
    ) -> list[str | Path]:
        """Compose the waifu2x argv for one pass.

        ``tile_size_override`` exists so the OOM-fallback wrapper can shrink
        without rebuilding the dataclass.
        """
        tile = tile_size_override if tile_size_override is not None else job.tile_size
        model_dir = self.resolve_model_dir(job.model_id)
        # Waifu2x denoise=-1 is encoded as `-n -1` (literal -1, not omitted).
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

        We never refuse to run — the caller decides — but we surface that
        the user is asking for something outside the supported matrix or
        the wrong model flavor for anime content.
        """
        warnings: list[str] = []
        info = _KNOWN_MODELS.get(model_id)
        if info is None:
            warnings.append(
                f"waifu2x model {model_id!r} is not in our catalog; "
                f"known models: {sorted(_KNOWN_MODELS.keys())}"
            )
            return warnings

        scales = info["scales"]  # type: ignore[assignment]
        denoises = info["denoise"]  # type: ignore[assignment]
        assert isinstance(scales, frozenset)
        assert isinstance(denoises, frozenset)

        if scale not in scales:
            warnings.append(
                f"waifu2x {model_id} does not ship a scale={scale} checkpoint "
                f"(supported: {sorted(scales)}). Pick scale=2 for cel anime, "
                f"or scale=1 with models-cunet for denoise-only passes."
            )
        if denoise not in denoises:
            warnings.append(
                f"waifu2x {model_id} does not ship a denoise={denoise} checkpoint "
                f"(supported: {sorted(denoises)}). Use -1 for no denoise; 3 is strongest."
            )
        if not info["anime_trained"]:
            warnings.append(
                f"waifu2x model {model_id!r} is a photo model and will produce "
                f"smoother, less line-faithful output on cel anime. "
                f"For anime episodes prefer models-cunet."
            )
        return warnings
