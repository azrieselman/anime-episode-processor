"""Anime4KCPP v3.2+ model identifiers (see upstream Model wiki)."""

from __future__ import annotations

# Default when preset/upscaler config does not set `model`.
DEFAULT_ANIME4K_MODEL = "acnet-f8b8-hdn"

# Catalog for validation warnings; aliases for pre-3.2 presets are included.
KNOWN_ANIME4K_MODELS: frozenset[str] = frozenset({
    # Legacy (pre-v3.2) — may still be accepted by some builds
    "acnet",
    "acnet-gan",
    "acnet-hdn",
    "acnet-hdn-gan",
    "acnet-hdn0",
    # v3.2 ACNet legacy
    "acnet-legacy-gan",
    "acnet-legacy-hdn0",
    "acnet-legacy-hdn1",
    "acnet-legacy-hdn2",
    "acnet-legacy-hdn3",
    # v3.2 ACNet F8B4
    "acnet-f8b4",
    "acnet-f8b4-hdn",
    "acnet-f8b4-box",
    "acnet-f8b4-box-hdn",
    # v3.2 ACNet F8B8
    "acnet-f8b8",
    "acnet-f8b8-hdn",
    "acnet-f8b8-box",
    "acnet-f8b8-box-hdn",
    # v3.2 ACNet F8B18
    "acnet-f8b18",
    "acnet-f8b18-hdn",
    "acnet-f8b18-box",
    "acnet-f8b18-box-hdn",
    # v3.2 ARNet
    "arnet-f8b8",
    "arnet-f8b8-hdn",
    "arnet-f8b8-box",
    "arnet-f8b8-box-hdn",
    "arnet-f8b16",
    "arnet-f8b16-hdn",
    "arnet-f8b16-box",
    "arnet-f8b16-box-hdn",
    "arnet-f8b32",
    "arnet-f8b32-hdn",
    "arnet-f8b32-box",
    "arnet-f8b32-box-hdn",
    "arnet-f8b64",
    "arnet-f8b64-hdn",
    "arnet-f8b64-box",
    "arnet-f8b64-box-hdn",
    # ArtCNN
    "artcnn-c4f16",
    "artcnn-c4f16-dn",
    "artcnn-c4f16-ds",
    "artcnn-c4f32",
    "artcnn-c4f32-dn",
    "artcnn-c4f32-ds",
    # FSRCNNX
    "fsrcnnx-f8b4",
    "fsrcnnx-f8b4-distort-plus",
    "fsrcnnx-f16b4",
    "fsrcnnx-f16b4-distort-plus",
})
