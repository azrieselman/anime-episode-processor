"""Mux package.

Stage 09 splits muxing into three pieces:

1. `mapping.py`   — pure planning. Consumes MediaInfo + StreamMappingCfg + container
                    target and produces a `StreamMappingPlan` (ffmpeg map/copy args
                    plus a list of post-mux mkvpropedit fixes). Side-effect-free.
2. `ffmpeg_mux.py` — executes the plan via ffmpeg's remux command builder. The default
                    path for most files; faster than mkvmerge for the common case of
                    "encoded video + a few audio/subs + chapters".
3. `mkvtoolnix_mux.py` — executes the plan via mkvmerge when the source has features
                    ffmpeg's mux is known to mangle (custom attachments, complex chapter
                    editions, tag/UID-sensitive metadata). Selection logic is
                    `decide_mux_tool(...)`.

Splitting "decide" from "execute" keeps stage 09 testable without spinning up real
binaries — most of our coverage hits `mapping.py` and `decide_mux_tool` with synthetic
MediaInfo.
"""
