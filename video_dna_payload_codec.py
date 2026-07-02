from __future__ import annotations

"""
video_dna_payload_codec.py

Streamlit-facing wrapper around video_dna_compressor_v21.py.

Purpose
-------
The original v21 module is an end-to-end benchmark pipeline.  It performs video
compression, internal Simple Mapping DNA substitution, reconstruction, and
metrics in one call.  This wrapper reuses the same visual/audio model, but splits
it into two app-friendly API calls:

    compress_video_to_payload(video_path, config) -> payload_bytes + manifest
    reconstruct_video_from_payload(payload_bytes, manifest, output_dir) -> video + metrics

This allows the web app to keep the shared six-step DNA pipeline:
    Video Compression -> DNA Encoding -> Strand Design/Error -> Reconstruction.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import video_dna_compressor_v21 as v21

MAGIC = b"VDNA21P1"  # optional marker for future standalone packaging


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_payload_length(data: bytes, expected: int) -> bytes:
    raw = bytes(data or b"")
    expected = max(0, int(expected))
    if len(raw) < expected:
        return raw + bytes(expected - len(raw))
    return raw[:expected]


def _vote3_bytes(a: bytes, b: bytes, c: bytes) -> bytes:
    """Byte-wise majority vote for three redundant copies.

    If all three bytes differ, keep the first copy.  For independent DNA
    substitution noise this preserves the correct byte whenever at least two
    copies survive unchanged.
    """
    n = min(len(a), len(b), len(c))
    if n <= 0:
        return b""
    aa = np.frombuffer(a[:n], dtype=np.uint8)
    bb = np.frombuffer(b[:n], dtype=np.uint8)
    cc = np.frombuffer(c[:n], dtype=np.uint8)
    out = aa.copy()
    mask_ab = aa == bb
    mask_ac = aa == cc
    mask_bc = bb == cc
    out[mask_ab | mask_ac] = aa[mask_ab | mask_ac]
    out[mask_bc & ~(mask_ab | mask_ac)] = bb[mask_bc & ~(mask_ab | mask_ac)]
    return out.tobytes()


def _read_audio_float_from_wav(path: str) -> Tuple[int, np.ndarray]:
    v21._require_audio()
    sr, pcm = v21.wavfile.read(path)
    return int(sr), v21.pcm16_to_float(pcm)


def _encode_frame_to_payload(frame_rgb: np.ndarray, config: v21.VideoDNAConfig) -> Tuple[bytes, Dict[str, Any]]:
    """Encode one RGB frame into fixed-layout Y/Cr/Cb wavelet payload bytes."""
    v21._require_cv2()
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    h, w = frame.shape[:2]

    ycrcb = v21.cv2.cvtColor(frame, v21.cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)

    Cr_small = v21.resize_chroma(Cr.astype(np.uint8), scale=int(config.chroma_downsample)).astype(np.float32)
    Cb_small = v21.resize_chroma(Cb.astype(np.uint8), scale=int(config.chroma_downsample)).astype(np.float32)

    payload_Y = v21.wavelet_encode_fixed_array(
        Y - 128.0,
        band_mode=config.y_band_mode,
        level=int(config.wavelet_level),
        wavelet=config.wavelet,
        ll_step=float(config.spatial_ll_step),
        detail_step=float(config.spatial_detail_step),
        q_bits=int(config.q_bits),
    )
    payload_Cr = v21.wavelet_encode_fixed_array(
        Cr_small - 128.0,
        band_mode=config.chroma_band_mode,
        level=int(config.wavelet_level),
        wavelet=config.wavelet,
        ll_step=float(config.spatial_ll_step),
        detail_step=float(config.spatial_detail_step),
        q_bits=int(config.q_bits),
    )
    payload_Cb = v21.wavelet_encode_fixed_array(
        Cb_small - 128.0,
        band_mode=config.chroma_band_mode,
        level=int(config.wavelet_level),
        wavelet=config.wavelet,
        ll_step=float(config.spatial_ll_step),
        detail_step=float(config.spatial_detail_step),
        q_bits=int(config.q_bits),
    )

    meta = {
        "frame_shape": [int(h), int(w), 3],
        "cr_shape": [int(Cr_small.shape[0]), int(Cr_small.shape[1])],
        "cb_shape": [int(Cb_small.shape[0]), int(Cb_small.shape[1])],
        "y_len": int(len(payload_Y)),
        "cr_len": int(len(payload_Cr)),
        "cb_len": int(len(payload_Cb)),
        "total_len": int(len(payload_Y) + len(payload_Cr) + len(payload_Cb)),
    }
    return payload_Y + payload_Cr + payload_Cb, meta


def _decode_frame_from_payload(frame_payload: bytes, frame_meta: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """Decode one frame payload into RGB frame using the v21 wavelet model."""
    h, w, _ = [int(x) for x in frame_meta["frame_shape"]]
    cr_shape = tuple(int(x) for x in frame_meta["cr_shape"])
    cb_shape = tuple(int(x) for x in frame_meta["cb_shape"])
    y_len = int(frame_meta["y_len"])
    cr_len = int(frame_meta["cr_len"])
    cb_len = int(frame_meta["cb_len"])

    payload = _ensure_payload_length(frame_payload, y_len + cr_len + cb_len)
    pY = payload[:y_len]
    pCr = payload[y_len:y_len + cr_len]
    pCb = payload[y_len + cr_len:y_len + cr_len + cb_len]

    y_band = str(config.get("y_band_mode", "LL"))
    c_band = str(config.get("chroma_band_mode", "LL"))
    level = int(config.get("wavelet_level", 2))
    wavelet = str(config.get("wavelet", "haar"))
    ll_step = float(config.get("spatial_ll_step", 16.0))
    detail_step = float(config.get("spatial_detail_step", 10.0))
    q_bits = int(config.get("q_bits", 6))

    Y_rec_c = v21.wavelet_decode_fixed_array(pY, (h, w), y_band, level, wavelet, ll_step, detail_step, q_bits)
    Cr_rec_c_small = v21.wavelet_decode_fixed_array(pCr, cr_shape, c_band, level, wavelet, ll_step, detail_step, q_bits)
    Cb_rec_c_small = v21.wavelet_decode_fixed_array(pCb, cb_shape, c_band, level, wavelet, ll_step, detail_step, q_bits)

    Y_rec = v21.clip_u8(Y_rec_c + 128.0)
    Cr_small_rec = v21.clip_u8(Cr_rec_c_small + 128.0)
    Cb_small_rec = v21.clip_u8(Cb_rec_c_small + 128.0)
    Cr_rec = v21.upsample_chroma(Cr_small_rec, target_shape=(h, w))
    Cb_rec = v21.upsample_chroma(Cb_small_rec, target_shape=(h, w))

    ycrcb_rec = np.stack([Y_rec, Cr_rec, Cb_rec], axis=-1).astype(np.uint8)
    rgb_rec = v21.cv2.cvtColor(ycrcb_rec, v21.cv2.COLOR_YCrCb2RGB)
    return v21.light_post_filter_color(rgb_rec)


def _encode_audio_to_payload(input_video: str | Path, output_dir: str | Path, config: v21.VideoDNAConfig) -> Tuple[bytes, Optional[Dict[str, Any]]]:
    """Extract video audio and encode it as robust mu-law payload bytes."""
    if not bool(config.audio_enabled):
        return b"", None
    if not v21.has_audio_stream(input_video):
        return b"", None

    v21._require_audio()
    out_dir = Path(output_dir) / "audio_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_audio_wav = out_dir / "original_audio_mono.wav"
    v21.extract_audio_wav(
        input_video,
        original_audio_wav,
        sample_rate=int(config.audio_sample_rate),
        max_seconds=config.max_seconds,
    )
    sr, audio_pcm = v21.wavfile.read(original_audio_wav)
    audio_float = v21.pcm16_to_float(audio_pcm)
    audio_q = v21.mulaw_encode_float(audio_float)
    payload = audio_q.tobytes()
    meta = {
        "enabled": True,
        "codec": "mu_law_uint8",
        "sample_rate": int(sr),
        "sample_count": int(len(audio_float)),
        "duration_sec": float(len(audio_float) / max(1, int(sr))),
        "payload_bytes": int(len(payload)),
        "original_audio_wav": str(original_audio_wav),
    }
    return payload, meta


def _decode_audio_from_payload(payload: bytes, audio_meta: Dict[str, Any], output_dir: str | Path) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Decode mu-law audio payload and write raw/clean WAV outputs."""
    if not audio_meta or not audio_meta.get("enabled"):
        return None, None, {}
    v21._require_audio()
    out_dir = Path(output_dir) / "audio_reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = int(audio_meta.get("sample_rate", 16000))
    sample_count = int(audio_meta.get("sample_count", 0))
    q = np.frombuffer(_ensure_payload_length(payload, int(audio_meta.get("payload_bytes", len(payload)))), dtype=np.uint8)
    if sample_count > 0:
        if q.size < sample_count:
            q = np.pad(q, (0, sample_count - q.size), constant_values=128)
        q = q[:sample_count]

    recovered_raw = v21.mulaw_decode_uint8(q)
    recovered_clean = v21.light_audio_cleanup(recovered_raw, sample_rate=sample_rate)

    raw_wav = out_dir / "recovered_audio_raw.wav"
    clean_wav = out_dir / "recovered_audio_clean.wav"
    v21.wavfile.write(raw_wav, sample_rate, v21.float_to_pcm16(recovered_raw))
    v21.wavfile.write(clean_wav, sample_rate, v21.float_to_pcm16(recovered_clean))

    metrics: Dict[str, Any] = {
        "audio_sample_rate": sample_rate,
        "audio_duration_sec": float(len(recovered_clean) / max(1, sample_rate)),
    }
    original_audio_wav = audio_meta.get("original_audio_wav")
    if original_audio_wav and Path(original_audio_wav).exists():
        try:
            _sr, original = _read_audio_float_from_wav(original_audio_wav)
            metrics["audio_snr_raw_db"] = v21.audio_snr_db(original, recovered_raw)
            metrics["audio_snr_clean_db"] = v21.audio_snr_db(original, recovered_clean)
        except Exception:
            metrics["audio_snr_raw_db"] = float("nan")
            metrics["audio_snr_clean_db"] = float("nan")
    return str(raw_wav), str(clean_wav), metrics


def build_video_config(
    *,
    output_dir: str,
    mode: str = "Recommended AV",
    include_audio: bool = True,
    process_full_video: bool = True,
    max_seconds: Optional[float] = None,
    substitution_rate: float = 0.02,
    random_seed: int = 42,
    target_fps: Optional[float] = None,
    temporal_repair_window_sec: float = 0.5,
    repair_strength: str = "Balanced",
    protection_mode: str = "None",
    keyframe_interval_sec: float = 1.0,
) -> v21.VideoDNAConfig:
    """Create v21 config from simple Streamlit mode names.

    Extra Streamlit-only options are attached to the config instance and later
    stored in the manifest.  They do not alter the upstream v21 dataclass API.
    """
    presets: Dict[str, Dict[str, Any]] = {
        "High quality AV": {
            "target_width": 224,
            "target_height": 126,
            "fps_policy": "manual",
            "manual_fps": 12.0,
            "max_allowed_fps": 24.0,
            "q_bits": 6,
            "chroma_downsample": 4,
        },
        "Recommended AV": {
            "target_width": 192,
            "target_height": 108,
            "fps_policy": "comfortable",
            "comfort_fps": 12.0,
            "max_allowed_fps": 24.0,
            "q_bits": 6,
            "chroma_downsample": 4,
        },
        "Small robust AV": {
            "target_width": 160,
            "target_height": 90,
            "fps_policy": "manual",
            "manual_fps": 8.0,
            "max_allowed_fps": 12.0,
            "q_bits": 5,
            "chroma_downsample": 4,
        },
    }
    p = dict(presets.get(mode, presets["Recommended AV"]))
    if target_fps is not None and float(target_fps) > 0:
        p["fps_policy"] = "manual"
        p["manual_fps"] = float(target_fps)
        p["max_allowed_fps"] = max(float(target_fps), float(p.get("max_allowed_fps", target_fps)))

    fps_for_radius = float(p.get("manual_fps") or p.get("comfort_fps") or p.get("max_allowed_fps") or 12.0)
    repair_radius = max(1, int(round(fps_for_radius * float(temporal_repair_window_sec))))

    strength = str(repair_strength or "Balanced")
    strength_presets: Dict[str, Dict[str, Any]] = {
        "Off": {
            "cleanup_enabled": False,
            "repair_blend_strength": 0.0,
            "repair_base_threshold": 1e9,
            "cleanup_blend_strength": 0.0,
            "cleanup_base_threshold": 1e9,
        },
        "Light": {
            "cleanup_enabled": True,
            "repair_base_threshold": 32.0,
            "repair_mad_scale": 4.0,
            "repair_blend_strength": 0.55,
            "cleanup_base_threshold": 28.0,
            "cleanup_mad_scale": 3.5,
            "cleanup_blend_strength": 0.35,
        },
        "Balanced": {
            "cleanup_enabled": True,
            "repair_base_threshold": 28.0,
            "repair_mad_scale": 3.5,
            "repair_blend_strength": 0.85,
            "cleanup_base_threshold": 22.0,
            "cleanup_mad_scale": 3.0,
            "cleanup_blend_strength": 0.55,
        },
        "Strong": {
            "cleanup_enabled": True,
            "repair_base_threshold": 20.0,
            "repair_mad_scale": 2.5,
            "repair_blend_strength": 0.92,
            "cleanup_base_threshold": 16.0,
            "cleanup_mad_scale": 2.2,
            "cleanup_blend_strength": 0.70,
        },
        "Very strong": {
            "cleanup_enabled": True,
            "repair_base_threshold": 14.0,
            "repair_mad_scale": 1.8,
            "repair_blend_strength": 0.97,
            "cleanup_base_threshold": 12.0,
            "cleanup_mad_scale": 1.8,
            "cleanup_blend_strength": 0.82,
        },
    }
    p.update(strength_presets.get(strength, strength_presets["Balanced"]))
    p["repair_radius"] = repair_radius
    p["cleanup_radius"] = repair_radius
    p["repair_top_k"] = 3
    p["cleanup_top_k"] = 3

    cfg = v21.VideoDNAConfig(
        output_dir=output_dir,
        max_seconds=None if process_full_video else max_seconds,
        audio_enabled=bool(include_audio),
        substitution_rate=float(substitution_rate),
        random_seed=int(random_seed),
        **p,
    )
    setattr(cfg, "streamlit_mode", mode)
    setattr(cfg, "target_fps_ui", float(target_fps) if target_fps is not None else None)
    setattr(cfg, "temporal_repair_window_sec", float(temporal_repair_window_sec))
    setattr(cfg, "repair_strength", strength)
    setattr(cfg, "protection_mode", str(protection_mode or "None"))
    setattr(cfg, "keyframe_interval_sec", float(keyframe_interval_sec))
    return cfg


def compress_video_to_payload(input_video: str | Path, config: v21.VideoDNAConfig) -> Dict[str, Any]:
    """Compress video/audio streams into robust payload bytes plus manifest.

    Protection mode is applied at the payload layer before DNA encoding:
      - None: no redundancy.
      - Light: triplicate Y/LL payload for keyframes only.
      - Strong: triplicate Y/LL payload for every frame.

    On reconstruction, the three copies are combined by byte-wise majority vote.
    This is not classical ECC; it is unequal redundancy for the most important
    visual component (luma low-frequency wavelet bytes).
    """
    start = time.perf_counter()
    input_video = str(input_video)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = v21.ffprobe_video(input_video)
    if video_info is None:
        raise ValueError("Could not read video stream with ffprobe.")
    target_fps = v21.choose_target_fps(video_info, config)

    frame_dir = out_dir / "original_frames"
    frame_paths = v21.extract_frames(input_video, frame_dir, target_fps, config.target_width, config.target_height, config.max_seconds)
    original_frames = v21.load_frames(frame_dir, color="rgb")
    if not original_frames:
        raise ValueError("No frames extracted from video.")

    original_preview = out_dir / "original_preview.mp4"
    v21.make_video_from_frames(frame_dir, original_preview, fps=target_fps)

    visual_parts: List[bytes] = []
    frame_metas: List[Dict[str, Any]] = []
    cursor = 0
    for idx, frame in enumerate(original_frames):
        part, fm = _encode_frame_to_payload(frame, config)
        fm["frame_index"] = int(idx)
        fm["offset"] = int(cursor)
        cursor += len(part)
        visual_parts.append(part)
        frame_metas.append(fm)

    visual_payload = b"".join(visual_parts)
    audio_payload, audio_meta = _encode_audio_to_payload(input_video, out_dir, config)

    protection_mode = str(getattr(config, "protection_mode", "None") or "None")
    keyframe_interval_sec = float(getattr(config, "keyframe_interval_sec", 1.0) or 1.0)
    keyframe_interval_frames = max(1, int(round(float(target_fps) * keyframe_interval_sec)))
    protection_parts: List[bytes] = []
    protection_segments: List[Dict[str, Any]] = []
    protection_cursor = 0

    if protection_mode != "None":
        for fm, part in zip(frame_metas, visual_parts):
            frame_idx = int(fm.get("frame_index", 0))
            protect_this = False
            if protection_mode.startswith("Light"):
                protect_this = (frame_idx % keyframe_interval_frames) == 0
            elif protection_mode.startswith("Strong") or protection_mode.startswith("Very"):
                protect_this = True
            if not protect_this:
                continue
            y_len = int(fm.get("y_len", 0))
            if y_len <= 0:
                continue
            y = part[:y_len]
            # Original Y already exists in visual payload. Append two extra copies.
            protection_parts.append(y + y)
            protection_segments.append({
                "frame_index": frame_idx,
                "channel": "Y",
                "band": "selected_wavelet",
                "original_offset": int(fm.get("offset", 0)),
                "length": int(y_len),
                "extra_offset": int(protection_cursor),
                "extra_copies": 2,
                "total_copies": 3,
            })
            protection_cursor += 2 * y_len

    protection_payload = b"".join(protection_parts)
    protection_offset = int(len(visual_payload) + len(audio_payload))
    payload = visual_payload + audio_payload + protection_payload
    raw_rgb_bytes = v21.frame_raw_size_bytes(original_frames)

    cfg = asdict(config)
    cfg.update({
        "streamlit_mode": getattr(config, "streamlit_mode", ""),
        "temporal_repair_window_sec": float(getattr(config, "temporal_repair_window_sec", 0.5)),
        "repair_strength": str(getattr(config, "repair_strength", "Balanced")),
        "protection_mode": protection_mode,
        "keyframe_interval_sec": keyframe_interval_sec,
        "keyframe_interval_frames": keyframe_interval_frames,
    })
    manifest = {
        "domain": "video",
        "version": "v21_streamlit_payload_protected",
        "config": cfg,
        "video_info": video_info,
        "target_fps": float(target_fps),
        "original_frames_count": int(len(original_frames)),
        "original_frames_dir": str(frame_dir),
        "original_preview": str(original_preview),
        "raw_rgb_bytes": int(raw_rgb_bytes),
        "visual": {
            "enabled": True,
            "resolution": f"{int(config.target_width)}x{int(config.target_height)}",
            "target_width": int(config.target_width),
            "target_height": int(config.target_height),
            "frame_count": int(len(original_frames)),
            "payload_offset": 0,
            "payload_bytes": int(len(visual_payload)),
            "frame_metas": frame_metas,
        },
        "audio": {
            **(audio_meta or {"enabled": False}),
            "payload_offset": int(len(visual_payload)),
            "payload_bytes": int(len(audio_payload)),
        },
        "protection": {
            "mode": protection_mode,
            "description": "Triplicate luma/Y wavelet payload for keyframes or all frames, decoded by byte-wise majority vote.",
            "payload_offset": protection_offset,
            "payload_bytes": int(len(protection_payload)),
            "segments": protection_segments,
            "protected_frames": int(len({int(s['frame_index']) for s in protection_segments})),
            "keyframe_interval_frames": int(keyframe_interval_frames),
            "keyframe_interval_sec": float(keyframe_interval_sec),
            "overhead_vs_unprotected": float(len(protection_payload) / max(1, len(visual_payload) + len(audio_payload))),
        },
        "payload_bytes": int(len(payload)),
        "unprotected_payload_bytes": int(len(visual_payload) + len(audio_payload)),
        "compression_vs_raw_rgb": float(raw_rgb_bytes / max(1, len(payload))),
        "compression_time_seconds": float(time.perf_counter() - start),
    }
    manifest_path = out_dir / "video_payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path)

    return {
        "payload_bytes": payload,
        "manifest": manifest,
        "preview_video": str(original_preview),
        "raw_rgb_bytes": int(raw_rgb_bytes),
        "frame_count": int(len(original_frames)),
        "target_fps": float(target_fps),
        "output_dir": str(out_dir),
    }


def reconstruct_video_from_payload(decoded_payload: bytes, manifest: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    """Reconstruct video/audio from recovered payload bytes and existing manifest."""
    start = time.perf_counter()
    v21._require_cv2()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_payload_len = int(manifest.get("payload_bytes", len(decoded_payload or b"")))
    payload = _ensure_payload_length(decoded_payload, expected_payload_len)
    cfg = dict(manifest.get("config", {}))
    visual = manifest.get("visual", {}) or {}
    audio = manifest.get("audio", {}) or {"enabled": False}

    frame_metas = list(visual.get("frame_metas", []))
    protection = manifest.get("protection", {}) or {}
    protection_offset = int(protection.get("payload_offset", int(visual.get("payload_bytes", 0)) + int(audio.get("payload_bytes", 0))))
    protection_by_frame: Dict[int, Dict[str, Any]] = {
        int(seg.get("frame_index", -1)): seg
        for seg in protection.get("segments", []) or []
        if str(seg.get("channel", "")) == "Y"
    }

    base_frames: List[np.ndarray] = []
    protected_used = 0
    for fm in frame_metas:
        off = int(fm.get("offset", 0))
        n = int(fm.get("total_len", 0))
        frame_payload = bytearray(_ensure_payload_length(payload[off:off + n], n))

        # Unequal protection: replace the noisy original Y payload with a
        # majority-voted Y payload from original + two redundant copies.
        seg = protection_by_frame.get(int(fm.get("frame_index", -1)))
        if seg:
            y_len = int(seg.get("length", fm.get("y_len", 0)))
            extra_off = protection_offset + int(seg.get("extra_offset", 0))
            extra = _ensure_payload_length(payload[extra_off:extra_off + 2 * y_len], 2 * y_len)
            if y_len > 0 and len(extra) >= 2 * y_len and len(frame_payload) >= y_len:
                y0 = bytes(frame_payload[:y_len])
                y1 = extra[:y_len]
                y2 = extra[y_len:2 * y_len]
                voted_y = _vote3_bytes(y0, y1, y2)
                frame_payload[:len(voted_y)] = voted_y
                protected_used += 1

        base_frames.append(_decode_frame_from_payload(bytes(frame_payload), fm, cfg))

    base_dir = output_dir / "base_frames"
    v21.save_frames(base_frames, base_dir)
    target_fps = float(manifest.get("target_fps", cfg.get("comfort_fps", 12.0)))
    base_video = output_dir / "base_reconstructed_video.mp4"
    v21.make_video_from_frames(base_dir, base_video, fps=target_fps)

    temporal_frames = v21.temporal_similarity_repair(
        base_frames,
        radius=int(cfg.get("repair_radius", 6)),
        top_k=int(cfg.get("repair_top_k", 3)),
        base_threshold=float(cfg.get("repair_base_threshold", 28.0)),
        mad_scale=float(cfg.get("repair_mad_scale", 3.5)),
        blend_strength=float(cfg.get("repair_blend_strength", 0.85)),
        max_mask_ratio=float(cfg.get("repair_max_mask_ratio", 0.40)),
    )
    final_frames = temporal_frames
    cleanup_enabled = bool(cfg.get("cleanup_enabled", True))
    if cleanup_enabled:
        final_frames = v21.reliability_local_cleanup(
            temporal_frames,
            radius=int(cfg.get("cleanup_radius", 6)),
            top_k=int(cfg.get("cleanup_top_k", 3)),
            base_threshold=float(cfg.get("cleanup_base_threshold", 22.0)),
            mad_scale=float(cfg.get("cleanup_mad_scale", 3.0)),
            blend_strength=float(cfg.get("cleanup_blend_strength", 0.55)),
            max_mask_ratio=float(cfg.get("cleanup_max_mask_ratio", 0.25)),
        )

    final_dir = output_dir / "final_frames"
    v21.save_frames(final_frames, final_dir)
    visual_video = output_dir / "recovered_visual_video.mp4"
    v21.make_video_from_frames(final_dir, visual_video, fps=target_fps)

    audio_raw_wav = audio_clean_wav = None
    audio_metrics: Dict[str, Any] = {}
    if audio.get("enabled"):
        aoff = int(audio.get("payload_offset", visual.get("payload_bytes", 0)))
        alen = int(audio.get("payload_bytes", 0))
        audio_payload = payload[aoff:aoff + alen]
        audio_raw_wav, audio_clean_wav, audio_metrics = _decode_audio_from_payload(audio_payload, audio, output_dir)

    final_video = str(visual_video)
    if audio_clean_wav:
        try:
            muxed = output_dir / "final_visual_plus_dna_audio.mp4"
            final_video = v21.mux_video_audio(visual_video, audio_clean_wav, muxed, audio_bitrate=str(cfg.get("audio_bitrate_for_mux", "96k")))
        except Exception:
            final_video = str(visual_video)

    metrics: Dict[str, Any] = {
        "playable": bool(v21.playable_video(final_video)),
        "target_fps": float(target_fps),
        "resolution": visual.get("resolution", ""),
        "frames_recovered": int(len(final_frames)),
        "reconstruction_time_seconds": float(time.perf_counter() - start),
        "output_video": str(final_video),
        "visual_video": str(visual_video),
        **audio_metrics,
    }

    original_frame_dir = manifest.get("original_frames_dir")
    try:
        if original_frame_dir and Path(original_frame_dir).exists():
            original_frames = v21.load_frames(original_frame_dir, color="rgb")
            base_metrics = v21.compute_video_metrics(original_frames, base_frames)
            final_metrics = v21.compute_video_metrics(original_frames, final_frames)
            metrics.update(final_metrics)
            metrics.update({
                "base_video_psnr": base_metrics.get("video_psnr", float("nan")),
                "base_video_ssim": base_metrics.get("video_ssim", float("nan")),
                "base_frame_correlation": base_metrics.get("frame_correlation", float("nan")),
                "psnr_gain_from_repair": float(final_metrics.get("video_psnr", float("nan"))) - float(base_metrics.get("video_psnr", float("nan"))),
                "ssim_gain_from_repair": float(final_metrics.get("video_ssim", float("nan"))) - float(base_metrics.get("video_ssim", float("nan"))),
            })
        else:
            metrics.update({"frames_compared": 0, "video_psnr": float("nan"), "video_ssim": float("nan"), "frame_correlation": float("nan")})
    except Exception:
        metrics.update({"frames_compared": 0, "video_psnr": float("nan"), "video_ssim": float("nan"), "frame_correlation": float("nan")})

    metrics.update({
        "base_video": str(base_video),
        "protection_mode": str((manifest.get("protection", {}) or {}).get("mode", "None")),
        "protected_frames_used": int(protected_used),
        "protected_frames_total": int((manifest.get("protection", {}) or {}).get("protected_frames", 0)),
        "protection_payload_bytes": int((manifest.get("protection", {}) or {}).get("payload_bytes", 0)),
        "repair_applied": bool(str(cfg.get("repair_strength", "Balanced")) != "Off"),
        "cleanup_applied": bool(cleanup_enabled),
    })

    summary_path = output_dir / "video_reconstruction_metrics.json"
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "final_video": str(final_video),
        "visual_video": str(visual_video),
        "base_video": str(base_video),
        "audio_raw_wav": audio_raw_wav,
        "audio_clean_wav": audio_clean_wav,
        "metrics": metrics,
        "metrics_json": str(summary_path),
    }
