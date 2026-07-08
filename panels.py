from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from compression_pipeline import CompressionCandidate
from config import WORK_ROOT, MAPPING_OPTIONS, DNA_PREVIEW_HEIGHT
from dna_codec import gc_content, homopolymer_stats
from dna_mapping import decode_dna_with_mapping, encode_bytes_to_dna, validate_container
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv
from restore_analysis import image_metrics, text_similarity, write_restored_file
from ui_helpers import download_bytes_button, fmt_bytes, get_domain, magic_dict, preview_file, save_upload, step_header

from utils_bits_v2 import detect_magic, bytes_to_bitstring
from ui_design_system.ui_labels import (
    PANEL_TITLES, BUTTONS, METRICS, DATA_SOURCES, FIELDS, MESSAGES, DOWNLOAD_FILES, display_mapping
)
from ui_design_system.design_tokens import REGION_COLORS


def _write_preview_inner(path: str, kind: str, title: str) -> str | None:
    """Compatibility fallback for older ui_helpers.py files."""
    return None


# -----------------------------------------------------------------------------
# Small shared helpers.  Keep this file intentionally simple: each helper below
# is used by exactly one or more visible pipeline panels.
# -----------------------------------------------------------------------------

MAX_BINARY_PREVIEW_BYTES = 512
MAX_BITS_PREVIEW_CHARS = 4096

UPLOAD_TYPES_BY_DOMAIN = {
    "image": ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
    "text": ["txt", "csv", "tsv", "json", "xml", "html", "md", "py", "log"],
    "audio": ["wav", "mp3", "m4a", "flac", "ogg", "aac"],
    "video": ["mp4", "mov", "avi", "mkv", "webm", "m4v"],
}

METHOD_OPTIONS_BY_DOMAIN = {
    "image": [
        "RGB pixels",
        "Grayscale pixels",
        "Binary pixels",
        "Robust Low-Resolution",
        "Base + Local Detail",
        "Local Block Coding",
        "Base + WebP Detail",
    ],
    "text": [
        "Dense fixed-vocab token",
        "Tokenization + CRC repair",
    ],
    "audio": [
        "High quality robust",
        "Recommended",
        "Small robust",
        "Experimental AI cleanup",
    ],
    "video": [
        "High quality AV",
        "Recommended AV",
        "Small robust AV",
    ],
}

METHOD_CAPTIONS_BY_DOMAIN = {
    "image": {
        "RGB pixels": "Raw RGB bytes. Larger payload, but DNA substitutions usually damage only local pixels.",
        "Grayscale pixels": "Raw 8-bit grayscale bytes. Smaller than RGB and easy to reconstruct.",
        "Binary pixels": "Packed black/white pixels. Best for MNIST, masks, sketches, and document-like images.",
        "Robust Low-Resolution": "Quantized low-resolution image. Very robust, but blurrier.",
        "Base + Local Detail": "Robust base image plus local detail. Main DNA-friendly image method.",
        "Local Block Coding": "Local DCT-style block payload. Errors remain spatially local.",
        "Base + WebP Detail": "Base layer with WebP detail tiles. Kept as an advanced comparison, not a standard WebP/JPEG benchmark.",
    },
    "text": {
        "Dense fixed-vocab token": "Fixed-width token baseline. No zlib stream, so substitutions do not destroy a compressed container.",
        "Tokenization + CRC repair": "BERT/RoBERTa token IDs with CRC detection, mask preview, and candidate repair support.",
    },
    "audio": {
        "High quality robust": "16 kHz mu-law 8-bit payload plus click repair.",
        "Recommended": "16 kHz mu-law 4-bit payload plus click repair. Balanced default.",
        "Small robust": "8 kHz mu-law 4-bit payload. Small and robust.",
        "Experimental AI cleanup": "16 kHz mu-law 4-bit payload plus optional AI cleanup after reconstruction.",
    },
    "video": {
        "High quality AV": "Higher visual quality, larger payload.",
        "Recommended AV": "Balanced quality and DNA length.",
        "Small robust AV": "Lower resolution/FPS and smaller payload for robustness tests.",
    },
}


def _active_domain() -> str:
    domain = str(st.session_state.get("active_domain") or "").lower().strip()
    if domain in METHOD_OPTIONS_BY_DOMAIN:
        return domain
    path = st.session_state.get("input_path") or ""
    data = st.session_state.get("input_bytes", b"") or b""
    detected = get_domain(path, data)
    return detected if detected in METHOD_OPTIONS_BY_DOMAIN else "image"


def _method_options_for_domain(domain: str) -> List[str]:
    return list(METHOD_OPTIONS_BY_DOMAIN.get(domain, []))


def _method_caption(domain: str, method: str) -> str:
    return METHOD_CAPTIONS_BY_DOMAIN.get(domain, {}).get(method, "")


def _selected_payload_dir(domain: str) -> Any:
    out_dir = WORK_ROOT / "selected_method_payloads" / str(domain or "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _safe_method_slug(method: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(method or "method")).strip("_").lower() or "method"


def _write_selected_payload(domain: str, method: str, payload: bytes, ext: str = ".bin") -> str:
    out_dir = _selected_payload_dir(domain)
    path = out_dir / f"{_safe_method_slug(method)}_payload{ext or '.bin'}"
    path.write_bytes(bytes(payload or b""))
    return str(path)


def _clear_after_method_selection() -> None:
    for key in [
        "dna", "bits", "codec_meta", "strand_rows",
        "decoded_data", "decoded_raw_pixels", "decoded_bits", "decoded_meta", "decoded_magic",
        "decoded_valid", "decoded_note", "raw_restore_info", "restored_info", "decode_error",
        "selected_compression_quality", "candidate_quality_cache", "compression_candidates",
        "selected_candidate", "pending_selected_candidate",
    ]:
        st.session_state.pop(key, None)


def _store_method_payload(domain: str, method: str, payload: bytes, payload_path: str, meta: Dict[str, Any], preview_path: str | None = None) -> None:
    kind = str(meta.get("kind") or meta.get("app_layer") or f"{domain}_payload")
    st.session_state.update({
        "stored_bytes": bytes(payload or b""),
        "stored_file_path": payload_path,
        "storage_method": method,
        "storage_kind": kind,
        "storage_meta": {**meta, "kind": kind, "domain": domain, "method_display": method, "preview_path": preview_path or meta.get("preview_path")},
        "compression_candidates": [],
        "selected_candidate": None,
        "pending_selected_candidate": None,
    })
    _clear_after_method_selection()
    # restore selected storage keys because _clear_after_method_selection removes downstream only
    st.session_state.update({
        "stored_bytes": bytes(payload or b""),
        "stored_file_path": payload_path,
        "storage_method": method,
        "storage_kind": kind,
        "storage_meta": {**meta, "kind": kind, "domain": domain, "method_display": method, "preview_path": preview_path or meta.get("preview_path")},
        "compression_candidates": [],
        "selected_candidate": None,
        "pending_selected_candidate": None,
    })


def _run_selected_domain_method(domain: str, method: str, path: str, data: bytes) -> Tuple[bytes, str, Dict[str, Any], str | None]:
    """Run exactly one selected method. This replaces all-method benchmarking."""
    domain = str(domain or "").lower()
    method = str(method or "")
    raw = bytes(data or b"")

    if domain == "image":
        if method in {"RGB pixels", "Grayscale pixels", "Binary pixels"}:
            representation = {
                "RGB pixels": "RGB pixels",
                "Grayscale pixels": "Grayscale pixels",
                "Binary pixels": "Binary image pixels",
            }[method]
            threshold = int(st.session_state.get("image_binary_threshold", 128))
            payload, meta, preview_png = _image_pixels_to_bytes(raw, representation, threshold=threshold)
            preview_path = str(_selected_payload_dir(domain) / f"{_safe_method_slug(method)}_preview.png")
            Path(preview_path).write_bytes(preview_png)
            meta.update({"method_display": method, "selected_method_family": "raw_pixels", "preview_path": preview_path})
            payload_path = _write_selected_payload(domain, method, payload, ".raw")
            return payload, payload_path, meta, preview_path

        try:
            import robust_image_pipeline as rip  # type: ignore
        except Exception as exc:
            raise RuntimeError("robust_image_pipeline.py is required for this image method.") from exc
        level = str(st.session_state.get("image_compression_level", "Balanced"))
        pixel_representation = str(st.session_state.get("image_pixel_representation", "Grayscale (8 bits/pixel)"))
        threshold = int(st.session_state.get("image_binary_threshold", 128))
        payload, meta, preview_path = rip.encode_image_to_payload(
            path,
            method,
            _selected_payload_dir(domain),
            compression_level=level,
            pixel_representation=pixel_representation,
            threshold=threshold,
        )
        meta.update({"kind": "robust_image_payload", "method_display": method, "preview_path": preview_path})
        payload_path = _write_selected_payload(domain, method, payload, ".imgpayload")
        return payload, payload_path, meta, preview_path

    if domain == "text":
        try:
            import text_dna_unified_panel as txtpanel  # type: ignore
        except Exception as exc:
            raise RuntimeError("text_dna_unified_panel.py is required for text token methods.") from exc
        text = txtpanel._decode_text_bytes(raw)
        if method == "Dense fixed-vocab token":
            package = txtpanel._make_text_method_package(text, "C_DENSE_TOKEN", 8192, 4096, 16)
        else:
            tokenizer = str(st.session_state.get("text_tokenizer_name", "bert-base-uncased"))
            error_detection = str(st.session_state.get("text_error_detection", "token_crc8"))
            package = txtpanel._make_text_method_package(
                text,
                "D_SPARSE_SEMANTIC",
                8192,
                4096,
                16,
                fixed_tokenizer_name=tokenizer,
                fixed_error_detection=error_detection,
                fixed_block_tokens=16,
            )
        payload = bytes(package.get("payload_bytes", b""))
        meta = txtpanel._jsonable_package(package)
        meta.update({"kind": "text_token_payload", "method_display": method, "text_package": package})
        payload_path = _write_selected_payload(domain, method, payload, ".textpayload")
        return payload, payload_path, meta, None

    if domain == "audio":
        try:
            import audio_dna_tab as audiopanel  # type: ignore
        except Exception as exc:
            raise RuntimeError("audio_dna_tab.py is required for audio robust methods.") from exc
        payload, original_wav, meta, _original = audiopanel._payload_from_audio(raw, method)
        preview_path = str(_selected_payload_dir(domain) / f"{_safe_method_slug(method)}_source.wav")
        Path(preview_path).write_bytes(original_wav)
        meta.update({"kind": "audio_payload", "method_display": method, "preview_path": preview_path})
        payload_path = _write_selected_payload(domain, method, payload, ".audiopayload")
        return payload, payload_path, meta, preview_path

    if domain == "video":
        try:
            from video_dna_payload_codec import build_video_config, compress_video_to_payload  # type: ignore
        except Exception as exc:
            raise RuntimeError("video_dna_payload_codec.py is required for video robust methods.") from exc
        include_audio = bool(st.session_state.get("video_include_audio", True))
        process_full = bool(st.session_state.get("video_process_full", False))
        max_seconds = None if process_full else float(st.session_state.get("video_max_seconds", 5.0))
        target_fps_value = float(st.session_state.get("video_target_fps", 0.0) or 0.0)
        protection = str(st.session_state.get("video_protection_mode", "None"))
        out_dir = _selected_payload_dir(domain) / _safe_method_slug(method)
        cfg = build_video_config(
            output_dir=str(out_dir),
            mode=method,
            include_audio=include_audio,
            process_full_video=process_full,
            max_seconds=max_seconds,
            target_fps=(target_fps_value if target_fps_value > 0 else None),
            repair_strength="Balanced",
            protection_mode=protection,
        )
        result = compress_video_to_payload(path, cfg)
        payload = bytes(result.get("payload_bytes", b""))
        manifest = result.get("manifest", {}) or {}
        preview_path = str(result.get("original_preview", "") or "") or None
        meta = {"kind": "video_payload", "method_display": method, "manifest": manifest, "compression_result": {k: v for k, v in result.items() if k != "payload_bytes"}, "preview_path": preview_path}
        payload_path = _write_selected_payload(domain, method, payload, ".videopayload")
        return payload, payload_path, meta, preview_path

    raise ValueError(f"Unsupported domain: {domain}")


def _apply_panel_typography() -> None:
    if st.session_state.get("_panel_typography_applied"):
        return
    st.session_state["_panel_typography_applied"] = True
    st.markdown(
        """
<style>
html, body, [class*="css"] {
  font-family: Inter, "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stDataFrame, .stTable, [data-testid="stMetric"] {
  font-family: Inter, "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.84rem;
  letter-spacing: 0;
}
[data-testid="stMetricValue"] {
  font-size: 1.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _preview_seq(seq: str, n: int = 80) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _candidate_file(cand: CompressionCandidate) -> str:
    active_domain = st.session_state.get("active_domain", "shared") or "shared"
    out_dir = WORK_ROOT / "selected_compression" / active_domain
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_method = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in cand.method)
    path = out_dir / f"selected_{safe_method}{cand.ext or '.bin'}"
    path.write_bytes(cand.data)
    return str(path)


def _selected_candidate_path(cand: CompressionCandidate) -> str:
    path = _candidate_file(cand)
    st.session_state["stored_file_path"] = path
    return path



def _display_mapping(mapping: str) -> str:
    return display_mapping(mapping)


def _decode_source() -> Tuple[str, str]:
    """Return (label, dna_text) for the currently selected decode source."""
    return "Original encoded DNA", st.session_state.get("dna", "")


def _dna_from_uploaded_strand_csv(data: bytes) -> Tuple[str, int, str]:
    text = bytes(data or b"").decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader]
    if not rows:
        return "", 0, "CSV has no strand rows."

    def row_no(row: Dict[str, Any]) -> int:
        try:
            return int(str(row.get("No.", row.get("No", "0")) or "0"))
        except Exception:
            return 0

    rows.sort(key=row_no)
    # Prefer the noisy/error payload if the file was generated by Advanced strand-noise simulation.
    if "Error payload" in rows[0]:
        dna = clean_dna("".join(str(row.get("Error payload", "") or row.get("Payload", "")) for row in rows))
        return dna, len(rows), "Loaded Error payload column from error-strand CSV."
    if "Payload" in rows[0]:
        dna = clean_dna("".join(str(row.get("Payload", "")) for row in rows))
        return dna, len(rows), "Loaded Payload column from prepared-strand CSV."
    if "Error full strand" in rows[0]:
        dna = clean_dna("".join(str(row.get("Error full strand", "") or row.get("Full strand", "")) for row in rows))
        return dna, len(rows), "Loaded Error full strand column; Payload/Error payload is preferred when available."
    if "Full strand" in rows[0]:
        dna = clean_dna("".join(str(row.get("Full strand", "")) for row in rows))
        return dna, len(rows), "Loaded Full strand column; Payload column is preferred for exact prepared-strand decode."
    return "", len(rows), "CSV must include Payload, Error payload, Full strand, or Error full strand."


def _bytes_to_bit_text(data: bytes, max_bytes: int = MAX_BINARY_PREVIEW_BYTES) -> str:
    raw = bytes(data or b"")
    shown = raw[:max_bytes]
    bits = bytes_to_bitstring(shown)
    if len(raw) > max_bytes:
        bits += f"\n\n... preview only: showing first {max_bytes:,} of {len(raw):,} bytes."
    return bits


def _bytes_to_full_bit_text(data: bytes) -> str:
    return bytes_to_bitstring(bytes(data or b""))


def _bits_preview_text(bits: str, max_chars: int = MAX_BITS_PREVIEW_CHARS) -> str:
    text = str(bits or "")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... preview only: showing first {max_chars:,} of {len(text):,} bits."
    return text


def _download_text_button(label: str, text: str, file_name: str) -> None:
    st.download_button(label, data=text.encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True)


def _download_full_binary_button(label: str, data: bytes, file_name: str) -> None:
    st.download_button(
        label,
        data=_bytes_to_full_bit_text(data).encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
    )


def _pipeline_file_metric_rows(path: str, data: bytes, *, compressed: bool = False) -> List[Dict[str, Any]]:
    rows = _file_info_rows(path, data)
    out: List[Dict[str, Any]] = []
    for row in rows:
        metric = str(row.get("Metric", ""))
        if metric in {"File name", "Container"}:
            continue
        if metric == "Size" and compressed:
            out.append({"Metric": "Compressed data", "Value": row.get("Value", "")})
        else:
            out.append(row)
    return out


def _rows_to_properties(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "Property": row.get("Metric", row.get("Property", "")),
            "Value": row.get("Value", ""),
        })
    return out


def _render_property_table(rows: List[Dict[str, Any]], height: int | None = None) -> None:
    if not rows:
        return
    kwargs = {"use_container_width": True, "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(pd.DataFrame(_rows_to_properties(rows)), **kwargs)


def _render_stats_table(rows: List[Dict[str, Any]], height: int | None = None) -> None:
    if not rows:
        return
    kwargs = {"use_container_width": True, "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(pd.DataFrame(rows), **kwargs)


def _format_chart_label(value: float, metric: str) -> str:
    if metric in {"Compression ratio"}:
        return f"{value:.2f}x"
    if metric in {"PSNR", "Keyframe PSNR", "SNR", "Signal-to-Noise Ratio"}:
        return f"{value:.2f} dB"
    if metric in {"SSIM", "Keyframe SSIM", "Waveform correlation", "Waveform Correlation", "Spectrogram similarity"}:
        return f"{value:.4f}"
    if metric in {"MAE", "Mean absolute error", "Duration difference"}:
        return f"{value:.3f}"
    if metric in {"Text accuracy"}:
        return f"{value:.2f}%"
    return f"{value:.2f}"


def _chart_scale_domain(metric: str) -> List[float] | None:
    if metric in {"SSIM", "Keyframe SSIM", "Waveform correlation", "Waveform Correlation", "Spectrogram similarity"}:
        return [0, 1]
    if metric in {"Text accuracy"}:
        return [0, 100]
    return None


def _chart_lollipop(
    rows: List[Dict[str, Any]],
    label_col: str,
    value_col: str,
    *,
    height: int | None = None,
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if label_col not in df or value_col not in df:
        return
    df = df.copy()
    df["Value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["Value"])
    if df.empty:
        return
    df["Order"] = range(len(df))
    df["Zero"] = 0.0
    df["Label"] = df["Value"].apply(lambda v: _format_chart_label(float(v), value_col))
    chart_height = height or max(120, min(260, 34 * len(df) + 24))
    domain = _chart_scale_domain(value_col)

    x_scale: Dict[str, Any] = {"zero": True}
    if domain:
        x_scale["domain"] = domain
    spec = {
        "height": chart_height,
        "encoding": {
            "y": {
                "field": label_col,
                "type": "nominal",
                "sort": {"field": "Order", "order": "ascending"},
                "axis": {"labelLimit": 180, "labelFontSize": 11, "title": None},
            },
            "x": {
                "field": "Value",
                "type": "quantitative",
                "title": value_col,
                "scale": x_scale,
                "axis": {"grid": True, "labelFontSize": 10, "titleFontSize": 11},
            },
        },
        "layer": [
            {
                "mark": {"type": "rule", "strokeWidth": 2, "color": "#94a3b8"},
                "encoding": {"x": {"field": "Zero", "type": "quantitative"}, "x2": {"field": "Value"}},
            },
            {"mark": {"type": "point", "filled": True, "size": 72, "color": "#2563eb", "opacity": 0.92}},
            {
                "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 8, "fontSize": 11, "color": "#0f172a"},
                "encoding": {"text": {"field": "Label", "type": "nominal"}},
            },
        ],
    }
    st.vega_lite_chart(df, spec, use_container_width=True)


def _chart_bar(rows: List[Dict[str, Any]], label_col: str, value_col: str) -> None:
    _chart_lollipop(rows, label_col, value_col)


def _format_method_name(method: str) -> str:
    text = str(method or "No compression").replace("_", " ").replace("-", " ")
    tokens = []
    for token in text.split():
        lower = token.lower()
        if lower in {"webp", "png", "jpeg", "jpg", "avif", "ogg", "opus", "mp4", "h264", "h265", "vp9", "gzip", "zlib", "zip", "bz2", "xz"}:
            tokens.append(lower.upper())
        elif lower.startswith("q") and lower[1:].isdigit():
            tokens.append(f"Q{lower[1:]}")
        elif lower.startswith("c") and lower[1:].isdigit():
            tokens.append(f"C{lower[1:]}")
        elif lower.startswith("lvl") and lower[3:].isdigit():
            tokens.append(f"LVL{lower[3:]}")
        elif lower.endswith("k") and lower[:-1].isdigit():
            tokens.append(lower.upper())
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


def _quality_columns_for_domain(domain: str) -> List[str]:
    if domain == "image":
        return ["PSNR", "SSIM", "MAE"]
    if domain == "audio":
        return ["SNR", "Waveform correlation", "Spectrogram similarity"]
    if domain == "video":
        return ["Keyframe PSNR", "Keyframe SSIM", "Duration difference", "Resolution"]
    if domain == "text":
        return ["Text accuracy", "Exact match", "Length delta"]
    return ["Exact match"]


def _quality_rows_to_candidate_columns(rows: List[Dict[str, Any]], domain: str | None = None) -> Dict[str, str]:
    aliases = {
        "PSNR": "PSNR",
        "SSIM": "SSIM",
        "Mean absolute error": "MAE",
        "Signal-to-Noise Ratio": "SNR",
        "Waveform Correlation": "Waveform correlation",
        "Spectrogram similarity": "Spectrogram similarity",
        "Keyframe PSNR": "Keyframe PSNR",
        "Keyframe SSIM": "Keyframe SSIM",
        "Text accuracy": "Text accuracy",
        "Exact text match": "Exact match",
        "Length delta": "Length delta",
        "Duration difference": "Duration difference",
        "Resolution": "Resolution",
    }
    wanted = _quality_columns_for_domain(domain or "") if domain else list(dict.fromkeys(aliases.values()))
    out = {name: "—" for name in wanted}
    for row in rows:
        metric = str(row.get("Metric", ""))
        if metric in aliases and aliases[metric] in out:
            out[aliases[metric]] = str(row.get("Value", "—"))
    return out


def _candidate_quality_cache_key(input_path: str | None, input_bytes: bytes, cand: CompressionCandidate) -> str:
    raw = bytes(input_bytes or b"")
    digest = hashlib.sha256(raw[:8192] + cand.data[:8192]).hexdigest()
    return "|".join([
        str(input_path or ""),
        str(len(raw)),
        str(cand.method),
        str(cand.size_bytes),
        str(cand.ext),
        digest,
    ])


def _candidate_quality_rows(
    input_path: str | None,
    input_bytes: bytes,
    cand: CompressionCandidate,
) -> List[Dict[str, Any]]:
    cache = st.session_state.setdefault("candidate_quality_cache", {})
    key = _candidate_quality_cache_key(input_path, input_bytes, cand)
    if key not in cache:
        cache[key] = _compression_quality_rows(input_path, input_bytes, _candidate_file(cand), cand.data)
    return cache.get(key, [])


def _candidate_quality_table_rows(
    input_path: str | None,
    input_bytes: bytes,
    candidates: List[CompressionCandidate],
) -> List[Dict[str, Any]]:
    domain = get_domain(input_path or "", bytes(input_bytes or b""))
    rows: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c.rank):
        q_rows = _candidate_quality_rows(input_path, input_bytes, cand)
        row = {
            "Rank": cand.rank,
            "Method": _format_method_name(cand.method),
            "Output": cand.ext or ".bin",
            "Compressed data": fmt_bytes(cand.size_bytes),
            "Compression ratio": f"{cand.compression_ratio:.2f}x",
            "Size reduction": f"{cand.saving_pct:.2f}%",
        }
        row.update(_quality_rows_to_candidate_columns(q_rows, domain))
        row["Estimated DNA"] = f"{cand.estimated_dna_nt:,} nt"
        rows.append(row)
    return rows


def _parse_chart_value(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text == "—" or text == "Unavailable":
        return None
    if text.startswith("∞"):
        return 100.0
    if text == "Yes":
        return 100.0
    if text == "No":
        return 0.0
    cleaned = (
        text.replace("dB", "")
        .replace("%", "")
        .replace("s", "")
        .replace("x", "")
        .replace("×", "")
        .strip()
    )
    if "->" in cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _render_quality_metric_charts(rows: List[Dict[str, Any]], metrics: List[str]) -> None:
    if not rows:
        return
    chart_metrics = [m for m in metrics if m != "Resolution"]
    for i in range(0, len(chart_metrics), 2):
        cols = st.columns(2)
        for col, metric in zip(cols, chart_metrics[i:i + 2]):
            chart_rows = []
            for row in rows:
                value = _parse_chart_value(row.get(metric))
                if value is not None:
                    chart_rows.append({"Method": row.get("Method", ""), metric: value})
            if chart_rows:
                with col:
                    st.markdown(f"##### {metric}")
                    _chart_lollipop(chart_rows, "Method", metric, height=max(120, min(210, 32 * len(chart_rows) + 20)))


def _quality_score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []
    for row in rows:
        metric = str(row.get("Metric", ""))
        value = str(row.get("Value", ""))
        try:
            if metric in {"SSIM", "Keyframe SSIM", "Waveform Correlation", "Spectrogram similarity"}:
                scores.append({"Metric": metric, "Score": max(0.0, min(100.0, float(value) * 100.0))})
            elif metric in {"Text accuracy"}:
                scores.append({"Metric": metric, "Score": max(0.0, min(100.0, float(value.rstrip("%"))))})
            elif metric == "Exact text match":
                scores.append({"Metric": metric, "Score": 100.0 if value == "Yes" else 0.0})
        except Exception:
            pass
    return scores


def _quality_numeric_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    numeric_metrics = {
        "PSNR",
        "SSIM",
        "Mean absolute error",
        "Signal-to-Noise Ratio",
        "Waveform Correlation",
        "Spectrogram similarity",
        "Keyframe PSNR",
        "Keyframe SSIM",
        "Duration difference",
    }
    for row in rows:
        metric = str(row.get("Metric", ""))
        value = str(row.get("Value", ""))
        if metric not in numeric_metrics or value == "Unavailable":
            continue
        if value.startswith("∞"):
            parsed = 100.0
        else:
            cleaned = value.replace("dB", "").replace("s", "").strip()
            try:
                parsed = float(cleaned)
            except Exception:
                continue
        out.append({"Metric": metric, "Value": parsed})
    return out


def _candidate_list_label(cand: CompressionCandidate) -> str:
    return (
        f"{cand.rank}. {_format_method_name(cand.method)} | {cand.ext or '.bin'} | "
        f"{fmt_bytes(cand.size_bytes)} | {cand.compression_ratio:.2f}x | "
        f"{cand.estimated_dna_nt:,} nt"
    )


def _set_selected_candidate(cand: CompressionCandidate) -> None:
    st.session_state.update({
        "selected_candidate": cand,
        "stored_bytes": cand.data,
        "stored_file_path": _selected_candidate_path(cand),
        "storage_method": cand.method,
        "storage_kind": cand.kind,
        "storage_meta": {"kind": "compressed_file", "method": cand.method, "file_kind": cand.kind, "ext": cand.ext},
    })
    for key in ["dna", "bits", "codec_meta", "strand_rows", "decoded_data", "restored_info", "decode_error"]:
        st.session_state.pop(key, None)


def _render_candidate_list(candidates: List[CompressionCandidate], selected: CompressionCandidate | None) -> None:
    """Show compression candidates and let the user explicitly choose one.

    The benchmark only produces candidates.  It does not automatically choose the
    smallest output for DNA encoding.  Panel 3 becomes available only after the
    user presses "Use selected method for DNA encoding".
    """
    if not candidates:
        return

    st.markdown("##### 🧾 Compression candidates")
    current_idx = 0
    if selected is not None:
        for i, cand in enumerate(candidates):
            if cand.method == selected.method and cand.size_bytes == selected.size_bytes and cand.ext == selected.ext:
                current_idx = i
                break

    selected_idx = st.selectbox(
        "Choose the compression output to continue to DNA encoding",
        list(range(len(candidates))),
        index=current_idx,
        format_func=lambda i: _candidate_list_label(candidates[int(i)]),
        key="compression_choice",
    )
    cand = candidates[int(selected_idx)]
    st.session_state["pending_selected_candidate"] = cand

    b1, b2 = st.columns([0.55, 0.45])
    with b1:
        if st.button("Use selected method for DNA encoding", key="use_selected_compression", type="primary", use_container_width=True):
            _set_selected_candidate(cand)
            st.success(f"Selected method: {_format_method_name(cand.method)}. Panel 3 can now encode this output to DNA.")
            st.rerun()
    with b2:
        if selected is None:
            st.info("No method has been selected yet.")
        else:
            st.success(f"Current method: {_format_method_name(selected.method)}")

    input_path = st.session_state.get("input_path")
    input_bytes = st.session_state.get("input_bytes", b"") or b""
    table_rows = _candidate_quality_table_rows(input_path, input_bytes, candidates)
    for row, c in zip(table_rows, sorted(candidates, key=lambda item: item.rank)):
        row["Selected"] = "Current" if selected is not None and c.method == selected.method and c.size_bytes == selected.size_bytes and c.ext == selected.ext else ("Preview" if c.method == cand.method and c.size_bytes == cand.size_bytes and c.ext == cand.ext else "")
    ordered = []
    for row in table_rows:
        selected_flag = row.pop("Selected", "")
        ordered.append({"Selected": selected_flag, **row})
    _render_stats_table(ordered)

def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"


def _previewable_payload_path(path: str | None, data: bytes | None, title: str) -> str | None:
    if not path:
        return None
    kind = magic_dict(bytes(data or b"")).get("kind", "unknown")
    if kind in {"zip", "gzip", "xz", "bz2", "zlib"}:
        inner = _write_preview_inner(path, kind, title)
        return inner or path
    return path


def _file_info_rows(path: str | None, data: bytes | None) -> List[Dict[str, Any]]:
    raw = bytes(data or b"")
    ext = os.path.splitext(path or "")[1].lower() or magic_dict(raw).get("ext", ".bin")
    domain = get_domain(path or "", raw) if raw else "unknown"
    m = magic_dict(raw)
    rows: List[Dict[str, Any]] = [
        {"Metric": "File name", "Value": os.path.basename(path or "—")},
        {"Metric": "Extension", "Value": ext or "—"},
        {"Metric": "Type", "Value": domain},
        {"Metric": "Size", "Value": fmt_bytes(len(raw))},
    ]
    if m.get("kind") and m.get("kind") != "unknown":
        rows.append({"Metric": "Container", "Value": m.get("kind")})

    if domain == "image" and Image is not None and raw:
        try:
            img = Image.open(io.BytesIO(raw))
            rows.append({"Metric": "Image size", "Value": f"{img.width} x {img.height} px"})
        except Exception:
            pass

    if domain in {"audio", "video"} and path:
        info = _run_ffprobe(path)
        duration = _duration_seconds(info)
        if duration is not None:
            rows.append({"Metric": "Duration", "Value": _format_duration(duration)})
        if domain == "video":
            stream = _first_stream(info, "video")
            if stream:
                rows.extend([
                    {"Metric": "Resolution", "Value": f"{stream.get('width', '?')} x {stream.get('height', '?')} px"},
                    {"Metric": "FPS", "Value": _fps_value(stream)},
                ])
    return rows


def _render_info_table(rows: List[Dict[str, Any]]) -> None:
    _render_property_table(rows)


def _render_metric_rows(rows: List[Dict[str, Any]], columns: int = 4) -> None:
    if not rows:
        return
    width = max(1, min(int(columns), 4))
    for start in range(0, len(rows), width):
        chunk = rows[start:start + width]
        cols = st.columns(len(chunk))
        for col, row in zip(cols, chunk):
            metric = str(row.get("Metric", ""))
            value = str(row.get("Value", ""))
            delta = row.get("Delta")
            col.metric(metric, value, delta=delta if delta else None)


def _render_workflow_overview() -> None:
    steps = [
        ("1", "Upload", "Input preview"),
        ("2", "Compress", "Stored bytes"),
        ("3", "DNA", "Mapping"),
        ("4", "Strands", "Design"),
        ("5", "Decode", "Restored file"),
        ("6", "Validate", "Compare"),
    ]
    html = ['<div class="workflow-strip">']
    for no, title, desc in steps:
        html.append(
            f'<div class="workflow-item"><div class="workflow-no">{no}</div>'
            f'<div class="workflow-title">{title}</div><div class="workflow-desc">{desc}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _clear_downstream_from_storage() -> None:
    for key in [
        "compression_candidates", "selected_candidate", "stored_bytes", "stored_file_path",
        "candidate_quality_cache",
        "storage_method", "storage_kind", "storage_meta",
        "dna", "bits", "codec_meta", "strand_rows",
        "decoded_data", "decoded_raw_pixels", "decoded_bits", "decoded_meta", "decoded_magic", "decoded_valid",
        "decoded_note", "raw_restore_info", "restored_info", "decode_error",
    ]:
        st.session_state.pop(key, None)


def _validate_and_write(data: bytes, preferred: str = "restored") -> Dict[str, Any]:
    active_domain = st.session_state.get("active_domain", "shared") or "shared"
    out_dir = WORK_ROOT / "decode_output" / active_domain
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_restored_file(data, str(out_dir), preferred_name=preferred)


def _is_uploaded_image(path: str, data: bytes) -> bool:
    """Return True when the uploaded file can be handled as an image."""
    if Image is None or not data:
        return False
    try:
        domain = get_domain(path, data)
        if domain == "image":
            return True
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _image_pixels_to_bytes(data: bytes, representation: str, threshold: int = 128) -> Tuple[bytes, Dict[str, Any], bytes]:
    """
    Convert an uploaded image to raw pixel bytes for no-compression storage.

    Returns: (pixel_bytes, metadata, preview_png_bytes).
    The bytes are not an image container; width/height/mode metadata are required
    to reconstruct them later.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image pixel conversion.")
    img = Image.open(io.BytesIO(data))
    if representation == "RGB pixels":
        out_img = img.convert("RGB")
        channels = 3
        raw_mode = "RGB"
        rep_label = "RGB pixels"
    elif representation == "Grayscale pixels":
        out_img = img.convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Grayscale pixels"
    elif representation == "Binary image pixels":
        gray = img.convert("L")
        out_img = gray.point(lambda p: 255 if p >= int(threshold) else 0).convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Binary image pixels"
    else:
        raise ValueError(f"Unknown image representation: {representation}")

    raw = out_img.tobytes()
    png = io.BytesIO()
    out_img.save(png, format="PNG")
    meta = {
        "kind": "raw_image_pixels",
        "representation": rep_label,
        "raw_mode": raw_mode,
        "width": int(out_img.width),
        "height": int(out_img.height),
        "channels": int(channels),
        "expected_bytes": int(len(raw)),
        "threshold": int(threshold),
        "output_ext": ".png",
    }
    return raw, meta, png.getvalue()


def _raw_image_bytes_to_png(data: bytes, meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Build a PNG preview/output from decoded raw image pixel bytes."""
    if Image is None:
        raise RuntimeError("Pillow is required to restore raw image pixels.")
    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))
    mode = str(meta.get("raw_mode", "L"))
    expected = int(meta.get("expected_bytes", width * height * (3 if mode == "RGB" else 1)))
    raw = bytes(data or b"")
    note = "Exact raw-pixel length."
    if len(raw) < expected:
        raw = raw + bytes(expected - len(raw))
        note = f"Decoded bytes were shorter than expected; padded {expected - len(data or b'')} bytes."
    elif len(raw) > expected:
        raw = raw[:expected]
        note = f"Decoded bytes were longer than expected; truncated {len(data or b'') - expected} bytes."
    img = Image.frombytes(mode, (width, height), raw)
    png = io.BytesIO()
    img.save(png, format="PNG")
    return png.getvalue(), {"note": note, "width": width, "height": height, "mode": mode, "expected_bytes": expected}


def _byte_accuracy_bytes(a: bytes, b: bytes) -> float:
    """Position-wise byte accuracy with length difference counted as errors."""
    a = bytes(a or b"")
    b = bytes(b or b"")
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / denom


def _bit_error_rate_bytes(a: bytes, b: bytes) -> float:
    a = bytes(a or b"")
    b = bytes(b or b"")
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0

    min_len = min(len(a), len(b))
    error_bits = 0
    for x, y in zip(a[:min_len], b[:min_len]):
        error_bits += (x ^ y).bit_count()
    error_bits += abs(len(a) - len(b)) * 8
    return error_bits / max(1, max_len * 8)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _sci(value: float) -> str:
    return f"{float(value):.2e}"


def _validation_row(stage: str, metric: str, value: Any, meaning: str | None = None, delta: str | None = None) -> Dict[str, Any]:
    row = {
        "Stage": stage,
        "Metric": metric,
        "Value": value,
    }
    if delta:
        row["Delta"] = delta
    return row


def _render_validation_metric_cards(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("Stage", "Validation")), []).append(row)

    for stage, stage_rows in grouped.items():
        st.markdown(f"#### {stage}")
        _render_metric_rows(stage_rows, columns=4)


def _file_cache_signature(path: str | None) -> Tuple[int, int]:
    if not path or not os.path.exists(path):
        return (0, 0)
    try:
        stat = os.stat(path)
        return (int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return (0, 0)


@st.cache_data(show_spinner=False)
def _run_ffprobe_cached(path: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if p.returncode != 0 or not p.stdout.strip():
            return {}
        return json.loads(p.stdout)
    except Exception:
        return {}


def _run_ffprobe(path: str | None) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    mtime_ns, size = _file_cache_signature(path)
    return _run_ffprobe_cached(path, mtime_ns, size)


def _duration_seconds(info: Dict[str, Any]) -> float | None:
    try:
        value = info.get("format", {}).get("duration")
        return float(value) if value is not None else None
    except Exception:
        return None


def _first_stream(info: Dict[str, Any], codec_type: str) -> Dict[str, Any]:
    for stream in info.get("streams", []) or []:
        if stream.get("codec_type") == codec_type:
            return stream
    return {}


def _fps_value(stream: Dict[str, Any]) -> str:
    raw = str(stream.get("avg_frame_rate") or "")
    if "/" not in raw:
        return raw or "unknown"
    try:
        num, den = raw.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return "unknown"
        return f"{float(num) / den_f:.2f}"
    except Exception:
        return "unknown"


@st.cache_data(show_spinner=False)
def _decode_audio_mono_cached(path: str, mtime_ns: int, size: int, sample_rate: int, seconds: int):
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-t",
                str(int(seconds)),
                "-ac",
                "1",
                "-ar",
                str(int(sample_rate)),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        audio = np.frombuffer(p.stdout, dtype=np.float32)
        return audio if audio.size else None
    except Exception:
        return None


def _decode_audio_mono(path: str | None, *, sample_rate: int = 16000, seconds: int = 30):
    if np is None or not path or not os.path.exists(path):
        return None
    mtime_ns, size = _file_cache_signature(path)
    return _decode_audio_mono_cached(path, mtime_ns, size, int(sample_rate), int(seconds))


def _spectrogram_similarity(path_a: str | None, path_b: str | None) -> float | None:
    if np is None:
        return None
    a = _decode_audio_mono(path_a)
    b = _decode_audio_mono(path_b)
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    if n < 1024:
        return None
    a = a[:n]
    b = b[:n]
    n_fft = 512
    hop = 256
    window = np.hanning(n_fft).astype(np.float32)

    def spec(x):
        frames = []
        for start in range(0, max(1, len(x) - n_fft + 1), hop):
            frame = x[start:start + n_fft]
            if frame.size < n_fft:
                break
            frames.append(np.log1p(np.abs(np.fft.rfft(frame * window))))
        if not frames:
            return None
        return np.asarray(frames, dtype=np.float32)

    sa = spec(a)
    sb = spec(b)
    if sa is None or sb is None:
        return None
    m = min(sa.shape[0], sb.shape[0])
    va = sa[:m].reshape(-1)
    vb = sb[:m].reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return None
    return max(0.0, min(1.0, float(np.dot(va, vb) / denom)))


def _audio_waveform_metrics(path_a: str | None, path_b: str | None) -> Dict[str, float]:
    if np is None:
        return {}
    a = _decode_audio_mono(path_a)
    b = _decode_audio_mono(path_b)
    if a is None or b is None:
        return {}
    n = min(a.size, b.size)
    if n < 1024:
        return {}
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    noise = a - b
    signal_power = float(np.mean(a * a))
    noise_power = float(np.mean(noise * noise))
    if noise_power <= 1e-18:
        snr = float("inf")
    elif signal_power <= 1e-18:
        snr = 0.0
    else:
        snr = float(10.0 * np.log10(signal_power / noise_power))

    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        corr = 1.0 if noise_power <= 1e-18 else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"snr": snr, "correlation": max(-1.0, min(1.0, corr))}


@st.cache_data(show_spinner=False)
def _extract_video_frame_cached(path: str, mtime_ns: int, size: int, at_millis: int):
    try:
        at_seconds = max(0.0, float(at_millis) / 1000.0)
        p = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, float(at_seconds)):.3f}",
                "-i",
                path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        return Image.open(io.BytesIO(p.stdout)).convert("RGB")
    except Exception:
        return None


def _extract_video_frame(path: str | None, at_seconds: float = 1.0):
    if Image is None or not path or not os.path.exists(path):
        return None
    mtime_ns, size = _file_cache_signature(path)
    at_millis = int(max(0.0, float(at_seconds)) * 1000)
    return _extract_video_frame_cached(path, mtime_ns, size, at_millis)


def _image_array_metrics(img_a, img_b) -> Dict[str, float]:
    if np is None or img_a is None or img_b is None:
        return {}
    try:
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)
        arr_a = np.asarray(img_a).astype("float32")
        arr_b = np.asarray(img_b).astype("float32")
        mse = float(np.mean((arr_a - arr_b) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        vals = []
        x = arr_a.reshape(-1, 3)
        y = arr_b.reshape(-1, 3)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        for ch in range(3):
            xx = x[:, ch]
            yy = y[:, ch]
            mux, muy = float(xx.mean()), float(yy.mean())
            vx, vy = float(xx.var()), float(yy.var())
            cov = float(((xx - mux) * (yy - muy)).mean())
            vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
        return {"psnr": psnr, "ssim": float(np.mean(vals))}
    except Exception:
        return {}


def _keyframe_metrics(path_a: str | None, path_b: str | None, duration: float | None) -> Dict[str, float]:
    at = 1.0
    if duration is not None and duration > 2:
        at = min(duration / 2.0, 10.0)
    frame_a = _extract_video_frame(path_a, at)
    frame_b = _extract_video_frame(path_b, at)
    return _image_array_metrics(frame_a, frame_b)


def _media_quality_rows(stage: str, domain: str, original_path: str | None, restored_path: str | None) -> List[Dict[str, Any]]:
    original = _run_ffprobe(original_path)
    restored = _run_ffprobe(restored_path)
    if not original or not restored:
        return [_validation_row(stage, "Media comparison", "Unavailable")]

    rows: List[Dict[str, Any]] = []
    orig_duration = _duration_seconds(original)
    new_duration = _duration_seconds(restored)
    if orig_duration is not None and new_duration is not None:
        diff = abs(orig_duration - new_duration)
        rows.append(_validation_row(
            stage,
            "Duration difference",
            f"{diff:.3f} s",
        ))

    if domain == "audio":
        spec_sim = _spectrogram_similarity(original_path, restored_path)
        wave_metrics = _audio_waveform_metrics(original_path, restored_path)
        snr = wave_metrics.get("snr")
        corr = wave_metrics.get("correlation")
        if snr is not None:
            if snr == float("inf"):
                rows.append(_validation_row(stage, "Signal-to-Noise Ratio", "∞ dB", delta="Perfect"))
            else:
                rows.append(_validation_row(stage, "Signal-to-Noise Ratio", f"{snr:.2f} dB"))
        if corr is not None:
            rows.append(_validation_row(stage, "Waveform Correlation", f"{corr:.4f}", delta="Perfect" if corr >= 0.9999 else None))
        rows.append(_validation_row(stage, "Spectrogram similarity", f"{spec_sim:.4f}" if spec_sim is not None else "Unavailable"))
    elif domain == "video":
        ov = _first_stream(original, "video")
        rv = _first_stream(restored, "video")
        orig_res = f"{ov.get('width', '?')}x{ov.get('height', '?')}"
        new_res = f"{rv.get('width', '?')}x{rv.get('height', '?')}"
        kmetrics = _keyframe_metrics(original_path, restored_path, orig_duration)
        rows.extend([
            _validation_row(stage, "Resolution", f"{orig_res} -> {new_res}"),
            _validation_row(stage, "Keyframe PSNR", f"{kmetrics['psnr']:.2f} dB" if "psnr" in kmetrics else "Unavailable"),
            _validation_row(stage, "Keyframe SSIM", f"{kmetrics['ssim']:.4f}" if "ssim" in kmetrics else "Unavailable"),
        ])
    return rows


def _path_if_exists(path: str | None) -> str | None:
    if path and os.path.exists(str(path)):
        return str(path)
    return None


def _internal_payload_extension(path: str | None) -> bool:
    return str(Path(path or "").suffix).lower() in {".imgpayload", ".raw", ".audpayload", ".vidpayload", ".payload"}


def _selected_payload_preview_path(stored_path: str | None, storage_meta: Dict[str, Any] | None = None) -> str | None:
    """Return a display artifact for an internal payload, never the payload itself.

    Files such as .imgpayload are intermediate DNA payloads and are not
    browser-previewable containers. The UI should display the PNG/audio/video
    preview generated by the original algorithm instead.
    """
    storage_meta = storage_meta or {}
    for key in ["preview_path", "image_preview_path", "selected_pixel_preview_path", "restored_preview_path"]:
        p = _path_if_exists(storage_meta.get(key))
        if p:
            return p
    if _internal_payload_extension(stored_path):
        return None
    return _path_if_exists(stored_path)


def _image_quality_reference_path(input_path: str | None, storage_meta: Dict[str, Any] | None = None) -> str | None:
    """Reference image for quality metrics.

    For image methods, quality should be measured against the Step-1 selected
    pixel representation when available, not against a hidden .imgpayload file.
    """
    storage_meta = storage_meta or {}
    for key in ["selected_pixel_preview_path", "input_preview_path"]:
        p = _path_if_exists(storage_meta.get(key))
        if p:
            return p
    return _path_if_exists(input_path)


def _image_quality_metric_rows(reference_path: str | None, preview_path: str | None, stage: str) -> List[Dict[str, Any]]:
    if not reference_path or not preview_path or Image is None:
        return []
    metrics = image_metrics(reference_path, preview_path)
    if not metrics.get("Validation"):
        return []
    return [
        _validation_row(stage, "PSNR", f"{float(metrics.get('psnr', 0)):.2f} dB"),
        _validation_row(stage, "SSIM", f"{float(metrics.get('ssim', 0)):.4f}"),
        _validation_row(stage, "MAE", f"{float(metrics.get('mae', 0)):.2f}"),
    ]


def _method_result_rows(
    input_path: str | None,
    input_bytes: bytes,
    stored_path: str | None,
    stored_bytes: bytes,
    storage_meta: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    storage_meta = storage_meta or {}
    stored = bytes(stored_bytes or b"")
    original = bytes(input_bytes or b"")
    if not stored:
        return []
    raw_pixel_bytes = int(storage_meta.get("raw_pixel_bytes") or storage_meta.get("expected_bytes") or 0)
    baseline = raw_pixel_bytes or len(original)
    preview_path = _selected_payload_preview_path(stored_path, storage_meta)
    rows: List[Dict[str, Any]] = [
        _validation_row("Payload", "Method", storage_meta.get("method_display", st.session_state.get("storage_method", "—"))),
        _validation_row("Payload", "Payload size", fmt_bytes(len(stored))),
        _validation_row("Payload", "Estimated DNA length", f"{len(stored) * 4:,} nt"),
    ]
    if baseline:
        rows.append(_validation_row("Payload", "Compression ratio", f"{baseline / max(1, len(stored)):.2f}×"))
    if storage_meta.get("pixel_representation"):
        rows.append(_validation_row("Payload", "Input representation", storage_meta.get("pixel_representation")))
    if preview_path:
        rows.append(_validation_row("Payload", "Preview", "Available"))
    else:
        rows.append(_validation_row("Payload", "Preview", "Internal payload only"))

    domain = str(storage_meta.get("domain") or get_domain(input_path or "", original)).lower()
    if domain == "image":
        ref = _image_quality_reference_path(input_path, storage_meta)
        rows.extend(_image_quality_metric_rows(ref, preview_path, "Image quality"))
    return rows


def _compression_quality_rows(
    input_path: str | None,
    input_bytes: bytes,
    stored_path: str | None,
    stored_bytes: bytes,
) -> List[Dict[str, Any]]:
    storage_meta = st.session_state.get("storage_meta", {}) or {}
    rows = _method_result_rows(input_path, input_bytes, stored_path, stored_bytes, storage_meta)

    # Keep the old generic media/text comparison path for non-internal payloads.
    original = bytes(input_bytes or b"")
    stored = bytes(stored_bytes or b"")
    if not input_path or not original or not stored_path or not stored or _internal_payload_extension(stored_path):
        return rows

    output_path = _previewable_payload_path(stored_path, stored, "compression_quality")
    if not output_path:
        return rows

    domain = get_domain(input_path, original)
    if domain == "text":
        sim = text_similarity(input_path, output_path)
        if sim.get("Validation"):
            rows.extend([
                _validation_row("Compression quality", "Text accuracy", _pct(float(sim.get("char_position_accuracy", 0)))),
                _validation_row("Compression quality", "Exact text match", "Yes" if sim.get("exact") else "No"),
                _validation_row("Compression quality", "Length delta", f"{int(sim.get('len_b', 0)) - int(sim.get('len_a', 0)):+,} chars"),
            ])
    elif domain in {"audio", "video"}:
        rows.extend(_media_quality_rows("Compression quality", domain, input_path, output_path))
    return rows


def _validation_rows(
    *,
    input_path: str | None,
    input_bytes: bytes,
    stored_file_path: str | None,
    stored_bytes: bytes,
    restored_preview_path: str | None,
    recovered_for_match: bytes,
    file_can_open: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stored = bytes(stored_bytes or b"")
    recovered = bytes(recovered_for_match or b"")
    original = bytes(input_bytes or b"")

    hash_match = bool(stored) and hashlib.sha256(stored).hexdigest() == hashlib.sha256(recovered).hexdigest()
    length_delta = len(recovered) - len(stored)

    rows.extend([
        _validation_row("DNA decode integrity", "Payload accuracy", _pct(_byte_accuracy_bytes(stored, recovered))),
        _validation_row("DNA decode integrity", "Bit error rate", _sci(_bit_error_rate_bytes(stored, recovered))),
        _validation_row("DNA decode integrity", "Length delta", f"{length_delta:+,} bytes"),
        _validation_row("DNA decode integrity", "Checksum", "Pass" if hash_match else "Fail"),
    ])

    return rows


def _compression_analysis_rows(
    input_path: str | None,
    input_bytes: bytes,
    stored_path: str | None,
    stored_bytes: bytes,
) -> List[Dict[str, Any]]:
    original = bytes(input_bytes or b"")
    stored = bytes(stored_bytes or b"")
    if not original or not stored:
        return []
    domain = get_domain(input_path or "", original)
    quality = _quality_rows_to_candidate_columns(_compression_quality_rows(input_path, original, stored_path, stored), domain)
    row = {
        "Method": _format_method_name(st.session_state.get("storage_method", "No compression")),
        "Original data": fmt_bytes(len(original)),
        "Original extension": os.path.splitext(input_path or "")[1].lower() or magic_dict(original).get("ext", ".bin"),
        "Compressed data": fmt_bytes(len(stored)),
        "Compressed extension": os.path.splitext(stored_path or "")[1].lower() or magic_dict(stored).get("ext", ".bin"),
        "Compression ratio": f"{len(original) / max(1, len(stored)):.2f}x",
        "Size reduction": _pct(1.0 - (len(stored) / max(1, len(original)))),
    }
    for key, value in quality.items():
        if value != "—":
            row[key] = value
    return [row]


def _encode_decode_analysis_rows(
    stored_bytes: bytes,
    recovered_bytes: bytes,
    dna: str,
    strand_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stored = bytes(stored_bytes or b"")
    recovered = bytes(recovered_bytes or b"")
    byte_accuracy = _byte_accuracy_bytes(stored, recovered)
    hash_match = bool(stored) and hashlib.sha256(stored).hexdigest() == hashlib.sha256(recovered).hexdigest()
    lengths = [len(clean_dna(r.get("Full strand", ""))) for r in strand_rows]
    hp_values = []
    gc_values = []
    for row in strand_rows:
        try:
            hp_values.append(int(row.get("Longest homopolymer", 0)))
            gc_values.append(float(row.get("GC content", 0)))
        except Exception:
            pass

    recovery = "Decoded exactly, every byte matches (100%)" if byte_accuracy >= 1.0 and hash_match else "Decoded with differences"
    return [
        {"Metric": "DNA design", "Value": _display_mapping(st.session_state.get("encoding_mapping", "—"))},
        {"Metric": "Strand format", "Value": "FBR + SI + payload + filler + RBR" if strand_rows else "Not prepared"},
        {"Metric": "DNA length", "Value": f"{len(dna or ''):,} nt"},
        {"Metric": "DNA GC content", "Value": f"{gc_content(dna):.3f}" if dna else "—"},
        {"Metric": "DNA longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0) if dna else "—"},
        {"Metric": "Strand count", "Value": len(strand_rows)},
        {"Metric": "Total strand length", "Value": f"{sum(lengths):,} nt" if lengths else "—"},
        {"Metric": "Average strand GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
        {"Metric": "Strand longest homopolymer", "Value": max(hp_values) if hp_values else "—"},
        {"Metric": "Payload accuracy", "Value": _pct(byte_accuracy)},
        {"Metric": "Checksum", "Value": "Pass" if hash_match else "Fail"},
        {"Metric": "Recovery result", "Value": recovery},
    ]


# -----------------------------------------------------------------------------
# DNA Strand Prep visualization
# -----------------------------------------------------------------------------

_REGION_COLORS = REGION_COLORS


def _row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return ordered strand regions for the prepared strand view."""
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", ""))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    """Render one region with optional red marking at 1-indexed full-strand positions."""
    bg, fg = _REGION_COLORS.get(name, ("#f8fafc", "#0f172a"))
    error_positions = error_positions or set()
    chars = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        if i in error_positions:
            ebg, efg = _REGION_COLORS["Error"]
            chars.append(f'<span class="error-base">{ch}</span>')
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span class="region-tag" style="background:{bg};color:{fg};">'
        f'<b>{name}</b>: {body}</span>'
    )


def _render_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    """Show FBR/SI/Payload/Filler/RBR as colored chunks."""
    parts = []
    cursor = 1
    for name, seq in _row_regions(row):
        parts.append(_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(seq)
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _region_for_position(row: Dict[str, Any], pos_1based: int) -> str:
    cursor = 1
    for name, seq in _row_regions(row):
        n = len(clean_dna(seq))
        if cursor <= int(pos_1based) < cursor + n:
            return name
        cursor += n
    return "Unknown"


def _strand_payload_dna(rows: List[Dict[str, Any]], original_len: int) -> str:
    parts: List[str] = []
    for row in rows:
        payload = ""
        if str(row.get("Advanced error source", "")).strip().lower() == "true":
            payload = str(row.get("Error payload", ""))
        parts.append(clean_dna(payload or row.get("Payload", "")))
    dna = "".join(parts)
    return dna[:int(original_len)]


def _direct_substitute_dna(dna: str, substitution_rate: float, seed: int) -> Tuple[str, int]:
    """Substitution-only noise directly on the encoded DNA payload."""
    seq = list(clean_dna(dna))
    rng = random.Random(str(int(seed)))
    n = 0
    for i, base in enumerate(seq):
        if rng.random() < float(substitution_rate):
            seq[i] = rng.choice([b for b in "ACGT" if b != base])
            n += 1
    return "".join(seq), n


def _mutate_prepared_strand(
    row: Dict[str, Any],
    *,
    scope: str,
    substitution_rate: float,
    insertion_rate: float,
    deletion_rate: float,
    seed: int,
    allow_indels: bool,
) -> Dict[str, Any]:
    rng = random.Random(str(seed) + "|" + str(row.get("No.", "")))
    full = clean_dna(row.get("Full strand", "")) or "".join(seq for _, seq in _row_regions(row))
    mutable_regions = {
        "Payload only": {"Payload"},
        "Index + Payload": {"SI", "Payload"},
        "Full strand": {"FBR", "SI", "Payload", "Filler", "RBR"},
    }.get(scope, {"Payload"})

    out: List[str] = []
    events: List[Dict[str, Any]] = []
    sub_count = ins_count = del_count = 0
    read_pos = 0

    for pos, base in enumerate(full, start=1):
        region = _region_for_position(row, pos)
        mutable = region in mutable_regions

        if mutable and allow_indels and rng.random() < float(deletion_rate):
            del_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos + 1,
                "region": region,
                "operation": "deletion",
                "from_base": base,
                "to_base": "",
            })
            if rng.random() < float(insertion_rate):
                nb = rng.choice("ACGT")
                out.append(nb)
                read_pos += 1
                ins_count += 1
                events.append({
                    "source_no": row.get("No.", ""),
                    "position_original": pos,
                    "position_read": read_pos,
                    "region": region,
                    "operation": "insertion",
                    "from_base": "",
                    "to_base": nb,
                })
            continue

        new_base = base
        if mutable and rng.random() < float(substitution_rate):
            new_base = rng.choice([b for b in "ACGT" if b != base])
            sub_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos + 1,
                "region": region,
                "operation": "substitution",
                "from_base": base,
                "to_base": new_base,
            })
        out.append(new_base)
        read_pos += 1

        if mutable and allow_indels and rng.random() < float(insertion_rate):
            nb = rng.choice("ACGT")
            out.append(nb)
            read_pos += 1
            ins_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos,
                "region": region,
                "operation": "insertion",
                "from_base": "",
                "to_base": nb,
            })

    err_full = "".join(out)
    fbr_len = len(clean_dna(row.get("FBR", "")))
    idx_len = len(clean_dna(row.get("Strand index", row.get("Index", ""))))
    payload_len = len(clean_dna(row.get("Payload", "")))
    payload_start = fbr_len + idx_len
    err_payload = clean_dna(err_full[payload_start:payload_start + payload_len])

    new = dict(row)
    new.update({
        "Clean full strand": full,
        "Clean payload": clean_dna(row.get("Payload", "")),
        "Error full strand": err_full,
        "Error payload": err_payload,
        "Full strand": err_full,
        "Advanced error source": "true",
        "Advanced error scope": scope,
        "Advanced error events": json.dumps(events, ensure_ascii=False),
        "Substitution count": str(sub_count),
        "Insertion count": str(ins_count),
        "Deletion count": str(del_count),
        "Error count": str(sub_count + ins_count + del_count),
        "Error full length": str(len(err_full)),
    })

    # Keep region columns aligned when only substitutions are used.
    if ins_count == 0 and del_count == 0:
        cursor = 0
        mutated_regions: Dict[str, str] = {}
        for name, seq in _row_regions(row):
            n = len(clean_dna(seq))
            mutated_regions[name] = err_full[cursor:cursor + n]
            cursor += n
        new["FBR"] = mutated_regions.get("FBR", new.get("FBR", ""))
        new["Strand index"] = mutated_regions.get("SI", new.get("Strand index", ""))
        new["Index"] = mutated_regions.get("SI", new.get("Index", new.get("Strand index", "")))
        new["Payload"] = mutated_regions.get("Payload", new.get("Payload", ""))
        new["Filler"] = mutated_regions.get("Filler", new.get("Filler", ""))
        new["RBR"] = mutated_regions.get("RBR", new.get("RBR", ""))

    return new


def _error_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "No.",
        "Advanced error scope",
        "Total length",
        "Error full length",
        "Substitution count",
        "Insertion count",
        "Deletion count",
        "Error count",
    ]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _error_events_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            events = json.loads(row.get("Advanced error events", "[]") or "[]")
        except Exception:
            events = []
        for ev in events:
            out.append({
                "Strand": ev.get("source_no", row.get("No.", "")),
                "Region": ev.get("region", ""),
                "Original position": ev.get("position_original", ""),
                "Error position": ev.get("position_read", ""),
                "Operation": ev.get("operation", ""),
                "Original base": ev.get("from_base", ""),
                "New/inserted base": ev.get("to_base", ""),
            })
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Panel 1 — Upload
# -----------------------------------------------------------------------------


def render_panel_1_upload() -> None:
    """Image input panel.

    This keeps the original image algorithm structure: RGB / grayscale /
    black-white is selected at Input as pixel representation, not as a
    compression method.
    """
    _apply_panel_typography()
    with st.container(border=True):
        step_header(1, PANEL_TITLES["input"])
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 📁 Input")
            uploaded = st.file_uploader(
                "Input image file",
                type=UPLOAD_TYPES_BY_DOMAIN.get("image"),
                key="upload_input_file",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                data_now = uploaded.getvalue()
                upload_sig = f"{uploaded.name}|{len(data_now)}|{hashlib.sha256(data_now).hexdigest()}"
                if st.session_state.get("upload_signature") != upload_sig:
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "upload_signature": upload_sig,
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })
                    _clear_downstream_from_storage()
                elif not st.session_state.get("input_bytes"):
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })

            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            if data and path:
                st.markdown("##### 📄 File properties")
                _render_info_table(_file_info_rows(path, data))

                st.markdown("##### 🧩 Input representation")
                try:
                    import robust_image_pipeline as rip  # type: ignore
                    reps = rip.list_pixel_representations()
                except Exception:
                    reps = ["Black-white (1 bit/pixel)", "Grayscale (8 bits/pixel)", "RGB (24 bits/pixel)"]

                current_rep = st.session_state.get("image_pixel_representation", "Grayscale (8 bits/pixel)")
                if current_rep not in reps:
                    current_rep = "Grayscale (8 bits/pixel)" if "Grayscale (8 bits/pixel)" in reps else reps[0]
                rep = st.selectbox(
                    "Pixel representation used by Image Step 2",
                    reps,
                    index=reps.index(current_rep),
                    key="image_pixel_representation",
                )
                threshold = int(st.session_state.get("image_binary_threshold", 128))
                if "black" in rep.lower():
                    threshold = st.slider(
                        "Black-white threshold",
                        min_value=0,
                        max_value=255,
                        value=threshold,
                        step=1,
                        key="image_binary_threshold",
                    )

                try:
                    import robust_image_pipeline as rip  # type: ignore
                    raw_info = rip.raw_pixel_info(path, rep, threshold=threshold)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Representation", raw_info.get("pixel_representation", rep))
                    m2.metric("Raw pixel payload", fmt_bytes(int(raw_info.get("raw_pixel_bytes", 0))))
                    m3.metric("Bits / pixel", raw_info.get("bits_per_pixel", "—"))
                except Exception:
                    st.caption("Raw-pixel size will be calculated in Step 2.")

                with st.expander(FIELDS["input_binary"], expanded=False):
                    st.text_area("Binary preview only", _bytes_to_bit_text(data), height=120)
                    _download_full_binary_button(BUTTONS["download_input_binary"], data, DOWNLOAD_FILES["input_binary"])

        with right:
            st.markdown("#### 🖼️ Preview")
            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            if not data or not path:
                st.info(MESSAGES["upload_to_start"])
                return
            preview_file(path, FIELDS["input_preview"])


def render_panel_2_compression() -> None:
    """Image method panel.

    No compression uses the pixel representation selected in Step 1.
    Compression uses the original robust_image_pipeline.py methods.
    No all-method benchmark is run; only the selected method runs.
    """
    with st.container(border=True):
        step_header(2, "Compression")
        data = st.session_state.get("input_bytes")
        path = st.session_state.get("input_path")
        if not data or not path:
            st.info(MESSAGES["upload_first"])
            return

        domain = "image"
        rep = st.session_state.get("image_pixel_representation", "Grayscale (8 bits/pixel)")
        threshold = int(st.session_state.get("image_binary_threshold", 128))
        controls, selected_output = st.columns([0.9, 1.1], gap="large")

        with controls:
            st.markdown("#### 🗂️ Configuration Setup")
            # storage_mode = st.radio(
            #     "Storage route",
            #     [DATA_SOURCES["no_compression"], DATA_SOURCES["compression"]],
            #     horizontal=True,
            #     key="storage_mode",
            # )
            storage_mode = DATA_SOURCES["compression"]
# st.caption("Compression mode is used by default.")
            if storage_mode == DATA_SOURCES["no_compression"]:
                if st.button("Run no-compression pixel payload", key="use_no_compression", type="primary", use_container_width=True):
                    try:
                        import robust_image_pipeline as rip  # type: ignore
                        payload, meta, preview_path = rip.image_to_raw_pixels(
                            path,
                            _selected_payload_dir(domain),
                            pixel_representation=rep,
                            threshold=threshold,
                        )
                        meta.update({
                            "kind": "raw_pixels",
                            "method_display": "No compression",
                            "preview_path": preview_path,
                            "domain": domain,
                        })
                        payload_path = _write_selected_payload(domain, "No compression", payload, ".raw")
                        _store_method_payload(domain, "No compression", payload, payload_path, meta, preview_path)
                        st.success("Completed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"No-compression pixel payload failed: {exc}")
            else:
                try:
                    import robust_image_pipeline as rip  # type: ignore
                    methods = [m for m in rip.list_image_compression_methods() if m != "No compression"]
                    levels = [m for m in rip.list_compression_levels() if m != "Custom"]
                    caption_fn = rip.method_caption
                except Exception:
                    methods = ["Robust Low-Resolution", "Base + Local Detail", "Base + WebP Detail", "Local Block Coding"]
                    levels = ["High quality", "Balanced", "High compression"]
                    caption_fn = lambda _m: ""

                method = st.selectbox("Image compression method", methods, key="image_method_select")
                level = st.selectbox("Compression level", levels, index=levels.index("Balanced") if "Balanced" in levels else 0, key="image_compression_level")

                if st.button("Run selected image method", key="run_selected_method", type="primary", use_container_width=True):
                    try:
                        import robust_image_pipeline as rip  # type: ignore
                        payload, meta, preview_path = rip.encode_image_to_payload(
                            path,
                            method,
                            _selected_payload_dir(domain),
                            compression_level=level,
                            pixel_representation=rep,
                            threshold=threshold,
                        )
                        meta.update({
                            "kind": "robust_image_payload",
                            "method_display": method,
                            "preview_path": preview_path,
                            "domain": domain,
                        })
                        payload_path = _write_selected_payload(domain, method, payload, ".imgpayload")
                        _store_method_payload(domain, method, payload, payload_path, meta, preview_path)
                        st.success("Completed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Selected image method failed: {exc}")

        with selected_output:
            st.markdown("#### 🧬 Estimated Compression Output")
            stored = st.session_state.get("stored_bytes")
            stored_path = st.session_state.get("stored_file_path")
            storage_meta = st.session_state.get("storage_meta", {}) or {}
            if not stored:
                st.info("Run no-compression pixel payload or one selected image compression method.")
            else:
                kind = st.session_state.get("storage_kind", magic_dict(stored).get("kind", "unknown"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Selected payload", fmt_bytes(len(stored)))
                c2.metric("Method", st.session_state.get("storage_method", "—"))
                c3.metric("Estimated DNA", f"{len(stored) * 4:,} nt")
                _render_property_table([
                    {"Metric": "Payload kind", "Value": kind},
                    {"Metric": "Input representation", "Value": storage_meta.get("pixel_representation", rep)},
                    {"Metric": "Payload file", "Value": os.path.basename(stored_path or "—")},
                ])
                metric_rows = _method_result_rows(path, data, stored_path, stored, storage_meta)
                if metric_rows:
                    _render_stats_table(metric_rows)
                d1, d2 = st.columns(2)
                with d1:
                    download_bytes_button(BUTTONS["download_stored_data"], stored, file_name=f"selected_payload{Path(stored_path or '').suffix or '.bin'}")
                with d2:
                    _download_full_binary_button(BUTTONS["download_stored_binary"], stored, DOWNLOAD_FILES["stored_binary"])

        st.markdown("#### Method result")
        before, after = st.columns(2, gap="large")
        with before:
            st.markdown("##### Before compression")
            preview_file(path, "Original preview")
            _render_property_table(_pipeline_file_metric_rows(path, data))
        with after:
            st.markdown("##### After compression")
            stored = st.session_state.get("stored_bytes")
            stored_path = st.session_state.get("stored_file_path")
            storage_meta = st.session_state.get("storage_meta", {}) or {}
            preview_path = storage_meta.get("preview_path")
            if stored and stored_path:
                display_preview = _selected_payload_preview_path(stored_path, storage_meta)
                if display_preview:
                    preview_file(display_preview, "Selected-method preview")
                _render_property_table(_pipeline_file_metric_rows(stored_path, stored, compressed=True))
            #     metric_rows = _method_result_rows(path, data, stored_path, stored, storage_meta)
            #     if metric_rows:
            #         _render_stats_table(metric_rows)
            # else:
            #     st.info("No payload selected yet.")

def render_panel_3_encoding() -> None:
    with st.container(border=True):
        step_header(3, "Encoding")
        payload = st.session_state.get("stored_bytes")
        if not payload:
            if st.session_state.get("compression_candidates"):
                st.info("Select one compression method in Panel 2 before DNA encoding.")
            else:
                st.info(MESSAGES["run_data_encoding_first"])
            return

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 🧬 Design")
            previous = st.session_state.get("encoding_mapping", "Simple Mapping")
            if previous not in MAPPING_OPTIONS:
                previous = "Simple Mapping"
            mapping = st.selectbox(
                "Mapping rule",
                MAPPING_OPTIONS,
                index=MAPPING_OPTIONS.index(previous),
                format_func=_display_mapping,
                key="encoding_mapping_select",
            )

            if st.button(BUTTONS["run_dna_encoding"], key="run_encoding", type="primary", use_container_width=True):
                dna, bits, meta = encode_bytes_to_dna(payload, mapping)
                st.session_state.update({
                    "encoding_mapping": mapping,
                    "dna": dna,
                    "bits": bits,
                    "codec_meta": meta,
                    "strand_rows": [],
                    "decoded_data": None,
                    "restored_info": None,
                    "decode_error": "",
                })

            st.markdown("##### 📄 Encoded data properties")
            _render_property_table([
                {"Metric": "Encoded data", "Value": fmt_bytes(len(payload))},
                {"Metric": "Estimated bits", "Value": f"{len(payload) * 8:,} bits"},
            ])
            _download_full_binary_button(BUTTONS["download_encoded_binary"], payload, DOWNLOAD_FILES["encoded_binary"])

        with right:
            st.markdown("#### 🧬 DNA output")
            dna = st.session_state.get("dna", "")
            if not dna:
                st.info(MESSAGES["run_data_encoding_first"])
                return

            _render_property_table([
                {"Metric": "DNA design", "Value": _display_mapping(st.session_state.get("encoding_mapping", mapping))},
                {"Metric": METRICS["dna_length"], "Value": f"{len(dna):,} nt"},
                {"Metric": "GC content", "Value": f"{gc_content(dna):.3f}"},
                {"Metric": "Longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0)},
            ])
            st.text_area("DNA payload preview", _preview_seq(dna, 600), height=DNA_PREVIEW_HEIGHT)
            _download_text_button(BUTTONS["download_encoded_dna"], dna, DOWNLOAD_FILES["encoded_dna"])

            st.markdown("##### 🧪 Payload-level noise")
            show_payload_noise = st.checkbox(
                "Advanced: add noise directly to encoded DNA payload",
                value=False,
                key="image_show_payload_noise",
            )
            if show_payload_noise:
                n1, n2 = st.columns(2)
                direct_sub_rate = n1.number_input(
                    "Substitution rate",
                    min_value=0.0,
                    max_value=0.20,
                    value=0.0050,
                    step=0.001,
                    format="%.4f",
                    key="image_direct_sub_rate",
                )
                direct_seed = n2.number_input("Seed", min_value=0, value=7, step=1, key="image_direct_sub_seed")
                st.caption("This is the fast payload-level substitution test. It skips strand design and is useful for checking robustness quickly.")
                if st.button("Add noise to DNA payload", key="image_add_direct_noise", type="primary", use_container_width=True):
                    noisy_dna, n_sub = _direct_substitute_dna(dna, float(direct_sub_rate), int(direct_seed))
                    st.session_state["noisy_dna"] = noisy_dna
                    st.session_state["direct_noisy_dna"] = noisy_dna
                    st.session_state["error_stats"] = {
                        "source": "Payload-level DNA noise",
                        "quick_skip_strand": True,
                        "substitution_rate": float(direct_sub_rate),
                        "seed": int(direct_seed),
                        "Substitute count": int(n_sub),
                        "noisy_dna_len": int(len(noisy_dna)),
                    }
                    st.success(f"Added {n_sub:,} substitutions to encoded DNA payload.")
                noisy_dna = st.session_state.get("noisy_dna", "")
                if noisy_dna and st.session_state.get("error_stats", {}).get("quick_skip_strand"):
                    qs = st.session_state.get("error_stats", {})
                    m1, m2 = st.columns(2)
                    m1.metric("Direct substitutions", f"{int(qs.get('Substitute count', 0)):,}")
                    m2.metric("Noisy encoded DNA", f"{len(noisy_dna):,} nt")
                    st.text_area("Noisy DNA preview", _preview_seq(noisy_dna, 600), height=120, key="image_noisy_dna_preview_payload")
                    _download_text_button("Download noisy encoded DNA", noisy_dna, "image_noisy_encoded_dna.txt")


# -----------------------------------------------------------------------------
# Panel 4 — DNA Strand Prep
# -----------------------------------------------------------------------------



def render_panel_4_experiment() -> None:
    with st.container(border=True):
        step_header(4, PANEL_TITLES["strand_preparation"])
        dna = st.session_state.get("dna", "")
        if not dna:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        mapping = st.session_state.get("encoding_mapping", "")
        st.markdown(f"#### 🧵 {PANEL_TITLES['strand_preparation']}")
        with st.expander(FIELDS["strand_design"], expanded=not bool(st.session_state.get("strand_rows"))):
            target_len = st.number_input(FIELDS["total_strand_length"], min_value=80, max_value=250, value=125, step=1, key="std_total_len")
            index_len = st.number_input(FIELDS["si_length"], min_value=0, max_value=24, value=8, step=1, key="std_index_len")
            fbr = st.text_input(FIELDS["fbr"], value="ACACGACGCTCTTCCGATCT", key="std_fbr")
            rbr = st.text_input(FIELDS["rbr"], value="AGATCGGAAGAGCACACGTCT", key="std_rbr")
            if st.button(BUTTONS["run_strand_preparation"], key="build_standard_strands"):
                cfg = choose_auto_strand_design(
                    len(dna), len(clean_dna(fbr)), len(clean_dna(rbr)), int(index_len),
                    min_total_len=int(target_len), max_total_len=int(target_len),
                )
                rows = prepare_dna_strands(
                    dna,
                    fbr=clean_dna(fbr),
                    rbr=clean_dna(rbr),
                    index_len=int(index_len),
                    target_total_len=int(cfg.get("target_total_len", cfg.get("total_len", target_len))),
                    add_filler=True,
                )
                for r in rows:
                    r["Type"] = FIELDS["prepared_strand"]
                st.session_state.update({
                    "strand_rows": rows,
                    "decoded_data": None,
                    "restored_info": None,
                })

        rows: List[Dict[str, Any]] = st.session_state.get("strand_rows", [])
        if not rows:
            st.info(MESSAGES["run_strand_preparation"])
            return

        lengths = [len(clean_dna(r.get("Full strand", ""))) for r in rows]
        total_strand_len = sum(lengths)
        gc_values = []
        hp_values = []
        for r in rows:
            try:
                gc_values.append(float(r.get("GC content", 0)))
                hp_values.append(int(r.get("Longest homopolymer", 0)))
            except Exception:
                pass
        st.markdown("##### 📄 Strand statistics")
        _render_property_table([
            {"Metric": METRICS["prepared_strands"], "Value": len(rows)},
            {"Metric": METRICS["total_strand_length"], "Value": f"{total_strand_len:,} nt"},
            {"Metric": "Average GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
            {"Metric": "Longest homopolymer", "Value": max(hp_values) if hp_values else "—"},
            {"Metric": "DNA design", "Value": _display_mapping(mapping or "—")},
        ])

        selected_index = int(st.number_input(
            "Strand ID",
            min_value=1,
            max_value=max(1, len(rows)),
            value=1,
            step=1,
            key="inspect_prepared_strand_no",
        ))
        selected_row = rows[selected_index - 1]
        _render_segmented_strand(selected_row, FIELDS["prepared_strand"])

        st.markdown("##### 🧾 Strand table")
        table_rows = []
        for r in rows[:500]:
            table_rows.append({
                "No.": r.get("No.", "—"),
                "SI": r.get("Strand index", "—"),
                "Payload length": r.get("Payload length", "—"),
                "Filler length": r.get("Filler length", "—"),
                "Total length": r.get("Total length", "—"),
                "GC content": r.get("GC content", "—"),
                "Longest homopolymer": r.get("Longest homopolymer", "—"),
                "Homo count": r.get("Homopolymer count", "—"),
            })
        _render_stats_table(table_rows, height=260)
        if len(rows) > 500:
            st.caption(f"Showing first 500 of {len(rows):,} strands.")

        st.download_button(BUTTONS["download_prepared_strands"], data=strand_rows_to_csv(rows), file_name=DOWNLOAD_FILES["prepared_strands"], mime="text/csv", use_container_width=True)

        st.markdown("##### 🧪 Strand-level noise")
        with st.expander("Advanced: add noise to prepared small strands", expanded=False):
            st.caption("This simulates sequencing/synthesis errors on the prepared strands generated by the app. The decoded payload is reconstructed from each strand Payload/Error payload region.")
            e1, e2, e3 = st.columns(3)
            scope = e1.selectbox("Error scope", ["Payload only", "Index + Payload", "Full strand"], index=0, key="image_error_scope")
            substitution_rate = e2.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.0200, step=0.001, format="%.4f", key="image_sub_rate")
            seed = e3.number_input("Seed", min_value=0, value=11, step=1, key="image_error_seed")
            allow_indels = st.checkbox("Include insertion/deletion errors", value=False, key="image_allow_indels")
            if allow_indels:
                i1, i2 = st.columns(2)
                insertion_rate = i1.number_input("Insertion", min_value=0.0, max_value=0.05, value=0.0000, step=0.0005, format="%.4f", key="image_ins_rate")
                deletion_rate = i2.number_input("Deletion", min_value=0.0, max_value=0.05, value=0.0000, step=0.0005, format="%.4f", key="image_del_rate")
            else:
                insertion_rate = 0.0
                deletion_rate = 0.0

            if st.button("Add noise to prepared strands", key="image_apply_strand_noise", type="primary", use_container_width=True):
                err_rows = [
                    _mutate_prepared_strand(
                        row,
                        scope=str(scope),
                        substitution_rate=float(substitution_rate),
                        insertion_rate=float(insertion_rate),
                        deletion_rate=float(deletion_rate),
                        seed=int(seed),
                        allow_indels=bool(allow_indels),
                    )
                    for row in rows
                ]
                noisy_dna = _strand_payload_dna(err_rows, len(clean_dna(dna)))
                st.session_state["error_rows"] = err_rows
                st.session_state["noisy_dna"] = noisy_dna
                st.session_state["error_stats"] = {
                    "source": "Strand-level noise",
                    "scope": str(scope),
                    "substitution_rate": float(substitution_rate),
                    "insertion_rate": float(insertion_rate),
                    "deletion_rate": float(deletion_rate),
                    "seed": int(seed),
                    "strand_count": int(len(err_rows)),
                    "noisy_dna_len": int(len(noisy_dna)),
                }
                st.success("Generated error strands and noisy encoded DNA.")

            err_rows = st.session_state.get("error_rows", [])
            noisy_dna = st.session_state.get("noisy_dna", "")
            if err_rows:
                a1, a2, a3 = st.columns(3)
                total_errors = sum(int(r.get("Error count", 0) or 0) for r in err_rows)
                a1.metric("Error strands", f"{len(err_rows):,}")
                a2.metric("Total error events", f"{total_errors:,}")
                a3.metric("Noisy encoded DNA", f"{len(noisy_dna):,} nt")
                _render_stats_table(_error_rows_table(err_rows).to_dict("records"), height=220)
                events_df = _error_events_table(err_rows)
                if not events_df.empty:
                    with st.expander("Error event table", expanded=False):
                        st.dataframe(events_df.head(500), use_container_width=True, hide_index=True)
                        if len(events_df) > 500:
                            st.caption(f"Showing first 500 of {len(events_df):,} error events.")
                st.download_button("Download error strands", strand_rows_to_csv(err_rows), "image_error_strands.csv", "text/csv", use_container_width=True)
                _download_text_button("Download noisy encoded DNA", noisy_dna, "image_noisy_encoded_dna.txt")


# -----------------------------------------------------------------------------
# Panel 5 — Decode
# -----------------------------------------------------------------------------


def render_panel_5_decoding() -> None:
    with st.container(border=True):
        step_header(5, "Decoding")
        mapping = st.session_state.get("encoding_mapping")
        if not mapping:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 📁 DNA input")
            source = st.radio(
                "DNA source",
                [
                    "Current encoded DNA",
                    "Noisy encoded DNA",
                    "Upload full DNA TXT",
                    "Upload prepared strands CSV",
                    "Upload error strands CSV",
                ],
                horizontal=False,
                key="decode_dna_source",
            )
            if source == "Upload full DNA TXT":
                uploaded_dna = st.file_uploader("Upload DNA payload TXT generated by this app", type=["txt"], key="decode_encoded_upload")
                if uploaded_dna is not None:
                    raw_text = uploaded_dna.getvalue().decode("utf-8", errors="ignore")
                    dna_text = clean_dna(raw_text)
                    st.session_state["decode_uploaded_dna"] = dna_text
                    source_label = f"Uploaded DNA TXT ({uploaded_dna.name})"
                else:
                    dna_text = st.session_state.get("decode_uploaded_dna", "")
                    source_label = "Uploaded DNA TXT"
            elif source == "Upload prepared strands CSV":
                uploaded_csv = st.file_uploader("Upload prepared strands CSV generated by this app", type=["csv"], key="decode_strand_csv_upload")
                if uploaded_csv is not None:
                    dna_text, row_count, note = _dna_from_uploaded_strand_csv(uploaded_csv.getvalue())
                    st.session_state["decode_uploaded_csv_dna"] = dna_text
                    st.session_state["decode_uploaded_csv_note"] = note
                    source_label = f"Uploaded prepared strands CSV ({row_count:,} strands)"
                else:
                    dna_text = st.session_state.get("decode_uploaded_csv_dna", "")
                    source_label = "Uploaded prepared strands CSV"
                note = st.session_state.get("decode_uploaded_csv_note")
                if note:
                    st.caption(note)
            elif source == "Upload error strands CSV":
                uploaded_csv = st.file_uploader("Upload error strands CSV generated by Advanced strand-noise simulation", type=["csv"], key="decode_error_strand_csv_upload")
                if uploaded_csv is not None:
                    dna_text, row_count, note = _dna_from_uploaded_strand_csv(uploaded_csv.getvalue())
                    st.session_state["decode_uploaded_error_csv_dna"] = dna_text
                    st.session_state["decode_uploaded_error_csv_note"] = note
                    source_label = f"Uploaded error strands CSV ({row_count:,} strands)"
                else:
                    dna_text = st.session_state.get("decode_uploaded_error_csv_dna", "")
                    source_label = "Uploaded error strands CSV"
                note = st.session_state.get("decode_uploaded_error_csv_note")
                if note:
                    st.caption(note)
            elif source == "Noisy encoded DNA":
                dna_text = st.session_state.get("noisy_dna", "")
                source_label = "Noisy encoded DNA from payload/strand noise"
                if not dna_text:
                    st.info("No noisy DNA exists yet. Use Advanced payload-level noise in Step 3 or Advanced strand-level noise in Step 4, or upload a noisy DNA/strand file.")
            else:
                source_label, dna_text = _decode_source()

            c1, c2 = st.columns(2)
            c1.metric("DNA design", _display_mapping(mapping))
            c2.metric(METRICS["input_dna"], source_label)
            st.text_area("Input DNA payload preview", _preview_seq(dna_text, 600), height=120)

            if st.button(BUTTONS["run_decode"], key="run_decode", type="primary", use_container_width=True):
                try:
                    if not clean_dna(dna_text):
                        raise ValueError("No valid encoded DNA sequence was provided.")
                    data, bits, meta = decode_dna_with_mapping(
                        dna_text,
                        mapping,
                        codec_meta=st.session_state.get("codec_meta", {}) or {},
                    )
                    storage_meta = st.session_state.get("storage_meta", {}) or {}
                    decoded_output = data
                    decoded_raw_pixels = None
                    raw_restore_info: Dict[str, Any] = {}
                    kind = str(storage_meta.get("kind", ""))
                    if kind == "raw_image_pixels":
                        decoded_raw_pixels = data
                        decoded_output, raw_restore_info = _raw_image_bytes_to_png(data, storage_meta)
                        m = detect_magic(decoded_output)
                        valid = True
                        note = f"Raw image pixels restored as PNG. {raw_restore_info.get('note', '')}"
                        info = _validate_and_write(decoded_output, preferred="restored_raw_image")
                    elif kind == "robust_image_payload":
                        try:
                            import robust_image_pipeline as rip  # type: ignore
                            out_dir = WORK_ROOT / "decode_output"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / "restored_robust_image.png"
                            raw_restore_info = rip.decode_payload_to_image(data, storage_meta, str(out_path))
                            decoded_output = out_path.read_bytes()
                            m = detect_magic(decoded_output)
                            valid = True
                            note = "Robust image payload restored as PNG."
                            info = _validate_and_write(decoded_output, preferred="restored_robust_image")
                        except Exception as exc:
                            raise RuntimeError(f"Could not restore robust image payload: {exc}") from exc
                    elif kind == "audio_payload":
                        try:
                            import audio_dna_tab as audiopanel  # type: ignore
                            decoded_output, raw_restore_info, _samples = audiopanel._reconstruct_audio(data, storage_meta)
                            m = detect_magic(decoded_output)
                            valid = True
                            note = "Audio payload restored as WAV."
                            info = _validate_and_write(decoded_output, preferred="restored_audio")
                        except Exception as exc:
                            raise RuntimeError(f"Could not restore audio payload: {exc}") from exc
                    elif kind == "video_payload":
                        try:
                            from video_dna_payload_codec import reconstruct_video_from_payload  # type: ignore
                            out_dir = WORK_ROOT / "decode_output" / "video_payload"
                            result = reconstruct_video_from_payload(data, storage_meta.get("manifest", {}) or {}, out_dir)
                            video_path = result.get("final_video") or result.get("visual_video")
                            if not video_path or not os.path.exists(video_path):
                                raise RuntimeError("Video reconstruction did not produce an output video.")
                            decoded_output = Path(video_path).read_bytes()
                            raw_restore_info = result.get("metrics", {}) or {}
                            m = detect_magic(decoded_output)
                            valid = True
                            note = "Video payload restored as MP4."
                            info = _validate_and_write(decoded_output, preferred="restored_video")
                            info["preview_path"] = video_path
                            info["file_path"] = video_path
                        except Exception as exc:
                            raise RuntimeError(f"Could not restore video payload: {exc}") from exc
                    elif kind == "text_token_payload":
                        try:
                            import text_dna_unified_panel as txtpanel  # type: ignore
                            package = storage_meta.get("text_package") or {}
                            bit_len = int(storage_meta.get("codec_bit_len") or package.get("codec_bit_len") or 0)
                            bits_text = bytes_to_bitstring(data)
                            if bit_len:
                                bits_text = bits_text[:bit_len]
                            codec_dna = txtpanel.project_bits_to_dna(bits_text)
                            codec = txtpanel._codec_from_package(package)
                            dec = codec.decode(codec_dna, package.get("meta", {}) or {})
                            decoded_output = str(dec.text or "").encode("utf-8")
                            raw_restore_info = {"decode_ok": bool(dec.decode_ok), **(dec.meta or {})}
                            m = detect_magic(decoded_output)
                            valid = True
                            note = "Text token payload restored as UTF-8 text."
                            info = _validate_and_write(decoded_output, preferred="restored_text")
                        except Exception as exc:
                            raise RuntimeError(f"Could not restore text token payload: {exc}") from exc
                    else:
                        m = detect_magic(decoded_output)
                        valid = False
                        note = "No recognizable file signature"
                        if m:
                            valid, note = validate_container(decoded_output, m.kind)
                        info = _validate_and_write(decoded_output, preferred="restored")
                    st.session_state.update({
                        "decoded_data": decoded_output,
                        "decoded_raw_pixels": decoded_raw_pixels,
                        "decoded_bits": bits,
                        "decoded_meta": meta,
                        "decoded_magic": m,
                        "decoded_valid": valid,
                        "decoded_note": note,
                        "raw_restore_info": raw_restore_info,
                        "restored_info": info,
                        "decode_source_label": source_label,
                        "decode_error": "",
                    })
                except Exception as exc:
                    st.session_state["decode_error"] = str(exc)
                    st.session_state["restored_info"] = None

            if st.session_state.get("decode_error"):
                st.error(st.session_state["decode_error"])

        with right:
            st.markdown("#### 🖼️ Decoded output")
            data = st.session_state.get("decoded_data")
            if data is None:
                st.info(MESSAGES["run_decode_first"])
                return
            m = st.session_state.get("decoded_magic")
            info = st.session_state.get("restored_info") or {}
            preview_path = info.get("preview_path") or info.get("file_path")
            if preview_path:
                preview_file(preview_path, "Decoded preview")
            _render_property_table(_pipeline_file_metric_rows(preview_path or "", data or b""))
            _render_property_table([
                {"Metric": METRICS["decoded_size"], "Value": fmt_bytes(len(data))},
                {"Metric": METRICS["restored_type"], "Value": m.kind if m else "unknown"},
            ])

            d1, d2 = st.columns(2)
            with d1:
                download_bytes_button(BUTTONS["download_decoded_file"], data, file_name=f"decoded{m.ext if m else '.bin'}")
            with d2:
                _download_full_binary_button(BUTTONS["download_decoded_binary"], data, DOWNLOAD_FILES["decoded_binary"])
            raw_pixels = st.session_state.get("decoded_raw_pixels")
            if raw_pixels is not None:
                r1, r2 = st.columns(2)
                with r1:
                    download_bytes_button("Download decoded raw pixels", raw_pixels, file_name=DOWNLOAD_FILES["decoded_raw_pixels"])
                with r2:
                    _download_full_binary_button("Download decoded raw-pixel binary", raw_pixels, DOWNLOAD_FILES["decoded_raw_pixel_binary"])


# -----------------------------------------------------------------------------
# Panel 6 — Analysis
# -----------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default




def _compact_analysis_rows(rows: List[Dict[str, Any]], keep_metrics: List[str]) -> List[Dict[str, Any]]:
    keep = set(keep_metrics)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows or []:
        metric = str(row.get("Metric", row.get("Property", "")))
        if metric in keep and metric not in seen:
            out.append(row)
            seen.add(metric)
    return out

def _analysis_section(title: str) -> str:
    section = title.replace(" Summary", "").replace(" Report", "")
    section = section.replace("Storage / Payload", "Input / Payload")
    section = section.replace("Error / Noise", "Error")
    section = section.replace("Recovery Quality", "Recovery")
    section = section.replace("DNA / Strand", "DNA / Strand")
    return section


def _metric_value(rows: List[Dict[str, Any]], metric_name: str, default: Any = "—") -> Any:
    for row in rows or []:
        if str(row.get("Metric", row.get("Property", ""))) == metric_name:
            value = row.get("Value", default)
            return default if value is None or value == "" else value
    return default


def _analysis_table(title: str, rows: List[Dict[str, Any]]) -> None:
    st.markdown(f"#### {title}")
    clean_rows = []
    for row in rows or []:
        prop = row.get("Property", row.get("Metric", ""))
        value = row.get("Value", "—")
        clean_rows.append({"Property": str(prop), "Value": "—" if value is None or value == "" else value})
    _render_stats_table(clean_rows)


def _strand_analysis_summary_rows(strand_rows: List[Dict[str, Any]], dna: str) -> List[Dict[str, Any]]:
    lengths: List[int] = []
    gc_values: List[float] = []
    hp_values: List[int] = []
    for row in strand_rows or []:
        full = clean_dna(row.get("Full strand", ""))
        if full:
            lengths.append(len(full))
            gc_values.append(_safe_float(row.get("GC content", gc_content(full))))
            hp_values.append(_safe_int(row.get("Longest homopolymer", homopolymer_stats(full).get("longest", 0))))
    return [
        {"Metric": "Mapping rule", "Value": _display_mapping(st.session_state.get("encoding_mapping", "—"))},
        {"Metric": "DNA length", "Value": f"{len(dna or ''):,} nt"},
        {"Metric": "GC content", "Value": f"{gc_content(dna):.3f}" if dna else "—"},
        {"Metric": "Longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0) if dna else "—"},
        {"Metric": "Strand count", "Value": f"{len(strand_rows or []):,}"},
        {"Metric": "Average strand length", "Value": f"{(sum(lengths) / len(lengths)):.1f} nt" if lengths else "—"},
        {"Metric": "Average strand GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
        {"Metric": "Max strand homopolymer", "Value": max(hp_values) if hp_values else "—"},
        {"Metric": "Strand architecture", "Value": "FBR + SI + Payload + Filler + RBR" if strand_rows else "—"},
    ]


def _image_storage_summary_rows(input_path: str | None, input_bytes: bytes, stored_bytes: bytes, storage_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    original_size = len(input_bytes or b"")
    payload_size = len(stored_bytes or b"")
    representation = (
        storage_meta.get("pixel_representation")
        or storage_meta.get("representation")
        or storage_meta.get("input_representation")
        or st.session_state.get("image_input_representation")
        or "—"
    )
    return [
        {"Metric": "Data type", "Value": "Image"},
        {"Metric": "Input file", "Value": os.path.basename(input_path or "—")},
        {"Metric": "Input representation", "Value": representation},
        {"Metric": "Method", "Value": st.session_state.get("storage_method", "—")},
        {"Metric": "Original size", "Value": fmt_bytes(original_size) if original_size else "—"},
        {"Metric": "Payload size", "Value": fmt_bytes(payload_size) if payload_size else "—"},
        {"Metric": "Compression ratio", "Value": f"{original_size / max(1, payload_size):.2f}x" if original_size and payload_size else "—"},
        {"Metric": "Estimated DNA length", "Value": f"{payload_size * 4:,} nt" if payload_size else "—"},
    ]


def _image_error_summary_rows(dna: str) -> List[Dict[str, Any]]:
    stats = st.session_state.get("error_stats", {}) or {}
    err_rows = st.session_state.get("error_rows", []) or []
    noisy = st.session_state.get("noisy_dna", "") or ""
    if not stats and not noisy:
        return [
            {"Metric": "Error status", "Value": "Clean"},
            {"Metric": "Error level", "Value": "Clean DNA"},
            {"Metric": "Error type", "Value": "None"},
            {"Metric": "Substitution rate", "Value": "0"},
            {"Metric": "Substituted bases", "Value": "0"},
            {"Metric": "Error scope", "Value": "—"},
            {"Metric": "Affected strands", "Value": "0"},
            {"Metric": "Seed", "Value": "—"},
        ]
    return [
        {"Metric": "Error status", "Value": "Noisy"},
        {"Metric": "Error level", "Value": "Payload-level" if stats.get("quick_skip_strand") else ("Strand-level" if err_rows else "Payload-level")},
        {"Metric": "Error type", "Value": "Substitution"},
        {"Metric": "Substitution rate", "Value": stats.get("substitution_rate", "—")},
        {"Metric": "Substituted bases", "Value": f"{_safe_int(stats.get('Substitute count', stats.get('total_errors', 0))):,}"},
        {"Metric": "Error scope", "Value": stats.get("scope", "payload")},
        {"Metric": "Affected strands", "Value": f"{len(err_rows):,}" if err_rows else ("—" if stats.get("quick_skip_strand") else "0")},
        {"Metric": "Seed", "Value": stats.get("seed", "—")},
    ]


def render_panel_6_analysis() -> None:
    with st.container(border=True):
        step_header(6, "Summarization")
        info = st.session_state.get("restored_info")
        if not info:
            st.info(MESSAGES["run_decode_first"])
            return

        path = info.get("file_path")
        preview_path = info.get("preview_path") or path
        data = st.session_state.get("decoded_data", b"")

        storage_meta = st.session_state.get("storage_meta", {}) or {}
        stored_bytes = st.session_state.get("stored_bytes", b"") or b""
        if storage_meta.get("kind") == "raw_image_pixels":
            recovered_for_match = st.session_state.get("decoded_raw_pixels", b"") or b""
        else:
            recovered_for_match = data or b""
        stored_path = st.session_state.get("stored_file_path")
        input_path = st.session_state.get("input_path")
        input_bytes = st.session_state.get("input_bytes", b"") or b""
        dna = st.session_state.get("dna", "")
        strand_rows = st.session_state.get("strand_rows", [])

        st.markdown("#### 📊 Summary")
        original_col, compressed_col, decoded_col = st.columns(3, gap="large")
        with original_col:
            st.markdown("##### Original")
            if input_path and input_bytes:
                original_preview = _image_quality_reference_path(input_path, storage_meta) or input_path
                preview_file(original_preview, "Original preview")
                _render_property_table(_pipeline_file_metric_rows(input_path, input_bytes))
            else:
                st.info(MESSAGES["upload_first"])
        with compressed_col:
            st.markdown("##### Compressed/Encoded")
            if stored_path and stored_bytes:
                stored_preview = _selected_payload_preview_path(stored_path, storage_meta)
                if stored_preview:
                    preview_file(stored_preview, "Compressed/Encoded preview", show_caption=False)
                else:
                    st.write("—")
                _render_property_table(_pipeline_file_metric_rows(stored_path, stored_bytes, compressed=True))
            else:
                st.info(MESSAGES["run_data_encoding_first"])
        with decoded_col:
            st.markdown("##### Decoded")
            if preview_path:
                preview_file(preview_path, "Decoded preview")
            _render_property_table(_pipeline_file_metric_rows(path, data or b""))

        _analysis_table("Compression analysis", _image_storage_summary_rows(input_path, input_bytes, stored_bytes, storage_meta))
        _analysis_table("Encode-decode analysis", _compact_analysis_rows(
            _strand_analysis_summary_rows(strand_rows, dna),
            ["Mapping rule", "DNA length", "GC content", "Longest homopolymer", "Strand count", "Average strand GC", "Max strand homopolymer"],
        ))
        _analysis_table("Error Adding Report", _compact_analysis_rows(
            _image_error_summary_rows(dna),
            ["Error status", "Error level", "Error type", "Substitution rate", "Substituted bases", "Error scope", "Affected strands", "Seed"],
        ))

        recovery_rows = _encode_decode_analysis_rows(stored_bytes, recovered_for_match, dna, strand_rows)
        final_quality: List[Dict[str, Any]] = []
        if str(storage_meta.get("domain", "image")).lower() == "image":
            ref = _image_quality_reference_path(input_path, storage_meta)
            decoded_preview = preview_path if preview_path and os.path.exists(str(preview_path)) else None
            final_quality = _image_quality_metric_rows(ref, decoded_preview, "Recovered image quality")

        payload_accuracy = _metric_value(recovery_rows, "Payload accuracy", "—")
        checksum = _metric_value(recovery_rows, "Checksum", "—")
        output_ok = bool(preview_path and os.path.exists(str(preview_path))) or bool(st.session_state.get("decoded_valid"))
        recovery_class = "Exact" if checksum == "Pass" else ("Usable" if output_ok else "Failed")
        recovery_report_rows = [
            {"Metric": "Decode source", "Value": st.session_state.get("decode_source_label", "Current encoded DNA")},
            {"Metric": "Decode status", "Value": "Success" if not st.session_state.get("decode_error") else "Failed"},
            {"Metric": "Output status", "Value": "Openable" if output_ok else "Failed"},
            {"Metric": "Payload accuracy", "Value": payload_accuracy},
            {"Metric": "Checksum", "Value": checksum},
            {"Metric": "Recovery class", "Value": recovery_class},
        ]
        for metric in ["PSNR", "SSIM", "MAE"]:
            value = _metric_value(final_quality, metric, None)
            if value is not None:
                recovery_report_rows.append({"Metric": metric, "Value": value})
        _analysis_table("Recovery Quality Report", recovery_report_rows)


        method_rows = _method_result_rows(input_path, input_bytes, stored_path, stored_bytes, storage_meta)
        if method_rows:
            with st.expander("Method-specific details", expanded=False):
                _render_property_table(method_rows)
