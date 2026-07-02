from __future__ import annotations

import io
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from config import MAPPING_OPTIONS, WORK_ROOT
from dna_codec import gc_content, homopolymer_stats
from dna_mapping import decode_dna_with_mapping, encode_bytes_to_dna
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv
from video_dna_payload_codec import build_video_config, compress_video_to_payload, reconstruct_video_from_payload
import video_dna_compressor_v21 as v21


VIDEO_STEPS = [
    (1, "Input"),
    (2, "Video Compression"),
    (3, "DNA Encoding"),
    (4, "Strand Design"),
    (5, "Video Reconstruction"),
    (6, "Validation"),
]

VIDEO_MODES = ["High quality AV", "Recommended AV", "Small robust AV"]


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    x = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"


def _display_mapping(mapping: str) -> str:
    return {"Simple Mapping": "SM", "RINF_B16": "RINF"}.get(mapping, mapping)
def _fmt_seconds(value: Any) -> str:
    try:
        if value is None or str(value).strip() == "":
            return "—"
        return f"{float(value):.2f}s"
    except Exception:
        return "—"


def _video_duration_seconds(manifest: dict | None) -> float | None:
    """
    Read video duration from all possible manifest locations.
    Prevents Step 6 report from showing '—' when duration exists.
    """
    if not isinstance(manifest, dict):
        return None

    candidates = [
        manifest.get("duration_seconds"),
        manifest.get("duration_sec"),
        manifest.get("duration"),
    ]

    video_info = manifest.get("video_info")
    if isinstance(video_info, dict):
        candidates.extend([
            video_info.get("duration"),
            video_info.get("duration_seconds"),
            video_info.get("duration_sec"),
        ])

    visual = manifest.get("visual")
    if isinstance(visual, dict):
        candidates.extend([
            visual.get("duration"),
            visual.get("duration_seconds"),
            visual.get("duration_sec"),
        ])

    for value in candidates:
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except Exception:
            pass

    # Fallback: estimate duration from frame_count / target_fps
    try:
        frame_count = None
        target_fps = None

        if isinstance(visual, dict):
            frame_count = (
                visual.get("frame_count")
                or visual.get("num_frames")
                or visual.get("frames")
            )

        target_fps = (
            manifest.get("target_fps")
            or manifest.get("fps")
        )

        if target_fps is None and isinstance(video_info, dict):
            target_fps = (
                video_info.get("target_fps")
                or video_info.get("fps")
            )

        if frame_count is not None and target_fps is not None and float(target_fps) > 0:
            return float(frame_count) / float(target_fps)
    except Exception:
        pass

    return None
def _direct_substitute_dna(dna: str, substitution_rate: float, seed: int) -> Tuple[str, int]:
    """Fast substitution-only DNA noise for skipping slow strand design.

    This keeps the DNA length unchanged, so byte/DNA decoding can still run.
    It is intended for quick video/audio robustness testing when full strand design
    is too slow for multi-million-nt DNA sequences.
    """
    clean = clean_dna(dna)
    if not clean or float(substitution_rate) <= 0.0:
        return clean, 0
    rng = random.Random(int(seed))
    out = list(clean)
    changed = 0
    for i, base in enumerate(out):
        if rng.random() < float(substitution_rate):
            out[i] = rng.choice([b for b in "ACGT" if b != base])
            changed += 1
    return "".join(out), int(changed)



def _step_header(number: int, title: str) -> None:
    st.markdown(
        f"""
<div class="step-heading">
  <span class="step-badge">{number}</span>
  <span class="step-title">{title}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _video_step_state(step_no: int) -> tuple[str, str]:
    checks = {
        1: bool(st.session_state.get("video_input_bytes")),
        2: bool(st.session_state.get("video_payload")),
        3: bool(st.session_state.get("video_dna")),
        4: bool(st.session_state.get("video_strand_rows") or st.session_state.get("video_noisy_dna")),
        5: bool(st.session_state.get("video_reconstructed_result") or st.session_state.get("video_decode_error")),
        6: bool(st.session_state.get("video_validation_metrics") or st.session_state.get("video_decode_error")),
    }
    if checks.get(step_no):
        return "done", "Done"
    previous_done = all(checks.get(i) for i in range(1, step_no)) if step_no > 1 else True
    if previous_done:
        return "current", "Next"
    return "", "Waiting"


def _render_video_stepper() -> None:
    parts = ['<div class="pipeline-steps">']
    for number, label in VIDEO_STEPS:
        css, state = _video_step_state(number)
        parts.append(
            f'<div class="pipeline-step {css}">'
            f'<div><span class="step-num">{number}</span><span class="step-name">{label}</span></div>'
            f'<div class="step-state">{state}</div>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _safe_video_path_from_upload(raw: bytes, name: str) -> str:
    out_dir = WORK_ROOT / "video_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(name or "uploaded.mp4").suffix or ".mp4"
    path = out_dir / f"video_input_{abs(hash((name, len(raw))))}{suffix}"
    path.write_bytes(raw)
    return str(path)


def _clear_video_after_input() -> None:
    for key in [
        "video_payload", "video_payload_manifest", "video_payload_result", "video_preview_path",
        "video_dna", "video_bits", "video_codec_meta", "video_mapping", "video_strand_rows",
        "video_error_rows", "video_noisy_dna", "video_error_stats", "video_uploaded_decode_rows",
        "video_uploaded_decode_dna", "video_reconstructed_result", "video_validation_metrics", "video_decode_error",
    ]:
        st.session_state.pop(key, None)


def _clear_video_after_payload() -> None:
    for key in [
        "video_dna", "video_bits", "video_codec_meta", "video_mapping", "video_strand_rows",
        "video_error_rows", "video_noisy_dna", "video_error_stats", "video_uploaded_decode_rows",
        "video_uploaded_decode_dna", "video_reconstructed_result", "video_validation_metrics", "video_decode_error",
    ]:
        st.session_state.pop(key, None)


def _clear_video_after_dna() -> None:
    for key in [
        "video_strand_rows", "video_error_rows", "video_noisy_dna", "video_error_stats",
        "video_uploaded_decode_rows", "video_uploaded_decode_dna", "video_reconstructed_result",
        "video_validation_metrics", "video_decode_error",
    ]:
        st.session_state.pop(key, None)


def _rows_from_uploaded_strand_csv(uploaded_file) -> List[Dict[str, str]]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str).fillna("")
    return [{str(k): str(v) for k, v in row.items()} for row in df.to_dict("records")]


def _strand_payload_dna(rows: List[Dict[str, Any]], original_len: int) -> str:
    parts: List[str] = []
    for row in rows:
        use_error = str(row.get("Advanced error source", "")).strip().lower() == "true"
        payload = row.get("Error payload", "") if use_error else ""
        parts.append(clean_dna(payload or row.get("Payload", "")))
    return clean_dna("".join(parts))[:int(original_len)]


# -----------------------------------------------------------------------------
# Strand rendering/error helpers, same structure as Audio/Image style
# -----------------------------------------------------------------------------

_VIDEO_REGION_COLORS = {
    "FBR": ("#dbeafe", "#1e3a8a"),
    "SI": ("#e0e7ff", "#3730a3"),
    "Payload": ("#dcfce7", "#166534"),
    "Filler": ("#fef3c7", "#92400e"),
    "RBR": ("#fee2e2", "#991b1b"),
    "Error": ("#fecaca", "#7f1d1d"),
}


def _video_row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", row.get("Index", "")))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _video_region_for_position(row: Dict[str, Any], pos1: int) -> str:
    cursor = 1
    for name, seq in _video_row_regions(row):
        end = cursor + len(seq) - 1
        if cursor <= int(pos1) <= end:
            return name
        cursor = end + 1
    return "Outside"


def _video_region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    bg, fg = _VIDEO_REGION_COLORS.get(name, ("#f8fafc", "#0f172a"))
    ebg, efg = _VIDEO_REGION_COLORS["Error"]
    marks = error_positions or set()
    chars = []
    for pos, ch in enumerate(clean_dna(seq), start=start_pos):
        if pos in marks:
            chars.append(
                f'<span style="background:{ebg};color:{efg};border-radius:3px;padding:0 1px;font-weight:700;">{ch}</span>'
            )
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span class="region-tag" style="background:{bg};color:{fg};display:inline-block;'
        'margin:2px 4px 2px 0;padding:4px 6px;border-radius:6px;font-family:monospace;word-break:break-all;">'
        f"<b>{name}</b>: {body}</span>"
    )


def _render_video_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    parts = []
    cursor = 1
    for name, seq in _video_row_regions(row):
        parts.append(_video_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(seq)
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _mutate_video_prepared_strand(
    row: Dict[str, Any],
    *,
    scope: str,
    substitution_rate: float,
    insertion_rate: float,
    deletion_rate: float,
    seed: int,
    allow_indels: bool,
) -> Dict[str, Any]:
    rng = random.Random(str(seed))
    full = clean_dna(row.get("Full strand", "")) or "".join(seq for _, seq in _video_row_regions(row))
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
        region = _video_region_for_position(row, pos)
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

    if ins_count == 0 and del_count == 0:
        cursor = 0
        mutated_regions: Dict[str, str] = {}
        for name, seq in _video_row_regions(row):
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


def _video_strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Type", "Total length", "Payload length", "Filler length", "GC content", "Longest homopolymer"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows[:50]])


def _video_error_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Advanced error scope", "Total length", "Error full length", "Substitution count", "Insertion count", "Deletion count", "Error count"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _video_error_events_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
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
# Main Streamlit panel
# -----------------------------------------------------------------------------


def render_video_dna_storage_panel() -> None:
    # Step status is rendered by app.py for the unified four-tab UI.

    # 1. Input
    with st.container(border=True):
        _step_header(1, "Input")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### 📁 Input")
            uploaded = st.file_uploader(
                "Input video file",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="video_dna_upload",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                raw = uploaded.getvalue()
                signature = f"{uploaded.name}|{len(raw)}"
                if st.session_state.get("video_input_signature") != signature:
                    st.session_state["video_input_signature"] = signature
                    st.session_state["video_input_name"] = uploaded.name
                    st.session_state["video_input_bytes"] = raw
                    st.session_state["video_input_path"] = _safe_video_path_from_upload(raw, uploaded.name)
                    _clear_video_after_input()

            raw = st.session_state.get("video_input_bytes")
            name = st.session_state.get("video_input_name", "—")
            if raw:
                st.markdown("##### 📄 File properties")
                c1, c2 = st.columns(2)
                c1.metric("Input file", name)
                c2.metric("Uploaded size", _fmt_bytes(len(raw)))
                try:
                    info = v21.ffprobe_video(st.session_state["video_input_path"])
                    has_audio = v21.has_audio_stream(st.session_state["video_input_path"])
                    if info:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Resolution", f"{info.get('width', 0)}×{info.get('height', 0)}")
                        dur = info.get("duration")
                        m2.metric("Duration", f"{float(dur):.2f}s" if dur else "—")
                        m3.metric("FPS", f"{float(info.get('fps') or 0):.2f}")
                        m4.metric("Audio track", "Yes" if has_audio else "No")
                except Exception as exc:
                    st.warning(f"Could not read video metadata. Check FFmpeg/ffprobe. Details: {exc}")
                st.caption("Preview is shown on the right. Compression/method selection is handled in Step 2.")
            else:
                st.info("Upload a video file to start.")

        with right:
            st.markdown("#### 🎬 Preview")
            raw = st.session_state.get("video_input_bytes")
            if raw:
                st.video(raw)
            else:
                st.info("Video preview will appear here after upload.")

    # 2. Video Compression
    with st.container(border=True):
        _step_header(2, "Compression / Method")
        if not st.session_state.get("video_input_path"):
            st.info("Upload a video first.")
        else:
            mode = st.selectbox("Video mode", VIDEO_MODES, index=1, key="video_mode_select")
            include_audio = st.checkbox("Include DNA-encoded audio track", value=True, key="video_include_audio")
            process_full = st.checkbox("Process full video", value=True, key="video_process_full")
            max_seconds = None
            if not process_full:
                max_seconds = st.number_input("Limit seconds for quick demo", min_value=1.0, max_value=120.0, value=4.0, step=1.0, key="video_limit_seconds")

            st.markdown("#### Fault-tolerant options")
            default_fps = 12.0 if mode != "Small robust AV" else 8.0
            f1, f2, f3, f4 = st.columns(4)
            target_fps = f1.number_input("Target FPS", min_value=1.0, max_value=30.0, value=float(default_fps), step=1.0, key="video_target_fps")
            temporal_window = f2.number_input("Temporal repair window (seconds)", min_value=0.10, max_value=2.00, value=0.50, step=0.05, format="%.2f", key="video_temporal_window_sec")
            repair_strength = f3.selectbox("Frame repair", ["Off", "Light", "Balanced", "Strong", "Very strong"], index=2, key="video_repair_strength")
            protection_mode = f4.selectbox("Visual protection", ["None", "Light keyframes", "Strong all frames"], index=0, key="video_protection_mode")
            keyframe_interval_sec = 1.0
            if protection_mode == "Light keyframes":
                keyframe_interval_sec = st.number_input("Keyframe protection interval (seconds)", min_value=0.25, max_value=5.0, value=1.0, step=0.25, key="video_keyframe_interval_sec")
            repair_radius_preview = max(1, int(round(float(target_fps) * float(temporal_window))))
            st.caption(
                f"Repair radius will be ±{repair_radius_preview} frames. "
                "Protection adds redundant Y/LL copies before DNA encoding and majority-votes them during reconstruction."
            )
            st.caption("DNA errors are not applied during compression; use Quick substitution after DNA Encoding or Strand Design in Step 4.")

            if st.button("Run Compression", key="run_video_compression", type="primary"):
                try:
                    out_dir = WORK_ROOT / "video_dna_v21_streamlit" / str(int(time.time() * 1000))
                    cfg = build_video_config(
                        output_dir=str(out_dir),
                        mode=mode,
                        include_audio=bool(include_audio),
                        process_full_video=bool(process_full),
                        max_seconds=max_seconds,
                        substitution_rate=0.0,
                        random_seed=42,
                        target_fps=float(target_fps),
                        temporal_repair_window_sec=float(temporal_window),
                        repair_strength=str(repair_strength),
                        protection_mode=str(protection_mode),
                        keyframe_interval_sec=float(keyframe_interval_sec),
                    )
                    result = compress_video_to_payload(st.session_state["video_input_path"], cfg)
                    st.session_state["video_payload"] = result["payload_bytes"]
                    st.session_state["video_payload_manifest"] = result["manifest"]
                    st.session_state["video_payload_result"] = result
                    st.session_state["video_preview_path"] = result.get("preview_video")
                    _clear_video_after_payload()
                except Exception as exc:
                    st.error(f"Video compression failed: {exc}")

            payload = st.session_state.get("video_payload")
            manifest = st.session_state.get("video_payload_manifest", {})
            if payload:
                uploaded_size = len(st.session_state.get("video_input_bytes", b""))
                raw_rgb = int(manifest.get("raw_rgb_bytes", 0))
                ratio_upload = uploaded_size / max(1, len(payload))
                ratio_raw = raw_rgb / max(1, len(payload))
                visual = manifest.get("visual", {})
                audio = manifest.get("audio", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Compressed payload", _fmt_bytes(len(payload)))
                m2.metric("Compression vs uploaded file", f"{ratio_upload:.2f}x")
                m3.metric("Compression vs raw RGB", f"{ratio_raw:.2f}x")
                m4.metric("Frames", f"{int(visual.get('frame_count', 0)):,}")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Target FPS", f"{float(manifest.get('target_fps', 0.0)):.2f}")
                r2.metric("Resolution", visual.get("resolution", "—"))
                r3.metric("Visual payload", _fmt_bytes(int(visual.get("payload_bytes", 0))))
                r4.metric("Audio payload", _fmt_bytes(int(audio.get("payload_bytes", 0))))
                prot = manifest.get("protection", {}) or {}
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Protection", prot.get("mode", "None"))
                p2.metric("Protected frames", f"{int(prot.get('protected_frames', 0)):,}")
                p3.metric("Protection overhead", _fmt_bytes(int(prot.get("payload_bytes", 0))))
                p4.metric("Repair radius", f"±{int((manifest.get('config', {}) or {}).get('repair_radius', 0))} frames")
                preview = st.session_state.get("video_preview_path")
                if preview and Path(preview).exists():
                    vp1, vp2 = st.columns([1, 1])
                    with vp1:
                        st.video(str(preview))
                st.download_button(
                    "Download video manifest",
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
                    "video_payload_manifest.json",
                    "application/json",
                )

    # 3. DNA Encoding
    with st.container(border=True):
        _step_header(3, "DNA Encoding")
        payload = st.session_state.get("video_payload")
        if not payload:
            st.info("Run Video Compression first.")
        else:
            options = [m for m in MAPPING_OPTIONS if m in {"Simple Mapping", "RINF_B16"}]
            if not options:
                options = ["Simple Mapping", "RINF_B16"]
            mapping = st.selectbox("Mapping rule", options, format_func=_display_mapping, key="video_mapping_select")
            if st.button("Run DNA Encoding", key="run_video_dna_encoding", type="primary"):
                try:
                    dna, bits, codec_meta = encode_bytes_to_dna(payload, mapping)
                    st.session_state["video_mapping"] = mapping
                    st.session_state["video_dna"] = dna
                    st.session_state["video_bits"] = bits
                    st.session_state["video_codec_meta"] = codec_meta
                    _clear_video_after_dna()
                except Exception as exc:
                    st.error(f"DNA encoding failed: {exc}")

            dna = st.session_state.get("video_dna")
            if dna:
                stats = homopolymer_stats(dna)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mapping rule", _display_mapping(st.session_state.get("video_mapping", mapping)))
                m2.metric("DNA length", f"{len(dna):,} nt")
                m3.metric("GC content", f"{gc_content(dna):.3f}")
                m4.metric("Longest homopolymer", int(stats.get("longest", 0)))
                with st.expander("DNA preview", expanded=False):
                    st.text_area("Encoded DNA", dna[:3000] + ("..." if len(dna) > 3000 else ""), height=120)
                    st.download_button("Download encoded DNA", dna.encode("utf-8"), "video_encoded_dna.txt", "text/plain")

                st.markdown("##### 🧪 Payload-level noise")
                show_payload_noise = st.checkbox(
                    "Advanced: add noise directly to encoded DNA payload",
                    value=False,
                    key="video_show_payload_noise",
                )
                if show_payload_noise:
                    q1, q2 = st.columns(2)
                    direct_sub_rate = q1.number_input("Direct substitution rate", min_value=0.0, max_value=0.20, value=0.0050, step=0.001, format="%.4f", key="video_direct_sub_rate")
                    direct_seed = q2.number_input("Quick seed", min_value=0, max_value=999999, value=23, step=1, key="video_direct_sub_seed")
                    st.caption("Fast substitution-only test on the full encoded DNA. This bypasses strand design.")
                    if st.button("Add direct substitutions", key="video_add_direct_substitutions", type="primary"):
                        noisy_dna, n_sub = _direct_substitute_dna(dna, float(direct_sub_rate), int(direct_seed))
                        st.session_state["video_noisy_dna"] = noisy_dna
                        st.session_state["video_direct_noisy_dna"] = noisy_dna
                        st.session_state["video_error_stats"] = {
                            "error_target": "Direct encoded DNA",
                            "substitution_rate": float(direct_sub_rate),
                            "insertion_rate": 0.0,
                            "deletion_rate": 0.0,
                            "seed": int(direct_seed),
                            "total_errors": int(n_sub),
                            "noisy_dna_len": int(len(noisy_dna)),
                            "Substitute count": int(n_sub),
                            "quick_skip_strand": True,
                        }
                        for key in ["video_reconstructed_result", "video_validation_metrics", "video_decode_error"]:
                            st.session_state.pop(key, None)
                    if st.session_state.get("video_noisy_dna") and st.session_state.get("video_error_stats", {}).get("quick_skip_strand"):
                        qs = st.session_state.get("video_error_stats", {})
                        q1, q2, q3 = st.columns(3)
                        q1.metric("Direct substitutions", f"{int(qs.get('Substitute count', 0)):,}")
                        q2.metric("Noisy encoded data", f"{int(qs.get('noisy_dna_len', 0)):,} nt")
                        q3.metric("Skip Strand Design", "Yes")

    # 4. Strand Design / DNA Error Simulation
    with st.container(border=True):
        _step_header(4, "Strand Design")
        dna = st.session_state.get("video_dna")
        if not dna:
            st.info("Run DNA Encoding first.")
        else:
            # st.markdown("#### Strand Design")
            with st.expander("Strand design", expanded=not bool(st.session_state.get("video_strand_rows"))):
                c1, c2 = st.columns(2)
                with c1:
                    target_len = st.number_input("Total strand length", min_value=80, max_value=250, value=125, step=1, key="video_strand_total_len")
                    index_len = st.number_input("Strand ID length", min_value=0, max_value=24, value=8, step=1, key="video_strand_index_len")
                with c2:
                    fbr = st.text_input("Forward primer", value="ACACGACGCTCTTCCGATCT", key="video_fbr")
                    rbr = st.text_input("Reverse primer", value="AGATCGGAAGAGCACACGTCT", key="video_rbr")

                if st.button("Run Strand Design", key="run_video_strand_design", type="primary"):
                    auto = choose_auto_strand_design(
                        len(dna),
                        fbr_len=len(clean_dna(fbr)),
                        rbr_len=len(clean_dna(rbr)),
                        index_len=int(index_len),
                        min_total_len=int(target_len),
                        max_total_len=int(target_len),
                    )
                    rows = prepare_dna_strands(
                        dna,
                        fbr=clean_dna(fbr),
                        rbr=clean_dna(rbr),
                        index_len=int(index_len),
                        target_total_len=int(auto["target_total_len"]),
                        add_filler=True,
                    )
                    for row in rows:
                        row["Type"] = "Designed strand"
                    st.session_state["video_strand_rows"] = rows
                    for key in ["video_error_rows", "video_noisy_dna", "video_error_stats", "video_reconstructed_result", "video_validation_metrics", "video_decode_error"]:
                        st.session_state.pop(key, None)

            rows = st.session_state.get("video_strand_rows", [])
            if not rows:
                st.info("Run Strand Design first.")
            else:
                total_strand_len = sum(len(clean_dna(row.get("Full strand", ""))) for row in rows)
                strand_expansion = total_strand_len / max(1, len(clean_dna(dna)))
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Designed strands", f"{len(rows):,}")
                s2.metric("Total strand length", f"{total_strand_len:,} nt")
                s3.metric("Strand Design expansion", f"{strand_expansion:.2f}×")
                s4.metric("DNA mapping", _display_mapping(st.session_state.get("video_mapping", "")))
                st.dataframe(_video_strand_summary(rows), use_container_width=True, hide_index=True)
                inspect_ids = [str(row.get("No.", i + 1)) for i, row in enumerate(rows)]
                selected_no = st.selectbox("Inspect designed strand", inspect_ids, index=0, key="video_inspect_designed_strand")
                selected_row = next((row for row in rows if str(row.get("No.", "")) == selected_no), rows[0])
                _render_video_segmented_strand(selected_row, "Designed strand")
                st.download_button("Download prepared strands", strand_rows_to_csv(rows), "video_prepared_strands.csv", "text/csv")

            st.markdown("##### 🧪 Strand-level noise")
            st.caption("Advanced: add noise to prepared small strands. Use this after Strand Design to simulate errors on app-generated strands.")
            if not rows:
                st.info("Run Strand Design first.")
            else:
                with st.container(border=True):
                    e1, e2, e3, e4 = st.columns(4)
                    error_target = e1.selectbox("Error target", ["Payload only", "Index + Payload", "Full strand"], index=0, key="video_error_target")
                    substitution_rate = e2.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.0200, step=0.001, format="%.4f", key="video_sub_rate")
                    insertion_rate = e3.number_input("Insertion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="video_ins_rate")
                    deletion_rate = e4.number_input("Deletion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="video_del_rate")
                    seed = st.number_input("Seed", min_value=0, max_value=999999, value=17, step=1, key="video_error_seed")

                    if st.button("Add errors", key="run_video_error_simulation", type="primary"):
                        allow_indels = bool(float(insertion_rate) > 0.0 or float(deletion_rate) > 0.0)
                        err_rows = []
                        for row in rows:
                            try:
                                row_no = int(str(row.get("No.", "0") or "0"))
                            except Exception:
                                row_no = len(err_rows) + 1
                            err_rows.append(_mutate_video_prepared_strand(
                                row,
                                scope=str(error_target),
                                substitution_rate=float(substitution_rate),
                                insertion_rate=float(insertion_rate),
                                deletion_rate=float(deletion_rate),
                                seed=int(seed) + row_no * 1000003,
                                allow_indels=allow_indels,
                            ))
                        events = _video_error_events_table(err_rows)
                        noisy_dna = _strand_payload_dna(err_rows, len(clean_dna(dna)))
                        st.session_state["video_error_rows"] = err_rows
                        st.session_state["video_noisy_dna"] = noisy_dna
                        st.session_state["video_error_stats"] = {
                            "error_target": error_target,
                            "substitution_rate": float(substitution_rate),
                            "insertion_rate": float(insertion_rate),
                            "deletion_rate": float(deletion_rate),
                            "seed": int(seed),
                            "total_errors": int(len(events)),
                            "noisy_dna_len": int(len(noisy_dna)),
                            "Substitute count": int(sum(int(str(row.get("Substitution count", "0") or "0")) for row in err_rows)),
                        }
                        for key in ["video_reconstructed_result", "video_validation_metrics", "video_decode_error"]:
                            st.session_state.pop(key, None)

                err_rows = st.session_state.get("video_error_rows", [])
                noisy_dna = st.session_state.get("video_noisy_dna", "")
                if err_rows:
                    events = _video_error_events_table(err_rows)
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Error strands", f"{len(err_rows):,}")
                    a2.metric("Added errors", f"{len(events):,}")
                    a3.metric("Noisy encoded data", f"{len(noisy_dna):,} nt")
                    st.dataframe(_video_error_rows_table(err_rows), use_container_width=True, hide_index=True)
                    eids = [str(row.get("No.", i + 1)) for i, row in enumerate(err_rows)]
                    eno = st.selectbox("Inspect error strand", eids, index=0, key="video_inspect_error_strand")
                    erow = next((row for row in err_rows if str(row.get("No.", "")) == eno), err_rows[0])
                    clean_for_error = next((row for row in rows if str(row.get("No.", "")) == eno), rows[0])
                    try:
                        ev_list = json.loads(erow.get("Advanced error events", "[]") or "[]")
                    except Exception:
                        ev_list = []
                    err_positions = {
                        int(ev.get("position_original"))
                        for ev in ev_list
                        if ev.get("operation") in {"substitution", "deletion"} and str(ev.get("position_original", "")).isdigit()
                    }
                    _render_video_segmented_strand(clean_for_error, "Clean strand", error_positions=err_positions)
                    _render_video_segmented_strand(erow, "Error strand", error_positions=err_positions)
                    if not events.empty:
                        st.dataframe(events, use_container_width=True, hide_index=True)
                    st.download_button("Download error strands", strand_rows_to_csv(err_rows), "video_error_strands.csv", "text/csv")
                    st.download_button("Download noisy encoded DNA", noisy_dna.encode("utf-8"), "video_noisy_encoded_dna.txt", "text/plain")

            with st.expander("Advanced sequencing simulation", expanded=False):
                st.info("Reserved for sequencing/read simulation. Video reconstruction currently decodes from clean, error, or uploaded strand payloads.")

    # 5. Video Reconstruction
    with st.container(border=True):
        _step_header(5, "Decode / Repair")
        if not st.session_state.get("video_dna"):
            st.info("Run DNA Encoding first.")
        else:
            uploaded_dna_txt = st.file_uploader("Upload full DNA TXT generated by this app", type=["txt"], key="video_decode_upload_dna_txt")
            if uploaded_dna_txt is not None:
                try:
                    st.session_state["video_uploaded_decode_dna_txt"] = clean_dna(uploaded_dna_txt.getvalue().decode("utf-8", errors="ignore"))
                    st.success("Loaded uploaded full DNA TXT for reconstruction.")
                except Exception as exc:
                    st.error(f"Could not load DNA TXT: {exc}")

            uploaded_strands = st.file_uploader("Upload prepared/error strands CSV generated by this app", type=["csv"], key="video_decode_upload_strands_csv")
            if uploaded_strands is not None:
                try:
                    uploaded_rows = _rows_from_uploaded_strand_csv(uploaded_strands)
                    uploaded_dna = _strand_payload_dna(uploaded_rows, len(st.session_state.get("video_dna", "")))
                    st.session_state["video_uploaded_decode_rows"] = uploaded_rows
                    st.session_state["video_uploaded_decode_dna"] = uploaded_dna
                    st.success(f"Loaded {len(uploaded_rows):,} uploaded strands for reconstruction.")
                except Exception as exc:
                    st.error(f"Could not load strands CSV: {exc}")

            decode_choices = ["Current encoded DNA", "Noisy encoded DNA", "Upload full DNA TXT", "Upload prepared/error strands CSV"]
            selected_decode_source = st.radio("Reconstruction source", decode_choices, horizontal=True, key="video_reconstruction_source")
            if selected_decode_source == "Upload prepared/error strands CSV":
                selected_input_dna = st.session_state.get("video_uploaded_decode_dna", "")
            elif selected_decode_source == "Upload full DNA TXT":
                selected_input_dna = st.session_state.get("video_uploaded_decode_dna_txt", "")
            elif selected_decode_source == "Noisy encoded DNA":
                selected_input_dna = st.session_state.get("video_noisy_dna", "")
            else:
                selected_input_dna = st.session_state.get("video_dna", "")
            if not selected_input_dna:
                st.info("No DNA is available for the selected reconstruction source.")

            if st.button("Run Decode", key="run_video_reconstruction", type="primary", disabled=not bool(selected_input_dna)):
                mapping = st.session_state.get("video_mapping", "Simple Mapping")
                codec_meta = st.session_state.get("video_codec_meta", {})
                manifest = st.session_state.get("video_payload_manifest", {})
                try:
                    start = time.perf_counter()
                    decoded_payload, _decoded_bits, decoded_meta = decode_dna_with_mapping(selected_input_dna, mapping, codec_meta)
                    out_dir = WORK_ROOT / "video_reconstruction" / str(int(time.time() * 1000))
                    result = reconstruct_video_from_payload(decoded_payload, manifest, out_dir)
                    reconstruction_time = float(time.perf_counter() - start)
                    original_payload = st.session_state.get("video_payload", b"")
                    byte_errors = (
                        sum(a != b for a, b in zip(original_payload, decoded_payload))
                        + abs(len(original_payload) - len(decoded_payload))
                    )
                    metrics = dict(result.get("metrics", {}))
                    metrics.update({
                        "decoded_bytes": len(decoded_payload),
                        "byte_exact": decoded_payload == original_payload,
                        "byte_error_rate": byte_errors / max(1, len(original_payload)),
                        "reconstruction_source": selected_decode_source,
                        "decode_meta": decoded_meta,
                        "reconstruction_time_seconds": reconstruction_time,
                    })
                    st.session_state["video_reconstructed_result"] = result
                    st.session_state["video_validation_metrics"] = metrics
                    st.session_state["video_decode_error"] = ""
                except Exception as exc:
                    st.session_state["video_reconstructed_result"] = {}
                    st.session_state["video_validation_metrics"] = {}
                    st.session_state["video_decode_error"] = str(exc)

            if st.session_state.get("video_decode_error"):
                st.warning(f"Video reconstruction warning: {st.session_state['video_decode_error']}")
            result = st.session_state.get("video_reconstructed_result", {}) or {}
            metrics = st.session_state.get("video_validation_metrics", {}) or {}
            final_video = result.get("final_video")
            if final_video and Path(final_video).exists():
                e = st.session_state.get("video_error_stats", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Added errors", f"{int(e.get('total_errors', 0)):,}")
                m2.metric("Byte error", f"{float(metrics.get('byte_error_rate', 0.0)) * 100:.2f}%")
                m3.metric("Video PSNR", f"{float(metrics.get('video_psnr', 0.0)):.2f} dB")
                m4.metric("Video SSIM", f"{float(metrics.get('video_ssim', 0.0)):.4f}")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Frame correlation", f"{float(metrics.get('frame_correlation', 0.0)):.4f}")
                r2.metric("Frames compared", f"{int(metrics.get('frames_compared', 0)):,}")
                r3.metric("Playable", "Yes" if metrics.get("playable") else "No")
                r4.metric("Reconstruction time", f"{float(metrics.get('reconstruction_time_seconds', 0.0)):.2f}s")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Protection", metrics.get("protection_mode", "None"))
                p2.metric("Protected frames used", f"{int(metrics.get('protected_frames_used', 0)):,}")
                p3.metric("Frame repair", st.session_state.get("video_repair_strength", metrics.get("repair_strength", "-")))
                p4.metric("PSNR gain", f"{float(metrics.get('psnr_gain_from_repair', 0.0)):+.2f} dB")
                rv1, rv2 = st.columns([1, 1])
                with rv1:
                    st.video(final_video)
                base_video = result.get("base_video") or metrics.get("base_video")
                if base_video and Path(base_video).exists():
                    with st.expander("Compare before/after frame repair", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Before temporal repair")
                            st.video(base_video)
                        with c2:
                            st.caption("After temporal repair + cleanup")
                            st.video(final_video)
                with open(final_video, "rb") as f:
                    st.download_button("Download recovered video", f.read(), "video_recovered.mp4", "video/mp4")
                metrics_json = result.get("metrics_json")
                if metrics_json and Path(metrics_json).exists():
                    st.download_button("Download reconstruction metrics", Path(metrics_json).read_bytes(), "video_reconstruction_metrics.json", "application/json")

    # 6. Analysis
    with st.container(border=True):
        _step_header(6, "Summarization")
        if not st.session_state.get("video_reconstructed_result") and not st.session_state.get("video_decode_error"):
            st.info("Run Video Reconstruction first.")
        else:
            metrics = st.session_state.get("video_validation_metrics", {}) or {}
            manifest = st.session_state.get("video_payload_manifest", {}) or {}
            visual = manifest.get("visual", {}) or {}
            audio = manifest.get("audio", {}) or {}
            cfg = manifest.get("config", {}) or {}
            protection = manifest.get("protection", {}) or {}
            payload = st.session_state.get("video_payload", b"") or b""
            uploaded_size = len(st.session_state.get("video_input_bytes", b""))
            raw_rgb = int(manifest.get("raw_rgb_bytes", 0))
            dna = st.session_state.get("video_dna", "") or ""
            strand_rows = st.session_state.get("video_strand_rows", []) or []
            err_stats = st.session_state.get("video_error_stats", {}) or {}
            noisy = st.session_state.get("video_noisy_dna", "") or ""
            error_rows = st.session_state.get("video_error_rows", []) or []

            def compact_rows(rows: List[Dict[str, Any]], keep_metrics: List[str]) -> List[Dict[str, Any]]:
                keep = set(keep_metrics)
                out: List[Dict[str, Any]] = []
                seen = set()
                for row in rows or []:
                    metric = str(row.get("Metric", ""))
                    if metric in keep and metric not in seen:
                        out.append(row)
                        seen.add(metric)
                return out

            def analysis_table(title: str, rows: List[Dict[str, Any]]) -> None:
                st.markdown(f"#### {title}")
                clean_rows = []
                for row in rows or []:
                    value = row.get("Value", "—")
                    clean_rows.append({
                        "Property": row.get("Property", row.get("Metric", "—")),
                        "Value": value if value not in (None, "") else "—",
                    })
                st.dataframe(pd.DataFrame(clean_rows), hide_index=True, use_container_width=True)

            st.markdown("#### 📊 Summary")
            original_col, encoded_col, decoded_col = st.columns(3, gap="large")
            with original_col:
                st.markdown("##### Original")
                original_path = st.session_state.get("video_input_path")
                if original_path and Path(original_path).exists():
                    st.video(original_path)
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    
                    {"Property": "Duration", "Value": _fmt_seconds(_video_duration_seconds(manifest))},
                    {"Property": "Resolution", "Value": visual.get("resolution", "—")},
                    {"Property": "Original size", "Value": _fmt_bytes(uploaded_size)},
                ]), hide_index=True, use_container_width=True)
            with encoded_col:
                st.markdown("##### Compressed/Encoded")
                encoded_video = st.session_state.get("video_preview_path")
                if encoded_video and Path(encoded_video).exists():
                    st.video(str(encoded_video))
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    {"Property": "Method", "Value": cfg.get("mode", st.session_state.get("video_mode_select", "—"))},
                    {"Property": "Payload size", "Value": _fmt_bytes(len(payload))},
                    {"Property": "DNA length", "Value": f"{len(dna):,} nt" if dna else "—"},
                ]), hide_index=True, use_container_width=True)
            with decoded_col:
                st.markdown("##### Decoded")
                result = st.session_state.get("video_reconstructed_result", {}) or {}
                final_video = result.get("final_video")
                if final_video and Path(final_video).exists():
                    st.video(final_video)
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    {"Property": "Decode source", "Value": metrics.get("reconstruction_source", "—")},
                    {"Property": "Output status", "Value": "Playable" if metrics.get("playable") else "Failed"},
                    {"Property": "Keyframe SSIM", "Value": f"{float(metrics.get('video_ssim', 0.0)):.4f}" if metrics else "—"},
                ]), hide_index=True, use_container_width=True)

            storage_rows = [
                {"Metric": "Data type", "Value": "Video"},
                {"Metric": "Video mode", "Value": st.session_state.get("video_mode_select", "-")},
                {"Metric": "Method", "Value": cfg.get("mode", st.session_state.get("video_mode_select", "-"))},
                
                {"Metric": "Duration", "Value": _fmt_seconds(_video_duration_seconds(manifest))},
                {"Metric": "Resolution", "Value": visual.get("resolution", "-")},
                {"Metric": "Target FPS", "Value": f"{float(manifest.get('target_fps', 0.0)):.2f}"},
                {"Metric": "Frame count", "Value": f"{int(visual.get('frame_count', 0)):,}"},
                {"Metric": "Audio included", "Value": "Yes" if audio.get("enabled") else "No"},
                {"Metric": "Original size", "Value": _fmt_bytes(uploaded_size)},
                {"Metric": "Payload size", "Value": _fmt_bytes(len(payload))},
                {"Metric": "Compression vs uploaded file", "Value": f"{uploaded_size / max(1, len(payload)):.2f}x"},
                {"Metric": "Compression vs raw RGB", "Value": f"{raw_rgb / max(1, len(payload)):.2f}x" if raw_rgb else "—"},
                {"Metric": "Visual payload", "Value": _fmt_bytes(int(visual.get("payload_bytes", 0)))},
                {"Metric": "Audio payload", "Value": _fmt_bytes(int(audio.get("payload_bytes", 0)))},
                {"Metric": "Estimated DNA length", "Value": f"{len(payload) * 4:,} nt" if payload else "—"},
            ]
            analysis_table("Compression analysis", compact_rows(
                storage_rows,
                ["Data type", "Method", "Duration", "Resolution", "Target FPS", "Audio included", "Original size", "Payload size", "Compression vs uploaded file", "Estimated DNA length"],
            ))

            lengths: List[int] = []
            gc_values: List[float] = []
            hp_values: List[int] = []
            for row in strand_rows:
                full = clean_dna(row.get("Full strand", ""))
                if full:
                    lengths.append(len(full))
                    try:
                        gc_values.append(float(row.get("GC content", gc_content(full))))
                    except Exception:
                        gc_values.append(gc_content(full))
                    try:
                        hp_values.append(int(row.get("Longest homopolymer", homopolymer_stats(full).get("longest", 0))))
                    except Exception:
                        hp_values.append(int(homopolymer_stats(full).get("longest", 0)))
            dna_rows = [
                
                {"Metric": "Mapping rule", "Value": _display_mapping(
    st.session_state.get("video_mapping", st.session_state.get("video_mapping_select", "—"))
)},
                {"Metric": "DNA length", "Value": f"{len(dna):,} nt" if dna else "—"},
                {"Metric": "GC content", "Value": f"{gc_content(dna):.3f}" if dna else "—"},
                {"Metric": "Longest homopolymer", "Value": homopolymer_stats(dna).get("longest", 0) if dna else "—"},
                {"Metric": "Strand count", "Value": f"{len(strand_rows):,}"},
                {"Metric": "Average strand length", "Value": f"{(sum(lengths) / len(lengths)):.1f} nt" if lengths else "—"},
                {"Metric": "Average strand GC", "Value": f"{(sum(gc_values) / len(gc_values)):.4f}" if gc_values else "—"},
                {"Metric": "Max strand homopolymer", "Value": max(hp_values) if hp_values else "—"},
                {"Metric": "Strand architecture", "Value": "FBR + SI + Payload + Filler + RBR" if strand_rows else "—"},
            ]
            analysis_table("Encode-decode analysis", compact_rows(
                dna_rows,
                ["Mapping rule", "DNA length", "GC content", "Longest homopolymer", "Strand count", "Average strand GC", "Max strand homopolymer"],
            ))

            is_noisy = bool(err_stats) or bool(noisy) or bool(error_rows)
            error_rows_summary = [
                {"Metric": "Error status", "Value": "Noisy" if is_noisy else "Clean"},
                {"Metric": "Error level", "Value": "Payload-level" if err_stats.get("quick_skip_strand") else ("Strand-level" if error_rows else "Clean DNA")},
                {"Metric": "Error type", "Value": "Substitution" if is_noisy else "None"},
                {"Metric": "Substitution rate", "Value": err_stats.get("substitution_rate", "0")},
                {"Metric": "Substituted bases", "Value": f"{int(err_stats.get('Substitute count', err_stats.get('total_errors', 0)) or 0):,}"},
                
                {"Metric": "Error scope", "Value": err_stats.get("error_target", "payload") if err_stats else "—"},
                {"Metric": "Affected strands", "Value": f"{len(error_rows):,}" if error_rows else ("—" if err_stats.get("quick_skip_strand") else "0")},
                {"Metric": "Seed", "Value": err_stats.get("seed", "—")},
            ]
            analysis_table("Error Adding Report", compact_rows(
                error_rows_summary,
                ["Error status", "Error level", "Error type", "Substitution rate", "Substituted bases", "Error scope", "Affected strands", "Seed"],
            ))

            playable = bool(metrics.get("playable"))
            byte_exact = bool(metrics.get("byte_exact"))
            recovery_class = "Exact" if byte_exact else ("Usable" if playable else "Failed")
            quality_rows = [
                # {"Metric": "Decode source", "Value": metrics.get("reconstruction_source", "—")},
                {"Metric": "Decode status", "Value": "Success" if playable and not st.session_state.get("video_decode_error") else "Failed"},
                # {"Metric": "Output status", "Value": "Playable" if playable else "Failed"},
                # {"Metric": "Recovery class", "Value": recovery_class},
                {"Metric": "Payload accuracy", "Value": f"{(1.0 - float(metrics.get('byte_error_rate', 0.0))) * 100:.2f}%"},
                # {"Metric": "Checksum", "Value": "Pass" if byte_exact else "Fail"},
                {"Metric": "Keyframe PSNR", "Value": f"{float(metrics.get('video_psnr', 0.0)):.2f} dB"},
                {"Metric": "Keyframe SSIM", "Value": f"{float(metrics.get('video_ssim', 0.0)):.4f}"},
                {"Metric": "Frame correlation", "Value": f"{float(metrics.get('frame_correlation', 0.0)):.4f}"},
                {"Metric": "Duration difference", "Value": f"{float(metrics.get('duration_difference_seconds', 0.0)):.3f}s" if metrics.get('duration_difference_seconds') is not None else "—"},
            ]
            analysis_table("Recovery Quality Report", compact_rows(
                quality_rows,
                ["Decode source", "Decode status", "Output status", "Recovery class", "Payload accuracy", "Checksum", "Keyframe PSNR", "Keyframe SSIM", "Frame correlation", "Duration difference"],
            ))


            method_rows = [
                {"Property": "Video mode", "Value": st.session_state.get("video_mode_select", "—")},
                {"Property": "Frame count", "Value": f"{int(visual.get('frame_count', 0)):,}"},
                {"Property": "Compression vs raw RGB", "Value": f"{raw_rgb / max(1, len(payload)):.2f}x" if raw_rgb and payload else "—"},
                {"Property": "Visual payload", "Value": _fmt_bytes(int(visual.get("payload_bytes", 0)))},
                {"Property": "Audio payload", "Value": _fmt_bytes(int(audio.get("payload_bytes", 0)))},
                {"Property": "Repair mode", "Value": cfg.get("repair_strength", st.session_state.get("video_repair_strength", "—"))},
                {"Property": "Visual protection", "Value": protection.get("mode", "None")},
            ]
            if method_rows:
                with st.expander("Method-specific details", expanded=False):
                    st.dataframe(pd.DataFrame([{"Property": r.get("Property", r.get("Metric", "—")), "Value": r.get("Value", "—")} for r in method_rows]), hide_index=True, use_container_width=True)
