from __future__ import annotations

import io
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import MAPPING_OPTIONS
from dna_codec import gc_content, homopolymer_stats
from dna_mapping import decode_dna_with_mapping, encode_bytes_to_dna
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv


AUDIO_MODES: Dict[str, Dict[str, Any]] = {
    "High quality robust": {"method": "mu-law 8-bit + click repair", "sample_rate": 16000, "bits": 8},
    "Recommended": {"method": "mu-law 4-bit + click repair", "sample_rate": 16000, "bits": 4},
    "Small robust": {"method": "mu-law 8kHz 4-bit + click repair", "sample_rate": 8000, "bits": 4},
    # "Experimental AI cleanup": {
    #     "method": "mu-law 4-bit + click repair + optional AI cleanup",
    #     "sample_rate": 16000,
    #     "bits": 4,
    #     "ai_cleanup": True,
    # },
}

AUDIO_STEPS = [
    (1, "Input"),
    (2, "Audio Compression"),
    (3, "DNA Encoding"),
    (4, "Strand Design"),
    (5, "Audio Reconstruction"),
    (6, "Validation"),
]


@dataclass
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int
    note: str


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


def _audio_step_state(step_no: int) -> tuple[str, str]:
    checks = {
        1: bool(st.session_state.get("audio_input_bytes")),
        2: bool(st.session_state.get("audio_payload")),
        3: bool(st.session_state.get("audio_dna")),
        4: bool(st.session_state.get("audio_strand_rows")),
        5: bool(st.session_state.get("audio_recovered_wav") or st.session_state.get("audio_decode_error")),
        6: bool(st.session_state.get("audio_recovery_metrics") or st.session_state.get("audio_decode_error")),
    }
    if checks.get(step_no):
        return "done", "Done"
    previous_done = all(checks.get(i) for i in range(1, step_no)) if step_no > 1 else True
    if previous_done:
        return "current", "Next"
    return "", "Waiting"


def _render_audio_stepper() -> None:
    parts = ['<div class="pipeline-steps">']
    for number, label in AUDIO_STEPS:
        css, state = _audio_step_state(number)
        parts.append(
            f'<div class="pipeline-step {css}">'
            f'<div><span class="step-num">{number}</span><span class="step-name">{label}</span></div>'
            f'<div class="step-state">{state}</div>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024.0 or unit == "GB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} GB"


def _display_mapping(mapping: str) -> str:
    return {
        "Simple Mapping": "Simple Mapping",
        "RINF_B16": "Rinf",
        "R0_B9": "R0",
        "R1_B12": "R1",
        "R2_B15": "R2",
        "New Design": "New Design",
    }.get(mapping, mapping)


def _clear_audio_after_input() -> None:
    for key in [
        "audio_payload",
        "audio_payload_meta",
        "audio_original_wav",
        "audio_dna",
        "audio_bits",
        "audio_codec_meta",
        "audio_mapping",
        "audio_strand_rows",
        "audio_advanced_error_rows",
        "audio_noisy_dna",
        "audio_error_stats",
        "audio_uploaded_decode_rows",
        "audio_uploaded_decode_dna",
        "audio_recovered_wav",
        "audio_recovery_metrics",
        "audio_decode_error",
    ]:
        st.session_state.pop(key, None)


def _clear_audio_after_payload() -> None:
    for key in [
        "audio_dna",
        "audio_bits",
        "audio_codec_meta",
        "audio_mapping",
        "audio_strand_rows",
        "audio_advanced_error_rows",
        "audio_noisy_dna",
        "audio_error_stats",
        "audio_uploaded_decode_rows",
        "audio_uploaded_decode_dna",
        "audio_recovered_wav",
        "audio_recovery_metrics",
        "audio_decode_error",
    ]:
        st.session_state.pop(key, None)


def _clear_audio_after_dna() -> None:
    for key in [
        "audio_strand_rows",
        "audio_advanced_error_rows",
        "audio_noisy_dna",
        "audio_error_stats",
        "audio_uploaded_decode_rows",
        "audio_uploaded_decode_dna",
        "audio_recovered_wav",
        "audio_recovery_metrics",
        "audio_decode_error",
    ]:
        st.session_state.pop(key, None)


def _clip_audio(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def _load_audio_bytes(raw: bytes) -> LoadedAudio:
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
        return LoadedAudio(_clip_audio(data.mean(axis=1)), int(sr), "Loaded with soundfile.")
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            width = int(wf.getsampwidth())
            frames = wf.readframes(wf.getnframes())
    except Exception as exc:
        raise RuntimeError("Audio tab can read WAV directly. Install soundfile to read FLAC/OGG/MP3 when supported.") from exc

    if width == 1:
        arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {width} bytes.")

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return LoadedAudio(_clip_audio(arr), sr, "Loaded as WAV.")


def _resample_linear(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = _clip_audio(samples)
    if int(sr_in) == int(sr_out):
        return x.copy()
    duration = len(x) / max(1, int(sr_in))
    out_len = max(1, int(round(duration * int(sr_out))))
    src_t = np.linspace(0.0, duration, num=len(x), endpoint=False)
    dst_t = np.linspace(0.0, duration, num=out_len, endpoint=False)
    return _clip_audio(np.interp(dst_t, src_t, x))


def _limit_seconds(samples: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    keep = max(1, int(round(float(seconds) * int(sample_rate))))
    return _clip_audio(samples[:keep])


def _mulaw_encode(samples: np.ndarray, bits: int) -> np.ndarray:
    x = _clip_audio(samples)
    mu = float((1 << int(bits)) - 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / math.log1p(mu)
    q = np.round((y + 1.0) * 0.5 * mu)
    return np.clip(q, 0, int(mu)).astype(np.uint8)


def _mulaw_decode(q: np.ndarray, bits: int) -> np.ndarray:
    mu = float((1 << int(bits)) - 1)
    y = (np.asarray(q, dtype=np.float32) / mu) * 2.0 - 1.0
    x = np.sign(y) * np.expm1(np.abs(y) * math.log1p(mu)) / mu
    return _clip_audio(x)


def _pack_payload(q: np.ndarray, bits: int) -> bytes:
    q = np.asarray(q, dtype=np.uint8).reshape(-1)
    if int(bits) == 8:
        return q.tobytes()
    if int(bits) == 4:
        if q.size % 2:
            q = np.append(q, 0).astype(np.uint8)
        packed = ((q[0::2] & 0x0F) << 4) | (q[1::2] & 0x0F)
        return packed.astype(np.uint8).tobytes()
    raise ValueError("Only 8-bit and 4-bit mu-law are supported.")


def _unpack_payload(payload: bytes, bits: int, expected_samples: int) -> np.ndarray:
    raw = np.frombuffer(bytes(payload or b""), dtype=np.uint8)
    if int(bits) == 8:
        out = raw
    elif int(bits) == 4:
        out = np.empty(raw.size * 2, dtype=np.uint8)
        out[0::2] = (raw >> 4) & 0x0F
        out[1::2] = raw & 0x0F
    else:
        raise ValueError("Only 8-bit and 4-bit mu-law are supported.")

    if out.size < expected_samples:
        out = np.pad(out, (0, expected_samples - out.size))
    return out[:expected_samples].astype(np.uint8)


def _click_repair(samples: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    y = _clip_audio(samples).copy()
    if y.size < 5:
        return y
    med = y.copy()
    med[2:-2] = np.median(np.vstack([y[:-4], y[1:-3], y[2:-2], y[3:-1], y[4:]]), axis=0)
    mask = np.abs(y - med) > float(threshold)
    y[mask] = med[mask]
    return _clip_audio(y)


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(_clip_audio(samples) * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _snr_db(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    if n <= 0:
        return float("nan")
    ref = _clip_audio(original[:n])
    rec = _clip_audio(recovered[:n])
    noise = ref - rec
    return float(10.0 * np.log10((np.mean(ref * ref) + 1e-12) / (np.mean(noise * noise) + 1e-12)))


def _mse(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    if n <= 0:
        return float("nan")
    d = _clip_audio(original[:n]) - _clip_audio(recovered[:n])
    return float(np.mean(d * d))


def _psnr_db(original: np.ndarray, recovered: np.ndarray) -> float:
    mse = _mse(original, recovered)
    if not np.isfinite(mse) or mse <= 1e-12:
        return float("inf")
    # Audio samples are normalized to [-1, 1], so peak amplitude is 1.0.
    return float(10.0 * np.log10(1.0 / mse))


def _waveform_correlation(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    if n <= 1:
        return float("nan")
    ref = _clip_audio(original[:n])
    rec = _clip_audio(recovered[:n])
    ref = ref - float(np.mean(ref))
    rec = rec - float(np.mean(rec))
    denom = float(np.sqrt(np.sum(ref * ref) * np.sum(rec * rec)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(ref * rec) / denom)


def _spectrogram_similarity(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    if n <= 512:
        return float("nan")
    ref = _clip_audio(original[:n])
    rec = _clip_audio(recovered[:n])
    frame = 512
    hop = 256
    win = np.hanning(frame).astype(np.float32)
    count = 1 + max(0, (n - frame) // hop)
    ref_specs = []
    rec_specs = []
    for i in range(count):
        start = i * hop
        a = ref[start:start + frame]
        b = rec[start:start + frame]
        if len(a) < frame or len(b) < frame:
            break
        ref_specs.append(np.log1p(np.abs(np.fft.rfft(a * win))))
        rec_specs.append(np.log1p(np.abs(np.fft.rfft(b * win))))
    if not ref_specs or not rec_specs:
        return float("nan")
    s1 = np.asarray(ref_specs, dtype=np.float32).reshape(-1)
    s2 = np.asarray(rec_specs, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(s1) * np.linalg.norm(s2))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(s1, s2) / denom)


def _maybe_ai_cleanup(samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, str]:
    try:
        from speechbrain.inference.separation import SepformerSeparation  # type: ignore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wav_path = tmpdir / "input.wav"
            wav_path.write_bytes(_wav_bytes(samples, sample_rate))
            model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-dns4-16k-enhancement",
                savedir=str(tmpdir / "speechbrain_model"),
                run_opts={"device": "cpu"},
            )
            enhanced = model.separate_file(path=str(wav_path))
            arr = enhanced.detach().cpu().numpy().reshape(-1).astype(np.float32)
            return _clip_audio(arr), "SpeechBrain SepFormer"
    except Exception as exc:
        return samples, f"not applied: {type(exc).__name__}"


def _payload_from_audio(raw: bytes, mode_name: str) -> Tuple[bytes, bytes, Dict[str, Any], np.ndarray]:
    loaded = _load_audio_bytes(raw)
    mode = AUDIO_MODES[mode_name]
    sample_rate = int(mode["sample_rate"])
    bits = int(mode["bits"])
    original = _resample_linear(loaded.samples, loaded.sample_rate, sample_rate)
    q = _mulaw_encode(original, bits)
    payload = _pack_payload(q, bits)
    meta = {
        "mode": mode_name,
        "method": mode["method"],
        "source_note": loaded.note,
        "uploaded_file_bytes": int(len(raw)),
        "source_sample_rate": int(loaded.sample_rate),
        "sample_rate": sample_rate,
        "bits": bits,
        "sample_count": int(len(original)),
        "duration_seconds": float(len(original) / sample_rate),
        "ai_cleanup": bool(mode.get("ai_cleanup")),
    }
    return payload, _wav_bytes(original, sample_rate), meta, original


def _reconstruct_audio(decoded_payload: bytes, meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], np.ndarray]:
    bits = int(meta["bits"])
    sample_count = int(meta["sample_count"])
    sample_rate = int(meta["sample_rate"])
    q = _unpack_payload(decoded_payload, bits, sample_count)
    recovered = _mulaw_decode(q, bits)
    recovered = _click_repair(recovered)
    ai_status = ""
    if meta.get("ai_cleanup"):
        recovered, ai_status = _maybe_ai_cleanup(recovered, sample_rate)
    return _wav_bytes(recovered, sample_rate), {"ai_cleanup": ai_status}, recovered


_AUDIO_REGION_COLORS = {
    "FBR": ("#dbeafe", "#1e3a8a"),
    "SI": ("#e0e7ff", "#3730a3"),
    "Payload": ("#dcfce7", "#166534"),
    "Filler": ("#fef3c7", "#92400e"),
    "RBR": ("#fee2e2", "#991b1b"),
    "Error": ("#fecaca", "#7f1d1d"),
}


def _audio_row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", row.get("Index", "")))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _audio_region_for_position(row: Dict[str, Any], pos1: int) -> str:
    cursor = 1
    for name, seq in _audio_row_regions(row):
        end = cursor + len(seq) - 1
        if cursor <= int(pos1) <= end:
            return name
        cursor = end + 1
    return "Outside"


def _audio_region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    bg, fg = _AUDIO_REGION_COLORS.get(name, ("#f8fafc", "#0f172a"))
    error_bg, error_fg = _AUDIO_REGION_COLORS["Error"]
    marks = error_positions or set()
    chars = []
    for pos, ch in enumerate(clean_dna(seq), start=start_pos):
        if pos in marks:
            chars.append(
                f'<span style="background:{error_bg};color:{error_fg};'
                'border-radius:3px;padding:0 1px;font-weight:700;">'
                f"{ch}</span>"
            )
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span class="region-tag" style="background:{bg};color:{fg};'
        'display:inline-block;margin:2px 4px 2px 0;padding:4px 6px;'
        'border-radius:6px;font-family:monospace;word-break:break-all;">'
        f"<b>{name}</b>: {body}</span>"
    )


def _render_audio_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    parts = []
    cursor = 1
    for name, seq in _audio_row_regions(row):
        parts.append(_audio_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(seq)
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _mutate_audio_prepared_strand(
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
    full = clean_dna(row.get("Full strand", "")) or "".join(seq for _, seq in _audio_row_regions(row))
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
        region = _audio_region_for_position(row, pos)
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
        for name, seq in _audio_row_regions(row):
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


def _audio_strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Type", "Total length", "Payload length", "Filler length"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows[:50]])


def _audio_error_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
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


def _audio_error_events_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
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


def _strand_payload_dna(rows: List[Dict[str, str]], original_len: int) -> str:
    parts = []
    for row in rows:
        payload = row.get("Error payload", "") if str(row.get("Advanced error source", "")).lower() == "true" else ""
        parts.append(clean_dna(payload or row.get("Payload", "")))
    dna = "".join(parts)
    return dna[:original_len]


def _direct_substitute_dna(dna: str, substitution_rate: float, seed: int) -> Tuple[str, int]:
    """Fast substitution-only noise directly on encoded DNA; skips strand design."""
    seq = list(clean_dna(dna))
    rng = random.Random(str(int(seed)))
    n = 0
    for i, b in enumerate(seq):
        if rng.random() < float(substitution_rate):
            seq[i] = rng.choice([x for x in "ACGT" if x != b])
            n += 1
    return "".join(seq), n


def _rows_from_uploaded_strand_csv(uploaded_file) -> List[Dict[str, str]]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str).fillna("")
    return [{str(k): str(v) for k, v in row.items()} for row in df.to_dict("records")]


def render_audio_dna_storage_panel() -> None:
    # Step status is rendered by app.py for the unified four-tab UI.

    # 1. Input
    with st.container(border=True):
        _step_header(1, "Input")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### 📁 Input")
            uploaded = st.file_uploader(
                "Input audio file",
                type=["wav", "flac", "ogg", "mp3", "m4a", "aac"],
                key="audio_dna_upload",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                raw = uploaded.getvalue()
                signature = f"{uploaded.name}|{len(raw)}"
                if st.session_state.get("audio_input_signature") != signature:
                    st.session_state["audio_input_signature"] = signature
                    st.session_state["audio_input_name"] = uploaded.name
                    st.session_state["audio_input_bytes"] = raw
                    _clear_audio_after_input()

            raw = st.session_state.get("audio_input_bytes")
            name = st.session_state.get("audio_input_name", "—")
            if raw:
                st.markdown("##### 📄 File properties")
                c1, c2 = st.columns(2)
                c1.metric("Input file", name)
                c2.metric("Uploaded size", _fmt_bytes(len(raw)))
                st.caption("Preview is shown on the right. Compression/method selection is handled in Step 2.")
            else:
                st.info("Upload an audio file to start.")

        with right:
            st.markdown("#### 🎧 Preview")
            raw = st.session_state.get("audio_input_bytes")
            if raw:
                st.audio(raw)
            else:
                st.info("Audio preview will appear here after upload.")

    # 2. Audio Compression
    with st.container(border=True):
        _step_header(2, "Compression")
        raw = st.session_state.get("audio_input_bytes")
        if not raw:
            st.info("Upload a file first.")
        else:
            mode_name = st.selectbox("Audio mode", list(AUDIO_MODES), index=1, key="audio_mode_select")
            # st.caption("The full uploaded audio file is processed. Large files can take longer.")

            if st.button("Run Compression", key="runAdvanced: add noise to prepared small strands. Use this after Strand Design to simulate errors on app-generated strands._audio_compression", type="primary"):
                try:
                    start = time.perf_counter()
                    payload, original_wav, meta, original_samples = _payload_from_audio(raw, mode_name)
                    meta["compression_time_seconds"] = float(time.perf_counter() - start)
                    st.session_state["audio_payload"] = payload
                    st.session_state["audio_original_wav"] = original_wav
                    st.session_state["audio_payload_meta"] = meta
                    st.session_state["audio_original_samples"] = original_samples
                    _clear_audio_after_payload()
                except Exception as exc:
                    st.error(f"Audio compression failed: {exc}")

            payload = st.session_state.get("audio_payload")
            meta = st.session_state.get("audio_payload_meta", {})
            if payload:
                pcm16_bytes = int(meta.get("sample_count", 0)) * 2
                ratio_pcm16 = pcm16_bytes / max(1, len(payload))
                ratio_upload = int(meta.get("uploaded_file_bytes", 0)) / max(1, len(payload))
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Compressed payload", _fmt_bytes(len(payload)))
                m2.metric("Compression vs uploaded file", f"{ratio_upload:.2f}x")
                m3.metric("Compression vs PCM16", f"{ratio_pcm16:.2f}x")
                m4.metric("Audio mode", meta.get("mode", "-"))
                m5.metric("Audio time", f"{float(meta.get('duration_seconds', 0.0)):.2f}s")
                st.audio(st.session_state.get("audio_original_wav", b""), format="audio/wav")

    # 3. DNA Encoding
    with st.container(border=True):
        _step_header(3, "DNA Encoding")
        payload = st.session_state.get("audio_payload")
        if not payload:
            st.info("Run Audio Compression first.")
        else:
            options = [m for m in MAPPING_OPTIONS if m in {"Simple Mapping", "RINF_B16"}]
            if not options:
                options = ["Simple Mapping"]
            mapping = st.selectbox("Mapping rule", options, format_func=_display_mapping, key="audio_mapping_select")
            if st.button("Run DNA Encoding", key="run_audio_dna_encoding", type="primary"):
                dna, bits, codec_meta = encode_bytes_to_dna(payload, mapping)
                st.session_state["audio_mapping"] = mapping
                st.session_state["audio_dna"] = dna
                st.session_state["audio_bits"] = bits
                st.session_state["audio_codec_meta"] = codec_meta
                _clear_audio_after_dna()

            dna = st.session_state.get("audio_dna")
            if dna:
                m1, m2, m3 = st.columns(3)
                m1.metric("Mapping rule", _display_mapping(st.session_state.get("audio_mapping", mapping)))
                m2.metric("DNA length", f"{len(dna):,} nt")
                m3.metric("GC content", f"{gc_content(dna):.3f}")
                with st.expander("DNA preview", expanded=False):
                    st.text_area("Encoded DNA", dna[:3000] + ("..." if len(dna) > 3000 else ""), height=120)
                    st.download_button("Download encoded DNA", dna.encode("utf-8"), "audio_encoded_dna.txt", "text/plain")

                st.markdown("##### 🧪 Payload-level noise")
                show_payload_noise = st.checkbox(
                    "Advanced: add noise directly to encoded DNA payload",
                    value=False,
                    key="audio_show_payload_noise",
                )
                if show_payload_noise:
                    q1, q2 = st.columns(2)
                    direct_sub_rate = q1.number_input("Direct substitution rate", min_value=0.0, max_value=0.20, value=0.0050, step=0.001, format="%.4f", key="audio_direct_sub_rate")
                    direct_seed = q2.number_input("Quick seed", min_value=0, max_value=999999, value=23, step=1, key="audio_direct_sub_seed")
                    # st.caption("Fast substitution-only test on the full encoded DNA. This bypasses strand design.")
                    if st.button("Add direct substitutions", key="audio_add_direct_substitutions", type="primary"):
                        noisy_dna, n_sub = _direct_substitute_dna(dna, float(direct_sub_rate), int(direct_seed))
                        st.session_state["audio_noisy_dna"] = noisy_dna
                        st.session_state["audio_direct_noisy_dna"] = noisy_dna
                        st.session_state["audio_error_stats"] = {
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
                        for key in ["audio_recovered_wav", "audio_recovered_samples", "audio_recovery_metrics", "audio_decode_error"]:
                            st.session_state.pop(key, None)
                    if st.session_state.get("audio_noisy_dna") and st.session_state.get("audio_error_stats", {}).get("quick_skip_strand"):
                        qs = st.session_state.get("audio_error_stats", {})
                        q1, q2, q3 = st.columns(3)
                        q1.metric("Direct substitutions", f"{int(qs.get('Substitute count', 0)):,}")
                        q2.metric("Noisy encoded data", f"{int(qs.get('noisy_dna_len', 0)):,} nt")
                        q3.metric("Skip Strand Design", "Yes")

    # 4. Strand Design / DNA Error Simulation
    with st.container(border=True):
        _step_header(4, "Strand Design")
        dna = st.session_state.get("audio_dna")
        if not dna:
            st.info("Run DNA Encoding first.")
        else:
            # st.markdown("#### Strand Design")
            with st.expander("Strand design", expanded=not bool(st.session_state.get("audio_strand_rows"))):
                c1, c2 = st.columns(2)
                with c1:
                    target_len = st.number_input("Total strand length", min_value=80, max_value=250, value=125, step=1, key="audio_strand_total_len")
                    index_len = st.number_input("Strand ID length", min_value=0, max_value=24, value=8, step=1, key="audio_strand_index_len")
                with c2:
                    fbr = st.text_input("Forward primer", value="ACACGACGCTCTTCCGATCT", key="audio_fbr")
                    rbr = st.text_input("Reverse primer", value="AGATCGGAAGAGCACACGTCT", key="audio_rbr")

                if st.button("Run Strand Design", key="run_audio_strand_design", type="primary"):
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
                    st.session_state["audio_strand_rows"] = rows
                    for key in [
                        "audio_advanced_error_rows",
                        "audio_noisy_dna",
                        "audio_error_stats",
                        "audio_recovered_wav",
                        "audio_recovered_samples",
                        "audio_recovery_metrics",
                        "audio_decode_error",
                    ]:
                        st.session_state.pop(key, None)

            rows = st.session_state.get("audio_strand_rows", [])
            if not rows:
                st.info("Run Strand Design first.")
            else:
                total_strand_len = sum(len(clean_dna(row.get("Full strand", ""))) for row in rows)
                strand_expansion = total_strand_len / max(1, len(clean_dna(dna)))
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Designed strands", f"{len(rows):,}")
                s2.metric("Total strand length", f"{total_strand_len:,} nt")
                s3.metric("Strand Design expansion", f"{strand_expansion:.2f}×")
                s4.metric("DNA mapping", _display_mapping(st.session_state.get("audio_mapping", "")))

                st.dataframe(_audio_strand_summary(rows), use_container_width=True, hide_index=True)
                inspect_ids = [str(row.get("No.", i + 1)) for i, row in enumerate(rows)]
                selected_no = st.selectbox("Inspect designed strand", inspect_ids, index=0, key="audio_inspect_designed_strand")
                selected_row = next((row for row in rows if str(row.get("No.", "")) == selected_no), rows[0])
                _render_audio_segmented_strand(selected_row, "Designed strand")
                st.download_button("Download prepared strands", strand_rows_to_csv(rows), "audio_prepared_strands.csv", "text/csv")

            st.markdown("##### 🧪 Strand-level noise")
            st.caption("Advanced: add noise to prepared small strands. Use this after Strand Design to simulate errors on app-generated strands.")
            if not rows:
                st.info("Run Strand Design first.")
            else:
                with st.container(border=True):
                    e1, e2, e3, e4 = st.columns(4)
                    error_target = e1.selectbox("Error target", ["Payload only", "Index + Payload", "Full strand"], index=0, key="audio_error_target")
                    substitution_rate = e2.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.0200, step=0.001, format="%.4f", key="audio_sub_rate")
                    insertion_rate = e3.number_input("Insertion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="audio_ins_rate")
                    deletion_rate = e4.number_input("Deletion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="audio_del_rate")
                    seed = st.number_input("Seed", min_value=0, max_value=999999, value=17, step=1, key="audio_error_seed")

                    if st.button("Add errors", key="run_audio_error_simulation", type="primary"):
                        allow_indels = bool(float(insertion_rate) > 0.0 or float(deletion_rate) > 0.0)
                        err_rows = []
                        for row in rows:
                            try:
                                row_no = int(str(row.get("No.", "0") or "0"))
                            except Exception:
                                row_no = len(err_rows) + 1
                            err_rows.append(_mutate_audio_prepared_strand(
                                row,
                                scope=str(error_target),
                                substitution_rate=float(substitution_rate),
                                insertion_rate=float(insertion_rate),
                                deletion_rate=float(deletion_rate),
                                seed=int(seed) + row_no * 1000003,
                                allow_indels=allow_indels,
                            ))
                        events = _audio_error_events_table(err_rows)
                        noisy_dna = _strand_payload_dna(err_rows, len(clean_dna(dna)))
                        st.session_state["audio_advanced_error_rows"] = err_rows
                        st.session_state["audio_noisy_dna"] = noisy_dna
                        st.session_state["audio_error_stats"] = {
                            "error_target": error_target,
                            "substitution_rate": float(substitution_rate),
                            "insertion_rate": float(insertion_rate),
                            "deletion_rate": float(deletion_rate),
                            "seed": int(seed),
                            "total_errors": int(len(events)),
                            "noisy_dna_len": int(len(noisy_dna)),
                            "Substitute count": int(sum(int(str(row.get("Substitution count", "0") or "0")) for row in err_rows)),
                        }
                        for key in ["audio_recovered_wav", "audio_recovered_samples", "audio_recovery_metrics", "audio_decode_error"]:
                            st.session_state.pop(key, None)

                err_rows = st.session_state.get("audio_advanced_error_rows", [])
                noisy_dna = st.session_state.get("audio_noisy_dna", "")
                if err_rows:
                    events = _audio_error_events_table(err_rows)
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Error strands", f"{len(err_rows):,}")
                    a2.metric("Added errors", f"{len(events):,}")
                    a3.metric("Noisy encoded data", f"{len(noisy_dna):,} nt")
                    st.dataframe(_audio_error_rows_table(err_rows), use_container_width=True, hide_index=True)

                    eids = [str(row.get("No.", i + 1)) for i, row in enumerate(err_rows)]
                    eno = st.selectbox("Inspect error strand", eids, index=0, key="audio_inspect_error_strand")
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
                    _render_audio_segmented_strand(clean_for_error, "Clean strand", error_positions=err_positions)
                    _render_audio_segmented_strand(erow, "Error strand", error_positions=err_positions)
                    if not events.empty:
                        st.dataframe(events, use_container_width=True, hide_index=True)
                    st.download_button("Download error strands", strand_rows_to_csv(err_rows), "audio_error_strands.csv", "text/csv")
                    st.download_button("Download noisy encoded DNA", noisy_dna.encode("utf-8"), "audio_noisy_encoded_dna.txt", "text/plain")

            with st.expander("Advanced sequencing simulation", expanded=False):
                st.info("Reserved for sequencing/read simulation. Audio reconstruction currently decodes from designed, error, or uploaded strands.")

    # 5. Audio Reconstruction
    with st.container(border=True):
        _step_header(5, "Decoding")
        noisy_dna = st.session_state.get("audio_noisy_dna")
        if not st.session_state.get("audio_dna"):
            st.info("Run DNA Encoding first.")
        elif not noisy_dna:
            st.info("Clean encoded data is available. Run DNA Error Simulation or upload a strands CSV to test other reconstruction sources.")
        else:
            pass

        if st.session_state.get("audio_dna"):
            uploaded_dna_txt = st.file_uploader(
                "Upload full DNA TXT generated by this app",
                type=["txt"],
                key="audio_decode_upload_dna_txt",
            )
            if uploaded_dna_txt is not None:
                try:
                    st.session_state["audio_uploaded_decode_dna_txt"] = clean_dna(uploaded_dna_txt.getvalue().decode("utf-8", errors="ignore"))
                    st.success("Loaded uploaded full DNA TXT for reconstruction.")
                except Exception as exc:
                    st.error(f"Could not load DNA TXT: {exc}")

            uploaded_strands = st.file_uploader(
                "Upload prepared/error strands CSV generated by this app",
                type=["csv"],
                key="audio_decode_upload_strands_csv",
            )
            if uploaded_strands is not None:
                try:
                    uploaded_rows = _rows_from_uploaded_strand_csv(uploaded_strands)
                    uploaded_dna = _strand_payload_dna(uploaded_rows, len(st.session_state.get("audio_dna", "")))
                    st.session_state["audio_uploaded_decode_rows"] = uploaded_rows
                    st.session_state["audio_uploaded_decode_dna"] = uploaded_dna
                    st.success(f"Loaded {len(uploaded_rows):,} uploaded strands for reconstruction.")
                except Exception as exc:
                    st.error(f"Could not load strands CSV: {exc}")

            decode_choices = ["Current encoded DNA", "Noisy encoded DNA", "Upload full DNA TXT", "Upload prepared/error strands CSV"]

            selected_decode_source = None
            if decode_choices:
                selected_decode_source = st.radio(
                    "Reconstruction source",
                    decode_choices,
                    horizontal=True,
                    key="audio_reconstruction_source",
                )

            if selected_decode_source == "Upload prepared/error strands CSV":
                selected_input_dna = st.session_state.get("audio_uploaded_decode_dna", "")
            elif selected_decode_source == "Upload full DNA TXT":
                selected_input_dna = st.session_state.get("audio_uploaded_decode_dna_txt", "")
            elif selected_decode_source == "Noisy encoded DNA":
                selected_input_dna = st.session_state.get("audio_noisy_dna", "")
            else:
                selected_input_dna = st.session_state.get("audio_dna", "")
            if not selected_input_dna:
                st.info("No DNA is available for the selected reconstruction source.")

        if st.session_state.get("audio_dna") and (st.session_state.get("audio_noisy_dna") or st.session_state.get("audio_uploaded_decode_dna") or st.session_state.get("audio_dna")):
            if st.button("Run Decode", key="run_audio_reconstruction", type="primary", disabled=not bool(selected_input_dna)):
                if selected_decode_source == "Upload prepared/error strands CSV":
                    input_dna = st.session_state.get("audio_uploaded_decode_dna", "")
                elif selected_decode_source == "Upload full DNA TXT":
                    input_dna = st.session_state.get("audio_uploaded_decode_dna_txt", "")
                elif selected_decode_source == "Noisy encoded DNA":
                    input_dna = st.session_state.get("audio_noisy_dna", "")
                else:
                    input_dna = st.session_state.get("audio_dna", "")

                mapping = st.session_state.get("audio_mapping", "Simple Mapping")
                codec_meta = st.session_state.get("audio_codec_meta", {})
                try:
                    start = time.perf_counter()
                    decoded_payload, _decoded_bits, decoded_meta = decode_dna_with_mapping(input_dna, mapping, codec_meta)
                    recovered_wav, repair_meta, recovered_samples = _reconstruct_audio(decoded_payload, st.session_state.get("audio_payload_meta", {}))
                    reconstruction_time = float(time.perf_counter() - start)
                    original_samples = st.session_state.get("audio_original_samples", np.zeros(1, dtype=np.float32))
                    mse = _mse(original_samples, recovered_samples)
                    st.session_state["audio_recovered_wav"] = recovered_wav
                    st.session_state["audio_recovered_samples"] = recovered_samples
                    st.session_state["audio_decode_error"] = ""
                    st.session_state["audio_recovery_metrics"] = {
                        "decoded_bytes": len(decoded_payload),
                        "byte_exact": decoded_payload == st.session_state.get("audio_payload", b""),
                        "byte_error_rate": (
                            sum(a != b for a, b in zip(st.session_state.get("audio_payload", b""), decoded_payload))
                            + abs(len(st.session_state.get("audio_payload", b"")) - len(decoded_payload))
                        )
                        / max(1, len(st.session_state.get("audio_payload", b""))),
                        "snr_db": _snr_db(original_samples, recovered_samples),
                        "psnr_db": _psnr_db(original_samples, recovered_samples),
                        "waveform_mse": mse,
                        "waveform_correlation": _waveform_correlation(original_samples, recovered_samples),
                        "spectrogram_similarity": _spectrogram_similarity(original_samples, recovered_samples),
                        "reconstruction_time_seconds": reconstruction_time,
                        "reconstruction_source": selected_decode_source or "DNA Error Simulation output",
                        "decode_meta": decoded_meta,
                        **repair_meta,
                    }
                except Exception as exc:
                    st.session_state["audio_recovered_wav"] = b""
                    st.session_state["audio_decode_error"] = str(exc)
                    st.session_state["audio_recovery_metrics"] = {}

            if st.session_state.get("audio_decode_error"):
                st.warning(f"DNA decode warning: {st.session_state['audio_decode_error']}")
            recovered_wav = st.session_state.get("audio_recovered_wav")
            if recovered_wav:
                err_stats = st.session_state.get("audio_error_stats", {})
                metrics = st.session_state.get("audio_recovery_metrics", {})
                payload_meta = st.session_state.get("audio_payload_meta", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Base errors", f"{int(err_stats.get('Substitute count', 0)):,}")
                m2.metric("Byte error", f"{float(metrics.get('byte_error_rate', 0.0)) * 100:.2f}%")
                m3.metric("Signal-to-Noise Ratio", f"{float(metrics.get('snr_db', 0.0)):.2f} dB")
                m4.metric("Audio time", f"{float(payload_meta.get('duration_seconds', 0.0)):.2f}s")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("PSNR", f"{float(metrics.get('psnr_db', 0.0)):.2f} dB")
                r2.metric("Waveform Correlation", f"{float(metrics.get('waveform_correlation', 0.0)):.4f}")
                r3.metric("Spectrogram similarity", f"{float(metrics.get('spectrogram_similarity', 0.0)):.4f}")
                r4.metric("Reconstruction time", f"{float(metrics.get('reconstruction_time_seconds', 0.0)):.2f}s")
                st.audio(recovered_wav, format="audio/wav")
                st.download_button("Download recovered audio", recovered_wav, "audio_recovered.wav", "audio/wav")

    # 6. Analysis
    with st.container(border=True):
        _step_header(6, "Summarization")
        if not st.session_state.get("audio_recovered_wav") and not st.session_state.get("audio_decode_error"):
            st.info("Run Decode first.")
        else:
            metrics = st.session_state.get("audio_recovery_metrics", {}) or {}
            payload_meta = st.session_state.get("audio_payload_meta", {}) or {}
            payload = st.session_state.get("audio_payload", b"") or b""
            dna = st.session_state.get("audio_dna", "") or ""
            strand_rows = st.session_state.get("audio_strand_rows", []) or []
            err_stats = st.session_state.get("audio_error_stats", {}) or {}
            noisy = st.session_state.get("audio_noisy_dna", "") or ""
            error_rows = st.session_state.get("audio_error_rows", []) or []

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
                original_wav = st.session_state.get("audio_input_bytes", b"")
                if original_wav:
                    st.audio(original_wav)
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    {"Property": "Duration", "Value": f"{float(payload_meta.get('duration_seconds', 0.0)):.2f}s"},
                    {"Property": "Sample rate", "Value": payload_meta.get("sample_rate", "—")},
                    {"Property": "Original size", "Value": _fmt_bytes(int(payload_meta.get('uploaded_file_bytes', 0)))},
                ]), hide_index=True, use_container_width=True)
            with encoded_col:
                st.markdown("##### Compressed/Encoded")
                encoded_audio = st.session_state.get("audio_original_wav", b"")
                if encoded_audio:
                    st.audio(encoded_audio, format="audio/wav")
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    {"Property": "Method", "Value": payload_meta.get("method", "—")},
                    {"Property": "Payload size", "Value": _fmt_bytes(len(payload))},
                    {"Property": "DNA length", "Value": f"{len(dna):,} nt" if dna else "—"},
                ]), hide_index=True, use_container_width=True)
            with decoded_col:
                st.markdown("##### Decoded")
                recovered_wav = st.session_state.get("audio_recovered_wav", b"")
                if recovered_wav:
                    st.audio(recovered_wav, format="audio/wav")
                else:
                    st.write("—")
                st.dataframe(pd.DataFrame([
                    {"Property": "Decode source", "Value": metrics.get("reconstruction_source", "—")},
                    # {"Property": "Output status", "Value": "Playable" if recovered_wav else "Failed"},
                    {"Property": "SNR", "Value": f"{float(metrics.get('snr_db', 0.0)):.2f} dB" if metrics else "—"},
                ]), hide_index=True, use_container_width=True)

            storage_rows = [
                {"Metric": "Data type", "Value": "Audio"},
                {"Metric": "Audio mode", "Value": payload_meta.get("mode", "-")},
                {"Metric": "Method", "Value": payload_meta.get("method", "-")},
                {"Metric": "Duration", "Value": f"{float(payload_meta.get('duration_seconds', 0.0)):.2f}s"},
                {"Metric": "Sample rate", "Value": payload_meta.get("sample_rate", "-")},
                {"Metric": "Channels", "Value": "1"},
                {"Metric": "Original size", "Value": _fmt_bytes(int(payload_meta.get('uploaded_file_bytes', 0)))},
                {"Metric": "Payload size", "Value": _fmt_bytes(len(payload))},
                {"Metric": "Compression vs uploaded file", "Value": f"{float(payload_meta.get('uploaded_file_bytes', 0)) / max(1, len(payload)):.2f}x"},
                {"Metric": "Compression vs PCM16", "Value": f"{(int(payload_meta.get('sample_count', 0)) * 2) / max(1, len(payload)):.2f}x"},
                {"Metric": "Estimated DNA length", "Value": f"{len(payload) * 4:,} nt" if payload else "—"},
            ]
            analysis_table("Compression analysis", compact_rows(
                storage_rows,
                ["Data type", "Method", "Duration", "Sample rate", "Channels", "Original size", "Payload size", "Compression vs uploaded file", "Estimated DNA length"],
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
    st.session_state.get("audio_mapping", st.session_state.get("audio_mapping_select", "—"))
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
                # {"Metric": "Error status", "Value": "Noisy" if is_noisy else "Clean"},
                # {"Metric": "Error level", "Value": "Payload-level" if err_stats.get("quick_skip_strand") else ("Strand-level" if error_rows else "Clean DNA")},
                {"Metric": "Error type", "Value": "Substitution" if is_noisy else "None"},
                {"Metric": "Substitution rate", "Value": err_stats.get("substitution_rate", "0")},
                {"Metric": "Substituted bases", "Value": f"{int(err_stats.get('Substitute count', err_stats.get('total_errors', 0)) or 0):,}"},
                # {"Metric": "Error scope", "Value": err_stats.get("scope", "payload") if err_stats else "—"},
                {"Metric": "Affected strands", "Value": f"{len(error_rows):,}" if error_rows else ("—" if err_stats.get("quick_skip_strand") else "0")},
                {"Metric": "Seed", "Value": err_stats.get("seed", "—")},
            ]
            analysis_table("Error Adding Report", compact_rows(
                error_rows_summary,
                ["Error status", "Error level", "Error type", "Substitution rate", "Substituted bases", "Error scope", "Affected strands", "Seed"],
            ))

            playable = bool(st.session_state.get("audio_recovered_wav"))
            byte_exact = bool(metrics.get("byte_exact"))
            recovery_class = "Exact" if byte_exact else ("Usable" if playable else "Failed")
            quality_rows = [
                {"Metric": "Decode source", "Value": metrics.get("reconstruction_source", "—")},
                {"Metric": "Decode status", "Value": "Success" if playable and not st.session_state.get("audio_decode_error") else "Failed"},
                # {"Metric": "Output status", "Value": "Playable" if playable else "Failed"},
                # {"Metric": "Recovery class", "Value": recovery_class},
                {"Metric": "Payload accuracy", "Value": f"{(1.0 - float(metrics.get('byte_error_rate', 0.0))) * 100:.2f}%"},
                # {"Metric": "Checksum", "Value": "Pass" if byte_exact else "Fail"},
                {"Metric": "Signal-to-Noise Ratio", "Value": f"{float(metrics.get('snr_db', 0.0)):.2f} dB"},
                {"Metric": "Waveform Correlation", "Value": f"{float(metrics.get('waveform_correlation', 0.0)):.4f}"},
                {"Metric": "Spectrogram similarity", "Value": f"{float(metrics.get('spectrogram_similarity', 0.0)):.4f}"},
                # {"Metric": "Duration difference", "Value": f"{float(metrics.get('duration_difference_seconds', 0.0)):.3f}s" if metrics.get('duration_difference_seconds') is not None else "—"},
            ]
            analysis_table("Recovery Quality Report", compact_rows(
                quality_rows,
                ["Decode source", "Decode status", "Output status", "Recovery class", "Payload accuracy", "Checksum", "Signal-to-Noise Ratio", "Waveform Correlation", "Spectrogram similarity", "Duration difference"],
            ))


            method_rows = [
                {"Property": "Audio mode", "Value": payload_meta.get("mode", "—")},
                {"Property": "Compression vs PCM16", "Value": f"{(int(payload_meta.get('sample_count', 0)) * 2) / max(1, len(payload)):.2f}x" if payload else "—"},
                {"Property": "Cleanup mode", "Value": "AI cleanup" if metrics.get("ai_cleanup") else ("Click repair" if metrics.get("click_repair", True) else "None")},
                {"Property": "Recovered duration", "Value": f"{float(metrics.get('recovered_duration_seconds', 0.0)):.3f}s" if metrics.get('recovered_duration_seconds') is not None else "—"},
            ]
            if method_rows:
                with st.expander("Method-specific details", expanded=False):
                    st.dataframe(pd.DataFrame([{"Property": r.get("Property", r.get("Metric", "—")), "Value": r.get("Value", "—")} for r in method_rows]), hide_index=True, use_container_width=True)
