"""
video_dna_compressor_v21.py

Best current no-ECC video DNA storage compressor module.

Final visual method:
    Color-preserving fixed-layout wavelet video representation
    + Simple Mapping DNA
    + temporal similarity repair
    + reliability-guided local cleanup

Optional audio method:
    mono 16 kHz μ-law audio -> Simple Mapping DNA -> noisy decode -> light cleanup

Important:
- This is not ECC.
- This does not recover original DNA bytes.
- Visual/audio repair only conceals artifacts after noisy reconstruction.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None
try:
    from scipy.io import wavfile
    from scipy.signal import medfilt, butter, filtfilt
except Exception:  # pragma: no cover
    wavfile = None
    medfilt = None
    butter = None
    filtfilt = None
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:  # pragma: no cover
    peak_signal_noise_ratio = None
    structural_similarity = None


@dataclass
class VideoDNAConfig:
    output_dir: str = "video_dna_v21_output"
    max_seconds: Optional[float] = 4.0

    target_width: int = 192
    target_height: int = 108

    fps_policy: str = "original"  # original | comfortable | manual
    max_allowed_fps: float = 24.0
    comfort_fps: float = 12.0
    manual_fps: float = 24.0

    substitution_rate: float = 0.02
    random_seed: int = 42

    wavelet: str = "haar"
    wavelet_level: int = 2
    q_bits: int = 6
    y_band_mode: str = "LL"
    chroma_band_mode: str = "LL"
    chroma_downsample: int = 4
    spatial_ll_step: float = 16.0
    spatial_detail_step: float = 10.0

    repair_radius: int = 6
    repair_top_k: int = 3
    repair_base_threshold: float = 28.0
    repair_mad_scale: float = 3.5
    repair_blend_strength: float = 0.85
    repair_max_mask_ratio: float = 0.40

    cleanup_enabled: bool = True
    cleanup_radius: int = 6
    cleanup_top_k: int = 3
    cleanup_base_threshold: float = 22.0
    cleanup_mad_scale: float = 3.0
    cleanup_blend_strength: float = 0.55
    cleanup_max_mask_ratio: float = 0.25

    audio_enabled: bool = False
    audio_sample_rate: int = 16000
    audio_bitrate_for_mux: str = "96k"


@dataclass
class VisualDNAResult:
    method: str
    fps: float
    resolution: str
    playable: bool
    payload_bytes: int
    dna_length_nt: int
    added_substitutions: int
    compression_vs_raw_rgb: float
    video_psnr: float
    video_ssim: float
    frame_correlation: float
    output_video: str
    frames_compared: int
    raw_rgb_bytes: int


@dataclass
class AudioDNAResult:
    method: str
    sample_rate: int
    payload_bytes: int
    dna_length_nt: int
    added_substitutions: int
    substitution_rate: float
    duration_sec: float
    snr_raw_db: float
    snr_clean_db: float
    original_audio_wav: str
    recovered_audio_raw_wav: str
    recovered_audio_clean_wav: str


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required: pip install opencv-python")


def _require_pywt() -> None:
    if pywt is None:
        raise ImportError("PyWavelets is required: pip install PyWavelets")


def _require_audio() -> None:
    if wavfile is None or medfilt is None or butter is None or filtfilt is None:
        raise ImportError("scipy is required for audio: pip install scipy")


def _require_metrics() -> None:
    if peak_signal_noise_ratio is None or structural_similarity is None:
        raise ImportError("scikit-image is required for metrics: pip install scikit-image")


# =========================================================
# FFmpeg helpers
# =========================================================

def run_cmd(cmd: Sequence[Any], quiet: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if not quiet:
            print(result.stderr[-4000:])
        raise RuntimeError("Command failed: " + " ".join(map(str, cmd)))
    return result


def parse_fps_fraction(frac: str) -> Optional[float]:
    try:
        a, b = frac.split("/")
        b = float(b)
        if b == 0:
            return None
        return float(a) / b
    except Exception:
        return None


def ffprobe_video(path: str | Path) -> Optional[Dict[str, Any]]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of", "json", str(path),
    ]
    result = run_cmd(cmd, quiet=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return None
    s = streams[0]
    fps = parse_fps_fraction(s.get("avg_frame_rate", "0/1"))
    if fps is None or fps <= 0:
        fps = parse_fps_fraction(s.get("r_frame_rate", "0/1"))
    duration = None
    if s.get("duration"):
        try:
            duration = float(s.get("duration"))
        except Exception:
            duration = None
    nb_frames = None
    if s.get("nb_frames") and str(s.get("nb_frames")).isdigit():
        nb_frames = int(s.get("nb_frames"))
    return {
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
    }


def has_audio_stream(video_path: str | Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool(result.stdout.strip())


def choose_target_fps(video_info: Dict[str, Any], config: VideoDNAConfig) -> float:
    original_fps = float(video_info.get("fps") or config.max_allowed_fps)
    if config.fps_policy == "original":
        return min(original_fps, float(config.max_allowed_fps))
    if config.fps_policy == "comfortable":
        return float(config.comfort_fps)
    if config.fps_policy == "manual":
        return float(config.manual_fps)
    return min(original_fps, float(config.max_allowed_fps))


def extract_frames(input_video: str | Path, out_dir: str | Path, fps: float, width: int, height: int, max_seconds: Optional[float] = None) -> List[Path]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale={width}:{height}:flags=lanczos"
    cmd = ["ffmpeg", "-y"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", str(input_video), "-vf", vf, "-vsync", "0", str(out_dir / "frame_%06d.png")]
    run_cmd(cmd, quiet=True)
    return sorted(out_dir.glob("frame_*.png"))


def load_frames(frame_dir: str | Path, color: str = "rgb") -> List[np.ndarray]:
    _require_cv2()
    files = sorted(Path(frame_dir).glob("frame_*.png"))
    frames: List[np.ndarray] = []
    for f in files:
        img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if color == "rgb":
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif color == "gray":
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("color must be 'rgb' or 'gray'")
        frames.append(img.astype(np.uint8))
    return frames


def save_frames(frames: Sequence[np.ndarray], out_dir: str | Path) -> List[Path]:
    _require_cv2()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(frames, start=1):
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 2:
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), arr)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), bgr)
    return sorted(out_dir.glob("frame_*.png"))


def make_video_from_frames(frame_dir: str | Path, output_path: str | Path, fps: float, crf: int = 23) -> str:
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-vf", "format=yuv420p", "-c:v", "libx264", "-profile:v", "baseline",
        "-level", "3.0", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    run_cmd(cmd, quiet=True)
    return str(output_path)


def playable_video(path: str | Path) -> bool:
    try:
        return ffprobe_video(path) is not None
    except Exception:
        return False


def mux_video_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path, audio_bitrate: str = "96k") -> str:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", audio_bitrate, "-shortest", str(output_path),
    ]
    run_cmd(cmd, quiet=True)
    return str(output_path)


def mux_original_audio_for_preview(recovered_video: str | Path, original_video: str | Path, output_video: str | Path, audio_bitrate: str = "128k") -> str:
    """Preview-only. Original audio is not encoded into DNA."""
    if not has_audio_stream(original_video):
        return str(recovered_video)
    cmd = [
        "ffmpeg", "-y", "-i", str(recovered_video), "-i", str(original_video),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", audio_bitrate, "-shortest", str(output_video),
    ]
    run_cmd(cmd, quiet=True)
    return str(output_video)


# =========================================================
# Simple Mapping DNA helpers
# =========================================================

def bytes_to_base_groups(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    groups = np.empty(arr.size * 4, dtype=np.uint8)
    groups[0::4] = (arr >> 6) & 3
    groups[1::4] = (arr >> 4) & 3
    groups[2::4] = (arr >> 2) & 3
    groups[3::4] = arr & 3
    return groups


def base_groups_to_bytes(groups: np.ndarray) -> bytes:
    groups = np.asarray(groups, dtype=np.uint8)
    n = (groups.size // 4) * 4
    groups = groups[:n].reshape(-1, 4)
    arr = ((groups[:, 0] << 6) | (groups[:, 1] << 4) | (groups[:, 2] << 2) | groups[:, 3]).astype(np.uint8)
    return arr.tobytes()


def base_groups_to_dna(groups: np.ndarray) -> str:
    alphabet = np.array(list("ACGT"))
    return "".join(alphabet[np.asarray(groups, dtype=np.uint8)].tolist())


def dna_to_base_groups(dna: str) -> np.ndarray:
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.array([lut[b] for b in dna.upper() if b in lut], dtype=np.uint8)


def transmit_through_dna_substitution(data: bytes, sub_rate: float = 0.02, rng: Optional[np.random.Generator] = None) -> Tuple[bytes, int, int]:
    if rng is None:
        rng = np.random.default_rng()
    groups = bytes_to_base_groups(data)
    dna_len = int(groups.size)
    if dna_len == 0:
        return data, 0, 0
    mask = rng.random(dna_len) < sub_rate
    n_errors = int(mask.sum())
    if n_errors > 0:
        changes = rng.integers(1, 4, size=n_errors, dtype=np.uint8)
        groups[mask] = (groups[mask] + changes) % 4
    noisy_bytes = base_groups_to_bytes(groups)
    return noisy_bytes, dna_len, n_errors


# =========================================================
# Metrics
# =========================================================

def frame_raw_size_bytes(frames: Sequence[np.ndarray]) -> int:
    return int(sum(np.asarray(f).nbytes for f in frames))


def compute_video_metrics(original_frames: Sequence[np.ndarray], recovered_frames: Sequence[np.ndarray]) -> Dict[str, float]:
    _require_metrics()
    _require_cv2()
    n = min(len(original_frames), len(recovered_frames))
    if n == 0:
        return {"frames_compared": 0, "video_psnr": float("nan"), "video_ssim": float("nan"), "frame_correlation": float("nan")}
    psnrs: List[float] = []
    ssims: List[float] = []
    cors: List[float] = []
    for i in range(n):
        a = np.asarray(original_frames[i], dtype=np.uint8)
        b = np.asarray(recovered_frames[i], dtype=np.uint8)
        if a.shape != b.shape:
            if b.ndim == 2 and a.ndim == 3:
                b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        try:
            p = peak_signal_noise_ratio(a, b, data_range=255)
            psnrs.append(100.0 if not np.isfinite(p) else float(p))
        except Exception:
            pass
        try:
            if a.ndim == 2:
                s = structural_similarity(a, b, data_range=255)
            else:
                s = structural_similarity(a, b, data_range=255, channel_axis=-1)
            ssims.append(float(s))
        except Exception:
            pass
        try:
            af = a.astype(np.float32).ravel()
            bf = b.astype(np.float32).ravel()
            if af.std() > 1e-6 and bf.std() > 1e-6:
                cors.append(float(np.corrcoef(af, bf)[0, 1]))
        except Exception:
            pass
    return {
        "frames_compared": int(n),
        "video_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "video_ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "frame_correlation": float(np.mean(cors)) if cors else float("nan"),
    }


# =========================================================
# Fixed-layout wavelet codec
# =========================================================

def pad_to_wavelet_multiple(arr: np.ndarray, level: int = 2) -> Tuple[np.ndarray, Tuple[int, int]]:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape[:2]
    m = 2 ** level
    ph = int(np.ceil(h / m) * m)
    pw = int(np.ceil(w / m) * m)
    padded = np.pad(arr, ((0, ph - h), (0, pw - w)), mode="reflect")
    return padded, (h, w)


def quantize_coeff(x: np.ndarray, step: float = 10.0, q_bits: int = 6) -> np.ndarray:
    max_q = (1 << q_bits) - 1
    center = max_q // 2
    q = np.round(x / step + center)
    return np.clip(q, 0, max_q).astype(np.uint8)


def dequantize_coeff(q: np.ndarray, step: float = 10.0, q_bits: int = 6) -> np.ndarray:
    max_q = (1 << q_bits) - 1
    center = max_q // 2
    q = np.asarray(q, dtype=np.uint8) & max_q
    return (q.astype(np.float32) - center) * step


def selected_coeff_specs(shape: Tuple[int, int], band_mode: str = "LL", level: int = 2, wavelet: str = "haar") -> List[Tuple[str, Tuple[int, int]]]:
    _require_pywt()
    dummy, _ = pad_to_wavelet_multiple(np.zeros(shape, dtype=np.float32), level)
    coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=level, mode="periodization")
    specs: List[Tuple[str, Tuple[int, int]]] = []
    if band_mode in ["LL", "LL_L2", "LL_L2_HV"]:
        specs.append(("LL", coeffs[0].shape))
    if band_mode == "LL_L2":
        cH2, cV2, cD2 = coeffs[1]
        specs.extend([("H2", cH2.shape), ("V2", cV2.shape), ("D2", cD2.shape)])
    if band_mode == "LL_L2_HV":
        cH2, cV2, _cD2 = coeffs[1]
        specs.extend([("H2", cH2.shape), ("V2", cV2.shape)])
    return specs


def coeff_step(label: str, ll_step: float, detail_step: float) -> float:
    return ll_step if label == "LL" else detail_step


def extract_selected_coeffs(coeffs: Sequence[Any], band_mode: str = "LL") -> List[Tuple[str, np.ndarray]]:
    arrays: List[Tuple[str, np.ndarray]] = []
    if band_mode in ["LL", "LL_L2", "LL_L2_HV"]:
        arrays.append(("LL", coeffs[0]))
    if band_mode == "LL_L2":
        cH2, cV2, cD2 = coeffs[1]
        arrays.extend([("H2", cH2), ("V2", cV2), ("D2", cD2)])
    if band_mode == "LL_L2_HV":
        cH2, cV2, _cD2 = coeffs[1]
        arrays.extend([("H2", cH2), ("V2", cV2)])
    return arrays


def build_zero_coeffs(shape: Tuple[int, int], level: int = 2, wavelet: str = "haar") -> List[Any]:
    _require_pywt()
    dummy, _ = pad_to_wavelet_multiple(np.zeros(shape, dtype=np.float32), level)
    coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=level, mode="periodization")
    zero_coeffs: List[Any] = [np.zeros_like(coeffs[0])]
    for details in coeffs[1:]:
        zero_coeffs.append(tuple(np.zeros_like(d) for d in details))
    return zero_coeffs


def insert_selected_coeffs(zero_coeffs: List[Any], selected_arrays: List[Tuple[str, np.ndarray]]) -> List[Any]:
    coeffs = list(zero_coeffs)
    lookup = {label: arr for label, arr in selected_arrays}
    if "LL" in lookup:
        coeffs[0] = lookup["LL"]
    if len(coeffs) > 1:
        cH2, cV2, cD2 = coeffs[1]
        if "H2" in lookup:
            cH2 = lookup["H2"]
        if "V2" in lookup:
            cV2 = lookup["V2"]
        if "D2" in lookup:
            cD2 = lookup["D2"]
        coeffs[1] = (cH2, cV2, cD2)
    return coeffs


def wavelet_encode_fixed_array(arr_float: np.ndarray, band_mode: str = "LL", level: int = 2, wavelet: str = "haar", ll_step: float = 16.0, detail_step: float = 10.0, q_bits: int = 6) -> bytes:
    _require_pywt()
    padded, _ = pad_to_wavelet_multiple(arr_float, level)
    coeffs = pywt.wavedec2(padded, wavelet=wavelet, level=level, mode="periodization")
    selected = extract_selected_coeffs(coeffs, band_mode=band_mode)
    q_parts = []
    for label, coeff_arr in selected:
        step = coeff_step(label, ll_step, detail_step)
        q_parts.append(quantize_coeff(coeff_arr, step=step, q_bits=q_bits).ravel())
    if not q_parts:
        return b""
    return np.concatenate(q_parts).astype(np.uint8).tobytes()


def wavelet_decode_fixed_array(payload: bytes, original_shape: Tuple[int, int], band_mode: str = "LL", level: int = 2, wavelet: str = "haar", ll_step: float = 16.0, detail_step: float = 10.0, q_bits: int = 6) -> np.ndarray:
    _require_pywt()
    specs = selected_coeff_specs(original_shape, band_mode=band_mode, level=level, wavelet=wavelet)
    expected_len = int(sum(np.prod(shape) for _, shape in specs))
    q = np.frombuffer(payload, dtype=np.uint8)
    if q.size < expected_len:
        q = np.pad(q, (0, expected_len - q.size), constant_values=(1 << q_bits) // 2)
    q = q[:expected_len]
    selected_arrays: List[Tuple[str, np.ndarray]] = []
    cursor = 0
    for label, shape in specs:
        n = int(np.prod(shape))
        q_part = q[cursor:cursor + n].reshape(shape)
        cursor += n
        step = coeff_step(label, ll_step, detail_step)
        selected_arrays.append((label, dequantize_coeff(q_part, step=step, q_bits=q_bits)))
    zero_coeffs = build_zero_coeffs(original_shape, level=level, wavelet=wavelet)
    coeffs_rec = insert_selected_coeffs(zero_coeffs, selected_arrays)
    rec = pywt.waverec2(coeffs_rec, wavelet=wavelet, mode="periodization")
    h, w = original_shape
    return rec[:h, :w].astype(np.float32)


def clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def resize_chroma(ch: np.ndarray, scale: int = 4) -> np.ndarray:
    _require_cv2()
    h, w = ch.shape
    new_w = max(1, w // scale)
    new_h = max(1, h // scale)
    return cv2.resize(ch, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upsample_chroma(ch_small: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    _require_cv2()
    h, w = target_shape
    return cv2.resize(ch_small, (w, h), interpolation=cv2.INTER_LINEAR)


def light_post_filter_color(rgb: np.ndarray) -> np.ndarray:
    _require_cv2()
    rgb = clip_u8(rgb)
    rgb = cv2.medianBlur(rgb, 3)
    return cv2.GaussianBlur(rgb, (3, 3), 0)


# =========================================================
# Visual DNA codec
# =========================================================

def run_color_wavelet_visual_noecc(frames_rgb: Sequence[np.ndarray], config: VideoDNAConfig, rng: Optional[np.random.Generator] = None, method_name: str = "base_simple_mapping") -> Tuple[List[np.ndarray], Dict[str, Any]]:
    _require_cv2()
    _require_pywt()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)
    if not frames_rgb:
        raise ValueError("frames_rgb is empty")
    recovered_frames: List[np.ndarray] = []
    total_payload_bytes = 0
    total_dna_len = 0
    total_errors = 0
    rgb0 = np.asarray(frames_rgb[0], dtype=np.uint8)
    h, w = rgb0.shape[:2]
    for frame in frames_rgb:
        frame = np.asarray(frame, dtype=np.uint8)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        Cr = ycrcb[:, :, 1].astype(np.float32)
        Cb = ycrcb[:, :, 2].astype(np.float32)
        Cr_small = resize_chroma(Cr.astype(np.uint8), scale=config.chroma_downsample).astype(np.float32)
        Cb_small = resize_chroma(Cb.astype(np.uint8), scale=config.chroma_downsample).astype(np.float32)
        Y_c = Y - 128.0
        Cr_c = Cr_small - 128.0
        Cb_c = Cb_small - 128.0
        payload_Y = wavelet_encode_fixed_array(Y_c, config.y_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        payload_Cr = wavelet_encode_fixed_array(Cr_c, config.chroma_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        payload_Cb = wavelet_encode_fixed_array(Cb_c, config.chroma_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        noisy_Y, dna_Y, err_Y = transmit_through_dna_substitution(payload_Y, config.substitution_rate, rng)
        noisy_Cr, dna_Cr, err_Cr = transmit_through_dna_substitution(payload_Cr, config.substitution_rate, rng)
        noisy_Cb, dna_Cb, err_Cb = transmit_through_dna_substitution(payload_Cb, config.substitution_rate, rng)
        Y_rec_c = wavelet_decode_fixed_array(noisy_Y, (h, w), config.y_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        Cr_rec_c_small = wavelet_decode_fixed_array(noisy_Cr, Cr_small.shape, config.chroma_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        Cb_rec_c_small = wavelet_decode_fixed_array(noisy_Cb, Cb_small.shape, config.chroma_band_mode, config.wavelet_level, config.wavelet, config.spatial_ll_step, config.spatial_detail_step, config.q_bits)
        Y_rec = clip_u8(Y_rec_c + 128.0)
        Cr_small_rec = clip_u8(Cr_rec_c_small + 128.0)
        Cb_small_rec = clip_u8(Cb_rec_c_small + 128.0)
        Cr_rec = upsample_chroma(Cr_small_rec, target_shape=(h, w))
        Cb_rec = upsample_chroma(Cb_small_rec, target_shape=(h, w))
        ycrcb_rec = np.stack([Y_rec, Cr_rec, Cb_rec], axis=-1).astype(np.uint8)
        rgb_rec = cv2.cvtColor(ycrcb_rec, cv2.COLOR_YCrCb2RGB)
        recovered_frames.append(light_post_filter_color(rgb_rec))
        total_payload_bytes += len(payload_Y) + len(payload_Cr) + len(payload_Cb)
        total_dna_len += dna_Y + dna_Cr + dna_Cb
        total_errors += err_Y + err_Cr + err_Cb
    meta = {
        "method": method_name,
        "payload_bytes": int(total_payload_bytes),
        "dna_length_nt": int(total_dna_len),
        "added_substitutions": int(total_errors),
    }
    return recovered_frames, meta


# =========================================================
# Temporal repair and reliability cleanup
# =========================================================

def ensure_u8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(frame), 0, 255).astype(np.uint8)


def to_luma(frame: np.ndarray) -> np.ndarray:
    _require_cv2()
    frame = ensure_u8(frame)
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[:, :, 0]


def frame_similarity_score(a: np.ndarray, b: np.ndarray, small_size: Tuple[int, int] = (64, 36)) -> float:
    _require_cv2()
    ya = to_luma(a)
    yb = to_luma(b)
    ya_small = cv2.resize(ya, small_size, interpolation=cv2.INTER_AREA)
    yb_small = cv2.resize(yb, small_size, interpolation=cv2.INTER_AREA)
    return float(np.mean((ya_small.astype(np.float32) - yb_small.astype(np.float32)) ** 2))


def find_similar_frame_indices(frames: Sequence[np.ndarray], index: int, radius: int = 6, top_k: int = 3) -> List[int]:
    n = len(frames)
    scores: List[Tuple[float, int]] = []
    for j in range(max(0, index - radius), min(n, index + radius + 1)):
        if j == index:
            continue
        scores.append((frame_similarity_score(frames[index], frames[j]), j))
    scores.sort(key=lambda x: x[0])
    return [j for _, j in scores[:top_k]]


def reference_from_similar_frames(frames: Sequence[np.ndarray], indices: Sequence[int]) -> np.ndarray:
    stack = np.stack([ensure_u8(frames[j]) for j in indices], axis=0)
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


def detect_error_mask(current: np.ndarray, reference: np.ndarray, base_threshold: float = 28, mad_scale: float = 3.5, dilate_iter: int = 1) -> np.ndarray:
    _require_cv2()
    cur = ensure_u8(current)
    ref = ensure_u8(reference)
    y_cur = to_luma(cur).astype(np.float32)
    y_ref = to_luma(ref).astype(np.float32)
    luma_diff = np.abs(y_cur - y_ref)
    if cur.ndim == 3:
        color_diff = np.sqrt(np.sum((cur.astype(np.float32) - ref.astype(np.float32)) ** 2, axis=2))
        combined = 0.75 * luma_diff + 0.25 * color_diff
    else:
        combined = luma_diff
    med = np.median(combined)
    mad = np.median(np.abs(combined - med)) + 1e-6
    threshold = max(base_threshold, med + mad_scale * mad)
    mask_u8 = ((combined > threshold).astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if dilate_iter > 0:
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iter)
    return mask_u8.astype(bool)


def temporal_similarity_repair(frames: Sequence[np.ndarray], radius: int = 6, top_k: int = 3, base_threshold: float = 28, mad_scale: float = 3.5, blend_strength: float = 0.85, max_mask_ratio: float = 0.40, dilate_iter: int = 1) -> List[np.ndarray]:
    repaired: List[np.ndarray] = []
    frames_u8 = [ensure_u8(f) for f in frames]
    for i, cur in enumerate(frames_u8):
        similar_ids = find_similar_frame_indices(frames_u8, index=i, radius=radius, top_k=top_k)
        if not similar_ids:
            repaired.append(cur.copy())
            continue
        ref = reference_from_similar_frames(frames_u8, similar_ids)
        mask = detect_error_mask(cur, ref, base_threshold, mad_scale, dilate_iter)
        mask_ratio = float(mask.mean())
        cur_f = cur.astype(np.float32)
        ref_f = ref.astype(np.float32)
        if mask_ratio > max_mask_ratio:
            out = 0.75 * cur_f + 0.25 * ref_f
        else:
            repaired_pixels = (1.0 - blend_strength) * cur_f + blend_strength * ref_f
            if cur.ndim == 3:
                out = np.where(mask[:, :, None], repaired_pixels, cur_f)
            else:
                out = np.where(mask, repaired_pixels, cur_f)
        repaired.append(np.clip(out, 0, 255).astype(np.uint8))
    return repaired


def robust_threshold_map(values: np.ndarray, base_threshold: float = 22, mad_scale: float = 3.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    return float(max(base_threshold, med + mad_scale * mad))


def build_nearby_temporal_reference(frames: Sequence[np.ndarray], index: int, radius: int = 6, top_k: int = 3) -> np.ndarray:
    frames_u8 = [ensure_u8(f) for f in frames]
    ids = find_similar_frame_indices(frames_u8, index=index, radius=radius, top_k=top_k)
    if ids:
        return reference_from_similar_frames(frames_u8, ids)
    n = len(frames_u8)
    fallback_ids = [j for j in range(max(0, index - radius), min(n, index + radius + 1)) if j != index]
    if not fallback_ids:
        return frames_u8[index].copy()
    return reference_from_similar_frames(frames_u8, fallback_ids[:top_k])


def detect_remaining_artifact_mask(current: np.ndarray, reference: np.ndarray, base_threshold: float = 22, mad_scale: float = 3.0, temporal_weight: float = 0.65, local_weight: float = 0.25, color_weight: float = 0.10, dilate_iter: int = 1) -> np.ndarray:
    _require_cv2()
    cur = ensure_u8(current)
    ref = ensure_u8(reference)
    cur_y = to_luma(cur).astype(np.float32)
    ref_y = to_luma(ref).astype(np.float32)
    temporal_diff = np.abs(cur_y - ref_y)
    local_med = cv2.medianBlur(cur_y.astype(np.uint8), 5).astype(np.float32)
    local_diff = np.abs(cur_y - local_med)
    if cur.ndim == 3:
        color_diff = np.sqrt(np.sum((cur.astype(np.float32) - ref.astype(np.float32)) ** 2, axis=2))
    else:
        color_diff = temporal_diff
    combined = temporal_weight * temporal_diff + local_weight * local_diff + color_weight * color_diff
    threshold = robust_threshold_map(combined, base_threshold, mad_scale)
    mask_u8 = ((combined > threshold).astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if dilate_iter > 0:
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iter)
    return mask_u8.astype(bool)


def reliability_local_cleanup(frames: Sequence[np.ndarray], radius: int = 6, top_k: int = 3, base_threshold: float = 22, mad_scale: float = 3.0, blend_strength: float = 0.55, max_mask_ratio: float = 0.25, dilate_iter: int = 1) -> List[np.ndarray]:
    _require_cv2()
    frames_u8 = [ensure_u8(f) for f in frames]
    cleaned: List[np.ndarray] = []
    for i, cur in enumerate(frames_u8):
        ref = build_nearby_temporal_reference(frames_u8, index=i, radius=radius, top_k=top_k)
        mask = detect_remaining_artifact_mask(cur, ref, base_threshold, mad_scale, dilate_iter=dilate_iter)
        mask_ratio = float(mask.mean())
        cur_f = cur.astype(np.float32)
        ref_f = ref.astype(np.float32)
        if mask_ratio > max_mask_ratio:
            out = 0.88 * cur_f + 0.12 * ref_f
        else:
            candidate = (1.0 - blend_strength) * cur_f + blend_strength * ref_f
            if cur.ndim == 3:
                out = np.where(mask[:, :, None], candidate, cur_f)
            else:
                out = np.where(mask, candidate, cur_f)
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        cleaned.append(cv2.medianBlur(out_u8, 3))
    return cleaned


def normalize_col(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    xmin = np.nanmin(arr)
    xmax = np.nanmax(arr)
    if abs(xmax - xmin) < 1e-9:
        return np.ones_like(arr)
    return (arr - xmin) / (xmax - xmin)


def choose_best_visual_result(results: Sequence[VisualDNAResult]) -> VisualDNAResult:
    psnr_n = normalize_col([r.video_psnr for r in results])
    ssim_n = normalize_col([r.video_ssim for r in results])
    corr_n = normalize_col([r.frame_correlation for r in results])
    scores = 0.45 * psnr_n + 0.35 * ssim_n + 0.20 * corr_n
    return results[int(np.nanargmax(scores))]


# =========================================================
# Full visual pipeline
# =========================================================

def _make_visual_result(method: str, fps: float, config: VideoDNAConfig, playable: bool, output_video: str, raw_rgb_bytes: int, payload_bytes: int, dna_length_nt: int, added_substitutions: int, metrics: Dict[str, float]) -> VisualDNAResult:
    return VisualDNAResult(
        method=method,
        fps=float(fps),
        resolution=f"{config.target_width}x{config.target_height}",
        playable=bool(playable),
        payload_bytes=int(payload_bytes),
        dna_length_nt=int(dna_length_nt),
        added_substitutions=int(added_substitutions),
        compression_vs_raw_rgb=float(raw_rgb_bytes / max(1, payload_bytes)),
        video_psnr=float(metrics["video_psnr"]),
        video_ssim=float(metrics["video_ssim"]),
        frame_correlation=float(metrics["frame_correlation"]),
        output_video=str(output_video),
        frames_compared=int(metrics["frames_compared"]),
        raw_rgb_bytes=int(raw_rgb_bytes),
    )


def run_best_visual_video_dna(input_video: str | Path, config: Optional[VideoDNAConfig] = None) -> Dict[str, Any]:
    if config is None:
        config = VideoDNAConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed)
    video_info = ffprobe_video(input_video)
    if video_info is None:
        raise ValueError("Could not read video stream with ffprobe.")
    target_fps = choose_target_fps(video_info, config)
    frame_dir = out_dir / "original_frames"
    extract_frames(input_video, frame_dir, target_fps, config.target_width, config.target_height, config.max_seconds)
    original_frames = load_frames(frame_dir, color="rgb")
    if not original_frames:
        raise ValueError("No frames extracted from video.")
    original_preview = out_dir / "original_preview.mp4"
    make_video_from_frames(frame_dir, original_preview, fps=target_fps)
    raw_rgb_bytes = frame_raw_size_bytes(original_frames)

    # Base
    base_frames, base_meta = run_color_wavelet_visual_noecc(original_frames, config=config, rng=rng, method_name="base_simple_mapping")
    base_dir = out_dir / "base_frames"
    save_frames(base_frames, base_dir)
    base_video = out_dir / "base_simple_mapping.mp4"
    make_video_from_frames(base_dir, base_video, target_fps)
    base_metrics = compute_video_metrics(original_frames, base_frames)
    base_result = _make_visual_result("base_simple_mapping", target_fps, config, playable_video(base_video), str(base_video), raw_rgb_bytes, base_meta["payload_bytes"], base_meta["dna_length_nt"], base_meta["added_substitutions"], base_metrics)

    # Temporal similarity repair
    temporal_frames = temporal_similarity_repair(
        base_frames,
        radius=config.repair_radius,
        top_k=config.repair_top_k,
        base_threshold=config.repair_base_threshold,
        mad_scale=config.repair_mad_scale,
        blend_strength=config.repair_blend_strength,
        max_mask_ratio=config.repair_max_mask_ratio,
    )
    temporal_dir = out_dir / "temporal_similarity_frames"
    save_frames(temporal_frames, temporal_dir)
    temporal_video = out_dir / "temporal_similarity_repair.mp4"
    make_video_from_frames(temporal_dir, temporal_video, target_fps)
    temporal_metrics = compute_video_metrics(original_frames, temporal_frames)
    temporal_result = _make_visual_result("temporal_similarity_repair", target_fps, config, playable_video(temporal_video), str(temporal_video), raw_rgb_bytes, base_result.payload_bytes, base_result.dna_length_nt, base_result.added_substitutions, temporal_metrics)

    # Reliability cleanup
    cleanup_result = None
    if config.cleanup_enabled:
        cleanup_frames = reliability_local_cleanup(
            temporal_frames,
            radius=config.cleanup_radius,
            top_k=config.cleanup_top_k,
            base_threshold=config.cleanup_base_threshold,
            mad_scale=config.cleanup_mad_scale,
            blend_strength=config.cleanup_blend_strength,
            max_mask_ratio=config.cleanup_max_mask_ratio,
        )
        cleanup_dir = out_dir / "reliability_cleanup_frames"
        save_frames(cleanup_frames, cleanup_dir)
        cleanup_video = out_dir / "temporal_similarity_plus_reliability_cleanup.mp4"
        make_video_from_frames(cleanup_dir, cleanup_video, target_fps)
        cleanup_metrics = compute_video_metrics(original_frames, cleanup_frames)
        cleanup_result = _make_visual_result("temporal_similarity_plus_reliability_cleanup", target_fps, config, playable_video(cleanup_video), str(cleanup_video), raw_rgb_bytes, base_result.payload_bytes, base_result.dna_length_nt, base_result.added_substitutions, cleanup_metrics)

    candidates = [base_result, temporal_result]
    if cleanup_result is not None:
        candidates.append(cleanup_result)
    best_result = choose_best_visual_result(candidates)
    summary = [asdict(r) for r in candidates]
    summary_path = out_dir / "visual_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {
        "config": asdict(config),
        "video_info": video_info,
        "target_fps": float(target_fps),
        "original_preview": str(original_preview),
        "base_result": asdict(base_result),
        "temporal_result": asdict(temporal_result),
        "cleanup_result": asdict(cleanup_result) if cleanup_result else None,
        "best_result": asdict(best_result),
        "summary": summary,
        "summary_json": str(summary_path),
        "original_frames_count": len(original_frames),
    }


# =========================================================
# Optional no-ECC audio branch
# =========================================================

def extract_audio_wav(input_video: str | Path, output_wav: str | Path, sample_rate: int = 16000, max_seconds: Optional[float] = None) -> str:
    cmd = ["ffmpeg", "-y"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", str(input_video), "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", str(output_wav)]
    run_cmd(cmd, quiet=True)
    return str(output_wav)


def pcm16_to_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 128.0) / 128.0
    return x.astype(np.float32)


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def mulaw_encode_float(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5)
    return np.clip(q, 0, 255).astype(np.uint8)


def mulaw_decode_uint8(q: np.ndarray, mu: int = 255) -> np.ndarray:
    q = np.asarray(q, dtype=np.uint8)
    y = (q.astype(np.float32) / 127.5) - 1.0
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def light_audio_cleanup(x: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    _require_audio()
    x = np.asarray(x, dtype=np.float32)
    x_med = medfilt(x, kernel_size=5).astype(np.float32)
    cutoff = min(6000.0, sample_rate * 0.45)
    b, a = butter(4, cutoff / (sample_rate / 2), btype="low")
    try:
        x_filt = filtfilt(b, a, x_med).astype(np.float32)
    except Exception:
        x_filt = x_med
    return np.clip(x_filt, -1.0, 1.0)


def audio_snr_db(original: np.ndarray, recovered: np.ndarray) -> float:
    n = min(len(original), len(recovered))
    if n == 0:
        return float("nan")
    a = np.asarray(original[:n], dtype=np.float32)
    b = np.asarray(recovered[:n], dtype=np.float32)
    noise = a - b
    return float(10 * np.log10((np.mean(a ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12)))


def run_audio_dna_noecc(input_video: str | Path, output_dir: str | Path, sub_rate: float = 0.02, sample_rate: int = 16000, max_seconds: Optional[float] = None, random_seed: int = 42, method_name: str = "audio_mulaw_noecc") -> Optional[AudioDNAResult]:
    _require_audio()
    if not has_audio_stream(input_video):
        return None
    rng = np.random.default_rng(random_seed)
    out_dir = Path(output_dir) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_wav = out_dir / "original_audio_mono.wav"
    extract_audio_wav(input_video, extracted_wav, sample_rate=sample_rate, max_seconds=max_seconds)
    sr, audio_pcm = wavfile.read(extracted_wav)
    audio_float = pcm16_to_float(audio_pcm)
    audio_q = mulaw_encode_float(audio_float)
    payload = audio_q.tobytes()
    noisy_payload, dna_len, n_errors = transmit_through_dna_substitution(payload, sub_rate=sub_rate, rng=rng)
    noisy_q = np.frombuffer(noisy_payload, dtype=np.uint8)
    recovered_float_raw = mulaw_decode_uint8(noisy_q)[:len(audio_float)]
    recovered_float_clean = light_audio_cleanup(recovered_float_raw, sample_rate=sample_rate)
    recovered_wav_raw = out_dir / "recovered_audio_raw.wav"
    recovered_wav_clean = out_dir / "recovered_audio_clean.wav"
    wavfile.write(recovered_wav_raw, sample_rate, float_to_pcm16(recovered_float_raw))
    wavfile.write(recovered_wav_clean, sample_rate, float_to_pcm16(recovered_float_clean))
    return AudioDNAResult(
        method=method_name,
        sample_rate=int(sample_rate),
        payload_bytes=int(len(payload)),
        dna_length_nt=int(dna_len),
        added_substitutions=int(n_errors),
        substitution_rate=float(sub_rate),
        duration_sec=float(len(audio_float) / sample_rate),
        snr_raw_db=audio_snr_db(audio_float, recovered_float_raw),
        snr_clean_db=audio_snr_db(audio_float, recovered_float_clean),
        original_audio_wav=str(extracted_wav),
        recovered_audio_raw_wav=str(recovered_wav_raw),
        recovered_audio_clean_wav=str(recovered_wav_clean),
    )


def run_full_av_video_dna(input_video: str | Path, config: Optional[VideoDNAConfig] = None) -> Dict[str, Any]:
    if config is None:
        config = VideoDNAConfig(audio_enabled=True)
    visual_dict = run_best_visual_video_dna(input_video, config=config)
    best_visual = VisualDNAResult(**visual_dict["best_result"])
    out_dir = Path(config.output_dir)
    audio_result = None
    final_video = best_visual.output_video
    total_payload = best_visual.payload_bytes
    total_dna = best_visual.dna_length_nt
    if config.audio_enabled:
        audio_result = run_audio_dna_noecc(
            input_video=input_video,
            output_dir=out_dir,
            sub_rate=config.substitution_rate,
            sample_rate=config.audio_sample_rate,
            max_seconds=config.max_seconds,
            random_seed=config.random_seed,
        )
        if audio_result is not None:
            final_video = mux_video_audio(
                best_visual.output_video,
                audio_result.recovered_audio_clean_wav,
                out_dir / "final_visual_plus_dna_audio.mp4",
                audio_bitrate=config.audio_bitrate_for_mux,
            )
            total_payload += audio_result.payload_bytes
            total_dna += audio_result.dna_length_nt
    full_dict = {
        "visual_pipeline": visual_dict,
        "audio_result": asdict(audio_result) if audio_result else None,
        "full_result": {
            "visual": asdict(best_visual),
            "audio": asdict(audio_result) if audio_result else None,
            "total_payload_bytes": int(total_payload),
            "total_dna_nt": int(total_dna),
            "final_video": str(final_video),
            "output_dir": str(out_dir),
        },
    }
    with open(out_dir / "full_av_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_dict, f, indent=2)
    return full_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="No-ECC color wavelet video DNA compressor v2.1")
    parser.add_argument("input_video", type=str)
    parser.add_argument("--output_dir", type=str, default="video_dna_v21_output")
    parser.add_argument("--max_seconds", type=float, default=4.0)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--fps_policy", type=str, default="original", choices=["original", "comfortable", "manual"])
    parser.add_argument("--max_allowed_fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=108)
    parser.add_argument("--substitution_rate", type=float, default=0.02)
    args = parser.parse_args()
    cfg = VideoDNAConfig(
        output_dir=args.output_dir,
        max_seconds=args.max_seconds,
        audio_enabled=args.audio,
        fps_policy=args.fps_policy,
        max_allowed_fps=args.max_allowed_fps,
        target_width=args.width,
        target_height=args.height,
        substitution_rate=args.substitution_rate,
    )
    result = run_full_av_video_dna(args.input_video, cfg)
    print(json.dumps(result["full_result"], indent=2))
