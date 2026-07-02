#!/usr/bin/env python3
"""
two_best_dna_image_codecs.py
============================

Clean final project with ONLY the two best no-ECC image-to-DNA methods:

1. hybrid_webp_highbase
   Practical method:
       strong fixed-length base layer
       + WebP tile enhancement
       + CRC32 per WebP tile
       If a WebP tile is corrupted by DNA substitution, that region falls back
       to the base layer. Good graceful degradation.

2. local_dct_ycbcr_strong
   Scientific/local method:
       RGB -> YCbCr
       Cb/Cr 4:2:0 downsampling
       local 8x8 DCT fixed packets
       bounded RLE coefficients
       CRC8 per local block packet
       No JPEG/WebP stream. Errors remain local at block/packet level.

Removed from this clean version:
    direct_jpeg
    direct_webp
    jpeg_tile_only
    webp_tile_only
    weaker hybrid/local variants

No ECC:
    No Reed-Solomon
    No fountain code
    No parity packet
    No majority-vote recovery

CRC is detection-only, not correction.

Install:
    pip install numpy pillow scipy scikit-image matplotlib pandas

Colab usage:
    from two_best_dna_image_codecs import run_benchmark, list_methods

    df = run_benchmark(
        image_path="blog_lofoten_islands.jpg",
        out_dir="two_best_results",
        methods="all",
        error_rates=[0, 0.0001, 0.001, 0.005, 0.01],
        show=True,
    )

CLI:
    python two_best_dna_image_codecs.py --image blog_lofoten_islands.jpg --out results --methods all --show
"""

from __future__ import annotations

__version__ = "two-best-dna-image-codecs-2026-06-02"

import argparse
import io
import json
import math
import os
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from skimage import metrics
except Exception:
    metrics = None

try:
    from scipy.fftpack import dct, idct
    from scipy.ndimage import median_filter
except Exception:
    dct = None
    idct = None
    median_filter = None

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


# =============================================================================
# DNA utilities
# =============================================================================

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

DNA_TO_VAL = np.full(256, 0, dtype=np.uint8)
DNA_TO_VAL[ord("A")] = 0
DNA_TO_VAL[ord("C")] = 1
DNA_TO_VAL[ord("G")] = 2
DNA_TO_VAL[ord("T")] = 3


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray, n_bytes: Optional[int] = None) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size % 8:
        bits = np.pad(bits, (0, 8 - bits.size % 8))
    arr = np.packbits(bits)
    if n_bytes is not None:
        arr = arr[:int(n_bytes)]
    return arr.tobytes()


def bits_to_dna(bits: np.ndarray) -> Tuple[str, int]:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = 0
    if bits.size % 2:
        bits = np.append(bits, 0).astype(np.uint8)
        pad = 1
    vals = bits[0::2] * 2 + bits[1::2]
    return BASES[vals].tobytes().decode("ascii"), pad


def dna_to_bits(dna: str, pad: int = 0) -> np.ndarray:
    clean = "".join(str(dna).split()).upper()
    vals = DNA_TO_VAL[np.frombuffer(clean.encode("ascii"), dtype=np.uint8)]
    bits = np.empty(vals.size * 2, dtype=np.uint8)
    bits[0::2] = vals >> 1
    bits[1::2] = vals & 1
    if pad:
        bits = bits[:-int(pad)]
    return bits


def substitute_dna(dna: str, rate: float = 0.001, seed: int = 123) -> Tuple[str, int]:
    """
    Substitute-only DNA error model.
    Length is unchanged. A/C/G/T base is randomly changed into another base.
    """
    rng = np.random.default_rng(seed)
    clean = "".join(str(dna).split()).upper()
    vals = DNA_TO_VAL[np.frombuffer(clean.encode("ascii"), dtype=np.uint8)].copy()

    mask = rng.random(vals.size) < float(rate)
    n = int(mask.sum())

    if n:
        delta = rng.integers(1, 4, size=n, dtype=np.uint8)
        vals[mask] = (vals[mask] + delta) % 4

    return BASES[vals].tobytes().decode("ascii"), n


def uints_to_bits(values: np.ndarray, n_bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32).ravel()
    values = np.clip(values, 0, 2 ** int(n_bits) - 1)
    shifts = np.arange(int(n_bits) - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def bits_to_uints(bits: np.ndarray, n_bits: int, n_values: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    needed = int(n_bits) * int(n_values)

    if bits.size < needed:
        bits = np.pad(bits, (0, needed - bits.size))
    else:
        bits = bits[:needed]

    mat = bits.reshape(int(n_values), int(n_bits))
    weights = (1 << np.arange(int(n_bits) - 1, -1, -1)).astype(np.uint32)
    return (mat * weights).sum(axis=1).astype(np.uint32)


def signed_to_bits(values: Sequence[int], n_bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32).ravel()
    offset = 2 ** (int(n_bits) - 1)
    values = np.clip(values, -offset, offset - 1)
    return uints_to_bits(values + offset, int(n_bits))


def bits_to_signed(bits: np.ndarray, n_bits: int, n_values: int) -> np.ndarray:
    u = bits_to_uints(bits, int(n_bits), int(n_values)).astype(np.int32)
    offset = 2 ** (int(n_bits) - 1)
    return u - offset


# =============================================================================
# General utilities
# =============================================================================

def image_info(image_path: str) -> Dict[str, Any]:
    im = Image.open(image_path)
    mode = "L" if im.mode == "L" else "RGB"
    w, h = im.size
    channels = 1 if mode == "L" else 3
    raw_bits = int(w * h * channels * 8)

    return {
        "image_path": image_path,
        "width": int(w),
        "height": int(h),
        "mode": mode,
        "channels": int(channels),
        "original_file_bytes": int(os.path.getsize(image_path)),
        "raw_bits": int(raw_bits),
        "raw_bytes": float(raw_bits / 8),
        "raw_dna_nt": float(raw_bits / 2),
    }


def compute_metrics(original_path: str, reconstructed_path: str) -> Tuple[float, float]:
    if metrics is None:
        return float("nan"), float("nan")

    orig = np.asarray(Image.open(original_path).convert("RGB"), dtype=np.uint8)
    rec = np.asarray(Image.open(reconstructed_path).convert("RGB"), dtype=np.uint8)

    psnr = metrics.peak_signal_noise_ratio(orig, rec, data_range=255)
    ssim = metrics.structural_similarity(orig, rec, data_range=255, channel_axis=-1)
    return float(psnr), float(ssim)


def add_crc32(data: bytes) -> bytes:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return data + crc.to_bytes(4, "big")


def check_crc32(packet: bytes) -> Tuple[bytes, bool]:
    if len(packet) < 4:
        return b"", False

    data = packet[:-4]
    crc_read = int.from_bytes(packet[-4:], "big")
    crc_true = zlib.crc32(data) & 0xFFFFFFFF
    return data, crc_read == crc_true


def pad_image_to_tile(im: Image.Image, tile_size: int) -> Tuple[Image.Image, int, int]:
    im = im.convert("RGB")
    w, h = im.size

    pad_w = (int(tile_size) - w % int(tile_size)) % int(tile_size)
    pad_h = (int(tile_size) - h % int(tile_size)) % int(tile_size)

    if pad_w == 0 and pad_h == 0:
        return im.copy(), 0, 0

    new_w = w + pad_w
    new_h = h + pad_h

    out = Image.new("RGB", (new_w, new_h))
    out.paste(im, (0, 0))

    if pad_w > 0:
        right_strip = im.crop((w - 1, 0, w, h)).resize((pad_w, h))
        out.paste(right_strip, (w, 0))

    if pad_h > 0:
        bottom_strip = out.crop((0, h - 1, new_w, h)).resize((new_w, pad_h))
        out.paste(bottom_strip, (0, h))

    return out, pad_h, pad_w


def save_webp_bytes(tile: Image.Image, quality: int = 45) -> bytes:
    bio = io.BytesIO()
    tile.convert("RGB").save(bio, format="WEBP", quality=int(quality), method=6)
    return bio.getvalue()


# =============================================================================
# METHOD 1: hybrid_webp_highbase
# =============================================================================

def encode_base_layer(im: Image.Image, base_downsample: int = 4, base_bits: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
    im = im.convert("RGB")
    w, h = im.size

    bw = max(1, int(math.ceil(w / int(base_downsample))))
    bh = max(1, int(math.ceil(h / int(base_downsample))))

    base = im.resize((bw, bh), RESAMPLE_BICUBIC)
    arr = np.asarray(base, dtype=np.uint8)

    levels = 2 ** int(base_bits) - 1
    q = np.round(arr.astype(np.float32) / 255.0 * levels).astype(np.uint32)
    bits = uints_to_bits(q.ravel(), int(base_bits))

    meta = {
        "base_shape": [int(bh), int(bw)],
        "base_downsample": int(base_downsample),
        "base_bits": int(base_bits),
        "base_values": int(q.size),
        "base_payload_bits": int(bits.size),
        "base_payload_bytes": float(bits.size / 8),
    }
    return bits, meta


def decode_base_layer(bits: np.ndarray, base_meta: Dict[str, Any], full_shape_hw: Tuple[int, int]) -> Image.Image:
    bh, bw = base_meta["base_shape"]
    base_bits = int(base_meta["base_bits"])
    n_values = int(bh * bw * 3)

    q = bits_to_uints(bits, base_bits, n_values)
    levels = 2 ** base_bits - 1

    arr = np.round(q.astype(np.float32) / levels * 255.0).astype(np.uint8).reshape(bh, bw, 3)
    base = Image.fromarray(arr, mode="RGB")

    h, w = full_shape_hw
    return base.resize((int(w), int(h)), RESAMPLE_BICUBIC)


def encode_webp_tile_payload(
    image_path: str,
    tile_size: int = 128,
    quality: int = 45,
    use_crc: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    im = Image.open(image_path).convert("RGB")
    w0, h0 = im.size

    padded, pad_h, pad_w = pad_image_to_tile(im, int(tile_size))
    wp, hp = padded.size

    tiles_x = wp // int(tile_size)
    tiles_y = hp // int(tile_size)

    packets: List[bytes] = []
    offsets: List[List[int]] = []
    tile_meta: List[Dict[str, Any]] = []
    cursor = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x = tx * int(tile_size)
            y = ty * int(tile_size)

            tile = padded.crop((x, y, x + int(tile_size), y + int(tile_size)))
            data = save_webp_bytes(tile, quality=int(quality))
            packet = add_crc32(data) if use_crc else data

            packets.append(packet)
            offsets.append([int(cursor), int(cursor + len(packet))])
            cursor += len(packet)

            tile_meta.append({
                "tx": int(tx),
                "ty": int(ty),
                "x": int(x),
                "y": int(y),
                "webp_bytes": int(len(data)),
                "packet_bytes": int(len(packet)),
            })

    payload = b"".join(packets)

    meta = {
        "format": "WEBP",
        "quality": int(quality),
        "tile_size": int(tile_size),
        "use_crc": bool(use_crc),
        "shape": [int(h0), int(w0)],
        "padded_shape": [int(hp), int(wp)],
        "tiles_x": int(tiles_x),
        "tiles_y": int(tiles_y),
        "n_tiles": int(tiles_x * tiles_y),
        "offsets": offsets,
        "tile_meta": tile_meta,
        "tile_payload_bytes": int(len(payload)),
        "tile_payload_bits": int(len(payload) * 8),
    }
    return payload, meta


def encode_hybrid_webp_highbase(
    image_path: str,
    tile_size: int = 128,
    quality: int = 45,
    base_downsample: int = 4,
    base_bits: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    im = Image.open(image_path).convert("RGB")
    h0, w0 = im.size[1], im.size[0]

    base_bits_arr, base_meta = encode_base_layer(
        im,
        base_downsample=base_downsample,
        base_bits=base_bits,
    )

    tile_payload, tile_meta = encode_webp_tile_payload(
        image_path,
        tile_size=tile_size,
        quality=quality,
        use_crc=True,
    )

    tile_bits = bytes_to_bits(tile_payload)
    payload_bits = np.concatenate([base_bits_arr, tile_bits]).astype(np.uint8)

    dna, pad = bits_to_dna(payload_bits)
    info = image_info(image_path)

    meta = {
        "method": "hybrid_webp_highbase",
        "method_type": "practical_hybrid",
        "codec": "hybrid_base_webp_tile_no_ecc",
        "shape": [int(h0), int(w0)],
        "tile_size": int(tile_size),
        "webp_quality": int(quality),
        "base_downsample": int(base_downsample),
        "base_bits": int(base_bits),
        "base": base_meta,
        "tile": tile_meta,
        "base_payload_bits": int(base_bits_arr.size),
        "base_payload_bytes": float(base_bits_arr.size / 8),
        "tile_payload_bytes": int(len(tile_payload)),
        "payload_bits": int(payload_bits.size),
        "payload_bytes": float(payload_bits.size / 8),
        "dna_nt": int(len(dna)),
        "dna_pad": int(pad),
        "raw_dna_nt": float(info["raw_dna_nt"]),
        "dna_reduction_vs_raw_pixels": float(info["raw_dna_nt"] / max(1, len(dna))),
        "payload_vs_original_file": float((payload_bits.size / 8) / max(1, info["original_file_bytes"])),
        "original_file_bytes": int(info["original_file_bytes"]),
    }
    return dna, meta


def decode_hybrid_webp_highbase(
    dna: str,
    meta: Dict[str, Any],
    output_path: str,
    substitution_rate: float = 0.0,
    seed: int = 123,
    repair: bool = True,
) -> Dict[str, Any]:
    n_sub = 0
    if substitution_rate > 0:
        dna, n_sub = substitute_dna(dna, substitution_rate, seed=seed)

    bits = dna_to_bits(dna, pad=int(meta["dna_pad"]))
    payload_bits = int(meta["payload_bits"])

    if bits.size < payload_bits:
        bits = np.pad(bits, (0, payload_bits - bits.size))
    else:
        bits = bits[:payload_bits]

    h0, w0 = meta["shape"]
    base_len = int(meta["base_payload_bits"])

    base_bits_arr = bits[:base_len]
    tile_bits = bits[base_len:]

    base_img = decode_base_layer(base_bits_arr, meta["base"], full_shape_hw=(int(h0), int(w0)))

    tile_meta = meta["tile"]
    tile_payload = bits_to_bytes(tile_bits, n_bytes=int(meta["tile_payload_bytes"]))

    tile_size = int(tile_meta["tile_size"])
    padded_base, _, _ = pad_image_to_tile(base_img, tile_size)
    canvas = padded_base.copy()

    valid = 0
    invalid = 0
    decode_failed = 0

    for (start, end), tm in zip(tile_meta["offsets"], tile_meta["tile_meta"]):
        packet = tile_payload[int(start):int(end)]
        data, ok_crc = check_crc32(packet)

        if ok_crc:
            try:
                tile = Image.open(io.BytesIO(data)).convert("RGB")
                tile.load()
                canvas.paste(tile, (int(tm["x"]), int(tm["y"])))
                valid += 1
            except Exception:
                invalid += 1
                decode_failed += 1
                # keep base layer in this region
        else:
            invalid += 1
            # keep base layer in this region

    out = canvas.crop((0, 0, int(w0), int(h0)))

    if repair:
        out = out.filter(ImageFilter.MedianFilter(size=3))

    out.save(output_path)

    return {
        "simulated_substituted_bases": int(n_sub),
        "decode_success": True,
        "valid_units": int(valid),
        "invalid_units": int(invalid),
        "decode_failed_units": int(decode_failed),
        "failure_rate": float(invalid / max(1, valid + invalid)),
    }


# =============================================================================
# METHOD 2: local_dct_ycbcr_strong
# =============================================================================

JPEG_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float32)

JPEG_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


def require_scipy() -> None:
    if dct is None or idct is None:
        raise RuntimeError("scipy is required for local_dct_ycbcr_strong. Install scipy.")


def dct2(block: np.ndarray) -> np.ndarray:
    require_scipy()
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    require_scipy()
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def zigzag_indices(n: int = 8) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = range(min(s, n - 1), max(-1, s - n), -1)
        else:
            r = range(max(0, s - n + 1), min(s, n - 1) + 1)
        for i in r:
            j = s - i
            if 0 <= i < n and 0 <= j < n:
                out.append((i, j))
    return out


ZZ8 = zigzag_indices(8)


def crc8_bits(bits: np.ndarray) -> int:
    crc = 0
    for b in np.asarray(bits, dtype=np.uint8).ravel():
        crc ^= int(b) << 7
        if crc & 0x80:
            crc = ((crc << 1) ^ 0x07) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    return int(crc)


def add_crc8(data_bits: np.ndarray) -> np.ndarray:
    return np.concatenate([data_bits.astype(np.uint8), uints_to_bits([crc8_bits(data_bits)], 8)]).astype(np.uint8)


def check_crc8(packet_bits: np.ndarray, data_len: int) -> Tuple[bool, np.ndarray]:
    packet_bits = np.asarray(packet_bits, dtype=np.uint8).ravel()
    needed = int(data_len) + 8

    if packet_bits.size < needed:
        packet_bits = np.pad(packet_bits, (0, needed - packet_bits.size))
    else:
        packet_bits = packet_bits[:needed]

    data_bits = packet_bits[:int(data_len)]
    crc_read = int(bits_to_uints(packet_bits[int(data_len):int(data_len) + 8], 8, 1)[0])
    return crc_read == crc8_bits(data_bits), data_bits


def coeff_to_code(v: int) -> int:
    v = int(np.clip(v, -31, 31))
    if v == 0:
        return 0
    return v if v > 0 else 31 + abs(v)


def code_to_coeff(code: int) -> int:
    code = int(code)
    if code == 0:
        return 0
    if 1 <= code <= 31:
        return code
    if 32 <= code <= 62:
        return -(code - 31)
    return 0


def token_to_bits(run: int, code: int) -> np.ndarray:
    return np.concatenate([uints_to_bits([run], 4), uints_to_bits([code], 6)]).astype(np.uint8)


def bits_to_token(bits10: np.ndarray) -> Tuple[int, int]:
    return int(bits_to_uints(bits10[:4], 4, 1)[0]), int(bits_to_uints(bits10[4:10], 6, 1)[0])


def encode_ac_rle(ac_values: np.ndarray, token_count: int) -> np.ndarray:
    ac_values = np.asarray(ac_values, dtype=np.int32).ravel()
    ac_values = np.clip(ac_values, -31, 31)

    tokens: List[Tuple[int, int]] = []
    zero_run = 0

    for v in ac_values:
        v = int(v)
        if v == 0:
            zero_run += 1
            continue

        while zero_run > 15 and len(tokens) < token_count:
            tokens.append((15, 0))
            zero_run -= 15

        if len(tokens) >= token_count:
            break

        tokens.append((min(zero_run, 15), coeff_to_code(v)))
        zero_run = 0

    if len(tokens) < token_count:
        tokens.append((0, 0))

    while len(tokens) < token_count:
        tokens.append((0, 0))

    return np.concatenate([token_to_bits(r, c) for r, c in tokens]).astype(np.uint8)


def decode_ac_rle(data_bits: np.ndarray, token_count: int) -> np.ndarray:
    data_bits = np.asarray(data_bits, dtype=np.uint8).ravel()
    needed = int(token_count) * 10

    if data_bits.size < needed:
        data_bits = np.pad(data_bits, (0, needed - data_bits.size))
    else:
        data_bits = data_bits[:needed]

    ac = np.zeros(63, dtype=np.int32)
    pos = 0

    for t in range(int(token_count)):
        run, code = bits_to_token(data_bits[t * 10:(t + 1) * 10])

        if run == 0 and code == 0:
            break

        if code == 0 and run > 0:
            pos += run
            if pos >= 63:
                break
            continue

        pos += run
        if pos >= 63:
            break

        ac[pos] = code_to_coeff(code)
        pos += 1

    return ac


def pad_to_multiple(arr: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]

    pad_h = (int(multiple) - h % int(multiple)) % int(multiple)
    pad_w = (int(multiple) - w % int(multiple)) % int(multiple)

    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded.astype(np.uint8), int(pad_h), int(pad_w)


def downsample_420(ch: np.ndarray) -> np.ndarray:
    im = Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8), mode="L")
    w, h = im.size
    return np.asarray(im.resize(((w + 1) // 2, (h + 1) // 2), RESAMPLE_BICUBIC), dtype=np.uint8)


def upsample_to(ch: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    im = Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(im.resize((w, h), RESAMPLE_BICUBIC), dtype=np.uint8)


def make_quant_table(kind: str, q_scale: float) -> np.ndarray:
    base = JPEG_LUMA_Q if kind == "luma" else JPEG_CHROMA_Q
    return np.maximum(1, np.round(base * float(q_scale))).astype(np.float32)


def encode_channel_packets(
    ch: np.ndarray,
    q_scale: float,
    packet_bits: int,
    kind: str = "luma",
    dc_bits: int = 12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ch = np.asarray(ch, dtype=np.uint8)
    padded, pad_h, pad_w = pad_to_multiple(ch, 8)

    h, w = padded.shape
    qtab = make_quant_table(kind, q_scale)

    token_count = max(1, (int(packet_bits) - int(dc_bits) - 8) // 10)
    data_len = int(dc_bits) + int(token_count) * 10
    actual_packet_bits = data_len + 8

    packets = []

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = padded[y:y + 8, x:x + 8].astype(np.float32) - 128.0
            coeff = np.round(dct2(block) / qtab).astype(np.int32)
            zz = np.array([coeff[i, j] for i, j in ZZ8], dtype=np.int32)

            dc = signed_to_bits([int(zz[0])], int(dc_bits))
            ac = encode_ac_rle(zz[1:], token_count)

            data = np.concatenate([dc, ac]).astype(np.uint8)
            packets.append(add_crc8(data))

    bits = np.concatenate(packets).astype(np.uint8)

    meta = {
        "original_shape": list(ch.shape),
        "padded_shape": [int(h), int(w)],
        "q_scale": float(q_scale),
        "kind": str(kind),
        "packet_bits": int(actual_packet_bits),
        "data_len": int(data_len),
        "dc_bits": int(dc_bits),
        "token_count": int(token_count),
        "payload_bits": int(bits.size),
        "n_blocks": int((h // 8) * (w // 8)),
    }
    return bits, meta


def decode_channel_packets(bits: np.ndarray, meta: Dict[str, Any], repair: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    bits = np.asarray(bits, dtype=np.uint8).ravel()

    h, w = meta["padded_shape"]
    qtab = make_quant_table(meta["kind"], meta["q_scale"])

    rec = np.zeros((h, w), dtype=np.float32)
    idx = 0
    valid = 0
    invalid = 0
    pb = int(meta["packet_bits"])

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            packet = bits[idx:idx + pb]
            idx += pb

            ok, data = check_crc8(packet, int(meta["data_len"]))

            qvec = np.zeros(64, dtype=np.int32)

            if ok:
                valid += 1
                qvec[0] = bits_to_signed(data[:int(meta["dc_bits"])], int(meta["dc_bits"]), 1)[0]
                qvec[1:] = decode_ac_rle(data[int(meta["dc_bits"]):], int(meta["token_count"]))
            else:
                invalid += 1
                # Keep neutral 128 block by leaving all coefficients zero.

            qcoeff = np.zeros((8, 8), dtype=np.float32)
            for k, (i, j) in enumerate(ZZ8):
                qcoeff[i, j] = qvec[k]

            block = idct2(qcoeff * qtab) + 128.0
            rec[y:y + 8, x:x + 8] = block

    oh, ow = meta["original_shape"]
    out = np.clip(rec[:oh, :ow], 0, 255).astype(np.uint8)

    if repair and median_filter is not None:
        med = median_filter(out, size=3)
        diff = np.abs(out.astype(np.int16) - med.astype(np.int16))
        mask = diff > 65
        out = out.copy()
        out[mask] = med[mask]

    return out, {
        "valid_packets": int(valid),
        "invalid_packets": int(invalid),
    }


def encode_local_dct_ycbcr_strong(
    image_path: str,
    y_q_scale: float = 1.5,
    c_q_scale: float = 2.5,
    y_packet_bits: int = 64,
    c_packet_bits: int = 24,
) -> Tuple[str, Dict[str, Any]]:
    require_scipy()

    im = Image.open(image_path)
    original_mode = "L" if im.mode == "L" else "RGB"

    if original_mode == "L":
        channels = [np.asarray(im.convert("L"), dtype=np.uint8)]
        names = ["L"]
    else:
        ycbcr = im.convert("YCbCr")
        y, cb, cr = [np.asarray(ch, dtype=np.uint8) for ch in ycbcr.split()]
        channels = [y, downsample_420(cb), downsample_420(cr)]
        names = ["Y", "Cb", "Cr"]

    bits_parts = []
    channel_metas = []
    offsets = []
    cursor = 0

    for name, ch in zip(names, channels):
        if name in ["Y", "L"]:
            q = y_q_scale
            pb = y_packet_bits
            kind = "luma"
        else:
            q = c_q_scale
            pb = c_packet_bits
            kind = "chroma"

        b, m = encode_channel_packets(
            ch,
            q_scale=q,
            packet_bits=pb,
            kind=kind,
        )

        offsets.append([int(cursor), int(cursor + b.size)])
        cursor += int(b.size)
        m["name"] = name

        bits_parts.append(b)
        channel_metas.append(m)

    payload_bits = np.concatenate(bits_parts).astype(np.uint8)
    dna, pad = bits_to_dna(payload_bits)
    info = image_info(image_path)

    meta = {
        "method": "local_dct_ycbcr_strong",
        "method_type": "scientific_local",
        "codec": "local_dct_ycbcr_fixed_packet_no_ecc",
        "original_mode": original_mode,
        "shape": [int(info["height"]), int(info["width"])],
        "channel_names": names,
        "channel_offsets": offsets,
        "channels": channel_metas,
        "params": {
            "y_q_scale": float(y_q_scale),
            "c_q_scale": float(c_q_scale),
            "y_packet_bits": int(y_packet_bits),
            "c_packet_bits": int(c_packet_bits),
            "chroma_mode": "420",
        },
        "payload_bits": int(payload_bits.size),
        "payload_bytes": float(payload_bits.size / 8),
        "dna_nt": int(len(dna)),
        "dna_pad": int(pad),
        "raw_dna_nt": float(info["raw_dna_nt"]),
        "dna_reduction_vs_raw_pixels": float(info["raw_dna_nt"] / max(1, len(dna))),
        "payload_vs_original_file": float((payload_bits.size / 8) / max(1, info["original_file_bytes"])),
        "original_file_bytes": int(info["original_file_bytes"]),
    }
    return dna, meta


def decode_local_dct_ycbcr_strong(
    dna: str,
    meta: Dict[str, Any],
    output_path: str,
    substitution_rate: float = 0.0,
    seed: int = 123,
    repair: bool = True,
) -> Dict[str, Any]:
    n_sub = 0
    if substitution_rate > 0:
        dna, n_sub = substitute_dna(dna, substitution_rate, seed=seed)

    bits = dna_to_bits(dna, pad=int(meta["dna_pad"]))

    if bits.size < int(meta["payload_bits"]):
        bits = np.pad(bits, (0, int(meta["payload_bits"]) - bits.size))
    else:
        bits = bits[:int(meta["payload_bits"])]

    decoded = {}
    valid_total = 0
    invalid_total = 0

    for (start, end), cm in zip(meta["channel_offsets"], meta["channels"]):
        ch_bits = bits[int(start):int(end)]
        ch, stats = decode_channel_packets(ch_bits, cm, repair=repair)

        decoded[cm["name"]] = ch
        valid_total += int(stats["valid_packets"])
        invalid_total += int(stats["invalid_packets"])

    h, w = meta["shape"]

    if meta["original_mode"] == "L":
        im = Image.fromarray(decoded["L"], mode="L").convert("RGB")
    else:
        y = decoded["Y"]
        cb = upsample_to(decoded["Cb"], (int(h), int(w)))
        cr = upsample_to(decoded["Cr"], (int(h), int(w)))

        im = Image.merge(
            "YCbCr",
            [
                Image.fromarray(y, mode="L"),
                Image.fromarray(cb, mode="L"),
                Image.fromarray(cr, mode="L"),
            ],
        ).convert("RGB")

    im.save(output_path)

    return {
        "simulated_substituted_bases": int(n_sub),
        "decode_success": True,
        "valid_units": int(valid_total),
        "invalid_units": int(invalid_total),
        "failure_rate": float(invalid_total / max(1, valid_total + invalid_total)),
    }


# =============================================================================
# Public method registry
# =============================================================================

METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "hybrid_webp_highbase": {
        "description": "Best practical method: strong base layer + WebP tile enhancement.",
        "method_type": "practical_hybrid",
    },
    "local_dct_ycbcr_strong": {
        "description": "Best scientific/local method: fixed-packet YCbCr DCT with local error containment.",
        "method_type": "scientific_local",
    },
}


def list_methods() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "method": name,
            "method_type": preset["method_type"],
            "description": preset["description"],
        }
        for name, preset in METHOD_PRESETS.items()
    ])


def encode_by_method(image_path: str, method: str) -> Tuple[str, Dict[str, Any]]:
    if method == "hybrid_webp_highbase":
        return encode_hybrid_webp_highbase(image_path)

    if method == "local_dct_ycbcr_strong":
        return encode_local_dct_ycbcr_strong(image_path)

    raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_PRESETS.keys())}")


def decode_by_method(
    dna: str,
    meta: Dict[str, Any],
    output_path: str,
    substitution_rate: float = 0.0,
    seed: int = 123,
) -> Dict[str, Any]:
    method = meta["method"]

    if method == "hybrid_webp_highbase":
        return decode_hybrid_webp_highbase(
            dna,
            meta,
            output_path,
            substitution_rate=substitution_rate,
            seed=seed,
        )

    if method == "local_dct_ycbcr_strong":
        return decode_local_dct_ycbcr_strong(
            dna,
            meta,
            output_path,
            substitution_rate=substitution_rate,
            seed=seed,
        )

    raise ValueError(f"Unknown method in metadata: {method}")


def resolve_methods(methods: Sequence[str] | str) -> List[str]:
    if methods == "all" or methods == ["all"]:
        return list(METHOD_PRESETS.keys())

    if isinstance(methods, str):
        if methods.strip().lower() == "all":
            return list(METHOD_PRESETS.keys())
        return [m.strip() for m in methods.split(",") if m.strip()]

    return list(methods)


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(
    image_path: str,
    out_dir: str = "two_best_results",
    methods: Sequence[str] | str = "all",
    error_rates: Sequence[float] = (0.0, 0.0001, 0.001, 0.005, 0.01),
    seed: int = 123,
    show: bool = False,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    method_list = resolve_methods(methods)

    rows = []

    for method in method_list:
        print(f"[encode] {method}")
        dna, meta = encode_by_method(image_path, method)

        dna_path = out / f"{method}.dna"
        meta_path = out / f"{method}.json"

        dna_path.write_text(dna, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        for err in error_rates:
            err_tag = str(err).replace(".", "p")
            rec_path = out / f"{method}_err_{err_tag}.png"

            stats = decode_by_method(
                dna=dna,
                meta=meta,
                output_path=str(rec_path),
                substitution_rate=float(err),
                seed=int(seed),
            )

            psnr, ssim = compute_metrics(image_path, str(rec_path))

            rows.append({
                "method": method,
                "method_type": meta["method_type"],
                "error_rate": float(err),
                "substituted_bases": int(stats["simulated_substituted_bases"]),
                "decode_success": bool(stats["decode_success"]),
                "failure_rate": float(stats["failure_rate"]),
                "valid_units": int(stats["valid_units"]),
                "invalid_units": int(stats["invalid_units"]),
                "payload_bytes": float(meta["payload_bytes"]),
                "payload_KB": float(meta["payload_bytes"] / 1024),
                "payload_vs_original_file": float(meta["payload_vs_original_file"]),
                "dna_nt": int(meta["dna_nt"]),
                "raw_dna_nt": float(meta["raw_dna_nt"]),
                "dna_reduction_vs_raw_pixels": float(meta["dna_reduction_vs_raw_pixels"]),
                "psnr": float(psnr),
                "ssim": float(ssim),
                "reconstructed_path": str(rec_path),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "two_best_benchmark_results.csv", index=False)

    if show:
        show_results(df, image_path, out)

    return df


def show_results(df: pd.DataFrame, image_path: str, out_dir: Path) -> None:
    if plt is None:
        print("matplotlib not available; skip plots.")
        return

    df0 = df[df["error_rate"] == 0.0].copy()

    if len(df0):
        plt.figure(figsize=(7, 5))
        plt.scatter(df0["payload_KB"], df0["psnr"], s=90)

        for _, r in df0.iterrows():
            plt.text(r["payload_KB"], r["psnr"], r["method"], fontsize=9)

        plt.xlabel("Payload size (KB)")
        plt.ylabel("PSNR, no DNA error")
        plt.title("Two best methods: compression vs no-error quality")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "payload_vs_psnr_no_error.png", dpi=180)
        plt.show()

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["error_rate"], sub["psnr"], marker="o", label=method)

    plt.xlabel("DNA substitution rate")
    plt.ylabel("PSNR")
    plt.title("PSNR vs DNA substitution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "psnr_vs_error.png", dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["error_rate"], sub["ssim"], marker="o", label=method)

    plt.xlabel("DNA substitution rate")
    plt.ylabel("SSIM")
    plt.title("SSIM vs DNA substitution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ssim_vs_error.png", dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["error_rate"], sub["failure_rate"], marker="o", label=method)

    plt.xlabel("DNA substitution rate")
    plt.ylabel("Failure rate")
    plt.title("Local failure rate vs DNA substitution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "failure_rate_vs_error.png", dpi=180)
    plt.show()

    target = 0.001
    sub = df[np.isclose(df["error_rate"], target)]

    if len(sub):
        cols = 3
        rows = 1
        plt.figure(figsize=(15, 5))

        plt.subplot(rows, cols, 1)
        plt.imshow(Image.open(image_path).convert("RGB"))
        plt.title("Original")
        plt.axis("off")

        for i, (_, r) in enumerate(sub.iterrows(), start=2):
            plt.subplot(rows, cols, i)
            plt.imshow(Image.open(r["reconstructed_path"]).convert("RGB"))
            plt.title(
                f"{r['method']}\n"
                f"PSNR={r['psnr']:.1f}, SSIM={r['ssim']:.3f}\n"
                f"fail={r['failure_rate']:.3f}",
                fontsize=9,
            )
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / "reconstructed_grid_err_0p001.png", dpi=180)
        plt.show()


# =============================================================================
# CLI
# =============================================================================

def parse_errors(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean two-method DNA image codec benchmark.")
    parser.add_argument("--image", required=False, help="Input image path")
    parser.add_argument("--out", default="two_best_results", help="Output directory")
    parser.add_argument("--methods", default="all", help="'all' or comma-separated method names")
    parser.add_argument("--errors", default="0,0.0001,0.001,0.005,0.01", help="Comma-separated substitution rates")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()

    if args.list_methods:
        print(list_methods().to_string(index=False))
        return

    if not args.image:
        raise SystemExit("--image is required unless --list-methods is used.")

    df = run_benchmark(
        image_path=args.image,
        out_dir=args.out,
        methods=args.methods,
        error_rates=parse_errors(args.errors),
        seed=args.seed,
        show=args.show,
    )

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
