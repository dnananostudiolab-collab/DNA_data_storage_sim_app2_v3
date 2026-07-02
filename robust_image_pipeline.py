from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

import two_best_dna_image_codecs as imgcodec
import dna_four_methods as fourcodec


METHOD_DISPLAY_TO_INTERNAL = {
    "No compression": "raw_pixels",
    "Robust Low-Resolution": "robust_low_resolution_image",
    "Robust Low-Resolution Image": "robust_low_resolution_image",
    "Base + Local Detail": "base_image_local_detail",
    "Base Image + Local Detail": "base_image_local_detail",
    "Base + WebP Detail": "base_webp_detail_tunable",
    "Base Image + WebP Detail": "base_webp_detail_tunable",
    "Local Block Coding": "local_block_coding_tunable",
}

METHOD_INTERNAL_TO_DISPLAY = {
    "raw_pixels": "No compression",
    "robust_low_resolution_image": "Robust Low-Resolution",
    "base_image_local_detail": "Base + Local Detail",
    "base_webp_detail_tunable": "Base + WebP Detail",
    "local_block_coding_tunable": "Local Block Coding",
    "hybrid_webp_highbase": "Base + WebP Detail",
    "local_dct_ycbcr_strong": "Local Block Coding",
}

PIXEL_REPRESENTATIONS = [
    "Black-white (1 bit/pixel)",
    "Grayscale (8 bits/pixel)",
    "RGB (24 bits/pixel)",
]

TUNABLE_METHODS = {"Base + WebP Detail", "Local Block Coding"}



LOWRES_PRESETS: Dict[str, Dict[str, Any]] = {
    "High quality": {
        "downsample": 3,
        "bits_per_channel": 6,
    },
    "Balanced": {
        "downsample": 4,
        "bits_per_channel": 5,
    },
    "High compression": {
        "downsample": 6,
        "bits_per_channel": 4,
    },
}

SMART_DETAIL_PRESETS: Dict[str, Dict[str, Any]] = {
    "High quality": {
        "base_downsample": 4,
        "base_bits": 6,
        "keep_coeffs": 6,
        "coeff_bits": 8,
        "q_step": 6.0,
    },
    "Balanced": {
        "base_downsample": 4,
        "base_bits": 5,
        "keep_coeffs": 4,
        "coeff_bits": 8,
        "q_step": 8.0,
    },
    "High compression": {
        "base_downsample": 6,
        "base_bits": 5,
        "keep_coeffs": 3,
        "coeff_bits": 8,
        "q_step": 10.0,
    },
}

HYBRID_PRESETS: Dict[str, Dict[str, Any]] = {
    "High quality": {
        "tile_size": 128,
        "quality": 60,
        "base_downsample": 4,
        "base_bits": 5,
    },
    "Balanced": {
        "tile_size": 128,
        "quality": 45,
        "base_downsample": 4,
        "base_bits": 5,
    },
    "High compression": {
        "tile_size": 192,
        "quality": 30,
        "base_downsample": 6,
        "base_bits": 4,
    },
}

LOCAL_DCT_PRESETS: Dict[str, Dict[str, Any]] = {
    "High quality": {
        "y_q_scale": 1.2,
        "c_q_scale": 2.0,
        "y_packet_bits": 80,
        "c_packet_bits": 32,
    },
    "Balanced": {
        "y_q_scale": 1.5,
        "c_q_scale": 2.5,
        "y_packet_bits": 64,
        "c_packet_bits": 24,
    },
    "High compression": {
        "y_q_scale": 2.0,
        "c_q_scale": 3.5,
        "y_packet_bits": 48,
        "c_packet_bits": 16,
    },
}


def list_image_compression_methods() -> list[str]:
    return [
        "No compression",
        "Robust Low-Resolution",
        "Base + Local Detail",
        "Base + WebP Detail",
        "Local Block Coding",
    ]


def list_compression_levels() -> list[str]:
    return ["High quality", "Balanced", "High compression", "Custom"]


def list_pixel_representations() -> list[str]:
    return list(PIXEL_REPRESENTATIONS)


def method_caption(method: str) -> str:
    return {
        "No compression": "Selected raw pixel data is sent directly to DNA Design. Black-white uses packed bits: 1 pixel = 1 bit.",
        "Robust Low-Resolution": "Stores only a small quantized image. Very robust, but blurry.",
        "Base + Local Detail": "Main proposed method: robust base image plus lightweight local detail.",
        "Base + WebP Detail": "Robust base image plus WebP detail tiles. Clean images look good, but tiles are more fragile under DNA errors.",
        "Local Block Coding": "Divides the image into local blocks so damage remains local.",
    }.get(method, "")


def _internal_method(method: str) -> str:
    if method in METHOD_DISPLAY_TO_INTERNAL:
        return METHOD_DISPLAY_TO_INTERNAL[method]
    if method in METHOD_INTERNAL_TO_DISPLAY:
        return method
    raise ValueError(f"Unknown image compression method: {method}")


def _pixel_mode_key(pixel_representation: str) -> str:
    s = str(pixel_representation or "").lower()
    if "black" in s or "binary" in s:
        return "black_white"
    if "gray" in s or "grey" in s:
        return "grayscale"
    return "rgb"


def prepare_pixel_image(
    image_path: str,
    pixel_representation: str = "Grayscale (8 bits/pixel)",
    threshold: int = 128,
) -> Tuple[Image.Image, Dict[str, Any]]:
    im0 = Image.open(image_path)
    key = _pixel_mode_key(pixel_representation)

    if key == "black_white":
        gray = im0.convert("L")
        im = gray.point(lambda p: 255 if p >= int(threshold) else 0).convert("L")
        channels = 1
        raw_mode = "1-bit packed"
        bits_per_pixel = 1
        representation = "Black-white (1 bit/pixel)"
    elif key == "grayscale":
        im = im0.convert("L")
        channels = 1
        raw_mode = "L"
        bits_per_pixel = 8
        representation = "Grayscale (8 bits/pixel)"
    else:
        im = im0.convert("RGB")
        channels = 3
        raw_mode = "RGB"
        bits_per_pixel = 24
        representation = "RGB (24 bits/pixel)"

    w, h = im.size
    raw_bits = int(w * h * bits_per_pixel)
    raw_bytes = int(math.ceil(raw_bits / 8))

    meta = {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "raw_mode": raw_mode,
        "pixel_representation": representation,
        "pixel_mode_key": key,
        "bits_per_pixel": int(bits_per_pixel),
        "raw_pixel_bits": int(raw_bits),
        "raw_pixel_bytes": int(raw_bytes),
        "threshold": int(threshold),
        "uploaded_file_bytes": int(Path(image_path).stat().st_size),
        "file_extension": Path(image_path).suffix.lower(),
        "original_mode": str(im0.mode),
    }
    return im, meta


def raw_pixel_info(
    image_path: str,
    pixel_representation: str = "Grayscale (8 bits/pixel)",
    threshold: int = 128,
) -> Dict[str, Any]:
    _im, meta = prepare_pixel_image(image_path, pixel_representation, threshold)
    return meta


def image_to_raw_pixels(
    image_path: str,
    out_dir: str | Path,
    pixel_representation: str = "Grayscale (8 bits/pixel)",
    threshold: int = 128,
) -> Tuple[bytes, Dict[str, Any], str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    im, pix = prepare_pixel_image(image_path, pixel_representation, threshold)
    key = pix["pixel_mode_key"]

    if key == "black_white":
        arr = np.asarray(im, dtype=np.uint8)
        bit_arr = (arr > 0).astype(np.uint8).ravel()
        raw = np.packbits(bit_arr).tobytes()
        payload_bits = int(bit_arr.size)
        expected_bytes = int(len(raw))
    else:
        raw = im.tobytes()
        payload_bits = int(len(raw) * 8)
        expected_bytes = int(len(raw))

    preview_path = out_dir / "selected_pixel_image.png"
    im.save(preview_path)

    meta = {
        "app_layer": "raw_pixels",
        "method": "raw_pixels",
        "method_display": "No compression",
        "kind": "raw_image_pixels",
        "output_ext": ".png",
        "payload_bits": int(payload_bits),
        "payload_bytes": float(payload_bits / 8),
        "payload_container_bytes": int(len(raw)),
        "expected_bytes": int(expected_bytes),
        "dna_nt": int(math.ceil(payload_bits / 2)),
        "compression_level": "No compression",
        "compression_fixed": True,
        "params": {},
        **pix,
    }
    return raw, meta, str(preview_path)


def _bits_from_payload(payload_bytes: bytes, n_bits: int) -> np.ndarray:
    bits = imgcodec.bytes_to_bits(bytes(payload_bytes or b""))
    n_bits = int(n_bits)
    if bits.size < n_bits:
        bits = np.pad(bits, (0, n_bits - bits.size))
    return bits[:n_bits].astype(np.uint8)


def _payload_bytes_from_dna(dna: str, meta: Dict[str, Any]) -> bytes:
    payload_bits = int(meta.get("payload_bits") or meta.get("impl_meta", {}).get("payload_bits") or 0)
    pad = int(meta.get("dna_pad", meta.get("impl_meta", {}).get("dna_pad", 0)))
    bits = imgcodec.dna_to_bits(dna, pad=pad)
    if payload_bits:
        bits = bits[:payload_bits]
    return imgcodec.bits_to_bytes(bits)


def image_payload_to_dna(payload_bytes: bytes, meta: Dict[str, Any]) -> str:
    n_bits = int(meta["payload_bits"])
    bits = _bits_from_payload(payload_bytes, n_bits)
    dna, _pad = imgcodec.bits_to_dna(bits)
    return dna



def _params_for_method(method: str, level: str, custom_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    internal = _internal_method(method)
    custom_params = custom_params or {}

    if internal == "robust_low_resolution_image":
        params = dict(LOWRES_PRESETS.get(level, LOWRES_PRESETS["Balanced"]))
        if level == "Custom":
            params.update(custom_params)
        return params

    if internal == "base_image_local_detail":
        params = dict(SMART_DETAIL_PRESETS.get(level, SMART_DETAIL_PRESETS["Balanced"]))
        if level == "Custom":
            params.update(custom_params)
        return params

    if internal == "base_webp_detail_tunable":
        params = dict(HYBRID_PRESETS.get(level, HYBRID_PRESETS["Balanced"]))
        if level == "Custom":
            params.update(custom_params)
        return params

    if internal == "local_block_coding_tunable":
        params = dict(LOCAL_DCT_PRESETS.get(level, LOCAL_DCT_PRESETS["Balanced"]))
        if level == "Custom":
            params.update(custom_params)
        return params

    return {}

def _pixel_meta_prefixed(pix_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pixel_representation": pix_meta.get("pixel_representation"),
        "pixel_mode_key": pix_meta.get("pixel_mode_key"),
        "pixel_width": int(pix_meta.get("width", 0)),
        "pixel_height": int(pix_meta.get("height", 0)),
        "pixel_channels": int(pix_meta.get("channels", 0)),
        "bits_per_pixel": int(pix_meta.get("bits_per_pixel", 0)),
        "raw_pixel_bits": int(pix_meta.get("raw_pixel_bits", 0)),
        "raw_pixel_bytes": int(pix_meta.get("raw_pixel_bytes", 0)),
        "threshold": int(pix_meta.get("threshold", 128)),
        "uploaded_file_bytes": int(pix_meta.get("uploaded_file_bytes", 0)),
        "file_extension": pix_meta.get("file_extension", ""),
        "selected_original_mode": pix_meta.get("original_mode", ""),
    }


def encode_image_to_payload(
    image_path: str,
    method: str,
    out_dir: str | Path,
    compression_level: str = "Balanced",
    custom_params: Dict[str, Any] | None = None,
    pixel_representation: str = "Grayscale (8 bits/pixel)",
    threshold: int = 128,
) -> Tuple[bytes, Dict[str, Any], str]:
    """
    Image -> payload bytes.

    Stops before DNA Design. Panel 3 maps these payload bytes using
    SM / FSM / Protected Design / Reed-Solomon.
    """
    internal = _internal_method(method)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_img, pix_meta = prepare_pixel_image(image_path, pixel_representation, threshold)
    selected_path = out_dir / "selected_pixel_image.png"
    selected_img.save(selected_path)

    if internal == "raw_pixels":
        return image_to_raw_pixels(
            image_path,
            out_dir,
            pixel_representation=pixel_representation,
            threshold=threshold,
        )

    params = _params_for_method(method, compression_level, custom_params)

    # Method family 1: methods from dna_four_methods.py, now with app-level presets.
    if internal == "robust_low_resolution_image":
        impl = fourcodec._load_impl("three_new")
        dna, impl_meta = impl.encode_base_only_resize_quant(
            str(selected_path),
            downsample=int(params["downsample"]),
            bits_per_channel=int(params["bits_per_channel"]),
        )
        four_meta = {
            "wrapper_version": getattr(fourcodec, "__version__", ""),
            "method": internal,
            "display_name": "Robust Low-Resolution",
            "simple_explanation": "Stores only a small quantized image. Very robust, but blurry.",
            "role": "ultra_robust_baseline",
            "implementation": "three_new.encode_base_only_resize_quant",
            "impl_meta": impl_meta,
            "payload_bytes": float(impl_meta.get("payload_bytes", 0)),
            "dna_nt": int(impl_meta.get("dna_nt", 0)),
            "dna_pad": int(impl_meta.get("dna_pad", 0)),
            "raw_dna_nt": float(impl_meta.get("raw_dna_nt", 0)),
            "dna_reduction_vs_raw_pixels": float(impl_meta.get("dna_reduction_vs_raw_pixels", 0)),
            "payload_vs_original_file": float(impl_meta.get("payload_vs_original_file", 0)),
            "original_file_bytes": int(impl_meta.get("original_file_bytes", 0)),
        }
        payload_bits = int(impl_meta.get("payload_bits", 0))
        payload_bytes = _payload_bytes_from_dna(dna, {"payload_bits": payload_bits, "dna_pad": four_meta.get("dna_pad", impl_meta.get("dna_pad", 0))})
        display = "Robust Low-Resolution"
        meta = dict(four_meta)
        meta.update({
            "app_layer": "robust_image_payload",
            "app_codec": "dna_four_methods",
            "method": internal,
            "method_display": display,
            "display_name": display,
            "compression_level": compression_level,
            "compression_fixed": False,
            "params": params,
            "payload_bits": int(payload_bits),
            "payload_bytes_exact": float(payload_bits / 8),
            "payload_container_bytes": int(len(payload_bytes)),
            "payload_byte_pad_bits": int(len(payload_bytes) * 8 - payload_bits),
            "selected_pixel_preview_path": str(selected_path),
            **_pixel_meta_prefixed(pix_meta),
        })

    elif internal == "base_image_local_detail":
        impl = fourcodec._load_impl("smart")
        custom_name = "_app_smart_base_residual_custom"
        impl.PRESETS[custom_name] = {
            "method_type": "smart_base_residual",
            "base_downsample": int(params["base_downsample"]),
            "base_bits": int(params["base_bits"]),
            "keep_coeffs": int(params["keep_coeffs"]),
            "coeff_bits": int(params["coeff_bits"]),
            "q_step": float(params["q_step"]),
        }
        dna, impl_meta = impl.encode_image(str(selected_path), method=custom_name)
        four_meta = {
            "wrapper_version": getattr(fourcodec, "__version__", ""),
            "method": internal,
            "display_name": "Base + Local Detail",
            "simple_explanation": "Stores a robust base image and adds local detail. If detail is damaged, that region falls back to the base.",
            "role": "main_proposed_method",
            "implementation": "smart.encode_image/app_custom",
            "impl_meta": impl_meta,
            "payload_bytes": float(impl_meta.get("payload_bytes", 0)),
            "dna_nt": int(impl_meta.get("dna_nt", 0)),
            "dna_pad": int(impl_meta.get("dna_pad", 0)),
            "raw_dna_nt": float(impl_meta.get("raw_dna_nt", 0)),
            "dna_reduction_vs_raw_pixels": float(impl_meta.get("dna_reduction_vs_raw_pixels", 0)),
            "payload_vs_original_file": float(impl_meta.get("payload_vs_original_file", 0)),
            "original_file_bytes": int(impl_meta.get("original_file_bytes", 0)),
        }
        payload_bits = int(impl_meta.get("payload_bits", 0))
        payload_bytes = _payload_bytes_from_dna(dna, {"payload_bits": payload_bits, "dna_pad": four_meta.get("dna_pad", impl_meta.get("dna_pad", 0))})
        display = "Base + Local Detail"
        meta = dict(four_meta)
        meta.update({
            "app_layer": "robust_image_payload",
            "app_codec": "dna_four_methods",
            "method": internal,
            "method_display": display,
            "display_name": display,
            "compression_level": compression_level,
            "compression_fixed": False,
            "params": params,
            "payload_bits": int(payload_bits),
            "payload_bytes_exact": float(payload_bits / 8),
            "payload_container_bytes": int(len(payload_bytes)),
            "payload_byte_pad_bits": int(len(payload_bytes) * 8 - payload_bits),
            "selected_pixel_preview_path": str(selected_path),
            **_pixel_meta_prefixed(pix_meta),
        })

    # Method family 2: tunable existing implementations, renamed for presentation.
    elif internal == "base_webp_detail_tunable":
        dna, impl_meta = imgcodec.encode_hybrid_webp_highbase(
            str(selected_path),
            tile_size=int(params["tile_size"]),
            quality=int(params["quality"]),
            base_downsample=int(params["base_downsample"]),
            base_bits=int(params["base_bits"]),
        )
        payload_bits_arr = imgcodec.dna_to_bits(dna, pad=int(impl_meta.get("dna_pad", 0)))[:int(impl_meta["payload_bits"])]
        payload_bytes = imgcodec.bits_to_bytes(payload_bits_arr)

        meta = dict(impl_meta)
        meta.update({
            "app_layer": "robust_image_payload",
            "app_codec": "two_best",
            "method": "hybrid_webp_highbase",
            "method_display": "Base + WebP Detail",
            "display_name": "Base + WebP Detail",
            "compression_level": compression_level,
            "compression_fixed": False,
            "params": params,
            "payload_bits": int(impl_meta["payload_bits"]),
            "payload_bytes_exact": float(impl_meta.get("payload_bytes", len(payload_bits_arr) / 8)),
            "payload_container_bytes": int(len(payload_bytes)),
            "payload_byte_pad_bits": int(len(payload_bytes) * 8 - int(impl_meta["payload_bits"])),
            "selected_pixel_preview_path": str(selected_path),
            **_pixel_meta_prefixed(pix_meta),
        })

    elif internal == "local_block_coding_tunable":
        dna, impl_meta = imgcodec.encode_local_dct_ycbcr_strong(
            str(selected_path),
            y_q_scale=float(params["y_q_scale"]),
            c_q_scale=float(params["c_q_scale"]),
            y_packet_bits=int(params["y_packet_bits"]),
            c_packet_bits=int(params["c_packet_bits"]),
        )
        payload_bits_arr = imgcodec.dna_to_bits(dna, pad=int(impl_meta.get("dna_pad", 0)))[:int(impl_meta["payload_bits"])]
        payload_bytes = imgcodec.bits_to_bytes(payload_bits_arr)

        meta = dict(impl_meta)
        meta.update({
            "app_layer": "robust_image_payload",
            "app_codec": "two_best",
            "method": "local_dct_ycbcr_strong",
            "method_display": "Local Block Coding",
            "display_name": "Local Block Coding",
            "compression_level": compression_level,
            "compression_fixed": False,
            "params": params,
            "payload_bits": int(impl_meta["payload_bits"]),
            "payload_bytes_exact": float(impl_meta.get("payload_bytes", len(payload_bits_arr) / 8)),
            "payload_container_bytes": int(len(payload_bytes)),
            "payload_byte_pad_bits": int(len(payload_bytes) * 8 - int(impl_meta["payload_bits"])),
            "selected_pixel_preview_path": str(selected_path),
            **_pixel_meta_prefixed(pix_meta),
        })

    else:
        raise ValueError(f"Unknown method: {method}")

    preview_path = out_dir / f"{internal}_before_dna_error.png"
    decode_payload_to_image(payload_bytes, meta, str(preview_path))

    meta_path = out_dir / f"{internal}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return payload_bytes, meta, str(preview_path)


def _coerce_decoded_output_channel(output_path: str, meta: Dict[str, Any]) -> str:
    """Force decoded preview/output to match the selected pixel representation channel.

    The image codecs may internally decode as RGB. For UI and validation consistency,
    grayscale input should return an L-mode image and RGB input should return RGB.
    """
    key = str(meta.get("pixel_mode_key", _pixel_mode_key(meta.get("pixel_representation", "")))).lower()
    if key == "rgb" or str(meta.get("raw_mode", "")).upper() == "RGB":
        target_mode = "RGB"
    elif key in {"grayscale", "black_white"} or str(meta.get("raw_mode", "")).upper() in {"L", "1-BIT PACKED"}:
        target_mode = "L"
    else:
        return "unchanged"

    try:
        im = Image.open(output_path)
        if target_mode == "L" and im.mode != "L":
            im = im.convert("L")
        elif target_mode == "RGB" and im.mode != "RGB":
            im = im.convert("RGB")
        if key == "black_white":
            im = im.point(lambda p: 255 if p >= 128 else 0).convert("L")
            target_mode = "L (black-white)"
        im.save(output_path)
        return target_mode
    except Exception:
        return "unchanged"


def decode_payload_to_image(
    payload_bytes: bytes,
    meta: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    output_path = str(output_path)

    if meta.get("method") == "raw_pixels" or meta.get("kind") == "raw_image_pixels":
        width = int(meta.get("width", meta.get("pixel_width", 0)))
        height = int(meta.get("height", meta.get("pixel_height", 0)))
        key = str(meta.get("pixel_mode_key", "grayscale"))
        expected_bits = int(meta.get("payload_bits", width * height))

        if key == "black_white":
            bits = _bits_from_payload(payload_bytes, expected_bits)
            arr = (bits[:width * height].reshape(height, width) * 255).astype(np.uint8)
            im = Image.fromarray(arr, mode="L")
            expected_bytes = int(math.ceil(width * height / 8))
        else:
            mode = "RGB" if key == "rgb" or str(meta.get("raw_mode")) == "RGB" else "L"
            channels = 3 if mode == "RGB" else 1
            expected_bytes = int(width * height * channels)
            raw = bytes(payload_bytes or b"")
            if len(raw) < expected_bytes:
                raw = raw + bytes(expected_bytes - len(raw))
            elif len(raw) > expected_bytes:
                raw = raw[:expected_bytes]
            im = Image.frombytes(mode, (width, height), raw)

        im.save(output_path)
        return {
            "decode_success": True,
            "valid_units": 1,
            "invalid_units": 0,
            "failure_rate": 0.0,
            "expected_bytes": int(expected_bytes),
            "decoded_bytes": int(len(payload_bytes or b"")),
            "pixel_representation": meta.get("pixel_representation", "—"),
            "output_channel": "RGB" if (key == "rgb" or str(meta.get("raw_mode")) == "RGB") else "L",
        }

    dna = image_payload_to_dna(payload_bytes, meta)
    app_codec = meta.get("app_codec", "two_best")

    if app_codec == "dna_four_methods":
        stats = fourcodec.decode_by_method(
            dna=dna,
            meta=dict(meta),
            output_path=output_path,
            substitution_rate=0.0,
            seed=123,
        )
    else:
        stats = imgcodec.decode_by_method(
            dna=dna,
            meta=dict(meta),
            output_path=output_path,
            substitution_rate=0.0,
            seed=123,
        )

    stats = dict(stats)
    stats["pixel_representation"] = meta.get("pixel_representation", "—")
    stats["output_channel"] = _coerce_decoded_output_channel(output_path, meta)
    return stats


def method_summary(meta: Dict[str, Any], raw_pixel_bytes: int | None = None) -> Dict[str, Any]:
    payload = int(meta.get("payload_container_bytes", meta.get("expected_bytes", 0)))
    raw = int(raw_pixel_bytes or meta.get("raw_pixel_bytes", 0))
    ratio = raw / max(1, payload)
    return {
        "Method": meta.get("method_display", meta.get("display_name", meta.get("method", "unknown"))),
        "Pixel representation": meta.get("pixel_representation", "—"),
        "Compression level": meta.get("compression_level", "—"),
        "Raw pixel data": raw,
        "Payload data": payload,
        "Ratio": round(ratio, 3),
    }
