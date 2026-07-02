from __future__ import annotations

import hashlib
import html
import io
import json
import re
import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from dna_codec import gc_content, homopolymer_stats
from dna_mapping import encode_bytes_to_dna, decode_dna_with_mapping
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv
from ui_helpers import fmt_bytes, step_header

from dna_sparse_semantic_project import (
    UTF8RawCodec,
    WholeZlibCodec,
    DenseTokenCodec,
    SparseSemanticTokenCodec,
    PretrainedSparseTokenCodec,
    FixedPretrainedCRCTokenCodec,
    XLMRContextRepairer,
    add_substitution_errors,
    dna_substitution_table,
    raw_utf8_dna_bases,
    simple_tokenize,
    text_similarity,
    classify_token_language,
    int_to_bits,
    dna_to_bits as project_dna_to_bits,
    bits_to_dna as project_bits_to_dna,
    detect_block_language,
    Decoded,
    bits_to_int,
    simple_detokenize,
)


FBR_DEFAULT = "ACACGACGCTCTTCCGATCT"
RBR_DEFAULT = "AGATCGGAAGAGCACACGTCT"
SUPPORTED_TEXT_UPLOADS = ["txt", "md", "csv", "json", "log"]

METHOD_ORDER = ["A_RAW_UTF8", "B_ZLIB", "C_DENSE_TOKEN", "D_SPARSE_SEMANTIC"]
METHOD_LABELS = {
    "A_RAW_UTF8": "A. No compression — raw UTF-8",
    "B_ZLIB": "B. Compression — whole zlib",
    "C_DENSE_TOKEN": "C. Compression — dense fixed-vocab token",
    "D_SPARSE_SEMANTIC": "Tokenization",
}
METHOD_SHORT = {
    "A_RAW_UTF8": "Raw UTF-8",
    "B_ZLIB": "zlib",
    "C_DENSE_TOKEN": "Dense token",
    "D_SPARSE_SEMANTIC": "Tokenization",
}
METHOD_DESCRIPTIONS = {
    "A_RAW_UTF8": "Exact baseline. Text is stored as UTF-8 bytes before DNA mapping. No compression.",
    "B_ZLIB": "Binary compression control. Small payload, but fragile under DNA substitution errors.",
    "C_DENSE_TOKEN": "Token baseline. Every fixed-width code is valid, so errors often become wrong tokens.",
    "D_SPARSE_SEMANTIC": "BERT/RoBERTa tokenizer method with CRC error detection and candidate-aware repair.",
}

TEXT_DNA_MAPPING_OPTIONS = ["Simple Mapping", "RINF_B16"]
TEXT_DNA_MAPPING_LABELS = {
    "Simple Mapping": "Simple Mapping (SM)",
    "RINF_B16": "RINF_B16 (R∞)",
}
DECODE_LANGUAGE_OPTIONS = ["auto", "en", "vi", "fr", "zh", "ko"]
DECODE_LANGUAGE_LABELS = {
    "auto": "Auto detect",
    "en": "English",
    "vi": "Vietnamese",
    "fr": "French",
    "zh": "Chinese",
    "ko": "Korean",
}
LANGUAGE_SCOPE_OPTIONS = ["global", "sentence_blocks"]
LANGUAGE_SCOPE_LABELS = {
    "global": "One language for whole text",
    "sentence_blocks": "Auto per sentence/block",
}
def _preview_seq(seq: str, n: int = 700) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _dna_difference_count(a: str, b: str) -> int:
    a = clean_dna(a)
    b = clean_dna(b)
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def _safe_widget_key(prefix: str, label: str, file_name: str, extra: str = "") -> str:
    raw = f"{prefix}|{label}|{file_name}|{extra}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(file_name or "download"))[:40]
    return f"{prefix}_{clean_name}_{digest}"


def _download_text(label: str, text: str, file_name: str, key: str | None = None) -> None:
    st.download_button(
        label,
        str(text or "").encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
        key=key or _safe_widget_key("download_text", label, file_name),
    )


def _download_bytes(label: str, data: bytes, file_name: str, key: str | None = None) -> None:
    st.download_button(
        label,
        bytes(data or b""),
        file_name=file_name,
        mime="application/octet-stream",
        use_container_width=True,
        key=key or _safe_widget_key("download_bytes", label, file_name),
    )


def _download_json(label: str, obj: Any, file_name: str, key: str | None = None) -> None:
    st.download_button(
        label,
        json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=file_name,
        mime="application/json",
        use_container_width=True,
        key=key or _safe_widget_key("download_json", label, file_name),
    )


def _strand_rows_from_uploaded_csv(uploaded_file) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str).fillna("")
    return [{str(k): str(v) for k, v in row.items()} for row in df.to_dict("records")]


def _df_csv_bytes(df: pd.DataFrame) -> bytes:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list, tuple)) else x)
    return safe.to_csv(index=False).encode("utf-8-sig")


def _download_df(label: str, df: pd.DataFrame, file_name: str, key: str | None = None) -> None:
    st.download_button(
        label,
        _df_csv_bytes(df),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        key=key or _safe_widget_key("download_df", label, file_name),
    )


def _decode_text_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_docx_text(raw: bytes) -> str:
    paragraphs: List[str] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("This DOCX file does not contain word/document.xml.")
        xml = zf.read("word/document.xml")

    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for p in root.findall(".//w:p", ns):
        parts: List[str] = []
        for node in p.iter():
            tag = node.tag.split("}")[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag == "br":
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs).strip()


def _extract_pdf_text(raw: bytes) -> str:
    errors: List[str] = []
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            text = "\n\n".join((page.extract_text() or "") for page in pdf.pages).strip()
        if text:
            return text
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    raise ValueError("Could not extract text from this PDF. Install pypdf or pdfplumber, or use a selectable-text PDF.")


def _normalise_json_text(text: str) -> str:
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return text


def _extract_text_from_upload(name: str, raw: bytes) -> Tuple[str, Dict[str, Any]]:
    suffix = Path(name or "uploaded.txt").suffix.lower().lstrip(".")
    meta: Dict[str, Any] = {
        "file_name": name or "uploaded file",
        "file_size": len(raw or b""),
        "file_type": suffix or "unknown",
        "extraction": "plain text",
    }

    if suffix == "docx":
        text = _extract_docx_text(raw)
        meta["extraction"] = "DOCX text"
    elif suffix == "pdf":
        text = _extract_pdf_text(raw)
        meta["extraction"] = "PDF text"
    elif suffix == "json":
        text = _normalise_json_text(_decode_text_bytes(raw))
        meta["extraction"] = "JSON text"
    else:
        text = _decode_text_bytes(raw)
        meta["extraction"] = f"{suffix.upper() or 'plain'} text"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    meta["characters"] = len(text)
    meta["words"] = len(text.split())
    meta["utf8_size"] = len(text.encode("utf-8"))
    return text, meta


def _uploaded_signature(name: str, raw: bytes) -> str:
    h = hashlib.sha256(raw or b"").hexdigest()[:16]
    return f"{name}|{len(raw or b'')}|{h}"


def _clear_text_downstream() -> None:
    for key in [
        "text_selected_package", "text_dna", "text_mapping_meta", "text_mapping_bits",
        "text_strand_rows", "text_noisy_dna", "text_noisy_codec_dna", "text_decoded_text",
        "text_repaired_text", "text_final_text", "text_decode_error", "text_validation",
        "text_changed_positions", "text_word_validation_df", "text_decode_ran",
        "text_codec_dna_from_decode", "text_substitution_table", "text_candidate_table",
        "text_repair_table", "text_decode_source", "text_decode_language",
    ]:
        st.session_state.pop(key, None)


def _sync_uploaded_file_to_text_area(uploaded: Any) -> None:
    if uploaded is None:
        return

    raw = uploaded.getvalue()
    sig = _uploaded_signature(uploaded.name, raw)
    if st.session_state.get("text_last_upload_sig") == sig:
        return

    try:
        extracted_text, meta = _extract_text_from_upload(uploaded.name, raw)
    except Exception as exc:
        st.session_state["text_upload_error"] = str(exc)
        st.session_state["text_last_upload_sig"] = sig
        return

    st.session_state["text_input_area"] = extracted_text
    st.session_state["text_input_value"] = extracted_text
    st.session_state["text_upload_meta"] = meta
    st.session_state["text_upload_error"] = ""
    st.session_state["text_last_upload_sig"] = sig
    _clear_text_downstream()
    st.rerun()


def _current_input_text() -> str:
    return st.session_state.get("text_input_area", st.session_state.get("text_input_value", "")) or ""


def _bits_to_padded_bytes(bits: str) -> Tuple[bytes, int]:
    bits = str(bits or "")
    pad_bits = (-len(bits)) % 8
    padded = bits + ("0" * pad_bits)
    data = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8)) if padded else b""
    return data, pad_bits


def _bytes_to_bits_exact(data: bytes, bit_len: int) -> str:
    bits = "".join(f"{b:08b}" for b in bytes(data or b""))
    return bits[:max(int(bit_len), 0)]


def _bytes_to_bits_full(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in bytes(data or b""))


def _method_payload_from_codec_dna(codec_dna: str) -> Tuple[bytes, Dict[str, Any]]:
    codec_bits = project_dna_to_bits(codec_dna)
    payload_bytes, pad_bits = _bits_to_padded_bytes(codec_bits)
    return payload_bytes, {
        "codec_bit_len": len(codec_bits),
        "payload_byte_len": len(payload_bytes),
        "payload_pad_bits": pad_bits,
    }


def _get_codec_vocab(codec: Any) -> List[str]:
    # Pretrained tokenizer codecs use a fixed external vocabulary and must not
    # store a per-file dictionary in app metadata or DNA metadata.
    if getattr(codec, "is_pretrained_tokenizer_codec", False):
        return []
    if hasattr(codec, "id_to_token"):
        return [codec.id_to_token[i] for i in range(len(codec.id_to_token))]
    return []


def _make_codec(method_key: str, text: str, dense_vocab_size: int, sparse_valid_vocab_size: int, sparse_code_bits: int, fixed_tokenizer_name: str = "bert-base-uncased", fixed_error_detection: str = "token_crc8", fixed_block_tokens: int = 16) -> Any:
    if method_key == "A_RAW_UTF8":
        return UTF8RawCodec()
    if method_key == "B_ZLIB":
        return WholeZlibCodec()
    if method_key == "C_DENSE_TOKEN":
        return DenseTokenCodec.from_text(text, vocab_size=int(dense_vocab_size))
    if method_key == "D_SPARSE_SEMANTIC":
        return FixedPretrainedCRCTokenCodec.from_text(text, tokenizer_name=fixed_tokenizer_name, error_detection=fixed_error_detection, block_tokens=int(fixed_block_tokens))
    raise ValueError(f"Unknown text method: {method_key}")


def _codec_from_package(package: Dict[str, Any]) -> Any:
    if package.get("codec_obj") is not None:
        return package["codec_obj"]

    method_key = package.get("method_key")
    params = package.get("params", {})
    vocab = package.get("codec_vocab", [])
    if method_key == "A_RAW_UTF8":
        return UTF8RawCodec()
    if method_key == "B_ZLIB":
        return WholeZlibCodec()
    if method_key == "C_DENSE_TOKEN":
        return DenseTokenCodec(vocab, vocab_size=int(params.get("dense_vocab_size", 8192)))
    if method_key == "D_SPARSE_SEMANTIC":
        return FixedPretrainedCRCTokenCodec(
            tokenizer_name=str(params.get("tokenizer_name", "bert-base-uncased")),
            error_detection=str(params.get("error_detection", "token_crc8")),
            block_tokens=int(params.get("block_tokens", 16)),
        )
    raise ValueError(f"Unknown text method in package: {method_key}")


def _make_text_method_package(text: str, method_key: str, dense_vocab_size: int, sparse_valid_vocab_size: int, sparse_code_bits: int, fixed_tokenizer_name: str = "bert-base-uncased", fixed_error_detection: str = "token_crc8", fixed_block_tokens: int = 16) -> Dict[str, Any]:
    codec = _make_codec(method_key, text, dense_vocab_size, sparse_valid_vocab_size, sparse_code_bits, fixed_tokenizer_name=fixed_tokenizer_name, fixed_error_detection=fixed_error_detection, fixed_block_tokens=int(fixed_block_tokens))
    enc = codec.encode(text)
    raw_bases = raw_utf8_dna_bases(text)
    codec_bits = project_dna_to_bits(enc.dna)
    package: Dict[str, Any] = {
        "method_key": method_key,
        "method_label": METHOD_LABELS[method_key],
        "method_short": METHOD_SHORT[method_key],
        "method_description": METHOD_DESCRIPTIONS[method_key],
        "codec_name": enc.method,
        "codec_dna": clean_dna(enc.dna),
        "codec_bit_len": len(codec_bits),
        "meta": enc.meta,
        "codec_obj": codec,
        "codec_vocab": _get_codec_vocab(codec),
        "original_text": text,
        "params": {
            "dense_vocab_size": int(dense_vocab_size),
            "sparse_valid_vocab_size": int(sparse_valid_vocab_size),
            "sparse_code_bits": int(sparse_code_bits),
            "tokenizer_name": enc.meta.get("tokenizer_name", fixed_tokenizer_name),
            "pretrained_sparse_code_bits": enc.meta.get("code_bits", enc.meta.get("token_id_bits")),
            "token_id_bits": enc.meta.get("token_id_bits"),
            "error_detection": enc.meta.get("error_detection", fixed_error_detection),
            "block_tokens": enc.meta.get("block_tokens", fixed_block_tokens),
            "no_file_specific_vocab": bool(enc.meta.get("no_file_specific_vocab", False)),
        },
        "raw_utf8_bytes": len(text.encode("utf-8")),
        "raw_utf8_dna_bases": raw_bases,
        "text_code_dna_bases": len(enc.dna),
        "text_code_reduction_percent": (1.0 - len(enc.dna) / max(raw_bases, 1)) * 100.0,
        "text_code_ratio_vs_utf8_raw": len(enc.dna) / max(raw_bases, 1),
        "tokens": enc.meta.get("tokens", simple_tokenize(text)),
        "num_tokens": enc.meta.get("num_tokens", len(simple_tokenize(text))),
    }

    payload_bytes, payload_meta = _method_payload_from_codec_dna(package["codec_dna"])
    package.update(payload_meta)
    package["payload_bytes"] = payload_bytes
    return package


def _jsonable_package(package: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in package.items():
        if k in {"codec_obj", "payload_bytes"}:
            continue
        if isinstance(v, bytes):
            out[k] = f"<bytes:{len(v)}>"
        else:
            out[k] = v
    return out


def _tokenization_table(package: Dict[str, Any]) -> pd.DataFrame:
    """Build a transparent tokenization/codeword table for the selected text method."""
    method_key = package.get("method_key", "")
    original_text = package.get("original_text", "")
    meta = package.get("meta", {}) or {}
    tokens = list(package.get("tokens") or meta.get("tokens") or simple_tokenize(original_text))
    token_ids = list(meta.get("token_ids") or package.get("token_ids") or [])

    # Raw UTF-8 and zlib do not use token IDs for storage, but showing the
    # tokenizer output is still useful for inspection/validation.
    uses_token_code = method_key in {"C_DENSE_TOKEN", "D_SPARSE_SEMANTIC"}

    codec = None
    if uses_token_code:
        try:
            codec = _codec_from_package(package)
        except Exception:
            codec = package.get("codec_obj")

    rows: List[Dict[str, Any]] = []
    for i, tok in enumerate(tokens):
        tid = token_ids[i] if i < len(token_ids) else None
        lang = classify_token_language(str(tok))
        row: Dict[str, Any] = {
            "position": i,
            "token": tok,
            "language": lang,
            "used_in_method_encoding": uses_token_code,
            "token_id": "" if tid is None else int(tid),
            "bit_width": "",
            "codeword_bits": "",
            "codeword_dna": "",
            "code_type": "UTF-8/zlib byte stream; tokenization preview only",
        }

        if method_key == "C_DENSE_TOKEN" and tid is not None:
            bit_width = int(meta.get("token_bits", getattr(codec, "token_bits", 0) or 0))
            code_bits = int_to_bits(int(tid), bit_width) if bit_width else ""
            row.update({
                "bit_width": bit_width,
                "codeword_bits": code_bits,
                "codeword_dna": project_bits_to_dna(code_bits) if code_bits else "",
                "code_type": "dense fixed-width token ID; every code is valid",
            })

        elif method_key == "D_SPARSE_SEMANTIC" and tid is not None:
            bit_width = int(meta.get("token_id_bits", meta.get("code_bits", getattr(codec, "token_id_bits", getattr(codec, "code_bits", 0)) or 0)))
            code_bits = ""
            if codec is not None and hasattr(codec, "code_bits_for_token_id"):
                code_bits = codec.code_bits_for_token_id(int(tid))
            elif codec is not None and hasattr(codec, "id_to_code_bits"):
                code_bits = getattr(codec, "id_to_code_bits", {}).get(int(tid), "")
            if not code_bits and codec is not None and hasattr(codec, "id_to_code"):
                code_int = getattr(codec, "id_to_code", {}).get(int(tid))
                if code_int is not None and bit_width:
                    code_bits = int_to_bits(int(code_int), bit_width)
            row.update({
                "bit_width": bit_width,
                "codeword_bits": code_bits,
                "codeword_dna": project_bits_to_dna(code_bits) if code_bits else "",
                "code_type": "BERT/RoBERTa token ID + CRC",
            })

        rows.append(row)

    return pd.DataFrame(rows)


def _token_recovery_preview_table(
    package: Dict[str, Any],
    decoded_meta: Dict[str, Any],
    repair_table: pd.DataFrame | None = None,
    max_rows: int = 250,
) -> pd.DataFrame:
    """Show which tokenizer units survived, failed CRC, or were repaired.

    This table is intentionally tokenizer-level rather than word-level.  It is
    the most direct view of the DNA-to-token recovery process: tokens with a
    failed CRC are shown separately from tokens that decoded cleanly.
    """
    if package.get("method_key") != "D_SPARSE_SEMANTIC":
        return pd.DataFrame()
    try:
        codec = _codec_from_package(package)
    except Exception:
        return pd.DataFrame()

    original_ids = [int(x) for x in list((package.get("meta", {}) or {}).get("token_ids", []))]
    decoded_ids = [int(x) for x in list(decoded_meta.get("decoded_ids", []))]
    invalid_positions = {int(x) for x in list(decoded_meta.get("invalid_positions", []))}
    candidate_map = decoded_meta.get("candidate_ids_by_position", {}) or {}

    repair_by_pos: Dict[int, Dict[str, Any]] = {}
    if isinstance(repair_table, pd.DataFrame) and not repair_table.empty:
        for _, row in repair_table.iterrows():
            try:
                repair_by_pos[int(row.get("token_position"))] = dict(row)
            except Exception:
                continue

    def piece(tid: int | None) -> str:
        if tid is None:
            return ""
        try:
            return str(codec.tokenizer.convert_ids_to_tokens([int(tid)])[0])
        except Exception:
            return ""

    def text_piece(tid: int | None) -> str:
        if tid is None:
            return ""
        try:
            return str(codec.tokenizer.decode([int(tid)], clean_up_tokenization_spaces=True)).strip()
        except Exception:
            return ""

    mask_id = int(getattr(codec, "mask_token_id", -1) or -1)
    eos_id = int(getattr(codec, "eos_token_id", -1) or -1)
    rows: List[Dict[str, Any]] = []
    n = min(max(len(original_ids), len(decoded_ids)), int(max_rows))
    for pos in range(n):
        oid = original_ids[pos] if pos < len(original_ids) else None
        did = decoded_ids[pos] if pos < len(decoded_ids) else None
        if oid == eos_id:
            break
        rep = repair_by_pos.get(pos, {})
        chosen_id = rep.get("chosen_token_id", None)
        try:
            chosen_id_int = int(chosen_id) if chosen_id is not None and str(chosen_id) != "nan" else None
        except Exception:
            chosen_id_int = None
        repaired_text = str(rep.get("chosen", "")) if rep else ""
        accepted = bool(rep.get("accepted", False)) if rep else False

        if pos in invalid_positions:
            status = "CRC fail"
            if accepted:
                status = "Repaired"
            elif did == mask_id:
                status = "Unresolved"
        elif oid is not None and did is not None and int(oid) == int(did):
            status = "Decoded"
        elif did is None:
            status = "Missing"
        else:
            status = "Changed"

        cand_ids = []
        try:
            cand_ids = [int(x) for x in list(candidate_map.get(pos, candidate_map.get(str(pos), [])))[:8]]
        except Exception:
            cand_ids = []
        rows.append({
            "position": pos,
            "status": status,
            "original_piece": piece(oid),
            "original_text": text_piece(oid),
            "decoded_piece": piece(did),
            "decoded_text": text_piece(did),
            "repaired_text": repaired_text if accepted else "",
            "original_id": "" if oid is None else int(oid),
            "decoded_id": "" if did is None else int(did),
            "candidate_count": int(len(candidate_map.get(pos, candidate_map.get(str(pos), [])) or [])),
            "candidate_preview": ", ".join(text_piece(x) for x in cand_ids if text_piece(x))[:180],
        })
    return pd.DataFrame(rows)


def _append_display_token(buffer: List[str], token_html: str, token_text: str) -> None:
    """Append token HTML with simple English spacing rules."""
    token_text = str(token_text or "").strip()
    if not token_text:
        return
    no_space_before = set(".,;:!?)]}”’%")
    no_space_after = set("([{“‘$")
    if not buffer:
        buffer.append(token_html)
        return
    if token_text in no_space_before:
        buffer.append(token_html)
    elif buffer[-1].endswith(tuple(no_space_after)):
        buffer.append(token_html)
    else:
        buffer.append(" " + token_html)


def _annotated_decoded_html(recovery_df: pd.DataFrame, fallback_text: str = "", max_tokens: int = 900) -> str:
    """Render decoded text with failed tokens as [MASK] and repaired tokens in bold.

    This is the professor-facing preview: it keeps the text readable while making
    failure and repair locations visually explicit.
    """
    style = """
<style>
.decoded-annotated-box {
  min-height: 280px;
  max-height: 420px;
  overflow-y: auto;
  padding: 0.85rem 0.95rem;
  border: 1px solid rgba(148, 163, 184, 0.45);
  border-radius: 0.75rem;
  background: #ffffff;
  line-height: 1.85;
  font-size: 0.95rem;
  color: #111827;
}
.decoded-mask {
  font-weight: 700;
  color: #991b1b;
  background: #fee2e2;
  border: 1px solid #fecaca;
  border-radius: 0.35rem;
  padding: 0.05rem 0.25rem;
  white-space: nowrap;
}
.decoded-repaired {
  font-weight: 800;
  color: #14532d;
  background: #dcfce7;
  border: 1px solid #86efac;
  border-radius: 0.35rem;
  padding: 0.05rem 0.25rem;
  white-space: nowrap;
}
.decoded-changed {
  color: #92400e;
  background: #fef3c7;
  border-radius: 0.25rem;
  padding: 0.03rem 0.18rem;
}
.decoded-legend {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #64748b;
}
</style>
"""
    if not isinstance(recovery_df, pd.DataFrame) or recovery_df.empty:
        safe = html.escape(str(fallback_text or ""))
        return style + f'<div class="decoded-annotated-box">{safe}</div>'

    parts: List[str] = []
    for _, row in recovery_df.head(int(max_tokens)).iterrows():
        status = str(row.get("status", ""))
        decoded_piece = str(row.get("decoded_text", "") or "").strip()
        repaired_piece = str(row.get("repaired_text", "") or "").strip()
        original_piece = str(row.get("original_text", "") or "").strip()

        if status == "Repaired" and repaired_piece:
            tok_text = repaired_piece
            tok_html = f'<span class="decoded-repaired"><b>{html.escape(tok_text)}</b></span>'
        elif status in {"CRC fail", "Unresolved", "Missing"}:
            tok_text = "[MASK]"
            tok_html = '<span class="decoded-mask">[MASK]</span>'
        elif status == "Changed" and decoded_piece:
            tok_text = decoded_piece
            tok_html = f'<span class="decoded-changed">{html.escape(tok_text)}</span>'
        else:
            tok_text = decoded_piece or original_piece
            tok_html = html.escape(tok_text)

        _append_display_token(parts, tok_html, tok_text)

    body = "".join(parts).strip() or html.escape(str(fallback_text or ""))
    legend = '<div class="decoded-legend"><span class="decoded-mask">[MASK]</span> = failed/unresolved token &nbsp; <span class="decoded-repaired"><b>bold</b></span> = repaired token &nbsp; <span class="decoded-changed">highlight</span> = changed token</div>'
    return style + f'<div class="decoded-annotated-box">{body}{legend}</div>'


def _tokenization_json(package: Dict[str, Any]) -> Dict[str, Any]:
    df = _tokenization_table(package)
    return {
        "method_key": package.get("method_key"),
        "method_label": package.get("method_label"),
        "codec_name": package.get("codec_name"),
        "num_tokens": int(package.get("num_tokens", len(df))),
        "notes": (
            "Raw UTF-8 and zlib store bytes, so tokenization is an inspection table only. "
            "Dense and sparse methods use token IDs/codewords for encoding."
        ),
        "tokens": df.to_dict(orient="records"),
    }


def _apply_dna_mapping(package: Dict[str, Any], mapping_name: str) -> Dict[str, Any]:
    payload = package.get("payload_bytes", b"")
    mapped_dna, mapped_bits, mapping_meta = encode_bytes_to_dna(payload, mapping_name)
    hp = homopolymer_stats(mapped_dna)
    updated = dict(package)
    updated.update({
        "dna_mapping": mapping_name,
        "mapped_dna": clean_dna(mapped_dna),
        "mapping_bits": mapped_bits,
        "mapping_meta": mapping_meta,
        "mapped_dna_bases": len(clean_dna(mapped_dna)),
        "mapped_gc_content": gc_content(mapped_dna),
        "mapped_longest_homopolymer": hp.get("longest", 0),
        "mapped_homopolymer_count_ge2": hp.get("count_ge2", 0),
        "final_reduction_percent": (1.0 - len(clean_dna(mapped_dna)) / max(int(package.get("raw_utf8_dna_bases", 1)), 1)) * 100.0,
        "final_ratio_vs_utf8_raw": len(clean_dna(mapped_dna)) / max(int(package.get("raw_utf8_dna_bases", 1)), 1),
    })
    return updated



def _split_sentence_blocks(text: str) -> List[str]:
    """Split text into sentence-like blocks while preserving multilingual punctuation."""
    text = str(text or "").strip()
    if not text:
        return []
    raw_blocks = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    blocks = [b.strip() for b in raw_blocks if b and b.strip()]
    return blocks or [text]


def _sentence_language_blocks(text: str, manual_language: str = "auto") -> Tuple[Dict[int, str], pd.DataFrame]:
    """Map token positions to a detected/manual language for each sentence-like block."""
    token_lang: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for block_id, block in enumerate(_split_sentence_blocks(text), start=1):
        toks = simple_tokenize(block)
        lang = detect_block_language(block, manual_language=manual_language)
        start = cursor
        end = cursor + len(toks) - 1 if toks else cursor - 1
        for pos in range(start, end + 1):
            token_lang[pos] = lang
        rows.append({
            "block_id": block_id,
            "language": lang,
            "token_start": start,
            "token_end": end,
            "token_count": len(toks),
            "text_preview": block[:220] + ("..." if len(block) > 220 else ""),
        })
        cursor += len(toks)
    return token_lang, pd.DataFrame(rows)


def _decode_sparse_with_token_languages(codec: Any, codec_dna: str, meta: Dict[str, Any], token_languages: Dict[int, str]) -> Decoded:
    """Decode sparse semantic DNA with a separate language constraint per token position."""
    bits = project_dna_to_bits(codec_dna)
    n = int(meta.get("num_tokens")) if meta and "num_tokens" in meta else None
    decoded_tokens: List[str] = []
    decoded_ids: List[int] = []
    candidate_rows: List[Dict[str, Any]] = []
    invalid_count = 0
    exact_count = 0
    pos = 0

    for i in range(0, len(bits), int(codec.code_bits)):
        chunk = bits[i:i + int(codec.code_bits)]
        if len(chunk) < int(codec.code_bits):
            break
        code = bits_to_int(chunk)
        if code in codec.code_to_id:
            tid = codec.code_to_id[code]
            tok = codec.id_to_token.get(tid, "[UNK]")
            if str(tok).startswith("[FILL_"):
                tok = "[UNK]"
            exact = True
            exact_count += 1
        else:
            tid = codec.unk_id
            tok = "[UNK]"
            exact = False
            invalid_count += 1

        prefer_language = token_languages.get(pos)
        cands = codec.nearest_candidates(chunk, top_k=8, prefer_language=prefer_language)
        for rank, cand in enumerate(cands, start=1):
            candidate_rows.append({
                "token_position": pos,
                "rank": rank,
                "language_constraint": prefer_language or "auto/global",
                "exact_valid_code": exact,
                **cand,
            })
        decoded_tokens.append(tok)
        decoded_ids.append(tid)
        pos += 1

    if n is not None:
        decoded_tokens = decoded_tokens[:n]
        decoded_ids = decoded_ids[:n]
        candidate_rows = [r for r in candidate_rows if r["token_position"] < n]

    return Decoded(codec.name, simple_detokenize([t for t in decoded_tokens if t != "[PAD]"]), True, {
        "decoded_tokens": decoded_tokens,
        "decoded_ids": decoded_ids,
        "invalid_code_count": invalid_count,
        "exact_valid_code_count": exact_count,
        "code_candidate_table": pd.DataFrame(candidate_rows),
        "language_block_mode": "sentence_blocks",
    })

def _decode_direct_method_level(package: Dict[str, Any], codec_dna: str, decode_language: str = "auto", language_scope: str = "global") -> Tuple[Any, str, Dict[str, Any]]:
    codec = _codec_from_package(package)
    decode_meta: Dict[str, Any] = {"decode_source": "method-level", "language_scope": language_scope}
    if package.get("method_key") == "D_SPARSE_SEMANTIC":
        if language_scope == "sentence_blocks" and not getattr(codec, "is_pretrained_tokenizer_codec", False):
            token_lang, block_df = _sentence_language_blocks(package.get("original_text", ""), manual_language=decode_language)
            dec = _decode_sparse_with_token_languages(codec, codec_dna, package.get("meta", {}), token_lang)
            decode_meta["language_blocks"] = block_df
        else:
            prefer_language = None if decode_language == "auto" else decode_language
            dec = codec.decode(codec_dna, package.get("meta", {}), prefer_language=prefer_language)
    else:
        dec = codec.decode(codec_dna, package.get("meta", {}))
    return dec, clean_dna(codec_dna), decode_meta


def _decode_full_pipeline(package: Dict[str, Any], mapped_dna: str, decode_language: str = "auto", language_scope: str = "global") -> Tuple[Any, str, Dict[str, Any]]:
    mapping_name = package.get("dna_mapping", "Simple Mapping")
    decoded_payload, _bits, mapping_decode_meta = decode_dna_with_mapping(
        mapped_dna,
        mapping_name,
        package.get("mapping_meta", {}),
    )
    payload_len = int(package.get("payload_byte_len", len(decoded_payload)))
    decoded_payload = bytes(decoded_payload or b"")[:payload_len]
    if package.get("method_key") == "D_SPARSE_SEMANTIC" and bool((package.get("meta", {}) or {}).get("no_file_specific_vocab", False)):
        # Pretrained sparse token uses a fixed EOS token, so no per-file bit-length
        # metadata is required. Decode all recovered payload bits and stop at EOS.
        codec_bits = _bytes_to_bits_full(decoded_payload)
    else:
        codec_bits = _bytes_to_bits_exact(decoded_payload, int(package.get("codec_bit_len", 0)))
    text_codec_dna = project_bits_to_dna(codec_bits)
    dec, codec_dna, method_decode_meta = _decode_direct_method_level(package, text_codec_dna, decode_language=decode_language, language_scope=language_scope)
    if isinstance(method_decode_meta, dict):
        mapping_decode_meta = dict(mapping_decode_meta or {})
        mapping_decode_meta.update(method_decode_meta)
    return dec, codec_dna, mapping_decode_meta


def _strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    keep = [
        "No.", "Text method", "DNA mapping", "Strand index", "Index length", "Payload length",
        "Filler length", "Total length", "GC content", "Longest homopolymer", "Full strand",
    ]
    return pd.DataFrame([{k: row.get(k, "—") for k in keep if k in row} for row in rows])


def _word_tokens_for_validation(value: str) -> List[str]:
    return [
        tok.lower()
        for tok in re.findall(r"[A-Za-zÀ-ỹ0-9]+(?:[-'][A-Za-zÀ-ỹ0-9]+)?|[\u4e00-\u9fff]+", str(value or ""), flags=re.UNICODE)
    ]


def _word_level_validation(original: str, decoded: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    orig = _word_tokens_for_validation(original)
    dec = _word_tokens_for_validation(decoded)
    n, m = len(orig), len(dec)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if orig[i - 1] == dec[j - 1]:
                best = (dp[i - 1][j - 1], "match")
            else:
                best = (dp[i - 1][j - 1] + 1, "substitute")
            delete = (dp[i - 1][j] + 1, "delete")
            insert = (dp[i][j - 1] + 1, "insert")
            dp[i][j], back[i][j] = min([best, delete, insert], key=lambda x: x[0])

    rows = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "substitute"}:
            rows.append({
                "Original position": i,
                "Decoded position": j,
                "Original word": orig[i - 1],
                "Decoded word": dec[j - 1],
                "Status": "Match" if op == "match" else "Substitution",
            })
            i -= 1
            j -= 1
        elif op == "delete":
            rows.append({
                "Original position": i,
                "Decoded position": "—",
                "Original word": orig[i - 1],
                "Decoded word": "",
                "Status": "Deletion",
            })
            i -= 1
        elif op == "insert":
            rows.append({
                "Original position": "—",
                "Decoded position": j,
                "Original word": "",
                "Decoded word": dec[j - 1],
                "Status": "Insertion",
            })
            j -= 1
        else:
            break

    rows.reverse()
    df = pd.DataFrame(rows)
    matches = int((df["Status"] == "Match").sum()) if len(df) else 0
    substitutions = int((df["Status"] == "Substitution").sum()) if len(df) else 0
    deletions = int((df["Status"] == "Deletion").sum()) if len(df) else 0
    insertions = int((df["Status"] == "Insertion").sum()) if len(df) else 0
    errors = substitutions + deletions + insertions
    metrics = {
        "original_words": n,
        "decoded_words": m,
        "exact_word_matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "word_error_count": errors,
        "word_error_rate": errors / max(n, 1),
        "word_accuracy": matches / max(n, 1),
        "sequence_similarity": text_similarity(original, decoded),
    }
    return metrics, df



# -----------------------------------------------------------------------------
# Shared strand-error helpers for the text branch.
# These mirror the image branch structure: FBR / SI / Payload / Filler / RBR.
# -----------------------------------------------------------------------------

def _text_row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", row.get("Index", "")))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _text_region_for_position(row: Dict[str, Any], pos1: int) -> str:
    cursor = 1
    for name, seq in _text_row_regions(row):
        end = cursor + len(seq) - 1
        if cursor <= int(pos1) <= end:
            return name
        cursor = end + 1
    return "Outside"


_TEXT_REGION_COLORS = {
    "FBR": ("#DBEAFE", "#1E3A8A"),
    "SI": ("#EDE9FE", "#5B21B6"),
    "Payload": ("#DCFCE7", "#166534"),
    "Filler": ("#F3F4F6", "#374151"),
    "RBR": ("#FFE4E6", "#9F1239"),
    "Error": ("#FEE2E2", "#991B1B"),
}


def _text_region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    bg, fg = _TEXT_REGION_COLORS.get(name, ("#F8FAFC", "#0F172A"))
    error_positions = error_positions or set()
    chars: List[str] = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        if i in error_positions:
            ebg, efg = _TEXT_REGION_COLORS["Error"]
            chars.append(
                f'<span style="background:{ebg};color:{efg};border-radius:4px;padding:0 2px;font-weight:800;">{ch}</span>'
            )
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span style="display:inline-block;margin:0.16rem 0.20rem 0.16rem 0; padding:0.28rem 0.42rem; '
        f'border-radius:10px;background:{bg};color:{fg};font-family:monospace;font-size:0.82rem;line-height:1.55;">'
        f'<b>{name}</b>: {body}</span>'
    )


def _render_text_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    parts: List[str] = []
    cursor = 1
    for name, seq in _text_row_regions(row):
        parts.append(_text_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(clean_dna(seq))
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _clean_decode_source_label(label: str) -> str:
    if label.startswith("Clean DNA") or label == "Current encoded DNA":
        return "Current encoded DNA"
    if label.startswith("Error DNA") or label == "Noisy encoded DNA":
        return "Noisy encoded DNA"
    return label


def _text_dna_from_strand_rows_for_decode(rows: List[Dict[str, Any]], original_dna_len: int = 0) -> str:
    parts: List[str] = []
    for row in rows or []:
        payload = clean_dna(row.get("Error payload", "")) if row.get("Advanced error source") else ""
        if not payload:
            payload = clean_dna(row.get("Payload", ""))
        parts.append(payload)
    dna = clean_dna("".join(parts))
    return dna[:int(original_dna_len)] if original_dna_len else dna


def _mutate_text_prepared_strand(
    row: Dict[str, Any],
    *,
    scope: str = "Payload only",
    substitution_rate: float = 0.0,
    insertion_rate: float = 0.0,
    deletion_rate: float = 0.0,
    seed: int = 17,
) -> Dict[str, Any]:
    rng = random.Random(str(seed))
    full = clean_dna(row.get("Full strand", "")) or clean_dna("".join(seq for _, seq in _text_row_regions(row)))
    mutable_regions = {
        "Payload only": {"Payload"},
        "Index + Payload": {"SI", "Payload"},
        "Full strand": {"FBR", "SI", "Payload", "Filler", "RBR"},
    }.get(scope, {"Payload"})

    allow_indels = bool(float(insertion_rate) > 0 or float(deletion_rate) > 0)
    out: List[str] = []
    events: List[Dict[str, Any]] = []
    sub_count = ins_count = del_count = 0
    read_pos = 0

    for pos, base in enumerate(full, start=1):
        region = _text_region_for_position(row, pos)
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
            continue

        new_base = base
        if mutable and rng.random() < float(substitution_rate):
            choices = [b for b in "ACGT" if b != base]
            new_base = rng.choice(choices)
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
            nb = rng.choice([b for b in "ACGT"])
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
    start = fbr_len + idx_len
    err_payload = clean_dna(err_full[start:start + payload_len])

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
        mutated: Dict[str, str] = {}
        for name, seq in _text_row_regions(row):
            n = len(seq)
            mutated[name] = err_full[cursor:cursor + n]
            cursor += n
        new["FBR"] = mutated.get("FBR", new.get("FBR", ""))
        new["Strand index"] = mutated.get("SI", new.get("Strand index", ""))
        new["Index"] = mutated.get("SI", new.get("Index", new.get("Strand index", "")))
        new["Payload"] = mutated.get("Payload", new.get("Payload", ""))
        new["Filler"] = mutated.get("Filler", new.get("Filler", ""))
        new["RBR"] = mutated.get("RBR", new.get("RBR", ""))

    return new


def _text_error_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Advanced error scope", "Total length", "Error full length", "Substitution count", "Insertion count", "Deletion count", "Error count"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _text_error_events_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
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


def _direct_substitute_dna(dna: str, substitution_rate: float, seed: int) -> Tuple[str, int]:
    """Substitution-only noise directly on encoded DNA; skips strand design."""
    seq = list(clean_dna(dna))
    rng = random.Random(int(seed))
    bases = "ACGT"
    n_sub = 0
    rate = max(0.0, float(substitution_rate))
    for i, base in enumerate(seq):
        if rng.random() < rate:
            options = [b for b in bases if b != base]
            seq[i] = rng.choice(options)
            n_sub += 1
    return "".join(seq), n_sub

def render_text_dna_storage_panel() -> None:

    with st.container(border=True):
        step_header(1, "Input")
        left, right = st.columns([1.35, 1.0], gap="large")

        with left:
            uploaded = st.file_uploader(
                "Upload text file (TXT, MD, CSV, JSON, LOG)",
                type=SUPPORTED_TEXT_UPLOADS,
                key="text_upload_file",
            )
            _sync_uploaded_file_to_text_area(uploaded)

            if st.session_state.get("text_upload_error"):
                st.error(st.session_state["text_upload_error"])

            default_text = st.session_state.get(
                "text_input_value",
                "Large numbers of people participate in sports that are extremely dangerous.\n\n"
                "Why do you think people do this?\n\nHow can the risks of participation be minimised?"
            )
            text = st.text_area("Text input", value=default_text, height=260, key="text_input_area")
            st.session_state["text_input_value"] = text

        with right:
            st.markdown("#### Input summary")
            if uploaded is not None:
                st.metric("Uploaded file", uploaded.name)
                st.metric("Uploaded size", fmt_bytes(len(uploaded.getvalue())))
            st.metric("Input size", fmt_bytes(len(text.encode("utf-8"))))
            st.metric("Characters", f"{len(text):,}")
            st.metric("Words", f"{len(text.split()):,}")
            if st.button("Use text", key="text_use_input"):
                _clear_text_downstream()
                st.session_state["text_input_value"] = text
                st.success("Ready")

    with st.container(border=True):
        step_header(2, "Compression")
        text = _current_input_text()
        if not text.strip():
            st.warning("Upload or paste text first.")

        storage_mode = st.radio(
            "Storage mode",
            ["No compression", "Compression"],
            horizontal=True,
            index=0 if st.session_state.get("text_storage_mode", "No compression") == "No compression" else 1,
            key="text_storage_mode",
        )

        if storage_mode == "No compression":
            method_key = "A_RAW_UTF8"
        else:
            method_key = "D_SPARSE_SEMANTIC"
            st.markdown("**Text compressor:** Tokenization")

        with st.expander("Method settings", expanded=False):
            a, b = st.columns(2)
            dense_vocab_size = 8192
            fixed_tokenizer_name = a.selectbox("Tokenizer", ["bert-base-uncased", "roberta-base"], index=0, key="text_fixed_tokenizer")
            fixed_error_detection_label = b.selectbox("Error detection", ["CRC8 per token", "CRC16 per block"], index=0, key="text_fixed_error_detection")
            fixed_error_detection = "token_crc8" if fixed_error_detection_label.startswith("CRC8") else "block_crc16"
            fixed_block_tokens = 16
            if fixed_error_detection == "block_crc16":
                fixed_block_tokens = st.selectbox("Block tokens", [8, 16, 24, 32], index=1, key="text_fixed_block_tokens")
            sparse_valid_vocab_size = 4096
            sparse_code_bits = 13

        if st.button("Run compression", key="text_run_selected_method", disabled=not bool(text.strip())):
            try:
                package = _make_text_method_package(
                    text,
                    method_key,
                    dense_vocab_size=int(dense_vocab_size),
                    sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                    sparse_code_bits=int(sparse_code_bits),
                    fixed_tokenizer_name=fixed_tokenizer_name,
                    fixed_error_detection=fixed_error_detection,
                    fixed_block_tokens=int(fixed_block_tokens),
                )
                st.session_state["text_selected_package"] = package
                for key in [
                    "text_dna", "text_strand_rows", "text_advanced_error_rows", "text_noisy_dna",
                    "text_decoded_text", "text_final_text", "text_validation", "text_substitution_table",
                    "text_candidate_table", "text_repair_table", "text_language_block_table", "text_token_recovery_df",
                ]:
                    st.session_state.pop(key, None)
                st.success("Done")
            except Exception as exc:
                st.error(f"Text method failed: {exc}")

        package = st.session_state.get("text_selected_package")
        if isinstance(package, dict):
            payload_binary_full = _bytes_to_bits_full(package.get("payload_bytes", b""))
            payload_binary_exact = _bytes_to_bits_exact(package.get("payload_bytes", b""), int(package.get("codec_bit_len", 0)))
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected method", package["method_short"])
            m2.metric("Payload size", fmt_bytes(len(package.get("payload_bytes", b""))))
            m3.metric("Reduction vs UTF-8 DNA", f"{package['text_code_reduction_percent']:.2f}%")
            if package.get("method_key") == "D_SPARSE_SEMANTIC":
                s1, s2, s3 = st.columns(3)
                meta = package.get("meta", {}) or {}
                s1.metric("Tokenizer", meta.get("tokenizer_name", "bert-base-uncased"))
                s2.metric("Token ID bits", meta.get("token_id_bits", meta.get("code_bits", "—")))
                s3.metric("Error detection", meta.get("error_detection", "CRC"))
            d1, d2 = st.columns(2)
            with d1:
                _download_bytes("Download payload bytes", package.get("payload_bytes", b""), "text_payload.bin", key="download_text_payload_bytes")
            with d2:
                _download_text("Download payload binary", payload_binary_full, "text_payload_binary.txt", key="download_text_payload_binary")
            with st.expander("Payload binary preview", expanded=False):
                st.text_area(
                    "Binary preview",
                    payload_binary_exact[:3000] + ("..." if len(payload_binary_exact) > 3000 else ""),
                    height=120,
                    key="text_payload_binary_preview",
                )

    with st.container(border=True):
        step_header(3, "DNA Encoding")
        package = st.session_state.get("text_selected_package")
        if not isinstance(package, dict):
            st.info("Run text compression first.")
        else:
            current_mapping = package.get("dna_mapping") if isinstance(package, dict) else None
            if current_mapping not in TEXT_DNA_MAPPING_OPTIONS:
                current_mapping = "Simple Mapping"
            mapping_name = st.selectbox(
                "Mapping rule",
                TEXT_DNA_MAPPING_OPTIONS,
                index=TEXT_DNA_MAPPING_OPTIONS.index(current_mapping),
                format_func=lambda m: TEXT_DNA_MAPPING_LABELS.get(m, m),
                key="text_mapping_rule_select",
            )

            if st.button("Run DNA Encoding", key="text_apply_mapping", type="primary"):
                try:
                    mapped_package = _apply_dna_mapping(package, mapping_name)
                    st.session_state["text_selected_package"] = mapped_package
                    st.session_state["text_dna"] = mapped_package["mapped_dna"]
                    st.session_state["text_mapping_meta"] = mapped_package["mapping_meta"]
                    st.session_state["text_mapping_bits"] = mapped_package["mapping_bits"]
                    for key in ["text_strand_rows", "text_advanced_error_rows", "text_noisy_dna", "text_decoded_text", "text_final_text", "text_validation"]:
                        st.session_state.pop(key, None)
                    st.success("Done")
                except Exception as exc:
                    st.error(f"DNA mapping failed: {exc}")

            package = st.session_state.get("text_selected_package")
            if isinstance(package, dict) and package.get("mapped_dna"):
                other_mapping = "RINF_B16" if package.get("dna_mapping") == "Simple Mapping" else "Simple Mapping"
                mapping_diff = "—"
                try:
                    alt_dna, _alt_bits, _alt_meta = encode_bytes_to_dna(package.get("payload_bytes", b""), other_mapping)
                    mapping_diff = f"{_dna_difference_count(package.get('mapped_dna', ''), alt_dna):,} bases"
                except Exception:
                    pass
                mapping_mode = (package.get("mapping_meta") or {}).get("mode", "—")
                roundtrip_status = "—"
                try:
                    _decoded_payload, _decoded_bits, _decoded_meta = decode_dna_with_mapping(
                        package.get("mapped_dna", ""),
                        package.get("dna_mapping", "Simple Mapping"),
                        package.get("mapping_meta", {}),
                    )
                    _expected_payload = bytes(package.get("payload_bytes", b""))
                    roundtrip_status = "Pass" if bytes(_decoded_payload)[:len(_expected_payload)] == _expected_payload else "Fail"
                except Exception:
                    roundtrip_status = "Fail"
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("DNA mapping", TEXT_DNA_MAPPING_LABELS.get(package.get("dna_mapping"), package.get("dna_mapping")))
                m2.metric("Mode", mapping_mode)
                m3.metric("Round-trip", roundtrip_status)
                m4.metric("Difference vs other", mapping_diff)
                m5.metric("Final DNA length", f"{package['mapped_dna_bases']:,} nt")
                preview_hash = hashlib.sha1((str(package.get("dna_mapping", "")) + "|" + package.get("mapped_dna", "")).encode("utf-8", errors="ignore")).hexdigest()[:10]
                st.text_area(
                    "Final mapped DNA preview",
                    _preview_seq(package["mapped_dna"]),
                    height=150,
                    key=f"text_mapped_dna_preview_{preview_hash}",
                )
                _download_text("Download final mapped DNA", package["mapped_dna"], "text_mapped_dna.txt", key=f"download_text_mapped_dna_{preview_hash}")
                _download_json("Download full text package metadata", _jsonable_package(package), "text_full_package_metadata.json", key="download_text_full_package_meta")

                st.markdown("##### 🧪 Payload-level noise")
                show_payload_noise = st.checkbox(
                    "Advanced: add noise directly to encoded DNA payload",
                    value=False,
                    key="text_show_payload_noise",
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
                        key="text_direct_sub_rate",
                    )
                    direct_seed = n2.number_input("Seed", min_value=0, max_value=999999, value=7, step=1, key="text_direct_sub_seed")
                    st.caption("Fast substitution-only test on the full encoded DNA. This bypasses strand design.")
                    if st.button("Add noise to DNA payload", key="text_add_direct_payload_noise", type="primary", use_container_width=True):
                        noisy_dna, n_sub = _direct_substitute_dna(package.get("mapped_dna", ""), float(direct_sub_rate), int(direct_seed))
                        st.session_state["text_noisy_dna"] = noisy_dna
                        st.session_state["text_direct_noisy_dna"] = noisy_dna
                        st.session_state["text_dna_error_stats"] = {
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
                        for key in ["text_decoded_text", "text_final_text", "text_validation", "text_decode_error"]:
                            st.session_state.pop(key, None)
                        st.success(f"Added {n_sub:,} substitutions to encoded DNA payload.")
                    if st.session_state.get("text_noisy_dna") and st.session_state.get("text_dna_error_stats", {}).get("quick_skip_strand"):
                        qs = st.session_state.get("text_dna_error_stats", {})
                        q1, q2, q3 = st.columns(3)
                        q1.metric("Direct substitutions", f"{int(qs.get('Substitute count', 0)):,}")
                        q2.metric("Noisy encoded DNA", f"{len(st.session_state.get('text_noisy_dna', '')):,} nt")
                        q3.metric("Skip Strand Design", "Yes")
                        st.text_area("Noisy DNA preview", _preview_seq(st.session_state.get("text_noisy_dna", ""), 600), height=120, key="text_direct_noisy_preview")
                        _download_text("Download noisy encoded DNA", st.session_state.get("text_noisy_dna", ""), "text_noisy_encoded_dna.txt", key="download_text_direct_noisy_dna")

    with st.container(border=True):
        step_header(4, "Strand Design")
        package = st.session_state.get("text_selected_package")
        dna = package.get("mapped_dna", "") if isinstance(package, dict) else ""
        if not dna:
            st.info("Run DNA Encoding first.")
        else:
            with st.expander("Strand design", expanded=not bool(st.session_state.get("text_strand_rows"))):
                target_len = st.number_input("Total strand length", min_value=80, max_value=250, value=125, step=1, key="text_std_total_len")
                index_len = st.number_input("SI length", min_value=0, max_value=24, value=8, step=1, key="text_std_index_len")
                fbr = st.text_input("FBR", value=FBR_DEFAULT, key="text_std_fbr")
                rbr = st.text_input("RBR", value=RBR_DEFAULT, key="text_std_rbr")
                if st.button("Prepare strands", key="text_build_standard_strands"):
                    try:
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
                        for row in rows:
                            row["Text method"] = package.get("method_short", "Text")
                            row["DNA mapping"] = TEXT_DNA_MAPPING_LABELS.get(package.get("dna_mapping"), package.get("dna_mapping"))
                        st.session_state.update({
                            "text_strand_rows": rows,
                            "text_advanced_error_rows": [],
                            "text_noisy_dna": "",
                            "text_dna_error_stats": {},
                            "text_decoded_text": "",
                            "text_final_text": "",
                            "text_validation": {},
                        })
                        st.success("Done")
                    except Exception as exc:
                        st.error(f"Strand design failed: {exc}")

            rows = st.session_state.get("text_strand_rows", [])
            if isinstance(rows, list) and rows:
                df = _strand_summary(rows)
                total_strand_len = sum(len(clean_dna(r.get("Full strand", ""))) for r in rows)
                baseline_nt = max(1, int(package.get("payload_byte_len", 0)) * 4)
                strand_expansion = total_strand_len / baseline_nt
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Designed strands", len(rows))
                c2.metric("Total strand length", f"{total_strand_len:,} nt")
                c3.metric("Strand Design expansion", f"{strand_expansion:.2f}×")
                c4.metric("DNA mapping", TEXT_DNA_MAPPING_LABELS.get(package.get("dna_mapping"), "—"))
                st.dataframe(df.head(200), use_container_width=True, hide_index=True)
                inspect_ids = [str(row.get("No.", i + 1)) for i, row in enumerate(rows)]
                selected_no = st.selectbox("Inspect designed strand", inspect_ids, index=0, key="text_inspect_designed_strand")
                selected_row = next((row for row in rows if str(row.get("No.", "")) == selected_no), rows[0])
                _render_text_segmented_strand(selected_row, "Designed strand")
                _download_df("Download prepared strands", pd.DataFrame(rows), "text_prepared_strands.csv", key="download_text_prepared_strands")

                st.markdown("---")
                st.markdown("##### 🧪 Strand-level noise")
                st.caption("Advanced: add noise to prepared small strands. Use this after Strand Design to simulate errors on app-generated strands.")
                enable_errors = st.checkbox("Advanced: add noise to prepared small strands", value=bool(st.session_state.get("text_advanced_error_rows")), key="text_enable_strand_level_errors")
                if enable_errors:
                    with st.container(border=True):
                        a, b, c, d = st.columns(4)
                        err_scope = a.selectbox("Error target", ["Payload only", "Index + Payload", "Full strand"], index=0, key="text_adv_scope")
                        err_sub = b.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.002, step=0.001, format="%.4f", key="text_adv_sub")
                        err_ins = c.number_input("Insertion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="text_adv_ins")
                        err_del = d.number_input("Deletion", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="text_adv_del")

                        if st.button("Add errors", key="text_run_advanced_errors"):
                            err_rows = []
                            for row in rows:
                                row_no = int(str(row.get("No.", "0") or "0"))
                                err_rows.append(_mutate_text_prepared_strand(
                                    row,
                                    scope=err_scope,
                                    substitution_rate=float(err_sub),
                                    insertion_rate=float(err_ins),
                                    deletion_rate=float(err_del),
                                    seed=17 + row_no * 1000003,
                                ))
                            noisy_dna = _text_dna_from_strand_rows_for_decode(err_rows, original_dna_len=len(clean_dna(package.get("mapped_dna", ""))))
                            events = _text_error_events_table(err_rows)
                            st.session_state.update({
                                "text_advanced_error_rows": err_rows,
                                "text_noisy_dna": noisy_dna,
                                "text_dna_error_stats": {
                                    "error_target": err_scope,
                                    "substitution_rate": float(err_sub),
                                    "insertion_rate": float(err_ins),
                                    "deletion_rate": float(err_del),
                                    "total_errors": int(len(events)),
                                    "noisy_dna_len": int(len(noisy_dna)),
                                },
                                "text_decoded_text": "",
                                "text_final_text": "",
                                "text_validation": {},
                            })
                            st.success("Done")
                else:
                    st.session_state["text_advanced_error_rows"] = []
                    st.session_state["text_noisy_dna"] = ""
                    st.session_state["text_dna_error_stats"] = {}

                err_rows = st.session_state.get("text_advanced_error_rows", []) if enable_errors else []
                if isinstance(err_rows, list) and err_rows:
                    events = _text_error_events_table(err_rows)
                    if isinstance(events, pd.DataFrame) and not events.empty and "Operation" in events.columns:
                        _ops = events["Operation"].astype(str).str.lower().value_counts().to_dict()
                    else:
                        _ops = {}
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Error strands", len(err_rows))
                    m2.metric("Added errors", int(sum(_ops.values())))
                    m3.metric("Noisy encoded data", f"{len(st.session_state.get('text_noisy_dna', '')):,} nt")
                    st.dataframe(_text_error_rows_table(err_rows), use_container_width=True, hide_index=True)

                    inspect_ids = [str(r.get("No.", i + 1)) for i, r in enumerate(err_rows)]
                    selected_err_no = st.selectbox("Inspect error strand", inspect_ids, index=0, key="text_inspect_error_strand")
                    error_row = next((r for r in err_rows if str(r.get("No.", "")) == selected_err_no), err_rows[0])
                    clean_row = next((r for r in rows if str(r.get("No.", "")) == selected_err_no), rows[0])
                    try:
                        ev_list = json.loads(error_row.get("Advanced error events", "[]") or "[]")
                    except Exception:
                        ev_list = []
                    err_positions = {
                        int(ev.get("position_original"))
                        for ev in ev_list
                        if ev.get("operation") in {"substitution", "deletion"} and str(ev.get("position_original", "")).isdigit()
                    }
                    _render_text_segmented_strand(clean_row, "Clean strand", error_positions=err_positions)
                    display_error_row = dict(error_row)
                    if int(str(error_row.get("Insertion count", "0") or "0")) == 0 and int(str(error_row.get("Deletion count", "0") or "0")) == 0:
                        full_err = clean_dna(error_row.get("Error full strand", ""))
                        cursor = 0
                        for region_name, region_seq in _text_row_regions(clean_row):
                            n = len(clean_dna(region_seq))
                            piece = full_err[cursor:cursor + n]
                            if region_name == "SI":
                                display_error_row["Strand index"] = piece
                                display_error_row["Index"] = piece
                            else:
                                display_error_row[region_name] = piece
                            cursor += n
                    _render_text_segmented_strand(display_error_row, "Error strand", error_positions=err_positions)

                    if not events.empty:
                        st.dataframe(events, use_container_width=True, hide_index=True)
                        _download_df("Download error table", events, "text_error_table.csv", key="download_text_error_table")
                    _download_df("Download error strands", pd.DataFrame(err_rows), "text_error_strands.csv", key="download_text_error_strands")
                    _download_text("Download noisy encoded DNA", st.session_state.get("text_noisy_dna", ""), "text_noisy_encoded_dna.txt", key="download_text_noisy_encoded_dna")

                with st.expander("Advanced sequencing simulation", expanded=False):
                    st.info("Reserved for sequencing/read simulation. Text decoding currently uses clean, error, or uploaded strands.")

    with st.container(border=True):
        step_header(5, "Decoding")
        package = st.session_state.get("text_selected_package")
        if not isinstance(package, dict) or not package.get("mapped_dna"):
            st.info("Run DNA Encoding first.")
        else:
            uploaded_dna_txt = st.file_uploader(
                "Upload full DNA TXT generated by this app",
                type=["txt"],
                key="text_decode_upload_dna_txt",
            )
            if uploaded_dna_txt is not None:
                try:
                    st.session_state["text_uploaded_decode_dna_txt"] = clean_dna(uploaded_dna_txt.getvalue().decode("utf-8", errors="ignore"))
                    st.success("Loaded uploaded full DNA TXT for decode.")
                except Exception as exc:
                    st.error(f"Could not load DNA TXT: {exc}")

            uploaded_strands = st.file_uploader(
                "Upload prepared/error strands CSV generated by this app",
                type=["csv"],
                key="text_decode_upload_strands_csv",
            )
            if uploaded_strands is not None:
                try:
                    upload_rows = _strand_rows_from_uploaded_csv(uploaded_strands)
                    upload_dna = _text_dna_from_strand_rows_for_decode(
                        upload_rows,
                        original_dna_len=len(clean_dna(package.get("mapped_dna", ""))),
                    )
                    st.session_state["text_uploaded_decode_strand_rows"] = upload_rows
                    st.session_state["text_uploaded_decode_dna"] = upload_dna
                    st.success(f"Loaded {len(upload_rows):,} uploaded strands for decode.")
                except Exception as exc:
                    st.error(f"Could not load strands CSV: {exc}")

            source_options = ["Current encoded DNA", "Noisy encoded DNA", "Upload full DNA TXT", "Upload prepared/error strands CSV"]
            source_label = st.radio("Reconstruction source", source_options, horizontal=True, key="text_decode_source_clean_error")
            decode_source_for_validation = _clean_decode_source_label(source_label)
            if source_label == "Upload prepared/error strands CSV":
                decode_source_for_validation = "Uploaded strands CSV"
                decode_dna = st.session_state.get("text_uploaded_decode_dna", "")
            elif source_label == "Upload full DNA TXT":
                decode_source_for_validation = "Uploaded DNA TXT"
                decode_dna = st.session_state.get("text_uploaded_decode_dna_txt", "")
            elif source_label == "Current encoded DNA":
                decode_dna = package.get("mapped_dna", "")
            else:
                decode_dna = st.session_state.get("text_noisy_dna", "")

            if not decode_dna:
                st.info("No DNA is available for the selected reconstruction source.")

            decode_language = "auto"
            language_scope = "global"
            confidence = 0.25
            use_repair = False
            if package.get("method_key") == "D_SPARSE_SEMANTIC":
                with st.expander("Tokenization repair options", expanded=False):
                    d, e = st.columns(2)
                    decode_language = d.selectbox(
                        "Decode language limit",
                        DECODE_LANGUAGE_OPTIONS,
                        format_func=lambda x: DECODE_LANGUAGE_LABELS.get(x, x),
                        index=0,
                        key="text_decode_language_select",
                    )
                    language_scope = e.selectbox(
                        "Language scope",
                        LANGUAGE_SCOPE_OPTIONS,
                        format_func=lambda x: LANGUAGE_SCOPE_LABELS.get(x, x),
                        index=0,
                        key="text_language_scope_select",
                    )
                    use_repair = st.checkbox("Use token repair", value=False, key="text_use_xlmr_repair")
                    if use_repair:
                        confidence = st.slider("Repair confidence", 0.05, 0.95, 0.25, 0.05, key="text_repair_confidence")

            with st.expander("DNA preview", expanded=False):
                decode_preview_hash = hashlib.sha1((source_label + "|" + decode_dna).encode("utf-8", errors="ignore")).hexdigest()[:10]
                st.text_area(
                    "Input DNA",
                    _preview_seq(decode_dna, 600),
                    height=120,
                    key=f"text_decode_input_dna_preview_{decode_preview_hash}",
                )

            if st.button("Run Decode", key="text_decode_button", disabled=not bool(decode_dna)):
                try:
                    dec, codec_dna_from_decode, mapping_decode_meta = _decode_full_pipeline(
                        package,
                        decode_dna,
                        decode_language=decode_language,
                        language_scope=language_scope,
                    )
                    decoded_text = dec.text
                    repaired_text = None
                    repair_table = pd.DataFrame()
                    repair_warning = ""
                    if use_repair and package.get("method_key") == "D_SPARSE_SEMANTIC" and dec.decode_ok and "decoded_tokens" in dec.meta:
                        try:
                            with st.spinner("Running fill-mask repair"):
                                codec_for_repair = _codec_from_package(package)
                                if getattr(codec_for_repair, "is_fixed_crc_token_codec", False):
                                    repaired_text, repair_table = codec_for_repair.repair_with_fill_mask(
                                        dec.meta.get("decoded_ids", []),
                                        dec.meta.get("invalid_positions", []),
                                        candidate_ids_by_position=dec.meta.get("candidate_ids_by_position", {}),
                                        confidence_threshold=float(confidence),
                                        prefer_language=(None if decode_language == "auto" else decode_language),
                                    )
                                else:
                                    repairer = XLMRContextRepairer()
                                    repaired_text, repair_table = repairer.repair_token_sequence(
                                        dec.meta["decoded_tokens"],
                                        code_candidate_table=dec.meta.get("code_candidate_table"),
                                        language_mode=("auto" if language_scope == "sentence_blocks" else decode_language),
                                        confidence_threshold=float(confidence),
                                    )
                        except ModuleNotFoundError as exc:
                            repair_warning = f"Fill-mask repair skipped: missing package {exc.name}."
                        except Exception as exc:
                            repair_warning = f"Fill-mask repair skipped: {exc}"

                    final_text = repaired_text if repaired_text is not None else decoded_text
                    metrics, word_df = _word_level_validation(package.get("original_text", ""), final_text)
                    sub_df = _text_error_events_table(st.session_state.get("text_advanced_error_rows", [])) if source_label == "Noisy encoded DNA" else pd.DataFrame()
                    dna_error_counts = {"substitution": 0, "deletion": 0, "insertion": 0}
                    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty and "Operation" in sub_df.columns:
                        vc = sub_df["Operation"].astype(str).str.lower().value_counts().to_dict()
                        dna_error_counts = {
                            "substitution": int(vc.get("substitution", 0)),
                            "deletion": int(vc.get("deletion", 0)),
                            "insertion": int(vc.get("insertion", 0)),
                        }

                    st.session_state["text_decoded_text"] = decoded_text
                    st.session_state["text_repaired_text"] = repaired_text
                    st.session_state["text_final_text"] = final_text
                    st.session_state["text_decode_error"] = ""
                    st.session_state["text_decode_ran"] = True
                    st.session_state["text_substitution_table"] = sub_df
                    st.session_state["text_candidate_table"] = dec.meta.get("code_candidate_table", pd.DataFrame())
                    st.session_state["text_repair_table"] = repair_table
                    st.session_state["text_token_recovery_df"] = _token_recovery_preview_table(package, dec.meta, repair_table)
                    st.session_state["text_codec_dna_from_decode"] = codec_dna_from_decode
                    st.session_state["text_validation"] = {
                        **metrics,
                        "decode_ok": bool(dec.decode_ok),
                        "decode_source": decode_source_for_validation,
                        "decode_source_label": source_label,
                        "repair_warning": repair_warning,
                        "decode_language": decode_language,
                        "language_scope": language_scope,
                        "dna_error_substitutions": int(dna_error_counts.get("substitution", 0)) if source_label == "Noisy encoded DNA" else 0,
                        "dna_error_deletions": int(dna_error_counts.get("deletion", 0)) if source_label == "Noisy encoded DNA" else 0,
                        "dna_error_insertions": int(dna_error_counts.get("insertion", 0)) if source_label == "Noisy encoded DNA" else 0,
                        "dna_error_total": int(sum(dna_error_counts.values())) if source_label == "Noisy encoded DNA" else 0,
                        "substituted_bases": int(dna_error_counts.get("substitution", 0)) if source_label == "Noisy encoded DNA" else 0,
                        "decoded_similarity": text_similarity(package.get("original_text", ""), decoded_text),
                        "final_similarity": text_similarity(package.get("original_text", ""), final_text),
                        "invalid_code_count": dec.meta.get("invalid_code_count"),
                        "exact_valid_code_count": dec.meta.get("exact_valid_code_count"),
                        "mapping_decode_meta": mapping_decode_meta,
                        "used_xlmr_repair": bool(repaired_text is not None),
                    }
                    if isinstance(mapping_decode_meta, dict) and isinstance(mapping_decode_meta.get("language_blocks"), pd.DataFrame):
                        st.session_state["text_language_block_table"] = mapping_decode_meta["language_blocks"]
                    st.session_state["text_word_validation_df"] = word_df
                    st.success("Done")
                except Exception as exc:
                    st.session_state["text_decode_error"] = str(exc)
                    st.error(f"Decode failed: {exc}")

            if st.session_state.get("text_decode_error"):
                st.error(st.session_state["text_decode_error"])

            validation = st.session_state.get("text_validation")
            if isinstance(validation, dict) and validation:
                st.markdown("#### Decode / Repair Summary")
                decode_summary_rows = [
                    {"Metric": "Source", "Value": validation.get("decode_source", "—")},
                    {"Metric": "DNA substitutions", "Value": validation.get("dna_error_substitutions", 0)},
                    {"Metric": "Language", "Value": DECODE_LANGUAGE_LABELS.get(validation.get("decode_language", "auto"), "Auto")},
                    {"Metric": "Invalid codes", "Value": validation.get("invalid_code_count", "—")},
                    {"Metric": "Decode status", "Value": "Pass" if validation.get("decode_ok") else "Review"},
                    {"Metric": "Token repair", "Value": "On" if validation.get("used_xlmr_repair") else "Off"},
                ]
                st.dataframe(pd.DataFrame(decode_summary_rows), use_container_width=True, hide_index=True)

                decoded_text = st.session_state.get("text_decoded_text", "") or ""
                final_text = st.session_state.get("text_final_text", "") or ""
                repaired_text = st.session_state.get("text_repaired_text")
                recovery_df = st.session_state.get("text_token_recovery_df", pd.DataFrame())
                st.markdown("#### Decoded Text Preview")
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("##### Decoded text")
                    st.text_area(
                        "Decoded text preview",
                        decoded_text[:5000],
                        height=240,
                        key="text_step5_decoded_preview",
                    )
                with p2:
                    st.markdown("##### Final recovered text" + (" / repaired" if repaired_text is not None else ""))
                    if isinstance(recovery_df, pd.DataFrame) and not recovery_df.empty:
                        st.markdown(_annotated_decoded_html(recovery_df, final_text[:5000]), unsafe_allow_html=True)
                    else:
                        st.text_area(
                            "Final recovered text preview",
                            final_text[:5000],
                            height=240,
                            key="text_step5_final_preview",
                        )

                d1, d2 = st.columns(2)
                with d1:
                    _download_text("Download decoded text", decoded_text, "text_decoded_output.txt", key="download_text_decoded_output_step5")
                with d2:
                    _download_text("Download final recovered text", final_text, "text_recovered_output.txt", key="download_text_recovered_output_step5")

                if isinstance(recovery_df, pd.DataFrame) and not recovery_df.empty:
                    display_recovery = recovery_df
                    if "status" in recovery_df.columns:
                        display_recovery = recovery_df[recovery_df["status"] != "Decoded"]
                    if not display_recovery.empty:
                        with st.expander("Token recovery table", expanded=False):
                            st.dataframe(display_recovery.head(120), use_container_width=True, hide_index=True)

    with st.container(border=True):
        step_header(6, "Summarization")
        package = st.session_state.get("text_selected_package")
        validation = st.session_state.get("text_validation")
        final_text = st.session_state.get("text_final_text", "")
        repaired_text = st.session_state.get("text_repaired_text")

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
            st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


        if not isinstance(package, dict) or not isinstance(validation, dict) or not validation:
            st.info("Decode text first.")
        else:
            original_text = package.get("original_text", "")
            payload = package.get("payload_bytes", b"") or b""
            dna = package.get("mapped_dna") or st.session_state.get("text_dna", "") or ""
            strand_rows = st.session_state.get("text_strand_rows", []) or []
            err_stats = st.session_state.get("text_dna_error_stats", {}) or {}
            recovery_df = st.session_state.get("text_token_recovery_df", pd.DataFrame())

            st.markdown("#### 📊 Summary")
            original_col, encoded_col, decoded_col = st.columns(3, gap="large")
            with original_col:
                st.markdown("##### Original")
                st.text_area("Original preview", original_text[:5000], height=220, key="text_summary_original_preview")
                st.dataframe(pd.DataFrame([
                    {"Property": "Characters", "Value": f"{len(original_text):,}"},
                    {"Property": "Words", "Value": f"{len(original_text.split()):,}"},
                    {"Property": "Original size", "Value": fmt_bytes(len(original_text.encode("utf-8")))},
                ]), hide_index=True, use_container_width=True)
            with encoded_col:
                st.markdown("##### Compressed/Encoded")
                method_preview = package.get("codec_dna", "") or dna
                st.text_area("Compressed/Encoded preview", _preview_seq(method_preview, 900), height=220, key="text_summary_encoded_preview")
                st.dataframe(pd.DataFrame([
                    {"Property": "Method", "Value": package.get("method_short", "—")},
                    {"Property": "Payload size", "Value": fmt_bytes(len(payload))},
                    {"Property": "DNA length", "Value": f"{len(dna):,} nt" if dna else "—"},
                ]), hide_index=True, use_container_width=True)
            with decoded_col:
                st.markdown("##### Decoded")
                st.text_area("Decoded preview", (final_text or "")[:5000], height=220, key="text_summary_decoded_preview")
                st.dataframe(pd.DataFrame([
                    {"Property": "Decode source", "Value": validation.get("decode_source_label", validation.get("decode_source", "—"))},
                    {"Property": "Decoded length", "Value": f"{len(final_text or ''):,} chars"},
                    {"Property": "Output status", "Value": "Readable" if final_text else "Failed"},
                ]), hide_index=True, use_container_width=True)

            storage_rows = [
                {"Metric": "Data type", "Value": "Text"},
                {"Metric": "Method", "Value": package.get("method_short", "—")},
                {"Metric": "Characters", "Value": f"{len(original_text):,}"},
                {"Metric": "Words", "Value": f"{len(original_text.split()):,}"},
                {"Metric": "Tokens", "Value": f"{int(package.get('num_tokens', 0)):,}" if package.get("num_tokens") is not None else "—"},
                {"Metric": "Original size", "Value": fmt_bytes(len(original_text.encode("utf-8")))},
                {"Metric": "Payload size", "Value": fmt_bytes(len(payload))},
                {"Metric": "Compression ratio", "Value": f"{len(original_text.encode('utf-8')) / max(1, len(payload)):.2f}x"},
                {"Metric": "Estimated DNA length", "Value": f"{len(payload) * 4:,} nt"},
            ]
            analysis_table("Compression analysis", compact_rows(
                storage_rows,
                ["Data type", "Method", "Characters", "Words", "Tokens", "Original size", "Payload size", "Compression ratio", "Estimated DNA length"],
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
                {"Metric": "Mapping rule", "Value": package.get("dna_mapping", "—")},
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

            noisy = st.session_state.get("text_noisy_dna", "") or ""
            error_rows = st.session_state.get("text_advanced_error_rows", []) or []
            is_noisy = bool(err_stats) or bool(noisy) or bool(error_rows) or int(validation.get("dna_error_total", 0) or 0) > 0
            error_rows_summary = [
                # {"Metric": "Error status", "Value": "Noisy" if is_noisy else "Clean"},
                # {"Metric": "Error level", "Value": "Payload-level" if err_stats.get("quick_skip_strand") else ("Strand-level" if error_rows else "Clean DNA")},
                {"Metric": "Error type", "Value": "Substitution" if is_noisy else "None"},
                {"Metric": "Substitution rate", "Value": err_stats.get("substitution_rate", "0")},
                {"Metric": "Substituted bases", "Value": f"{int(err_stats.get('Substitute count', err_stats.get('total_errors', validation.get('substituted_bases', validation.get('dna_error_substitutions', 0)))) or 0):,}"},
                {"Metric": "Error scope", "Value": err_stats.get("scope", "payload") if err_stats else "—"},
                {"Metric": "Affected strands", "Value": f"{len(error_rows):,}" if error_rows else ("—" if err_stats.get("quick_skip_strand") else "0")},
                # {"Metric": "Seed", "Value": err_stats.get("seed", "—")},
            ]
            analysis_table("Error Adding Report", compact_rows(
                error_rows_summary,
                ["Error status", "Error level", "Error type", "Substitution rate", "Substituted bases", "Error scope", "Affected strands", "Seed"],
            ))

            non_decoded = pd.DataFrame()
            if isinstance(recovery_df, pd.DataFrame) and not recovery_df.empty:
                non_decoded = recovery_df[recovery_df["status"] != "Decoded"] if "status" in recovery_df.columns else recovery_df
            repaired_count = 0
            unrepaired_count = 0
            masked_count = 0
            if isinstance(non_decoded, pd.DataFrame) and not non_decoded.empty:
                if "status" in non_decoded.columns:
                    repaired_count = int((non_decoded["status"].astype(str).str.contains("Repair", case=False, na=False)).sum())
                    unrepaired_count = int((non_decoded["status"].astype(str).str.contains("Fail|Unrepair|Invalid", case=False, na=False)).sum())
                masked_count = len(non_decoded)
            final_similarity = float(validation.get("final_similarity", 0.0))
            length_delta = len(final_text or "") - len(original_text or "")
            recovery_class = "Exact" if final_similarity >= 1.0 else ("Usable" if final_similarity >= 0.8 else ("Degraded" if final_similarity > 0 else "Failed"))
            quality_rows = [
                # {"Metric": "Decode source", "Value": validation.get("decode_source_label", validation.get("decode_source", "—"))},
                # {"Metric": "Decode status", "Value": "Success" if validation.get("decode_ok", True) else "Partial / failed"},
                # {"Metric": "Output status", "Value": "Readable" if final_text else "Failed"},
                # {"Metric": "Recovery class", "Value": recovery_class},
                {"Metric": "Text accuracy", "Value": f"{final_similarity * 100:.2f}%"},
                # {"Metric": "Exact match", "Value": "Yes" if final_similarity >= 1.0 else "No"},
                # {"Metric": "Length delta", "Value": f"{length_delta:+,} chars"},
                {"Metric": "Word accuracy", "Value": f"{float(validation.get('word_accuracy', 0.0)) * 100:.2f}%"},
                {"Metric": "Sequence similarity", "Value": f"{final_similarity:.4f}"},
                {"Metric": "Repaired tokens", "Value": f"{repaired_count:,}" if repaired_count else "—"},
                # {"Metric": "Unrepaired tokens", "Value": f"{unrepaired_count:,}" if unrepaired_count else "—"},
            ]
            analysis_table("Recovery Quality Report", compact_rows(
                quality_rows,
                ["Decode source", "Decode status", "Output status", "Recovery class", "Text accuracy", "Exact match", "Length delta", "Word accuracy", "Sequence similarity", "Repaired tokens", "Unrepaired tokens"],
            ))


            method_rows = [
                {"Property": "Full method label", "Value": METHOD_LABELS.get(package.get("method", ""), package.get("method_short", "—"))},
                {"Property": "Payload codec", "Value": package.get("codec_name", package.get("method_short", "—"))},
                {"Property": "Token repair", "Value": "On" if validation.get("used_xlmr_repair") else "Off"},
                {"Property": "Masked tokens", "Value": f"{masked_count:,}" if masked_count else "—"},
            ]
            word_df = st.session_state.get("text_word_validation_df", pd.DataFrame())
            display_df = pd.DataFrame()
            if isinstance(word_df, pd.DataFrame) and not word_df.empty:
                display_df = word_df[word_df["Status"] != "Match"] if "Status" in word_df.columns else word_df
            if method_rows or (isinstance(display_df, pd.DataFrame) and not display_df.empty) or (isinstance(non_decoded, pd.DataFrame) and not non_decoded.empty):
                with st.expander("Method-specific details", expanded=False):
                    if method_rows:
                        st.dataframe(pd.DataFrame([{"Property": r.get("Property", r.get("Metric", "—")), "Value": r.get("Value", "—")} for r in method_rows]), use_container_width=True, hide_index=True)
                    if isinstance(display_df, pd.DataFrame) and not display_df.empty:
                        st.markdown("##### Word-level differences")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    if isinstance(non_decoded, pd.DataFrame) and not non_decoded.empty:
                        st.markdown("##### Token recovery table")
                        st.dataframe(non_decoded.head(200), use_container_width=True, hide_index=True)



render_text_dna_storage_page = render_text_dna_storage_panel
