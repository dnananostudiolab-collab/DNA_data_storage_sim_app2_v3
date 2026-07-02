"""
text_sparse_semantic_dna_streamlit.py

Streamlit-ready module for text DNA storage using the best conservative version:

    Sparse semantic token code
    + invalid token-code detection -> [UNK]
    + code-distance candidate table
    + optional XLM-R repair for [UNK]/invalid tokens ONLY

This file is designed to be added to your existing DNA data storage Streamlit project
as a separate text-storage panel/tab.

Important design choice
-----------------------
This module intentionally DOES NOT do aggressive valid-token rewriting. It only repairs
explicit uncertainty ([UNK] / invalid token-code / wrong-language token). This avoids the
over-repair problem where a language model changes correct words such as:
    sports -> games
    essay -> article
    minimize -> reduce

The goal is:
    compressed + openable + non-exact readable text recovery under substitution errors

Not:
    exact reconstruction
    free-form rewriting
    ECC
"""

from __future__ import annotations

import difflib
import json
import math
import random
import re
import zlib
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import streamlit as st
except Exception:  # Allows core functions to be imported outside Streamlit.
    st = None


# =============================================================================
# DNA / binary helpers
# =============================================================================

BITS_TO_DNA = {"00": "A", "01": "C", "10": "G", "11": "T"}
DNA_TO_BITS = {v: k for k, v in BITS_TO_DNA.items()}


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    bits = bits[: len(bits) // 8 * 8]
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def int_to_bits(x: int, width: int) -> str:
    return format(int(x), f"0{width}b")


def bits_to_int(bits: str) -> int:
    return int(bits, 2) if bits else 0


def bits_to_dna(bits: str) -> str:
    if len(bits) % 2:
        bits += "0"
    return "".join(BITS_TO_DNA[bits[i:i + 2]] for i in range(0, len(bits), 2))


def dna_to_bits(dna: str) -> str:
    dna = clean_dna(dna)
    return "".join(DNA_TO_BITS.get(base, "00") for base in dna)


def clean_dna(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGT")


def raw_utf8_dna_bases(text: str) -> int:
    """Notepad-like baseline: 1 UTF-8 byte = 8 bits = 4 DNA bases."""
    return len(text.encode("utf-8")) * 4


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def add_substitution_errors(
    dna: str,
    error_rate: float,
    seed: int = 7,
    protect_first_bases: int = 0,
) -> Tuple[str, int, List[int]]:
    """
    Add substitution-only DNA errors.

    A -> C/G/T
    C -> A/G/T
    G -> A/C/T
    T -> A/C/G

    Length is unchanged.
    """
    rng = random.Random(seed)
    bases = ["A", "C", "G", "T"]
    out = list(clean_dna(dna))
    changed_positions: List[int] = []

    for i in range(protect_first_bases, len(out)):
        if rng.random() < float(error_rate):
            old = out[i]
            choices = [b for b in bases if b != old]
            out[i] = rng.choice(choices)
            changed_positions.append(i)

    return "".join(out), len(changed_positions), changed_positions


def dna_substitution_table(
    original_dna: str,
    noisy_dna: str,
    changed_positions: Sequence[int],
    window: int = 12,
    max_show: int = 50,
) -> pd.DataFrame:
    original_dna = clean_dna(original_dna)
    noisy_dna = clean_dna(noisy_dna)
    rows = []

    for pos in list(changed_positions)[:max_show]:
        start = max(0, pos - window)
        end = min(len(original_dna), pos + window + 1)
        rows.append({
            "position": pos,
            "old_base": original_dna[pos] if pos < len(original_dna) else "",
            "new_base": noisy_dna[pos] if pos < len(noisy_dna) else "",
            "original_context": original_dna[start:end],
            "noisy_context": noisy_dna[start:end],
            "marker": " " * (pos - start) + "^",
        })

    return pd.DataFrame(rows)


def gc_content(dna: str) -> float:
    dna = clean_dna(dna)
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna)


def longest_homopolymer(dna: str) -> int:
    dna = clean_dna(dna)
    if not dna:
        return 0
    best = cur = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return max(best, cur)


def preview_seq(seq: str, n: int = 300) -> str:
    seq = str(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


# =============================================================================
# Tokenization / language helpers
# =============================================================================

def simple_tokenize(text: str) -> List[str]:
    """
    Simple Unicode-aware tokenizer:
    - English/Vietnamese/French-like Latin words
    - CJK character groups
    - numbers
    - punctuation
    """
    return re.findall(
        r"[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|[\u4e00-\u9fff]+|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )


def simple_detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for tok in tokens:
        if tok in ".,;:!?)]}":
            out = out.rstrip() + tok + " "
        elif tok in "([{":
            out += tok
        elif re.search(r"[\u4e00-\u9fff]", tok):
            out += tok
        else:
            out += tok + " "
    return out.strip()


VI_SPECIFIC_CHARS = set(
    "ăâêôơưđĂÂÊÔƠƯĐ"
    "áàảãạấầẩẫậắằẳẵặ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
)

FR_CUES = {
    "la", "le", "les", "des", "du", "une", "un", "de", "données",
    "numériques", "technologie", "stockage", "préserver", "peut",
    "avec", "pour", "dans"
}

EN_CUES = {
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in",
    "for", "with", "text", "data", "storage", "technology", "digital",
    "errors", "sports", "people", "games", "risk", "dangerous",
}


def classify_token_language(tok: str) -> str:
    if tok in {"[PAD]", "[UNK]"}:
        return "special"
    if re.fullmatch(r"[0-9]+", tok):
        return "number"
    if re.fullmatch(r"[\W_]+", tok) and not re.search(r"[\u4e00-\u9fff]", tok):
        return "punct"
    if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
        return "zh"

    low = tok.lower()
    if any(ch in VI_SPECIFIC_CHARS for ch in tok):
        return "vi"
    if low in FR_CUES or any(ch in "éèêëàâîïôùûçÉÈÊËÀÂÎÏÔÙÛÇ" for ch in tok):
        return "fr"
    return "en"


def count_words_matching(block: str, vocab: set[str]) -> int:
    toks = re.findall(r"[A-Za-zÀ-ỹ]+", block.lower())
    return sum(1 for t in toks if t in vocab)


def detect_block_language(block: str, manual_language: str = "auto") -> str:
    if manual_language and manual_language != "auto":
        return manual_language

    total = max(len(block), 1)
    cjk_ratio = len(re.findall(r"[\u4e00-\u9fff]", block)) / total
    if cjk_ratio > 0.15:
        return "zh"

    vi_ratio = sum(ch in VI_SPECIFIC_CHARS for ch in block) / total
    if vi_ratio > 0.025:
        return "vi"

    fr_score = count_words_matching(block, FR_CUES)
    en_score = count_words_matching(block, EN_CUES)
    if fr_score >= 2 and fr_score >= en_score:
        return "fr"
    if en_score >= 1:
        return "en"

    latin_ratio = len(re.findall(r"[A-Za-zÀ-ỹ]", block)) / total
    return "en" if latin_ratio > 0.4 else "unknown"


def is_valid_for_language(word: str, lang: str) -> bool:
    w = str(word).strip()
    if w == "" or w == "[UNK]" or "<0x" in w:
        return False

    if re.fullmatch(r"[\W_]+", w) and not re.search(r"[\u4e00-\u9fff]", w):
        return True

    if lang == "en":
        return re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", w) is not None
    if lang == "fr":
        return re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?", w) is not None
    if lang == "vi":
        return re.fullmatch(r"[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?", w) is not None
    if lang == "zh":
        return re.fullmatch(r"[\u4e00-\u9fff]+", w) is not None

    return bool(re.fullmatch(r"[A-Za-zÀ-ỹ]+|[\u4e00-\u9fff]+", w))


DEFAULT_EXTRA_VOCAB = [
    "[PAD]", "[UNK]",
    # English / domain
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "for",
    "with", "as", "by", "from", "text", "data", "storage", "DNA",
    "technology", "digital", "preservation", "semantic", "structure",
    "repeated", "words", "contextual", "redundancy", "compressed", "binary",
    "stream", "token", "tokens", "IDs", "substitution", "errors", "occur",
    "decoded", "openable", "wrong", "promising", "long-term", "different",
    "valid", "may", "become", "some", "although", "model", "language",
    "context", "repair", "candidate", "people", "sports", "games",
    "dangerous", "risk", "risks", "participate", "participation", "players",
    "money", "professional", "training", "protection", "equipment",
    # Vietnamese / domain
    "Công", "nghệ", "lưu", "trữ", "có", "thể", "hữu", "ích", "cho", "bảo",
    "quản", "dữ", "liệu", "lâu", "dài", "văn", "bản", "cấu", "trúc", "ngữ",
    "nghĩa", "nhiều", "từ", "lặp", "lại",
    # French / domain
    "La", "Le", "Les", "la", "le", "les", "technologie", "de", "stockage",
    "ADN", "peut", "préserver", "données", "numériques",
    # Chinese chunks
    "数据存储", "长期保存", "非常重要", "数据", "存储", "长期", "保存", "重要",
]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Encoded:
    method: str
    dna: str
    meta: Dict[str, Any]


@dataclass
class Decoded:
    method: str
    text: str
    decode_ok: bool
    meta: Dict[str, Any]


# =============================================================================
# Codecs
# =============================================================================

class UTF8RawTextCodec:
    name = "A_no_compression_utf8_raw"

    def encode(self, text: str) -> Encoded:
        data = text.encode("utf-8")
        dna = bits_to_dna(bytes_to_bits(data))
        return Encoded(self.name, dna, {
            "raw_utf8_bytes": len(data),
            "dna_bases": len(dna),
        })

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        data = bits_to_bytes(dna_to_bits(dna))
        text = data.decode("utf-8", errors="replace")
        return Decoded(self.name, text, True, {"decoded_bytes": len(data)})


class WholeZlibTextCodec:
    name = "B_bad_binary_compression_whole_zlib"

    def __init__(self, level: int = 6):
        self.level = level

    def encode(self, text: str) -> Encoded:
        raw = text.encode("utf-8")
        comp = zlib.compress(raw, self.level)
        dna = bits_to_dna(bytes_to_bits(comp))
        return Encoded(self.name, dna, {
            "raw_utf8_bytes": len(raw),
            "compressed_bytes": len(comp),
            "dna_bases": len(dna),
        })

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        comp = bits_to_bytes(dna_to_bits(dna))
        try:
            raw = zlib.decompress(comp)
            text = raw.decode("utf-8", errors="replace")
            return Decoded(self.name, text, True, {"decoded_compressed_bytes": len(comp)})
        except Exception as e:
            return Decoded(
                self.name,
                f"[DECODE FAILED: {type(e).__name__}({str(e)!r})]",
                False,
                {"error": repr(e), "decoded_compressed_bytes": len(comp)},
            )


class DenseFixedVocabTokenTextCodec:
    name_prefix = "C_dense_fixed_vocab_token"

    def __init__(self, vocab: Sequence[str], vocab_size: int = 8192):
        self.vocab_size = 1 << math.ceil(math.log2(vocab_size))
        self.token_bits = int(math.log2(self.vocab_size))
        self.name = f"{self.name_prefix}_{self.vocab_size}"

        ordered: List[str] = []
        for tok in ["[PAD]", "[UNK]"]:
            if tok not in ordered:
                ordered.append(tok)

        for tok in vocab:
            if tok not in ordered:
                ordered.append(tok)
            if len(ordered) >= self.vocab_size:
                break

        while len(ordered) < self.vocab_size:
            ordered.append(f"[FILL_{len(ordered)}]")

        self.id_to_token = dict(enumerate(ordered))
        self.token_to_id = {tok: i for i, tok in self.id_to_token.items()}
        self.unk_id = self.token_to_id["[UNK]"]

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 8192) -> "DenseFixedVocabTokenTextCodec":
        counts: Dict[str, int] = {}
        for tok in simple_tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        sorted_tokens = sorted(counts.keys(), key=lambda x: (-counts[x], x))
        return cls(DEFAULT_EXTRA_VOCAB + sorted_tokens, vocab_size=vocab_size)

    def encode(self, text: str) -> Encoded:
        tokens = simple_tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        bits = "".join(int_to_bits(i, self.token_bits) for i in ids)
        dna = bits_to_dna(bits)
        return Encoded(self.name, dna, {
            "num_tokens": len(ids),
            "token_bits": self.token_bits,
            "vocab_size": self.vocab_size,
            "token_ids": ids,
            "tokens": tokens,
            "vocab": [self.id_to_token[i] for i in range(self.vocab_size)],
            "dna_bases": len(dna),
        })

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        bits = dna_to_bits(dna)
        num_tokens = int(meta.get("num_tokens", 0)) if meta else 0
        ids: List[int] = []

        for i in range(0, len(bits), self.token_bits):
            chunk = bits[i:i + self.token_bits]
            if len(chunk) < self.token_bits:
                break
            tid = bits_to_int(chunk)
            if tid >= self.vocab_size:
                tid = self.unk_id
            ids.append(tid)

        if num_tokens:
            ids = ids[:num_tokens]

        tokens = []
        for tid in ids:
            tok = self.id_to_token.get(tid, "[UNK]")
            if tok.startswith("[FILL_"):
                tok = "[UNK]"
            if tok != "[PAD]":
                tokens.append(tok)

        return Decoded(self.name, simple_detokenize(tokens), True, {
            "decoded_ids": ids,
            "decoded_tokens": tokens,
        })


def _bit_parity(x: int) -> int:
    return int(x.bit_count() % 2)


def _even_parity_codewords(code_bits: int) -> List[int]:
    """
    Return all codewords with even bit parity.
    For 13 bits, there are 4096 valid even-parity codewords out of 8192 possible.
    """
    total = 1 << int(code_bits)
    return [x for x in range(total) if _bit_parity(x) == 0]


class SparseSemanticTokenTextCodec:
    """
    Proposed conservative text DNA codec:

        token -> semantic grouped ID -> sparse even-parity codeword -> DNA

    Decode:
        exact valid codeword -> token
        invalid codeword -> [UNK]
        candidate table -> nearest valid tokens by Hamming distance

    This is not ECC. It is error-detecting/uncertainty-shaping token coding.
    """

    name_prefix = "D_proposed_sparse_semantic_token"

    def __init__(
        self,
        vocab: Sequence[str],
        valid_vocab_size: int = 4096,
        code_bits: int = 13,
    ):
        self.valid_vocab_size = int(valid_vocab_size)
        self.code_bits = int(code_bits)
        self.codespace_size = 1 << self.code_bits

        valid_codes = _even_parity_codewords(self.code_bits)
        if self.valid_vocab_size > len(valid_codes):
            raise ValueError(
                f"valid_vocab_size={self.valid_vocab_size} is too large for "
                f"{self.code_bits}-bit even-parity space ({len(valid_codes)} valid codes)."
            )

        self.valid_codewords = valid_codes[: self.valid_vocab_size]
        self.name = f"{self.name_prefix}_{self.valid_vocab_size}_codespace_{self.codespace_size}"

        ordered = self._semantic_order(vocab)
        ordered = ordered[: self.valid_vocab_size]
        while len(ordered) < self.valid_vocab_size:
            ordered.append(f"[FILL_{len(ordered)}]")

        # Ensure PAD/UNK exist at the front if possible.
        if "[PAD]" not in ordered:
            ordered[0] = "[PAD]"
        if "[UNK]" not in ordered:
            ordered[1] = "[UNK]"

        self.id_to_token = dict(enumerate(ordered))
        self.token_to_id = {tok: i for i, tok in self.id_to_token.items()}
        self.unk_id = self.token_to_id.get("[UNK]", 1)

        self.token_id_to_codeword = {
            tid: self.valid_codewords[tid]
            for tid in range(self.valid_vocab_size)
        }
        self.codeword_to_token_id = {
            code: tid
            for tid, code in self.token_id_to_codeword.items()
        }
        self.token_id_to_code_bits = {
            tid: int_to_bits(code, self.code_bits)
            for tid, code in self.token_id_to_codeword.items()
        }
        self.token_language = {
            tok: classify_token_language(tok)
            for tok in self.token_to_id.keys()
        }

    @staticmethod
    def _semantic_order(vocab: Sequence[str]) -> List[str]:
        groups: Dict[str, List[str]] = {
            "special": [],
            "punct": [],
            "number": [],
            "en": [],
            "vi": [],
            "fr": [],
            "zh": [],
            "other": [],
        }
        seen = set()
        for tok in vocab:
            if tok in seen:
                continue
            seen.add(tok)
            lang = classify_token_language(tok)
            if lang not in groups:
                lang = "other"
            groups[lang].append(tok)

        ordered: List[str] = []
        for group_name in ["special", "punct", "number", "en", "vi", "fr", "zh", "other"]:
            ordered.extend(groups[group_name])
        return ordered

    @classmethod
    def from_text(
        cls,
        text: str,
        valid_vocab_size: int = 4096,
        code_bits: int = 13,
    ) -> "SparseSemanticTokenTextCodec":
        counts: Dict[str, int] = {}
        for tok in simple_tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        sorted_tokens = sorted(counts.keys(), key=lambda x: (-counts[x], x))
        return cls(
            DEFAULT_EXTRA_VOCAB + sorted_tokens,
            valid_vocab_size=valid_vocab_size,
            code_bits=code_bits,
        )

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "SparseSemanticTokenTextCodec":
        vocab = meta.get("vocab")
        if not vocab:
            raise ValueError("Sparse codec metadata requires a saved 'vocab' list.")
        return cls(
            vocab,
            valid_vocab_size=int(meta.get("valid_vocab_size", 4096)),
            code_bits=int(meta.get("code_bits", 13)),
        )

    def encode(self, text: str) -> Encoded:
        tokens = simple_tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        bits = "".join(self.token_id_to_code_bits[i] for i in ids)
        dna = bits_to_dna(bits)
        vocab = [self.id_to_token[i] for i in range(self.valid_vocab_size)]

        return Encoded(self.name, dna, {
            "num_tokens": len(ids),
            "code_bits": self.code_bits,
            "codespace_size": self.codespace_size,
            "valid_vocab_size": self.valid_vocab_size,
            "token_ids": ids,
            "tokens": tokens,
            "vocab": vocab,
            "dna_bases": len(dna),
            "encoding": "sparse_even_parity_code",
        })

    def candidate_tokens_from_bits(
        self,
        noisy_bits: str,
        top_k: int = 8,
        prefer_language: Optional[str] = None,
        max_hamming: int = 4,
    ) -> List[Dict[str, Any]]:
        rows = []
        for tid in range(self.valid_vocab_size):
            tok = self.id_to_token.get(tid, "[UNK]")
            if tok.startswith("[FILL_"):
                continue
            code_bits = self.token_id_to_code_bits[tid]
            dist = sum(a != b for a, b in zip(noisy_bits, code_bits))
            if dist > max_hamming:
                continue
            lang = self.token_language.get(tok, "other")
            lang_penalty = 0
            if prefer_language and lang not in {prefer_language, "punct", "number", "special"}:
                lang_penalty = 2
            score = dist + lang_penalty
            rows.append({
                "token_id": tid,
                "token": tok,
                "language": lang,
                "hamming_distance": dist,
                "score": score,
            })

        rows.sort(key=lambda r: (r["score"], r["hamming_distance"], r["token_id"]))
        return rows[:top_k]

    def decode(
        self,
        dna: str,
        meta: Optional[Dict[str, Any]] = None,
        candidate_top_k: int = 8,
        prefer_language: Optional[str] = None,
    ) -> Decoded:
        bits = dna_to_bits(dna)
        num_tokens = int(meta.get("num_tokens", 0)) if meta else 0

        decoded_tokens: List[str] = []
        decoded_ids: List[int] = []
        candidate_rows: List[Dict[str, Any]] = []
        invalid_code_count = 0
        exact_valid_code_count = 0
        token_index = 0

        for i in range(0, len(bits), self.code_bits):
            chunk = bits[i:i + self.code_bits]
            if len(chunk) < self.code_bits:
                break

            code_int = bits_to_int(chunk)
            exact_valid = code_int in self.codeword_to_token_id

            if exact_valid:
                tid = self.codeword_to_token_id[code_int]
                tok = self.id_to_token.get(tid, "[UNK]")
                if tok.startswith("[FILL_"):
                    tok = "[UNK]"
                exact_valid_code_count += 1
            else:
                tid = self.unk_id
                tok = "[UNK]"
                invalid_code_count += 1

            decoded_ids.append(tid)
            decoded_tokens.append(tok)

            candidates = self.candidate_tokens_from_bits(
                chunk,
                top_k=candidate_top_k,
                prefer_language=prefer_language,
            )
            for rank, cand in enumerate(candidates, start=1):
                candidate_rows.append({
                    "token_position": token_index,
                    "rank": rank,
                    "exact_valid_code": exact_valid,
                    **cand,
                })

            token_index += 1

        if num_tokens:
            decoded_tokens = decoded_tokens[:num_tokens]
            decoded_ids = decoded_ids[:num_tokens]
            candidate_rows = [r for r in candidate_rows if r["token_position"] < num_tokens]

        text = simple_detokenize([t for t in decoded_tokens if t != "[PAD]"])
        return Decoded(self.name, text, True, {
            "decoded_ids": decoded_ids,
            "decoded_tokens": decoded_tokens,
            "code_candidate_table": pd.DataFrame(candidate_rows),
            "invalid_code_count": invalid_code_count,
            "exact_valid_code_count": exact_valid_code_count,
        })


# =============================================================================
# Conservative XLM-R repair: [UNK]/invalid only
# =============================================================================

def _load_xlmr_model_uncached(model_name: str = "xlm-roberta-base"):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


if st is not None:
    @st.cache_resource(show_spinner=False)
    def _load_xlmr_model(model_name: str = "xlm-roberta-base"):
        return _load_xlmr_model_uncached(model_name)
else:
    _XLMR_CACHE: Dict[str, Any] = {}

    def _load_xlmr_model(model_name: str = "xlm-roberta-base"):
        if model_name not in _XLMR_CACHE:
            _XLMR_CACHE[model_name] = _load_xlmr_model_uncached(model_name)
        return _XLMR_CACHE[model_name]


class ExplicitUncertaintyXLMRRepairer:
    """
    Conservative repairer:
    - repairs only [UNK] or invalid-language tokens
    - does not rewrite valid tokens
    - returns a transparent candidate table
    """

    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model_name = model_name
        self.tokenizer, self.model, self.torch, self.device = _load_xlmr_model(model_name)

    def mlm_candidates_for_position(
        self,
        tokens: Sequence[str],
        pos: int,
        language: str = "en",
        top_k: int = 20,
        window: int = 35,
    ) -> List[Dict[str, Any]]:
        start = max(0, pos - window)
        end = min(len(tokens), pos + window + 1)

        local = list(tokens[start:end])
        local_pos = pos - start
        local[local_pos] = self.tokenizer.mask_token

        text = simple_detokenize(local)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        with self.torch.no_grad():
            logits = self.model(**inputs).logits

        mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            return []

        idx = mask_positions[1][0]
        probs = self.torch.softmax(logits[0, idx], dim=-1)
        top = self.torch.topk(probs, k=top_k * 80)

        out: List[Dict[str, Any]] = []
        seen = set()

        for tid, prob in zip(top.indices.tolist(), top.values.tolist()):
            cand = self.tokenizer.decode([tid]).replace("▁", "").strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)

            if not is_valid_for_language(cand, language):
                continue
            if language != "zh" and len(cand) <= 1:
                continue

            out.append({
                "candidate": cand,
                "probability": float(prob),
                "mlm_token_id": int(tid),
            })
            if len(out) >= top_k:
                break

        return out

    @staticmethod
    def code_candidates_for_position(
        code_candidate_table: pd.DataFrame,
        token_position: int,
        language: str = "en",
        top_k: int = 12,
        max_hamming: int = 4,
    ) -> List[str]:
        if code_candidate_table is None or len(code_candidate_table) == 0:
            return []

        sub = code_candidate_table[code_candidate_table["token_position"] == token_position].copy()
        if len(sub) == 0:
            return []

        if "hamming_distance" in sub.columns:
            sub = sub[sub["hamming_distance"] <= max_hamming]

        if "language" in sub.columns:
            sub = sub[sub["language"].isin([language, "punct", "number", "special", "en"])]

        if len(sub) == 0:
            return []

        sort_cols = []
        if "score" in sub.columns:
            sort_cols.append("score")
        if "hamming_distance" in sub.columns:
            sort_cols.append("hamming_distance")
        if sort_cols:
            sub = sub.sort_values(sort_cols)

        toks: List[str] = []
        seen = set()
        for tok in sub["token"].tolist():
            tok = str(tok)
            if tok.startswith("[FILL_") or tok == "[PAD]":
                continue
            if tok not in seen:
                toks.append(tok)
                seen.add(tok)
            if len(toks) >= top_k:
                break

        return toks

    @staticmethod
    def _norm(x: str) -> str:
        return str(x).replace("▁", "").strip().lower()

    def choose_replacement(
        self,
        mlm_candidates: List[Dict[str, Any]],
        code_candidates: List[str],
        confidence_threshold: float = 0.25,
        mlm_only_threshold: float = 0.25,
    ) -> Tuple[str, bool, str, float]:
        if not mlm_candidates:
            return "[UNK]", False, "no_mlm_candidates", 0.0

        code_set = {self._norm(x) for x in code_candidates}

        # Prefer agreement between code-distance and MLM.
        for cand in mlm_candidates:
            c = cand["candidate"]
            p = float(cand["probability"])
            if self._norm(c) in code_set and p >= confidence_threshold:
                return c, True, "mlm_and_code_agree", p

        # For explicit [UNK], MLM-only is allowed, but only above threshold.
        top = mlm_candidates[0]
        if float(top["probability"]) >= mlm_only_threshold:
            return top["candidate"], True, "mlm_only_high_confidence", float(top["probability"])

        return "[UNK]", False, "below_threshold", float(top["probability"])

    def repair_tokens(
        self,
        decoded_tokens: Sequence[str],
        code_candidate_table: Optional[pd.DataFrame] = None,
        language_mode: str = "auto",
        confidence_threshold: float = 0.25,
        mlm_only_threshold: float = 0.25,
        top_k: int = 20,
        window: int = 35,
    ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        tokens = list(decoded_tokens)

        if language_mode == "auto":
            language = detect_block_language(simple_detokenize(tokens), manual_language="auto")
        else:
            language = language_mode

        repaired = list(tokens)
        rows: List[Dict[str, Any]] = []

        explicit_positions = []
        for i, tok in enumerate(tokens):
            if tok == "[UNK]":
                explicit_positions.append(i)
            elif tok and tok not in {"[PAD]"} and not is_valid_for_language(tok, language):
                explicit_positions.append(i)

        for pos in explicit_positions:
            old = tokens[pos]
            mlm_cands = self.mlm_candidates_for_position(
                repaired,
                pos,
                language=language,
                top_k=top_k,
                window=window,
            )
            code_cands = self.code_candidates_for_position(
                code_candidate_table if code_candidate_table is not None else pd.DataFrame(),
                pos,
                language=language,
                top_k=12,
            )

            chosen, accepted, reason, prob = self.choose_replacement(
                mlm_cands,
                code_cands,
                confidence_threshold=confidence_threshold,
                mlm_only_threshold=mlm_only_threshold,
            )

            repaired[pos] = chosen if accepted else "[UNK]"
            rows.append({
                "token_position": pos,
                "old_token": old,
                "chosen": repaired[pos],
                "language": language,
                "probability": prob,
                "accepted": accepted,
                "repair_type": "explicit_uncertainty_only",
                "reason": reason,
                "code_candidates": code_cands,
                "mlm_candidates": mlm_cands[:10],
            })

        repaired_text = simple_detokenize([t for t in repaired if t != "[PAD]"])
        repair_report = pd.DataFrame(rows)
        block_report = pd.DataFrame([{
            "block": 0,
            "dominant_language": language,
            "decoded_text": simple_detokenize(tokens),
            "repaired_text": repaired_text,
            "num_explicit_uncertain_tokens": len(explicit_positions),
        }])
        return repaired_text, repair_report, block_report


# =============================================================================
# Benchmark / inspection API
# =============================================================================

def make_text_codecs(
    text: str,
    dense_vocab_size: int = 8192,
    sparse_valid_vocab_size: int = 4096,
    sparse_code_bits: int = 13,
) -> List[Any]:
    return [
        UTF8RawTextCodec(),
        WholeZlibTextCodec(),
        DenseFixedVocabTokenTextCodec.from_text(text, vocab_size=dense_vocab_size),
        SparseSemanticTokenTextCodec.from_text(
            text,
            valid_vocab_size=sparse_valid_vocab_size,
            code_bits=sparse_code_bits,
        ),
    ]


def run_single_text_method(
    text: str,
    codec: Any,
    error_rate: float,
    seed: int = 7,
) -> Dict[str, Any]:
    enc = codec.encode(text)
    noisy_dna, n_subs, changed_positions = add_substitution_errors(enc.dna, error_rate=error_rate, seed=seed)
    dec = codec.decode(noisy_dna, enc.meta)

    raw_bases = raw_utf8_dna_bases(text)

    row: Dict[str, Any] = {
        "group": enc.method.split("_", 1)[0],
        "method": enc.method,
        "error_rate": error_rate,
        "dna_bases": len(enc.dna),
        "substituted_bases": n_subs,
        "actual_substitution_rate": n_subs / max(len(enc.dna), 1),
        "compression_ratio_vs_utf8_raw": len(enc.dna) / max(raw_bases, 1),
        "dna_reduction_percent": (1.0 - len(enc.dna) / max(raw_bases, 1)) * 100.0,
        "decode_ok": dec.decode_ok,
        "decoded_similarity": text_similarity(text, dec.text),
        "decoded_preview": dec.text[:180].replace("\n", " "),
        "encoded": enc,
        "decoded": dec,
        "noisy_dna": noisy_dna,
        "changed_positions": changed_positions,
    }

    if "invalid_code_count" in dec.meta:
        row["invalid_code_count"] = dec.meta.get("invalid_code_count")
        row["exact_valid_code_count"] = dec.meta.get("exact_valid_code_count")
        cand = dec.meta.get("code_candidate_table", pd.DataFrame())
        row["num_code_candidates"] = len(cand)

    return row


def run_text_dna_benchmark(
    text: str,
    error_rates: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
    dense_vocab_size: int = 8192,
    sparse_valid_vocab_size: int = 4096,
    sparse_code_bits: int = 13,
    seed: int = 7,
    include_objects: bool = False,
) -> pd.DataFrame:
    codecs = make_text_codecs(
        text,
        dense_vocab_size=dense_vocab_size,
        sparse_valid_vocab_size=sparse_valid_vocab_size,
        sparse_code_bits=sparse_code_bits,
    )

    rows: List[Dict[str, Any]] = []
    for codec in codecs:
        for er in error_rates:
            row = run_single_text_method(text, codec, er, seed=seed)
            if not include_objects:
                row = {
                    k: v for k, v in row.items()
                    if k not in {"encoded", "decoded", "noisy_dna", "changed_positions"}
                }
            rows.append(row)

    return pd.DataFrame(rows)


def inspect_sparse_text_dna(
    text: str,
    error_rate: float = 0.01,
    sparse_valid_vocab_size: int = 4096,
    sparse_code_bits: int = 13,
    seed: int = 7,
    use_xlmr_repair: bool = True,
    repair_language_mode: str = "auto",
    repair_confidence_threshold: float = 0.25,
    repair_mlm_only_threshold: float = 0.25,
) -> Dict[str, Any]:
    codec = SparseSemanticTokenTextCodec.from_text(
        text,
        valid_vocab_size=sparse_valid_vocab_size,
        code_bits=sparse_code_bits,
    )
    enc = codec.encode(text)
    noisy_dna, n_subs, changed_positions = add_substitution_errors(enc.dna, error_rate=error_rate, seed=seed)
    dec = codec.decode(noisy_dna, enc.meta)

    decoded_tokens = dec.meta.get("decoded_tokens", simple_tokenize(dec.text))
    candidate_table = dec.meta.get("code_candidate_table", pd.DataFrame())

    repaired_text = None
    repair_report = pd.DataFrame()
    block_report = pd.DataFrame()
    repaired_similarity = None

    if use_xlmr_repair:
        repairer = ExplicitUncertaintyXLMRRepairer()
        repaired_text, repair_report, block_report = repairer.repair_tokens(
            decoded_tokens,
            code_candidate_table=candidate_table,
            language_mode=repair_language_mode,
            confidence_threshold=repair_confidence_threshold,
            mlm_only_threshold=repair_mlm_only_threshold,
        )
        repaired_similarity = text_similarity(text, repaired_text)

    raw_bases = raw_utf8_dna_bases(text)

    return {
        "method": codec.name,
        "original_text": text,
        "encoded_dna": enc.dna,
        "noisy_dna": noisy_dna,
        "dna_preview": preview_seq(enc.dna, 500),
        "noisy_dna_preview": preview_seq(noisy_dna, 500),
        "decoded_text": dec.text,
        "repaired_text": repaired_text,
        "substitution_table": dna_substitution_table(enc.dna, noisy_dna, changed_positions),
        "code_candidate_table": candidate_table,
        "repair_report": repair_report,
        "block_report": block_report,
        "compression_ratio_vs_utf8_raw": len(enc.dna) / max(raw_bases, 1),
        "dna_reduction_percent": (1.0 - len(enc.dna) / max(raw_bases, 1)) * 100.0,
        "substituted_bases": n_subs,
        "actual_substitution_rate": n_subs / max(len(enc.dna), 1),
        "decoded_similarity": text_similarity(text, dec.text),
        "repaired_similarity": repaired_similarity,
        "repair_gain": None if repaired_similarity is None else repaired_similarity - text_similarity(text, dec.text),
        "invalid_code_count": dec.meta.get("invalid_code_count"),
        "exact_valid_code_count": dec.meta.get("exact_valid_code_count"),
        "metadata_json": json.dumps(enc.meta, ensure_ascii=False),
    }


def encode_text_to_sparse_dna_package(
    text: str,
    sparse_valid_vocab_size: int = 4096,
    sparse_code_bits: int = 13,
) -> Dict[str, Any]:
    """
    Production-style helper for Streamlit session export.
    The metadata contains the vocabulary; save it with the DNA if you want to decode later.
    """
    codec = SparseSemanticTokenTextCodec.from_text(
        text,
        valid_vocab_size=sparse_valid_vocab_size,
        code_bits=sparse_code_bits,
    )
    enc = codec.encode(text)
    return {
        "dna": enc.dna,
        "meta": enc.meta,
        "method": enc.method,
        "text_bytes": text.encode("utf-8"),
    }


def decode_text_from_sparse_dna_package(
    dna: str,
    meta: Dict[str, Any],
    use_xlmr_repair: bool = False,
    repair_language_mode: str = "auto",
    repair_confidence_threshold: float = 0.25,
) -> Dict[str, Any]:
    codec = SparseSemanticTokenTextCodec.from_meta(meta)
    dec = codec.decode(dna, meta)

    out = {
        "decoded_text": dec.text,
        "decoded_tokens": dec.meta.get("decoded_tokens", []),
        "code_candidate_table": dec.meta.get("code_candidate_table", pd.DataFrame()),
        "invalid_code_count": dec.meta.get("invalid_code_count"),
        "exact_valid_code_count": dec.meta.get("exact_valid_code_count"),
    }

    if use_xlmr_repair:
        repairer = ExplicitUncertaintyXLMRRepairer()
        repaired_text, repair_report, block_report = repairer.repair_tokens(
            out["decoded_tokens"],
            code_candidate_table=out["code_candidate_table"],
            language_mode=repair_language_mode,
            confidence_threshold=repair_confidence_threshold,
        )
        out.update({
            "repaired_text": repaired_text,
            "repair_report": repair_report,
            "block_report": block_report,
        })

    return out


# =============================================================================
# Streamlit panel
# =============================================================================

def _download_df_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    if st is None:
        return
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def _download_text_button(label: str, text: str, file_name: str) -> None:
    if st is None:
        return
    st.download_button(
        label,
        data=str(text or "").encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
    )


def render_text_dna_storage_panel() -> None:
    """
    Add this as a separate tab/page in your Streamlit DNA Data Storage app:

        from text_sparse_semantic_dna_streamlit import render_text_dna_storage_panel

        with st.tab("Text DNA Storage"):
            render_text_dna_storage_panel()
    """
    if st is None:
        raise RuntimeError("Streamlit is required to render the panel.")

    st.markdown("## Text DNA Storage — Sparse Semantic Token Coding")
    st.caption(
        "Conservative version: sparse token code + invalid-code/[UNK] repair only. "
        "No aggressive valid-token rewriting."
    )

    with st.container(border=True):
        st.markdown("### 1. Input text")

        uploaded = st.file_uploader(
            "Upload .txt file",
            type=["txt", "md", "csv", "json"],
            key="text_dna_upload",
        )
        default_text = st.session_state.get(
            "text_dna_input_text",
            "DNA storage is a promising technology for long-term digital preservation. "
            "Text data contains semantic structure, repeated words, and contextual redundancy. "
            "When DNA substitution errors occur, some token codes may become invalid. "
            "The decoded text remains openable, although some words may become [UNK]."
        )

        if uploaded is not None:
            try:
                default_text = uploaded.getvalue().decode("utf-8", errors="replace")
            except Exception:
                default_text = ""

        text = st.text_area(
            "Text content",
            value=default_text,
            height=220,
            key="text_dna_input_text_area",
        )
        st.session_state["text_dna_input_text"] = text

        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{len(text):,}")
        c2.metric("UTF-8 bytes", f"{len(text.encode('utf-8')):,}")
        c3.metric("UTF-8 DNA baseline", f"{raw_utf8_dna_bases(text):,} nt")

    with st.container(border=True):
        st.markdown("### 2. Settings")

        a, b, c = st.columns(3)
        dense_vocab_size = a.selectbox("Dense vocab size", [2048, 4096, 8192], index=2, key="text_dense_vocab_size")
        sparse_valid_vocab_size = b.selectbox("Sparse valid vocab", [1024, 2048, 4096], index=2, key="text_sparse_vocab_size")
        sparse_code_bits = c.selectbox("Sparse code bits", [12, 13, 14], index=1, key="text_sparse_code_bits")

        d, e, f = st.columns(3)
        seed = int(d.number_input("Random seed", min_value=1, max_value=999999, value=7, step=1, key="text_error_seed"))
        inspect_error_rate = float(e.number_input("Inspect substitution error", min_value=0.0, max_value=0.20, value=0.01, step=0.005, format="%.4f", key="text_inspect_error_rate"))
        language_mode = f.selectbox("Repair language", ["auto", "en", "vi", "fr", "zh"], index=0, key="text_repair_language")

        use_repair = st.checkbox(
            "Use XLM-R repair for [UNK]/invalid tokens only",
            value=False,
            key="text_use_xlmr_repair",
            help="Downloads xlm-roberta-base on first use. This does not rewrite valid words.",
        )
        confidence = st.slider("Repair confidence threshold", 0.05, 0.95, 0.25, 0.05, key="text_repair_confidence")

    with st.container(border=True):
        st.markdown("### 3. Benchmark")

        if st.button("Run text benchmark", key="run_text_dna_benchmark", use_container_width=True):
            df = run_text_dna_benchmark(
                text,
                error_rates=[0.01, 0.02, 0.05, 0.10],
                dense_vocab_size=int(dense_vocab_size),
                sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                sparse_code_bits=int(sparse_code_bits),
                seed=int(seed),
            )
            st.session_state["text_dna_benchmark_df"] = df

        df = st.session_state.get("text_dna_benchmark_df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            _download_df_button("Download benchmark CSV", df, "text_dna_benchmark.csv")

            try:
                chart_df = df[["method", "error_rate", "decoded_similarity"]].copy()
                st.line_chart(chart_df, x="error_rate", y="decoded_similarity", color="method")
            except Exception:
                pass

    with st.container(border=True):
        st.markdown("### 4. Inspect proposed sparse method")

        if st.button("Inspect sparse semantic token method", key="inspect_sparse_text_dna", use_container_width=True):
            with st.spinner("Running sparse token simulation" + (" + XLM-R repair..." if use_repair else "...")):
                result = inspect_sparse_text_dna(
                    text,
                    error_rate=float(inspect_error_rate),
                    sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                    sparse_code_bits=int(sparse_code_bits),
                    seed=int(seed),
                    use_xlmr_repair=bool(use_repair),
                    repair_language_mode=language_mode,
                    repair_confidence_threshold=float(confidence),
                    repair_mlm_only_threshold=float(confidence),
                )
            st.session_state["text_dna_sparse_result"] = result

        result = st.session_state.get("text_dna_sparse_result")
        if isinstance(result, dict):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("DNA length", f"{len(result.get('encoded_dna', '')):,} nt")
            m2.metric("DNA reduction", f"{float(result.get('dna_reduction_percent', 0)):.2f}%")
            m3.metric("Invalid codes", result.get("invalid_code_count", "—"))
            m4.metric("Substituted bases", result.get("substituted_bases", "—"))

            s1, s2, s3 = st.columns(3)
            s1.metric("Decoded similarity", f"{float(result.get('decoded_similarity', 0)):.4f}")
            if result.get("repaired_similarity") is not None:
                s2.metric("Repaired similarity", f"{float(result.get('repaired_similarity', 0)):.4f}")
                s3.metric("Repair gain", f"{float(result.get('repair_gain', 0)):+.4f}")
            else:
                s2.metric("Repaired similarity", "repair off")
                s3.metric("Repair gain", "—")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "DNA",
                "Decoded text",
                "Repair report",
                "Code candidates",
                "Downloads",
            ])

            with tab1:
                st.markdown("#### Encoded DNA")
                st.text_area("Encoded DNA preview", result.get("dna_preview", ""), height=120)
                st.markdown("#### Noisy DNA")
                st.text_area("Noisy DNA preview", result.get("noisy_dna_preview", ""), height=120)
                st.markdown("#### DNA substitutions")
                st.dataframe(result.get("substitution_table", pd.DataFrame()), use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("#### Decoded text")
                st.text_area("Decoded text", result.get("decoded_text", ""), height=220)
                if result.get("repaired_text") is not None:
                    st.markdown("#### Conservative XLM-R repaired text")
                    st.text_area("Repaired text", result.get("repaired_text", ""), height=220)
                else:
                    st.info("Repair is off. Enable XLM-R repair to produce repaired text.")

            with tab3:
                repair_df = result.get("repair_report", pd.DataFrame())
                if isinstance(repair_df, pd.DataFrame) and not repair_df.empty:
                    st.dataframe(repair_df, use_container_width=True, hide_index=True)
                    _download_df_button("Download repair report", repair_df, "text_dna_repair_report.csv")
                else:
                    st.info("No repair report available.")

                block_df = result.get("block_report", pd.DataFrame())
                if isinstance(block_df, pd.DataFrame) and not block_df.empty:
                    st.markdown("#### Block/language report")
                    st.dataframe(block_df, use_container_width=True, hide_index=True)

            with tab4:
                cand_df = result.get("code_candidate_table", pd.DataFrame())
                if isinstance(cand_df, pd.DataFrame) and not cand_df.empty:
                    st.dataframe(cand_df.head(500), use_container_width=True, hide_index=True)
                    _download_df_button("Download code candidate table", cand_df, "text_dna_code_candidates.csv")
                else:
                    st.info("No code candidate table available.")

            with tab5:
                _download_text_button("Download encoded DNA", result.get("encoded_dna", ""), "text_sparse_encoded_dna.txt")
                _download_text_button("Download noisy DNA", result.get("noisy_dna", ""), "text_sparse_noisy_dna.txt")
                _download_text_button("Download decoded text", result.get("decoded_text", ""), "text_sparse_decoded.txt")
                if result.get("repaired_text") is not None:
                    _download_text_button("Download repaired text", result.get("repaired_text", ""), "text_sparse_repaired.txt")
                _download_text_button("Download sparse codec metadata JSON", result.get("metadata_json", ""), "text_sparse_codec_meta.json")


# Alias for apps that prefer page naming.
render_text_dna_storage_page = render_text_dna_storage_panel



# =============================================================================
# Six-step Streamlit text pipeline UI override
# =============================================================================
# This section intentionally overrides the older render_text_dna_storage_panel()
# with a six-panel workflow matching the image pipeline:
#   1. Input
#   2. Text Compression
#   3. DNA Encoding
#   4. Strand Design
#   5. Text Reconstruction
#   6. Validation
#
# It also fixes:
#   _csv.Error: need to escape, but no escapechar set
# by exporting DataFrames using robust CSV quoting/escaping.
# =============================================================================


def _safe_dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Safe CSV exporter for DataFrames containing lists, dictionaries, quotes,
    backslashes, newlines, candidate tables, and model reports.

    Fixes pandas/csv errors such as:
        _csv.Error: need to escape, but no escapechar set
    """
    if df is None:
        df = pd.DataFrame()

    safe_df = df.copy()

    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list, tuple)) else x
        )

    return safe_df.to_csv(
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    ).encode("utf-8-sig")


def _download_df_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    if st is None:
        return
    st.download_button(
        label,
        data=_safe_dataframe_csv_bytes(df),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def _download_text_button(label: str, text: str, file_name: str) -> None:
    if st is None:
        return
    st.download_button(
        label,
        data=str(text or "").encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
    )


def _text_metric_row(text: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", f"{len(text):,}")
    c2.metric("UTF-8 bytes", f"{len(text.encode('utf-8')):,}")
    c3.metric("UTF-8 DNA baseline", f"{raw_utf8_dna_bases(text):,} nt")
    toks = simple_tokenize(text)
    c4.metric("Tokens", f"{len(toks):,}")


def _step_header(step_no: int, title: str, subtitle: str = "") -> None:
    st.markdown(f"### {step_no}. {title}")
    if subtitle:
        st.caption(subtitle)


def _text_pipeline_stepper(active_step: int = 1) -> None:
    steps = [
        (1, "Input"),
        (2, "Text Compression"),
        (3, "DNA Encoding"),
        (4, "Strand Design"),
        (5, "Text Reconstruction"),
        (6, "Validation"),
    ]

    html = ['<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 18px 0;">']
    for no, name in steps:
        if no < active_step:
            bg = "#E8F5E9"
            border = "#66BB6A"
            state = "Done"
        elif no == active_step:
            bg = "#E8F0FE"
            border = "#4285F4"
            state = "Current"
        else:
            bg = "#F7F7F7"
            border = "#DDDDDD"
            state = "Waiting"

        html.append(
            f"<div style='flex:1; min-width:155px; padding:12px 12px; "
            f"background:{bg}; border:1px solid {border}; border-radius:14px;'>"
            f"<div style='font-weight:700;'>{no}. {name}</div>"
            f"<div style='font-size:12px; opacity:0.75;'>{state}</div>"
            f"</div>"
        )

    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _get_text_from_input(uploaded: Any, pasted_text: str) -> str:
    if uploaded is not None:
        try:
            return uploaded.getvalue().decode("utf-8", errors="replace")
        except Exception:
            return ""
    return str(pasted_text or "")


def _encode_sparse_for_session(
    text: str,
    sparse_valid_vocab_size: int,
    sparse_code_bits: int,
) -> Dict[str, Any]:
    package = encode_text_to_sparse_dna_package(
        text,
        sparse_valid_vocab_size=int(sparse_valid_vocab_size),
        sparse_code_bits=int(sparse_code_bits),
    )
    dna = clean_dna(package["dna"])
    meta = package["meta"]
    return {
        "dna": dna,
        "meta": meta,
        "method": package["method"],
        "dna_bases": len(dna),
        "gc_content": gc_content(dna),
        "longest_homopolymer": longest_homopolymer(dna),
        "raw_utf8_dna_bases": raw_utf8_dna_bases(text),
        "compression_ratio_vs_utf8_raw": len(dna) / max(raw_utf8_dna_bases(text), 1),
        "dna_reduction_percent": (1.0 - len(dna) / max(raw_utf8_dna_bases(text), 1)) * 100.0,
    }


def _design_text_strands(
    dna: str,
    payload_len: int = 120,
    index_len: int = 8,
    forward_primer: str = "ACACGACGCTCATCCGATCT",
    reverse_primer: str = "AGATCGGAAGAGCACACGTCT",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple deterministic strand design for text DNA.

    This is not ECC. The index is only an ID for inspection/ordering.
    """
    dna = clean_dna(dna)
    payload_len = max(int(payload_len), 20)
    chunks = [dna[i:i + payload_len] for i in range(0, len(dna), payload_len)]

    def index_to_dna(idx: int, width: int) -> str:
        bits = int_to_bits(idx, width * 2)
        return bits_to_dna(bits)[:width]

    rows = []
    for i, payload in enumerate(chunks):
        idx_dna = index_to_dna(i, int(index_len))
        strand = clean_dna(forward_primer) + idx_dna + payload + clean_dna(reverse_primer)
        rows.append({
            "strand_id": i,
            "index_dna": idx_dna,
            "payload_len": len(payload),
            "strand_len": len(strand),
            "gc_content": gc_content(strand),
            "longest_homopolymer": longest_homopolymer(strand),
            "payload": payload,
            "full_strand": strand,
        })

    df = pd.DataFrame(rows)
    info = {
        "num_strands": len(df),
        "payload_len": payload_len,
        "index_len": index_len,
        "forward_primer": clean_dna(forward_primer),
        "reverse_primer": clean_dna(reverse_primer),
        "mean_strand_len": float(df["strand_len"].mean()) if len(df) else 0,
        "mean_gc": float(df["gc_content"].mean()) if len(df) else 0,
        "max_homopolymer": int(df["longest_homopolymer"].max()) if len(df) else 0,
    }
    return df, info


def render_text_dna_storage_panel() -> None:
    """
    Six-step Text DNA Storage panel.

    This replaces the older long single-page UI with a pipeline matching
    the image branch:
        1. Input
        2. Text Compression
        3. DNA Encoding
        4. Strand Design
        5. Text Reconstruction
        6. Validation
    """
    if st is None:
        raise RuntimeError("Streamlit is required to render the panel.")

    st.markdown("## Text DNA Storage")
    st.caption(
        "Sparse semantic token coding for compressed, openable, non-exact readable text recovery under DNA substitution errors."
    )

    active_step = int(st.session_state.get("text_active_step", 1))
    _text_pipeline_stepper(active_step)

    with st.container(border=True):
        _step_header(
            1,
            "Input",
            "Upload or paste text. This branch is for readable/semantic recovery, not exact archival recovery.",
        )

        uploaded = st.file_uploader(
            "Upload text file",
            type=["txt", "md", "csv", "json"],
            key="six_text_upload",
        )

        default_text = st.session_state.get(
            "six_text_input",
            "DNA storage is a promising technology for long-term digital preservation. "
            "Text data contains semantic structure, repeated words, and contextual redundancy. "
            "When DNA substitution errors occur, some token codes may become invalid. "
            "The decoded text remains openable, although some words may become [UNK]."
        )

        if uploaded is not None:
            default_text = _get_text_from_input(uploaded, "")

        text = st.text_area(
            "Input text",
            value=default_text,
            height=220,
            key="six_text_input_area",
        )
        st.session_state["six_text_input"] = text

        _text_metric_row(text)

        if st.button("Use this text", key="six_text_use_input", use_container_width=True):
            st.session_state["text_active_step"] = 2
            st.session_state["six_text_input"] = text
            st.rerun()

    with st.container(border=True):
        _step_header(
            2,
            "Text Compression",
            "Benchmark text representations: raw UTF-8, whole zlib, dense token coding, and sparse semantic token coding.",
        )

        a, b, c = st.columns(3)
        dense_vocab_size = a.selectbox(
            "Dense vocab size",
            [2048, 4096, 8192],
            index=2,
            key="six_dense_vocab_size",
        )
        sparse_valid_vocab_size = b.selectbox(
            "Sparse valid vocab",
            [1024, 2048, 4096],
            index=2,
            key="six_sparse_vocab_size",
        )
        sparse_code_bits = c.selectbox(
            "Sparse code bits",
            [12, 13, 14],
            index=1,
            key="six_sparse_code_bits",
        )

        d, e = st.columns(2)
        benchmark_errors_str = d.text_input(
            "Benchmark error rates",
            value="0,0.001,0.005,0.01,0.05,0.10",
            key="six_benchmark_errors",
        )
        seed = int(e.number_input(
            "Random seed",
            min_value=1,
            max_value=999999,
            value=7,
            step=1,
            key="six_text_seed",
        ))

        try:
            benchmark_errors = [float(x.strip()) for x in benchmark_errors_str.split(",") if x.strip()]
        except Exception:
            benchmark_errors = [0, 0.001, 0.005, 0.01, 0.05, 0.10]
            st.warning("Invalid error-rate list. Using default values.")

        if st.button("Run text compression benchmark", key="six_run_text_benchmark", use_container_width=True):
            with st.spinner("Running text benchmark..."):
                df = run_text_dna_benchmark(
                    text,
                    error_rates=benchmark_errors,
                    dense_vocab_size=int(dense_vocab_size),
                    sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                    sparse_code_bits=int(sparse_code_bits),
                    seed=int(seed),
                )
            st.session_state["six_text_benchmark_df"] = df
            st.session_state["text_active_step"] = 3
            st.rerun()

        df = st.session_state.get("six_text_benchmark_df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            show_cols = [
                col for col in [
                    "method",
                    "error_rate",
                    "dna_bases",
                    "compression_ratio_vs_utf8_raw",
                    "dna_reduction_percent",
                    "decode_ok",
                    "decoded_similarity",
                    "invalid_code_count",
                    "decoded_preview",
                ]
                if col in df.columns
            ]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
            _download_df_button("Download benchmark CSV", df, "text_dna_benchmark.csv")

            try:
                chart_df = df[["method", "error_rate", "decoded_similarity"]].copy()
                st.line_chart(chart_df, x="error_rate", y="decoded_similarity", color="method")
            except Exception:
                pass

    with st.container(border=True):
        _step_header(
            3,
            "DNA Encoding",
            "Encode the text with the proposed Sparse Semantic Text Coding method.",
        )

        text = st.session_state.get("six_text_input", "")

        if st.button("Encode text to sparse semantic DNA", key="six_encode_sparse_text", use_container_width=True):
            with st.spinner("Encoding text to DNA..."):
                enc_info = _encode_sparse_for_session(
                    text,
                    sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                    sparse_code_bits=int(sparse_code_bits),
                )
            st.session_state["six_text_encoded"] = enc_info
            st.session_state["text_active_step"] = 4
            st.rerun()

        enc_info = st.session_state.get("six_text_encoded")
        if isinstance(enc_info, dict):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("DNA length", f"{enc_info['dna_bases']:,} nt")
            m2.metric("DNA reduction", f"{enc_info['dna_reduction_percent']:.2f}%")
            m3.metric("GC content", f"{enc_info['gc_content']:.3f}")
            m4.metric("Longest homopolymer", enc_info["longest_homopolymer"])

            st.text_area("Encoded DNA preview", preview_seq(enc_info["dna"], 800), height=150)
            _download_text_button("Download encoded DNA", enc_info["dna"], "text_sparse_encoded_dna.txt")
            _download_text_button(
                "Download metadata JSON",
                json.dumps(enc_info["meta"], ensure_ascii=False, indent=2),
                "text_sparse_metadata.json",
            )

    with st.container(border=True):
        _step_header(
            4,
            "Strand Design",
            "Fragment the encoded DNA into indexed strands. This is not ECC; indices are for inspection/ordering.",
        )

        enc_info = st.session_state.get("six_text_encoded")
        if not isinstance(enc_info, dict):
            st.info("Encode DNA in step 3 first.")
        else:
            a, b, c, d = st.columns(4)
            payload_len = int(a.number_input("Payload bases/strand", min_value=40, max_value=300, value=120, step=10))
            index_len = int(b.number_input("Index length (bases)", min_value=4, max_value=20, value=8, step=1))
            fwd = c.text_input("Forward primer", value="ACACGACGCTCATCCGATCT")
            rev = d.text_input("Reverse primer", value="AGATCGGAAGAGCACACGTCT")

            if st.button("Design text DNA strands", key="six_design_text_strands", use_container_width=True):
                strand_df, strand_info = _design_text_strands(
                    enc_info["dna"],
                    payload_len=payload_len,
                    index_len=index_len,
                    forward_primer=fwd,
                    reverse_primer=rev,
                )
                st.session_state["six_text_strand_df"] = strand_df
                st.session_state["six_text_strand_info"] = strand_info
                st.session_state["text_active_step"] = 5
                st.rerun()

            strand_df = st.session_state.get("six_text_strand_df")
            strand_info = st.session_state.get("six_text_strand_info")
            if isinstance(strand_df, pd.DataFrame) and isinstance(strand_info, dict):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Number of strands", f"{strand_info['num_strands']:,}")
                s2.metric("Mean strand length", f"{strand_info['mean_strand_len']:.1f}")
                s3.metric("Mean GC", f"{strand_info['mean_gc']:.3f}")
                s4.metric("Max homopolymer", strand_info["max_homopolymer"])

                display_cols = [
                    "strand_id",
                    "index_dna",
                    "payload_len",
                    "strand_len",
                    "gc_content",
                    "longest_homopolymer",
                    "full_strand",
                ]
                st.dataframe(
                    strand_df[display_cols].head(200),
                    use_container_width=True,
                    hide_index=True,
                )
                _download_df_button("Download strand table CSV", strand_df, "text_dna_strands.csv")

    with st.container(border=True):
        _step_header(
            5,
            "Text Reconstruction",
            "Add DNA substitution errors, decode text, and optionally repair explicit [UNK]/invalid tokens only.",
        )

        enc_info = st.session_state.get("six_text_encoded")
        if not isinstance(enc_info, dict):
            st.info("Encode DNA in step 3 first.")
        else:
            a, b, c = st.columns(3)
            inspect_error_rate = float(a.number_input(
                "Substitution error rate",
                min_value=0.0,
                max_value=0.20,
                value=0.01,
                step=0.005,
                format="%.4f",
                key="six_inspect_error_rate",
            ))
            repair_language_mode = b.selectbox(
                "Repair language",
                ["auto", "en", "vi", "fr", "zh"],
                index=0,
                key="six_repair_language_mode",
            )
            use_repair = c.checkbox(
                "Use XLM-R repair",
                value=False,
                key="six_use_xlmr_repair",
                help="Repairs only [UNK]/invalid tokens. OFF by default because it downloads a large model.",
            )

            confidence = st.slider(
                "Repair confidence threshold",
                0.05,
                0.95,
                0.25,
                0.05,
                key="six_repair_confidence",
            )

            text = st.session_state.get("six_text_input", "")

            if st.button("Reconstruct text from noisy DNA", key="six_reconstruct_text", use_container_width=True):
                with st.spinner("Simulating DNA errors and decoding text..."):
                    result = inspect_sparse_text_dna(
                        text,
                        error_rate=float(inspect_error_rate),
                        sparse_valid_vocab_size=int(sparse_valid_vocab_size),
                        sparse_code_bits=int(sparse_code_bits),
                        seed=int(seed),
                        use_xlmr_repair=bool(use_repair),
                        repair_language_mode=repair_language_mode,
                        repair_confidence_threshold=float(confidence),
                        repair_mlm_only_threshold=float(confidence),
                    )
                st.session_state["six_text_reconstruction"] = result
                st.session_state["text_active_step"] = 6
                st.rerun()

            result = st.session_state.get("six_text_reconstruction")
            if isinstance(result, dict):
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Substituted bases", result.get("substituted_bases", "—"))
                r2.metric("Invalid codes", result.get("invalid_code_count", "—"))
                r3.metric("Decoded similarity", f"{float(result.get('decoded_similarity', 0)):.4f}")
                if result.get("repaired_similarity") is not None:
                    r4.metric("Repaired similarity", f"{float(result.get('repaired_similarity', 0)):.4f}")
                else:
                    r4.metric("Repaired similarity", "repair off")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Decoded text")
                    st.text_area("Decoded text", result.get("decoded_text", ""), height=240)
                with col2:
                    st.markdown("#### Repaired text")
                    if result.get("repaired_text") is not None:
                        st.text_area("Repaired text", result.get("repaired_text", ""), height=240)
                    else:
                        st.info("Repair is off.")

                _download_text_button("Download decoded text", result.get("decoded_text", ""), "text_sparse_decoded.txt")
                if result.get("repaired_text") is not None:
                    _download_text_button("Download repaired text", result.get("repaired_text", ""), "text_sparse_repaired.txt")

    with st.container(border=True):
        _step_header(
            6,
            "Validation",
            "Inspect substitutions, candidate tokens, repair reports, and downloadable validation artifacts.",
        )

        result = st.session_state.get("six_text_reconstruction")
        if not isinstance(result, dict):
            st.info("Run reconstruction in step 5 first.")
        else:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("DNA reduction", f"{float(result.get('dna_reduction_percent', 0)):.2f}%")
            v2.metric("Compression ratio", f"{float(result.get('compression_ratio_vs_utf8_raw', 0)):.4f}")
            v3.metric("Actual error rate", f"{float(result.get('actual_substitution_rate', 0)):.4f}")
            gain = result.get("repair_gain")
            v4.metric("Repair gain", "—" if gain is None else f"{float(gain):+.4f}")

            tab1, tab2, tab3, tab4 = st.tabs([
                "DNA substitutions",
                "Code candidates",
                "Repair report",
                "Downloads",
            ])

            with tab1:
                sub_df = result.get("substitution_table", pd.DataFrame())
                if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                    st.dataframe(sub_df, use_container_width=True, hide_index=True)
                    _download_df_button("Download substitution table", sub_df, "text_dna_substitutions.csv")
                else:
                    st.info("No substitution table available.")

            with tab2:
                cand_df = result.get("code_candidate_table", pd.DataFrame())
                if isinstance(cand_df, pd.DataFrame) and not cand_df.empty:
                    st.dataframe(cand_df.head(500), use_container_width=True, hide_index=True)
                    _download_df_button("Download code candidate table", cand_df, "text_dna_code_candidates.csv")
                else:
                    st.info("No candidate table available.")

            with tab3:
                repair_df = result.get("repair_report", pd.DataFrame())
                block_df = result.get("block_report", pd.DataFrame())
                if isinstance(repair_df, pd.DataFrame) and not repair_df.empty:
                    st.markdown("#### Repair report")
                    st.dataframe(repair_df, use_container_width=True, hide_index=True)
                    _download_df_button("Download repair report", repair_df, "text_dna_repair_report.csv")
                else:
                    st.info("No repair report available.")

                if isinstance(block_df, pd.DataFrame) and not block_df.empty:
                    st.markdown("#### Block/language report")
                    st.dataframe(block_df, use_container_width=True, hide_index=True)
                    _download_df_button("Download block report", block_df, "text_dna_block_report.csv")

            with tab4:
                _download_text_button("Download encoded DNA", result.get("encoded_dna", ""), "text_sparse_encoded_dna.txt")
                _download_text_button("Download noisy DNA", result.get("noisy_dna", ""), "text_sparse_noisy_dna.txt")
                _download_text_button("Download decoded text", result.get("decoded_text", ""), "text_sparse_decoded.txt")
                if result.get("repaired_text") is not None:
                    _download_text_button("Download repaired text", result.get("repaired_text", ""), "text_sparse_repaired.txt")
                _download_text_button("Download metadata JSON", result.get("metadata_json", ""), "text_sparse_codec_meta.json")


# Alias for apps that prefer page naming.
render_text_dna_storage_page = render_text_dna_storage_panel
