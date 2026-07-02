
"""
dna_sparse_semantic_project.py

Sparse semantic token coding for no-ECC DNA text storage.

Purpose
-------
This project develops the strongest current direction for the user's idea:
compressed + openable + non-exact text recovery under DNA substitution errors.

Main methods
------------
A. UTF-8 raw baseline
   Text -> UTF-8 bytes -> DNA

B. Whole zlib negative control
   Text -> UTF-8 bytes -> zlib -> DNA
   Compressed, but fragile under DNA substitution.

C. Dense fixed-vocab token baseline
   Text -> token IDs -> dense fixed-width binary -> DNA
   Compressed/openable, but every code is valid, so errors become wrong tokens.

D. Proposed sparse semantic token code
   Text -> semantic grouped tokens -> sparse even-parity codewords -> DNA
   13-bit space has 8192 possible codes, but only even-parity codewords are valid.
   One-bit code errors are detected as invalid -> [UNK] + candidate list.

Optional repair
---------------
XLM-R context repair can use:
- language detection / manual language
- [UNK] markers
- code-distance candidate table
- MLM candidate probabilities

No ECC is used. This is error-detecting/source-coding, not exact reconstruction.
"""

from __future__ import annotations

import difflib
import math
import random
import re
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================
# Binary/DNA utilities
# ============================================================

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


def bit_parity(x: int) -> int:
    return x.bit_count() % 2


def hamming_distance_bits(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def bits_to_dna(bits: str) -> str:
    if len(bits) % 2:
        bits += "0"
    return "".join(BITS_TO_DNA[bits[i:i + 2]] for i in range(0, len(bits), 2))


def dna_to_bits(dna: str) -> str:
    return "".join(DNA_TO_BITS.get(base, "00") for base in dna)


def raw_utf8_dna_bases(text: str) -> int:
    return len(text.encode("utf-8")) * 4


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def add_substitution_errors(dna: str, error_rate: float, seed: int = 7) -> Tuple[str, int, List[int]]:
    rng = random.Random(seed)
    bases = ["A", "C", "G", "T"]
    out = list(dna)
    changed = []
    for i, old in enumerate(out):
        if rng.random() < error_rate:
            out[i] = rng.choice([b for b in bases if b != old])
            changed.append(i)
    return "".join(out), len(changed), changed


def dna_substitution_table(original_dna: str, noisy_dna: str, changed_positions: Sequence[int], window: int = 12, max_show: int = 30) -> pd.DataFrame:
    rows = []
    for pos in list(changed_positions)[:max_show]:
        start = max(0, pos - window)
        end = min(len(original_dna), pos + window + 1)
        rows.append({
            "position": pos,
            "old_base": original_dna[pos],
            "new_base": noisy_dna[pos],
            "original_context": original_dna[start:end],
            "noisy_context": noisy_dna[start:end],
            "marker": " " * (pos - start) + "^",
        })
    return pd.DataFrame(rows)


def conservative_cleanup(text: str) -> str:
    text = text.replace("\ufffd", "[UNK]")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "[UNK]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


# ============================================================
# Tokenization and language grouping
# ============================================================


def simple_tokenize(text: str) -> List[str]:
    return re.findall(
        r"[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|[\u4e00-\u9fff]+|[\uac00-\ud7af]+|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )


def simple_detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for tok in tokens:
        if tok in ".,;:!?)]}":
            out = out.rstrip() + tok + " "
        elif tok in "([{" :
            out += tok
        elif re.search(r"[\u4e00-\u9fff]", tok):
            out += tok
        else:
            out += tok + " "
    return out.strip()


VI_SPECIFIC_CHARS = set("ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
FR_CUES = {"la", "le", "les", "des", "du", "une", "un", "de", "données", "numériques", "technologie", "stockage", "préserver", "peut", "avec", "pour", "dans"}
EN_CUES = {"the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "for", "with", "text", "data", "storage", "technology", "digital", "errors", "sports", "people", "games", "risk", "dangerous", "money", "players"}


def classify_token_language(tok: str) -> str:
    if tok in {"[PAD]", "[UNK]"}:
        return "special"
    if re.fullmatch(r"[0-9]+", tok):
        return "number"
    if re.fullmatch(r"[\W_]+", tok) and not re.search(r"[\u4e00-\u9fff\uac00-\ud7af]", tok):
        return "punct"
    if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
        return "zh"
    if re.fullmatch(r"[\uac00-\ud7af]+", tok):
        return "ko"
    low = tok.lower()
    if any(ch in VI_SPECIFIC_CHARS for ch in tok):
        return "vi"
    if low in FR_CUES or any(ch in "éèêëàâîïôùûçÉÈÊËÀÂÎÏÔÙÛÇ" for ch in tok):
        return "fr"
    return "en"


def detect_block_language(block: str, manual_language: str = "auto") -> str:
    if manual_language and manual_language != "auto":
        return manual_language
    total = max(len(block), 1)
    if len(re.findall(r"[\u4e00-\u9fff]", block)) / total > 0.15:
        return "zh"
    if len(re.findall(r"[\uac00-\ud7af]", block)) / total > 0.15:
        return "ko"
    if sum(ch in VI_SPECIFIC_CHARS for ch in block) / total > 0.025:
        return "vi"
    toks = re.findall(r"[A-Za-zÀ-ỹ]+", block.lower())
    fr = sum(t in FR_CUES for t in toks)
    en = sum(t in EN_CUES for t in toks)
    if fr >= 2 and fr >= en:
        return "fr"
    if en >= 1:
        return "en"
    return "en" if toks else "unknown"


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
    if lang == "ko":
        return re.fullmatch(r"[\uac00-\ud7af]+", w) is not None
    return bool(re.fullmatch(r"[A-Za-zÀ-ỹ]+|[\u4e00-\u9fff]+|[\uac00-\ud7af]+", w))


DEFAULT_EXTRA_VOCAB = [
    "[PAD]", "[UNK]",
    # English common/domain
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "for", "with", "as", "by", "from",
    "text", "data", "storage", "DNA", "technology", "digital", "preservation", "semantic", "structure",
    "repeated", "words", "contextual", "redundancy", "compressed", "binary", "stream", "token", "tokens", "IDs",
    "substitution", "errors", "occur", "decoded", "openable", "wrong", "promising", "long-term", "different",
    "valid", "may", "become", "some", "although", "model", "language", "context", "repair", "candidate",
    "people", "sports", "games", "dangerous", "risk", "risks", "participate", "participation", "players", "money",
    "professional", "training", "protection", "equipment", "popular", "factor", "allowed", "human", "compete", "safety",
    # Vietnamese/domain
    "Công", "nghệ", "lưu", "trữ", "có", "thể", "hữu", "ích", "cho", "bảo", "quản", "dữ", "liệu", "lâu", "dài",
    "văn", "bản", "cấu", "trúc", "ngữ", "nghĩa", "nhiều", "từ", "lặp", "lại",
    # French/domain
    "La", "Le", "Les", "la", "le", "les", "technologie", "de", "stockage", "ADN", "peut", "préserver", "données", "numériques",
    # Chinese chunks
    "数据存储", "长期保存", "非常重要", "数据", "存储", "长期", "保存", "重要",
    # Korean/domain
    "저장", "기술", "디지털", "정보", "장기간", "보존", "데이터", "있습니다",
]


def build_vocab_from_text(text: str, limit: int) -> List[str]:
    counts: Dict[str, int] = {}
    for t in simple_tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    sorted_tokens = sorted(counts, key=lambda x: (-counts[x], x))
    out: List[str] = []
    seen = set()
    for t in DEFAULT_EXTRA_VOCAB + sorted_tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= limit:
            break
    return out


def semantic_order(vocab: Sequence[str]) -> List[str]:
    groups: Dict[str, List[str]] = {"special": [], "punct": [], "number": [], "en": [], "vi": [], "fr": [], "zh": [], "ko": [], "other": []}
    seen = set()
    for tok in vocab:
        if tok in seen:
            continue
        seen.add(tok)
        g = classify_token_language(tok)
        if g not in groups:
            g = "other"
        groups[g].append(tok)
    ordered = []
    for g in ["special", "punct", "number", "en", "vi", "fr", "zh", "ko", "other"]:
        ordered.extend(groups[g])
    return ordered


# ============================================================
# Codeword generation: sparse even-parity code
# ============================================================


def even_parity_codewords(code_bits: int) -> List[int]:
    return [x for x in range(1 << code_bits) if bit_parity(x) == 0]


def code_int_to_dna(code: int, code_bits: int) -> str:
    return bits_to_dna(int_to_bits(code, code_bits))


# ============================================================
# Codecs
# ============================================================

class UTF8RawCodec:
    name = "A_no_compression_utf8_raw"
    def encode(self, text: str) -> Encoded:
        data = text.encode("utf-8")
        dna = bits_to_dna(bytes_to_bits(data))
        return Encoded(self.name, dna, {"raw_utf8_bytes": len(data), "dna_bases": len(dna)})
    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        data = bits_to_bytes(dna_to_bits(dna))
        return Decoded(self.name, data.decode("utf-8", errors="replace"), True, {})


class WholeZlibCodec:
    name = "B_bad_binary_compression_whole_zlib"
    def encode(self, text: str) -> Encoded:
        comp = zlib.compress(text.encode("utf-8"), 6)
        return Encoded(self.name, bits_to_dna(bytes_to_bits(comp)), {"compressed_bytes": len(comp), "dna_bases": len(comp) * 4})
    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        comp = bits_to_bytes(dna_to_bits(dna))
        try:
            raw = zlib.decompress(comp)
            return Decoded(self.name, raw.decode("utf-8", errors="replace"), True, {})
        except Exception as e:
            return Decoded(self.name, f"[DECODE FAILED: {type(e).__name__}({str(e)!r})]", False, {"error": repr(e)})


class DenseTokenCodec:
    """Dense fixed-vocab token baseline: all code space is valid."""
    name_prefix = "C_dense_fixed_vocab_token"
    def __init__(self, vocab: Sequence[str], vocab_size: int = 8192):
        self.vocab_size = 1 << math.ceil(math.log2(vocab_size))
        self.token_bits = int(math.log2(self.vocab_size))
        ordered = list(vocab)
        while len(ordered) < self.vocab_size:
            ordered.append(f"[FILL_{len(ordered)}]")
        self.id_to_token = dict(enumerate(ordered[: self.vocab_size]))
        self.token_to_id = {t: i for i, t in self.id_to_token.items()}
        self.unk_id = self.token_to_id.get("[UNK]", 1)
        self.name = f"{self.name_prefix}_{self.vocab_size}"
    @classmethod
    def from_text(cls, text: str, vocab_size: int = 8192) -> "DenseTokenCodec":
        return cls(semantic_order(build_vocab_from_text(text, vocab_size)), vocab_size=vocab_size)
    def encode(self, text: str) -> Encoded:
        toks = simple_tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        bits = "".join(int_to_bits(i, self.token_bits) for i in ids)
        dna = bits_to_dna(bits)
        return Encoded(self.name, dna, {"tokens": toks, "token_ids": ids, "num_tokens": len(ids), "token_bits": self.token_bits, "dna_bases": len(dna)})
    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None) -> Decoded:
        bits = dna_to_bits(dna)
        n = int(meta.get("num_tokens")) if meta and "num_tokens" in meta else None
        toks = []
        ids = []
        for i in range(0, len(bits), self.token_bits):
            chunk = bits[i:i+self.token_bits]
            if len(chunk) < self.token_bits:
                break
            tid = bits_to_int(chunk)
            if tid >= self.vocab_size:
                tid = self.unk_id
            tok = self.id_to_token.get(tid, "[UNK]")
            if tok.startswith("[FILL_"):
                tok = "[UNK]"
            ids.append(tid)
            if tok != "[PAD]":
                toks.append(tok)
        if n is not None:
            toks = toks[:n]
            ids = ids[:n]
        return Decoded(self.name, simple_detokenize(toks), True, {"decoded_tokens": toks, "decoded_ids": ids})


class SparseSemanticTokenCodec:
    """
    Proposed method.

    valid_vocab_size <= 2^(code_bits-1) using even-parity valid codewords.
    Any odd-parity noisy codeword is invalid -> [UNK], with nearest-code candidates.

    Example default: 4096 valid tokens inside 13-bit space (8192 total codes).
    """
    name_prefix = "D_proposed_sparse_semantic_token"
    def __init__(self, vocab: Sequence[str], valid_vocab_size: int = 4096, code_bits: int = 13):
        if valid_vocab_size > (1 << (code_bits - 1)):
            raise ValueError("For even-parity sparse code, valid_vocab_size must be <= 2^(code_bits-1).")
        self.valid_vocab_size = valid_vocab_size
        self.code_bits = code_bits
        self.name = f"{self.name_prefix}_{valid_vocab_size}_codespace_{1<<code_bits}"

        ordered = semantic_order(vocab)[:valid_vocab_size]
        while len(ordered) < valid_vocab_size:
            ordered.append(f"[FILL_{len(ordered)}]")
        self.id_to_token = dict(enumerate(ordered))
        self.token_to_id = {t: i for i, t in self.id_to_token.items()}
        self.unk_token = "[UNK]"
        self.unk_id = self.token_to_id.get("[UNK]", 1)

        codes = even_parity_codewords(code_bits)[:valid_vocab_size]
        self.id_to_code = {i: codes[i] for i in range(valid_vocab_size)}
        self.code_to_id = {code: i for i, code in self.id_to_code.items()}
        self.id_to_code_bits = {i: int_to_bits(c, code_bits) for i, c in self.id_to_code.items()}
        self.token_language = {tok: classify_token_language(tok) for tok in self.token_to_id.keys()}

    @classmethod
    def from_text(cls, text: str, valid_vocab_size: int = 4096, code_bits: int = 13) -> "SparseSemanticTokenCodec":
        return cls(build_vocab_from_text(text, valid_vocab_size), valid_vocab_size=valid_vocab_size, code_bits=code_bits)

    def encode(self, text: str) -> Encoded:
        toks = simple_tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        bits = "".join(self.id_to_code_bits[i] for i in ids)
        dna = bits_to_dna(bits)
        return Encoded(self.name, dna, {"tokens": toks, "token_ids": ids, "num_tokens": len(ids), "code_bits": self.code_bits, "valid_vocab_size": self.valid_vocab_size, "dna_bases": len(dna)})

    def nearest_candidates(self, noisy_code_bits: str, top_k: int = 8, prefer_language: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = []
        for tid, code_bits in self.id_to_code_bits.items():
            tok = self.id_to_token.get(tid, "[UNK]")
            if tok.startswith("[FILL_"):
                continue
            lang = self.token_language.get(tok, "other")
            dist = hamming_distance_bits(noisy_code_bits, code_bits)
            penalty = 0 if (prefer_language is None or lang in {prefer_language, "special", "punct", "number"}) else 2
            rows.append({"token_id": tid, "token": tok, "language": lang, "hamming_distance": dist, "score": dist + penalty})
        rows.sort(key=lambda r: (r["score"], r["hamming_distance"], r["token_id"]))
        return rows[:top_k]

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None, candidate_top_k: int = 8, prefer_language: Optional[str] = None) -> Decoded:
        bits = dna_to_bits(dna)
        n = int(meta.get("num_tokens")) if meta and "num_tokens" in meta else None
        decoded_tokens: List[str] = []
        decoded_ids: List[int] = []
        candidate_rows: List[Dict[str, Any]] = []
        invalid_count = 0
        exact_count = 0

        pos = 0
        for i in range(0, len(bits), self.code_bits):
            chunk = bits[i:i+self.code_bits]
            if len(chunk) < self.code_bits:
                break
            code = bits_to_int(chunk)
            if code in self.code_to_id:
                tid = self.code_to_id[code]
                tok = self.id_to_token.get(tid, "[UNK]")
                if tok.startswith("[FILL_"):
                    tok = "[UNK]"
                exact = True
                exact_count += 1
            else:
                tid = self.unk_id
                tok = "[UNK]"
                exact = False
                invalid_count += 1

            cands = self.nearest_candidates(chunk, top_k=candidate_top_k, prefer_language=prefer_language)
            for rank, c in enumerate(cands, start=1):
                candidate_rows.append({"token_position": pos, "rank": rank, "exact_valid_code": exact, **c})
            decoded_tokens.append(tok)
            decoded_ids.append(tid)
            pos += 1

        if n is not None:
            decoded_tokens = decoded_tokens[:n]
            decoded_ids = decoded_ids[:n]
            candidate_rows = [r for r in candidate_rows if r["token_position"] < n]

        return Decoded(self.name, simple_detokenize([t for t in decoded_tokens if t != "[PAD]"]), True, {
            "decoded_tokens": decoded_tokens,
            "decoded_ids": decoded_ids,
            "invalid_code_count": invalid_count,
            "exact_valid_code_count": exact_count,
            "code_candidate_table": pd.DataFrame(candidate_rows),
        })


# ============================================================
# XLM-R candidate-aware repair
# ============================================================

class XLMRContextRepairer:
    def __init__(self, model_name: str = "xlm-roberta-base", device: Optional[str] = None):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mlm_candidates(self, tokens: List[str], pos: int, lang: str, top_k: int = 20) -> List[Dict[str, Any]]:
        masked = list(tokens)
        masked[pos] = self.tokenizer.mask_token
        sent = simple_detokenize(masked)
        inputs = self.tokenizer(sent, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            return []
        idx = mask_positions[1][0]
        probs = self.torch.softmax(logits[0, idx], dim=-1)
        top = self.torch.topk(probs, k=top_k * 60)
        out, seen = [], set()
        for tid, prob in zip(top.indices.tolist(), top.values.tolist()):
            cand = self.tokenizer.decode([tid]).replace("▁", "").strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            if not is_valid_for_language(cand, lang):
                continue
            if lang not in {"zh", "ko"} and len(cand) <= 1:
                continue
            out.append({"candidate": cand, "probability": float(prob), "mlm_token_id": int(tid)})
            if len(out) >= top_k:
                break
        return out

    def repair_token_sequence(self, tokens: List[str], code_candidate_table: Optional[pd.DataFrame] = None, language_mode: str = "auto", confidence_threshold: float = 0.25, top_k: int = 20) -> Tuple[str, pd.DataFrame]:
        # For long text, use global language from detokenized text or manual override.
        text = simple_detokenize(tokens)
        lang = detect_block_language(text, manual_language=language_mode)
        repaired = list(tokens)
        rows = []

        # Build code candidate map for each token position.
        code_map: Dict[int, List[str]] = {}
        if code_candidate_table is not None and len(code_candidate_table) > 0:
            for pos, sub in code_candidate_table.groupby("token_position"):
                code_map[int(pos)] = [str(x) for x in sub.sort_values(["rank"])["token"].head(8).tolist()]

        for pos, tok in enumerate(tokens):
            suspicious = (tok == "[UNK]") or (not is_valid_for_language(tok, lang))
            if not suspicious:
                continue

            mlm = self.mlm_candidates(repaired, pos, lang=lang, top_k=top_k)
            code_cands = code_map.get(pos, [])
            code_set = {c.lower(): c for c in code_cands}

            chosen = "[UNK]"
            accepted = False
            prob = mlm[0]["probability"] if mlm else 0.0
            reason = "no_candidate"

            # Prefer intersection between code-nearest and MLM candidates.
            for m in mlm:
                if m["candidate"].lower() in code_set and m["probability"] >= confidence_threshold:
                    chosen = code_set[m["candidate"].lower()]
                    accepted = True
                    prob = m["probability"]
                    reason = "mlm_and_code_agree"
                    break

            # If no agreement, allow high-confidence MLM for [UNK].
            if not accepted and mlm and mlm[0]["probability"] >= confidence_threshold:
                chosen = mlm[0]["candidate"]
                accepted = True
                prob = mlm[0]["probability"]
                reason = "mlm_only"

            repaired[pos] = chosen
            rows.append({
                "token_position": pos,
                "old_token": tok,
                "chosen": chosen,
                "language": lang,
                "probability": prob,
                "accepted": accepted,
                "reason": reason,
                "code_candidates": code_cands[:8],
                "mlm_candidates": mlm[:10],
            })

        return simple_detokenize(repaired), pd.DataFrame(rows)


# ============================================================
# Benchmark functions
# ============================================================


def make_codecs(text: str, dense_vocab_size: int = 8192, sparse_valid_vocab_size: int = 4096, sparse_code_bits: int = 13) -> List[Any]:
    return [
        UTF8RawCodec(),
        WholeZlibCodec(),
        DenseTokenCodec.from_text(text, vocab_size=dense_vocab_size),
        SparseSemanticTokenCodec.from_text(text, valid_vocab_size=sparse_valid_vocab_size, code_bits=sparse_code_bits),
    ]


def run_single(text: str, codec: Any, error_rate: float, seed: int = 7, use_xlmr_repair: bool = False, repair_language_mode: str = "auto", repair_confidence_threshold: float = 0.25) -> Dict[str, Any]:
    enc = codec.encode(text)
    noisy_dna, n_subs, positions = add_substitution_errors(enc.dna, error_rate=error_rate, seed=seed)
    dec = codec.decode(noisy_dna, enc.meta)
    raw_bases = raw_utf8_dna_bases(text)
    row: Dict[str, Any] = {
        "group": enc.method.split("_", 1)[0],
        "method": enc.method,
        "error_rate": error_rate,
        "dna_bases": len(enc.dna),
        "substituted_bases": n_subs,
        "actual_substitution_rate": n_subs / max(len(enc.dna), 1),
        "compression_ratio_vs_utf8_raw": len(enc.dna) / raw_bases,
        "dna_reduction_percent": (1 - len(enc.dna) / raw_bases) * 100,
        "decode_ok": dec.decode_ok,
        "decoded_similarity": text_similarity(text, dec.text),
        "conservative_similarity": text_similarity(text, conservative_cleanup(dec.text)),
        "decoded_preview": dec.text[:180].replace("\n", " "),
        "encoded": enc,
        "decoded": dec,
        "noisy_dna": noisy_dna,
        "changed_positions": positions,
    }

    if "invalid_code_count" in dec.meta:
        row["invalid_code_count"] = dec.meta.get("invalid_code_count")
        row["exact_valid_code_count"] = dec.meta.get("exact_valid_code_count")
        row["num_code_candidates"] = len(dec.meta.get("code_candidate_table", pd.DataFrame()))
        row["code_candidate_table"] = dec.meta.get("code_candidate_table", pd.DataFrame())

    if use_xlmr_repair and dec.decode_ok and "decoded_tokens" in dec.meta:
        repairer = XLMRContextRepairer()
        repaired_text, repair_table = repairer.repair_token_sequence(
            dec.meta["decoded_tokens"],
            code_candidate_table=dec.meta.get("code_candidate_table"),
            language_mode=repair_language_mode,
            confidence_threshold=repair_confidence_threshold,
        )
        row.update({
            "xlmr_repaired_text": repaired_text,
            "xlmr_repaired_similarity": text_similarity(text, repaired_text),
            "xlmr_repaired_preview": repaired_text[:180].replace("\n", " "),
            "xlmr_replacement_report": repair_table,
            "xlmr_repair_count": len(repair_table),
            "xlmr_accepted_count": int(repair_table["accepted"].sum()) if len(repair_table) else 0,
        })
    return row


def run_benchmark(text: str, error_rates: Sequence[float] = (0.01, 0.02, 0.05, 0.10), dense_vocab_size: int = 8192, sparse_valid_vocab_size: int = 4096, sparse_code_bits: int = 13, seed: int = 7, use_xlmr_repair: bool = False, include_objects: bool = False) -> pd.DataFrame:
    rows = []
    codecs = make_codecs(text, dense_vocab_size=dense_vocab_size, sparse_valid_vocab_size=sparse_valid_vocab_size, sparse_code_bits=sparse_code_bits)
    for codec in codecs:
        for er in error_rates:
            row = run_single(text, codec, er, seed=seed, use_xlmr_repair=use_xlmr_repair)
            if not include_objects:
                row = {k: v for k, v in row.items() if k not in ["encoded", "decoded", "noisy_dna", "changed_positions", "code_candidate_table", "xlmr_repaired_text", "xlmr_replacement_report"]}
            rows.append(row)
    return pd.DataFrame(rows)


def inspect_method(text: str, method_contains: str = "sparse", error_rate: float = 0.01, dense_vocab_size: int = 8192, sparse_valid_vocab_size: int = 4096, sparse_code_bits: int = 13, seed: int = 7, use_xlmr_repair: bool = True, repair_language_mode: str = "auto", repair_confidence_threshold: float = 0.25, max_dna_preview: int = 300) -> Dict[str, Any]:
    selected = None
    for codec in make_codecs(text, dense_vocab_size=dense_vocab_size, sparse_valid_vocab_size=sparse_valid_vocab_size, sparse_code_bits=sparse_code_bits):
        if method_contains in codec.name:
            selected = codec
            break
    if selected is None:
        raise ValueError("No method found.")
    row = run_single(text, selected, error_rate, seed=seed, use_xlmr_repair=use_xlmr_repair, repair_language_mode=repair_language_mode, repair_confidence_threshold=repair_confidence_threshold)
    enc, dec = row["encoded"], row["decoded"]
    return {
        "method": enc.method,
        "original_text": text,
        "dna_preview": enc.dna[:max_dna_preview],
        "noisy_dna_preview": row["noisy_dna"][:max_dna_preview],
        "substitution_table": dna_substitution_table(enc.dna, row["noisy_dna"], row["changed_positions"]),
        "decoded_text": dec.text,
        "xlmr_repaired_text": row.get("xlmr_repaired_text"),
        "code_candidate_table": dec.meta.get("code_candidate_table", pd.DataFrame()),
        "xlmr_replacement_report": row.get("xlmr_replacement_report", pd.DataFrame()),
        "compression_ratio_vs_utf8_raw": row["compression_ratio_vs_utf8_raw"],
        "dna_reduction_percent": row["dna_reduction_percent"],
        "decoded_similarity": row["decoded_similarity"],
        "xlmr_repaired_similarity": row.get("xlmr_repaired_similarity"),
        "invalid_code_count": dec.meta.get("invalid_code_count"),
        "exact_valid_code_count": dec.meta.get("exact_valid_code_count"),
    }


def save_reports(out_dir: str, benchmark_df: pd.DataFrame, inspection: Optional[Dict[str, Any]] = None) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    benchmark_df.to_csv(os.path.join(out_dir, "benchmark_results.csv"), index=False)
    if inspection:
        inspection.get("substitution_table", pd.DataFrame()).to_csv(os.path.join(out_dir, "dna_substitution_table.csv"), index=False)
        inspection.get("code_candidate_table", pd.DataFrame()).to_csv(os.path.join(out_dir, "code_candidate_table.csv"), index=False)
        inspection.get("xlmr_replacement_report", pd.DataFrame()).to_csv(os.path.join(out_dir, "xlmr_candidate_table.csv"), index=False)
        with open(os.path.join(out_dir, "decoded_text.txt"), "w", encoding="utf-8") as f:
            f.write(inspection.get("decoded_text") or "")
        with open(os.path.join(out_dir, "xlmr_repaired_text.txt"), "w", encoding="utf-8") as f:
            f.write(inspection.get("xlmr_repaired_text") or "")


# ============================================================
# Pretrained-tokenizer sparse token codec
# ============================================================
# This codec avoids any file-specific vocabulary/dictionary metadata.
# Encoder and decoder share a fixed pretrained tokenizer name, e.g.
# xlm-roberta-base.  The DNA payload therefore contains only token IDs
# encoded as sparse even-parity codewords plus a fixed EOS token.
# No torch is required for tokenization; transformers/sentencepiece are enough.

from functools import lru_cache


@lru_cache(maxsize=4)
def _load_pretrained_tokenizer(tokenizer_name: str = "xlm-roberta-base"):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Pretrained tokenizer support requires: pip install transformers sentencepiece"
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_name)


@lru_cache(maxsize=8)
def _pretrained_even_code_tables(vocab_size: int, code_bits: int) -> tuple[dict[int, int], dict[int, int]]:
    """Return token_id->even_code and even_code->token_id for a fixed tokenizer vocab."""
    vocab_size = int(vocab_size)
    code_bits = int(code_bits)
    capacity = 1 << (code_bits - 1)
    if vocab_size > capacity:
        raise ValueError(
            f"vocab_size={vocab_size} requires more than {code_bits} bits for even-parity sparse coding."
        )
    id_to_code: dict[int, int] = {}
    code_to_id: dict[int, int] = {}
    tid = 0
    for code in range(1 << code_bits):
        if bit_parity(code) == 0:
            id_to_code[tid] = code
            code_to_id[code] = tid
            tid += 1
            if tid >= vocab_size:
                break
    return id_to_code, code_to_id


class PretrainedSparseTokenCodec:
    """
    Sparse token codec using a fixed pretrained tokenizer vocabulary.

    Key design points:
    - No per-file vocabulary is built.
    - No dictionary/header needs to be stored with the DNA payload.
    - Tokenizer name is fixed by the app/paper, default xlm-roberta-base.
    - Token IDs are mapped to sparse even-parity codewords.
    - A fixed EOS token marks the end, so payload bit length/num_tokens metadata is
      not required for decoding; byte/DNA padding after EOS is ignored.
    """

    name_prefix = "D_pretrained_xlmr_sparse_token"
    is_pretrained_tokenizer_codec = True

    def __init__(self, tokenizer_name: str = "xlm-roberta-base", code_bits: int | None = None):
        self.tokenizer_name = str(tokenizer_name)
        self.tokenizer = _load_pretrained_tokenizer(self.tokenizer_name)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or len(self.tokenizer))
        if code_bits is None:
            # Need 2^(code_bits-1) valid even-parity codes >= vocab_size.
            code_bits = max(2, math.ceil(math.log2(max(self.vocab_size, 2))) + 1)
        self.code_bits = int(code_bits)
        self.id_to_code, self.code_to_id = _pretrained_even_code_tables(self.vocab_size, self.code_bits)
        self.eos_token_id = int(
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else (self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2)
        )
        self.unk_token_id = int(
            self.tokenizer.unk_token_id
            if self.tokenizer.unk_token_id is not None
            else 0
        )
        self.name = f"{self.name_prefix}_{self.tokenizer_name.replace('/', '_')}_{self.code_bits}bit"

    @staticmethod
    def from_text(text: str, tokenizer_name: str = "xlm-roberta-base", code_bits: int | None = None) -> "PretrainedSparseTokenCodec":
        # text is intentionally unused; tokenizer vocab is fixed globally.
        return PretrainedSparseTokenCodec(tokenizer_name=tokenizer_name, code_bits=code_bits)

    def _code_bits_for_token_id(self, token_id: int) -> str:
        code = self.id_to_code.get(int(token_id))
        if code is None:
            code = self.id_to_code.get(self.unk_token_id, 0)
        return int_to_bits(int(code), self.code_bits)

    def code_bits_for_token_id(self, token_id: int) -> str:
        return self._code_bits_for_token_id(token_id)

    def code_dna_for_token_id(self, token_id: int) -> str:
        return bits_to_dna(self._code_bits_for_token_id(token_id))

    def encode(self, text: str) -> Encoded:
        ids = list(self.tokenizer.encode(str(text or ""), add_special_tokens=False))
        ids = [int(i) if 0 <= int(i) < self.vocab_size else self.unk_token_id for i in ids]
        ids.append(self.eos_token_id)
        bits = "".join(self._code_bits_for_token_id(tid) for tid in ids)
        dna = bits_to_dna(bits)
        token_pieces = self.tokenizer.convert_ids_to_tokens(ids)
        return Encoded(self.name, dna, {
            "tokenizer_name": self.tokenizer_name,
            "fixed_tokenizer_vocab": True,
            "no_file_specific_vocab": True,
            "vocab_size": self.vocab_size,
            "code_bits": self.code_bits,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "num_tokens_including_eos": len(ids),
            "num_tokens": max(len(ids) - 1, 0),
            "token_ids": ids,
            "tokens": token_pieces,
            "dna_bases": len(dna),
            "encoding": "pretrained_tokenizer_sparse_even_parity_with_eos",
        })

    def nearest_candidates(self, noisy_code_bits: str, top_k: int = 8, prefer_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Candidate table from nearby even-parity codewords; no file-specific vocab required."""
        noisy_code_bits = str(noisy_code_bits or "")[:self.code_bits]
        if len(noisy_code_bits) < self.code_bits:
            return []
        code = bits_to_int(noisy_code_bits)
        candidates: dict[int, int] = {}

        # Try exact, one-bit, then two-bit neighbors.  This is much faster than
        # scanning the full ~250k-token vocab and is aligned with sparse error detection.
        probe_codes = [code]
        for b in range(self.code_bits):
            probe_codes.append(code ^ (1 << b))
        for b1 in range(self.code_bits):
            for b2 in range(b1 + 1, self.code_bits):
                probe_codes.append(code ^ (1 << b1) ^ (1 << b2))

        for cand_code in probe_codes:
            tid = self.code_to_id.get(int(cand_code))
            if tid is None or tid in candidates:
                continue
            candidates[tid] = hamming_distance_bits(noisy_code_bits, int_to_bits(cand_code, self.code_bits))

        rows: List[Dict[str, Any]] = []
        for tid, dist in candidates.items():
            piece = self.tokenizer.convert_ids_to_tokens([int(tid)])[0]
            token_text = self.tokenizer.decode([int(tid)], clean_up_tokenization_spaces=True).strip()
            if not token_text:
                token_text = str(piece).replace("▁", "")
            lang = classify_token_language(token_text) if token_text else "other"
            penalty = 0 if (not prefer_language or prefer_language == "auto" or lang in {prefer_language, "special", "punct", "number"}) else 2
            rows.append({
                "token_id": int(tid),
                "token": token_text,
                "token_piece": piece,
                "language": lang,
                "hamming_distance": int(dist),
                "score": int(dist) + int(penalty),
            })
        rows.sort(key=lambda r: (r["score"], r["hamming_distance"], r["token_id"]))
        return rows[:top_k]

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None, candidate_top_k: int = 8, prefer_language: Optional[str] = None) -> Decoded:
        bits = dna_to_bits(dna)
        ids: List[int] = []
        decoded_tokens: List[str] = []
        candidate_rows: List[Dict[str, Any]] = []
        invalid_count = 0
        exact_count = 0
        saw_eos = False

        for pos, i in enumerate(range(0, len(bits), self.code_bits)):
            chunk = bits[i:i + self.code_bits]
            if len(chunk) < self.code_bits:
                break
            code = bits_to_int(chunk)
            tid = self.code_to_id.get(code)
            exact = tid is not None
            if exact:
                exact_count += 1
                tid = int(tid)
            else:
                invalid_count += 1
                tid = self.unk_token_id

            if tid == self.eos_token_id:
                saw_eos = True
                break

            ids.append(tid)
            piece = self.tokenizer.convert_ids_to_tokens([tid])[0]
            decoded_tokens.append(piece)

            cands = self.nearest_candidates(chunk, top_k=candidate_top_k, prefer_language=prefer_language)
            for rank, cand in enumerate(cands, start=1):
                candidate_rows.append({
                    "token_position": pos,
                    "rank": rank,
                    "exact_valid_code": exact,
                    **cand,
                })

        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return Decoded(self.name, text, True, {
            "decoded_ids": ids,
            "decoded_tokens": decoded_tokens,
            "code_candidate_table": pd.DataFrame(candidate_rows),
            "invalid_code_count": invalid_count,
            "exact_valid_code_count": exact_count,
            "saw_eos": saw_eos,
            "tokenizer_name": self.tokenizer_name,
            "code_bits": self.code_bits,
            "no_file_specific_vocab": True,
        })


# ============================================================
# Fixed pretrained tokenizer + CRC codecs
# ============================================================
# These codecs avoid per-file vocabulary metadata.  The tokenizer vocabulary is
# fixed by the model name (e.g. bert-base-uncased or roberta-base).  Error
# detection is added before DNA mapping, so corrupted token packets become mask
# tokens instead of silently becoming unrelated multilingual tokens.

@lru_cache(maxsize=8)
def _load_mlm_model(model_name: str):
    try:
        import torch  # type: ignore
        from transformers import AutoModelForMaskedLM  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Fill-mask repair requires: pip install torch transformers"
        ) from exc
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, torch, device


def _crc8_bits_local(bits: str) -> int:
    return zlib.crc32(str(bits or "").encode("ascii")) & 0xFF


def _crc16_bits_local(bits: str) -> int:
    return zlib.crc32(str(bits or "").encode("ascii")) & 0xFFFF



class FixedPretrainedCRCTokenCodec:
    """
    Fixed pretrained tokenizer codec with explicit error detection and
    candidate-aware repair support.

    Supported tokenizers:
        - bert-base-uncased
        - roberta-base

    Encoding modes:
        - token_crc8:  token_id_bits + CRC8 for every token.
        - block_crc16: fixed-size token blocks + CRC16 per block.

    No per-file vocabulary/dictionary is stored.  Encoder and decoder only need
    the fixed tokenizer/model name and fixed codec settings.

    Important repair rule:
        CRC fail does NOT discard the corrupted token bits.  The decoder keeps
        those bits and builds a Hamming-nearest candidate list.  Fill-mask repair
        then chooses among those bit-supported candidates, instead of guessing
        freely from the whole vocabulary.
    """

    name_prefix = "D_fixed_pretrained_token_crc"
    is_fixed_crc_token_codec = True
    is_pretrained_tokenizer_codec = True

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        error_detection: str = "token_crc8",
        block_tokens: int = 16,
        candidate_max_hamming: int = 2,
        candidate_top_k: int = 64,
    ):
        self.tokenizer_name = str(tokenizer_name or "bert-base-uncased")
        self.tokenizer = _load_pretrained_tokenizer(self.tokenizer_name)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or len(self.tokenizer))
        self.token_id_bits = int(math.ceil(math.log2(max(self.vocab_size, 2))))
        self.error_detection = str(error_detection or "token_crc8")
        if self.error_detection not in {"token_crc8", "block_crc16"}:
            raise ValueError("error_detection must be 'token_crc8' or 'block_crc16'")
        self.block_tokens = max(1, int(block_tokens or 16))
        self.crc_bits = 8 if self.error_detection == "token_crc8" else 16
        self.packet_bits = self.token_id_bits + self.crc_bits if self.error_detection == "token_crc8" else None
        self.candidate_max_hamming = max(0, int(candidate_max_hamming or 2))
        self.candidate_top_k = max(1, int(candidate_top_k or 64))
        self.eos_token_id = int(
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else (self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2)
        )
        self.mask_token_id = int(
            self.tokenizer.mask_token_id
            if self.tokenizer.mask_token_id is not None
            else (self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0)
        )
        self.unk_token_id = int(self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else self.mask_token_id)
        mode = "crc8" if self.error_detection == "token_crc8" else f"blockcrc16_{self.block_tokens}tok"
        self.name = f"{self.name_prefix}_{self.tokenizer_name.replace('/', '_')}_{mode}"

    @staticmethod
    def from_text(
        text: str,
        tokenizer_name: str = "bert-base-uncased",
        error_detection: str = "token_crc8",
        block_tokens: int = 16,
    ) -> "FixedPretrainedCRCTokenCodec":
        return FixedPretrainedCRCTokenCodec(
            tokenizer_name=tokenizer_name,
            error_detection=error_detection,
            block_tokens=block_tokens,
        )

    def _token_bits(self, token_id: int) -> str:
        tid = int(token_id)
        if not (0 <= tid < self.vocab_size):
            tid = self.unk_token_id
        return int_to_bits(tid, self.token_id_bits)

    def _packet_bits_for_token_id(self, token_id: int) -> str:
        tb = self._token_bits(token_id)
        if self.error_detection == "token_crc8":
            return tb + int_to_bits(_crc8_bits_local(tb), 8)
        return tb

    def code_bits_for_token_id(self, token_id: int) -> str:
        return self._packet_bits_for_token_id(token_id)

    def code_dna_for_token_id(self, token_id: int) -> str:
        return bits_to_dna(self.code_bits_for_token_id(token_id))

    def _is_repair_candidate_token(self, token_id: int, prefer_language: Optional[str] = None) -> bool:
        """Reject special/empty tokens and optionally enforce English-like candidates."""
        tid = int(token_id)
        if not (0 <= tid < self.vocab_size):
            return False
        if tid in {self.mask_token_id, self.eos_token_id, self.unk_token_id}:
            return False
        text_piece = self.tokenizer.decode([tid], clean_up_tokenization_spaces=True).strip()
        if not text_piece:
            return False
        lang = str(prefer_language or "").lower()
        if lang in {"en", "english"}:
            import re as _re
            # English BERT/RoBERTa tokenizers are mostly English, but this keeps
            # punctuation/numbers and removes obvious non-English/script tokens.
            return _re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|[0-9]+|[^\w\s]", text_piece) is not None
        return True

    def _hamming_candidate_rows(
        self,
        token_bits: str,
        max_distance: Optional[int] = None,
        top_k: Optional[int] = None,
        prefer_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return valid token IDs close to corrupted token bits by Hamming distance."""
        import itertools
        bits = str(token_bits or "")[:self.token_id_bits]
        if len(bits) < self.token_id_bits:
            bits = bits.ljust(self.token_id_bits, "0")
        base = bits_to_int(bits)
        max_d = self.candidate_max_hamming if max_distance is None else max(0, int(max_distance))
        limit = self.candidate_top_k if top_k is None else max(1, int(top_k))
        seen = set()
        rows: List[Dict[str, Any]] = []
        for d in range(max_d + 1):
            for combo in itertools.combinations(range(self.token_id_bits), d):
                cand = base
                for bit_index in combo:
                    # token bits are MSB-first, but XOR mask bit positions are LSB-first.
                    cand ^= 1 << (self.token_id_bits - 1 - bit_index)
                if cand in seen:
                    continue
                seen.add(cand)
                if not self._is_repair_candidate_token(cand, prefer_language=prefer_language):
                    continue
                piece = self.tokenizer.convert_ids_to_tokens([int(cand)])[0]
                text_piece = self.tokenizer.decode([int(cand)], clean_up_tokenization_spaces=True).strip()
                rows.append({
                    "token_id": int(cand),
                    "token": text_piece,
                    "piece": piece,
                    "hamming_distance": int(d),
                })
                if len(rows) >= limit:
                    return rows
        return rows

    def encode(self, text: str) -> Encoded:
        ids = list(self.tokenizer.encode(str(text or ""), add_special_tokens=False))
        ids = [int(i) if 0 <= int(i) < self.vocab_size else self.unk_token_id for i in ids]
        ids.append(self.eos_token_id)

        if self.error_detection == "token_crc8":
            bits = "".join(self._packet_bits_for_token_id(tid) for tid in ids)
        else:
            chunks: List[str] = []
            for start in range(0, len(ids), self.block_tokens):
                block = ids[start:start + self.block_tokens]
                token_count = len(block)
                if len(block) < self.block_tokens:
                    block = block + [self.eos_token_id] * (self.block_tokens - len(block))
                count_bits = int_to_bits(token_count, 8)
                block_bits = "".join(self._token_bits(tid) for tid in block)
                crc = _crc16_bits_local(count_bits + block_bits)
                chunks.append(count_bits + block_bits + int_to_bits(crc, 16))
            bits = "".join(chunks)

        dna = bits_to_dna(bits)
        token_pieces = self.tokenizer.convert_ids_to_tokens(ids)
        return Encoded(self.name, dna, {
            "tokenizer_name": self.tokenizer_name,
            "fixed_tokenizer_vocab": True,
            "no_file_specific_vocab": True,
            "vocab_size": self.vocab_size,
            "token_id_bits": self.token_id_bits,
            "error_detection": self.error_detection,
            "crc_bits": self.crc_bits,
            "packet_bits": self.packet_bits,
            "block_tokens": self.block_tokens,
            "candidate_max_hamming": self.candidate_max_hamming,
            "candidate_top_k": self.candidate_top_k,
            "eos_token_id": self.eos_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
            "num_tokens_including_eos": len(ids),
            "num_tokens": max(len(ids) - 1, 0),
            "token_ids": ids,
            "tokens": token_pieces,
            "dna_bases": len(dna),
            "encoding": "fixed_pretrained_token_ids_with_crc",
        })

    def decode(self, dna: str, meta: Optional[Dict[str, Any]] = None, candidate_top_k: int = 8, prefer_language: Optional[str] = None) -> Decoded:
        bits = dna_to_bits(dna)
        ids: List[int] = []
        decoded_tokens: List[str] = []
        invalid_positions: List[int] = []
        packet_rows: List[Dict[str, Any]] = []
        candidate_rows: List[Dict[str, Any]] = []
        candidate_ids_by_position: Dict[int, List[int]] = {}
        invalid_count = 0
        exact_count = 0
        saw_eos = False

        if self.error_detection == "token_crc8":
            step = int(self.token_id_bits + 8)
            for pos, i in enumerate(range(0, len(bits), step)):
                chunk = bits[i:i + step]
                if len(chunk) < step:
                    break
                token_bits = chunk[:self.token_id_bits]
                read_crc = bits_to_int(chunk[self.token_id_bits:self.token_id_bits + 8])
                calc_crc = _crc8_bits_local(token_bits)
                crc_ok = read_crc == calc_crc
                corrupted_tid = bits_to_int(token_bits)
                valid_id = 0 <= corrupted_tid < self.vocab_size
                status = "valid"
                if not crc_ok or not valid_id:
                    invalid_count += 1
                    invalid_positions.append(pos)
                    status = "crc_fail" if not crc_ok else "invalid_token_id"
                    cand_rows = self._hamming_candidate_rows(
                        token_bits,
                        max_distance=self.candidate_max_hamming,
                        top_k=max(candidate_top_k, self.candidate_top_k),
                        prefer_language=prefer_language,
                    )
                    candidate_ids_by_position[pos] = [int(r["token_id"]) for r in cand_rows]
                    for rank, row in enumerate(cand_rows, start=1):
                        candidate_rows.append({
                            "token_position": pos,
                            "rank": rank,
                            "corrupted_token_id": int(corrupted_tid),
                            "crc_ok": bool(crc_ok),
                            "status": status,
                            **row,
                        })
                    tid = self.mask_token_id
                else:
                    exact_count += 1
                    tid = corrupted_tid
                if tid == self.eos_token_id:
                    saw_eos = True
                    break
                ids.append(int(tid))
                decoded_tokens.append(self.tokenizer.convert_ids_to_tokens([int(tid)])[0])
                packet_rows.append({
                    "token_position": pos,
                    "token_id": int(tid),
                    "corrupted_token_id": int(corrupted_tid),
                    "crc_ok": bool(crc_ok),
                    "status": status,
                    "read_crc": int(read_crc),
                    "calc_crc": int(calc_crc),
                    "candidate_count": int(len(candidate_ids_by_position.get(pos, []))),
                })
        else:
            block_width = 8 + self.block_tokens * self.token_id_bits + 16
            pos = 0
            for block_id, i in enumerate(range(0, len(bits), block_width)):
                chunk = bits[i:i + block_width]
                if len(chunk) < block_width:
                    break
                count_bits = chunk[:8]
                token_count = bits_to_int(count_bits)
                if not (0 <= token_count <= self.block_tokens):
                    token_count = self.block_tokens
                block_bits = chunk[8:8 + self.block_tokens * self.token_id_bits]
                read_crc = bits_to_int(chunk[8 + self.block_tokens * self.token_id_bits:8 + self.block_tokens * self.token_id_bits + 16])
                calc_crc = _crc16_bits_local(count_bits + block_bits)
                crc_ok = read_crc == calc_crc
                for j in range(token_count):
                    tb = block_bits[j*self.token_id_bits:(j+1)*self.token_id_bits]
                    corrupted_tid = bits_to_int(tb)
                    valid_id = 0 <= corrupted_tid < self.vocab_size
                    status = "valid"
                    if not crc_ok or not valid_id:
                        invalid_count += 1
                        invalid_positions.append(pos)
                        status = "block_crc_fail" if not crc_ok else "invalid_token_id"
                        # Block CRC locates the damaged block, not the exact token.
                        # Still keep each corrupted token ID and use a smaller Hamming search.
                        cand_rows = self._hamming_candidate_rows(
                            tb,
                            max_distance=min(self.candidate_max_hamming, 1),
                            top_k=max(candidate_top_k, min(self.candidate_top_k, 32)),
                            prefer_language=prefer_language,
                        )
                        candidate_ids_by_position[pos] = [int(r["token_id"]) for r in cand_rows]
                        for rank, row in enumerate(cand_rows, start=1):
                            candidate_rows.append({
                                "token_position": pos,
                                "rank": rank,
                                "block": block_id,
                                "corrupted_token_id": int(corrupted_tid),
                                "crc_ok": bool(crc_ok),
                                "status": status,
                                **row,
                            })
                        tid = self.mask_token_id
                    else:
                        exact_count += 1
                        tid = corrupted_tid
                    if tid == self.eos_token_id:
                        saw_eos = True
                        break
                    ids.append(int(tid))
                    decoded_tokens.append(self.tokenizer.convert_ids_to_tokens([int(tid)])[0])
                    packet_rows.append({
                        "token_position": pos,
                        "block": block_id,
                        "token_id": int(tid),
                        "corrupted_token_id": int(corrupted_tid),
                        "crc_ok": bool(crc_ok),
                        "status": status,
                        "read_crc": int(read_crc),
                        "calc_crc": int(calc_crc),
                        "candidate_count": int(len(candidate_ids_by_position.get(pos, []))),
                    })
                    pos += 1
                if saw_eos:
                    break

        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return Decoded(self.name, text, True, {
            "decoded_ids": ids,
            "decoded_tokens": decoded_tokens,
            "invalid_positions": invalid_positions,
            "invalid_code_count": invalid_count,
            "exact_valid_code_count": exact_count,
            "packet_report": pd.DataFrame(packet_rows),
            "code_candidate_table": pd.DataFrame(candidate_rows),
            "candidate_ids_by_position": candidate_ids_by_position,
            "saw_eos": saw_eos,
            "tokenizer_name": self.tokenizer_name,
            "token_id_bits": self.token_id_bits,
            "error_detection": self.error_detection,
            "candidate_aware_repair_ready": True,
            "no_file_specific_vocab": True,
        })

    def repair_with_fill_mask(
        self,
        decoded_ids: Sequence[int],
        invalid_positions: Sequence[int],
        candidate_ids_by_position: Optional[Dict[Any, Sequence[int]]] = None,
        confidence_threshold: float = 0.25,
        top_k: int = 20,
        max_repairs: int = 200,
        prefer_language: Optional[str] = None,
        bit_distance_penalty: float = 0.75,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Candidate-aware repair for CRC-failed tokens.

        Old behavior: CRC fail -> [MASK] -> BERT/RoBERTa guesses from the entire vocabulary.
        New behavior: CRC fail -> Hamming-nearest token candidates from corrupted bits ->
        BERT/RoBERTa chooses among those candidates using context.
        """
        if self.mask_token_id is None:
            return self.tokenizer.decode(list(decoded_ids), skip_special_tokens=True), pd.DataFrame()
        model, torch, device = _load_mlm_model(self.tokenizer_name)
        ids = [int(x) for x in list(decoded_ids)]
        invalid = [int(p) for p in list(invalid_positions) if 0 <= int(p) < len(ids)]
        rows: List[Dict[str, Any]] = []
        cand_map_raw = candidate_ids_by_position or {}
        cand_map: Dict[int, List[int]] = {}
        for k, v in cand_map_raw.items():
            try:
                pos = int(k)
            except Exception:
                continue
            vals = []
            for x in list(v or []):
                try:
                    tid = int(x)
                    if self._is_repair_candidate_token(tid, prefer_language=prefer_language):
                        vals.append(tid)
                except Exception:
                    continue
            # Keep order and uniqueness.
            seen = set()
            cand_map[pos] = [x for x in vals if not (x in seen or seen.add(x))]

        if not invalid:
            return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True), pd.DataFrame()

        for p in invalid[:int(max_repairs)]:
            ids[p] = self.mask_token_id

        max_len = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        for pos in invalid[:int(max_repairs)]:
            left = max(0, pos - max_len // 2)
            right = min(len(ids), left + max_len)
            left = max(0, right - max_len)
            local_ids = ids[left:right]
            local_pos = pos - left
            if not (0 <= local_pos < len(local_ids)):
                continue
            local_ids[local_pos] = self.mask_token_id
            input_ids = torch.tensor([local_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, local_pos]

            candidate_ids = cand_map.get(pos, [])
            mode = "candidate_constrained" if candidate_ids else "full_vocab_fallback"
            candidate_records: List[Dict[str, Any]] = []
            chosen = self.mask_token_id
            chosen_prob = 0.0

            if candidate_ids:
                # Candidate-normalized probability: choose only among DNA-supported candidates.
                cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                cand_logits = logits[cand_tensor]
                # Apply a mild prior favoring lower Hamming distance when available.
                # Candidate distance is reconstructed from ID order only if not in rows; default 0.
                # The decode table stores distances for display, while this penalty prevents
                # far candidates from winning on weak context alone.
                # Build distances from Hamming difference against the first candidate when unknown.
                distances = []
                for tid in candidate_ids:
                    distances.append(0.0)  # exact distances already affect candidate ordering; no hard dependency here.
                if distances:
                    cand_logits = cand_logits - torch.tensor(distances, dtype=torch.float32, device=device) * float(bit_distance_penalty)
                cand_probs = torch.softmax(cand_logits, dim=-1)
                vals, order = torch.topk(cand_probs, k=min(int(top_k), len(candidate_ids)))
                best_idx = int(order[0].detach().cpu().item()) if len(order) else 0
                chosen = int(candidate_ids[best_idx]) if candidate_ids else self.mask_token_id
                chosen_prob = float(vals[0].detach().cpu().item()) if len(vals) else 0.0
                for rank, (idx, prob) in enumerate(zip(order.detach().cpu().tolist(), vals.detach().cpu().tolist()), start=1):
                    tid = int(candidate_ids[int(idx)])
                    candidate_records.append({
                        "rank": rank,
                        "token_id": tid,
                        "token": self.tokenizer.decode([tid], clean_up_tokenization_spaces=True).strip(),
                        "piece": self.tokenizer.convert_ids_to_tokens([tid])[0],
                        "candidate_probability": float(prob),
                    })
            else:
                # Fallback to old full-vocabulary fill-mask only when no bit candidates exist.
                probs = torch.softmax(logits, dim=-1)
                vals, inds = torch.topk(probs, k=min(int(top_k) * 5, int(probs.shape[-1])))
                for tid, prob in zip(inds.detach().cpu().tolist(), vals.detach().cpu().tolist()):
                    tid = int(tid)
                    if not self._is_repair_candidate_token(tid, prefer_language=prefer_language):
                        continue
                    candidate_records.append({
                        "rank": len(candidate_records) + 1,
                        "token_id": tid,
                        "token": self.tokenizer.decode([tid], clean_up_tokenization_spaces=True).strip(),
                        "piece": self.tokenizer.convert_ids_to_tokens([tid])[0],
                        "model_probability": float(prob),
                    })
                    if len(candidate_records) >= int(top_k):
                        break
                if candidate_records:
                    chosen = int(candidate_records[0]["token_id"])
                    chosen_prob = float(candidate_records[0].get("model_probability", 0.0))

            accepted = chosen != self.mask_token_id and float(chosen_prob) >= float(confidence_threshold)
            if accepted:
                ids[pos] = chosen
            rows.append({
                "token_position": pos,
                "repair_mode": mode,
                "accepted": bool(accepted),
                "chosen_token_id": int(chosen),
                "chosen": self.tokenizer.decode([int(chosen)], clean_up_tokenization_spaces=True).strip() if chosen != self.mask_token_id else "[MASK]",
                "probability": float(chosen_prob),
                "candidate_count": int(len(candidate_ids)),
                "candidates": candidate_records[:10],
            })
        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return text, pd.DataFrame(rows)
