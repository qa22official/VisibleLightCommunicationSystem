#!/usr/bin/env python3
"""2048-Hanzi codec for binary files.

This tool initializes a 2048-character dictionary in dictionary.db,
encodes binary files into Hanzi text files, and decodes them back.
"""

from __future__ import annotations

import argparse
import binascii
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DICT_SIZE = 2048
BITS_PER_CHAR = 11
MAGIC = "VLCS-HANZI-2048"
PAYLOAD_MARKER = "---PAYLOAD---"
TABLE_NAME = "hanzi_map"


def extract_unique_hanzi(text: str) -> List[str]:
    seen = set()
    chars: List[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" and ch not in seen:
            seen.add(ch)
            chars.append(ch)
    return chars


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            code INTEGER PRIMARY KEY,
            chinese TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS codec_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def init_dictionary(db_path: Path, source_path: Path, force: bool) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    source_text = source_path.read_text(encoding="utf-8")
    hanzi_chars = extract_unique_hanzi(source_text)
    if len(hanzi_chars) < DICT_SIZE:
        raise ValueError(
            f"Not enough unique Hanzi in {source_path}. "
            f"Found {len(hanzi_chars)}, need at least {DICT_SIZE}."
        )

    selected = hanzi_chars[:DICT_SIZE]

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_tables(conn)
        cur = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        row_count = int(cur.fetchone()[0])

        if row_count > 0 and not force:
            cur = conn.execute(f"SELECT code, chinese FROM {TABLE_NAME} ORDER BY code")
            existing = [row[1] for row in cur.fetchall()]
            if len(existing) == DICT_SIZE and existing == selected:
                print(f"Dictionary already initialized in {db_path}.")
                return
            raise ValueError(
                f"{TABLE_NAME} already has {row_count} rows. "
                "Use --force to overwrite."
            )

        with conn:
            if row_count > 0:
                conn.execute(f"DELETE FROM {TABLE_NAME}")
            conn.executemany(
                f"INSERT INTO {TABLE_NAME}(code, chinese) VALUES (?, ?)",
                list(enumerate(selected)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO codec_meta(key, value) VALUES('dict_size', ?)",
                (str(DICT_SIZE),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO codec_meta(key, value) VALUES('bits_per_char', ?)",
                (str(BITS_PER_CHAR),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO codec_meta(key, value) VALUES('magic', ?)",
                (MAGIC,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO codec_meta(key, value) VALUES('source', ?)",
                (str(source_path),),
            )

        print(
            f"Initialized {TABLE_NAME} with {DICT_SIZE} Hanzi in {db_path} "
            f"(source: {source_path})."
        )
    finally:
        conn.close()


def load_dictionary(db_path: Path) -> Tuple[List[str], Dict[str, int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_tables(conn)
        cur = conn.execute(f"SELECT code, chinese FROM {TABLE_NAME} ORDER BY code")
        rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) != DICT_SIZE:
        raise ValueError(
            f"{TABLE_NAME} must contain exactly {DICT_SIZE} rows; found {len(rows)}. "
            "Run init-dict first."
        )

    code_to_char: List[Optional[str]] = [None] * DICT_SIZE
    char_to_code: Dict[str, int] = {}

    for code_value, char_value in rows:
        if not isinstance(code_value, int):
            raise ValueError(f"Invalid code type: {type(code_value)}")
        if code_value < 0 or code_value >= DICT_SIZE:
            raise ValueError(f"Code out of range: {code_value}")

        if not isinstance(char_value, str) or len(char_value) != 1:
            raise ValueError(f"Invalid Hanzi entry for code {code_value}: {char_value!r}")

        if code_to_char[code_value] is not None:
            raise ValueError(f"Duplicate code in dictionary: {code_value}")
        if char_value in char_to_code:
            raise ValueError(f"Duplicate Hanzi in dictionary: {char_value}")

        code_to_char[code_value] = char_value
        char_to_code[char_value] = code_value

    if any(ch is None for ch in code_to_char):
        raise ValueError("Dictionary has missing code slots.")

    return [ch for ch in code_to_char if ch is not None], char_to_code


def encode_bytes_to_hanzi(data: bytes, code_to_char: List[str]) -> str:
    mask = (1 << BITS_PER_CHAR) - 1
    acc = 0
    acc_bits = 0
    out_chars: List[str] = []

    for b in data:
        acc = (acc << 8) | b
        acc_bits += 8
        while acc_bits >= BITS_PER_CHAR:
            acc_bits -= BITS_PER_CHAR
            idx = (acc >> acc_bits) & mask
            out_chars.append(code_to_char[idx])
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    if acc_bits > 0:
        idx = (acc << (BITS_PER_CHAR - acc_bits)) & mask
        out_chars.append(code_to_char[idx])

    return "".join(out_chars)


def decode_hanzi_to_bytes(payload: str, expected_bytes: int, char_to_code: Dict[str, int]) -> bytes:
    acc = 0
    acc_bits = 0
    out = bytearray()

    for pos, ch in enumerate(payload, start=1):
        if ch not in char_to_code:
            raise ValueError(f"Unknown Hanzi at payload position {pos}: {ch!r}")

        acc = (acc << BITS_PER_CHAR) | char_to_code[ch]
        acc_bits += BITS_PER_CHAR

        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
            if acc_bits > 0:
                acc &= (1 << acc_bits) - 1
            else:
                acc = 0

    if len(out) < expected_bytes:
        raise ValueError(
            f"Decoded bytes shorter than expected: got {len(out)}, expected {expected_bytes}."
        )

    return bytes(out[:expected_bytes])


def write_encoded_text(path: Path, payload: str, raw: bytes, line_width: int) -> None:
    crc = binascii.crc32(raw) & 0xFFFFFFFF

    lines = [
        MAGIC,
        f"bytes={len(raw)}",
        f"crc32={crc:08x}",
        PAYLOAD_MARKER,
    ]

    if line_width <= 0:
        lines.append(payload)
    else:
        for i in range(0, len(payload), line_width):
            lines.append(payload[i : i + line_width])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_encoded_text(path: Path) -> Tuple[int, Optional[int], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError("Encoded text format error: file is too short.")

    if lines[0].strip() != MAGIC:
        raise ValueError(
            f"Invalid header magic in {path}. Expected {MAGIC!r}, got {lines[0].strip()!r}."
        )

    if not lines[1].startswith("bytes="):
        raise ValueError("Invalid header: missing bytes=...")
    expected_bytes = int(lines[1].split("=", 1)[1])
    if expected_bytes < 0:
        raise ValueError("Invalid header: bytes must be >= 0")

    crc_value: Optional[int] = None
    if lines[2].startswith("crc32="):
        crc_text = lines[2].split("=", 1)[1].strip().lower()
        if len(crc_text) != 8 or any(ch not in "0123456789abcdef" for ch in crc_text):
            raise ValueError("Invalid header: crc32 must be 8 hex chars")
        crc_value = int(crc_text, 16)
    else:
        raise ValueError("Invalid header: missing crc32=...")

    try:
        marker_idx = next(i for i, line in enumerate(lines) if line.strip() == PAYLOAD_MARKER)
    except StopIteration as exc:
        raise ValueError(f"Invalid format: missing marker {PAYLOAD_MARKER!r}") from exc

    payload = ""
    for line in lines[marker_idx + 1 :]:
        payload += "".join(ch for ch in line if not ch.isspace())

    if not payload and expected_bytes != 0:
        raise ValueError("Encoded file has empty payload.")

    return expected_bytes, crc_value, payload


def command_init_dict(args: argparse.Namespace) -> None:
    init_dictionary(args.db, args.source, args.force)


def command_encode(args: argparse.Namespace) -> None:
    code_to_char, _ = load_dictionary(args.db)

    raw = args.input.read_bytes()
    payload = encode_bytes_to_hanzi(raw, code_to_char)
    write_encoded_text(args.output, payload, raw, args.line_width)

    print(
        f"Encoded {args.input} ({len(raw)} bytes) -> {args.output} "
        f"({len(payload)} Hanzi)."
    )


def command_decode(args: argparse.Namespace) -> None:
    _, char_to_code = load_dictionary(args.db)

    expected_bytes, crc_value, payload = parse_encoded_text(args.input)
    decoded = decode_hanzi_to_bytes(payload, expected_bytes, char_to_code)

    if crc_value is not None:
        actual_crc = binascii.crc32(decoded) & 0xFFFFFFFF
        if actual_crc != crc_value:
            raise ValueError(
                f"CRC mismatch: expected {crc_value:08x}, got {actual_crc:08x}."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(decoded)

    print(
        f"Decoded {args.input} ({len(payload)} Hanzi) -> {args.output} "
        f"({len(decoded)} bytes)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2048-Hanzi binary codec backed by dictionary.db"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_p = subparsers.add_parser(
        "init-dict", help="Initialize 2048-Hanzi dictionary in SQLite"
    )
    init_p.add_argument(
        "--db",
        type=Path,
        default=Path("dictionary.db"),
        help="Path to dictionary SQLite DB (default: dictionary.db)",
    )
    init_p.add_argument(
        "--source",
        type=Path,
        default=Path("example/CH_original.txt"),
        help="Hanzi source file used to build 2048 unique chars",
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dictionary table if non-empty",
    )
    init_p.set_defaults(func=command_init_dict)

    enc_p = subparsers.add_parser("encode", help="Encode binary file to Hanzi txt")
    enc_p.add_argument("--db", type=Path, default=Path("dictionary.db"))
    enc_p.add_argument("--input", type=Path, required=True, help="Input binary file")
    enc_p.add_argument("--output", type=Path, required=True, help="Output Hanzi txt file")
    enc_p.add_argument(
        "--line-width",
        type=int,
        default=128,
        help="Payload line width in output txt (0 means no wrapping)",
    )
    enc_p.set_defaults(func=command_encode)

    dec_p = subparsers.add_parser("decode", help="Decode Hanzi txt back to binary")
    dec_p.add_argument("--db", type=Path, default=Path("dictionary.db"))
    dec_p.add_argument("--input", type=Path, required=True, help="Input Hanzi txt file")
    dec_p.add_argument("--output", type=Path, required=True, help="Output binary file")
    dec_p.set_defaults(func=command_decode)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
        return 0
    except Exception as exc:  # pragma: no cover - CLI level guard
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

