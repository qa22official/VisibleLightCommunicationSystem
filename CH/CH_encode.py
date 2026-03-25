#!/usr/bin/env python3
"""将 ECC 编码后的 .bin 按 11bit 映射为汉字文本。"""

from __future__ import annotations

import argparse
import binascii
import shutil
import sqlite3
import struct
from pathlib import Path
from typing import List, Optional

DICT_SIZE = 2048
BITS_PER_CHAR = 11
TABLE_NAME = "hanzi_map"


def load_code_to_char(db_path: Path) -> List[str]:
    if not db_path.exists():
        raise FileNotFoundError(f"未找到字典数据库: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            f"SELECT code, chinese FROM {TABLE_NAME} ORDER BY code"
        ).fetchall()
    finally:
        conn.close()

    if len(rows) != DICT_SIZE:
        raise ValueError(
            f"{TABLE_NAME} 需要 {DICT_SIZE} 条映射，当前为 {len(rows)}"
        )

    code_to_char: List[Optional[str]] = [None] * DICT_SIZE
    for code, ch in rows:
        if not isinstance(code, int) or code < 0 or code >= DICT_SIZE:
            raise ValueError(f"非法 code: {code}")
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(f"非法汉字映射: code={code}, chinese={ch!r}")
        if code_to_char[code] is not None:
            raise ValueError(f"重复 code: {code}")
        code_to_char[code] = ch

    if any(v is None for v in code_to_char):
        raise ValueError("字典存在缺失 code")

    return [v for v in code_to_char if v is not None]


def bytes_to_hanzi(data: bytes, code_to_char: List[str]) -> str:
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


def find_single_bin(input_dir: Path) -> Path:
    files = sorted(p for p in input_dir.glob("*.bin") if p.is_file())
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 .bin 文件")
    if len(files) > 1:
        names = ", ".join(p.name for p in files)
        raise ValueError(f"在 {input_dir} 中找到多个 .bin 文件: {names}")
    return files[0]


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="将 ECC/encode_output 中唯一 .bin 按 11bit 映射为汉字 txt"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root_dir / "ECC" / "encode_output",
        help="输入目录（默认: ECC/encode_output）",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=script_dir / "dictionary.db",
        help="字典数据库路径（默认: CH/dictionary.db）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "encode_output",
        help="输出目录（默认: CH/encode_output）",
    )
    return parser


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    args = build_parser().parse_args()

    src = find_single_bin(args.input_dir)
    code_to_char = load_code_to_char(args.db)

    raw = src.read_bytes()
    payload = bytes_to_hanzi(raw, code_to_char)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{src.stem}.txt"
    out_path.write_text(payload, encoding="utf-8")

    print(f"输入文件: {src}")
    print(f"输出文件: {out_path}")
    print(f"原始字节数: {len(raw)}")
    print(f"汉字数量: {len(payload)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)


def encode_ch(data_bytes, pad_bits=0):
    """构造带 CHv1 头和 CRC 的数据包（兼容旧接口）。"""
    magic = b"CHv1"
    crc = binascii.crc32(data_bytes) & 0xFFFFFFFF
    header = magic + struct.pack(">I", len(data_bytes)) + struct.pack(">I", pad_bits)
    return header + data_bytes + struct.pack(">I", crc)
