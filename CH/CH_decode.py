#!/usr/bin/env python3
"""将 OCR/output 下按文件名排序的汉字 txt 解码为单个 .bin。"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict
import struct 
import binascii
DICT_SIZE = 2048
BITS_PER_CHAR = 11
TABLE_NAME = "hanzi_map"


def load_char_to_code(db_path: Path) -> Dict[str, int]:
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

    char_to_code: Dict[str, int] = {}
    for code, ch in rows:
        if not isinstance(code, int) or code < 0 or code >= DICT_SIZE:
            raise ValueError(f"非法 code: {code}")
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(f"非法汉字映射: code={code}, chinese={ch!r}")
        if ch in char_to_code:
            raise ValueError(f"重复汉字映射: {ch}")
        char_to_code[ch] = code

    return char_to_code


def list_txt_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.glob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 .txt 文件")
    return files


def decode_txt_files_to_bytes(files: list[Path], char_to_code: Dict[str, int]) -> bytes:
    acc = 0
    acc_bits = 0
    out = bytearray()

    for txt_path in files:
        text = txt_path.read_text(encoding="utf-8")
        symbol_pos = 0
        for ch in text:
            if ch.isspace():
                continue

            symbol_pos += 1
            if ch not in char_to_code:
                raise ValueError(
                    f"文件 {txt_path.name} 第 {symbol_pos} 个有效字符无法映射: {ch!r}"
                )

            acc = (acc << BITS_PER_CHAR) | char_to_code[ch]
            acc_bits += BITS_PER_CHAR

            while acc_bits >= 8:
                acc_bits -= 8
                out.append((acc >> acc_bits) & 0xFF)
                if acc_bits > 0:
                    acc &= (1 << acc_bits) - 1
                else:
                    acc = 0

    return bytes(out)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="按文件名顺序解码 OCR/output 下所有 txt 并合并输出 .bin"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root_dir / "OCR" / "output",
        help="输入目录（默认: OCR/output）",
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
        default=script_dir / "decode_output",
        help="输出目录（默认: CH/decode_output）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ocr_merged.bin",
        help="输出文件名（默认: ocr_merged.bin）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    files = list_txt_files(args.input_dir)
    char_to_code = load_char_to_code(args.db)
    decoded = decode_txt_files_to_bytes(files, char_to_code)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    out_path.write_bytes(decoded)

    print("输入文件顺序:")
    for p in files:
        print(f"- {p.name}")
    print(f"输出文件: {out_path}")
    print(f"输出字节数: {len(decoded)}")
    return 0

def decode_ch(encoded_packet):
    """
    解析 CH 编码数据包，验证头部和 CRC，提取原始数据和填充位信息。
    """
    MAGIC = b'CHv1'
    MIN_LENGTH = 4 + 4 + 4 + 4
    
    if len(encoded_packet) < MIN_LENGTH:
        raise ValueError("数据包长度不足，无法解析头部")
    
    magic = encoded_packet[0:4]
    if magic != MAGIC:
        raise ValueError(f"无效的 MAGIC 头：{magic}, 期望 {MAGIC}")
    
    data_len = struct.unpack('>I', encoded_packet[4:8])[0]
    pad_bits = struct.unpack('>I', encoded_packet[8:12])[0]
    
    expected_crc_pos = 12 + data_len
    if len(encoded_packet) < expected_crc_pos + 4:
        raise ValueError("数据包截断，缺少 CRC 校验码")
        
    data_bytes = encoded_packet[12:expected_crc_pos]
    received_crc = struct.unpack('>I', encoded_packet[expected_crc_pos:expected_crc_pos+4])[0]
    
    calculated_crc = binascii.crc32(data_bytes) & 0xffffffff
    if calculated_crc != received_crc:
        raise ValueError(f"CRC 校验失败：计算值 {calculated_crc:#x}, 接收值 {received_crc:#x}")
    return data_bytes, pad_bits


if __name__ == "__main__":
    raise SystemExit(main())
