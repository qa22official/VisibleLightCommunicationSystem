#!/usr/bin/env python3
"""将 OCR/output 下按文件名排序的汉字 txt 解码为单个 .bin。"""

from __future__ import annotations

import argparse
import binascii
import shutil
import sqlite3
import struct
from pathlib import Path
from typing import Dict, Set, Tuple
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


def parse_eraser_file(path: Path) -> set[tuple[int, int, int]]:
    rows: set[tuple[int, int, int]] = set()
    if not path.exists():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line or line.startswith("frame"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            frame_no = int(parts[0])
            row_no = int(parts[1])
            col_no = int(parts[2])
        except ValueError:
            continue
        rows.add((frame_no, row_no, col_no))
    return rows


def write_eraser_file(path: Path, rows: set[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame\trow\tcol"]
    for frame_no, row_no, col_no in sorted(rows):
        lines.append(f"{frame_no}\t{row_no}\t{col_no}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decode_txt_files_to_bytes(
    files: list[Path],
    char_to_code: Dict[str, int],
    fill_unknown_zero: bool,
    existing_eraser: Set[Tuple[int, int, int]],
    rows_per_frame: int,
    cols_per_row: int,
) -> tuple[bytes, set[tuple[int, int, int]]]:
    out = bytearray()
    unknown_positions: set[tuple[int, int, int]] = set()

    def append_row_codes(codes: list[int]) -> None:
        acc = 0
        acc_bits = 0
        for code in codes:
            acc = (acc << BITS_PER_CHAR) | code
            acc_bits += BITS_PER_CHAR
            while acc_bits >= 8:
                acc_bits -= 8
                out.append((acc >> acc_bits) & 0xFF)
                if acc_bits > 0:
                    acc &= (1 << acc_bits) - 1
                else:
                    acc = 0
        if acc_bits > 0:
            out.append((acc << (8 - acc_bits)) & 0xFF)

    for order, txt_path in enumerate(files, start=1):
        try:
            frame_no = int(txt_path.stem)
        except ValueError:
            frame_no = order

        text = txt_path.read_text(encoding="utf-8")
        row_lines = [ln.replace(" ", "").replace("\u3000", "") for ln in text.splitlines()]
        if not row_lines and text.strip():
            row_lines = ["".join(ch for ch in text if not ch.isspace())]

        if rows_per_frame > 0:
            row_count = rows_per_frame
        else:
            frame_eraser_rows = [r for f, r, _ in existing_eraser if f == frame_no]
            max_row_in_eraser = max(frame_eraser_rows) if frame_eraser_rows else 0
            row_count = max(len(row_lines), max_row_in_eraser)

        for row_no in range(1, row_count + 1):
            row_text = row_lines[row_no - 1] if row_no - 1 < len(row_lines) else ""
            idx = 0

            if cols_per_row <= 0:
                cols = len(row_text)
            else:
                cols = cols_per_row

            row_codes: list[int] = []
            for col_no in range(1, cols + 1):
                if (frame_no, row_no, col_no) in existing_eraser:
                    code = 0
                else:
                    if idx >= len(row_text):
                        unknown_positions.add((frame_no, row_no, col_no))
                        code = 0
                    else:
                        ch = row_text[idx]
                        idx += 1

                        if ch not in char_to_code:
                            if not fill_unknown_zero:
                                raise ValueError(
                                    f"文件 {txt_path.name} 第 {row_no} 行第 {col_no} 列字符无法映射: {ch!r}"
                                )
                            unknown_positions.add((frame_no, row_no, col_no))
                            code = 0
                        else:
                            code = char_to_code[ch]
                row_codes.append(code)
            append_row_codes(row_codes)

    return bytes(out), unknown_positions


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
    parser.add_argument(
        "--eraser-path",
        type=Path,
        default=root_dir / "ECC" / "eraser.txt",
        help="缺失/不可解码位置文件（默认: ECC/eraser.txt）",
    )
    parser.add_argument(
        "--fill-unknown-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="遇到无法映射字符时按 11 位 0 写入并记录到 eraser（默认开启）",
    )
    parser.add_argument(
        "--rows-per-frame",
        type=int,
        default=26,
        help="每帧固定行数（默认: 26）",
    )
    parser.add_argument(
        "--cols-per-row",
        type=int,
        default=40,
        help="每行固定列数（默认: 40）",
    )
    return parser


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    args = build_parser().parse_args()

    files = list_txt_files(args.input_dir)
    char_to_code = load_char_to_code(args.db)
    existing_eraser = parse_eraser_file(args.eraser_path)
    decoded, unknown_positions = decode_txt_files_to_bytes(
        files,
        char_to_code,
        args.fill_unknown_zero,
        existing_eraser,
        args.rows_per_frame,
        args.cols_per_row,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    out_path.write_bytes(decoded)

    merged_eraser = set(existing_eraser)
    merged_eraser.update(unknown_positions)
    write_eraser_file(args.eraser_path, merged_eraser)

    print("输入文件顺序:")
    for p in files:
        print(f"- {p.name}")
    print(f"输出文件: {out_path}")
    print(f"输出字节数: {len(decoded)}")
    print(f"新增无法解码位置: {len(unknown_positions)}")
    print(f"eraser 总位置数: {len(merged_eraser)}")
    return 0

def decode_ch(encoded_packet):
    """解析 CHv1 头并做 CRC 校验（兼容旧接口）。"""
    MAGIC = b"CHv1"
    MIN_LENGTH = 4 + 4 + 4 + 4
    
    if len(encoded_packet) < MIN_LENGTH:
        raise ValueError("数据包长度不足，无法解析头部")
    
    magic = encoded_packet[0:4]
    if magic != MAGIC:
        raise ValueError(f"无效的 MAGIC 头：{magic}, 期望 {MAGIC}")
    
    data_len = struct.unpack(">I", encoded_packet[4:8])[0]
    pad_bits = struct.unpack(">I", encoded_packet[8:12])[0]
    
    expected_crc_pos = 12 + data_len
    if len(encoded_packet) < expected_crc_pos + 4:
        raise ValueError("数据包截断，缺少 CRC 校验码")
        
    data_bytes = encoded_packet[12:expected_crc_pos]
    received_crc = struct.unpack(">I", encoded_packet[expected_crc_pos : expected_crc_pos + 4])[0]
    
    calculated_crc = binascii.crc32(data_bytes) & 0xFFFFFFFF
    if calculated_crc != received_crc:
        raise ValueError(f"CRC 校验失败：计算值 {calculated_crc:#x}, 接收值 {received_crc:#x}")
    return data_bytes, pad_bits


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)
