#!/usr/bin/env python3
"""用于 test/*.bin 的批量 RS(11-bit) 编码器。

设计目标：
- Reed-Solomon 以 11bit 为一个数据元。
- 默认按 5% 的综合数据元损坏率（擦除模型）设计参数。
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Iterable, List

from rs11_core import RS11, RS11Config, bytes_to_11bit_symbols, pack_11bit_symbols

MAGIC = b"RS11ECC1"
HEADER_FMT = "<8sHHBQQBI"
# magic, n, k, pad_bits, orig_bytes, total_data_symbols, payload_fmt, block_count
PAYLOAD_FMT_PACKED11 = 1


def encode_file(src: Path, dst: Path, rs: RS11) -> None:
    raw = src.read_bytes()
    data_symbols, pad_bits = bytes_to_11bit_symbols(raw)

    k = rs.cfg.k
    n = rs.cfg.n

    blocks: List[List[int]] = []
    for i in range(0, len(data_symbols), k):
        block = data_symbols[i : i + k]
        if len(block) < k:
            # 最后一个分组使用 shortened RS：保持校验符号数量不变，
            # 同时避免把人为补的前导零数据写入文件。
            shorten = k - len(block)
            padded = [0] * shorten + block
            full_cw = rs.encode_block(padded)
            blocks.append(full_cw[shorten:])
        else:
            blocks.append(rs.encode_block(block))

    block_count = len(blocks)
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        n,
        k,
        pad_bits,
        len(raw),
        len(data_symbols),
        PAYLOAD_FMT_PACKED11,
        block_count,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(header)
        for cw in blocks:
            f.write(pack_11bit_symbols(cw))


def iter_bin_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob("*.bin")):
        if p.is_file():
            yield p


def design_k_from_damage_rate(n: int, damage_rate: float) -> int:
    if n <= 1:
        raise ValueError("n 必须大于 1")
    if not (0.0 < damage_rate < 1.0):
        raise ValueError("damage-rate 必须在 (0, 1) 区间内")

    parity = math.ceil(n * damage_rate)
    if parity <= 0:
        parity = 1
    if parity >= n:
        raise ValueError("damage-rate 对当前 n 来说过大")
    return n - parity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 11bit 数据元的 RS 对 .bin 文件编码（默认按 5% 损坏率设计）"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test"),
        help="输入 .bin 文件目录（默认：test）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ECC/encode_output"),
        help="编码后 .bin 输出目录（默认：ECC/encode_output）",
    )
    parser.add_argument("--n", type=int, default=255, help="RS 码字长度 n（默认：255）")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="RS 数据长度 k（默认：由 --damage-rate 自动计算）",
    )
    parser.add_argument(
        "--damage-rate",
        type=float,
        default=0.05,
        help="用于擦除设计的目标综合数据元损坏率（默认：0.05）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"未找到输入目录：{args.input_dir}")

    k = args.k if args.k is not None else design_k_from_damage_rate(args.n, args.damage_rate)
    cfg = RS11Config(n=args.n, k=k)
    rs = RS11(cfg)

    files = list(iter_bin_files(args.input_dir))
    if not files:
        raise SystemExit(f"在 {args.input_dir} 中未找到 .bin 文件")

    print(
        f"使用 RS({cfg.n},{cfg.k})，数据元宽度 11bit。"
        f"冗余占比={(cfg.n - cfg.k) / cfg.n:.2%}，最大可恢复擦除率={(cfg.n - cfg.k) / cfg.n:.2%}，"
        f"目标损坏率={args.damage_rate:.2%}。"
    )

    for src in files:
        dst = args.output_dir / f"{src.stem}.encoded.bin"
        encode_file(src, dst, rs)
        print(f"已编码：{src} -> {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
