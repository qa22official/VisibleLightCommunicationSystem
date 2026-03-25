#!/usr/bin/env python3
"""用于 test/*.bin 的批量 RS(11-bit) 编码器。

设计目标：
- Reed-Solomon 以 11bit 为一个数据元。
- 默认按“擦除 + 少量随机错误”联合模型设计参数。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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


def design_k_for_channel(
    n: int,
    erasure_rate: float,
    random_error_rate: float,
    safety_factor: float,
    min_parity_symbols: int,
) -> int:
    """按 RS 约束 2t + s <= nsym 设计 k。

    - s: 预期擦除数量
    - t: 预期随机错误数量
    """
    if n <= 1:
        raise ValueError("n 必须大于 1")
    if not (0.0 <= erasure_rate < 1.0):
        raise ValueError("erasure-rate 必须在 [0, 1) 区间内")
    if not (0.0 <= random_error_rate < 1.0):
        raise ValueError("random-error-rate 必须在 [0, 1) 区间内")
    if safety_factor < 1.0:
        raise ValueError("safety-factor 不能小于 1.0")

    required = n * (erasure_rate + 2.0 * random_error_rate) * safety_factor
    parity = max(min_parity_symbols, math.ceil(required))
    if parity <= 0:
        parity = 1
    if parity >= n:
        raise ValueError("通道参数导致 parity >= n，请降低损坏率或安全系数")
    return n - parity


def load_shared_rs_redundancy(default_value: float) -> float:
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    if not cfg_path.exists():
        return default_value

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    ecc = data.get("ecc", {})
    # 兼容旧字段 expected_damage_rate。
    val = float(ecc.get("rs_redundancy_rate", ecc.get("expected_damage_rate", default_value)))
    if not (0.0 < val < 1.0):
        return default_value
    return val


def load_shared_rs_channel_defaults() -> dict:
    """读取 config.json 中 ecc.* 的联合信道参数。"""
    defaults = {
        "erasure_rate": 0.05,
        "random_error_rate": 0.005,
        "safety_factor": 1.20,
        "min_parity_symbols": 4,
    }

    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    if not cfg_path.exists():
        return defaults

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    ecc = data.get("ecc", {})

    # 兼容旧字段：没有 erasure_rate 时继续使用 rs_redundancy_rate/expected_damage_rate。
    legacy_rate = float(ecc.get("rs_redundancy_rate", ecc.get("expected_damage_rate", defaults["erasure_rate"])))

    erasure_rate = float(ecc.get("erasure_rate", legacy_rate))
    random_error_rate = float(ecc.get("random_error_rate", defaults["random_error_rate"]))
    safety_factor = float(ecc.get("design_safety_factor", defaults["safety_factor"]))
    min_parity_symbols = int(ecc.get("min_parity_symbols", defaults["min_parity_symbols"]))

    if not (0.0 <= erasure_rate < 1.0):
        erasure_rate = defaults["erasure_rate"]
    if not (0.0 <= random_error_rate < 1.0):
        random_error_rate = defaults["random_error_rate"]
    if safety_factor < 1.0:
        safety_factor = defaults["safety_factor"]
    if min_parity_symbols < 1:
        min_parity_symbols = defaults["min_parity_symbols"]

    return {
        "erasure_rate": erasure_rate,
        "random_error_rate": random_error_rate,
        "safety_factor": safety_factor,
        "min_parity_symbols": min_parity_symbols,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 11bit 数据元的 RS 对 .bin 文件编码（默认按擦除+少量随机错误模型设计）"
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
        default=None,
        help="用于擦除设计的目标综合数据元损坏率（默认读取 config.json 的 ecc.rs_redundancy_rate）",
    )
    parser.add_argument(
        "--erasure-rate",
        type=float,
        default=None,
        help="擦除率估计 s/n（默认读取 config.json 的 ecc.erasure_rate）",
    )
    parser.add_argument(
        "--random-error-rate",
        type=float,
        default=None,
        help="随机错误率估计 t/n（默认读取 config.json 的 ecc.random_error_rate）",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=None,
        help="参数设计安全系数（默认读取 config.json 的 ecc.design_safety_factor）",
    )
    parser.add_argument(
        "--min-parity-symbols",
        type=int,
        default=None,
        help="最小冗余符号数下限（默认读取 config.json 的 ecc.min_parity_symbols）",
    )
    return parser


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    args = build_parser().parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"未找到输入目录：{args.input_dir}")

    channel_defaults = load_shared_rs_channel_defaults()

    if args.k is not None:
        k = args.k
        design_info = "k 由命令行直接指定"
    else:
        # 若明确给了旧参数 damage-rate，保留旧行为。
        if args.damage_rate is not None:
            damage_rate = args.damage_rate
            k = design_k_from_damage_rate(args.n, damage_rate)
            design_info = f"旧模型：仅擦除，目标损坏率={damage_rate:.2%}"
        else:
            erasure_rate = (
                args.erasure_rate
                if args.erasure_rate is not None
                else channel_defaults["erasure_rate"]
            )
            random_error_rate = (
                args.random_error_rate
                if args.random_error_rate is not None
                else channel_defaults["random_error_rate"]
            )
            safety_factor = (
                args.safety_factor
                if args.safety_factor is not None
                else channel_defaults["safety_factor"]
            )
            min_parity_symbols = (
                args.min_parity_symbols
                if args.min_parity_symbols is not None
                else channel_defaults["min_parity_symbols"]
            )

            k = design_k_for_channel(
                args.n,
                erasure_rate,
                random_error_rate,
                safety_factor,
                min_parity_symbols,
            )
            design_info = (
                "联合模型："
                f"erasure={erasure_rate:.2%}, random_error={random_error_rate:.2%}, "
                f"safety={safety_factor:.2f}"
            )

    cfg = RS11Config(n=args.n, k=k)
    rs = RS11(cfg)

    files = list(iter_bin_files(args.input_dir))
    if not files:
        raise SystemExit(f"在 {args.input_dir} 中未找到 .bin 文件")

    print(
        f"使用 RS({cfg.n},{cfg.k})，数据元宽度 11bit。"
        f"冗余占比={(cfg.n - cfg.k) / cfg.n:.2%}，最大可恢复擦除率={(cfg.n - cfg.k) / cfg.n:.2%}。"
    )
    print(f"参数设计：{design_info}")

    for src in files:
        dst = args.output_dir / f"{src.stem}.encoded.bin"
        encode_file(src, dst, rs)
        print(f"已编码：{src} -> {dst}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)
