#!/usr/bin/env python3
"""用于解码 ECC_encode.py 生成的 RS(11-bit) 文件。

默认会进行 syndrome 校验并解码无损码字。
若已知擦除位置，可通过 --erasures-json 在每个 RS 分组中恢复最多 n-k 个擦除。
编码器默认按 5% 数据元损坏率进行参数设计。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional

from rs11_core import RS11, RS11Config, symbols11_to_bytes, unpack_11bit_symbols

MAGIC = b"RS11ECC1"
HEADER_FMT = "<8sHHBQQBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PAYLOAD_FMT_U16 = 0
PAYLOAD_FMT_PACKED11 = 1


def auto_find_erasures_json(input_path: Path) -> Optional[Path]:
    """自动查找与输入文件同目录下的擦除位置文件。"""
    parent = input_path.parent
    base = input_path.stem.split(".")[0]
    candidates = sorted(parent.glob(f"{base}.erasures*.json"))
    if candidates:
        return candidates[0]
    return None


def default_check_output_path(output_path: Path) -> Path:
    """根据输出文件名生成默认校验文件路径。"""
    if output_path.suffix.lower() == ".bin":
        return output_path.with_name(f"{output_path.stem}.check.bin")
    return output_path.with_name(f"{output_path.name}.check.bin")


def default_block_status_output_path(output_path: Path) -> Path:
    """默认块状态输出路径。"""
    return output_path.parent / "block_status.txt"


def default_rs_usage_output_path(output_path: Path) -> Path:
    """默认 RS 冗余使用统计输出路径。"""
    return output_path.parent / "rs_usage.json"


def write_block_status_file(path: Path, block_status: List[int]) -> None:
    """写入块状态：每 40 个块换行，每 26 行空一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    row_count = 0
    for i in range(0, len(block_status), 40):
        chunk = block_status[i : i + 40]
        lines.append("".join("1" if v else "0" for v in chunk))
        row_count += 1
        if row_count % 26 == 0:
            lines.append("")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_erasures(path: Path | None) -> Dict[int, List[int]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, List[int]] = {}
    for k, v in data.items():
        out[int(k)] = [int(x) for x in v]
    return out


def write_rs_usage_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bit_validity_file(
    symbol_validity: List[int], pad_bits: int, expected_bytes: int
) -> bytes:
    """把数据元级有效性展开为按位校验文件（1=可确认正确，0=无法还原）。"""
    if pad_bits < 0 or pad_bits >= 11:
        raise ValueError("pad_bits 必须在 [0, 10] 区间内")

    total_bits = len(symbol_validity) * 11 - pad_bits
    if total_bits < 0:
        raise ValueError("总位数非法")

    out = bytearray()
    acc = 0
    acc_bits = 0
    emitted = 0

    for valid in symbol_validity:
        bit_value = 1 if valid else 0
        for _ in range(11):
            if emitted >= total_bits:
                break
            acc = (acc << 1) | bit_value
            acc_bits += 1
            emitted += 1
            if acc_bits == 8:
                out.append(acc)
                acc = 0
                acc_bits = 0

    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)

    # 按原始字节数裁剪，防止异常输入造成长度不一致。
    return bytes(out[:expected_bytes])


def decode_file(
    src: Path,
    dst: Path,
    erasures_map: Dict[int, List[int]],
    check_output: Optional[Path] = None,
    block_status_output: Optional[Path] = None,
    rs_usage_output: Optional[Path] = None,
    allow_partial: bool = False,
) -> dict:
    raw = src.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError("编码文件过小")

    magic, n, k, pad_bits, orig_bytes, total_data_symbols, payload_fmt, block_count = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError("文件魔数无效，不是 RS11 编码文件")

    rs = RS11(RS11Config(n=n, k=k))
    nsym = n - k
    payload = raw[HEADER_SIZE:]
    if total_data_symbols == 0:
        expected_blocks = 0
    else:
        expected_blocks = (total_data_symbols + k - 1) // k
    if block_count != expected_blocks:
        raise ValueError(
            f"分组数量无效：实际为 {block_count}，预期为 {expected_blocks}"
        )

    block_symbol_counts: List[int] = []
    for block_idx in range(block_count):
        if block_idx < block_count - 1:
            data_len = k
        else:
            data_len = total_data_symbols - (block_count - 1) * k
            if data_len == 0:
                data_len = k
        if data_len <= 0 or data_len > k:
            raise ValueError(f"分组 {block_idx} 的数据元长度无效：{data_len}")
        block_symbol_counts.append(data_len + nsym)

    if payload_fmt == PAYLOAD_FMT_PACKED11:
        expected_payload_len = sum(math.ceil(sym_count * 11 / 8) for sym_count in block_symbol_counts)
    elif payload_fmt == PAYLOAD_FMT_U16:
        expected_payload_len = sum(sym_count * 2 for sym_count in block_symbol_counts)
    else:
        raise ValueError(f"未知的负载格式标记：{payload_fmt}")

    if len(payload) != expected_payload_len:
        raise ValueError(
            f"负载大小无效：实际为 {len(payload)}，预期为 {expected_payload_len}"
        )

    data_symbols: List[int] = []
    symbol_validity: List[int] = []
    block_status: List[int] = []
    bad_blocks = 0
    rs_ea_fixed_blocks = 0
    usage_rows: List[dict] = []
    offset = 0
    for block_idx in range(block_count):
        block_n = block_symbol_counts[block_idx]
        data_len = block_n - nsym
        shorten = k - data_len

        if payload_fmt == PAYLOAD_FMT_PACKED11:
            block_bytes = math.ceil(block_n * 11 / 8)
        else:
            block_bytes = block_n * 2

        block_data = payload[offset : offset + block_bytes]
        offset += block_bytes

        if payload_fmt == PAYLOAD_FMT_PACKED11:
            cw = unpack_11bit_symbols(block_data, block_n)
        else:
            cw = list(struct.unpack(f"<{block_n}H", block_data))

        full_cw = ([0] * shorten) + cw
        block_recovered = True

        erasures = erasures_map.get(block_idx, [])
        if any(pos < 0 or pos >= block_n for pos in erasures):
            raise ValueError(f"分组 {block_idx} 的擦除位置超出范围")
        shifted_erasures = [pos + shorten for pos in erasures]

        syn = rs.syndromes(full_cw)
        usage_item = {
            "block": block_idx,
            "symbols": block_n,
            "data_symbols": data_len,
            "erasures": len(shifted_erasures),
            "unknown_errors": 0,
            "budget_used": len(shifted_erasures),
            "budget_total": nsym,
            "budget_ratio": (len(shifted_erasures) / nsym) if nsym > 0 else 0.0,
            "status": 1,
        }
        if any(syn):
            try:
                full_cw, stats = rs.correct_errors_and_erasures_with_stats(full_cw, shifted_erasures)
                usage_item["unknown_errors"] = int(stats.get("unknown_errors", 0))
                usage_item["budget_used"] = int(stats.get("budget_used", usage_item["budget_used"]))
                usage_item["budget_total"] = int(stats.get("budget_total", nsym))
                bt = usage_item["budget_total"]
                usage_item["budget_ratio"] = (usage_item["budget_used"] / bt) if bt > 0 else 0.0
                rs_ea_fixed_blocks += 1
            except Exception:
                if not allow_partial:
                    if erasures:
                        raise
                    raise ValueError(
                        f"分组 {block_idx} 的 syndrome 非零；如需恢复擦除，请传入 --erasures-json"
                    )
                block_recovered = False
                bad_blocks += 1
                usage_item["status"] = 0

        if block_recovered:
            cw = full_cw[shorten:]
            data_symbols.extend(cw[:data_len])
            symbol_validity.extend([1] * data_len)
            block_status.append(1)
        else:
            # 不可恢复时仍输出占位数据，并将对应位全部标记为 0。
            data_symbols.extend([0] * data_len)
            symbol_validity.extend([0] * data_len)
            block_status.append(0)

        usage_rows.append(usage_item)

    data_symbols = data_symbols[:total_data_symbols]
    out_bytes = symbols11_to_bytes(data_symbols, pad_bits)
    out_bytes = out_bytes[:orig_bytes]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out_bytes)

    if check_output is not None:
        check_bytes = build_bit_validity_file(symbol_validity, pad_bits, orig_bytes)
        check_output.parent.mkdir(parents=True, exist_ok=True)
        check_output.write_bytes(check_bytes)

    if block_status_output is not None:
        write_block_status_file(block_status_output, block_status)

    recovered_blocks = sum(block_status)
    if usage_rows:
        budget_used_list = [int(x["budget_used"]) for x in usage_rows]
        budget_ratio_list = [float(x["budget_ratio"]) for x in usage_rows]
        p95_index = max(0, min(len(budget_used_list) - 1, math.ceil(0.95 * len(budget_used_list)) - 1))
        p95_budget_used = sorted(budget_used_list)[p95_index]
        p95_budget_ratio = sorted(budget_ratio_list)[p95_index]
        max_budget_used = max(budget_used_list)
        max_budget_ratio = max(budget_ratio_list)
        avg_budget_used = sum(budget_used_list) / len(budget_used_list)
        avg_budget_ratio = sum(budget_ratio_list) / len(budget_ratio_list)
    else:
        p95_budget_used = 0
        p95_budget_ratio = 0.0
        max_budget_used = 0
        max_budget_ratio = 0.0
        avg_budget_used = 0.0
        avg_budget_ratio = 0.0

    usage_summary = {
        "n": n,
        "k": k,
        "nsym": nsym,
        "block_count": block_count,
        "recovered_blocks": recovered_blocks,
        "bad_blocks": bad_blocks,
        "max_budget_used": max_budget_used,
        "p95_budget_used": p95_budget_used,
        "avg_budget_used": avg_budget_used,
        "max_budget_ratio": max_budget_ratio,
        "p95_budget_ratio": p95_budget_ratio,
        "avg_budget_ratio": avg_budget_ratio,
        "blocks": usage_rows,
    }

    if rs_usage_output is not None:
        write_rs_usage_file(rs_usage_output, usage_summary)

    if bad_blocks > 0 and not allow_partial:
        raise ValueError(f"存在不可恢复分组：{bad_blocks}")

    if rs_ea_fixed_blocks > 0:
        print(f"已通过完整 errors-and-erasures 修复分组数：{rs_ea_fixed_blocks}")

    print(
        "RS 冗余使用："
        f"max={max_budget_used}/{nsym} ({max_budget_ratio:.2%}), "
        f"p95={p95_budget_used}/{nsym} ({p95_budget_ratio:.2%}), "
        f"avg={avg_budget_used:.2f}/{nsym} ({avg_budget_ratio:.2%}), "
        f"坏块={bad_blocks}/{block_count}"
    )

    return usage_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解码 RS(11-bit) .bin 文件")
    parser.add_argument("--input", type=Path, required=True, help="输入的编码 .bin 文件路径")
    parser.add_argument("--output", type=Path, required=True, help="输出的原始 .bin 文件路径")
    parser.add_argument(
        "--erasures-json",
        type=Path,
        default=None,
        help="可选 JSON 映射：{\"0\": [pos1,pos2], \"1\": [...]}，用于提供已知擦除位置",
    )
    parser.add_argument(
        "--check-output",
        type=Path,
        default=None,
        help="可选：按位校验文件输出路径（默认自动生成）",
    )
    parser.add_argument(
        "--block-status-output",
        type=Path,
        default=None,
        help="可选：块状态输出路径（默认: 输出目录/block_status.txt）",
    )
    parser.add_argument(
        "--rs-usage-output",
        type=Path,
        default=None,
        help="可选：RS 冗余使用统计输出 JSON 路径（默认: 输出目录/rs_usage.json）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许部分不可恢复分组继续解码（默认已启用）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：遇到不可恢复分组立即报错",
    )
    return parser


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    args = build_parser().parse_args()
    erasures_path = args.erasures_json
    if erasures_path is None:
        erasures_path = auto_find_erasures_json(args.input)
        if erasures_path is not None:
            print(f"已自动加载擦除位置文件：{erasures_path}")

    erasures = parse_erasures(erasures_path)

    check_output = args.check_output if args.check_output is not None else default_check_output_path(args.output)
    block_status_output = (
        args.block_status_output
        if args.block_status_output is not None
        else default_block_status_output_path(args.output)
    )
    rs_usage_output = (
        args.rs_usage_output
        if args.rs_usage_output is not None
        else default_rs_usage_output_path(args.output)
    )
    allow_partial = False if args.strict else True

    decode_file(
        args.input,
        args.output,
        erasures,
        check_output,
        block_status_output,
        rs_usage_output,
        allow_partial,
    )
    print(f"已解码：{args.input} -> {args.output}")
    print(f"已生成校验文件：{check_output}")
    print(f"已生成块状态文件：{block_status_output}")
    print(f"已生成冗余使用统计：{rs_usage_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)
