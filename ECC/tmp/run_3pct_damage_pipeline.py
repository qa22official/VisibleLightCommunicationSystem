#!/usr/bin/env python3
"""在 ECC/tmp 下执行：编码 -> 3% 数据元损坏模拟。"""

from __future__ import annotations

import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
ECC_DIR = ROOT / "ECC"
if str(ECC_DIR) not in sys.path:
    sys.path.insert(0, str(ECC_DIR))

from ECC_encode import encode_file  # type: ignore
from rs11_core import RS11, RS11Config, pack_11bit_symbols, unpack_11bit_symbols  # type: ignore

MAGIC = b"RS11ECC1"
HEADER_FMT = "<8sHHBQQBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PAYLOAD_FMT_U16 = 0
PAYLOAD_FMT_PACKED11 = 1


def pick_first_bin(test_dir: Path) -> Path:
    files = sorted(p for p in test_dir.glob("*.bin") if p.is_file())
    if not files:
        raise FileNotFoundError(f"在 {test_dir} 中未找到 .bin 文件")
    return files[0]


def parse_blocks(total_data_symbols: int, block_count: int, k: int, nsym: int) -> List[int]:
    counts: List[int] = []
    for block_idx in range(block_count):
        if block_idx < block_count - 1:
            data_len = k
        else:
            data_len = total_data_symbols - (block_count - 1) * k
            if data_len == 0:
                data_len = k
        counts.append(data_len + nsym)
    return counts


def mutate_encoded_file(encoded_path: Path, damaged_path: Path, damage_rate: float, seed: int) -> Dict[int, List[int]]:
    raw = bytearray(encoded_path.read_bytes())
    if len(raw) < HEADER_SIZE:
        raise ValueError("编码文件过小")

    magic, n, k, _pad_bits, _orig_bytes, total_data_symbols, payload_fmt, block_count = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError("文件魔数无效，不是 RS11 编码文件")

    nsym = n - k
    block_symbol_counts = parse_blocks(total_data_symbols, block_count, k, nsym)
    payload = raw[HEADER_SIZE:]

    rng = random.Random(seed)
    erasures: Dict[int, List[int]] = {}
    mutated_payload = bytearray()

    offset = 0
    for block_idx, block_n in enumerate(block_symbol_counts):
        if payload_fmt == PAYLOAD_FMT_PACKED11:
            block_bytes = math.ceil(block_n * 11 / 8)
        elif payload_fmt == PAYLOAD_FMT_U16:
            block_bytes = block_n * 2
        else:
            raise ValueError(f"未知负载格式：{payload_fmt}")

        block_data = payload[offset : offset + block_bytes]
        offset += block_bytes

        if payload_fmt == PAYLOAD_FMT_PACKED11:
            symbols = unpack_11bit_symbols(bytes(block_data), block_n)
        else:
            symbols = list(struct.unpack(f"<{block_n}H", bytes(block_data)))

        e = max(1, math.ceil(block_n * damage_rate))
        e = min(e, nsym)
        positions = sorted(rng.sample(range(block_n), e))

        for pos in positions:
            delta = rng.randint(1, 2047)
            symbols[pos] ^= delta

        erasures[block_idx] = positions

        if payload_fmt == PAYLOAD_FMT_PACKED11:
            mutated_payload.extend(pack_11bit_symbols(symbols))
        else:
            mutated_payload.extend(struct.pack(f"<{block_n}H", *symbols))

    damaged = bytes(raw[:HEADER_SIZE]) + bytes(mutated_payload)
    damaged_path.write_bytes(damaged)

    return erasures


def main() -> int:
    test_dir = ROOT / "test"
    tmp_dir = ECC_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src = pick_first_bin(test_dir)

    cfg = RS11Config(n=255, k=242)
    rs = RS11(cfg)

    encoded_path = tmp_dir / f"{src.stem}.encoded.bin"
    damaged_path = tmp_dir / f"{src.stem}.damaged.3pct.bin"
    erasures_path = tmp_dir / f"{src.stem}.erasures.3pct.json"

    encode_file(src, encoded_path, rs)
    erasures = mutate_encoded_file(encoded_path, damaged_path, damage_rate=0.03, seed=20260319)
    erasures_path.write_text(json.dumps({str(k): v for k, v in erasures.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"源文件：{src}")
    print(f"编码文件：{encoded_path}")
    print(f"损坏文件：{damaged_path}")
    print(f"擦除位置：{erasures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
