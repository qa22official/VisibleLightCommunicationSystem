#!/usr/bin/env python3
"""基于视频特征清理 frames 中的无效帧。"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 在导入 paddleocr 前关闭模型源连通性检查，避免启动时网络探测。
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "错误：需要安装 opencv-python 才能运行 main_decode.py。"
    ) from exc

try:
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "错误：需要安装 paddleocr 才能运行 main_decode.py 的 OCR 步骤。"
    ) from exc


@dataclass
class FrameFeature:
    path: Path
    mean_luma: float


def run_cmd(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "命令执行失败:\n"
            + " ".join(cmd)
            + "\n\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    return result.stdout


def load_ocr_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"OCR 配置文件不存在: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_model_dir(base_dir: Path, path_value: str) -> str:
    model_path = Path(path_value)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"OCR 模型目录不存在: {model_path}")
    return str(model_path)


def init_ocr_from_config(root_dir: Path, config: dict) -> PaddleOCR:
    ocr_dir = root_dir / "OCR"
    det_model_dir = resolve_model_dir(ocr_dir, str(config.get("text_detection_model_dir", "")))
    rec_model_dir = resolve_model_dir(ocr_dir, str(config.get("text_recognition_model_dir", "")))

    if bool(config.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", False)):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    kwargs = dict(
        use_doc_orientation_classify=bool(config.get("use_doc_orientation_classify", False)),
        use_doc_unwarping=bool(config.get("use_doc_unwarping", False)),
        use_textline_orientation=bool(config.get("use_textline_orientation", False)),
        device=str(config.get("device", "cpu")),
        text_detection_model_dir=det_model_dir,
        text_recognition_model_dir=rec_model_dir,
        text_det_thresh=float(config.get("thresh", 0.3)),
        text_det_box_thresh=float(config.get("box_thresh", 0.5)),
        text_det_unclip_ratio=float(config.get("unclip_ratio", 2.0)),
        char_detection=bool(config.get("char_detection", True)),
    )
    try:
        return PaddleOCR(**kwargs)
    except (TypeError, ValueError) as exc:
        # 兼容不支持 char_detection 构造参数的 paddleocr 版本。
        if "char_detection" not in str(exc):
            raise
        kwargs.pop("char_detection", None)
        return PaddleOCR(**kwargs)


def extract_ocr_items(
    result: object,
) -> list[tuple[str, float | None, list[list[float]] | None, list[str] | None, object]]:
    """提取 OCR 条目：(text, score, polygon_points, char_texts, char_boxes)。"""
    items: list[tuple[str, float | None, list[list[float]] | None, list[str] | None, object]] = []
    if not result:
        return items

    for block in result:
        if isinstance(block, dict):
            rec_texts = block.get("rec_texts", [])
            rec_scores = block.get("rec_scores", [])
            dt_polys = block.get("dt_polys", [])
            word_texts = block.get("text_word", [])
            word_boxes = block.get("text_word_boxes", [])
            for i, text in enumerate(rec_texts):
                score = None
                if isinstance(rec_scores, list) and i < len(rec_scores):
                    try:
                        score = float(rec_scores[i])
                    except Exception:
                        score = None

                points = None
                if isinstance(dt_polys, list) and i < len(dt_polys):
                    poly = dt_polys[i]
                    try:
                        points = [[float(p[0]), float(p[1])] for p in poly]
                    except Exception:
                        points = None

                chars = None
                if isinstance(word_texts, list) and i < len(word_texts):
                    try:
                        chars = [str(ch) for ch in word_texts[i]]
                    except Exception:
                        chars = None

                cboxes = None
                if isinstance(word_boxes, list) and i < len(word_boxes):
                    cboxes = word_boxes[i]

                items.append((str(text), score, points, chars, cboxes))
            continue

        if isinstance(block, list):
            for item in block:
                if isinstance(item, list) and len(item) >= 2:
                    poly_obj = item[0]
                    rec = item[1]
                    if isinstance(rec, tuple) and len(rec) == 2:
                        text, score_obj = rec
                        score = None
                        try:
                            score = float(score_obj)
                        except Exception:
                            score = None

                        points = None
                        try:
                            points = [[float(p[0]), float(p[1])] for p in poly_obj]
                        except Exception:
                            points = None

                        items.append((str(text), score, points, None, None))

    return items


def write_ocr_text_only(
    path: Path,
    items: list[tuple[str, float | None, list[list[float]] | None, list[str] | None, object]],
) -> None:
    texts = [text.replace(" ", "").replace("\u3000", "") for text, _score, _points, _chars, _cboxes in items if text]
    path.write_text("\n".join(texts) + ("\n" if texts else ""), encoding="utf-8")


def write_ocr_detail(
    path: Path,
    items: list[tuple[str, float | None, list[list[float]] | None, list[str] | None, object]],
) -> None:
    lines: list[str] = []
    for idx, (text, score, points, chars, cboxes) in enumerate(items, start=1):
        text = text.replace(" ", "").replace("\u3000", "")
        score_str = "N/A" if score is None else f"{score:.6f}"
        if points is None:
            point_str = "N/A"
        else:
            point_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in points)

        char_info = "N/A"
        if chars and cboxes is not None:
            pairs: list[str] = []
            for ci, ch in enumerate(chars):
                box_obj = cboxes[ci] if ci < len(cboxes) else None
                if box_obj is None:
                    pairs.append(f"{ch}:N/A")
                    continue

                try:
                    pts = [[float(p[0]), float(p[1])] for p in box_obj]
                    pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)
                    pairs.append(f"{ch}:{pts_str}")
                except Exception:
                    try:
                        vals = [float(v) for v in box_obj]
                        pairs.append(f"{ch}:{','.join(f'{v:.2f}' for v in vals)}")
                    except Exception:
                        pairs.append(f"{ch}:N/A")

            if pairs:
                char_info = "|".join(pairs)

        lines.append(
            f"index={idx}\ttext={text}\tconfidence={score_str}\tpoints={point_str}\tchar_boxes={char_info}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def infer_missing_positions_from_output_texts(
    ocr_output_dir: Path,
    ocr_detail_output_dir: Path,
    rows_per_frame: int,
    cols_per_row: int,
) -> list[tuple[int, int, int]]:
    """按 OCR/output 每帧文本行长度推断缺字位置，返回 (frame,row,col)（1-based）。"""
    if rows_per_frame <= 0 or cols_per_row <= 0:
        return []

    txt_files = sorted(p for p in ocr_output_dir.glob("*.txt") if p.is_file())
    if not txt_files:
        return []

    rows: list[tuple[int, int, int]] = []
    total_frames = len(txt_files)

    def parse_char_centers(detail_path: Path) -> dict[int, list[float]]:
        centers_by_row: dict[int, list[float]] = {}
        if not detail_path.exists():
            return centers_by_row

        lines = detail_path.read_text(encoding="utf-8").splitlines()
        for row_idx, line in enumerate(lines, start=1):
            key = "char_boxes="
            pos = line.find(key)
            if pos < 0:
                continue

            raw = line[pos + len(key) :].strip()
            if not raw or raw == "N/A":
                continue

            xs: list[float] = []
            for token in raw.split("|"):
                if ":" not in token:
                    continue
                _ch, box = token.split(":", 1)
                box = box.strip()
                if not box or box == "N/A":
                    continue

                try:
                    nums = [float(v) for v in box.split(",")]
                    if len(nums) >= 4:
                        xs.append((nums[0] + nums[2]) / 2.0)
                except Exception:
                    continue

            if xs:
                centers_by_row[row_idx] = sorted(xs)

        return centers_by_row

    def infer_missing_cols_by_centers(
        centers: list[float],
        char_count: int,
        cols: int,
        no_tail: bool,
    ) -> list[int]:
        if char_count <= 0 or cols <= 0:
            return []
        miss_count = max(0, cols - char_count)
        if miss_count == 0:
            return []
        if len(centers) < 2:
            return []

        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        base_gaps = sorted(g for g in gaps if g > 1e-6)
        if not base_gaps:
            return []

        # 使用较小一半间隔估计正常字距，避免缺字大间隔干扰。
        small_half = base_gaps[: max(1, len(base_gaps) // 2)]
        step = statistics.median(small_half)
        if step <= 1e-6:
            return []

        gap_missing: list[int] = []
        for g in gaps:
            m = int(round(g / step)) - 1
            gap_missing.append(max(0, m))

        total_internal = sum(gap_missing)
        if total_internal > miss_count:
            # 内部缺字总量超过上限时，按顺序截断到 miss_count。
            remain = miss_count
            clipped: list[int] = []
            for m in gap_missing:
                keep = min(m, remain)
                clipped.append(keep)
                remain -= keep
            gap_missing = clipped
            total_internal = sum(gap_missing)

        missing_cols: list[int] = []
        col = 1
        for i in range(char_count):
            col += 1
            if i < len(gap_missing):
                for _ in range(gap_missing[i]):
                    missing_cols.append(col)
                    col += 1

        tail_missing = miss_count - total_internal
        if not no_tail and tail_missing > 0:
            for _ in range(tail_missing):
                missing_cols.append(col)
                col += 1

        return [c for c in missing_cols if 1 <= c <= cols]

    for order, txt_path in enumerate(txt_files, start=1):
        try:
            frame_no = int(txt_path.stem)
        except ValueError:
            frame_no = order

        detail_path = ocr_detail_output_dir / txt_path.name
        centers_by_row = parse_char_centers(detail_path)

        content = txt_path.read_text(encoding="utf-8")
        line_texts = [ln.replace(" ", "").replace("\u3000", "") for ln in content.splitlines()]
        frame_rows = [line_texts[r - 1] if r - 1 < len(line_texts) else "" for r in range(1, rows_per_frame + 1)]

        # 空白帧不进行缺字推算。
        if all(row_text == "" for row_text in frame_rows):
            continue

        first_blank_row: int | None = None
        for r, row_text in enumerate(frame_rows, start=1):
            if row_text == "":
                first_blank_row = r
                break

        is_last_frame = order == total_frames

        for row in range(1, rows_per_frame + 1):
            # 最后一帧最后一行不参与缺字推算。
            if is_last_frame and row == rows_per_frame:
                continue

            line = frame_rows[row - 1]
            # 空白行不进行缺字推算。
            if line == "":
                continue

            char_count = len(line)
            if char_count >= cols_per_row:
                continue

            miss_count = cols_per_row - char_count
            # 第一个空白行的前一行：仅做中部缺字推算，不做尾部缺字推算。
            if first_blank_row is not None and row == first_blank_row - 1:
                inferred = infer_missing_cols_by_centers(
                    centers_by_row.get(row, []),
                    char_count=char_count,
                    cols=cols_per_row,
                    no_tail=True,
                )
                if inferred:
                    for col in inferred:
                        rows.append((frame_no, row, col))
                else:
                    # 该特殊行必须依赖坐标定位；若无可用坐标则跳过，避免尾部误判。
                    continue
            else:
                inferred = infer_missing_cols_by_centers(
                    centers_by_row.get(row, []),
                    char_count=char_count,
                    cols=cols_per_row,
                    no_tail=False,
                )
                if inferred:
                    for col in inferred:
                        rows.append((frame_no, row, col))
                else:
                    for col in range(char_count + 1, cols_per_row + 1):
                        rows.append((frame_no, row, col))

    return rows


def write_eraser_file(path: Path, rows: list[tuple[int, int, int]]) -> None:
    """写出缺失字符位置：frame,row,col（1-based）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame\trow\tcol"]
    for frame_no, row_no, col_no in rows:
        lines.append(f"{frame_no}\t{row_no}\t{col_no}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_eraser_file(path: Path) -> list[tuple[int, int, int]]:
    if not path.exists():
        return []
    rows: list[tuple[int, int, int]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line or line.startswith("frame"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def build_rs_erasures_json_from_eraser(
    encoded_bin_path: Path,
    eraser_rows: list[tuple[int, int, int]],
    rows_per_frame: int,
    cols_per_row: int,
    frames_order: list[int],
    out_json_path: Path,
) -> Path | None:
    """把 frame/row/col 位置映射为 RS 分组擦除参数（block -> symbol positions）。"""
    if not encoded_bin_path.exists() or not eraser_rows:
        return None

    MAGIC = b"RS11ECC1"
    HEADER_FMT = "<8sHHBQQBI"
    header_size = struct.calcsize(HEADER_FMT)

    raw = encoded_bin_path.read_bytes()
    if len(raw) < header_size:
        return None

    magic, n, k, pad_bits, orig_bytes, total_data_symbols, payload_fmt, block_count = struct.unpack(
        HEADER_FMT, raw[:header_size]
    )
    if magic != MAGIC:
        return None

    payload = raw[header_size:]
    nsym = n - k
    block_symbol_counts: list[int] = []
    for bi in range(block_count):
        if bi < block_count - 1:
            data_len = k
        else:
            data_len = total_data_symbols - (block_count - 1) * k
            if data_len == 0:
                data_len = k
        block_symbol_counts.append(data_len + nsym)

    if payload_fmt == 1:
        block_byte_sizes = [math.ceil(c * 11 / 8) for c in block_symbol_counts]
    elif payload_fmt == 0:
        block_byte_sizes = [c * 2 for c in block_symbol_counts]
    else:
        return None

    if sum(block_byte_sizes) > len(payload):
        return None

    frame_to_order = {frame_no: i + 1 for i, frame_no in enumerate(frames_order)}

    # 先把 frame/row/col 转为 OCR 符号序号（0-based）。
    symbol_indices: list[int] = []
    for frame_no, row_no, col_no in eraser_rows:
        order = frame_to_order.get(frame_no)
        if order is None:
            continue
        if not (1 <= row_no <= rows_per_frame and 1 <= col_no <= cols_per_row):
            continue
        idx = ((order - 1) * rows_per_frame + (row_no - 1)) * cols_per_row + (col_no - 1)
        symbol_indices.append(idx)

    if not symbol_indices:
        return None

    # OCR 符号对应整段 encoded.bin 的 11bit 分组，先映射到 payload 位域。
    payload_start_bit = header_size * 8
    payload_end_bit = payload_start_bit + len(payload) * 8

    erasures_map: dict[int, set[int]] = {}

    block_bit_cursor = 0
    block_ranges: list[tuple[int, int, int, int]] = []
    for bi, (sym_count, byte_size) in enumerate(zip(block_symbol_counts, block_byte_sizes)):
        bit_start = block_bit_cursor
        bit_end = bit_start + byte_size * 8
        valid_bits = sym_count * 11
        block_ranges.append((bi, bit_start, bit_end, valid_bits))
        block_bit_cursor = bit_end

    for sym_idx in symbol_indices:
        src_bit_start = sym_idx * 11
        src_bit_end = src_bit_start + 11
        ov_start = max(src_bit_start, payload_start_bit)
        ov_end = min(src_bit_end, payload_end_bit)
        if ov_end <= ov_start:
            continue

        payload_local_start = ov_start - payload_start_bit
        payload_local_end = ov_end - payload_start_bit

        for bi, blk_start, blk_end, valid_bits in block_ranges:
            bstart = max(payload_local_start, blk_start)
            bend = min(payload_local_end, blk_end)
            if bend <= bstart:
                continue

            local_start = bstart - blk_start
            local_end = bend - blk_start
            if local_start >= valid_bits:
                continue
            local_end = min(local_end, valid_bits)

            sym_start = local_start // 11
            sym_end = (local_end - 1) // 11
            s = erasures_map.setdefault(bi, set())
            for pos in range(sym_start, sym_end + 1):
                if 0 <= pos < block_symbol_counts[bi]:
                    s.add(pos)

    if not erasures_map:
        return None

    payload_json = {str(k): sorted(v) for k, v in sorted(erasures_map.items()) if v}
    if not payload_json:
        return None

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json_path


def ensure_ecc_encoded_bin_length(src_bin: Path) -> Path:
    """若 CH 解码产物负载长度小于头部预期，则在末尾补 0 后返回可用文件路径。"""
    MAGIC = b"RS11ECC1"
    HEADER_FMT = "<8sHHBQQBI"
    header_size = struct.calcsize(HEADER_FMT)

    raw = src_bin.read_bytes()
    if len(raw) < header_size:
        return src_bin

    try:
        magic, n, k, pad_bits, orig_bytes, total_data_symbols, payload_fmt, block_count = struct.unpack(
            HEADER_FMT, raw[:header_size]
        )
    except Exception:
        return src_bin

    if magic != MAGIC:
        return src_bin

    nsym = n - k
    block_symbol_counts: list[int] = []
    for bi in range(block_count):
        if bi < block_count - 1:
            data_len = k
        else:
            data_len = total_data_symbols - (block_count - 1) * k
            if data_len == 0:
                data_len = k
        block_symbol_counts.append(data_len + nsym)

    if payload_fmt == 1:
        expected_payload_len = sum(math.ceil(c * 11 / 8) for c in block_symbol_counts)
    elif payload_fmt == 0:
        expected_payload_len = sum(c * 2 for c in block_symbol_counts)
    else:
        return src_bin

    payload = raw[header_size:]
    if len(payload) >= expected_payload_len:
        return src_bin

    miss = expected_payload_len - len(payload)
    padded = raw + (b"\x00" * miss)
    out_path = src_bin.with_name(f"{src_bin.stem}.padded{src_bin.suffix}")
    out_path.write_bytes(padded)
    print(f"警告：CH 解码结果负载不足，已补齐 {miss} 字节后继续 ECC 解码 -> {out_path}")
    return out_path


def iter_bits(data: bytes):
    for b in data:
        for i in range(7, -1, -1):
            yield (b >> i) & 1


def evaluate_recovery_metrics(source_bin: Path, recovered_bin: Path, check_bin: Path) -> dict[str, float] | None:
    """评估恢复质量：
    1) 源文件与恢复文件按位匹配度；
    2) 不匹配位中，被校验文件标注(0)的占比；
    3) 校验文件多余标注(位其实匹配)在所有标注中的占比。
    """
    if not source_bin.exists() or not recovered_bin.exists() or not check_bin.exists():
        return None

    src = source_bin.read_bytes()
    rec = recovered_bin.read_bytes()
    chk = check_bin.read_bytes()

    src_bits = list(iter_bits(src))
    rec_bits = list(iter_bits(rec))
    chk_bits = list(iter_bits(chk))

    total_bits = max(len(src_bits), len(rec_bits))
    if total_bits == 0:
        return {
            "bit_match_rate": 1.0,
            "mismatch_marked_rate": 0.0,
            "over_mark_rate": 0.0,
            "total_bits": 0,
            "mismatch_bits": 0,
            "marked_bits": 0,
        }

    mismatch_bits = 0
    mismatch_marked_bits = 0
    marked_bits = 0
    over_mark_bits = 0

    for i in range(total_bits):
        sb = src_bits[i] if i < len(src_bits) else 0
        rb = rec_bits[i] if i < len(rec_bits) else 0
        mismatch = sb != rb

        # check 位语义：1=可确认正确，0=被标注为不可确认/错误。
        cb = chk_bits[i] if i < len(chk_bits) else 1
        marked = cb == 0

        if mismatch:
            mismatch_bits += 1
            if marked:
                mismatch_marked_bits += 1

        if marked:
            marked_bits += 1
            if not mismatch:
                over_mark_bits += 1

    matched_bits = total_bits - mismatch_bits
    bit_match_rate = matched_bits / total_bits
    mismatch_marked_rate = (mismatch_marked_bits / mismatch_bits) if mismatch_bits > 0 else 0.0
    over_mark_rate = (over_mark_bits / marked_bits) if marked_bits > 0 else 0.0

    return {
        "bit_match_rate": bit_match_rate,
        "mismatch_marked_rate": mismatch_marked_rate,
        "over_mark_rate": over_mark_rate,
        "total_bits": float(total_bits),
        "mismatch_bits": float(mismatch_bits),
        "marked_bits": float(marked_bits),
    }


def build_ecc_tuning_suggestion(config_path: Path, rs_usage_json: Path) -> dict | None:
    if not config_path.exists() or not rs_usage_json.exists():
        return None

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    usage = json.loads(rs_usage_json.read_text(encoding="utf-8"))
    ecc_cfg = cfg.get("ecc", {})

    n = int(usage.get("n", 0))
    nsym = int(usage.get("nsym", 0))
    p95_budget_used = float(usage.get("p95_budget_used", 0.0))
    max_budget_used = float(usage.get("max_budget_used", 0.0))
    max_budget_ratio = float(usage.get("max_budget_ratio", 0.0))
    bad_blocks = int(usage.get("bad_blocks", 0))
    block_count = int(usage.get("block_count", 0))

    if n <= 0 or nsym <= 0:
        return None

    cur_erasure_rate = float(ecc_cfg.get("erasure_rate", ecc_cfg.get("rs_redundancy_rate", 0.05)))
    cur_random_error_rate = float(ecc_cfg.get("random_error_rate", 0.003))
    cur_safety = float(ecc_cfg.get("design_safety_factor", 1.25))
    cur_min_parity = int(ecc_cfg.get("min_parity_symbols", 6))

    target_budget = max(p95_budget_used, max_budget_used)
    margin = 1.10 if bad_blocks == 0 else 1.30
    target_budget_with_margin = min(float(n), max(2.0, target_budget * margin))

    # 当当前样本非常稳（无坏块且峰值使用率低）时，鼓励下调冗余；
    # 否则保持更保守的安全系数。
    if bad_blocks == 0 and max_budget_ratio < 0.60:
        sugg_safety = 1.15
    elif bad_blocks == 0:
        sugg_safety = max(1.15, min(1.35, cur_safety))
    else:
        sugg_safety = max(1.25, min(1.60, cur_safety + 0.10))

    sugg_min_parity = max(2, int(math.ceil(target_budget_with_margin)))
    sugg_rs_redundancy_rate = max(0.001, min(0.95, sugg_min_parity / n))

    # 使 erasure+2*random 与建议冗余保持一致：
    # parity_rate ~= (erasure + 2*random) * safety_factor。
    combined_needed = sugg_rs_redundancy_rate / max(1.0, sugg_safety)
    current_combined_rate = max(1e-6, cur_erasure_rate + 2.0 * cur_random_error_rate)
    erasure_weight = cur_erasure_rate / current_combined_rate
    random_weight = (2.0 * cur_random_error_rate) / current_combined_rate

    sugg_erasure_rate = max(0.0, min(0.95, combined_needed * erasure_weight))
    sugg_random_error_rate = max(0.0, min(0.20, (combined_needed * random_weight) / 2.0))

    return {
        "observed": {
            "n": n,
            "nsym": nsym,
            "bad_blocks": bad_blocks,
            "block_count": block_count,
            "p95_budget_used": p95_budget_used,
            "max_budget_used": max_budget_used,
        },
        "current": {
            "erasure_rate": cur_erasure_rate,
            "random_error_rate": cur_random_error_rate,
            "design_safety_factor": cur_safety,
            "min_parity_symbols": cur_min_parity,
            "rs_redundancy_rate": float(ecc_cfg.get("rs_redundancy_rate", nsym / n)),
        },
        "suggested": {
            "erasure_rate": sugg_erasure_rate,
            "random_error_rate": sugg_random_error_rate,
            "design_safety_factor": sugg_safety,
            "min_parity_symbols": sugg_min_parity,
            "rs_redundancy_rate": sugg_rs_redundancy_rate,
            "expected_damage_rate": sugg_rs_redundancy_rate,
        },
    }


def run_ocr_on_frames(
    root_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    detail_output_dir: Path,
    ocr_config_path: Path,
) -> tuple[int, int]:
    config = load_ocr_config(ocr_config_path)
    ocr = init_ocr_from_config(root_dir, config)
    return_word_box = bool(config.get("char_detection", True))

    allowed_ext = {str(ext).lower() for ext in config.get("image_formats", [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"])}
    frame_files = sorted(
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext
    )

    if not frame_files:
        print(f"未在 {frames_dir} 中找到可 OCR 的图像帧。")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.txt"):
        old.unlink(missing_ok=True)
    for old in detail_output_dir.glob("*.txt"):
        old.unlink(missing_ok=True)

    total = len(frame_files)
    success = 0
    for idx, frame in enumerate(frame_files, start=1):
        out_txt = output_dir / f"{frame.stem}.txt"
        out_detail = detail_output_dir / f"{frame.stem}.txt"
        try:
            result = ocr.predict(str(frame), return_word_box=return_word_box)
            items = extract_ocr_items(result)
            write_ocr_text_only(out_txt, items)
            write_ocr_detail(out_detail, items)

            success += 1
            print(f"OCR 进度: {idx}/{total}，已处理 {frame.name}（成功）", flush=True)
        except Exception as exc:
            print(f"OCR 失败: {frame.name} -> {exc}")
            out_txt.write_text("", encoding="utf-8")
            out_detail.write_text("", encoding="utf-8")
            print(f"OCR 进度: {idx}/{total}，已处理 {frame.name}（失败）", flush=True)

    return total, success


def load_video_grid(root_dir: Path) -> tuple[int, int]:
    cfg_path = root_dir / "config.json"
    if not cfg_path.exists():
        return 26, 40
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    video = data.get("video", {})
    rows = int(video.get("rows_per_frame", 26))
    cols = int(video.get("cols_per_row", 40))
    if rows <= 0:
        rows = 26
    if cols <= 0:
        cols = 40
    return rows, cols


def run_ffmpeg(video_path: Path, output_dir: Path) -> None:
    """调用 ffmpeg 将视频拆帧到 frames 目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先清空旧帧，避免新视频帧数更少时残留脏数据。
    for p in [
        *output_dir.glob("*.png"),
        *output_dir.glob("*.jpg"),
        *output_dir.glob("*.jpeg"),
        *output_dir.glob("*.bmp"),
    ]:
        p.unlink(missing_ok=True)

    output_pattern = output_dir / "%06d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        str(output_pattern),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 拆帧失败: {video_path.name}\n"
            f"stderr:\n{result.stderr.strip()}"
        )


def sample_left_band_mean(path: Path, left_band_ratio: float) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")

    h, w = img.shape[:2]
    y0, y1 = int(h * 0.10), int(h * 0.90)
    band_w = max(4, int(w * left_band_ratio))
    left_band = img[y0:y1, :band_w]
    return float(np.mean(left_band))


def build_features(frame_files: list[Path], left_band_ratio: float) -> list[FrameFeature]:
    means = [sample_left_band_mean(p, left_band_ratio) for p in frame_files]
    return [FrameFeature(path=path, mean_luma=m) for path, m in zip(frame_files, means)]


def select_valid_indices(
    features: list[FrameFeature],
    peak_min_delta: float,
    peak_flat_tolerance: float,
) -> tuple[set[int], set[int]]:
    # 有效帧定义：左侧条带平均亮度相对前后帧都更亮或都更暗（局部极值）。
    n = len(features)
    if n == 0:
        return set(), set()
    if n < 3:
        return set(range(n)), set()

    candidates: list[tuple[int, int, float]] = []
    for i in range(1, n - 1):
        prev_m = features[i - 1].mean_luma
        cur_m = features[i].mean_luma
        next_m = features[i + 1].mean_luma

        # 允许平顶/平谷近似极值，降低拍摄噪声与压缩抖动导致的漏检。
        is_local_max = (
            cur_m >= prev_m - peak_flat_tolerance
            and cur_m >= next_m - peak_flat_tolerance
            and (cur_m > prev_m + peak_flat_tolerance or cur_m > next_m + peak_flat_tolerance)
        )
        is_local_min = (
            cur_m <= prev_m + peak_flat_tolerance
            and cur_m <= next_m + peak_flat_tolerance
            and (cur_m < prev_m - peak_flat_tolerance or cur_m < next_m - peak_flat_tolerance)
        )
        if not (is_local_max or is_local_min):
            continue

        d_prev = abs(cur_m - prev_m)
        d_next = abs(cur_m - next_m)
        if max(d_prev, d_next) < peak_min_delta:
            continue

        # type: +1=峰, -1=谷; strength 越大说明局部极值越可靠。
        extrema_type = 1 if is_local_max else -1
        strength = min(d_prev, d_next)
        candidates.append((i, extrema_type, strength))

    # 必须交替峰谷：若出现连续同类型候选，只保留强度更高的一个。
    selected: list[tuple[int, int, float]] = []
    for c in candidates:
        if not selected:
            selected.append(c)
            continue

        last_i, last_type, last_strength = selected[-1]
        cur_i, cur_type, cur_strength = c
        if cur_type != last_type:
            selected.append(c)
            continue

        if cur_strength > last_strength:
            selected[-1] = c

    valid = {idx for idx, _, _ in selected}

    # 若峰谷被压平导致未命中，保守回退为全保留，避免误删有效帧。
    if not valid:
        valid = set(range(n))

    invalid = {i for i in range(n) if i not in valid}
    return valid, invalid


def cleanup_frames(
    frames_dir: Path,
    left_band_ratio: float,
    peak_min_delta: float,
    peak_flat_tolerance: float,
    dry_run: bool,
    backup_dir: Path | None,
) -> tuple[int, int]:
    frame_files = sorted(
        [
            *frames_dir.glob("*.png"),
            *frames_dir.glob("*.jpg"),
            *frames_dir.glob("*.jpeg"),
            *frames_dir.glob("*.bmp"),
        ]
    )
    if not frame_files:
        print(f"未在 {frames_dir} 中找到图像帧。")
        return 0, 0

    features = build_features(frame_files, left_band_ratio)
    means = [ft.mean_luma for ft in features]
    span = float(np.percentile(means, 90) - np.percentile(means, 10))
    adaptive_delta = max(0.5, span * 0.015)
    delta = peak_min_delta if peak_min_delta > 0 else adaptive_delta
    keep_set, invalid = select_valid_indices(features, delta, peak_flat_tolerance)

    valid_count = len(keep_set)
    invalid_count = len(invalid)

    if dry_run:
        print("[Dry Run] 仅统计，不删除文件。")
        return valid_count, invalid_count

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for i in sorted(invalid):
        src = features[i].path
        if backup_dir is not None:
            dst = backup_dir / src.name
            shutil.move(str(src), str(dst))
        else:
            src.unlink(missing_ok=True)

    return valid_count, invalid_count


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="清理 frames 中的重复帧、重叠帧和多余帧")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=root / "frames",
        help="帧目录（默认: frames）",
    )
    parser.add_argument(
        "--left-band-ratio",
        type=float,
        default=0.05,
        help="左侧条带采样宽度占比（默认: 0.05）",
    )
    parser.add_argument(
        "--peak-min-delta",
        type=float,
        default=0.0,
        help="局部极值与前后帧最小亮度差；<=0 时自动估计（默认: 0）",
    )
    parser.add_argument(
        "--peak-flat-tolerance",
        type=float,
        default=1.5,
        help="平顶/平谷近似极值容差（默认: 1.5）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计要删除的帧，不实际删除",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="可选：将无效帧移动到该目录，而不是直接删除",
    )
    parser.add_argument(
        "--ocr-config",
        type=Path,
        default=root / "OCR" / "ocr_config.json",
        help="OCR 配置文件路径（默认: OCR/ocr_config.json）",
    )
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        default=root / "OCR" / "output",
        help="OCR 文本输出目录（默认: OCR/output）",
    )
    parser.add_argument(
        "--ocr-detail-output-dir",
        type=Path,
        default=root / "OCR" / "output_detail",
        help="OCR 明细输出目录（坐标+置信度，默认: OCR/output_detail）",
    )
    parser.add_argument(
        "--eraser-output",
        type=Path,
        default=root / "ECC" / "eraser.txt",
        help="OCR 推断缺字位置输出（默认: ECC/eraser.txt）",
    )
    parser.add_argument(
        "--ch-db",
        type=Path,
        default=root / "CH" / "dictionary.db",
        help="CH 解码字典路径（默认: CH/dictionary.db）",
    )
    parser.add_argument(
        "--ch-decode-output-dir",
        type=Path,
        default=root / "CH" / "decode_output",
        help="CH 解码输出目录（默认: CH/decode_output）",
    )
    parser.add_argument(
        "--ch-output-name",
        type=str,
        default="ocr_merged.bin",
        help="CH 解码输出文件名（默认: ocr_merged.bin）",
    )
    parser.add_argument(
        "--ecc-decode-output-dir",
        type=Path,
        default=root / "ECC" / "decode_output",
        help="ECC 解码输出目录（默认: ECC/decode_output）",
    )
    parser.add_argument(
        "--ecc-output-name",
        type=str,
        default="recovered.bin",
        help="ECC 恢复输出文件名（默认: recovered.bin）",
    )
    parser.add_argument(
        "--source-bin",
        type=Path,
        default=root / "test" / "ceshi.bin",
        help="用于质量评估的源文件路径（默认: test/ceshi.bin）",
    )
    return parser


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    args = build_parser().parse_args()
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        print("错误：未检测到 ffmpeg，请先安装并加入 PATH。", file=sys.stderr)
        return 1

    videos = sorted(p for p in root_dir.glob("*.mp4") if p.is_file())
    if len(videos) > 1:
        print("检测到多个 .mp4 文件。默认仅允许存在一个视频，请只保留一个后再运行。", file=sys.stderr)
        return 1
    if len(videos) == 1:
        video = videos[0]
        print(f"检测到视频：{video.name}，开始拆帧...")
        run_ffmpeg(video, args.frames_dir)
        print(f"拆帧完成：{args.frames_dir}")
    else:
        print(f"未在根目录找到 .mp4 文件，跳过拆帧，仅处理已有帧：{args.frames_dir}")

    valid_count, invalid_count = cleanup_frames(
        frames_dir=args.frames_dir,
        left_band_ratio=args.left_band_ratio,
        peak_min_delta=args.peak_min_delta,
        peak_flat_tolerance=args.peak_flat_tolerance,
        dry_run=args.dry_run,
        backup_dir=args.backup_dir,
    )

    print(f"有效帧数量: {valid_count}")
    print(f"无效帧数量: {invalid_count}")
    if args.dry_run:
        print("未执行删除。")
        print("Dry Run 模式下跳过 OCR。")
    elif args.backup_dir is not None:
        print(f"无效帧已移动到: {args.backup_dir}")
    else:
        print("无效帧已从 frames 中删除。")

    if not args.dry_run:
        rows_per_frame, cols_per_row = load_video_grid(root_dir)
        total, success = run_ocr_on_frames(
            root_dir=root_dir,
            frames_dir=args.frames_dir,
            output_dir=args.ocr_output_dir,
            detail_output_dir=args.ocr_detail_output_dir,
            ocr_config_path=args.ocr_config,
        )
        eraser_rows = infer_missing_positions_from_output_texts(
            ocr_output_dir=args.ocr_output_dir,
            ocr_detail_output_dir=args.ocr_detail_output_dir,
            rows_per_frame=rows_per_frame,
            cols_per_row=cols_per_row,
        )
        write_eraser_file(args.eraser_output, eraser_rows)

        # 1) 调用 CH/CH_decode.py：按顺序读取 OCR/output 所有 txt 解码为一个 bin。
        run_cmd(
            [
                sys.executable,
                str(root_dir / "CH" / "CH_decode.py"),
                "--input-dir",
                str(args.ocr_output_dir),
                "--db",
                str(args.ch_db),
                "--output-dir",
                str(args.ch_decode_output_dir),
                "--output-name",
                args.ch_output_name,
                "--eraser-path",
                str(args.eraser_output),
                "--fill-unknown-zero",
                "--rows-per-frame",
                str(rows_per_frame),
                "--cols-per-row",
                str(cols_per_row),
            ],
            root_dir,
        )

        ch_bin = args.ch_decode_output_dir / args.ch_output_name
        ecc_input_bin = ensure_ecc_encoded_bin_length(ch_bin)

        # 2) 把 eraser.txt 转换为 ECC 可用的 erasures-json。
        frames_sorted = sorted(
            int(p.stem) if p.stem.isdigit() else i + 1
            for i, p in enumerate(sorted(args.ocr_output_dir.glob("*.txt")))
        )
        merged_eraser_rows = parse_eraser_file(args.eraser_output)
        erasures_json_path = args.ecc_decode_output_dir / "erasures.json"
        erasures_json = build_rs_erasures_json_from_eraser(
            encoded_bin_path=ecc_input_bin,
            eraser_rows=merged_eraser_rows,
            rows_per_frame=rows_per_frame,
            cols_per_row=cols_per_row,
            frames_order=frames_sorted,
            out_json_path=erasures_json_path,
        )

        # 3) 调用 ECC/ECC_decode.py 恢复，并输出恢复文件与校验文件到 ECC/decode_output。
        args.ecc_decode_output_dir.mkdir(parents=True, exist_ok=True)
        ecc_out = args.ecc_decode_output_dir / args.ecc_output_name
        ecc_cmd = [
            sys.executable,
            str(root_dir / "ECC" / "ECC_decode.py"),
            "--input",
            str(ecc_input_bin),
            "--output",
            str(ecc_out),
            "--allow-partial",
        ]
        if erasures_json is not None:
            ecc_cmd.extend(["--erasures-json", str(erasures_json)])
        run_cmd(ecc_cmd, root_dir)

        print(f"OCR 输入图片数: {total}")
        print(f"OCR 成功输出数: {success}")
        print(f"OCR 文本目录: {args.ocr_output_dir}")
        print(f"OCR 明细目录: {args.ocr_detail_output_dir}")
        print(f"缺字位置文件: {args.eraser_output}")
        print(f"缺字位置条目数: {len(eraser_rows)}")
        print(f"CH 解码输出: {ch_bin}")
        if ecc_input_bin != ch_bin:
            print(f"ECC 实际输入: {ecc_input_bin}")
        if erasures_json is not None:
            print(f"RS 擦除参数: {erasures_json}")
        else:
            print("RS 擦除参数: 未生成（将按无擦除参数解码）")
        print(f"ECC 恢复输出: {ecc_out}")
        check_path = ecc_out.with_name(ecc_out.stem + '.check.bin')
        print(f"ECC 校验文件: {check_path}")
        rs_usage_path = args.ecc_decode_output_dir / "rs_usage.json"
        if rs_usage_path.exists():
            rs_usage = json.loads(rs_usage_path.read_text(encoding="utf-8"))
            nsym = int(rs_usage.get("nsym", 0))
            print("\n=== RS 冗余使用情况 ===")
            print(
                f"max 使用: {rs_usage.get('max_budget_used', 0)}/{nsym} "
                f"({float(rs_usage.get('max_budget_ratio', 0.0)):.2%})"
            )
            print(
                f"p95 使用: {rs_usage.get('p95_budget_used', 0)}/{nsym} "
                f"({float(rs_usage.get('p95_budget_ratio', 0.0)):.2%})"
            )
            print(
                f"avg 使用: {float(rs_usage.get('avg_budget_used', 0.0)):.2f}/{nsym} "
                f"({float(rs_usage.get('avg_budget_ratio', 0.0)):.2%})"
            )
            print(
                f"坏块: {rs_usage.get('bad_blocks', 0)}/{rs_usage.get('block_count', 0)}"
            )

            sugg = build_ecc_tuning_suggestion(root_dir / "config.json", rs_usage_path)
            if sugg is not None:
                cur = sugg["current"]
                rec = sugg["suggested"]
                print("\n=== ECC 参数调整建议（基于当前样本）===")
                print(
                    f"erasure_rate: {cur['erasure_rate']:.6f} -> {rec['erasure_rate']:.6f}"
                )
                print(
                    f"random_error_rate: {cur['random_error_rate']:.6f} -> {rec['random_error_rate']:.6f}"
                )
                print(
                    f"design_safety_factor: {cur['design_safety_factor']:.4f} -> {rec['design_safety_factor']:.4f}"
                )
                print(
                    f"min_parity_symbols: {int(cur['min_parity_symbols'])} -> {int(rec['min_parity_symbols'])}"
                )
                print(
                    f"rs_redundancy_rate: {cur['rs_redundancy_rate']:.6f} -> {rec['rs_redundancy_rate']:.6f}"
                )
        else:
            print("\nRS 冗余使用情况: 未找到 rs_usage.json，跳过建议计算")

        metrics = evaluate_recovery_metrics(args.source_bin, ecc_out, check_path)
        if metrics is None:
            print("质量评估: 跳过（源文件/恢复文件/校验文件缺失）")
        else:
            total_bits = int(metrics["total_bits"])
            mismatch_bits = int(metrics["mismatch_bits"])
            marked_bits = int(metrics["marked_bits"])
            print("\n=== ECC 按位质量评估 ===")
            print(f"源文件: {args.source_bin}")
            print(f"恢复文件: {ecc_out}")
            print(f"总位数: {total_bits}")
            print(f"按位匹配度: {metrics['bit_match_rate']:.4%}")
            print(
                "不匹配位被校验文件标注占比: "
                f"{metrics['mismatch_marked_rate']:.4%} "
                f"(不匹配位 {mismatch_bits})"
            )
            print(
                "校验文件多余标注占比: "
                f"{metrics['over_mark_rate']:.4%} "
                f"(标注位 {marked_bits})"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)
