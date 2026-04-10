#!/usr/bin/env python3
"""编码主程序：test/.bin -> ECC -> CH -> 文本视频。"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

try:
	from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
	raise SystemExit("错误：需要安装 Pillow 才能逐帧绘制文字。") from exc

FPS = 30
ROWS_PER_FRAME = 20
COLS_PER_ROW = 40
CHARS_PER_FRAME = ROWS_PER_FRAME * COLS_PER_ROW
MAX_TEXT_FRAMES = 30
LEAD_IN_SECONDS = 0.5
TAIL_OUT_SECONDS = 0.5
SIDE_BAND_WIDTH = 180
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FONT_SIZE = 0
FONT_PATH = ""
LINE_SPACING = 0
CHAR_SPACING = 0

ECC_HEADER_FMT = "<8sHHBQQBI"
ECC_HEADER_SIZE = struct.calcsize(ECC_HEADER_FMT)
ECC_SYMBOL_BITS = 11
ECC_DEFAULT_N = 255


def run_cmd(cmd: list[str], cwd: Path) -> None:
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


def find_single_file(input_dir: Path, pattern: str) -> Path:
	files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
	if not files:
		raise FileNotFoundError(f"在 {input_dir} 中未找到 {pattern} 文件")
	if len(files) > 1:
		names = ", ".join(p.name for p in files)
		raise ValueError(f"在 {input_dir} 中找到多个 {pattern} 文件: {names}")
	return files[0]


def split_text_chunks(text: str, chunk_size: int) -> list[str]:
	if not text:
		return [""]
	return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def wrap_chunk_for_ass(chunk: str) -> str:
	# 每帧固定 20 行，每行 40 个字符；不足部分使用全角空格补齐。
	padded = chunk.ljust(CHARS_PER_FRAME, "\u3000")
	lines = [padded[i : i + COLS_PER_ROW] for i in range(0, CHARS_PER_FRAME, COLS_PER_ROW)]

	escaped_lines: list[str] = []
	for line in lines[:ROWS_PER_FRAME]:
		escaped = line.replace("\\", r"\\")
		escaped = escaped.replace("{", r"\{").replace("}", r"\}")
		escaped_lines.append(escaped)
	return r"\N".join(escaped_lines)


def chunk_to_lines(chunk: str) -> list[str]:
	# 每帧固定 20 行、每行 40 字，不足部分全角空格补齐。
	padded = chunk.ljust(CHARS_PER_FRAME, "\u3000")
	return [padded[i : i + COLS_PER_ROW] for i in range(0, CHARS_PER_FRAME, COLS_PER_ROW)][:ROWS_PER_FRAME]


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
	font_candidates: list[Path] = []
	if FONT_PATH:
		font_candidates.append(Path(FONT_PATH))

	font_candidates.extend(
		[
			Path("~/.local/share/fonts/arial.ttf"),
		]
	)
	for fp in font_candidates:
		if fp.exists():
			try:
				return ImageFont.truetype(str(fp), size=max(1, size))
			except Exception:
				continue

	# 尽量回退到可缩放字体，避免固定像素字体导致字号不可调。
	try:
		return ImageFont.truetype("DejaVuSans.ttf", size=max(1, size))
	except Exception:
		pass

	return ImageFont.load_default()


def load_shared_config(root_dir: Path) -> None:
	global FPS, ROWS_PER_FRAME, COLS_PER_ROW, CHARS_PER_FRAME
	global LEAD_IN_SECONDS, TAIL_OUT_SECONDS, SIDE_BAND_WIDTH
	global FRAME_WIDTH, FRAME_HEIGHT, FONT_SIZE, FONT_PATH, LINE_SPACING, CHAR_SPACING, MAX_TEXT_FRAMES

	cfg_path = root_dir / "config.json"
	if not cfg_path.exists():
		return

	data = json.loads(cfg_path.read_text(encoding="utf-8"))
	v = data.get("video", {})

	FPS = int(v.get("fps", FPS))
	ROWS_PER_FRAME = int(v.get("rows_per_frame", ROWS_PER_FRAME))
	COLS_PER_ROW = int(v.get("cols_per_row", COLS_PER_ROW))
	MAX_TEXT_FRAMES = max(1, int(v.get("max_text_frames", MAX_TEXT_FRAMES)))
	CHARS_PER_FRAME = ROWS_PER_FRAME * COLS_PER_ROW
	FONT_SIZE = int(v.get("font_size", FONT_SIZE))
	FONT_PATH = str(v.get("font_path", FONT_PATH)).strip()
	LINE_SPACING = int(v.get("line_spacing", LINE_SPACING))
	# 同时兼容 char_spacing 与 col_spacing 两种命名。
	CHAR_SPACING = int(v.get("char_spacing", v.get("col_spacing", CHAR_SPACING)))
	LEAD_IN_SECONDS = float(v.get("lead_in_seconds", LEAD_IN_SECONDS))
	TAIL_OUT_SECONDS = float(v.get("tail_out_seconds", TAIL_OUT_SECONDS))
	SIDE_BAND_WIDTH = int(v.get("side_band_width", SIDE_BAND_WIDTH))
	FRAME_WIDTH = int(v.get("frame_width", FRAME_WIDTH))
	FRAME_HEIGHT = int(v.get("frame_height", FRAME_HEIGHT))


def pick_text_font() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
	text_w = FRAME_WIDTH - SIDE_BAND_WIDTH * 2
	text_h = FRAME_HEIGHT
	if FONT_SIZE > 0:
		if FONT_SIZE < 8:
			print(f"提示：当前 font_size={FONT_SIZE}，文字会非常小，建议设置为 12-48。")
		font = pick_font(FONT_SIZE)
		bbox = font.getbbox("汉")
		char_w = max(1, bbox[2] - bbox[0])
		char_h = max(1, bbox[3] - bbox[1])
		return font, char_w, char_h

	for size in range(56, 20, -1):
		font = pick_font(size)
		bbox = font.getbbox("汉")
		char_w = max(1, bbox[2] - bbox[0])
		char_h = max(1, bbox[3] - bbox[1])
		block_w = COLS_PER_ROW * char_w + (COLS_PER_ROW - 1) * CHAR_SPACING
		block_h = ROWS_PER_FRAME * char_h + (ROWS_PER_FRAME - 1) * LINE_SPACING
		if block_w <= text_w - 16 and block_h <= text_h - 16:
			return font, char_w, char_h
	font = pick_font(20)
	bbox = font.getbbox("汉")
	return font, max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def render_text_frame(
	frame_idx: int,
	chunks: list[str],
	lead_in_frames: int,
	text_frame_count: int,
	font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
	char_w: int,
	char_h: int,
) -> Image.Image:
	img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), "white")
	draw = ImageDraw.Draw(img)

	text_start = lead_in_frames
	text_end = text_start + text_frame_count - 1
	if text_start <= frame_idx <= text_end:
		text_idx = frame_idx - text_start

		draw.rectangle(
			[SIDE_BAND_WIDTH, 0, FRAME_WIDTH - SIDE_BAND_WIDTH - 1, FRAME_HEIGHT - 1],
			fill="white",
		)
		draw.rectangle(
			[FRAME_WIDTH - SIDE_BAND_WIDTH, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1],
			fill="black",
		)
		left_color = "black" if text_idx % 2 == 1 else "white"
		draw.rectangle([0, 0, SIDE_BAND_WIDTH - 1, FRAME_HEIGHT - 1], fill=left_color)

		lines = chunk_to_lines(chunks[text_idx])
		block_w = COLS_PER_ROW * char_w + (COLS_PER_ROW - 1) * CHAR_SPACING
		block_h = ROWS_PER_FRAME * char_h + (ROWS_PER_FRAME - 1) * LINE_SPACING
		text_x = SIDE_BAND_WIDTH + max(8, (FRAME_WIDTH - 2 * SIDE_BAND_WIDTH - block_w) // 2)
		text_y = max(0, (FRAME_HEIGHT - block_h) // 2)
		for row, line in enumerate(lines):
			y = text_y + row * (char_h + LINE_SPACING)
			for col, ch in enumerate(line):
				x = text_x + col * (char_w + CHAR_SPACING)
				draw.text((x, y), ch, fill="black", font=font)

	return img


def generate_frame_images(
	frame_images_dir: Path,
	chunks: list[str],
	lead_in_frames: int,
	tail_out_frames: int,
) -> tuple[int, int]:
	text_frame_count = max(len(chunks), 1)
	total_frames = lead_in_frames + text_frame_count + tail_out_frames
	font, char_w, char_h = pick_text_font()

	clean_png_frames(frame_images_dir)
	for frame_idx in range(total_frames):
		img = render_text_frame(frame_idx, chunks, lead_in_frames, text_frame_count, font, char_w, char_h)
		img.save(frame_images_dir / f"{frame_idx + 1:06d}.png")

	return text_frame_count, total_frames


def clean_png_frames(frames_dir: Path) -> None:
	frames_dir.mkdir(parents=True, exist_ok=True)
	for p in frames_dir.glob("*.png"):
		p.unlink(missing_ok=True)


def cleanup_pycache(root: Path) -> None:
	for p in root.rglob("__pycache__"):
		if p.is_dir():
			shutil.rmtree(p, ignore_errors=True)


def ceil_div(a: int, b: int) -> int:
	if a <= 0:
		return 0
	return (a + b - 1) // b


def load_ecc_nk(root_dir: Path) -> tuple[int, int]:
	n = ECC_DEFAULT_N
	k = None
	cfg_path = root_dir / "config.json"
	if cfg_path.exists():
		data = json.loads(cfg_path.read_text(encoding="utf-8"))
		ecc = data.get("ecc", {})
		n = int(ecc.get("n", n))
		if "k" in ecc and ecc["k"] is not None:
			k = int(ecc["k"])
		else:
			erasure_rate = float(ecc.get("erasure_rate", ecc.get("rs_redundancy_rate", 0.05)))
			random_error_rate = float(ecc.get("random_error_rate", 0.005))
			safety_factor = float(ecc.get("design_safety_factor", 1.20))
			min_parity_symbols = int(ecc.get("min_parity_symbols", 4))

			erasure_rate = min(max(erasure_rate, 0.0), 0.999999)
			random_error_rate = min(max(random_error_rate, 0.0), 0.999999)
			safety_factor = max(safety_factor, 1.0)
			min_parity_symbols = max(min_parity_symbols, 1)

			required = n * (erasure_rate + 2.0 * random_error_rate) * safety_factor
			parity = max(min_parity_symbols, math.ceil(required))
			parity = min(max(parity, 1), n - 1)
			k = n - parity

	if k is None:
		k = 242

	if n <= 1 or k <= 0 or k >= n:
		raise ValueError(f"ECC 参数无效：n={n}, k={k}")
	return n, k


def estimate_ecc_encoded_bytes(raw_bytes_len: int, n: int, k: int) -> int:
	if raw_bytes_len < 0:
		raise ValueError("raw_bytes_len 不能为负数")

	data_symbols = ceil_div(raw_bytes_len * 8, ECC_SYMBOL_BITS)
	if data_symbols == 0:
		return ECC_HEADER_SIZE

	full_blocks = data_symbols // k
	remainder = data_symbols % k
	nsym = n - k

	payload_bytes = full_blocks * ceil_div(n * ECC_SYMBOL_BITS, 8)
	if remainder > 0:
		shortened_symbols = nsym + remainder
		payload_bytes += ceil_div(shortened_symbols * ECC_SYMBOL_BITS, 8)

	return ECC_HEADER_SIZE + payload_bytes


def estimate_text_frames_from_raw(raw_bytes_len: int, n: int, k: int) -> int:
	ecc_bytes = estimate_ecc_encoded_bytes(raw_bytes_len, n, k)
	hanzi_count = ceil_div(ecc_bytes * 8, ECC_SYMBOL_BITS)
	return max(1, ceil_div(hanzi_count, CHARS_PER_FRAME))


def truncate_source_for_frame_budget(src_bin: Path, root_dir: Path) -> tuple[int, int]:
	n, k = load_ecc_nk(root_dir)
	raw = src_bin.read_bytes()
	orig_len = len(raw)

	max_hanzi = MAX_TEXT_FRAMES * CHARS_PER_FRAME
	max_ecc_bytes = (max_hanzi * ECC_SYMBOL_BITS) // 8

	if max_ecc_bytes <= ECC_HEADER_SIZE:
		raise RuntimeError("当前帧预算过小，连 ECC 头都无法承载")

	if estimate_text_frames_from_raw(orig_len, n, k) <= MAX_TEXT_FRAMES:
		print(
			f"帧预算检查：原始文件 {orig_len} 字节（{orig_len * 8} bit）在 {MAX_TEXT_FRAMES} 文字帧内，无需截断。"
		)
		return orig_len, orig_len

	lo = 0
	hi = orig_len
	best = 0
	while lo <= hi:
		mid = (lo + hi) // 2
		ecc_bytes = estimate_ecc_encoded_bytes(mid, n, k)
		if ecc_bytes <= max_ecc_bytes:
			best = mid
			lo = mid + 1
		else:
			hi = mid - 1

	truncated = raw[:best]
	src_bin.write_bytes(truncated)

	print(
		f"帧预算检查：最多 {MAX_TEXT_FRAMES} 文字帧，承载上限约 {max_hanzi * ECC_SYMBOL_BITS} bit。"
	)
	print(
		f"源文件已截断：{orig_len} -> {best} 字节（{orig_len * 8} -> {best * 8} bit），ECC 参数 n={n}, k={k}。"
	)
	return orig_len, best


def main() -> int:
	root_dir = Path(__file__).resolve().parent
	load_shared_config(root_dir)
	test_dir = root_dir / "test"
	ecc_out_dir = root_dir / "ECC" / "encode_output"
	ch_out_dir = root_dir / "CH" / "encode_output"

	if shutil.which("ffmpeg") is None:
		print("错误：未检测到 ffmpeg，请先安装并加入 PATH。", file=sys.stderr)
		return 1

	# 1) 读取 test 中唯一 .bin，并调用 ECC 编码
	src_bin = find_single_file(test_dir, "*.bin")
	truncate_source_for_frame_budget(src_bin, root_dir)
	print(f"步骤1：ECC 编码输入文件 -> {src_bin.name}")
	run_cmd(
		[
			sys.executable,
			str(root_dir / "ECC" / "ECC_encode.py"),
			"--input-dir",
			str(test_dir),
			"--output-dir",
			str(ecc_out_dir),
		],
		root_dir,
	)

	# 2) 调用 CH 编码输出到 CH/encode_output
	print("步骤2：执行 CH 编码")
	run_cmd(
		[
			sys.executable,
			str(root_dir / "CH" / "CH_encode.py"),
			"--input-dir",
			str(ecc_out_dir),
			"--output-dir",
			str(ch_out_dir),
		],
		root_dir,
	)

	txt_file = find_single_file(ch_out_dir, "*.txt")
	txt = txt_file.read_text(encoding="utf-8")

	# 3) 先逐帧绘图，再按 30fps 合成视频
	print("步骤3：先生成逐帧图片")
	chunks = split_text_chunks(txt, CHARS_PER_FRAME)
	if len(chunks) > MAX_TEXT_FRAMES:
		print(f"警告：文字帧超出预算，已从 {len(chunks)} 帧截断到 {MAX_TEXT_FRAMES} 帧。")
		chunks = chunks[:MAX_TEXT_FRAMES]
	lead_in_frames = int(round(LEAD_IN_SECONDS * FPS))
	tail_out_frames = int(round(TAIL_OUT_SECONDS * FPS))

	frame_images_dir = root_dir / "_main_encode_frames"
	frame_pattern = frame_images_dir / "%06d.png"
	text_frame_count, total_frames = generate_frame_images(
		frame_images_dir,
		chunks,
		lead_in_frames,
		tail_out_frames,
	)

	generated_frames = len(list(frame_images_dir.glob("*.png")))
	if generated_frames != total_frames:
		raise RuntimeError(
			f"逐帧图片数量异常：期望 {total_frames}，实际 {generated_frames}。"
		)

	print("步骤 4：将图片序列组合为 30fps 视频")

	out_video = root_dir / f"{txt_file.stem}.mp4"
	compose_cmd = [
		"ffmpeg",
		"-y",
		"-framerate",
		str(FPS),
		"-i",
		str(frame_pattern),
		"-r",
		str(FPS),
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		str(out_video),
	]

	try:
		run_cmd(compose_cmd, root_dir)
	except RuntimeError as exc:
		if "Permission denied" not in str(exc):
			raise
		fallback_video = root_dir / f"{txt_file.stem}.regen.mp4"
		print(f"目标文件被占用，改为输出：{fallback_video}")
		compose_cmd[-1] = str(fallback_video)
		run_cmd(compose_cmd, root_dir)
		out_video = fallback_video

	if frame_images_dir.exists():
		shutil.rmtree(frame_images_dir)
		print(f"已清理临时图片目录：{frame_images_dir}")

	print("编码流程完成。")
	print(f"文本文件: {txt_file}")
	print(f"图片帧目录: {frame_images_dir}")
	print(f"图片帧数量: {generated_frames}")
	print(f"输出视频: {out_video}")
	print(f"文字帧数: {text_frame_count}")
	print(f"视频总帧数: {total_frames}")
	print(
		f"编码参数: rows={ROWS_PER_FRAME}, cols={COLS_PER_ROW}, font_size={FONT_SIZE}, "
		f"line_spacing={LINE_SPACING}, col_spacing={CHAR_SPACING}, max_text_frames={MAX_TEXT_FRAMES}"
	)
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	finally:
		cleanup_pycache(Path(__file__).resolve().parent)
