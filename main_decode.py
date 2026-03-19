#!/usr/bin/env python3
"""读取根目录中的 .mp4，调用 ffmpeg 拆帧到 frames 目录。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run_ffmpeg(video_path: Path, output_dir: Path) -> None:
    """调用 ffmpeg 将视频拆帧为 PNG。"""
    output_dir.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    frames_root = root_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        print("错误：未检测到 ffmpeg，请先安装并加入 PATH。", file=sys.stderr)
        return 1

    videos = sorted(p for p in root_dir.glob("*.mp4") if p.is_file())
    if not videos:
        print(f"未在根目录找到 .mp4 文件：{root_dir}")
        return 0

    if len(videos) > 1:
        print("检测到多个 .mp4 文件。默认仅允许存在一个视频，请只保留一个后再运行。", file=sys.stderr)
        return 1

    video = videos[0]
    print(f"检测到视频：{video.name}，开始拆帧...")
    print(f"处理：{video.name} -> {frames_root}")
    run_ffmpeg(video, frames_root)

    print(f"完成。拆帧结果已保存至：{frames_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
