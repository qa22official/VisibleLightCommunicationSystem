#!/usr/bin/env python3
"""检查依赖、必要时创建虚拟环境，然后运行 main_encode.py。"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MAIN_SCRIPT = PROJECT_ROOT / "main_encode.py"
VENV_DIR = PROJECT_ROOT / ".venv"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def parse_requirements(requirements_file: Path) -> list[tuple[str, str | None]]:
    requirements: list[tuple[str, str | None]] = []
    version_pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$")

    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        match = version_pattern.match(line)
        if match:
            requirements.append((match.group(1), match.group(2)))
        else:
            requirements.append((line, None))

    return requirements


def requirements_satisfied(requirements: list[tuple[str, str | None]]) -> bool:
    for package_name, expected_version in requirements:
        try:
            installed_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return False

        if expected_version is not None and installed_version != expected_version:
            return False

    return True


def get_venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_venv_and_install() -> Path:
    venv_python = get_venv_python()

    if not venv_python.exists():
        print(f"当前环境不满足 requirements，开始创建虚拟环境：{VENV_DIR}")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)], PROJECT_ROOT)

    print("正在安装/修复依赖...")
    run_command([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], PROJECT_ROOT)
    return venv_python


def main() -> None:
    if not REQUIREMENTS_FILE.exists():
        raise SystemExit(f"未找到依赖文件：{REQUIREMENTS_FILE}")
    if not MAIN_SCRIPT.exists():
        raise SystemExit(f"未找到入口脚本：{MAIN_SCRIPT}")

    requirements = parse_requirements(REQUIREMENTS_FILE)

    if not is_frozen() and requirements_satisfied(requirements):
        python_executable = Path(sys.executable)
        print("当前环境已满足 requirements，直接运行 main_encode.py")
    else:
        python_executable = ensure_venv_and_install()

    print(f"使用解释器：{python_executable}")
    run_command([str(python_executable), str(MAIN_SCRIPT)], PROJECT_ROOT)


if __name__ == "__main__":
    main()