#!/usr/bin/env python
"""版本号管理脚本"""

import re
import sys
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def parse_version(version_str: str) -> tuple[int, int, int]:
    """解析版本号字符串"""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}. Expected: major.minor.patch")
    return tuple(map(int, match.groups()))


def format_version(major: int, minor: int, patch: int) -> str:
    """格式化版本号"""
    return f"{major}.{minor}.{patch}"


def get_current_version() -> str:
    """从 pyproject.toml 读取当前版本号"""
    root = get_project_root()
    pyproject_path = root / "pyproject.toml"

    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Cannot find version in pyproject.toml")

    return match.group(1)


def update_pyproject(new_version: str) -> None:
    """更新 pyproject.toml 中的版本号"""
    root = get_project_root()
    pyproject_path = root / "pyproject.toml"

    content = pyproject_path.read_text(encoding="utf-8")
    updated = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE
    )

    pyproject_path.write_text(updated, encoding="utf-8")
    print(f"✓ Updated pyproject.toml: {new_version}")


def update_readme(new_version: str) -> None:
    """更新 README.md 中的版本号徽章"""
    root = get_project_root()
    readme_path = root / "README.md"

    content = readme_path.read_text(encoding="utf-8")

    # 更新版本徽章
    updated = re.sub(
        r'(https://img\.shields\.io/badge/version-)[^-]+(-green\.svg)',
        rf'\g<1>{new_version}\g<2>',
        content
    )

    readme_path.write_text(updated, encoding="utf-8")
    print(f"✓ Updated README.md: {new_version}")


def bump_version(bump_type: str) -> str:
    """升级版本号

    Args:
        bump_type: "major", "minor", "patch" 或具体版本号（如 "1.2.3"）

    Returns:
        新版本号
    """
    current = get_current_version()
    major, minor, patch = parse_version(current)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    elif re.match(r"^\d+\.\d+\.\d+$", bump_type):
        # 直接指定版本号
        return bump_type
    else:
        raise ValueError(f"Invalid bump type: {bump_type}. Use 'major', 'minor', 'patch' or 'x.y.z'")

    return format_version(major, minor, patch)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/bump_version.py <bump_type>")
        print()
        print("Examples:")
        print("  python scripts/bump_version.py patch    # 1.2.2 -> 1.2.3")
        print("  python scripts/bump_version.py minor    # 1.2.2 -> 1.3.0")
        print("  python scripts/bump_version.py major    # 1.2.2 -> 2.0.0")
        print("  python scripts/bump_version.py 1.5.0    # Set to 1.5.0")
        sys.exit(1)

    bump_type = sys.argv[1]

    try:
        current = get_current_version()
        print(f"Current version: {current}")

        new_version = bump_version(bump_type)
        print(f"New version: {new_version}")
        print()

        # 更新文件
        update_pyproject(new_version)
        update_readme(new_version)

        print()
        print(f"✅ Version bumped: {current} -> {new_version}")
        print()
        print("Next steps:")
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git add -A && git commit -m 'chore: bump version to {new_version}'")
        print(f"  3. Tag: git tag v{new_version}")
        print(f"  4. Push: git push && git push --tags")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
