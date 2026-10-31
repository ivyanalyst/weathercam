import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import psutil

from weathercam.core.config import PROJECT_ROOT, load_config
from weathercam.core.memory import MemoryGuard, MemoryLimitError, rss_mb

REQUIRED = ["numpy", "pyyaml", "psutil", "weathercam"]
OPTIONAL = ["pytest", "torch", "torchvision", "open_clip_torch", "pillow", "fastapi"]
BYTES_PER_MB = 1024 * 1024


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    ok = True

    print(f"python      {platform.python_version()} ({sys.executable})")
    print(f"platform    {platform.platform()}")
    print(f"root        {PROJECT_ROOT}")

    print("\nrequired packages")
    for name in REQUIRED:
        found = package_version(name)
        print(f"  {name:<18}{found or 'MISSING'}")
        if found is None:
            ok = False

    print("\noptional packages")
    for name in OPTIONAL:
        print(f"  {name:<18}{package_version(name) or 'not installed'}")

    mem = psutil.virtual_memory()
    total_mb = mem.total / BYTES_PER_MB
    available_mb = mem.available / BYTES_PER_MB
    print(f"\nram         total {total_mb:.0f} MB, available {available_mb:.0f} MB")
    print(f"process     rss {rss_mb():.0f} MB")

    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as err:
        print(f"\nconfig      FAILED: {err}")
        return 1

    print(f"\nconfig      seed {cfg.seed}, memory limit {cfg.memory_limit_mb} MB")
    if cfg.memory_limit_mb > available_mb:
        print("  warning: limit is above available RAM, the guard may never fire")

    for label, folder in [("data", cfg.paths.data_dir), ("dawn", cfg.paths.dawn_dir)]:
        exists = folder.is_dir()
        print(f"  {label:<6}{folder} {'ok' if exists else 'MISSING'}")
        if not exists:
            ok = False

    try:
        current = MemoryGuard(cfg.memory_limit_mb).check()
        print(f"  guard ok at {current:.0f} MB")
    except MemoryLimitError as err:
        print(f"  guard FAILED: {err}")
        ok = False

    print("\nresult      " + ("OK" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
