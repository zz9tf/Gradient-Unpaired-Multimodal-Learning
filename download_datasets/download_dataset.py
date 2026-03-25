#!/usr/bin/env python3
"""Download a dataset file from a URL into a fixed data directory."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


CHUNK_SIZE = 1024 * 1024  # 1 MB

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download a file from URL into the project data directory."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Source file URL, e.g. https://.../ILSVRC2012_img_train.tar",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default="",
        help="Output directory, e.g. /home/zz/zheng/Unpaired-Multimodal-Learning/data",
    )
    return parser.parse_args()


def infer_filename(url: str) -> str:
    """Infer output filename from URL path."""
    path = urlparse(url).path
    filename = Path(path).name
    if not filename:
        raise ValueError(f"Cannot infer filename from URL: {url}")
    return filename


def format_bytes(num_bytes: float) -> str:
    """Convert bytes to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024
    raise RuntimeError("Unreachable code in format_bytes.")


def render_progress(
    downloaded: int,
    total_size: int | None,
    start_time: float,
    bar_width: int = 30,
) -> None:
    """Render one-line download progress in terminal."""
    elapsed = max(time.time() - start_time, 1e-9)
    speed = downloaded / elapsed

    if total_size is not None and total_size > 0:
        ratio = min(downloaded / total_size, 1.0)
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        message = (
            f"\r[{bar}] {ratio * 100:6.2f}% "
            f"{format_bytes(downloaded)}/{format_bytes(total_size)} "
            f"{format_bytes(speed)}/s"
        )
    else:
        message = (
            f"\rDownloaded {format_bytes(downloaded)} "
            f"at {format_bytes(speed)}/s (total size unknown)"
        )

    sys.stdout.write(message)
    sys.stdout.flush()


def download_to_fixed_dir(url: str, output_dir: str) -> Path:
    """Download URL content into the fixed project data directory."""
    filename = infer_filename(url)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    with urlopen(url) as response, output_path.open("wb") as f:
        content_length = response.getheader("Content-Length")
        total_size = int(content_length) if content_length is not None else None
        downloaded = 0
        start_time = time.time()

        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            render_progress(downloaded, total_size, start_time)

    sys.stdout.write("\n")
    sys.stdout.flush()

    return output_path


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    output_path = download_to_fixed_dir(args.url, args.output_dir)
    print(f"Downloaded: {output_path}")

if __name__ == "__main__":
    main()
