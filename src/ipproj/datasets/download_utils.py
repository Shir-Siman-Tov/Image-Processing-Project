"""Shared download-and-cache helper for KITTI archives."""
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_and_extract(url: str, dest_dir: Path, marker_name: str) -> Path:
    """Download `url` into `dest_dir` and extract it, unless already done.

    `marker_name` names the file/dir expected inside `dest_dir` after
    extraction; its presence is treated as "already downloaded".
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / marker_name
    if marker.exists():
        return dest_dir

    zip_path = dest_dir / Path(url).name
    if not zip_path.exists():
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=zip_path.name) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    if not marker.exists():
        raise FileNotFoundError(
            f"Expected '{marker_name}' inside {dest_dir} after extracting {url}, but it's missing. "
            "The KITTI archive layout may have changed - inspect the zip contents."
        )
    return dest_dir
