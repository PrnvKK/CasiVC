# data/libritts_download.py
"""
Automated LibriTTS Downloader & Extractor
========================================

Downloads and unpacks LibriTTS subsets (e.g. dev-clean, 1.2 GB)
from OpenSLR mirrors with streaming chunk download and progress reporting.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

OPENSLR_URLS = {
    "dev-clean": [
        "https://openslr.trmal.net/resources/60/dev-clean.tar.gz",
        "https://www.openslr.org/resources/60/dev-clean.tar.gz",
        "https://openslr.elda.org/resources/60/dev-clean.tar.gz"
    ],
    "dev-other": [
        "https://openslr.trmal.net/resources/60/dev-other.tar.gz",
        "https://www.openslr.org/resources/60/dev-other.tar.gz"
    ],
    "test-clean": [
        "https://openslr.trmal.net/resources/60/test-clean.tar.gz",
        "https://www.openslr.org/resources/60/test-clean.tar.gz"
    ],
    "train-clean-100": [
        "https://openslr.trmal.net/resources/60/train-clean-100.tar.gz",
        "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        "https://openslr.elda.org/resources/60/train-clean-100.tar.gz"
    ],
    "train-clean-360": [
        "https://openslr.trmal.net/resources/60/train-clean-360.tar.gz",
        "https://www.openslr.org/resources/60/train-clean-360.tar.gz"
    ],
    "train-other-500": [
        "https://openslr.trmal.net/resources/60/train-other-500.tar.gz",
        "https://www.openslr.org/resources/60/train-other-500.tar.gz"
    ]
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_libritts(subset="dev-clean", target_dir="data/LibriTTS"):
    dest_path = Path(target_dir) / subset
    if dest_path.exists() and len(list(dest_path.rglob("*.wav"))) > 100:
        print(f"[Dataset] '{dest_path}' already exists with {len(list(dest_path.rglob('*.wav')))} utterances.")
        return dest_path

    os.makedirs(target_dir, exist_ok=True)
    archive_path = Path(target_dir) / f"{subset}.tar.gz"

    urls = OPENSLR_URLS.get(subset, [])
    if not urls:
        raise ValueError(f"Unknown subset: {subset}. Choose from {list(OPENSLR_URLS.keys())}")

    download_success = False
    for url in urls:
        print(f"\n[Download] Connecting to {url}...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {subset}") as t:
                urllib.request.urlretrieve(url, filename=str(archive_path), reporthook=t.update_to)
            download_success = True
            break
        except Exception as e:
            print(f"[Warning] Download failed from {url}: {e}. Trying mirror...")

    if not download_success:
        raise RuntimeError(f"Failed to download {subset} from all mirrors.")

    print(f"\n[Extract] Unpacking {archive_path} to {target_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    # LibriTTS tar extracts to LibriTTS/<subset>
    extracted_subset = Path(target_dir) / "LibriTTS" / subset
    if extracted_subset.exists() and not dest_path.exists():
        # Move or link if needed
        import shutil
        for item in extracted_subset.iterdir():
            shutil.move(str(item), str(dest_path))
        shutil.rmtree(str(Path(target_dir) / "LibriTTS"), ignore_errors=True)

    # Clean up archive to save disk space
    if archive_path.exists():
        os.remove(archive_path)
        print(f"[Clean] Removed temporary archive {archive_path.name}")

    wav_count = len(list(dest_path.rglob("*.wav")))
    print(f"✅ Successfully extracted {subset}: {wav_count} audio utterances found in '{dest_path}'.")
    return dest_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="dev-clean", help="LibriTTS subset (dev-clean, dev-other)")
    parser.add_argument("--target_dir", type=str, default="data/LibriTTS", help="Extraction directory")
    args = parser.parse_args()

    download_and_extract_libritts(subset=args.subset, target_dir=args.target_dir)
