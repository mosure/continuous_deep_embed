#!/usr/bin/env python3

"""
dataset_get.py — ImageNet helpers for experiments

Features
--------
1) Open subsets download (no credentials):
   - Imagenette / Imagewoof (fast.ai mirrors)
   - Tiny-ImageNet-200 (CS231n)

2) Full ImageNet-1k preparation (license-respecting):
   - Uses local tarballs you already have:
       ILSVRC2012_img_train.tar
       ILSVRC2012_img_val.tar
       ILSVRC2012_devkit_t12.tar.gz  (for restructuring val/ into class folders)
   - Optional: accepts user-provided magnet URIs (Academic Torrents) or
     cookies.txt for the official site, but DOES NOT ship any URLs itself.
     You must supply them explicitly (see flags).

3) Subsetting from a full ImageNet tree:
   - Given a list of WNIDs, create a new dataset containing only those classes
     by symlink or copy (your choice). Works for train/ and val/.

4) Validation split restructuring:
   - Given the devkit tarball, moves/links val images into class subfolders.

5) Paper-level helper (NEW):
   - `imagenet-paper` prints ready-to-run commands for Linux (bash) or
     Windows (PowerShell) that:
       a) Pull the raw tarballs using YOUR magnet URIs or cookies+URLs
       b) Prepare the final train/ and val/ tree used in the paper
       c) Optionally create a subset (e.g., ImageNet-100) via wnids list

Usage examples
--------------
# Download open subsets
python dataset_get.py imagenette --out /data/imagenette --size 320
python dataset_get.py imagewoof  --out /data/imagewoof --size 320
python dataset_get.py tiny-imagenet --out /data/tiny-imagenet

# Prepare full ImageNet from local tarballs
python dataset_get.py imagenet-full \
  --train-tar /data/raw/ILSVRC2012_img_train.tar \
  --val-tar   /data/raw/ILSVRC2012_img_val.tar \
  --devkit    /data/raw/ILSVRC2012_devkit_t12.tar.gz \
  --out       /data/imagenet

# Create a subset (e.g., ImageNet-100) from a full copy using a wnids.txt list
python dataset_get.py subset \
  --src /data/imagenet \
  --wnids wnids.txt \
  --out /data/imagenet-100 \
  --copy false

# (Advanced) Attempt download of full tars using your magnet URIs or cookies
python dataset_get.py fetch-tars \
  --out /data/raw \
  --train-magnet "magnet:?xt=urn:btih:..." \
  --val-magnet   "magnet:?xt=urn:btih:..."

# (NEW) Print the paper-level commands to fetch & prepare ImageNet-1k
python dataset_get.py imagenet-paper \
  --raw /data/raw --out /data/imagenet \
  --shell bash \
  --train-magnet "magnet:?xt=urn:btih:..." \
  --val-magnet   "magnet:?xt=urn:btih:..." \
  --devkit-magnet "magnet:?xt=urn:btih:..." \
  --wnids /path/to/imagenet100_wnids.txt --subset-out /data/imagenet-100

Notes
-----
• For FULL ImageNet downloads you are responsible for license compliance.
• This script does not include any restricted URLs and will never bypass
  authentication. It only uses inputs you provide.
• On Linux, symlinking subsets is recommended to save disk space.
"""

import argparse
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

# ----------------- small utils -----------------

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def human(n):
    for unit in ['','K','M','G','T']:
        if abs(n) < 1024.0: return f"{n:3.1f}{unit}B"
        n /= 1024.0
    return f"{n:.1f}PB"

def download(url: str, dst: Path, chunk=1024*1024):
    """Simple downloader with resume support."""
    mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".part")
    headers = {}
    pos = 0
    if tmp.exists():
        pos = tmp.stat().st_size
        headers['Range'] = f'bytes={pos}-'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r, open(tmp, 'ab') as f:
        total = r.length + pos if r.length else None
        last = time.time()
        while True:
            b = r.read(chunk)
            if not b: break
            f.write(b)
            if time.time() - last > 1.0:
                s = tmp.stat().st_size
                if total:
                    pct = 100.0 * s / total
                    print(f"\r{dst.name}: {human(s)} / {human(total)} ({pct:4.1f}%)", end='')
                else:
                    print(f"\r{dst.name}: {human(s)}", end='')
                last = time.time()
    tmp.rename(dst)
    print(f"\nSaved {dst} ({human(dst.stat().st_size)})")

def safe_extract_tar(tar_path: Path, out_dir: Path):
    with tarfile.open(tar_path, 'r:*') as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        for m in tar.getmembers():
            target = out_dir / m.name
            if not is_within_directory(out_dir, target):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(out_dir)

def safe_extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        for n in z.namelist():
            target = out_dir / n
            if not str(target.resolve()).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Blocked path traversal: {n}")
        z.extractall(out_dir)

def is_windows() -> bool:
    return os.name == "nt"

def q(path: Path) -> str:
    # Quote a path for shell; both bash/Powershell accept double quotes
    return f"\"{str(path)}\""

# ----------------- public subsets -----------------

IMAGENETTE_URLS = {
    160: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    320: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    0:   "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",  # original ~256px
}
IMAGEWOOF_URLS = {
    160: "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz",
    320: "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
    0:   "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
}
TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

def cmd_imagenette(args):
    size = int(args.size)
    url = IMAGENETTE_URLS.get(size, IMAGENETTE_URLS[320])
    out = Path(args.out)
    mkdir(out)
    tgz = out / Path(url).name
    print(f"Downloading Imagenette ({size or 'orig'}): {url}")
    download(url, tgz)
    print("Extracting...")
    safe_extract_tar(tgz, out)
    print("Done. Train/val are inside the extracted folder.")

def cmd_imagewoof(args):
    size = int(args.size)
    url = IMAGEWOOF_URLS.get(size, IMAGEWOOF_URLS[320])
    out = Path(args.out)
    mkdir(out)
    tgz = out / Path(url).name
    print(f"Downloading Imagewoof ({size or 'orig'}): {url}")
    download(url, tgz)
    print("Extracting...")
    safe_extract_tar(tgz, out)
    print("Done. Train/val are inside the extracted folder.")

def cmd_tiny_imagenet(args):
    out = Path(args.out)
    mkdir(out)
    z = out / "tiny-imagenet-200.zip"
    print(f"Downloading Tiny-ImageNet-200: {TINY_IMAGENET_URL}")
    download(TINY_IMAGENET_URL, z)
    print("Extracting...")
    safe_extract_zip(z, out)
    print("Done. Train/val folders under tiny-imagenet-200/.")

# ----------------- full imagenet prep -----------------

def extract_train_tar(train_tar: Path, out_train: Path):
    """
    ILSVRC2012_img_train.tar contains 1000 tars (one per class).
    This will extract the inner tars into class subfolders under out_train.
    """
    mkdir(out_train)
    with tarfile.open(train_tar, 'r:*') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tar")]
        print(f"Found {len(members)} class tarballs in {train_tar.name}")
        for i, m in enumerate(members, 1):
            wnid = Path(m.name).stem  # e.g., n01440764
            dst_dir = out_train / wnid
            mkdir(dst_dir)
            f = tar.extractfile(m)
            if f is None:
                print(f"Warning: cannot extract {m.name}")
                continue
            with tarfile.open(fileobj=io.BytesIO(f.read()), mode='r:*') as t2:
                t2.extractall(dst_dir)
            if i % 25 == 0 or i == len(members):
                print(f"  extracted {i}/{len(members)}")

def extract_val_tar(val_tar: Path, out_val: Path):
    """
    ILSVRC2012_img_val.tar contains 50k images in a single folder.
    We'll extract them flat here; later, devkit labels will reorganize.
    """
    mkdir(out_val)
    print(f"Extracting {val_tar} to {out_val} ...")
    safe_extract_tar(val_tar, out_val)

def reorganize_val_with_devkit(val_dir: Path, devkit_tar: Path):
    """
    Use ILSVRC2012_devkit_t12.tar.gz to move validation images into class folders.
    """
    import scipy.io  # requires scipy installed
    tmp = val_dir.parent / "_devkit_tmp"
    mkdir(tmp)
    print(f"Extracting devkit {devkit_tar} ...")
    safe_extract_tar(devkit_tar, tmp)
    # Find ground truth and meta.mat
    gt_txt = next(tmp.rglob("ILSVRC2012_validation_ground_truth.txt"))
    meta_mat = next(tmp.rglob("meta.mat"))
    # Load mapping from synsets to numeric labels
    meta = scipy.io.loadmat(str(meta_mat))["synsets"]
    # synsets is structured; build list of wnids sorted by ILSVRC2012 ID
    # entry: [ILSVRC2012_ID, WNID, words, ..., num_children]
    id_to_wnid = {}
    for row in meta:
        ilsvrc_id = int(row[0][0][0])  # nested arrays
        wnid = str(row[0][1][0])
        id_to_wnid[ilsvrc_id] = wnid
    # Read 50k integers, one per image index
    with open(gt_txt, 'r') as f:
        labels = [int(x.strip()) for x in f.readlines() if x.strip()]
    # Images are named ILSVRC2012_val_00000001.JPEG ... 50000
    print("Reorganizing validation images into class directories...")
    for i, cls_id in enumerate(labels, 1):
        wnid = id_to_wnid[cls_id]
        dst = val_dir / wnid
        dst.mkdir(exist_ok=True)
        img = val_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
        if not img.exists():
            print(f"Warning: {img.name} missing; skipping")
            continue
        shutil.move(str(img), str(dst / img.name))
        if i % 2000 == 0 or i == len(labels):
            print(f"  moved {i}/{len(labels)}")
    # cleanup
    shutil.rmtree(tmp, ignore_errors=True)
    print("Validation split reorganized.")

def cmd_imagenet_full(args):
    out = Path(args.out)
    train_tar = Path(args.train_tar)
    val_tar = Path(args.val_tar)
    devkit = Path(args.devkit) if args.devkit else None
    out_train = out / "train"
    out_val = out / "val"
    print(f"Preparing ImageNet-1k under {out}")
    extract_train_tar(train_tar, out_train)
    extract_val_tar(val_tar, out_val)
    if devkit is not None:
        try:
            reorganize_val_with_devkit(out_val, devkit)
        except Exception as e:
            print("Devkit reorg failed (install scipy?). Val will remain flat.")
            print("Error:", e)
    print("Done.")

# ----------------- subsetting from full copy -----------------

def read_wnids(path: Path) -> List[str]:
    xs = []
    with open(path, 'r') as f:
        for line in f:
            w = line.strip()
            if not w: continue
            xs.append(w)
    return xs

def subset_split(src_split: Path, dst_split: Path, wnids: List[str], copy: bool):
    for wnid in wnids:
        src = src_split / wnid
        if not src.exists():
            print(f"Warning: missing class {wnid} in {src_split.name}; skipping")
            continue
        dst = dst_split / wnid
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            # symlink the folder if possible; fallback to file-level symlinks
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                dst.mkdir(exist_ok=True)
                for p in src.glob("*"):
                    os.symlink(p, dst / p.name)

def cmd_subset(args):
    src = Path(args.src)
    out = Path(args.out)
    wnids = read_wnids(Path(args.wnids))
    do_copy = str(args.copy).lower() == "true"
    print(f"Creating subset with {len(wnids)} classes from {src} -> {out} (copy={do_copy})")
    for split in ["train", "val"]:
        subset_split(src / split, out / split, wnids, copy=do_copy)
    print("Done.")

# ----------------- fetch raw tars (advanced, BYO magnets/cookies) ---------

def have_bin(name: str) -> bool:
    from shutil import which
    return which(name) is not None

def cmd_fetch_tars(args):
    """
    Download raw ImageNet tarballs using user-supplied magnets or cookies.
    Requires 'aria2c' for magnet links or 'curl/wget' for HTTP.
    No URLs are included in this script by design.
    """
    out = Path(args.out); mkdir(out)
    if args.train_magnet:
        if not have_bin("aria2c"):
            print("aria2c not found. Please install it to use magnet URIs."); sys.exit(2)
        print("Fetching train tar via magnet (user-supplied)...")
        os.system(f'aria2c --dir="{out}" --summary-interval=5 --seed-time=0 "{args.train_magnet}"')
    if args.val_magnet:
        if not have_bin("aria2c"):
            print("aria2c not found. Please install it to use magnet URIs."); sys.exit(2)
        print("Fetching val tar via magnet (user-supplied)...")
        os.system(f'aria2c --dir="{out}" --summary-interval=5 --seed-time=0 "{args.val_magnet}"')
    if args.cookies and args.train_url and args.val_url:
        if not (have_bin("curl") or have_bin("wget")):
            print("curl/wget not found. Install one to use cookies+HTTP mode."); sys.exit(2)
        cookies = Path(args.cookies)
        if not cookies.exists(): print("cookies file missing"); sys.exit(2)
        # Example: curl --cookie cookies.txt -L -o file.tar URL
        print("Fetching via cookies+HTTP (user-supplied URLs)...")
        if have_bin("curl"):
            os.system(f'curl -L --cookie "{cookies}" -o "{out / "ILSVRC2012_img_train.tar"}" "{args.train_url}"')
            os.system(f'curl -L --cookie "{cookies}" -o "{out / "ILSVRC2012_img_val.tar"}" "{args.val_url}"')
            if args.devkit_url:
                os.system(f'curl -L --cookie "{cookies}" -o "{out / "ILSVRC2012_devkit_t12.tar.gz"}" "{args.devkit_url}"')
        else:
            os.system(f'wget --load-cookies="{cookies}" -O "{out / "ILSVRC2012_img_train.tar"}" "{args.train_url}"')
            os.system(f'wget --load-cookies="{cookies}" -O "{out / "ILSVRC2012_img_val.tar"}" "{args.val_url}"')
            if args.devkit_url:
                os.system(f'wget --load-cookies="{cookies}" -O "{out / "ILSVRC2012_devkit_t12.tar.gz"}" "{args.devkit_url}"')
    print("Fetch stage complete. Verify files and checksums before extraction.")

# ----------------- paper-level command printer (NEW) -----------------

def cmd_imagenet_paper(args):
    """
    Print OS-specific commands to download & prepare the paper-level ImageNet-1k dataset.
    No restricted URLs are embedded; you may provide your own magnets/URLs to auto-fill.
    """
    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()
    wnids = Path(args.wnids).resolve() if args.wnids else None
    subset_out = Path(args.subset_out).resolve() if args.subset_out else None
    shell = args.shell or ("powershell" if is_windows() else "bash")

    print("\n=== ImageNet-1k: Paper-level preparation ===\n")
    print(f"Target dataset root: {out}")
    print(f"Raw tarballs directory: {raw}")
    if wnids:
        print(f"Subset WNIDs file: {wnids}")
        print(f"Subset output root: {subset_out or (out.parent / (out.name + '-subset'))}")
    print("\nLicense note: You must have permission to download/use ImageNet. "
          "This tool does not include URLs and will not bypass authentication.\n")

    # Fill-ins
    TRAIN_TAR = raw / "ILSVRC2012_img_train.tar"
    VAL_TAR   = raw / "ILSVRC2012_img_val.tar"
    DEVKIT_TGZ= raw / "ILSVRC2012_devkit_t12.tar.gz"

    tmag = args.train_magnet or "<PASTE_TRAIN_MAGNET_URI_HERE>"
    vmag = args.val_magnet   or "<PASTE_VAL_MAGNET_URI_HERE>"
    dmag = args.devkit_magnet or "<OPTIONAL_DEVKIT_MAGNET_URI_HERE>"
    cookies = args.cookies or "<cookies.txt>"
    train_url = args.train_url or "<AUTHENTICATED_TRAIN_URL>"
    val_url   = args.val_url   or "<AUTHENTICATED_VAL_URL>"
    dev_url   = args.devkit_url or "<AUTHENTICATED_DEVKIT_URL>"

    # Emit commands
    if shell == "bash":
        print("### Bash (Linux/macOS) ###\n")
        print(f"mkdir -p {q(raw)} {q(out)}")
        print("\n# Option A: Download via magnet URIs (requires aria2c)")
        print(f'aria2c --dir={q(raw)} --summary-interval=5 --seed-time=0 "{tmag}"')
        print(f'aria2c --dir={q(raw)} --summary-interval=5 --seed-time=0 "{vmag}"')
        print(f'# (optional) devkit:')
        print(f'aria2c --dir={q(raw)} --summary-interval=5 --seed-time=0 "{dmag}"')
        print("\n# Option B: Authenticated HTTP with cookies (curl). You must supply valid URLs.")
        print(f'curl -L --cookie {q(Path(cookies))} -o {q(TRAIN_TAR)} "{train_url}"')
        print(f'curl -L --cookie {q(Path(cookies))} -o {q(VAL_TAR)}   "{val_url}"')
        print(f'# (optional) devkit:')
        print(f'curl -L --cookie {q(Path(cookies))} -o {q(DEVKIT_TGZ)} "{dev_url}"')
        print("\n# (Recommended) Verify checksums manually before extraction (example shows SHA1):")
        print(f"sha1sum {q(TRAIN_TAR)}")
        print(f"sha1sum {q(VAL_TAR)}")
        print(f"sha1sum {q(DEVKIT_TGZ)}  # if downloaded")
        print("\n# Prepare the final train/ and val/ tree (devkit reorganizes val/ into class folders):")
        print(f"python {q(Path(__file__))} imagenet-full "
              f"--train-tar {q(TRAIN_TAR)} --val-tar {q(VAL_TAR)} "
              f"--devkit {q(DEVKIT_TGZ)} --out {q(out)}")
        if wnids:
            so = subset_out or (out.parent / (out.name + "-subset"))
            print("\n# (Optional) Create subset from the full copy using your WNID list:")
            print(f"python {q(Path(__file__))} subset --src {q(out)} --wnids {q(wnids)} "
                  f"--out {q(so)} --copy false")
        print("\n# Final dataset path to use in training scripts:")
        print(f"#   --imagenet {q(out)}   # contains train/ and val/")
        if wnids:
            so = subset_out or (out.parent / (out.name + "-subset"))
            print(f"#   --imagenet {q(so)}   # for subset runs")
    else:
        # PowerShell
        print("### PowerShell (Windows) ###\n")
        print(f"New-Item -ItemType Directory -Force -Path {q(raw)} | Out-Null")
        print(f"New-Item -ItemType Directory -Force -Path {q(out)} | Out-Null")
        print("\n# Option A: Download via magnet URIs (requires aria2c)")
        print(f'aria2c --dir="{raw}" --summary-interval=5 --seed-time=0 "{tmag}"')
        print(f'aria2c --dir="{raw}" --summary-interval=5 --seed-time=0 "{vmag}"')
        print(f'# (optional) devkit:')
        print(f'aria2c --dir="{raw}" --summary-interval=5 --seed-time=0 "{dmag}"')
        print("\n# Option B: Authenticated HTTP with cookies (curl). You must supply valid URLs.")
        print(f'curl -L --cookie "{cookies}" -o "{TRAIN_TAR}" "{train_url}"')
        print(f'curl -L --cookie "{cookies}" -o "{VAL_TAR}"   "{val_url}"')
        print(f'# (optional) devkit:')
        print(f'curl -L --cookie "{cookies}" -o "{DEVKIT_TGZ}" "{dev_url}"')
        print("\n# (Recommended) Verify checksums manually before extraction (example shows SHA1):")
        print(f'Get-FileHash -Algorithm SHA1 "{TRAIN_TAR}"')
        print(f'Get-FileHash -Algorithm SHA1 "{VAL_TAR}"')
        print(f'Get-FileHash -Algorithm SHA1 "{DEVKIT_TGZ}"  # if downloaded')
        print("\n# Prepare the final train/ and val/ tree (devkit reorganizes val/ into class folders):")
        print(f'python "{Path(__file__)}" imagenet-full '
              f'--train-tar "{TRAIN_TAR}" --val-tar "{VAL_TAR}" '
              f'--devkit "{DEVKIT_TGZ}" --out "{out}"')
        if wnids:
            so = subset_out or (out.parent / (out.name + "-subset"))
            print("\n# (Optional) Create subset from the full copy using your WNID list:")
            print(f'python "{Path(__file__)}" subset --src "{out}" --wnids "{wnids}" '
                  f'--out "{so}" --copy false')
        print("\n# Final dataset path to use in training scripts:")
        print(f'#   --imagenet "{out}"   # contains train/ and val/')
        if wnids:
            so = subset_out or (out.parent / (out.name + "-subset"))
            print(f'#   --imagenet "{so}"   # for subset runs')

    print("\n(End of commands)\n")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="ImageNet dataset helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("imagenette"); p.add_argument("--out", required=True); p.add_argument("--size", type=int, default=320); p.set_defaults(func=cmd_imagenette)
    p = sub.add_parser("imagewoof");  p.add_argument("--out", required=True); p.add_argument("--size", type=int, default=320); p.set_defaults(func=cmd_imagewoof)
    p = sub.add_parser("tiny-imagenet"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_tiny_imagenet)

    p = sub.add_parser("imagenet-full")
    p.add_argument("--train-tar", required=True, help="Path to ILSVRC2012_img_train.tar")
    p.add_argument("--val-tar", required=True, help="Path to ILSVRC2012_img_val.tar")
    p.add_argument("--devkit", default=None, help="Path to ILSVRC2012_devkit_t12.tar.gz (optional, for val reorg)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_imagenet_full)

    p = sub.add_parser("subset")
    p.add_argument("--src", required=True, help="Path to full ImageNet root with train/ and val/")
    p.add_argument("--wnids", required=True, help="Text file with one WNID per line")
    p.add_argument("--out", required=True, help="Where to create the subset")
    p.add_argument("--copy", default="false", help="true to copy files, false to symlink (default)")
    p.set_defaults(func=cmd_subset)

    p = sub.add_parser("fetch-tars")
    p.add_argument("--out", required=True)
    p.add_argument("--train-magnet", default=None, help="Magnet URI for the training tarball (user-supplied)")
    p.add_argument("--val-magnet",   default=None, help="Magnet URI for the validation tarball (user-supplied)")
    p.add_argument("--devkit-magnet", default=None, help="Magnet URI for the devkit tarball (user-supplied)")
    p.add_argument("--cookies", default=None, help="cookies.txt (Netscape format) for authenticated HTTP downloads")
    p.add_argument("--train-url", default=None, help="Authenticated HTTP URL for training tarball")
    p.add_argument("--val-url",   default=None, help="Authenticated HTTP URL for validation tarball")
    p.add_argument("--devkit-url", default=None, help="Authenticated HTTP URL for devkit tarball")
    p.set_defaults(func=cmd_fetch_tars)

    # NEW: print reproducible, OS-specific commands for the paper dataset
    p = sub.add_parser("imagenet-paper", help="Print commands to download & prepare the paper-level ImageNet-1k dataset (no embedded URLs).")
    p.add_argument("--raw", required=True, help="Directory where raw tarballs (train/val/devkit) will be stored")
    p.add_argument("--out", required=True, help="Final ImageNet root to create (contains train/ and val/)")
    p.add_argument("--shell", choices=["bash","powershell"], default=None, help="Force shell style for printed commands (auto-detect default)")
    # Optional auto-fill for commands (still user-provided)
    p.add_argument("--train-magnet", default=None, help="Your magnet URI for ILSVRC2012_img_train.tar")
    p.add_argument("--val-magnet",   default=None, help="Your magnet URI for ILSVRC2012_img_val.tar")
    p.add_argument("--devkit-magnet", default=None, help="Your magnet URI for ILSVRC2012_devkit_t12.tar.gz")
    p.add_argument("--cookies", default=None, help="Path to cookies.txt if using authenticated HTTP")
    p.add_argument("--train-url", default=None, help="Your authenticated HTTP URL for the training tarball")
    p.add_argument("--val-url",   default=None, help="Your authenticated HTTP URL for the validation tarball")
    p.add_argument("--devkit-url", default=None, help="Your authenticated HTTP URL for the devkit tarball")
    # Optional subset printing
    p.add_argument("--wnids", default=None, help="If provided, also print commands to subset using this WNIDs file")
    p.add_argument("--subset-out", default=None, help="Where to place the subset (default: <out>-subset)")
    p.set_defaults(func=cmd_imagenet_paper)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
