"""Retrieve the two identified source PDFs on a network-enabled machine.

Default: print immutable URLs only. --download explicitly opts into network.
This does not execute PDFs, access secrets, or assert redistribution rights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "audits/papers/jev-engineering-sources.json"


def source_url(manifest: dict, item: dict) -> str:
    if manifest["repository"] != "yibie/jev-engineering-zh":
        raise ValueError("unexpected source repository")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"]):
        raise ValueError("an immutable commit is required")
    path = item["path"]
    if path not in ("assets/original.pdf", "source-notes/original.pdf"):
        raise ValueError("unexpected PDF path")
    if not re.fullmatch(r"[a-z0-9-]+\.pdf", item["filename"]):
        raise ValueError("invalid destination filename")
    return f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/{path}"


def verify(data: bytes, item: dict) -> str:
    if len(data) != item["size_bytes"] or not data.startswith(b"%PDF-"):
        raise ValueError("PDF signature or byte length mismatch")
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if digest != item["git_blob_sha"]:
        raise ValueError("Git blob hash mismatch")
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, maximum: int) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read(maximum + 1)


def download(manifest: dict, item: dict, directory: Path) -> dict:
    url = source_url(manifest, item)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / item["filename"]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace {destination}")
    size = item["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 5_000_000:
        raise ValueError("unexpected source size")
    data = fetch(url, size)
    sha256 = verify(data, item)
    fd, temporary = tempfile.mkstemp(prefix=".paper-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        # Hard-link creation is exclusive even with a concurrent destination writer.
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return {"id": item["id"], "path": str(destination), "sha256": sha256,
            "git_blob_verified": True, "size_bytes": len(data), "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path("/tmp/omni-jev-papers"))
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text())
        records = [download(manifest, d, args.directory) if args.download else
                   {"id": d["id"], "url": source_url(manifest, d)} for d in manifest["documents"]]
        print(json.dumps({"downloaded": args.download, "documents": records}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
