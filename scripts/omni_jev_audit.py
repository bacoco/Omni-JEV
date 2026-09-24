#!/usr/bin/env python3
"""Metadata-only model audit. No model execution, weight downloads, or credentials.

A valid audit is not a ready model. `validate --require-ready` enforces that
separate gate. Hugging Face metadata hashes are not local file verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
MODALITIES = {"text", "image", "audio", "video", "audio_video"}
MAX_METADATA_BYTES = 8 * 1024 * 1024


class AuditError(ValueError):
    """Invalid evidence, metadata, or artifact boundary."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def valid_sha(value, pattern=SHA1):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def safe_path(value):
    require(isinstance(value, str) and bool(value), "empty artifact path")
    require(not any(c in value for c in ("\\", ":", "\x00")), "unsafe artifact path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ("", ".", "..") for x in value.split("/")),
            "unsafe artifact path")
    return p


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise AuditError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def write_new_json(path, value):
    """Never silently replace an existing lock or successful report."""
    target = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_audit(data, require_ready=False):
    require(isinstance(data, dict) and data.get("schema_version") == 1, "unknown audit schema")
    components = data.get("components")
    models = data.get("models")
    require(isinstance(components, list) and components, "no components")
    require(isinstance(models, list) and models, "no model candidates")
    seen = set()
    for component in components:
        ident = component.get("id")
        require(isinstance(ident, str) and ident and ident not in seen, "duplicate/invalid component ID")
        seen.add(ident)
        require(REPO.fullmatch(component.get("repo_id", "")), "invalid code repository")
        require(valid_sha(component.get("revision")), "code revisions must be full commit SHAs")
        require(component.get("inspected_files"), "missing inspected code files")
        paths = set()
        for artifact in component["inspected_files"]:
            path = str(safe_path(artifact.get("path")))
            require(path not in paths, "duplicate inspected file")
            paths.add(path)
            require(valid_sha(artifact.get("git_blob_sha1")), "missing inspected-file Git blob hash")
        require(component.get("runtime_status") == "not_run", "code inspection is not an inference run")
    seen = set()
    ready = []
    for model in models:
        ident = model.get("id")
        require(isinstance(ident, str) and ident and ident not in seen, "duplicate/invalid model ID")
        seen.add(ident)
        require(REPO.fullmatch(model.get("repo_id", "")), "invalid model repository")
        revision = model.get("revision")
        require(revision is None or valid_sha(revision), "model revision must be null or a full SHA")
        observed = model.get("observed_revision_hint")
        require(observed is None or valid_sha(observed), "invalid historical revision hint")
        require(set(model.get("declared_modalities", [])) <= MODALITIES, "invalid declared modality")
        runtime = model.get("runtime", {})
        status = runtime.get("status")
        require(status in {"not_run", "blocked", "failed", "passed"}, "invalid runtime status")
        if status == "passed":
            require(valid_sha(revision), "a model cannot pass without an immutable revision")
            require(valid_sha(runtime.get("verified_snapshot_lock_sha256"), SHA256), "missing snapshot lock digest")
            require(valid_sha(runtime.get("report_sha256"), SHA256), "missing inference report digest")
            require(runtime.get("device") and runtime.get("dtype"), "missing device/precision")
            require(runtime.get("output_shapes"), "missing measured output shapes")
            tested = runtime.get("tested_modalities", [])
            require(tested and set(tested) <= MODALITIES, "missing tested modality paths")
            ready.append(ident)
        else:
            require(runtime.get("reason"), "unexecuted candidate needs an explicit reason")
            require(not runtime.get("tested_modalities"), "unexecuted model claims tested modalities")
            require(not runtime.get("output_shapes"), "unexecuted model claims measured shapes")
    require(not require_ready or bool(ready), "no pretrained baseline has passed native smoke tests")
    return {"schema_valid": True, "code_revisions_pinned": len(components),
            "model_candidates": len(models), "runtime_evidence_declared_for": ready,
            "artifact_authenticity_proven": False}


def validate_lock(lock):
    require(isinstance(lock, dict) and lock.get("schema_version") == 1
            and lock.get("kind") == "hf_snapshot", "unknown snapshot schema")
    require(REPO.fullmatch(lock.get("repo_id", "")), "invalid model repository")
    require(valid_sha(lock.get("revision")), "snapshot needs an immutable revision")
    files = lock.get("files")
    require(isinstance(files, list) and files, "empty snapshot inventory")
    seen = set()
    for item in files:
        path = str(safe_path(item.get("path")))
        require(path not in seen, "duplicate snapshot path")
        seen.add(path)
        require(type(item.get("size")) is int and item["size"] >= 0, "invalid artifact size")
        sha256, git_sha = item.get("sha256"), item.get("git_blob_sha1")
        require((sha256 is None) != (git_sha is None), "each file needs exactly one content digest")
        require(valid_sha(sha256, SHA256) if sha256 is not None else valid_sha(git_sha),
                "invalid content digest")
    return lock


def snapshot_from_info(info, repo_id):
    require(isinstance(info, dict) and info.get("id") == repo_id, "metadata repository mismatch")
    require(valid_sha(info.get("sha")), "metadata lacks a full revision")
    siblings = info.get("siblings")
    require(isinstance(siblings, list) and siblings, "metadata has no file inventory")
    files = []
    for item in siblings:
        record = {"path": item.get("rfilename"), "size": item.get("size")}
        if item.get("lfs"):
            record["sha256"] = item["lfs"].get("sha256")
            require(record["size"] == item["lfs"].get("size"), "inconsistent LFS size")
        else:
            record["git_blob_sha1"] = item.get("blobId")
        files.append(record)
    lock = {"schema_version": 1, "kind": "hf_snapshot", "repo_id": repo_id,
            "revision": info["sha"], "captured_at": utc_now(),
            "hash_provenance": "server_metadata_not_locally_verified",
            "files": sorted(files, key=lambda f: str(f["path"])),
            "declared_license": (info.get("cardData") or {}).get("license"),
            "inference_tested": False}
    return validate_lock(lock)


def get_metadata(url):
    request = Request(url, headers={"User-Agent": "Omni-JEV-artifact-audit/1", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
    except HTTPError as exc:
        raise AuditError(f"metadata HTTP {exc.code}; this does not prove non-release") from exc
    except (URLError, OSError) as exc:
        raise AuditError(f"metadata transport failure: {type(exc).__name__}; no release conclusion") from exc
    require(len(raw) <= MAX_METADATA_BYTES, "oversized metadata response")
    return json.loads(raw)


def resolve_hf(repo_id, revision=None, fetch=get_metadata):
    require(isinstance(repo_id, str) and REPO.fullmatch(repo_id), "invalid model repository")
    require(revision is None or valid_sha(revision), "requested revision must be a full SHA")
    root = f"https://huggingface.co/api/models/{repo_id}"
    if revision is None:
        discovery = fetch(root + "?blobs=true")
        require(isinstance(discovery, dict) and discovery.get("id") == repo_id, "discovery repository mismatch")
        revision = discovery.get("sha")
        require(valid_sha(revision), "cannot resolve an immutable model revision")
    pinned = fetch(root + f"/revision/{revision}?blobs=true")
    require(isinstance(pinned, dict) and pinned.get("sha") == revision, "pinned revision mismatch")
    return snapshot_from_info(pinned, repo_id)


def verify_snapshot(lock, directory):
    validate_lock(lock)
    root = Path(directory).resolve(strict=True)
    require(root.is_dir(), "snapshot is not a directory")
    verified = []
    total = 0
    for item in lock["files"]:
        path = root.joinpath(*safe_path(item["path"]).parts).resolve(strict=True)
        require(path.is_relative_to(root), "snapshot symlink escapes the materialized directory")
        require(path.is_file() and path.stat().st_size == item["size"], f"size/type mismatch: {item['path']}")
        if item.get("sha256"):
            digest = hashlib.sha256()
            expected = item["sha256"]
        else:
            digest = hashlib.sha1(b"blob " + str(item["size"]).encode("ascii") + b"\x00")
            expected = item["git_blob_sha1"]
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        require(digest.hexdigest() == expected, f"content digest mismatch: {item['path']}")
        verified.append(item["path"])
        total += item["size"]
    return {"status": "verified", "revision": lock["revision"], "files": verified,
            "total_bytes": total, "inference_tested": False,
            "note": "Only inventoried files verified; this is not a code-safety or inference certificate."}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "colpali-engine", "peft", "Pillow",
                 "numpy", "huggingface-hub", "safetensors", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": packages, "inference_tested": False,
            "note": "Presence is not import compatibility. This command does not load a model."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("manifest", type=Path)
    p.add_argument("--require-ready", action="store_true")
    p = sub.add_parser("resolve-hf")
    p.add_argument("repo_id")
    p.add_argument("--revision")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("verify-snapshot")
    p.add_argument("lock", type=Path)
    p.add_argument("directory", type=Path)
    sub.add_parser("environment")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_audit(read_json(args.manifest), args.require_ready)
        elif args.command == "resolve-hf":
            result = resolve_hf(args.repo_id, args.revision)
            write_new_json(args.output, result)
        elif args.command == "verify-snapshot":
            result = verify_snapshot(read_json(args.lock), args.directory)
        else:
            result = environment()
    except (AuditError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(json.dumps({"status": "blocked_or_invalid", "error": str(exc),
                          "inference_tested": False}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
