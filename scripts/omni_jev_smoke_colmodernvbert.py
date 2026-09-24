#!/usr/bin/env python3
"""Opt-in native ColModernVBERT smoke test using verified local artifacts only.

This is an integration runner, not a completed experiment or a decision benchmark.
It never installs packages or downloads models. Review the pinned upstream code
and prepare its environment separately before opting in.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from omni_jev_audit import (AuditError, environment, read_json, require,
                            utc_now, verify_snapshot, write_new_json)

ENGINE_REVISION = "97487f8871ff4d5d2284411fe61bdcd2cfe99894"
MODEL_REPO = "ModernVBERT/colmodernvbert-merged"
REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer.json",
                  "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"}


def check_engine(directory):
    root = Path(directory).resolve(strict=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=20).stdout.strip()
    require(git("rev-parse", "HEAD") == ENGINE_REVISION, "wrong colpali_engine commit")
    require(not git("status", "--porcelain", "--untracked-files=all"), "upstream checkout is not clean")
    require((root / "colpali_engine").is_dir(), "missing colpali_engine source")
    return root


def run(lock_path, snapshot, engine_source, allow_reviewed_code=False):
    require(allow_reviewed_code, "explicit --allow-reviewed-code is required")
    lock = read_json(lock_path)
    require(lock.get("repo_id") == MODEL_REPO, "not the audited ColModernVBERT candidate")
    require(REQUIRED_FILES <= {f["path"] for f in lock.get("files", [])}, "incomplete native model snapshot")
    verification = verify_snapshot(lock, snapshot)
    engine_root = check_engine(engine_source)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["USE_TF"] = "0"
    sys.path.insert(0, str(engine_root))
    import torch
    from PIL import Image, ImageDraw
    import colpali_engine
    require(Path(colpali_engine.__file__).resolve().is_relative_to(engine_root), "wrong imported engine location")
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    torch.manual_seed(0)
    torch.set_num_threads(2)
    local = str(Path(snapshot).resolve())
    processor = ColModernVBertProcessor.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    model, loading = ColModernVBert.from_pretrained(
        local, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        require(not loading.get(key), f"non-exact checkpoint load: {key}")
    model.to("cpu").eval()
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((30, 50, 180, 200), fill="red")
    texts = ["A rectangle.", "A synthetic image containing a red rectangle on a white background."]
    q_inputs = processor.process_texts(texts)
    d_inputs = processor.process_images([image])
    with torch.inference_mode():
        queries = model(**q_inputs)
        documents = model(**d_inputs)
        scores = processor.score(list(queries), list(documents), device="cpu")
    for name, tensor, inputs in (("queries", queries, q_inputs), ("documents", documents, d_inputs)):
        require(tensor.ndim == 3 and tensor.shape[-1] == model.dim, f"wrong {name} shape")
        require(bool(torch.isfinite(tensor).all()), f"non-finite {name} output")
        mask = inputs["attention_mask"].bool()
        require(tensor.shape[:2] == mask.shape and bool(mask.any(dim=1).all()), "invalid native mask")
        require(bool(torch.all(tensor[~mask] == 0)), "padding output is not zero")
        require(bool(torch.allclose(tensor[mask].norm(dim=-1), torch.ones_like(tensor[mask][:, 0]),
                                    atol=1e-4, rtol=1e-4)), "non-unit token embeddings")
    require(tuple(scores.shape) == (2, 1) and bool(torch.isfinite(scores).all()), "invalid score tensor")
    qmask, dmask = q_inputs["attention_mask"].bool(), d_inputs["attention_mask"].bool()
    # A single native document has no batch padding. Query padding contributes zero.
    reference = torch.stack([(q[qmask[i]] @ documents[0][dmask[0]].T).max(-1).values.sum()
                             for i, q in enumerate(queries)]).reshape(2, 1)
    require(bool(torch.allclose(scores, reference, atol=1e-4, rtol=1e-4)), "native MaxSim parity failed")
    return {"status": "passed", "timestamp": utc_now(), "model_repo": MODEL_REPO,
            "revision": lock["revision"], "engine_revision": ENGINE_REVISION,
            "verified_snapshot_lock_sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
            "device": "cpu", "dtype": str(queries.dtype), "tested_modalities": ["text", "image"],
            "output_shapes": {"text": list(queries.shape), "image": list(documents.shape), "scores": list(scores.shape)},
            "native_processor": processor.to_dict(), "scores": scores.tolist(),
            "snapshot_verification": verification, "environment": environment(),
            "fixture": "Procedurally drawn 256x256 rectangle; two fixed English strings; no user media.",
            "limitations": ["No semantic correctness assertion", "No audio/video path", "No model-quality benchmark"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--engine-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-reviewed-code", action="store_true")
    args = parser.parse_args(argv)
    require(not args.report.exists(), "report exists; choose a new output path")
    try:
        result = run(args.lock, args.snapshot, args.engine_source, args.allow_reviewed_code)
        code = 0
    except Exception as exc:
        result = {"status": "blocked_or_failed", "timestamp": utc_now(),
                  "error": f"{type(exc).__name__}: {exc}", "native_smoke_passed": False,
                  "environment": environment()}
        code = 2
    write_new_json(args.report, result)
    print(f"{result['status']}: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
