"""Offline audit tests. All model metadata and weights below are synthetic."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import omni_jev_audit as audit
import omni_jev_smoke_colmodernvbert as smoke


def metadata():
    return {"id": "org/model", "sha": "a" * 40, "siblings": [
        {"rfilename": "config.json", "size": 2,
         "blobId": hashlib.sha1(b"blob 2\x00{}").hexdigest()},
        {"rfilename": "model.safetensors", "size": 3, "blobId": "c" * 40,
         "lfs": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}]}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.data = audit.read_json(ROOT / "audits/OJ-001/manifest.json")

    def test_repository_manifest_is_valid_but_not_ready(self):
        result = audit.validate_audit(self.data)
        self.assertEqual(result["code_revisions_pinned"], 7)
        self.assertEqual(result["model_candidates"], 5)
        self.assertEqual(result["runtime_evidence_declared_for"], [])
        with self.assertRaises(audit.AuditError):
            audit.validate_audit(self.data, require_ready=True)

    def test_reject_mutable_code_revision(self):
        for revision in ("main", "v1", "1234567", None):
            with self.subTest(revision=revision):
                d = copy.deepcopy(self.data); d["components"][0]["revision"] = revision
                with self.assertRaises(audit.AuditError): audit.validate_audit(d)

    def test_reject_mutable_model_revision(self):
        self.data["models"][0]["revision"] = "main"
        with self.assertRaises(audit.AuditError): audit.validate_audit(self.data)

    def test_no_measured_shape_on_blocked_model(self):
        self.data["models"][0]["runtime"]["output_shapes"] = {"image": [1, 2048]}
        with self.assertRaises(audit.AuditError): audit.validate_audit(self.data)

    def test_no_tested_modality_on_blocked_model(self):
        self.data["models"][0]["runtime"]["tested_modalities"] = ["audio"]
        with self.assertRaises(audit.AuditError): audit.validate_audit(self.data)

    def test_pass_requires_revision_and_evidence(self):
        self.data["models"][0]["runtime"]["status"] = "passed"
        with self.assertRaises(audit.AuditError): audit.validate_audit(self.data, True)

    def test_pass_evidence_is_structural_not_an_authenticity_claim(self):
        model = self.data["models"][0]; model["revision"] = "b" * 40
        runtime = model["runtime"]
        runtime.update(status="passed", verified_snapshot_lock_sha256="d" * 64,
                       report_sha256="e" * 64, device="cpu", dtype="float32",
                       tested_modalities=["text"], output_shapes={"text": [1, 10]})
        self.assertFalse(audit.validate_audit(self.data, True)["artifact_authenticity_proven"])
        for field in ("report_sha256", "device", "output_shapes", "tested_modalities"):
            with self.subTest(field=field):
                d = copy.deepcopy(self.data); d["models"][0]["runtime"].pop(field)
                with self.assertRaises(audit.AuditError): audit.validate_audit(d, True)

    def test_reject_duplicate_identifiers(self):
        for key in ("models", "components"):
            with self.subTest(key=key):
                d = copy.deepcopy(self.data); d[key].append(d[key][0])
                with self.assertRaises(audit.AuditError): audit.validate_audit(d)

    def test_blocker_reason_is_mandatory(self):
        self.data["models"][0]["runtime"].pop("reason")
        with self.assertRaises(audit.AuditError): audit.validate_audit(self.data)

    def test_safe_path_rejects_traversal_and_ambiguity(self):
        for value in ("../x", "/x", "a/../x", "a//x", "./x", "C:/x", "a\\x", "x/", "", "x\x00y"):
            with self.subTest(value=value):
                with self.assertRaises(audit.AuditError): audit.safe_path(value)
        self.assertEqual(str(audit.safe_path("model/config.json")), "model/config.json")

    def test_metadata_distinguishes_lfs_content_from_pointer(self):
        lock = audit.snapshot_from_info(metadata(), "org/model")
        self.assertIn("git_blob_sha1", lock["files"][0])
        self.assertIn("sha256", lock["files"][1])
        self.assertNotIn("git_blob_sha1", lock["files"][1])
        self.assertFalse(lock["inference_tested"])

    def test_metadata_rejects_missing_digest(self):
        m = metadata(); m["siblings"][0].pop("blobId")
        with self.assertRaises(audit.AuditError): audit.snapshot_from_info(m, "org/model")

    def test_metadata_rejects_lfs_size_mismatch(self):
        m = metadata(); m["siblings"][1]["lfs"]["size"] = 4
        with self.assertRaises(audit.AuditError): audit.snapshot_from_info(m, "org/model")

    def test_metadata_rejects_invalid_size(self):
        for size in (True, -1, None, 1.2):
            with self.subTest(size=size):
                m = metadata(); m["siblings"][0]["size"] = size
                with self.assertRaises(audit.AuditError): audit.snapshot_from_info(m, "org/model")

    def test_metadata_rejects_duplicate_files(self):
        m = metadata(); m["siblings"].append(m["siblings"][0])
        with self.assertRaises(audit.AuditError): audit.snapshot_from_info(m, "org/model")

    def test_snapshot_needs_exactly_one_digest(self):
        lock = audit.snapshot_from_info(metadata(), "org/model")
        lock["files"][0]["sha256"] = "f" * 64
        with self.assertRaises(audit.AuditError): audit.validate_lock(lock)

    def test_resolve_uses_immutable_second_request(self):
        calls = []
        def fetch(url): calls.append(url); return metadata()
        lock = audit.resolve_hf("org/model", fetch=fetch)
        self.assertEqual(len(calls), 2)
        self.assertIn("/revision/" + "a" * 40, calls[-1])
        self.assertEqual(lock["revision"], "a" * 40)

    def test_explicit_revision_never_reads_main(self):
        calls = []
        def fetch(url): calls.append(url); return metadata()
        audit.resolve_hf("org/model", "a" * 40, fetch)
        self.assertEqual(len(calls), 1)
        self.assertIn("/revision/", calls[0])

    def test_resolver_rejects_wrong_identity(self):
        with self.assertRaises(audit.AuditError):
            audit.resolve_hf("org/other", fetch=lambda url: metadata())

    def test_resolver_rejects_changed_pinned_response(self):
        with self.assertRaises(audit.AuditError):
            audit.resolve_hf("org/model", "b" * 40, lambda url: metadata())

    def test_bad_repo_cannot_become_a_url(self):
        for name in ("org/model?token=x", "../model", "http://elsewhere", "org/model/path"):
            with self.subTest(name=name):
                with self.assertRaises(audit.AuditError):
                    audit.resolve_hf(name, fetch=lambda url: self.fail("network must not be called"))

    def test_metadata_error_is_not_non_release(self):
        errors = [HTTPError("https://huggingface.co", 401, "unauthorized", {}, None), URLError("DNS")]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch.object(audit, "urlopen", side_effect=error):
                with self.assertRaises(audit.AuditError) as caught: audit.get_metadata("https://huggingface.co/api/models/org/model")
                self.assertTrue("does not prove" in str(caught.exception) or "no release conclusion" in str(caught.exception))

    def test_metadata_size_is_bounded(self):
        class Response(io.BytesIO):
            pass
        with patch.object(audit, "urlopen", return_value=Response(b"x" * (audit.MAX_METADATA_BYTES + 1))):
            with self.assertRaises(audit.AuditError): audit.get_metadata("https://huggingface.co/api/models/org/model")

    def test_verify_real_bytes_with_both_digest_kinds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "config.json").write_bytes(b"{}"); (root / "model.safetensors").write_bytes(b"abc")
            result = audit.verify_snapshot(audit.snapshot_from_info(metadata(), "org/model"), root)
            self.assertEqual(result["total_bytes"], 5)
            self.assertFalse(result["inference_tested"])

    def test_verify_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "config.json").write_bytes(b"[]"); (root / "model.safetensors").write_bytes(b"abc")
            with self.assertRaises(audit.AuditError): audit.verify_snapshot(audit.snapshot_from_info(metadata(), "org/model"), root)

    def test_verify_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); inside = root / "snapshot"; inside.mkdir()
            (root / "config.json").write_bytes(b"{}"); (inside / "config.json").symlink_to(root / "config.json")
            with self.assertRaises(audit.AuditError): audit.verify_snapshot(audit.snapshot_from_info(metadata(), "org/model"), inside)

    def test_no_overwrite_of_existing_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lock.json"; audit.write_new_json(p, {"old": True})
            with self.assertRaises(FileExistsError): audit.write_new_json(p, {"old": False})
            self.assertEqual(audit.read_json(p), {"old": True})

    def test_duplicate_json_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}'):
                with self.subTest(text=text):
                    p.write_text(text)
                    with self.assertRaises(audit.AuditError): audit.read_json(p)

    def test_environment_does_not_claim_inference(self):
        self.assertFalse(audit.environment()["inference_tested"])

    def test_cli_readiness_fails(self):
        command = [sys.executable, str(ROOT / "scripts/omni_jev_audit.py"), "validate",
                   str(ROOT / "audits/OJ-001/manifest.json"), "--require-ready"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no pretrained baseline", result.stderr)

    def test_native_smoke_is_opt_in(self):
        with self.assertRaises(audit.AuditError): smoke.run("missing", "missing", "missing")

    def test_native_smoke_wrong_repo_blocked_before_import(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lock.json"; audit.write_new_json(p, metadata())
            with self.assertRaises(audit.AuditError): smoke.run(p, d, d, True)

    def test_native_smoke_failure_gets_a_report_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            report = Path(d) / "report.json"
            result = smoke.main(["--lock", "missing", "--snapshot", d, "--engine-source", d, "--report", str(report)])
            self.assertEqual(result, 2)
            self.assertEqual(audit.read_json(report)["status"], "blocked_or_failed")


if __name__ == "__main__":
    unittest.main()
