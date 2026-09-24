import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/omni_jev_fetch_papers.py"
spec = importlib.util.spec_from_file_location("papers", path)
papers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(papers)


def fixture():
    data = b"%PDF-1.4\nSynthetic bytes for integrity tests only.\n"
    item = {"id": "toy", "path": "assets/original.pdf", "filename": "toy.pdf",
            "size_bytes": len(data),
            "git_blob_sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
    manifest = {"repository": "yibie/jev-engineering-zh", "revision": "a" * 40}
    return data, item, manifest


def test_committed_sources_are_pinned():
    manifest = json.loads(papers.MANIFEST.read_text())
    assert len(manifest["documents"]) == 2
    for item in manifest["documents"]:
        assert manifest["revision"] in papers.source_url(manifest, item)


def test_verified_download_is_exclusive(tmp_path):
    data, item, manifest = fixture()
    with patch.object(papers, "fetch", return_value=data) as fetch:
        result = papers.download(manifest, item, tmp_path)
        assert result["sha256"] == hashlib.sha256(data).hexdigest()
        assert (tmp_path / "toy.pdf").read_bytes() == data
        with pytest.raises(FileExistsError): papers.download(manifest, item, tmp_path)
        assert fetch.call_count == 1


@pytest.mark.parametrize("field,value", [("path", "../bad.pdf"), ("filename", "../bad.pdf")])
def test_path_escape_rejected(field, value):
    data, item, manifest = fixture();item[field] = value
    with pytest.raises(ValueError): papers.source_url(manifest, item)


def test_mutable_revision_rejected():
    data, item, manifest = fixture();manifest["revision"] = "main"
    with pytest.raises(ValueError): papers.source_url(manifest, item)


@pytest.mark.parametrize("change", ["size", "hash", "signature"])
def test_corruption_rejected(change):
    data, item, _ = fixture()
    if change == "size": item["size_bytes"] += 1
    if change == "hash": item["git_blob_sha"] = "0" * 40
    if change == "signature": data = b"FAIL-" + data[5:]
    with pytest.raises(ValueError): papers.verify(data, item)


def test_symlink_not_overwritten(tmp_path):
    data, item, manifest = fixture()
    (tmp_path / "toy.pdf").symlink_to(tmp_path / "missing.pdf")
    with patch.object(papers, "fetch") as fetch:
        with pytest.raises(FileExistsError): papers.download(manifest, item, tmp_path)
        fetch.assert_not_called()


def test_failure_leaves_no_pdf(tmp_path):
    data, item, manifest = fixture()
    with patch.object(papers, "fetch", return_value=b"wrong"):
        with pytest.raises(ValueError): papers.download(manifest, item, tmp_path)
    assert not list(tmp_path.iterdir())
