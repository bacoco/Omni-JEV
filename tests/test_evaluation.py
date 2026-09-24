"""Synthetic, offline contract/metric tests. Never a model benchmark."""
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
from omni_jev.evaluation import Benchmark, Prediction, connected_groups, evaluate


def case(i="a", kind="noul", split="test", status="ok"):
    question = {"id": "q", "type": kind, "instructions": "Evaluate the supplied state.",
                "sources": [{"source_id": "s"}], "required_modalities": ["text"]}
    target = "yes"
    if kind == "choice":
        question["options"] = [{"id": "x", "description": "First category"},
                               {"id": "y", "description": "Second category"}]
        target = "x"
    if kind == "score":
        question["levels"] = [{"id": f"l{j}", "description": f"Rubric level {j}", "index": j} for j in range(3)]
        target = "l2"
    return {"id": i, "split": split, "predicate_family": "toy", "language": "en",
            "source_groups": ["acquisition:" + i], "annotation_origin": "synthetic",
            "annotation_note": "Handcrafted unit-test data; not an observed model prediction.",
            "request": {"state": {"id": i, "sources": [{"id": "s", "modality": "text", "text": "Toy " + i}]},
                        "questions": [question]}, "gold_status": status,
            "target_id": target if status == "ok" else None}


def benchmark(*cases, held=()):
    return Benchmark.model_validate_json(json.dumps({"id": "toy", "cases": cases, "test_only_predicates": held}))


def pred(c, values=None, status="ok", revision="synthetic-not-a-model", timing=None):
    kind = c["request"]["questions"][0]["type"]
    result = {"question_id": "q", "type": kind, "status": status,
              "metadata": {"model_revision": revision,
                           "processed_sources": [{"source_id": "s", "modalities": ["text"]}]}}
    if status != "ok":
        result["reason"] = "Synthetic status test."
    elif kind == "noul":
        result["answer"] = {"type": "noul", "noul": .8 if values is None else values[0]}
    elif kind == "choice":
        values = values or [.8, .2]
        options = c["request"]["questions"][0]["options"]
        pairs = [{"id": o["id"], "probability": p} for o, p in zip(options, values)]
        result["answer"] = {"type": "choice", "choice": min(pairs, key=lambda x: (-x["probability"], x["id"]))["id"],
                            "probabilities": pairs}
    else:
        values = values or [.1, .2, .7]
        result["answer"] = {"type": "score", "score": sum(i * p for i, p in enumerate(values)),
                            "probabilities": [{"id": f"l{i}", "index": i, "probability": p} for i, p in enumerate(values)]}
    data = {"case_id": c["id"], "response": {"state_id": c["id"], "results": [result]}}
    if timing is not None:
        data["timing"] = timing
    return Prediction.model_validate_json(json.dumps(data))


def run(*cs, values=None, **kwargs):
    return evaluate(benchmark(*cs), tuple(pred(c, values=values) for c in cs), split="test", bootstrap_iterations=30, **kwargs)


def test_binary_arithmetic():
    r = run(case())
    assert r["answered_metrics"]["nll"] == pytest.approx(-math.log(.8))
    assert r["answered_metrics"]["multiclass_brier"] == pytest.approx(.08)
    assert r["answered_metrics"]["toplabel_ece"] == pytest.approx(.2)
    assert r["answerable_top1_success"] == 1


def test_ordinal_arithmetic_and_top1_distinction():
    r = run(case(kind="score"))
    assert r["score"]["expected_index_mae"] == pytest.approx(.4)
    assert r["score"]["normalized_ranked_probability_score"] == pytest.approx(.05)
    assert r["answered_metrics"]["top1_accuracy"] == 1


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_three_types(kind):
    r = run(case(kind=kind))
    assert r["synthetic_only"] is True
    assert r["slices"]["type"][kind]["n"] == 1


def test_choice_reordering():
    c = case(kind="choice")
    before = run(c)
    c["request"]["questions"][0]["options"].reverse()
    after = run(c, values=[.2, .8])
    assert before["answered_metrics"] == after["answered_metrics"]


@pytest.mark.parametrize("mutation", ["group", "hash", "ancestor", "text"])
def test_leakage_rejected(mutation):
    a, b = case("a", split="train"), case("b")
    if mutation == "group":
        b["source_groups"] = a["source_groups"]
    elif mutation == "hash":
        for c in (a, b): c["request"]["state"]["sources"][0]["sha256"] = "a" * 64
    elif mutation == "ancestor":
        a["request"]["state"]["sources"][0]["sha256"] = "b" * 64
        b["ancestor_sha256"] = ["b" * 64]
    else:
        b["request"]["state"]["sources"][0]["text"] = "Toy a"
    with pytest.raises(ValueError, match="leakage"):
        benchmark(a, b)


def test_transitive_multisource_cluster():
    a, b, c = case("a"), case("b"), case("c")
    a["source_groups"] = ["origin:x"]
    b["source_groups"] = ["origin:x", "origin:y"]
    c["source_groups"] = ["origin:y"]
    bm = benchmark(a, b, c)
    assert len(set(connected_groups(bm.cases).values())) == 1


def test_request_local_source_ids_not_global():
    assert len(set(connected_groups(benchmark(case("a"), case("b")).cases).values())) == 2


@pytest.mark.parametrize("split", ["train", "validation", "calibration"])
def test_predicate_holdout_includes_calibration(split):
    with pytest.raises(ValueError, match="held-out"):
        benchmark(case("a", split=split), held=["toy"])


def test_unknown_declared_holdout():
    with pytest.raises(ValueError, match="no examples"):
        benchmark(case(), held=["missing"])


def test_unavailable_does_not_become_no():
    a, b = case("a"), case("b", status="insufficient_evidence")
    r = evaluate(benchmark(a, b), (pred(a), pred(b, status="insufficient_evidence")), split="test")
    assert r["answerable_cases"] == 1
    assert r["unavailable_evidence"] == {"n": 1, "answered": 0, "recognized": 1}


def test_unsupported_not_recognized_as_missing_evidence():
    c = case(status="insufficient_evidence")
    r = evaluate(benchmark(c), (pred(c, status="unsupported_modality"),), split="test")
    assert r["unavailable_evidence"]["recognized"] == 0
    assert r["answered_metrics"]["nll"] is None


def test_fabricated_answer_counted():
    c = case(status="insufficient_evidence")
    r = evaluate(benchmark(c), (pred(c),), split="test")
    assert r["unavailable_evidence"]["answered"] == 1


def test_abstention_denominator():
    a, b = case("a"), case("b")
    r = evaluate(benchmark(a, b), (pred(a), pred(b, status="abstained")), split="test")
    assert r["answerable_top1_success"] == .5
    assert r["answered_metrics"]["top1_accuracy"] == 1
    assert r["answered_metrics"]["risk_coverage"][-1]["coverage_of_answerable"] == .5


def test_ties_are_admitted_together():
    a, b = case("a"), case("b")
    b["target_id"] = "no"
    r = run(a, b)
    curve = r["answered_metrics"]["risk_coverage"]
    assert len(curve) == 2
    assert curve[-1]["accepted"] == 2
    assert curve[-1]["top1_risk"] == .5


def test_zero_probability_is_finite():
    r = run(case(), values=[0.0])
    assert r["answered_metrics"]["nll"] == pytest.approx(-math.log(1e-12))
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_prediction_coverage(mode):
    c = case()
    preds = () if mode == "missing" else (pred(c), pred(case("b"))) if mode == "extra" else (pred(c), pred(c))
    with pytest.raises(ValueError): evaluate(benchmark(c), preds, split="test")


def test_mixed_model_revisions_rejected():
    a, b = case("a"), case("b")
    with pytest.raises(ValueError, match="revision"):
        evaluate(benchmark(a, b), (pred(a), pred(b, revision="other")), split="test")


def test_mismatched_response_ids_rejected():
    c = case()
    p = pred(c).model_dump(mode="json")
    p["response"]["state_id"] = "wrong"
    with pytest.raises(ValueError, match="different state"):
        evaluate(benchmark(c), (Prediction.model_validate_json(json.dumps(p)),), split="test")


@pytest.mark.parametrize("field,value", [("bins", 0), ("bins", 101), ("bootstrap_iterations", -1), ("bootstrap_iterations", 10001)])
def test_configuration_limits(field, value):
    c = case()
    with pytest.raises(ValueError): evaluate(benchmark(c), (pred(c),), split="test", **{field: value})


def test_bootstrap_reproducible_and_cluster_aware():
    a, b = case("a"), case("b")
    b["target_id"] = "no"
    first = run(a, b)["answerable_top1_success_interval"]
    assert first == run(b, a)["answerable_top1_success_interval"]
    assert first["groups"] == 2
    assert first["low"] == 0 and first["high"] == 1
    assert run(a)["answerable_top1_success_interval"]["low"] is None


def test_timings_and_cache_accounting():
    a, b = case("a"), case("b")
    predictions = (pred(a, timing={"wall_ms": 100.0}), pred(b, timing={"wall_ms": 10.0, "cache_hit": True}))
    r = evaluate(benchmark(a, b), predictions, split="test")
    assert r["wall_by_cache_ms"]["cold"]["p50"] == 100
    assert r["wall_by_cache_ms"]["warm"]["p50"] == 10


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_timing(bad):
    with pytest.raises(ValueError): pred(case(), timing={"wall_ms": bad})


def test_source_without_audio_cannot_have_gold_speech_answer():
    c = case()
    c["request"]["state"]["sources"][0] = {"id": "s", "modality": "video", "uri": "fixture://not-loaded", "has_audio": False}
    c["request"]["questions"][0]["required_modalities"] = ["audio"]
    with pytest.raises(ValueError, match="modalities"): benchmark(c)
    c["gold_status"], c["target_id"] = "insufficient_evidence", None
    assert benchmark(c).cases[0].gold_status == "insufficient_evidence"


def test_missing_required_source_rejected():
    c = case()
    c["request"]["state"]["sources"][0]["available"] = False
    with pytest.raises(ValueError, match="unavailable"): benchmark(c)


def test_synthetic_target_cannot_mark_unavailable_as_false():
    c = case(status="insufficient_evidence")
    c["target_id"] = "no"
    with pytest.raises(ValueError, match="negative target"): benchmark(c)


def test_duplicate_case_ids():
    with pytest.raises(ValueError, match="case IDs"): benchmark(case(), case())


def test_two_questions_rejected():
    c = case()
    q = copy.deepcopy(c["request"]["questions"][0]);q["id"] = "q2"
    c["request"]["questions"].append(q)
    with pytest.raises(ValueError, match="one question"): benchmark(c)


def test_empty_requested_split():
    with pytest.raises(ValueError, match="empty"):
        evaluate(benchmark(case(split="train")), (), split="test")


def test_probability_metrics_match_reference_random_vectors():
    rng = random.Random(42)
    for _ in range(25):
        raw = [rng.random() for _ in range(3)]
        p = [x / sum(raw) for x in raw]
        r = run(case(kind="score"), values=p)
        assert r["score"]["expected_index_mae"] == pytest.approx(abs(p[1] + 2*p[2] - 2))
        assert r["score"]["normalized_ranked_probability_score"] == pytest.approx((p[0]**2+(p[0]+p[1])**2)/2)


def test_cli_validate_and_evaluate(tmp_path):
    c = case()
    dataset = tmp_path / "data.json"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(benchmark(c).model_dump_json())
    predictions.write_text(pred(c).model_dump_json() + "\n")
    cmd = [sys.executable, "-m", "omni_jev.evaluation", str(dataset)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    done = subprocess.run(cmd + ["--predictions", str(predictions), "--bootstrap-iterations", "10"], capture_output=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["synthetic_only"] is True
    assert subprocess.run(cmd + ["--require-reviewed"], capture_output=True).returncode == 2


def test_empty_answered_family_remains_visible():
    c = case()
    r = evaluate(benchmark(c), (pred(c, status="abstained"),), split="test")
    assert r["slices"]["family"]["toy"]["n"] == 0
    assert r["slices"]["family"]["toy"]["answerable_cases"] == 1


def test_compare_identifies_lost_decisions_not_just_agreement():
    from omni_jev.evaluation import compare_runs
    a, b = case("a"), case("b")
    r = compare_runs(benchmark(a, b), (pred(a), pred(b, [.1])), (pred(a, [.1]), pred(b)), split="test")
    assert r["transitions"]["correct_to_incorrect_or_unanswered"] == 1
    assert r["transitions"]["incorrect_or_unanswered_to_correct"] == 1
    assert r["answer_disagreements"] == 2


def test_compression_agreement_can_be_wrong():
    from omni_jev.evaluation import compare_runs
    c = case()
    r = compare_runs(benchmark(c), (pred(c, [.1]),), (pred(c, [.1]),), split="test")
    assert r["answer_disagreements"] == 0
    assert r["candidate_success"] == 0


def test_compare_rejects_incomplete_candidate():
    from omni_jev.evaluation import compare_runs
    c = case()
    with pytest.raises(ValueError): compare_runs(benchmark(c), (pred(c),), (), split="test")


def test_checked_in_fixture_roundtrip_and_evaluation():
    root = Path(__file__).resolve().parents[1]
    bm = Benchmark.model_validate_json((root / "datasets/oj003/synthetic-smoke.json").read_text())
    ps = tuple(Prediction.model_validate_json(line) for line in (root / "datasets/oj003/synthetic-predictions.jsonl").read_text().splitlines())
    result = evaluate(bm, ps, split="test", bootstrap_iterations=10)
    assert result["total_cases"] == 4
    assert result["synthetic_only"]
    assert Benchmark.model_validate_json(bm.model_dump_json()) == bm
