"""CPU-only benchmark validation and evaluation of supplied typed predictions.

No model invocation, downloading, calibration fitting, or deployment decisions.
See docs/omni-jev/evaluation-protocol.md for denominators and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from .contracts import (
    Contract, DecisionRequest, DecisionResponse, Identifier, Nonnegative,
    Text, unique, validate_response,
)

Split = Literal["train", "validation", "calibration", "test"]


def option_ids(request: DecisionRequest) -> tuple[str, ...]:
    q = request.questions[0]
    if q.type == "noul":
        return ("no", "yes")
    return tuple(x.id for x in (q.options if q.type == "choice" else q.levels))


class BenchmarkCase(Contract):
    id: Identifier
    split: Split
    predicate_family: Identifier
    language: Text
    # Namespace these globally: acquisition:..., speaker:..., document:..., etc.
    source_groups: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    ancestor_sha256: tuple[Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")], ...] = ()
    annotation_origin: Literal["synthetic", "human_reviewed", "unreviewed"]
    annotation_note: Text
    request: DecisionRequest
    gold_status: Literal["ok", "insufficient_evidence"]
    target_id: Identifier | None = None

    @model_validator(mode="after")
    def valid_target(self) -> Self:
        unique(self.source_groups, "source groups")
        unique(self.ancestor_sha256, "ancestor hashes")
        if len(self.request.questions) != 1:
            raise ValueError("one question per benchmark case; related cases share source_groups")
        if self.gold_status == "ok":
            if self.target_id not in option_ids(self.request):
                raise ValueError("answerable case requires a request-local target option")
            q = self.request.questions[0]
            sources = {s.id: s for s in self.request.state.sources}
            scoped = [sources[r.source_id] for r in q.sources] if q.sources else list(sources.values())
            if q.sources and any(not s.available for s in scoped):
                raise ValueError("answerable case references an unavailable source")
            modalities = {s.modality for s in scoped if s.available}
            if any(s.available and s.modality == "video" and s.has_audio is True for s in scoped):
                modalities.add("audio")
            if not modalities or not set(q.required_modalities) <= modalities:
                raise ValueError("gold answer lacks declared available modalities")
        elif self.target_id is not None:
            raise ValueError("insufficient evidence is not a negative target label")
        return self


class Benchmark(Contract):
    schema_version: Literal["0.1"] = "0.1"
    id: Identifier
    test_only_predicates: tuple[Identifier, ...] = ()
    cases: Annotated[tuple[BenchmarkCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def leakage_free(self) -> Self:
        unique(tuple(c.id for c in self.cases), "case IDs")
        unique(self.test_only_predicates, "test-only predicates")
        families = {c.predicate_family for c in self.cases}
        if not set(self.test_only_predicates) <= families:
            raise ValueError("declared held-out family has no examples")
        for c in self.cases:
            if c.predicate_family in self.test_only_predicates and c.split != "test":
                raise ValueError("held-out predicate appears outside test")
        groups = connected_groups(self.cases)
        splits: dict[str, set[str]] = defaultdict(set)
        for c in self.cases:
            splits[groups[c.id]].add(c.split)
        if any(len(s) != 1 for s in splits.values()):
            raise ValueError("source, ancestor or exact-content leakage across splits")
        return self


def connected_groups(cases: tuple[BenchmarkCase, ...]) -> dict[str, str]:
    """Union all lineage keys, including cross-linked/multi-source acquisitions.

    Request-local source IDs are NOT global identities. Declared hashes are
    grouped but not verified against external media bytes by this module.
    """
    parent = {c.id: c.id for c in cases}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for c in cases:
        keys = {"group:" + g for g in c.source_groups}
        keys.update("sha256:" + h for h in c.ancestor_sha256)
        for s in c.request.state.sources:
            if s.sha256:
                keys.add("sha256:" + s.sha256)
            if s.text is not None:
                keys.add("sha256:" + hashlib.sha256(s.text.encode("utf-8")).hexdigest())
        for key in sorted(keys):
            if key in owner:
                a, b = find(c.id), find(owner[key])
                parent[max(a, b)] = min(a, b)
            else:
                owner[key] = c.id
    return {c.id: find(c.id) for c in cases}


class Timing(Contract):
    media_decode_ms: Nonnegative | None = None
    state_encode_ms: Nonnegative | None = None
    question_ms: Nonnegative | None = None
    wall_ms: Nonnegative | None = None
    cache_hit: bool = False
    # Components may overlap: do not equate their sum with elapsed wall time.


class Prediction(Contract):
    case_id: Identifier
    response: DecisionResponse
    timing: Timing | None = None


def probabilities(case: BenchmarkCase, prediction: Prediction) -> dict[str, float] | None:
    validate_response(case.request, prediction.response)
    result = prediction.response.results[0]
    if result.status != "ok":
        return None
    answer = result.answer
    if answer.type == "noul":
        return {"no": 1.0 - answer.noul, "yes": answer.noul}
    return {p.id: p.probability for p in answer.probabilities}


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def distribution_metrics(rows: list[dict], bins: int) -> dict:
    """Only emitted answers on answerable cases; denominators are returned."""
    if not rows:
        return {"n": 0, "top1_accuracy": None, "nll": None, "multiclass_brier": None,
                "toplabel_ece": None, "reliability": [], "risk_coverage": []}
    n = len(rows)
    reliability = []
    for i in range(bins):
        selected = [r for r in rows if min(bins - 1, int(r["pmax"] * bins)) == i]
        if selected:
            reliability.append({"bin": i, "n": len(selected),
                                "mean_probability": _mean([r["pmax"] for r in selected]),
                                "accuracy": _mean([r["correct"] for r in selected])})
    ece = math.fsum(b["n"] / n * abs(b["mean_probability"] - b["accuracy"]) for b in reliability)
    curve = [{"threshold": None, "accepted": 0, "coverage_of_answered": 0.0, "top1_risk": None}]
    # Admit an entire tie group together; ordering cannot make the curve optimistic.
    by_confidence: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_confidence[row["pmax"]].append(row)
    accepted = errors = 0
    for threshold in sorted(by_confidence, reverse=True):
        group = by_confidence[threshold]
        accepted += len(group)
        errors += sum(1 - r["correct"] for r in group)
        curve.append({"threshold": threshold, "accepted": accepted,
                      "coverage_of_answered": accepted / n, "top1_risk": errors / accepted})
    return {"n": n, "top1_accuracy": _mean([r["correct"] for r in rows]),
            "nll": _mean([r["nll"] for r in rows]),
            "multiclass_brier": _mean([r["brier"] for r in rows]),
            "toplabel_ece": ece, "reliability": reliability, "risk_coverage": curve}


def group_interval(rows: list[dict], iterations: int, seed: int) -> dict:
    grouped: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        grouped[r["cluster"]].append(r["correct"])
    result = {"method": "cluster_percentile_bootstrap", "groups": len(grouped),
              "iterations": iterations, "seed": seed, "low": None, "high": None}
    if len(grouped) < 2 or iterations == 0:
        return result
    clusters = [grouped[g] for g in sorted(grouped)]
    rng = random.Random(seed)
    stats = []
    for _ in range(iterations):
        draw = [x for _ in clusters for x in rng.choice(clusters)]
        stats.append(math.fsum(draw) / len(draw))
    result.update(low=quantile(stats, .025), high=quantile(stats, .975))
    return result


def evaluate(benchmark: Benchmark, predictions: tuple[Prediction, ...], *, split: Split,
             bins: int = 10, bootstrap_iterations: int = 1000, seed: int = 0) -> dict:
    if split not in ("train", "validation", "calibration", "test"):
        raise ValueError("unknown evaluation split")
    if not 1 <= bins <= 100 or not 0 <= bootstrap_iterations <= 10000:
        raise ValueError("bins must be 1..100; bootstrap iterations must be 0..10000")
    cases = tuple(c for c in benchmark.cases if c.split == split)
    if not cases:
        raise ValueError("evaluation split is empty")
    unique(tuple(p.case_id for p in predictions), "prediction case IDs")
    indexed = {p.case_id: p for p in predictions}
    if set(indexed) != {c.id for c in cases}:
        raise ValueError("supply exactly all cases in the selected split; no missing/extra predictions")
    groups = connected_groups(benchmark.cases)
    rows, all_answerable, score_errors, rps = [], [], [], []
    status_counts, revisions = Counter(), set()
    unavailable = unavailable_answered = unavailable_recognized = 0
    timings: dict[str, list[float]] = defaultdict(list)
    cold_wall, warm_wall = [], []
    for case in cases:
        prediction = indexed[case.id]
        dist = probabilities(case, prediction)  # Invalid output aborts, never silently excluded.
        result = prediction.response.results[0]
        status_counts[result.status] += 1
        revisions.add((result.metadata.model_revision, result.metadata.calibration_revision))
        if prediction.timing is not None:
            t = prediction.timing
            for key in ("media_decode_ms", "state_encode_ms", "question_ms", "wall_ms"):
                value = getattr(t, key)
                if value is not None:
                    timings[key].append(value)
            if t.wall_ms is not None:
                (warm_wall if t.cache_hit else cold_wall).append(t.wall_ms)
        if case.gold_status != "ok":
            unavailable += 1
            unavailable_answered += dist is not None
            unavailable_recognized += result.status == "insufficient_evidence"
            continue
        winner = min(dist, key=lambda k: (-dist[k], k)) if dist is not None else None
        correct = int(winner == case.target_id)
        all_answerable.append({"cluster": groups[case.id], "correct": correct})
        if dist is None:
            continue
        row = {"id": case.id, "cluster": groups[case.id], "correct": correct,
               "family": case.predicate_family, "type": case.request.questions[0].type,
               "language": case.language, "pmax": max(dist.values()),
               "nll": -math.log(max(dist[case.target_id], 1e-12)),
               "brier": math.fsum((p - float(k == case.target_id)) ** 2 for k, p in dist.items())}
        rows.append(row)
        if row["type"] == "score":
            ids = option_ids(case.request)
            target = ids.index(case.target_id)
            mean = math.fsum(i * dist[k] for i, k in enumerate(ids))
            score_errors.append(abs(mean - target))
            cdf = 0.0
            terms = []
            for i, k in enumerate(ids[:-1]):
                cdf += dist[k]
                terms.append((cdf - float(target <= i)) ** 2)
            rps.append(math.fsum(terms) / (len(ids) - 1))
    if len(revisions) != 1:
        raise ValueError("one model/calibration revision pair per evaluation run")
    answered_metrics = distribution_metrics(rows, bins)
    for point in answered_metrics["risk_coverage"]:
        point["coverage_of_answerable"] = point["accepted"] / len(all_answerable) if all_answerable else None
    slices = {}
    for field in ("family", "type", "language"):
        def case_value(c):
            return c.predicate_family if field == "family" else c.request.questions[0].type if field == "type" else c.language
        slices[field] = {}
        for value in sorted({case_value(c) for c in cases}):
            subset = [c for c in cases if case_value(c) == value]
            metric = distribution_metrics([r for r in rows if r[field] == value], bins)
            metric["total_cases"] = len(subset)
            metric["answerable_cases"] = sum(c.gold_status == "ok" for c in subset)
            metric["status_counts"] = dict(Counter(indexed[c.id].response.results[0].status for c in subset))
            slices[field][value] = metric
    return {"schema_version": "0.1", "benchmark": benchmark.id, "split": split,
            "total_cases": len(cases), "answerable_cases": len(all_answerable),
            "answered_answerable_cases": len(rows), "status_counts": dict(status_counts),
            "model_revision": next(iter(revisions))[0], "calibration_revision": next(iter(revisions))[1],
            "annotation_origins": dict(Counter(c.annotation_origin for c in cases)),
            "synthetic_only": all(c.annotation_origin == "synthetic" for c in cases),
            "answered_metrics": answered_metrics, "slices": slices,
            "answerable_top1_success": _mean([r["correct"] for r in all_answerable]),
            "answerable_top1_success_interval": group_interval(all_answerable, bootstrap_iterations, seed),
            "score": {"answered_n": len(score_errors), "expected_index_mae": _mean(score_errors),
                      "normalized_ranked_probability_score": _mean(rps)},
            "unavailable_evidence": {"n": unavailable, "answered": unavailable_answered,
                                     "recognized": unavailable_recognized},
            "timing_ms": {k: {"n": len(v), "p50": quantile(v, .5), "p95": quantile(v, .95)}
                          for k, v in timings.items()},
            "wall_by_cache_ms": {"cold": {"n": len(cold_wall), "p50": quantile(cold_wall, .5)},
                                 "warm": {"n": len(warm_wall), "p50": quantile(warm_wall, .5)}},
            "notes": ["NLL clips zero probability at 1e-12.",
                      "Top-label concentration is not calibrated correctness.",
                      "Score top1 metrics evaluate a discrete level, not its expected index.",
                      "No media decoding, neural inference or calibration fitting occurs here."]}



def compare_runs(benchmark: Benchmark, reference: tuple[Prediction, ...],
                 candidate: tuple[Prediction, ...], *, split: Split) -> dict:
    """Paired correctness, not merely agreement, for compression/selection ablations.

    No statistical significance claim; use grouped task-level resampling for
    research conclusions. Non-ok outputs count as unanswered, not correct.
    """
    reports = [evaluate(benchmark, run, split=split, bootstrap_iterations=0)
               for run in (reference, candidate)]
    left, right = ({p.case_id: p for p in run} for run in (reference, candidate))
    transitions = Counter()
    disagreements = 0
    for c in benchmark.cases:
        if c.split != split or c.gold_status != "ok":
            continue
        winners = []
        for indexed in (left, right):
            dist = probabilities(c, indexed[c.id])
            winners.append(min(dist, key=lambda k: (-dist[k], k)) if dist is not None else None)
        flags = [w == c.target_id for w in winners]
        key = ("correct" if flags[0] else "incorrect_or_unanswered") + "_to_" + ("correct" if flags[1] else "incorrect_or_unanswered")
        transitions[key] += 1
        disagreements += winners[0] != winners[1]
    return {"split": split, "reference_success": reports[0]["answerable_top1_success"],
            "candidate_success": reports[1]["answerable_top1_success"],
            "transitions": dict(transitions), "answer_disagreements": disagreements,
            "reference_unavailable_evidence": reports[0]["unavailable_evidence"],
            "candidate_unavailable_evidence": reports[1]["unavailable_evidence"]}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "calibration", "test"], default="test")
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    try:
        benchmark = Benchmark.model_validate_json(args.benchmark.read_text())
        if args.require_reviewed and any(c.annotation_origin != "human_reviewed" for c in benchmark.cases):
            raise ValueError("dataset is not fully human-reviewed")
        if args.predictions:
            records = tuple(Prediction.model_validate_json(line)
                            for line in args.predictions.read_text().splitlines() if line.strip())
            report = evaluate(benchmark, records, split=args.split, bootstrap_iterations=args.bootstrap_iterations)
        else:
            report = {"valid": True, "cases": len(benchmark.cases),
                      "splits": dict(Counter(c.split for c in benchmark.cases)),
                      "source_components": len(set(connected_groups(benchmark.cases).values())),
                      "test_only_predicates": benchmark.test_only_predicates}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
