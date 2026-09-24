"""Compare identical causal prompts with/without prefix reuse on a local model.

Usage: python -m omni_jev.prefix_benchmark --help
No model download, API call, training, or inference server is started.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import time
from pathlib import Path

from .contracts import DecisionRequest
from .prefix_backend import load_local_backend
from .prefix_cache import PROMPT_VERSION, PrefixDecisionEngine, Run


def read_cases(path: str | Path) -> list[tuple[str, DecisionRequest]]:
    cases, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if set(item) != {"case_id", "request"} or not isinstance(item["case_id"], str) or not item["case_id"].strip():
                raise ValueError("each row requires a nonempty case_id and request")
            if item["case_id"] in seen:
                raise ValueError("duplicate case_id")
            seen.add(item["case_id"])
            request = DecisionRequest.model_validate_json(json.dumps(item["request"]))
            cases.append((item["case_id"], request))
    if not cases:
        raise ValueError("no request cases supplied")
    return cases


def parity(left: Run, right: Run, *, atol: float) -> dict:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    lres = {r.question_id: r for r in left.response.results}
    rres = {r.question_id: r for r in right.response.results}
    same = left.response.state_id == right.response.state_id and lres.keys() == rres.keys()
    prob_delta, logit_delta = 0.0, 0.0
    for key in lres.keys() & rres.keys():
        a, b = lres[key], rres[key]
        same &= (a.type, a.status, a.metadata) == (b.type, b.status, b.metadata)
        if a.answer is None or b.answer is None:
            same &= a.answer is b.answer
            continue
        if a.answer.type != b.answer.type:
            same = False
            continue
        if a.type == "noul":
            pairs = [(a.answer.noul, b.answer.noul)]
        else:
            ap = {p.id: p.probability for p in a.answer.probabilities}
            bp = {p.id: p.probability for p in b.answer.probabilities}
            same &= ap.keys() == bp.keys()
            pairs = [(ap[k], bp[k]) for k in ap.keys() & bp.keys()]
            if a.type == "choice":
                same &= a.answer.choice == b.answer.choice
            else:
                same &= math.isclose(a.answer.score, b.answer.score, abs_tol=atol, rel_tol=0)
        prob_delta = max(prob_delta, *(abs(x - y) for x, y in pairs))
    lt, rt = {t.question_id: t for t in left.traces}, {t.question_id: t for t in right.traces}
    same &= lt.keys() == rt.keys()
    for key in lt.keys() & rt.keys():
        a, b = lt[key], rt[key]
        same &= (a.prompt_sha256, a.candidate_ids, a.codes, a.greedy_code) == (
            b.prompt_sha256, b.candidate_ids, b.codes, b.greedy_code)
        if len(a.logits) != len(b.logits):
            same = False
        else:
            logit_delta = max(logit_delta, *(abs(x - y) for x, y in zip(a.logits, b.logits)))
    return {"passed": bool(same and prob_delta <= atol and logit_delta <= atol and lt),
            "compared_questions": len(lt), "max_abs_logit_difference": logit_delta,
            "max_abs_probability_difference": prob_delta, "absolute_tolerance": atol}


def compare(engine: PrefixDecisionEngine, cases, *, repeats: int = 2, atol: float = 1e-5, scope="benchmark"):
    if type(repeats) is not int or repeats < 1 or not cases or len({k for k, _ in cases}) != len(cases):
        raise ValueError("positive repeats and unique, nonempty cases are required")
    samples, latest = [], {}
    engine.clear_cache()
    for repeat in range(repeats):
        for case_id, request in cases:
            # Alternate order to reduce systematic warm-up bias; retain every timing.
            order = (False, True) if repeat % 2 == 0 else (True, False)
            runs = {cached: engine.run(request, use_cache=cached, scope=scope) for cached in order}
            result = parity(runs[False], runs[True], atol=atol)
            samples.append({"case_id": case_id, "repeat": repeat, "order": list(order), "parity": result,
                "uncached": runs[False].report(), "cached": runs[True].report()})
            latest[case_id] = runs
    measured = [s for s in samples if s["parity"]["compared_questions"]]
    def summary(kind):
        times = [s[kind]["wall_seconds"] for s in measured]
        traces = [t for s in measured for t in s[kind]["traces"]]
        return {"median_request_seconds": statistics.median(times) if times else None,
                "total_request_seconds": sum(times),
                "computed_tokens": sum(t["computed_tokens"] for t in traces),
                "reused_tokens": sum(t["reused_tokens"] for t in traces),
                "cache_hits": sum(t["cache_hit"] for t in traces),
                "prefix_prefill_seconds": sum(t["prefill_seconds"] for t in traces),
                "cache_clone_seconds": sum(t["clone_seconds"] for t in traces)}
    report = {"format_version": 1, "prompt_version": PROMPT_VERSION, "prompt_mode": engine.mode,
        "quality_evaluated": False, "calibration_fitted": False,
        "cache_parity_passed": bool(measured) and all(s["parity"]["passed"] for s in measured),
        "unscored_cases": [s["case_id"] for s in samples if not s["parity"]["compared_questions"]],
        "scored_pair_count": len(measured), "repeats": repeats,
        "uncached": summary("uncached"), "cached": summary("cached"),
        "cache_at_end": engine.cache_info(), "samples": samples}
    return report, latest


def _write_jsonl(path, items):
    with path.open("x", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="local materialized safetensors model directory")
    parser.add_argument("--requests", required=True, help="JSONL {case_id,request}; no media URI is read")
    parser.add_argument("--output-dir", required=True, help="new directory; existing directories are refused")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mode", choices=("chat", "plain"), default="chat")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--cache-scope", default="local")
    args = parser.parse_args(argv)
    try:
        target = Path(args.output_dir)
        if target.exists():
            raise ValueError("output directory already exists")
        if args.repeats < 1 or not math.isfinite(args.atol) or args.atol < 0:
            raise ValueError("invalid repeats or tolerance")
        cases = read_cases(args.requests)
        start = time.perf_counter()
        backend = load_local_backend(args.snapshot, device=args.device, dtype=args.dtype, max_context=args.max_context)
        load_seconds = time.perf_counter() - start
        engine = PrefixDecisionEngine(backend, mode=args.mode)
        report, latest = compare(engine, cases, repeats=args.repeats, atol=args.atol, scope=args.cache_scope)
        report.update({"snapshot_hash_and_load_seconds": load_seconds,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "pydantic": importlib.metadata.version("pydantic"), "backend": backend.environment})
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        with (target / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        for cached, filename in ((True, "cached-predictions.jsonl"), (False, "uncached-predictions.jsonl")):
            _write_jsonl(target / filename, ({"case_id": key, "response": runs[cached].response.model_dump(mode="json")}
                                          for key, runs in latest.items()))
        _write_jsonl(target / "values.jsonl", ({"case_id": key, "values": {
            r.question_id: None if r.answer is None else getattr(r.answer, r.type)
            for r in runs[True].response.results}, "statuses": {r.question_id: r.status for r in runs[True].response.results}}
            for key, runs in latest.items()))
        print(json.dumps({"cache_parity_passed": report["cache_parity_passed"], "quality_evaluated": False,
                          "scored_pair_count": report["scored_pair_count"], "output_dir": str(target)}))
        return 0 if report["cache_parity_passed"] else 2
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        # Do not dump input text, model internals, credentials, or partial predictions.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "hint": "Check local snapshot, optional dependencies, context and tokenizer compatibility."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
