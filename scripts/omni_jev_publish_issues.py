#!/usr/bin/env python3
"""Publish the prepared Omni-JEV issues using an existing GitHub CLI login.

Dry-run by default. Writes require --apply. Reads the target repository and all
existing issues before posting. Serial reruns skip marked issues; existing bodies
are never edited. This tool does not commit files, change settings, or add labels.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "planning/omni-jev/issues.json"


class PublishError(RuntimeError):
    """A validation or GitHub operation failed; no automatic write retry occurs."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("issues", [])
    if not items or data.get("schema_version") != 1:
        raise PublishError("Expected a non-empty version-1 issue manifest.")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise PublishError("Duplicate planning IDs.")
    for item in items:
        if not re.fullmatch(r"OJ-\d{3}", item["id"]):
            raise PublishError(f"Invalid planning ID: {item['id']}")
        body_path = (ROOT / item["body_file"]).resolve()
        if not body_path.is_relative_to(ROOT) or not body_path.is_file():
            raise PublishError(f"Invalid body file: {item['body_file']}")
        body = body_path.read_text(encoding="utf-8")
        if marker(item["id"]) not in body:
            raise PublishError(f"Missing stable marker in {item['body_file']}")
        if len(body.encode("utf-8")) > 60000:
            raise PublishError(f"Body too large: {item['id']}")
        if not set(item.get("depends_on", [])).issubset(set(ids)):
            raise PublishError(f"Unknown dependency in {item['id']}")
    graph = {i["id"]: i.get("depends_on", []) for i in items}
    seen: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise PublishError("Dependency cycle in issue manifest.")
        if node in seen:
            return
        active.add(node)
        for dep in graph[node]:
            visit(dep)
        active.remove(node)
        seen.add(node)

    for ident in ids:
        visit(ident)
    return data["target_repository"], items


def marker(ident: str) -> str:
    return f"<!-- omni-jev:{ident} -->"


class GitHubClient:
    def _run(self, args: list[str], payload: Any = None) -> Any:
        if payload is not None:
            args += ["--input", "-"]
        try:
            result = subprocess.run(
                ["gh", *args], input=None if payload is None else json.dumps(payload),
                text=True, capture_output=True, timeout=90, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PublishError(f"GitHub CLI did not complete: {exc}") from exc
        if result.returncode:
            raise PublishError(result.stderr.strip() or "GitHub CLI failed.")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PublishError("GitHub CLI returned non-JSON output.") from exc

    def request(self, method: str, endpoint: str, payload: Any = None) -> Any:
        return self._run([
            "api", "--hostname", "github.com", "--method", method,
            "-H", "Accept: application/vnd.github+json", endpoint,
        ], payload)

    def all_issues(self, repo: str) -> list[dict[str, Any]]:
        pages = self._run([
            "api", "--hostname", "github.com", "--method", "GET",
            f"repos/{repo}/issues?state=all&per_page=100", "--paginate", "--slurp",
        ])
        return [issue for page in pages for issue in page if "pull_request" not in issue]


def replace_block(body: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.escape(start) + r".*?" + re.escape(end)
    return re.sub(pattern, lambda _: start + "\n" + content + "\n" + end,
                  body, flags=re.DOTALL)


def render_body(item: dict[str, Any], items: list[dict[str, Any]],
                known: dict[str, dict[str, Any]]) -> str:
    body = (ROOT / item["body_file"]).read_text(encoding="utf-8")
    deps = item.get("depends_on", [])
    dep_text = "None." if not deps else "\n".join(
        f"- [{dep}]({known[dep]['html_url']})" if dep in known
        else f"- `{dep}` (not published or linked in this run)." for dep in deps
    )
    body = replace_block(body, "dependencies", dep_text)
    if item["id"] == "OJ-000":
        lines = []
        for other in items:
            if other["id"] == "OJ-000":
                continue
            ident = other["id"]
            label = f"{ident} — {other['title']}"
            lines.append(f"- [{label}]({known[ident]['html_url']})" if ident in known
                         else f"- {label} (local draft; not linked in this run).")
        body = replace_block(body, "backlog", "\n".join(lines))
    return body


def publish(client: Any, repo: str, items: list[dict[str, Any]],
            selected: set[str], report: dict[str, Any]) -> None:
    info = client.request("GET", f"repos/{repo}")
    if info.get("full_name", "").casefold() != repo.casefold():
        raise PublishError("The repository response does not match the requested target.")
    if not info.get("has_issues", False):
        raise PublishError("Issues are disabled. No repository settings were changed.")
    existing = client.all_issues(repo)
    known: dict[str, dict[str, Any]] = {}
    for item in items:
        matches = [i for i in existing if marker(item["id"]) in (i.get("body") or "")]
        if len(matches) > 1:
            raise PublishError(f"Several issues contain {item['id']}; resolve the conflict first.")
        if matches:
            known[item["id"]] = matches[0]
        elif any(i.get("title") == item["title"] for i in existing):
            raise PublishError(f"Title conflict without a stable marker: {item['title']}")
    for item in items:
        ident = item["id"]
        if ident not in selected:
            continue
        if ident in known:
            report["items"].append({"id": ident, "action": "skipped_existing",
                                    "url": known[ident]["html_url"]})
            continue
        result = client.request("POST", f"repos/{repo}/issues", {
            "title": item["title"], "body": render_body(item, items, known),
        })
        # Record the response before verifying, so a later failure is not hidden.
        row = {"id": ident, "action": "created_unverified", "url": result.get("html_url"),
               "number": result.get("number")}
        report["items"].append(row)
        if not isinstance(result.get("number"), int):
            raise PublishError(f"Creation response missing an issue number for {ident}.")
        verified = client.request("GET", f"repos/{repo}/issues/{result['number']}")
        expected_prefix = f"https://github.com/{repo}/issues/".casefold()
        if (marker(ident) not in (verified.get("body") or "")
                or verified.get("title") != item["title"]
                or not verified.get("html_url", "").casefold().startswith(expected_prefix)):
            raise PublishError(f"Read-back verification failed for {ident}; check the report before rerunning.")
        known[ident] = verified
        row.update(action="created_verified", url=verified["html_url"])
        print(f"Created {ident}: {verified['html_url']}")
    report["completed"] = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="bacoco/Omni-JEV")
    parser.add_argument("--apply", action="store_true", help="Explicitly authorize GitHub issue writes.")
    parser.add_argument("--only", action="append", default=[], metavar="OJ-000",
                        help="Limit to a planning ID; repeat this flag to select several.")
    parser.add_argument("--report", type=Path, default=Path("PUBLICATION_RESULT.json"))
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
            raise PublishError("Repository must have the form owner/name.")
        _, items = load_manifest()
        ids = {i["id"] for i in items}
        selected = set(args.only) if args.only else ids
        if not selected.issubset(ids):
            raise PublishError("Unknown --only planning ID.")
        print(f"Target: {args.repo}; mode: {'APPLY' if args.apply else 'DRY RUN'}")
        for item in items:
            if item["id"] in selected:
                print(f"  {item['id']} {item['title']}")
        if not args.apply:
            print("No network calls or GitHub writes performed. Use --apply after reviewing the drafts.")
            return 0
        if shutil.which("gh") is None:
            raise PublishError("GitHub CLI is required. Install gh and sign in with gh auth login.")
        report: dict[str, Any] = {"repository": args.repo, "completed": False, "items": [],
                                 "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            publish(GitHubClient(), args.repo, items, selected, report)
        except PublishError as exc:
            report["error"] = str(exc)
            raise
        finally:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Publication report: {args.report}")
        return 0
    except (PublishError, OSError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
