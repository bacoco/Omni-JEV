# Publishing this bootstrap

## Current state

The initial documentation and publication tooling have been committed to `main` in `bacoco/Omni-JEV`, and all 13 issues have been created through the authenticated GitHub integration. The repository was inspected and found empty before initialization. Existing issue searches were empty before publication. No repository settings, license, labels, or assignees were changed.

See [PUBLICATION_RESULT.json](PUBLICATION_RESULT.json) for the real issue URLs and verification record. Preparation-only status is preserved in Git history. The following instructions remain useful for future imports or other repositories; preserve existing work and review conflicts before writing.

## Route A: authenticated GitHub integration

Connect the GitHub integration to the intended account and repository with the access needed for the requested writes. Read the actual repository and existing issues first. Then use the issue manifest and Markdown bodies to create the missing issues and publish the documentation on a reviewable branch. Return only real issue/commit URLs after verifying the writes.

The local drafts are under `planning/omni-jev/issues/`; `OJ-000` is a self-contained synthesis suitable for a single initial issue. Stable planning IDs are not GitHub issue numbers.

## Route B: publish issues with GitHub CLI

Requirements: Python 3.10+ and an installed GitHub CLI authenticated to the intended account. Run from the extracted package directory. No access token needs to be pasted into this package or a chat.

```bash
# Sign in using the normal GitHub CLI flow, if not already authenticated.
gh auth login

# Local preview: no network calls or writes.
python scripts/omni_jev_publish_issues.py --repo bacoco/Omni-JEV

# Create the twelve task issues and then the linked synthesis issue.
python scripts/omni_jev_publish_issues.py --repo bacoco/Omni-JEV --apply
```

To publish only the synthesis instead:

```bash
python scripts/omni_jev_publish_issues.py --repo bacoco/Omni-JEV --only OJ-000 --apply
```

The helper first reads the actual repository and all existing issues. It recognizes its stable markers on serial reruns and skips existing marked issues without editing their bodies. A matching unmarked title stops the import for review. It does not push files, add labels, assign users, close issues, or change repository settings.

Every created issue is read back. `PUBLICATION_RESULT.json` records real URLs and whether creation was verified. No automatic retry is performed after an ambiguous write failure; inspect the result, then rerun so existing markers can prevent duplication. Do not run two imports concurrently. Repository-wide atomicity is not promised: a failure can leave a successfully published prefix of the backlog.

Existing issues, including an already-published synthesis, are not automatically refreshed. A later full import will not rewrite an earlier synthesis-only post; review its index manually when needed. The publication report still lists the real task URLs.

## Commit the documentation separately

Use the normal repository workflow, inspecting existing contents first. The namespaced documentation lives under `docs/omni-jev/`, and the backlog under `planning/omni-jev/`. Merge the supplied root README into an existing README rather than overwriting it blindly. The script and its tests are optional tooling, not model code.

A documentation branch is preferable when the repository already has a base branch. Review the diff, avoid force pushes, and create a pull request when appropriate. For a genuinely empty repository, establish the first commit according to the repository owner's chosen default branch. This package does not assume its name.

## Local verification

```bash
python -m unittest discover -s tests -v
python scripts/omni_jev_publish_issues.py --only OJ-000
```

The tests use fake GitHub responses. They validate the helper's dry-run behavior, serial idempotence, conflict checks, dependency links, and failure reporting. They do **not** establish network authentication, target-repository permissions, successful remote publication, or any model behavior.

## Source references

https://cli.github.com/manual/gh_api
https://cli.github.com/manual/gh_issue_create
