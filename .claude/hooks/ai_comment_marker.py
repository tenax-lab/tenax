#!/usr/bin/env python3
"""PreToolUse hook: require a 🤖 AI marker in the BODY of gh commands that post
to GitHub (``gh pr|issue comment|create``, ``gh pr review``).

Hardened over the original raw-``grep`` version (PR #623 review):

* It only fires on an *actual* ``gh`` invocation, not a mention. Tokenising with
  ``shlex`` (which respects quotes) means ``rg "gh pr create" docs`` parses to
  ``argv[0] == "rg"`` and is ignored — so harmless searches/echoes that contain
  these command strings are no longer denied.
* It checks the marker inside the ``--body``/``-b`` value specifically, not
  anywhere on the line. ``shlex.split(..., comments=True)`` also strips shell
  comments, so ``gh pr comment 1 --body "x" # 🤖`` no longer passes (the marker
  was in a comment GitHub never sees); a marker in ``--title`` or a filename no
  longer counts either.

Known best-effort limits (documented, not enforced): a body supplied via
``--body-file``/``-F`` or stdin/heredoc can't be inspected here, so those are
not blocked; and command substitution inside ``--body`` is matched on its
literal text. This is a convention nudge for the agent, not a security boundary.

Reads the PreToolUse event JSON on stdin; emits a ``deny`` decision (and exits 0)
when an unmarked posting command is detected, otherwise stays silent (allow).
"""
from __future__ import annotations

import json
import re
import shlex
import sys

_MARKER = re.compile(r"🤖|AI-generated|generated with", re.IGNORECASE)
# (subcommand, action) pairs that write to GitHub
_POSTING = {
    ("pr", "comment"),
    ("pr", "create"),
    ("pr", "review"),
    ("issue", "comment"),
    ("issue", "create"),
}
# tokens that separate one simple command from the next in a shell line
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "{", "}"}
_BODY_FLAGS = {"--body", "-b"}
_BODY_FILE_FLAGS = {"--body-file", "-F"}

_DENY_REASON = (
    "AI-comment convention (see CLAUDE.md): this gh command posts to GitHub but "
    "its --body has no 🤖 AI marker. Put a 🤖 AI label inside the --body/--body-file "
    "so AI comments are distinguishable from human ones."
)


def _deny() -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _DENY_REASON,
                }
            }
        )
    )


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a flat token list into simple-command segments on shell separators."""
    out: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if tok in _SEPARATORS:
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def _body_of(seg: list[str]) -> tuple[str, bool]:
    """Return (concatenated inline --body values, uses_body_file) for a gh segment."""
    bodies: list[str] = []
    uses_file = False
    i = 0
    while i < len(seg):
        tok = seg[i]
        if tok in _BODY_FLAGS and i + 1 < len(seg):
            bodies.append(seg[i + 1])
            i += 2
            continue
        if tok.startswith("--body="):
            bodies.append(tok.split("=", 1)[1])
            i += 1
            continue
        if tok in _BODY_FILE_FLAGS:
            uses_file = True
            i += 2
            continue
        i += 1
    return "\n".join(bodies), uses_file


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # not parseable -> stay out of the way (allow)
    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    try:
        tokens = shlex.split(cmd, comments=True)
    except ValueError:
        return  # unbalanced quotes etc. -> allow rather than block on a parse error

    for seg in _segments(tokens):
        if len(seg) < 3 or seg[0] != "gh":
            continue
        if (seg[1], seg[2]) not in _POSTING:
            continue
        body, uses_file = _body_of(seg[3:])
        if uses_file and not body:
            # body comes from a file/stdin we can't inspect — don't block.
            continue
        if not _MARKER.search(body):
            _deny()
            return


if __name__ == "__main__":
    main()
