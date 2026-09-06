"""Print the rollup hashes for a config's declared inputs, ready to paste.

    DO NOT MAKE THIS WRITE TO THE CONFIG.

    It would be a small, obvious "improvement" to have this update the
    rollup_sha256 fields in place instead of printing them. Do not. Guard 3
    exists to stop a run when the data has changed unexpectedly; a tool that
    rewrites the declared hash to match whatever is currently on disk makes the
    guard agree with reality by definition and therefore checks nothing. The
    friction of pasting the value is the point -- it makes declaring what the
    data is a deliberate act, performed once, by a person.


    PYTHONPATH=src python scripts/declare_inputs.py --config configs/p1_build_manifest.yaml

A config declares each input by path AND by hash, and guard 3 refuses to start if
they disagree. That is the point -- but it means a config for data you have not
hashed yet cannot be written by hand. This walks the declared paths, prints what
they actually hash to, and shows the YAML to paste in.

Run it once when the data arrives. After that the declared hash is a lock: if the
directory ever changes, every run stops until someone explains why.

Deliberately read-only. It never edits the config, because a tool that silently
rewrites a declared hash to match whatever is on disk turns the guard off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cleft.config import load_config  # noqa: E402
from cleft.provenance import hash_path  # noqa: E402
from cleft.provenance.context import resolve_declared_path  # noqa: E402

PLACEHOLDER = "0" * 64


def _as_declared(entry: dict) -> str:
    """What the config SAYS, not what it resolved to.

    The paste block has to echo the declaration. Printing the resolved path would
    invite pasting a machine-specific path back into a config whose whole point is
    that it does not contain one -- the tool would hand you the defect it was just
    fixed to avoid, and it would look like the tool's own suggestion.
    """
    return entry.get("path_declared_as") or entry["path"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    inputs = config["inputs"]
    if not inputs:
        print(f"{args.config} declares no inputs")
        return 0

    print(f"{args.config}: {len(inputs)} declared input(s)\n")
    problems = 0
    lines = []

    for entry in inputs:
        # **The run's own resolver, not a second copy of it.** This used to do
        # `Path(entry["path"]).is_absolute()` and join onto the repo root
        # otherwise -- a platform-NATIVE check, where the run uses a
        # platform-NEUTRAL one. On Linux `Path("C:/data/...").is_absolute()` is
        # False, so a laptop path was concatenated into
        # `/home/user/codex/cleft-aesthetics/C:/data/...` and reported MISSING:
        # a malformed declaration presenting as absent data. The predicate itself
        # was correct and tested all along; the defect was having two of them.
        path = resolve_declared_path(entry["path"], REPO)

        print(f"  {entry['name']}")
        if entry.get("path_from_env"):
            print(
                f"    declared : {entry['path_declared_as']}"
                f"  (from {entry['path_from_env']})"
            )
        print(f"    path     : {path}")

        if not path.exists():
            print("    MISSING  : this path does not exist on this machine\n")
            problems += 1
            lines.append((entry["name"], entry["path"], None))
            continue

        actual = hash_path(path)
        declared = entry["rollup_sha256"]
        state = (
            "placeholder"
            if declared == PLACEHOLDER
            else ("MATCHES" if declared == actual["rollup"] else "DISAGREES")
        )
        print(f"    files    : {actual['file_count']}, {actual['total_bytes']} bytes")
        print(f"    actual   : {actual['rollup']}")
        print(f"    declared : {declared}  [{state}]\n")
        if state == "DISAGREES":
            problems += 1
        lines.append((entry["name"], _as_declared(entry), actual["rollup"]))

    print("-" * 72)
    print("Paste into the config's inputs: block:\n")
    for name, path, rollup in lines:
        print(f"  - name: {name}")
        print(f"    path: {path}")
        print(f"    rollup_sha256: {rollup or '<path missing>'}")

    if problems:
        print(f"\n{problems} input(s) missing or disagreeing.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
