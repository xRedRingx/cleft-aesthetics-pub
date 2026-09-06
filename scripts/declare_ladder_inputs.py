"""Hash every unresolved Phase 7 ladder input, once, and print one paste block.

    PYTHONPATH=src python scripts/declare_ladder_inputs.py

``declare_inputs.py`` does one config at a time, which for nineteen configs is
nineteen round trips and nineteen pastes. This walks them all, hashes each
DISTINCT path once — sixteen embedding sets are shared across the nineteen
configs, so hashing per config would hash the same directory repeatedly — and
prints a single block keyed by path.

**Read-only, for the reason declare_inputs.py gives at length.** It never
edits a config. A tool that rewrites the declared hash to whatever is on disk
makes guard 3 agree with reality by definition and therefore checks nothing;
the friction of pasting is what makes declaring the data a deliberate act.

**Paste by PATH, not per config.** A path's contents are immutable, so one
hash serves every config declaring it — and pasting per path makes the
cross-config invariant (``tests/test_smoke_run.py``) satisfied by
construction rather than by remembering to update the sibling. The one
already-observed failure of that was exactly a sibling forgotten.

Anything already declared with a real hash is reported as ALREADY and is not
re-hashed: it has been verified, and re-deriving it here would invite pasting
over a verified value with whatever is on disk today.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cleft.config import load_config  # noqa: E402
from cleft.provenance import hash_path  # noqa: E402
from cleft.provenance.context import (  # noqa: E402
    is_absolute_path,
    resolve_declared_path,
)

PLACEHOLDER = "0" * 64
UNRESOLVED = "UNRESOLVED_GLOB"

#: Characters that make a declared path a pattern rather than a location.
GLOB_CHARACTERS = "*?["


def expand_glob(pattern: str) -> list[str]:
    """Every path a declared pattern matches on THIS machine, sorted.

    Run directories are ``<config-stem>__<sha8>__<job-id>``, and neither the
    SHA nor the job id is derivable on a laptop -- so a config that consumes
    another run's outputs can only name it by pattern until someone stands on
    the cluster. Expanding here is what turns that pattern into a path the
    config can carry.

    **The caller refuses a pattern that matches more than once.** Choosing
    would be the dangerous behaviour: this project has three rounds of Phase
    7C arms on disk under the same config stems, two of them VOID.
    """
    from glob import glob as expand

    target = pattern if is_absolute_path(pattern) else str(REPO / pattern)
    return sorted(expand(target))


#: **The discovery pattern, and why it is not ``p7_*``.**
#:
#: [MEASURED 2026-08-02] It was ``p7_*.yaml``, which does not match
#: ``p7b_search.yaml`` -- so the tool reported "nothing to paste" while that
#: config carried twelve placeholders. **"Nothing to paste" reading as
#: "everything is declared" is the failure mode**, and it is the same shape as
#: every other check in this project that reported success on its own blind
#: spot: the tool was right about what it looked at and silent about what it
#: did not.
#:
#: Two fixes, because widening alone would leave the next gap just as quiet:
#: the pattern now covers the whole Phase 7 family, AND every config in
#: ``configs/`` that the pattern excluded is named in the output. A reader can
#: then see what was scanned rather than inferring it from a clean result.
#: **[MEASURED 2026-08-08] Widened a SECOND time, for the same reason.**
#: Road B's configs are ``roadb_p2_*.yaml`` and do not match ``p7*``, so the
#: tool would have reported "nothing to paste" over eight unresolved
#: artifacts. Identical to the p7b_search gap this pattern was widened for the
#: first time -- which is evidence that a single prefix is the wrong shape,
#: not that the prefix was wrong.
#:
#: It is a LIST now, and the NOT SCANNED report below is what makes the next
#: gap visible rather than quiet: a reader sees what was excluded instead of
#: inferring it from a clean result.
DISCOVERY_GLOBS = ("p7*.yaml", "roadb_*.yaml", "roadB_*.yaml")


def main(argv: list[str] | None = None) -> int:
    all_configs = sorted((REPO / "configs").glob("*.yaml"))
    configs = sorted({
        path for pattern in DISCOVERY_GLOBS
        for path in (REPO / "configs").glob(pattern)
    })
    if not configs:
        print(f"no {list(DISCOVERY_GLOBS)} configs found; run a generator first")
        return 1

    skipped = [path.name for path in all_configs if path not in configs]
    print(f"patterns {list(DISCOVERY_GLOBS)}: {len(configs)} scanned, "
          f"{len(skipped)} skipped")
    if skipped:
        # Grouped by prefix rather than listed: sixty-odd filenames is noise a
        # reader skims past, and the point is that the exclusion is VISIBLE,
        # not that every name is printed.
        groups: dict[str, int] = defaultdict(int)
        for name in skipped:
            groups[name.split("_", 1)[0]] += 1
        summary = ", ".join(
            f"{prefix}* x{count}" for prefix, count in sorted(groups.items())
        )
        print(f"  NOT SCANNED: {summary}")
        print("  (use scripts/declare_inputs.py --config for one of those)")

    # path -> {"names": {...}, "configs": [...], "declared": {...}}
    wanted: dict[str, dict] = defaultdict(
        lambda: {"names": set(), "configs": [], "declared": set()}
    )
    for config in configs:
        for entry in load_config(config)["inputs"]:
            record = wanted[entry["path"]]
            record["names"].add(entry["name"])
            record["configs"].append(config.name)
            record["declared"].add(entry["rollup_sha256"])

    print(f"{len(configs)} configs, {len(wanted)} distinct input paths\n")

    resolved: list[tuple] = []
    already = missing = 0
    for path, record in sorted(wanted.items()):
        names = ", ".join(sorted(record["names"]))
        # A path declared under two different names in different configs is a
        # naming slip worth seeing, not an error -- guard 3 keys on the hash.
        if len(record["names"]) > 1:
            print(f"  NOTE   {path}\n         declared as {names}")

        real = {h for h in record["declared"] if h != PLACEHOLDER}
        if len(real) > 1:
            print(
                f"  CONFLICT {path}\n"
                f"           declared with {len(real)} different hashes across "
                f"{len(record['configs'])} config(s) -- a path is immutable, so "
                "fix this before pasting anything"
            )
            missing += 1
            continue
        if real and PLACEHOLDER not in record["declared"]:
            already += 1
            continue

        if UNRESOLVED in path:
            print(
                f"  UNRESOLVED {path}\n"
                "             a run directory carries its job id, which no "
                "laptop can derive; resolve the glob on this machine and "
                "paste the real path with its hash"
            )
            missing += 1
            continue

        if any(char in path for char in GLOB_CHARACTERS):
            matches = expand_glob(path)
            if not matches:
                print(
                    f"  NO MATCH {path}\n"
                    "           the glob matches nothing on this machine"
                )
                missing += 1
                continue
            if len(matches) > 1:
                # **Refuse rather than choose.** Phase 7C ran its arms three
                # times under the SAME config stems -- v1 (the augmenter never
                # ran) and v2 (every arm stopped at epoch 1) are both VOID and
                # both left prediction files behind. Picking the newest, or the
                # first, would silently pair void vectors and produce entirely
                # plausible intervals. The one safe answer is to stop and make
                # the maintainer name the run.
                listing = "\n".join(f"             {m}" for m in matches)
                print(
                    f"  AMBIGUOUS {path}\n"
                    f"            matches {len(matches)} paths -- refusing to "
                    "choose. Name the run explicitly in the config:\n"
                    f"{listing}"
                )
                missing += 1
                continue
            target = Path(matches[0])
            paste_path = matches[0]
            print(f"  RESOLVED {path}\n           -> {target}")
        else:
            target = resolve_declared_path(path, REPO)
            paste_path = path
        if not Path(target).exists():
            print(f"  MISSING  {path}\n           does not exist on this machine")
            missing += 1
            continue

        actual = hash_path(target)
        # For a glob, the paste carries the RESOLVED path: the config must end
        # up naming one run directory, not a pattern that could match a
        # different set tomorrow.
        resolved.append((paste_path, actual["rollup"], names, len(record["configs"])))

    print(
        f"\n{already} already verified, {len(resolved)} hashed, {missing} "
        "unresolved\n"
    )
    if not resolved:
        # **Never just "nothing to paste".** That sentence is true when every
        # hash is verified and equally true when the tool could not reach the
        # data, and a reader cannot tell which from the words alone -- which is
        # exactly how the p7b_search gap stayed quiet.
        if missing:
            print(
                f"NOTHING COULD BE HASHED, and {missing} input(s) are still "
                "unresolved -- see the lines above. This is NOT the same as "
                "everything being declared."
            )
            return 1
        print(
            f"nothing to paste: all {already} declared input(s) across "
            f"{len(configs)} scanned config(s) already carry a verified hash."
        )
        return 0

    print("=" * 72)
    print("PASTE BLOCK -- one entry per PATH. Set rollup_sha256 to the value")
    print("below in EVERY config declaring that path (the count is shown), or")
    print("the cross-config invariant will fail on the sibling you missed.")
    print("=" * 72 + "\n")
    for path, rollup, names, count in resolved:
        print(f"# {names}  ({count} config{'s' if count != 1 else ''})")
        print(f"  path: {path}")
        print(f"  rollup_sha256: {rollup}\n")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
