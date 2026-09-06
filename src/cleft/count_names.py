"""R11: a count named for its elements, in a collection with a non-element.

**The rule.** *A count constant whose NAME describes the elements, in a
collection that also contains an appended non-element, will be read as
the element count by someone downstream.* Name the two quantities
separately, or the reader who needs the element count will take the
total.

----------------------------------------------------------------------------
WHY IT IS A RULE AND NOT A THIRD RECORD
----------------------------------------------------------------------------
[CORRECTED 2026-08-14] It looked like three instances and it is **two** --
``srs_boxes``/``model_regions`` in ``roadb_tasks`` is the FIX for the
AG-Net episode, not a separate occurrence. Two is the more interesting
number, because the two are not independent:

**Both graph backbones append the whole image as a final node, and both
name the post-append total with a word describing the pre-append
elements.** So this is not a coincidence repeated -- it is one
architectural convention, and the naming trap follows from it. That
earns the rule a PREDICTION rather than a tally:

    the next instance is AG-Net's, when arm C's keypoint attention is
    extracted -- ``agnet`` has its own ``forward_from_features`` and its
    own 36/37 pair, and nothing has yet needed its per-region weights.

**Both were caught by reading, not by a test**, which is why the sweep
below exists: it cannot judge whether a count is miscounted, but it can
refuse to let a new ``X = Y + n`` definition pass unclassified.

----------------------------------------------------------------------------
WHAT IT COST, THE SECOND TIME
----------------------------------------------------------------------------
In Phase 8 the appended node would have entered arm B's node-weight gate
as a 27th "region". A shared near-constant element **inflates the
between-patient baseline**, and that baseline is the bar a randomised
vector must fall BELOW -- so the gate would have become more permissive
for a reason having nothing to do with explanation quality, and a pass
would have meant less. It would not have raised anything.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: R11, stated once.
RULE = (
    "a count constant whose NAME describes the elements, in a collection "
    "that also contains an appended non-element, will be read as the "
    "element count by someone downstream. Name the two quantities "
    "separately"
)

#: Where the rule lives. R10 ("read the API before calling it") is also a
#: code-side convention -- the PLAN's headings stop at R9 -- so R11 is
#: recorded here rather than creating a gap in a document that never had
#: R10 either.
NUMBERING = {
    "rule": "R11",
    "plan_headings_stop_at": "R9",
    "r10_is_also_code_side": True,
}

#: The instances, both in graph models, both from the same convention.
INSTANCES = {
    "agnet": {
        "pre_append": 36, "total": 37,
        "appended": "the whole image, as the last region",
        "named_by": "generate_srs's box count against region_count",
        "resolved_as": (
            "roadb_tasks reports srs_boxes and model_regions separately"
        ),
        "found_by": "reading a gate report that compared the two under one name",
    },
    "srgnn": {
        "pre_append": 26, "total": 27,
        "appended": "the whole feature map, appended before attention",
        "named_by": "N_ROIS against N_REGIONS",
        "resolved_as": (
            "node_weights.split_whole_image ranks the 26 and reports the "
            "whole-image weight separately"
        ),
        "found_by": "checking the count before writing the gate that reads it",
    },
}

#: The prediction the rule makes, registered before the run that would
#: test it.
PREDICTS = (
    "AG-Net's per-region weights, when arm C's keypoint attention is "
    "extracted: it has its own forward_from_features and its own 36/37 "
    "pair, and nothing has needed its weights yet"
)

#: ``X = Y + n`` definitions that are NOT collection counts. Parameter
#: arithmetic -- a Linear layer's weights plus its bias -- has the same
#: shape and none of the hazard, so it is classified rather than
#: excluded by a name heuristic that would rot.
PARAMETER_ARITHMETIC = {
    "models/agnet.py:fusion",
    "models/srgnn.py:weighted_attention",
}

#: Collection counts the sweep expects to find, by ``module:name``.
COLLECTION_COUNTS = {"models/srgnn.py:N_REGIONS"}


def count_plus_constant_definitions(root: Path) -> dict:
    """Every module-level ``X = Y + <int>``, as ``module:name -> text``.

    Deliberately syntactic. The sweep cannot tell a region count from a
    parameter count -- that is a judgement -- so it finds the SHAPE and
    requires each one to have been judged.
    """
    found = {}
    for path in sorted(Path(root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value
            if not isinstance(target, ast.Name):
                continue
            if (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.Name)
                and isinstance(value.right, ast.Constant)
                and isinstance(value.right.value, int)
            ):
                key = f"{path.relative_to(root).as_posix()}:{target.id}"
                found[key] = f"{target.id} = {value.left.id} + {value.right.value}"
    return found
