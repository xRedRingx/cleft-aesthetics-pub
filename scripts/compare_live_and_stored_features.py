"""Is the stored embedding set what live extraction produces today?

    PYTHONPATH=src python scripts/compare_live_and_stored_features.py \
        --set data/embeddings/embeddings_v1/vit_b16__imagenet__g2 \
        --manifest data/manifests/cleft_v1 --staged data/staged/staged_v1

Fifteen Phase 7 arms came back depressed against numbers the same nominal
configurations measured earlier, and two factors differ at once: the ladder
runs at G2 where the 0.2529 bar was G1, and it reads stored pooled artifacts
where Phase 3 extracts live. This isolates the second, without fitting
anything.

----------------------------------------------------------------------------
WHAT IS ALREADY RULED OUT, AND WHAT THAT LEAVES
----------------------------------------------------------------------------
**[MEASURED 2026-08-01, laptop] The two extraction CODE PATHS are bitwise
identical.** ``torch_backbone.extract_embeddings`` builds the model at
``num_outputs=0`` and calls ``model(batch)``; ``extract.extract_features``
builds it at ``num_outputs=1`` and calls ``forward_features`` then
``forward_head(pre_logits=True)``. Given the same weights and the same input
those agree to 0.0 for both ViT-B/16 and Swin-B -- ``head`` is an Identity at
``num_classes=0``, and ``pre_logits=True`` stops before it.

So a computation difference is excluded. What is NOT excluded, and what this
script tests, is whether the STORED array is what that path would produce
now:

* **the wrong images** -- an artifact named ``__g2`` built from G1 pixels
  would be undetectable downstream; every shape, hash and pairing check
  passes. So the stored set is compared against live extraction at BOTH
  geometries, and matching the wrong one is the finding.
* **row order** -- the artifact records its own ``patient_ids``; a
  permutation trains every patient against another's label and still fits.
* **scale, dtype, precision** -- a float16 round trip or a missed
  normalisation shows up as a norm ratio, not as an error.
* **staleness** -- an artifact from an earlier code state.

It reports numbers rather than a verdict. A per-row correlation of 0.999 with
a scale factor of 3 is a different problem from a correlation of 0.2, and
collapsing both into "differs" would discard the part that says which.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from cleft import embeddings as emb  # noqa: E402
from cleft.data.manifest import load_manifest  # noqa: E402
from cleft.train import phase3  # noqa: E402


def compare(stored: np.ndarray, live: np.ndarray) -> dict:
    """Everything cheap that distinguishes one failure mode from another."""
    if stored.shape != live.shape:
        return {"same_shape": False, "stored": list(stored.shape),
                "live": list(live.shape)}

    delta = np.abs(stored - live)
    stored_norm = np.linalg.norm(stored, axis=1)
    live_norm = np.linalg.norm(live, axis=1)
    # Per-row correlation: insensitive to scale, so it separates "the same
    # features at a different scale" from "different features".
    rows = []
    for a, b in zip(stored, live):
        if a.std() > 0 and b.std() > 0:
            rows.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "same_shape": True,
        "identical": bool(np.array_equal(stored, live)),
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "stored_dtype": str(stored.dtype),
        "live_dtype": str(live.dtype),
        "mean_row_norm_stored": float(stored_norm.mean()),
        "mean_row_norm_live": float(live_norm.mean()),
        "norm_ratio": float(
            stored_norm.mean() / live_norm.mean() if live_norm.mean() else np.inf
        ),
        "mean_per_row_correlation": float(np.mean(rows)) if rows else None,
        "min_per_row_correlation": float(np.min(rows)) if rows else None,
    }


def permutation_check(stored: np.ndarray, live: np.ndarray) -> dict:
    """Is the stored array the live one in a DIFFERENT ORDER?

    The failure that produces a plausible number from misaligned data. Each
    stored row is matched to its nearest live row; if the best match is not
    the row's own index for most rows, the arrays hold the same content under
    a permutation -- which no hash or shape check can see.
    """
    if stored.shape != live.shape or len(stored) > 512:
        return {"checked": False, "why": "shape mismatch, or too large to pair"}
    # Squared distances via the expansion, to keep this cheap.
    d = (
        (stored**2).sum(axis=1)[:, None]
        + (live**2).sum(axis=1)[None, :]
        - 2.0 * stored @ live.T
    )
    nearest = np.argmin(d, axis=1)
    on_diagonal = int(np.sum(nearest == np.arange(len(stored))))
    return {
        "checked": True,
        "rows_whose_nearest_live_row_is_itself": on_diagonal,
        "n_rows": int(len(stored)),
        "looks_permuted": bool(
            on_diagonal < len(stored) and on_diagonal >= 0.5 * len(stored) is False
        ),
        "is_a_permutation_of_live": bool(
            on_diagonal < len(stored) and len(set(nearest.tolist())) == len(stored)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--staged", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    manifest_ids = [
        int(row["patient_id"])
        for row in load_manifest(args.manifest / "manifest.csv")
    ]
    stored, metadata = emb.load(args.set, manifest_ids=manifest_ids)
    print(f"stored set : {args.set.name}")
    print(
        f"  kind {metadata['kind']}, backbone {metadata['backbone']}, "
        f"init {metadata['init']}, geometry {metadata['geometry']}"
    )
    print(f"  shape {stored.shape} {stored.dtype}")

    if metadata["kind"] != "pooled":
        print("  this script compares POOLED sets; a feature map has no live "
              "single-call equivalent here")
        return 1
    if metadata["init"] != "imagenet":
        print(
            "  NOTE: live extraction is ImageNet-only, so a pretrained-init "
            "set has no live counterpart to compare against. Run this on the "
            "imagenet set of the same backbone and geometry."
        )
        return 1

    from cleft.train.torch_backbone import extract_embeddings

    # **Both geometries.** The sharpest available check is whether a set named
    # __g2 actually matches G1 pixels -- an artifact built from the wrong
    # images passes every hash, shape and pairing check downstream.
    results = {}
    for geometry in ("g1", "g2"):
        images, _, patient_ids, _ = phase3.load_inputs(
            args.manifest, args.staged, geometry, "mean"
        )
        # **The artifact's OWN backbone, not the default.**
        # ``extract_embeddings`` defaults to ``DEFAULT_BACKBONE`` (ViT-B/16),
        # so without this a Swin set would be compared against ViT features:
        # 1024 dims against 768, reported as a shape mismatch that says
        # nothing about the artifact. `swin_b__imagenet__g2` at 0.0065 is the
        # most diagnostic set there is, so the one that must work.
        live, _report = extract_embeddings(
            images, name=metadata["backbone"], batch_size=args.batch_size
        )
        live = np.asarray(live, dtype=stored.dtype)
        results[geometry] = compare(stored, live)
        results[geometry]["permutation"] = permutation_check(stored, live)
        if patient_ids != manifest_ids:
            print("  WARNING: staged row order differs from the manifest")

    print()
    for geometry, report in results.items():
        marker = "  <-- the set claims this geometry" if (
            geometry == metadata["geometry"]
        ) else ""
        print(f"live at {geometry}{marker}")
        for key, value in report.items():
            if key == "permutation":
                continue
            print(f"    {key:32s} {value}")
        perm = report["permutation"]
        if perm.get("checked"):
            print(
                f"    nearest-row diagonal              "
                f"{perm['rows_whose_nearest_live_row_is_itself']}/{perm['n_rows']}"
            )
        print()

    own = results[metadata["geometry"]]
    other = results["g1" if metadata["geometry"] == "g2" else "g2"]
    print("-" * 68)
    if own.get("identical"):
        print(
            "VERDICT: the stored set reproduces live extraction EXACTLY at the "
            "geometry it claims.\n"
            "         The feature source is not the cause; the remaining factor "
            "is geometry."
        )
        return 0
    if other.get("identical"):
        print(
            "VERDICT: the stored set matches live extraction at the OTHER "
            f"geometry ({'g1' if metadata['geometry'] == 'g2' else 'g2'}).\n"
            "         The artifact was built from the wrong images. Every hash, "
            "shape and pairing\n         check would pass, and nothing "
            "downstream could see it."
        )
        return 1
    print(
        "VERDICT: the stored set matches neither geometry exactly. Read the "
        "numbers above:\n"
        "  a high per-row correlation with norm_ratio != 1 is a scale or "
        "normalisation difference;\n"
        "  a low correlation with an off-diagonal nearest-row count is a row "
        "permutation;\n"
        "  a small max_abs_diff is a precision or dtype round trip."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
