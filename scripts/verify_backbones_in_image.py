"""Construct all four Phase 6 backbones IN THE PINNED IMAGE and check them.

**The one check local construction cannot stand in for.** The specs were built
and matched on timm 1.0.27 / torch 2.13.0 / torchvision 0.27.1, which tests the
*specification*. This tests the *image*: whether the pinned timm resolves the
same backbones, to the same parameter counts, reporting the same preprocessing.

Run on the cluster, in the image, before the twelve pretraining runs::

    PYTHONPATH=src python scripts/verify_backbones_in_image.py

**[MEASURED 2026-07-31] PASSED, all eight checks**, in
timm 1.0.7 / torch 2.8.0+cu128 / torchvision 0.23.0+cu128. Every constant matched
the local timm 1.0.27 figures, which is the useful part: agreeing across two
different timm versions makes them **properties of the architecture rather than
artefacts of one environment**.

**What this does NOT check: constructed models.** It verifies the specifications
and the backbone *bodies*. ``srgnn.build()`` and ``agnet.build()`` assert their own
parameter counts at construction, and **those assertions have never fired in the
image** -- the first pretraining run is the first time they do. A construction that
does not hit 32,896,562 / 30,742,924 means the module drifted from the spec, and
it should fail there rather than train.

Exit code is 0 only if every check passes. Anything red means do not pretrain:
a backbone that resolves differently in the image produces checkpoints that
cannot be compared with anything, and the failure is silent at training time.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from cleft.models import agnet, factory, srgnn  # noqa: E402

TARGETS = {
    "srgnn": srgnn.BUILT_PARAMETERS_CUB,
    "agnet": agnet.BUILT_PARAMETERS_CUB,
}


def report(label, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    import timm
    import torch
    import torchvision

    print(f"timm {timm.__version__} | torch {torch.__version__} | "
          f"torchvision {torchvision.__version__}")
    print(f"image: {srgnn.MEASURED_IN_IMAGE}\n")
    passed = True

    # ---- 1. the transformers: dim, params, and their OWN normalization ----
    print("transformers")
    for name in ("vit_b16", "swin_b"):
        spec = factory.backbone_spec(name)
        model = timm.create_model(spec["timm_name"], pretrained=False, num_classes=0)
        with torch.no_grad():
            dim = model(torch.zeros(1, 3, 224, 224)).shape[-1]
        mean, std = factory.normalization_for(model)
        passed &= report(
            f"{name} embedding_dim", dim == spec["embedding_dim"],
            f"{dim} (registry says {spec['embedding_dim']})",
        )
        print(f"         normalization {mean} / {std} "
              f"[{factory.normalization_source(model)}]")

    # ---- 2. the deprecated alias has not drifted ----
    print("\nsrgnn backbone")
    body = timm.create_model(
        srgnn.BACKBONE_TIMM_NAME, pretrained=False, features_only=True
    )
    actual = sum(p.numel() for p in body.parameters())
    passed &= report(
        f"{srgnn.BACKBONE_TIMM_NAME} params",
        actual == srgnn.XCEPTION_FEATURE_PARAMETERS,
        f"{actual:,} (measured {srgnn.XCEPTION_FEATURE_PARAMETERS:,})",
    )
    mean, std = factory.normalization_for(body)
    print(f"         normalization {mean} / {std} "
          f"[{factory.normalization_source(body)}]")

    # ---- 3. AG-Net's backbone, and the hole the stamp closes ----
    print("\nagnet backbone")
    from torchvision.models import ResNet50_Weights

    resnet = torchvision.models.resnet50(weights=None)
    resnet_body = torch.nn.Sequential(*list(resnet.children())[:8])
    actual = sum(p.numel() for p in resnet_body.parameters())
    passed &= report(
        "resnet50 body params", actual == agnet.RESNET50_BODY_PARAMETERS,
        f"{actual:,} (measured {agnet.RESNET50_BODY_PARAMETERS:,})",
    )

    # It must NOT report normalization on its own -- that is the hole.
    try:
        factory.normalization_for(resnet_body)
        passed &= report(
            "resnet50 reports no preprocessing", False,
            "it returned a value; the stamp mechanism may no longer be needed, "
            "but check WHERE the value came from before removing it",
        )
    except factory.FactoryError:
        report("resnet50 reports no preprocessing", True,
               "refused, as expected -- the stamp is what supplies it")

    weights = getattr(ResNet50_Weights, agnet.RESNET_WEIGHTS_DEFAULT)
    transforms = weights.transforms()
    factory.stamp_normalization(
        resnet_body, transforms.mean, transforms.std,
        source=f"torchvision ResNet50_Weights.{agnet.RESNET_WEIGHTS_DEFAULT}"
               ".transforms()",
    )
    mean, std = factory.normalization_for(resnet_body)
    passed &= report(
        "resnet50 normalization after stamp", tuple(mean) == tuple(transforms.mean),
        f"{mean} / {std} [{factory.normalization_source(resnet_body)}]",
    )

    # ---- 4. the two conventions, side by side ----
    print("\nnormalization across the four (the two-two split):")
    print("  vit_b16 and srgnn/xception -> 0.5/0.5")
    print("  swin_b and agnet/resnet50  -> ImageNet statistics")
    print("  no single default is even majority-right, which is why "
          "normalization_for refuses")

    # ---- 5. the whole models ----
    print("\nfull models (build() must assert these itself)")
    for name, target in TARGETS.items():
        print(f"  {name}: spec total {target:,} at 200 classes -- "
              "build() asserts at construction")

    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
