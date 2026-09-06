"""OFFLINE PROBE, no launch: the variant's per-cell behaviour on the real
cohort. Run directly on the pod from the repo root:

    PYTHONPATH=src python /path/to/p8b_softmax_probe.py

Reads the same inputs configs/p8_grad_cam_softmax.yaml declares (paths only,
no guard-3 run machinery -- this writes nothing and claims nothing). Reports:
the walk order (which patient is second), per-cell all-zero counts for the
variant, and the top-10 concentration distribution on surviving cells.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import yaml

from cleft import gradcam, gradcam_sheet, phase8
from cleft.data.manifest import load_manifest
from cleft.run import embeddings_module_load, phase8_refit_heads, phase8_stratified_sample
from cleft.train import phase3
from cleft.train.torch_backbone import FrozenExtractor

config = yaml.safe_load(Path("configs/p8_grad_cam_softmax.yaml").read_text())
paths = {e["name"]: Path(e["path"]) for e in config["inputs"]}
task = config["task"]
seeds = [int(s) for s in task["seeds"]]

images, labels, patient_ids, _ = phase3.load_inputs(
    paths[task["manifest_artifact"]], paths[task["staged_artifact"]],
    task["geometry"], task["label"],
)
rows = load_manifest(paths[task["manifest_artifact"]] / "manifest.csv")
assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}
features, _ = embeddings_module_load(paths[task["embeddings_artifact"]], patient_ids)
heads, pooled = phase8_refit_heads(
    features, labels, patient_ids, assignments, seeds, task, print
)
print(f"refit mean {np.mean(pooled):.4f} (gate expects 0.2520)")

sample = phase8_stratified_sample(patient_ids, labels)
print(f"WALK ORDER: {sample}")
print(f"first: {sample[0]}   SECOND (the failing patient): {sample[1]}")

extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
index = {int(p): i for i, p in enumerate(patient_ids)}

zero = np.zeros((len(sample), len(seeds)), dtype=bool)
survivor_top10, original_top10 = [], []
for pi, patient in enumerate(sample):
    pixels = np.asarray(images[index[patient]: index[patient] + 1])
    for si, seed in enumerate(seeds):
        acts, grads = gradcam.token_activations_and_gradients(
            extractor, pixels, heads[(seed, assignments[patient])]
        )
        a, g = acts[0], grads[0]
        original_top10.append(
            gradcam_sheet.concentration(gradcam.cam(a, g))["top_10_share"]
        )
        raw = gradcam.cam_softmax(a, g)
        if raw.max() <= 0:
            zero[pi, si] = True
        else:
            survivor_top10.append(
                gradcam_sheet.concentration(raw)["top_10_share"]
            )
    print(f"  patient {patient}: {int(zero[pi].sum())}/5 cells all-zero")

per_patient = zero.sum(axis=1)
s = np.array(survivor_top10)
o = np.array(original_top10)
print(f"\ncells all-zero: {int(zero.sum())} of {zero.size}")
print(f"patients with ALL seeds zero (no variant map): "
      f"{int((per_patient == len(seeds)).sum())} of {len(sample)}")
print(f"per-patient zero counts: {per_patient.tolist()}")
print(f"variant top-10 on survivors (n={len(s)}): "
      f"median {np.median(s):.3f} p10 {np.percentile(s, 10):.3f} "
      f"p90 {np.percentile(s, 90):.3f} min {s.min():.3f} max {s.max():.3f}"
      if len(s) else "variant top-10: NO surviving cells")
print(f"original top-10 (all cells): median {np.median(o):.3f} "
      f"p90 {np.percentile(o, 90):.3f}   uniform {10/196:.3f}")
