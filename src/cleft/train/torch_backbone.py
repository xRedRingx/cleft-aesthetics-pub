"""The real backbone, satisfying the same protocol as the stub.

Everything torch-shaped lives behind ``__init__``, so importing this module costs
nothing on a laptop and the package's tests never need torch. It is constructed
only when a config actually asks for it.

The head is initialised to the training-fold mean before any optimiser step,
which is what makes gate 3 pass rather than something the gate has to tolerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models.factory import (
    DEFAULT_BACKBONE,
    count_parameters,
    create_backbone,
    freeze_backbone,
    normalization_for,
)


def extract_embeddings(
    images: np.ndarray,
    *,
    name: str = DEFAULT_BACKBONE,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Run the frozen backbone once and return (N, D) penultimate features.

    With the backbone frozen, its output is a fixed function of the image -- the
    same for every fold and every seed. Extracting once turns a 10-seed sweep
    from ten full passes over 237 images into one pass plus ten cheap head fits,
    and makes the *features* byte-identical across seeds rather than merely
    equivalent.

    That removes one source of seed-to-seed difference. It does not remove them
    all: the seed still drives ``harness.inner_val_split``, so which patients
    train varies with it. The band that comes out is initialisation plus split.
    """
    import torch

    # The staged input's own size, so a non-224 artifact builds the dynamic
    # variant instead of dying at timm's patch-embed assertion -- and so 224
    # keeps the exact historical path (factory.timm_kwargs_for).
    input_size = tuple(int(v) for v in np.asarray(images).shape[1:3])
    model = create_backbone(
        name, pretrained=True, num_outputs=0, input_size=input_size
    )
    mean, std = normalization_for(model)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(target)

    mean_t = torch.tensor(mean, device=target).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=target).view(1, -1, 1, 1)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(
                np.asarray(images[start : start + batch_size], dtype=np.float32) / 255.0
            )
            batch = batch.permute(0, 3, 1, 2).to(target)
            batch = (batch - mean_t) / std_t
            out.append(model(batch).float().cpu().numpy())

    embeddings = np.concatenate(out).astype(np.float32)

    # [MEASURED defect, 2026-07-28] This block used to splat count_parameters()
    # verbatim, so metrics.json recorded `trainable_parameters: 85798656` and
    # `trainable_fraction: 1.0` on a run whose train_config said
    # `trainable: head` and whose head is 769 parameters. Nothing was wrong with
    # the run -- the *report* was, and a reader had no way to tell which.
    #
    # The extraction model is never trained: it runs under no_grad and is
    # discarded. So it reports what it is -- a frozen feature extractor -- under
    # names that cannot be mistaken for the arm's trainable set. The arm-level
    # numbers are assembled in phase3.parameter_summary.
    counts = count_parameters(model)
    return embeddings, {
        "backbone": name,
        "embedding_dim": int(embeddings.shape[1]),
        "n_images": int(embeddings.shape[0]),
        # What the backbone actually consumed. At non-224 the ViT runs with
        # dynamic sizing and an interpolated position grid, which is part of
        # the procedure this arm's numbers describe.
        "input_size": [int(v) for v in input_size],
        "dynamic_input": input_size != (224, 224),
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "backbone_parameters": int(counts["total_parameters"]),
        "backbone_trainable_parameters": 0,
        "backbone_frozen": True,
    }


def extract_block_embeddings(
    images: np.ndarray,
    *,
    block: int,
    token: str,
    name: str = DEFAULT_BACKBONE,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Phase 7B's pooling axis: the representation after an INTERMEDIATE block.

    **[LITERATURE]** intermediate layers often transfer better than the last
    for tasks unlike the pretraining objective, and judging a surgical outcome
    is very unlike ImageNet classification -- the final block is optimised for
    an object-category decision this task does not make.

    ----------------------------------------------------------------------
    THE FINAL NORM IS APPLIED AT EVERY DEPTH, AND THAT IS THE WHOLE POINT
    ----------------------------------------------------------------------
    ``model.norm`` is applied after the selected block, not only after the
    twelfth. Two reasons, and the second is the one that matters:

    * it is what timm's own ``forward_intermediates(norm=True)`` does, so the
      representation is the conventional one rather than this project's
      invention;
    * **without it the axis would vary depth AND normalisation together**, and
      a difference between block 6 and block 12 could not be attributed to
      either. That is the one-factor rule the whole ladder is built on, in a
      place where nothing downstream would notice it had been broken.

    **The consequence is a free self-check**: at ``block=12, token=cls`` this
    reproduces exactly what ``extract_embeddings`` returns for ViT-B/16, since
    timm's ``num_classes=0`` forward is the class token after ``model.norm``.
    The shipped config asserts that bit-for-bit against the already-hashed
    ``vit_b16__imagenet__g1``, which validates the whole intermediate path
    against a known-good artifact for free.
    """
    import torch

    from ..embedding_plan import POOLING_TOKENS

    if token not in POOLING_TOKENS:
        raise ValueError(f"unknown token {token!r}; expected {POOLING_TOKENS}")

    # The frames' own size, like extract_embeddings: 7B ran at 224 only,
    # where this is a no-op, and a future pooling axis at a new resolution
    # dies here at build rather than on the cluster (the builder sweep).
    input_size = tuple(int(v) for v in np.asarray(images).shape[1:3])
    model = create_backbone(
        name, pretrained=True, num_outputs=0, input_size=input_size
    )
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError(
            f"{name} has no `blocks`; the pooling axis is defined for the "
            "plain ViT stack and refuses anything else rather than guessing "
            "where its depth lives"
        )
    depth = len(blocks)
    if not 1 <= int(block) <= depth:
        raise ValueError(f"block {block} outside 1..{depth} for {name}")

    mean, std = normalization_for(model)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(target)

    mean_t = torch.tensor(mean, device=target).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=target).view(1, -1, 1, 1)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))

    def identity(x):
        return x

    patch_drop = getattr(model, "patch_drop", identity)
    norm_pre = getattr(model, "norm_pre", identity)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(
                np.asarray(images[start : start + batch_size], dtype=np.float32) / 255.0
            )
            batch = batch.permute(0, 3, 1, 2).to(target)
            batch = (batch - mean_t) / std_t

            x = model.patch_embed(batch)
            x = model._pos_embed(x)
            x = patch_drop(x)
            x = norm_pre(x)
            for layer in blocks[: int(block)]:
                x = layer(x)
            x = model.norm(x)

            if token == "cls":
                if n_prefix < 1:
                    raise ValueError(
                        f"{name} carries no prefix token, so 'cls' names "
                        "nothing; use mean_patch"
                    )
                pooled = x[:, 0]
            else:
                pooled = x[:, n_prefix:].mean(dim=1)
            out.append(pooled.float().cpu().numpy())

    embeddings = np.concatenate(out).astype(np.float32)
    counts = count_parameters(model)
    return embeddings, {
        "backbone": name,
        "block": int(block),
        "depth": depth,
        "token": token,
        "num_prefix_tokens": n_prefix,
        "final_norm_applied": True,
        "embedding_dim": int(embeddings.shape[1]),
        "n_images": int(embeddings.shape[0]),
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "backbone_parameters": int(counts["total_parameters"]),
        "backbone_trainable_parameters": 0,
        "backbone_frozen": True,
        "note": (
            "model.norm is applied at every depth so the axis varies depth "
            "alone; block=depth with token=cls reproduces extract_embeddings"
        ),
    }


def extract_patch_embeddings(
    images: np.ndarray,
    geometry_rows: list[dict],
    patches,
    *,
    name: str = DEFAULT_BACKBONE,
    pooling: str = "mean",
    output_size: int | None = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Embed every patch of every patient with the frozen backbone, then pool.

    Same frozen backbone and the same normalization as the whole-image probe, so
    the comparison between them is about **where the model looks** rather than
    about what it is. One patient at a time, because the full patch stack is
    ~960 MB materialised and nothing needs it all at once.

    Deterministic by construction: no dropout, no shuffling, ``no_grad``, and a
    fixed patient order. That is the property the Phase 4 determinism re-run
    checks rather than assumes -- pooling is exactly where a nondeterministic
    reduction would hide.
    """
    import torch

    from ..geometry import patch_features

    output_size = output_size or patch_features.BACKBONE_INPUT_SIZE
    if pooling not in patch_features.POOLINGS:
        raise patch_features.PatchFeatureError(
            f"unknown pooling {pooling!r}; expected one of {patch_features.POOLINGS}"
        )

    # The size the model actually consumes is the PATCH crop size -- every
    # crop is resized to output_size before the forward -- not the staged
    # frame. Threading the frame here would be the wrong quantity (R2); at
    # the default 224 this is a no-op either way (the builder sweep).
    model = create_backbone(
        name, pretrained=True, num_outputs=0,
        input_size=(int(output_size), int(output_size)),
    )
    mean, std = normalization_for(model)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(target)

    mean_t = torch.tensor(mean, device=target).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=target).view(1, -1, 1, 1)

    pooled: list[np.ndarray] = []
    with torch.no_grad():
        for stack in patch_features.iter_patient_patches(
            images, geometry_rows, patches, output_size
        ):
            per_patch: list[np.ndarray] = []
            for start in range(0, len(stack), batch_size):
                batch = torch.as_tensor(
                    np.asarray(stack[start : start + batch_size], dtype=np.float32)
                    / 255.0
                )
                batch = batch.permute(0, 3, 1, 2).to(target)
                batch = (batch - mean_t) / std_t
                per_patch.append(model(batch).float().cpu().numpy())
            pooled.append(
                patch_features.pool(np.concatenate(per_patch), pooling)
            )

    embeddings = np.stack(pooled).astype(np.float32)
    counts = count_parameters(model)
    return embeddings, {
        "features": "frozen_backbone_patch_embeddings",
        "backbone": name,
        "embedding_dim": int(embeddings.shape[1]),
        "n_images": int(embeddings.shape[0]),
        "n_patches": len(patches),
        "pooling": pooling,
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        # Same naming discipline as the whole-image path: these are the frozen
        # extractor's parameters and must not read as the arm's trainable set.
        "backbone_parameters": int(counts["total_parameters"]),
        "backbone_trainable_parameters": 0,
        "backbone_frozen": True,
    }


class FrozenExtractor:
    """The frozen backbone, built ONCE and reused across epochs and folds.

    ``extract_embeddings`` constructs the model on every call, which is right
    for a one-shot extraction and wrong for the Phase 7C path: that runs the
    backbone every epoch, and rebuilding ViT-B/16 each time would dominate the
    arm. Nothing about the model changes between calls -- it is frozen, in
    eval mode, under ``no_grad`` -- so it is built once and held.
    """

    def __init__(
        self, name: str = DEFAULT_BACKBONE, *, batch_size: int = 32,
        device: str = "cuda",
    ):
        import torch

        self.name = name
        self.batch_size = int(batch_size)
        self._model = create_backbone(name, pretrained=True, num_outputs=0)
        self._mean, self._std = normalization_for(self._model)
        self._device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._model.eval().to(self._device)
        self._mean_t = torch.tensor(self._mean, device=self._device).view(1, -1, 1, 1)
        self._std_t = torch.tensor(self._std, device=self._device).view(1, -1, 1, 1)
        self.parameter_report = {
            "backbone_parameters": int(
                count_parameters(self._model)["total_parameters"]
            ),
            "backbone_trainable_parameters": 0,
            "backbone_frozen": True,
            "normalization_mean": list(self._mean),
            "normalization_std": list(self._std),
        }

    @property
    def model(self):
        """The frozen model itself.

        **Exposed for Phase 8's Grad-CAM**, which needs to hang a forward hook
        on an intermediate block. ``phase8.LIVE_PATH_REUSE``: the gradient pass
        is a method on this object rather than a second class, so the frozen
        boundary stays defined in one place.
        """
        return self._model

    def preprocess(self, images: np.ndarray):
        """(n, H, W, 3) pixels in 0-255 -> a normalised NCHW tensor on device.

        **Factored out of ``__call__`` rather than duplicated.** Grad-CAM needs
        the identical preprocessing and must NOT run under ``no_grad``, so it
        cannot reuse ``__call__``; a second copy of the /255, permute and
        normalise would be two definitions of what this model is fed, which is
        the shape of defect this project keeps finding.
        """
        import torch

        stack = np.asarray(images, dtype=np.float32)
        batch = torch.as_tensor(stack / 255.0)
        batch = batch.permute(0, 3, 1, 2).to(self._device)
        return (batch - self._mean_t) / self._std_t

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """(n, H, W, 3) pixels in 0-255 -> (n, D) frozen embeddings."""
        import torch

        stack = np.asarray(images, dtype=np.float32)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(stack), self.batch_size):
                batch = self.preprocess(stack[start : start + self.batch_size])
                out.append(self._model(batch).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


@dataclass
class AugmentingHeadBackbone:
    """Phase 7C: augment the training pixels, extract, fit the head.

    **Slots into the frozen harness rather than replacing it.**
    ``harness.run_fold`` calls ``train_epoch`` once per epoch and ``predict``
    for inner-val and test, so augmenting inside ``train_epoch`` and not in
    ``predict`` makes the never-augment-evaluation rule a property of the call
    graph. The training loop is untouched and gate 1 still tests the harness
    it was measured on.

    **The features it receives are (n, H, W, 4)**: three colour channels and a
    per-patient augmentation-strength mask (``phase3.prepare_features``). The
    mask rides along as a channel because the harness slices ``features[rows]``
    and anything held separately could not be sliced with it -- recovering the
    row indices inside the backbone would mean recomputing ``inner_val_split``
    outside the frozen harness.
    """

    policy: Any
    photometric_settings: dict
    geometric_settings: dict
    inside_strength: float = 1.0
    sigma_fraction: float = 0.0
    seed: int = 1337
    #: The ordinal position of this fold in the sequence ``run_cv`` walks --
    #: NOT the fold id, which the zero-argument factory protocol cannot pass.
    #: It is only used to separate the augmentation streams of different
    #: folds, for which an ordinal is sufficient and honest.
    fold_ordinal: int = 0
    extractor: Any = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.01

    _head: Any = field(default=None, repr=False)
    _epoch: int = field(default=0, repr=False)

    @property
    def parameter_report(self) -> dict:
        return {} if self._head is None else self._head.parameter_report

    @staticmethod
    def split(features: np.ndarray) -> tuple:
        """(n, H, W, 4) -> pixels, strength."""
        array = np.asarray(features)
        if array.ndim != 4 or array.shape[3] != 4:
            raise ValueError(
                f"the augmenting path expects (n, H, W, 4) -- three colour "
                f"channels and a strength mask -- and got {array.shape}"
            )
        return array[..., :3], array[..., 3]

    def reset(self, train_labels: np.ndarray) -> None:
        self._head = EmbeddingHeadBackbone(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )
        self._head.reset(train_labels)
        self._epoch = 0

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """**The only place augmentation happens.**"""
        from .augment import augment_batch

        images, strength = self.split(features)
        augmented = augment_batch(
            images,
            policy=self.policy,
            seed=self.seed,
            fold=self.fold_ordinal,
            epoch=self._epoch,
            photometric_settings=self.photometric_settings,
            geometric_settings=self.geometric_settings,
            strength_per_image=strength if self.policy.region_aware else None,
        )
        self._epoch += 1
        return self._head.train_epoch(self.extractor(augmented), labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """**No augmentation, ever.** Inner-val and test arrive here."""
        images, _ = self.split(features)
        return self._head.predict(self.extractor(images))


@dataclass
class EmbeddingHeadBackbone:
    """A linear head over precomputed frozen-backbone embeddings.

    This is what ``trainable: head`` actually trains. Exactly equivalent to
    freezing the backbone and fine-tuning the head, and far cheaper, because the
    frozen backbone's output does not depend on the seed or the fold.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 50
    seed: int = 1337
    #: **[ADDED 2026-08-15, Phase 8c's animation]** Called after EVERY
    #: optimizer step as ``on_step(step_index)`` (1-based within the epoch)
    #: when set. The head trains by full-batch steps, so each call marks a
    #: real parameter state -- the training-trajectory axis's checkpoint
    #: hook. None (the default) changes nothing: the hook is observation
    #: only, and a capturing caller must not alter what it captures.
    on_step: Any = None
    #: What this arm ACTUALLY trains, filled in once the embedding width is
    #: known. For ViT-B/16 that is 768 weights + 1 bias = 769 -- five orders of
    #: magnitude below the backbone it sits on, which is the whole point of the
    #: policy and was exactly what the old report did not say.
    parameter_report: dict = field(default_factory=dict)

    _weights: Any = field(default=None, repr=False)
    _bias: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Weights at zero and bias at the training-fold mean, so an untrained
        # head predicts the mean exactly -- which is what gate 3 checks.
        self._weights = None
        self._bias = torch.tensor(
            float(np.mean(train_labels)), device=self._device, requires_grad=True
        )
        self._optimizer = None

    def _ensure(self, dim: int) -> None:
        import torch

        if self._weights is None:
            self._weights = torch.zeros(
                dim, device=self._device, requires_grad=True, dtype=torch.float32
            )
            self._optimizer = torch.optim.AdamW(
                [self._weights, self._bias],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            # Counted off the tensors the optimiser was actually handed, not
            # derived from the width a second time.
            self.parameter_report = {
                "total_parameters": int(
                    sum(p.numel() for p in (self._weights, self._bias))
                ),
                "trainable_parameters": int(
                    sum(
                        p.numel()
                        for p in (self._weights, self._bias)
                        if p.requires_grad
                    )
                ),
            }

    def head_weights(self) -> np.ndarray:
        """The fitted weight vector, for Phase 8's Grad-CAM.

        **A copy, not the live tensor.** Grad-CAM needs the direction this
        head reads the embedding along, so it can weight the backbone's output
        before taking gradients. Handing out the fitted tensor would let an
        explanation routine mutate a trained model.
        """
        if self._weights is None:
            raise ValueError(
                "the head has not been fitted; there are no weights to read"
            )
        return self._weights.detach().cpu().numpy().reshape(-1).copy()

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        import torch

        self._ensure(features.shape[1])
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self._device)
        y = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=self._device)

        loss_value = 0.0
        for step in range(self.max_steps):
            self._optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(x @ self._weights + self._bias, y)
            loss.backward()
            self._optimizer.step()
            loss_value = float(loss.item())
            if self.on_step is not None:
                self.on_step(step + 1)
        return loss_value

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self._device)
        with torch.no_grad():
            if self._weights is None:
                return np.full(len(features), float(self._bias.item()))
            return (x @ self._weights + self._bias).float().cpu().numpy()


@dataclass
class TorchBackbone:
    """timm backbone + regression head, MSE on the raw 1-5 scale."""

    name: str = DEFAULT_BACKBONE
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    device: str = "cuda"
    seed: int = 1337
    #: "head" freezes the backbone; "full" trains everything and, on 237 images,
    #: destroyed the representation -- see factory.TRAINABLE_POLICIES.
    trainable: str = "head"
    parameter_report: dict = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _mean: Any = field(default=None, repr=False)
    _std: Any = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        self._model = create_backbone(self.name, pretrained=self.pretrained)

        # Normalization from the model's own config -- never a hardcoded triple.
        mean, std = normalization_for(self._model)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, -1, 1, 1)

        # The head starts at the training-fold mean, so an untrained model
        # predicts the mean. Gate 3 then checks that it actually did.
        head = self._model.get_classifier()
        with torch.no_grad():
            if hasattr(head, "weight"):
                torch.nn.init.zeros_(head.weight)
            if getattr(head, "bias", None) is not None:
                head.bias.fill_(float(np.mean(train_labels)))

        if self.trainable == "head":
            freeze_backbone(self._model)
        self.parameter_report = count_parameters(self._model)

        self._model.to(device)
        self._optimizer = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _batches(self, features: np.ndarray, labels: np.ndarray | None, shuffle: bool):
        import torch

        device = next(self._model.parameters()).device
        order = np.arange(len(features))
        if shuffle:
            order = np.random.default_rng(self.seed).permutation(order)

        for start in range(0, len(order), self.batch_size):
            index = order[start : start + self.batch_size]
            batch = torch.as_tensor(
                np.asarray(features[index], dtype=np.float32) / 255.0
            )
            # (N, H, W, C) -> (N, C, H, W)
            batch = batch.permute(0, 3, 1, 2).to(device)
            batch = (batch - self._mean) / self._std
            target = (
                None
                if labels is None
                else torch.as_tensor(
                    np.asarray(labels[index], dtype=np.float32), device=device
                )
            )
            yield batch, target

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        import torch

        self._model.train()
        total, seen = 0.0, 0
        for batch, target in self._batches(features, labels, shuffle=True):
            self._optimizer.zero_grad(set_to_none=True)
            prediction = self._model(batch).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            self._optimizer.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        return total / max(seen, 1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for batch, _ in self._batches(features, None, shuffle=False):
                out.append(self._model(batch).squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)
