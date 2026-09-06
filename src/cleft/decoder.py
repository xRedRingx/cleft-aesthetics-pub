"""Phase 13's decoder: the representation-measurement instrument.

The model, the SSIM metric, the band machinery and the SCUT-embedding
artifact I/O live here, ONCE, so the training task and the later cohort
task cannot drift apart (``phase13.DECODER_ARCHITECTURE_PROPOSED``,
``phase13.DOUBLE_DIFFERENCE_RULE``).

**Not the embeddings-set format, deliberately.** The ``embeddings`` module's
loaders enforce cleft-manifest row order and cleft pairing semantics; a
SCUT set imitating that format would invite a ladder loader to consume
it. This module's artifact is values + stems + metadata, with its own
reader that checks what it wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: The architecture's fixed shape (phase13.DECODER_ARCHITECTURE_PROPOSED).
#: [2026-08-24, the pick on PARAMETER_BAND_MISESTIMATED:
#: resolution (a), the 7x7x96 shape -- the smaller decoder is the more
#: conservative instrument for a what-the-embedding-keeps reading; the
#: original 7x7x256 seed constructed to 10,628,771 against a ~3-5M
#: estimate. The registration is now the CONSTRUCTED count, held by
#: exact equality -- no band, no estimate.]
EMBED_DIM = 768
SEED_CHANNELS = 96
SEED_SIZE = 7
OUTPUT_SIZE = 224
#: Five upsample+conv stages: 7 -> 14 -> 28 -> 56 -> 112 -> 224.
STAGE_CHANNELS = (96, 96, 64, 32, 16)
#: The registered parameter count -- a LITERAL, deliberately, so an edit
#: to the constants above cannot silently re-register itself: both
#: expected_parameters() (venv, torch-free) and build() (under torch)
#: must equal it exactly.
REGISTERED_PARAMETERS = 3_862_339

#: Content-pixel rule for band errors, BOTH families: a pixel at or above
#: this on every channel is baked white or pad, and is excluded exactly as
#: the pad is (phase13.DOUBLE_DIFFERENCE_RULE's registered mitigation;
#: the basal.py precedent).
WHITE_LEVEL = 250

#: The bands: content-box horizontal thirds, named by the anatomy
#: vocabulary on the cleft crop. Positional, not anatomical -- the
#: registered residual limitation.
BANDS = ("eyes", "nose", "lips")
GRADER_RELEVANT = ("nose", "lips")
GRADE_IRRELEVANT = ("eyes",)


class DecoderError(RuntimeError):
    """A Phase 13 decoder contract is not usable."""


def checkpoint_name(epoch: int) -> str:
    """The per-epoch checkpoint filename, ONCE: the shakedown task
    writes it and the cohort task reads it, so the format cannot drift
    between the writer and the reader (phase13.STOP_2_BUILT)."""
    return f"epoch_{int(epoch):02d}.pt"


def expected_parameters(
    seed_channels: int = SEED_CHANNELS,
    stage_channels: tuple = STAGE_CHANNELS,
    embed_dim: int = EMBED_DIM,
    seed_size: int = SEED_SIZE,
) -> int:
    """The constructed parameter count, by PURE INTEGER ARITHMETIC over
    the layer dims -- no torch, so it runs in the pinned venv.

    **[2026-08-24, phase13.PARAMETER_BAND_MISESTIMATED]** This exists
    because the registration carried an ESTIMATED count (~3-5M) that the
    first real construction -- on the cluster, since the torch test
    skips in the venv -- measured at 10,628,771, and the guard refused
    it, correctly. An estimated count is not a constructed count; this
    function is the constructed count without needing the construction,
    and the venv test pins the registration to it so a mismatch fails on
    the laptop before any launch.
    """
    seed_map = seed_size * seed_size * seed_channels
    total = embed_dim * seed_map + seed_map          # the seed Linear
    channels = seed_channels
    for out_channels in stage_channels:              # 3x3 convs, W + b
        total += channels * out_channels * 9 + out_channels
        channels = out_channels
    total += channels * 3 * 9 + 3                    # the output conv
    return total


def build(*, seed: int = 1337):
    """The bottleneck decoder: Linear 768 -> 7x7x96, then five
    upsample+conv stages to 224x224x3, sigmoid output in [0, 1].

    Upsample+conv rather than ConvTranspose, so checkerboard artifacts
    cannot contaminate the regional comparison. REGISTERED_PARAMETERS
    (3,862,339) asserted EXACTLY at construction -- the re-registration
    rule after phase13.PARAMETER_BAND_MISESTIMATED: no band, no
    estimate.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class BottleneckDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seed_layer = nn.Linear(
                EMBED_DIM, SEED_CHANNELS * SEED_SIZE * SEED_SIZE
            )
            blocks = []
            channels = SEED_CHANNELS
            for out_channels in STAGE_CHANNELS:
                blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                blocks.append(
                    nn.Conv2d(channels, out_channels, 3, padding=1)
                )
                blocks.append(nn.ReLU())
                channels = out_channels
            blocks.append(nn.Conv2d(channels, 3, 3, padding=1))
            self.stages = nn.Sequential(*blocks)

        def forward(self, z):
            import torch as _torch

            seed_map = self.seed_layer(z).reshape(
                -1, SEED_CHANNELS, SEED_SIZE, SEED_SIZE
            )
            return _torch.sigmoid(self.stages(seed_map))

    model = BottleneckDecoder()
    total = sum(p.numel() for p in model.parameters())
    if total != REGISTERED_PARAMETERS:
        raise DecoderError(
            f"{total:,} parameters constructed, {REGISTERED_PARAMETERS:,} "
            "registered -- exact equality is the rule "
            "(phase13.PARAMETER_BAND_MISESTIMATED); the architecture "
            "drifted from its registration"
        )
    return model


# --------------------------------------------------------------------------
# SSIM -- deterministic, dependency-light (the registered metric choice)
# --------------------------------------------------------------------------


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    offsets = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.outer(kernel, kernel)


def ssim(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean SSIM over channels, standard constants, 11x1.5 gaussian.

    Inputs are HxWx3 in [0, 255] (uint8 or float). Implemented here in
    numpy because SSIM is REPORTED, never optimised -- optimising a
    perceptual metric would bake 'looks face-like' into the objective,
    the registered failure mode by construction.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(reconstruction, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise DecoderError(f"shapes {a.shape} vs {b.shape}: not comparable")
    window = _gaussian_window()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def _filter(image: np.ndarray) -> np.ndarray:
        pad = window.shape[0] // 2
        padded = np.pad(image, pad, mode="reflect")
        out = np.zeros_like(image)
        for dy in range(window.shape[0]):
            for dx in range(window.shape[1]):
                out += window[dy, dx] * padded[
                    dy : dy + image.shape[0], dx : dx + image.shape[1]
                ]
        return out

    values = []
    for channel in range(3):
        x, y = a[:, :, channel], b[:, :, channel]
        mu_x, mu_y = _filter(x), _filter(y)
        sigma_x = _filter(x * x) - mu_x ** 2
        sigma_y = _filter(y * y) - mu_y ** 2
        sigma_xy = _filter(x * y) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        )
        values.append(float(score.mean()))
    return float(np.mean(values))


# --------------------------------------------------------------------------
# bands and content pixels (phase13.DOUBLE_DIFFERENCE_RULE)
# --------------------------------------------------------------------------


def content_mask(image: np.ndarray, level: int = WHITE_LEVEL) -> np.ndarray:
    """True where the pixel is CONTENT: below the white level on any
    channel. Baked white and pad are excluded identically, both families
    -- the registered mitigation for the baked-white baseline defect."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise DecoderError(f"expected HxWx3, got {array.shape}")
    return ~np.all(array >= level, axis=2)


def band_rows(content_box: tuple, band: str) -> tuple:
    """The output-pixel row range of one band: the content box's
    horizontal thirds. Positional by registration."""
    if band not in BANDS:
        raise DecoderError(f"unknown band {band!r}; bands are {BANDS}")
    _, y0, _, height = content_box
    index = BANDS.index(band)
    top = y0 + int(round(height * index / 3))
    bottom = y0 + int(round(height * (index + 1) / 3))
    return top, max(bottom, top + 1)


def band_errors(reference: np.ndarray, reconstruction: np.ndarray,
                content_box: tuple) -> dict:
    """Per-band mean squared error over CONTENT pixels, plus each band's
    content-pixel fraction and content variance (the comparability
    table's rows come from here -- one implementation for both
    families)."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(reconstruction, dtype=np.float64)
    mask = content_mask(reference)
    out: dict = {}
    for band in BANDS:
        top, bottom = band_rows(content_box, band)
        band_mask = np.zeros(mask.shape, dtype=bool)
        band_mask[top:bottom, :] = True
        band_mask &= mask
        n = int(band_mask.sum())
        if n == 0:
            out[band] = {
                "mse": float("nan"), "content_fraction": 0.0,
                "content_variance": float("nan"), "n_pixels": 0,
            }
            continue
        difference = (a - b)[band_mask]
        rows = bottom - top
        out[band] = {
            "mse": float(np.mean(difference ** 2)),
            "content_fraction": float(n / (rows * mask.shape[1])),
            "content_variance": float(np.var(a[band_mask])),
            "n_pixels": n,
        }
    return out


def band_asymmetry(
    image: np.ndarray, content_box: tuple, band: str,
    mask_source: np.ndarray | None = None,
) -> float:
    """Mean ``|left - right|`` over CONTENT pixels within one band -- the
    Phase 4 mirror difference, restricted (phase13.ASYMMETRY_RETENTION).

    Three departures from ``mirror.mirror_features``, each forced and each
    registered rather than silent:

    * **The midline is the CONTENT BOX's, not the image's.** These are G1
      crops, where ``mirror`` refuses its own feature set precisely because
      the trapezium pad is symmetric and dilutes the index. The face is
      centred in its content box, not in the padded square, so flipping
      about the image midline would measure the box's offset.
    * **Content pixels only, and SYMMETRICALLY so.** A pixel whose mirror
      partner is pad would score |content - white| and read as asymmetry
      that is really the crop's shape, so the mask is ``mask & mask[:,
      ::-1]`` -- both members present or neither counts.

    ``mask_source`` supplies the image the mask is read FROM, which for a
    PAIRED comparison must be the REFERENCE for both members.

    **[2026-08-24, DEFECT (c-i)] Reading the mask from each image
    separately compares different pixel sets.** A decoder reconstruction
    is smooth and almost never reaches the white level, so its own mask
    covers nearly the whole window INCLUDING the reconstructed pad, while
    the original's mask excludes the pad exactly. The reconstruction's
    "asymmetry" then measures mostly background rendering, which has no
    reason to correlate with the face's asymmetry -- the leading
    suspect for r(asym_orig, asym_recon) landing at +0.0000. A retention
    RATIO between two different regions is not a ratio at all, so the
    reference's mask is now used for both.
    * **The bands are ``band_rows``' positional thirds**, the same rows the
      double difference uses, so the two statistics speak about the same
      pixels.

    Returns NaN when the band has no symmetric content pair, which the
    caller reports rather than silently averaging away.

    **[2026-08-24, MEASURED before use] Units are scaled HERE, not
    inherited.** ``mirror.as_float`` scales uint8 but TRUSTS float as
    already [0, 1] -- and this phase's two families arrive in different
    dtypes: staged originals are uint8, decoder reconstructions are
    float in 0-255. Dtype-trusting would have made every retention ratio
    ~255x too large and read as "asymmetry fully retained", the exact
    finding under test. Both families are therefore scaled by the same
    explicit rule before the mirror machinery sees them.
    """
    from .geometry import mirror

    raw = np.asarray(image, dtype=np.float64)
    x0, _, width, _ = content_box
    top, bottom = band_rows(content_box, band)
    left, right = int(x0), int(x0) + int(width)
    window = raw[top:bottom, left:right]
    if window.size == 0:
        return float("nan")
    # The mask reads RAW 0-255 (WHITE_LEVEL is a 0-255 threshold); the
    # residual reads the SCALED copy. Mixing the two units is the defect
    # the docstring above records.
    source = window if mask_source is None else np.asarray(
        mask_source, dtype=np.float64
    )[top:bottom, left:right]
    if source.shape[:2] != window.shape[:2]:
        raise DecoderError(
            f"mask source {source.shape[:2]} does not match the measured "
            f"window {window.shape[:2]}: the paired mask must come from "
            "the SAME region of the reference"
        )
    mask = content_mask(source)
    mask &= mask[:, ::-1]
    if not mask.any():
        return float("nan")
    window = window / 255.0
    # residual_map is |image - mirror(image)| on [0, 1], channel-averaged,
    # and is itself mirror-symmetric -- mirror.mirror_symmetry_error is the
    # standing check on that property.
    residual = mirror.residual_map(window)
    return float(residual[mask].mean())


#: The non-anatomical global statistics (phase13.CLOSING_ADDENDUM_REGISTERED
#: P1). Every one is a property of the IMAGE as a whole -- none of them
#: can see a nose, a lip or a scar -- which is exactly what makes a probe
#: over them a confound ceiling rather than a model.
GLOBAL_STATISTIC_NAMES = (
    "aspect_ratio", "brightness_mean", "contrast_sd",
    "content_fraction", "corner_white_fraction",
)


def global_statistics(image: np.ndarray, content_box: tuple) -> dict:
    """The five non-anatomical statistics for one staged crop.

    Measured over CONTENT pixels where the statistic is about the face
    (brightness, contrast) and over the whole crop where it is about the
    framing (content fraction, corner white). The aspect ratio is the
    content box's own, which is the quantity phase12's
    BASAL_CONFOUND_OBSERVED already measured against grade one variable
    at a time -- this is the multivariate version of that same worry.
    """
    from .geometry import basal

    array = np.asarray(image, dtype=np.float64)
    x0, y0, width, height = (int(v) for v in content_box)
    mask = content_mask(array)
    content = array[mask] if mask.any() else array.reshape(-1, array.shape[-1])
    return {
        "aspect_ratio": float(width / height) if height else float("nan"),
        "brightness_mean": float(content.mean()),
        "contrast_sd": float(content.std()),
        "content_fraction": float(mask.mean()),
        "corner_white_fraction": float(
            basal.corner_white_fraction(np.clip(array, 0, 255).astype(np.uint8))
        ),
    }


def occlude_band(image: np.ndarray, content_box: tuple, band: str) -> np.ndarray:
    """A COPY of the crop with one positional third blanked to white
    (phase13.CLOSING_ADDENDUM_REGISTERED P2).

    White, not black or noise: white is what the staging pad already is,
    so an occluded band is indistinguishable from pad to anything
    downstream -- and the pad is the one thing the phase already
    established as trivially reconstructible and excluded everywhere.
    Blanking to black would introduce a value the pipeline never sees.

    The rows are ``band_rows``' thirds, the same machinery the double
    difference and the asymmetry measure use, so all three statistics
    speak about the same regions.
    """
    occluded = np.array(image, copy=True)
    top, bottom = band_rows(content_box, band)
    occluded[top:bottom, :] = 255
    return occluded


def double_difference(cleft_band_mse: dict, scut_band_mse: dict) -> dict:
    """The registered statistic: cleft band error normalised by the same
    band's SCUT baseline, then grader-relevant against grade-irrelevant.

    Positive gap = grader-relevant regions reconstruct WORSE relative to
    baseline -- the direction the failure-mode reading fires on.
    """
    ratios = {}
    for band in BANDS:
        baseline = scut_band_mse[band]
        if not np.isfinite(baseline) or baseline <= 0:
            raise DecoderError(
                f"band {band!r}: SCUT baseline {baseline} is unusable -- "
                "the double difference divides by it"
            )
        ratios[band] = float(cleft_band_mse[band] / baseline)
    relevant = float(np.mean([ratios[b] for b in GRADER_RELEVANT]))
    irrelevant = float(np.mean([ratios[b] for b in GRADE_IRRELEVANT]))
    return {
        "normalised_ratio_by_band": ratios,
        "grader_relevant_mean": relevant,
        "grade_irrelevant_mean": irrelevant,
        "double_difference": relevant - irrelevant,
        "positive_means": (
            "grader-relevant regions reconstruct WORSE relative to the "
            "SCUT baseline -- the direction the failure-mode reading "
            "fires on"
        ),
    }


# --------------------------------------------------------------------------
# the SCUT-embedding artifact (values + stems, NOT the embeddings-set format)
# --------------------------------------------------------------------------


def save_scut_embeddings(directory, values: np.ndarray, stems: list,
                         metadata: dict) -> dict:
    """Write the SCUT embedding artifact. Owns existence and creation
    atomically -- the guard-after-mkdir lesson
    (phase12.EXTRACT_GUARD_AFTER_MKDIR): callers must NOT pre-create or
    pre-check this path."""
    from .provenance import atomic_write_text
    from .provenance.hashing import hash_dir

    directory = Path(directory)
    if directory.exists():
        raise DecoderError(
            f"{directory} already exists. Data artifacts are immutable: "
            "create a new version, never overwrite."
        )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != EMBED_DIM:
        raise DecoderError(f"values {values.shape}: expected (n, {EMBED_DIM})")
    if len(stems) != len(values):
        raise DecoderError(f"{len(stems)} stems for {len(values)} rows")
    if len(set(stems)) != len(stems):
        raise DecoderError("repeated stem: two rows would claim one face")
    directory.mkdir(parents=True)
    np.save(directory / "values.npy", values)
    atomic_write_text(
        directory / "metadata.json",
        json.dumps(
            {"stems": list(stems), **metadata}, indent=2, sort_keys=True
        ) + "\n",
    )
    payload = hash_dir(directory)
    atomic_write_text(
        directory / "MANIFEST.json",
        json.dumps(
            {
                "artifact": directory.name,
                "payload_rollup": payload["rollup"],
                "payload_files": payload["files"],
                "note": (
                    "SCUT embeddings, values+stems format -- deliberately "
                    "NOT the cleft embeddings-set format, whose loaders "
                    "enforce cleft-manifest semantics (cleft.decoder)"
                ),
            },
            indent=2, sort_keys=True,
        ) + "\n",
    )
    return payload


def load_scut_embeddings(directory) -> tuple:
    """Read what save_scut_embeddings wrote, re-checking it."""
    directory = Path(directory)
    values = np.load(directory / "values.npy")
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    stems = metadata["stems"]
    if values.ndim != 2 or values.shape[1] != EMBED_DIM:
        raise DecoderError(f"values {values.shape}: expected (n, {EMBED_DIM})")
    if len(stems) != len(values):
        raise DecoderError(
            f"{len(stems)} stems for {len(values)} rows -- the artifact "
            "was edited or truncated"
        )
    return values, stems, metadata
