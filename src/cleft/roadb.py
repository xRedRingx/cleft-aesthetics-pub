"""Road B: the same phase sequence, with the branches Road A never tried.

**Road A is not superseded. It is the control arm.** Every Road A result is
reported, and a Road B result is meaningful precisely because there is a
matched Road A number beside it. Both roads converge at Phase 9, which is not
duplicated.

**[REFRAMED 2026-08-16, phase9.ROAD_B_IS_THE_ANNEX -- the paragraph above
stands as history, corrected here in prose.]** Road A is the MAIN PROJECT
and Road B is its ANNEX of additional and alternative ideas -- never a
parallel or co-equal road, and the "control arm" sentence above inverted
the actual hierarchy (it cast the main project as the control of its own
annex). There is no convergence at Phase 9 because there was never a
fork. Every measurement in this module stands unchanged evidentially;
the framing in any write-up is "additional ideas explored", and the
``roadb_`` prefixes stay exactly as they are -- renaming history would
damage provenance.

**And the finding that governs every Road B comparison** is
``ladder.COHORT_CANNOT_RESOLVE``: 29 of 30 paired comparisons on Road A are
unresolvable, with per-seed BCa intervals ~0.28 wide at n=237. **A resolution
improvement of 0.03 will not be claimable.** That is why the resolution axis
has three points rather than two -- 224, 512, 768 as a monotone *trend* is far
harder to attribute to chance than any single delta, and a trend across three
points is reportable where a pair is not. The arms are designed so the trend
is the result.
"""

from __future__ import annotations

#: **[MEASURED 2026-08-08] Non-content fraction by staging, and this is why
#: geometry is an axis rather than a detail.**
#:
#: The first draft of the brief said non-square staging means "every patch,
#: token and region the backbone sees is face rather than padding". True of
#: PADDING, false of non-face content: ``content_mask`` is a **trapezium
#: inside a rectangle**, so removing the pad leaves the corners.
#:
#:     G1 square @ median AR   0.4064   pad + corners
#:     G1 non-square           0.1989   the corners survive
#:     G2 non-square           0.0000   nothing
#:
#: So non-square HALVES the non-content at G1 and only G2 reaches zero. Four
#: staging settings could not have answered the question; ten can.
NON_CONTENT_FRACTION = {
    "measured": "2026-08-08",
    "via": "augment_sheet.content_mask",
    "g1_square_median_ar": 0.4064,
    "g1_non_square": 0.1989,
    "g2_non_square": 0.0000,
    "why_it_is_an_axis": (
        "the mask is a trapezium inside a rectangle, so removing the pad "
        "leaves the corners. Non-square halves the non-content at G1; only "
        "G2 reaches zero"
    ),
    "consequence_for_saliency": (
        "0.1989 and 0.0000 are different expectations, and a normalised "
        "saliency map read at either needs its own -- phase8's refuted "
        "framing finding is what happens when one expectation is used for a "
        "different region"
    ),
}

#: The resolution axis. 224 is Road A's, reused as the control rather than
#: re-run.
RESOLUTIONS = (224, 512, 768)
CONTROL_RESOLUTION = 224

#: **[MEASURED 2026-08-07, brief §2] Upsampling counts by target**, over the
#: 473 source images. **768 mixes two populations** -- 17.1% interpolated,
#: 82.9% downsampled -- so its arm is reported BOTH ways: all patients, and
#: restricted to those that genuinely downsample. If the two agree the trend
#: is clean; if not, the 768 result is partly about interpolation.
#:
#: **[LABELLED 2026-08-09] Counted by the SHORTER side, and staging scales by
#: the LONGER** -- so these are CEILINGS on the interpolation counts, not the
#: counts (a longer-side-flagged source is always shorter-side-flagged too).
#: The values stay because they are true of what they measured; the flag now
#: measures the governing side. ``UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE``,
#: including what this does to the both-ways plan above.
UPSAMPLING_COUNTS = {
    224: 0, 384: 1, 512: 7, 768: 81,
}
N_SOURCES = 473

#: **[DECIDED 2026-08-08] Eight staging settings, not four.**
#:
#: 512 and 768 x square and non-square x G1 and G2. The brief's original four
#: collapsed geometry, and ``NON_CONTENT_FRACTION`` is why that could not
#: work: non-square alone does not reach zero non-content.
#:
#: **The eighth cell -- 768 non-square G2 -- is the one that answers whether
#: zero non-content helps at all.** Every other cell leaves some.
#:
#: **[DECIDED 2026-08-08] Plus 224 NON-SQUARE, at both geometries -- so TEN
#: settings, not eight, and not nine.**
#:
#: The brief proposed one extra cell. It is two: both non-square families need
#: a third point, and ``(nonsquare, g1)`` and ``(nonsquare, g2)`` are distinct
#: settings.
#:
#: **Why it is meaningful, and it is more than arithmetic.** ``stage`` pads to
#: square and THEN resizes, so at a given target the content occupies the same
#: pixel box either way -- 166x224 at the median aspect ratio. **Non-square
#: does not change the content resolution at all; it removes the pad.** The
#: aspect axis and the resolution axis are therefore orthogonal.
#:
#: Which makes 224 non-square the **cleanest test of Branch 1's pad claim**:
#: it has the same content pixels as Road A's square 224 and differs only by
#: the pad. Every 512 comparison confounds pad removal with resolution; this
#: pair does not.
#:
#: And it converts both non-square families from two points to three, so §5's
#: trend argument -- a monotone trend is reportable where a single delta is
#: not -- covers all four families rather than two. Without it the G2
#: non-square cell, the one that answers whether zero non-content helps, rests
#: on a delta, which is exactly what 29-of-30 says cannot be claimed.
#:
#: **224 SQUARE is not re-staged.** It is Road A's ``staged_v1``, reused as
#: the control.
#:
#: Cost is not a consideration (brief §1: never scope out on cost; cost it and
#: report the cost).
STAGING_RESOLUTIONS = (224, 512, 768)

TWO_TWENTY_FOUR_NON_SQUARE = {
    "decided": "2026-08-08",
    "added": ["roadB_p2_224_nonsquare_g1", "roadB_p2_224_nonsquare_g2"],
    "count": "two cells, not one -- both non-square families need a third point",
    "totals": {"settings": 10, "artifacts": 10, "new_runs": 10},
    "why_meaningful": (
        "stage pads THEN resizes, so at a given target the content occupies "
        "the same pixel box either way. Non-square does not change content "
        "resolution, it removes the pad -- so the aspect and resolution axes "
        "are orthogonal"
    ),
    "cleanest_pad_test": (
        "224 non-square has the same content pixels as Road A's square 224 "
        "and differs only by the pad. Every 512 comparison confounds pad "
        "removal with resolution; this pair does not"
    ),
    "fixes_the_two_point_families": (
        "both non-square families go from two points to three, so §5's trend "
        "argument covers all four. Without it the G2 non-square cell rests on "
        "a delta, which 29-of-30 says cannot be claimed"
    ),
    "224_square_is_not_restaged": "it is Road A's staged_v1, reused as the control",
}


def staging_settings() -> list[dict]:
    """The TEN Road B staging settings, as data.

    Named ``roadB_p2_<resolution>_<aspect>_<geometry>``, so a run directory
    says what it is without a lookup.
    """
    out = []
    for resolution in STAGING_RESOLUTIONS:
        for aspect in ("square", "nonsquare"):
            for geometry in ("g1", "g2"):
                # 224 SQUARE is Road A's staged_v1, reused as the control
                # rather than re-staged. Only its non-square counterparts are
                # new. TWO_TWENTY_FOUR_NON_SQUARE.
                if resolution == CONTROL_RESOLUTION and aspect == "square":
                    continue
                out.append({
                    "name": f"roadB_p2_{resolution}_{aspect}_{geometry}",
                    "resolution": resolution,
                    "aspect": aspect,
                    "geometry": geometry,
                    "upsampled_sources": UPSAMPLING_COUNTS[resolution],
                    "mixes_populations": UPSAMPLING_COUNTS[resolution] > 0,
                    # The expectation a saliency map read on this staging
                    # needs. Reported per setting so it is never inherited.
                    "expected_non_content": (
                        None if aspect == "square"
                        else NON_CONTENT_FRACTION[f"{geometry}_non_square"]
                    ),
                    # Its OWN artifact, so a partial re-run moves only this
                    # hash. ARTIFACT_PER_SETTING.
                    "artifact": (
                        f"roadb_{resolution}_{aspect}_{geometry}_v1"
                    ),
                })
    return out


#: **[MEASURED 2026-08-08] The asymmetry gate cannot be run per setting,
#: because it never sees a setting. Running it eight times would test
#: nothing.**
#:
#: ``trapezium.asymmetry_is_preserved(trapezium, heights, offsets, tolerance)``
#: takes **no size argument**. It is purely analytic -- it evaluates
#: ``half_width_at`` and ``unwarp_x``, both fractions of the content box -- so
#: it returns the identical result at 224, 512 and 768. "Run the gate per
#: setting and record the residual per setting" would produce eight identical
#: passes and eight identical residuals.
#:
#: **This is the transfer question one level deeper than expected.** The
#: concern was that a gate verified at 224 might not hold at 512. The truth is
#: that the gate is blind to resolution, so it neither holds nor fails to hold
#: -- it was never measuring the thing that varies.
#:
#: **What DOES vary is the pixel path**, ``trapezium.unwarp``, and it has
#: never been gated at any resolution. Measured by unwarping a mark placed at
#: a known offset and comparing its recovered centre against ``unwarp_x``:
#:
#:     224   residual 3.29e-03
#:     512   residual 1.40e-03
#:     768   residual 9.14e-04
#:
#: Shrinking roughly as 1/size, which is resampling quantisation rather than a
#: defect. But it is **six orders of magnitude looser than the analytic 1e-9**,
#: so "Road A verified G2 to machine epsilon" is true of the MAP and not of
#: the staged pixels -- and a reader will take it for the latter.
#:
#: **So Phase 2's per-setting gate is the PIXEL residual, and it is new work
#: rather than a re-run.** The analytic gate still runs once, because it is
#: still the thing that establishes the map is right; running it eight times
#: is what would be theatre.
ASYMMETRY_GATE_IS_RESOLUTION_BLIND = {
    "measured": "2026-08-08",
    "analytic_gate": "trapezium.asymmetry_is_preserved -- takes no size argument",
    "consequence": (
        "running it per setting yields eight identical passes and tests "
        "nothing about the settings"
    ),
    "what_varies": "the pixel path, trapezium.unwarp, never gated at any size",
    "pixel_residual": {224: 3.29e-03, 512: 1.40e-03, 768: 9.14e-04},
    "scales_as": "roughly 1/size -- resampling quantisation, not a defect",
    "machine_epsilon_means_the_map": (
        "Road A's 1e-9 verification is of the analytic map. The staged pixels "
        "are six orders looser, and a reader will take the claim for the "
        "pixels"
    ),
    "phase_2_gate": (
        "the PIXEL residual, per setting -- new work, not a re-run. The "
        "analytic gate runs ONCE"
    ),
}


#: **[DECIDED 2026-08-08] Two sheets, each with one job, because one sheet
#: cannot do both and a sheet nobody reads is the defect this exists against.**
#:
#: **The axis sheet: a few patients x all ten settings, panels downscaled to
#: a common display size.** The review question at Phase 2 is whether the
#: framing, the pad and the trapezium are right AND how they differ across the
#: axis -- so the axis has to be on one page. Downscaling makes ten columns
#: legible on screen.
#:
#: **But downscaling destroys exactly what a resolution axis is about**, so it
#: cannot be the only sheet: everything rendered at a common size looks equally
#: sharp, and that is the one thing 512-versus-768 should show.
#:
#: **The detail strip: the flagged upsampled patients, at NATIVE resolution,
#: one setting per row.** They are the patients where interpolation is real,
#: so they are the ones whose sharpness is worth looking at, and they are a
#: small enough set to render natively.
#:
#: **[CORRECTED 2026-08-09]** "eight" throughout predated the two 224
#: non-square cells; the axis is ten. And the strip's counts were quoted from
#: the 473-source measurement -- the staged population's measured frontal
#: counts are **1 at 512 and 41 at 768** (first staging runs, 2026-08-09).
#:
#: One sheet per setting with a shared sample was the alternative. Rejected:
#: it puts the axis across ten files, so comparing settings becomes flipping
#: between images, and the comparison is the review.
SHEET_DENSITY = {
    "decided": "2026-08-08",
    "corrected": (
        "2026-08-09: 'eight' predated the 224 non-square cells; the strip's "
        "7/81 were the 473-source ceilings, and the measured frontal counts "
        "are 1 and 41"
    ),
    "axis_sheet": {
        "rows": "patients", "columns": "all ten settings",
        "scale": "downscaled to a common display size",
        "answers": "is the framing right, and how does it differ across the axis",
    },
    "detail_strip": {
        "rows": "one per setting",
        "patients": (
            "the flagged upsampled ones -- measured frontal counts 1 at 512, "
            "41 at 768 (the 7/81 were ceilings over all 473 sources)"
        ),
        "scale": "NATIVE",
        "answers": "do the upsampled patients look interpolated",
    },
    "why_two": (
        "downscaling to fit ten columns destroys the sharpness difference a "
        "resolution axis is about, so the axis sheet cannot answer it and the "
        "detail strip does"
    ),
    "rejected": (
        "one sheet per setting with a shared sample -- it puts the axis across "
        "ten files, and comparing settings is the review"
    ),
}


#: **[MEASURED 2026-08-09, first gate run] Within an aspect, g1 and g2 record
#: IDENTICAL residuals to full precision -- and that is by construction, not
#: the aliasing defect returning.**
#:
#: "Identical to 16 digits across a factor that should matter" is exactly the
#: signature the G2-aliasing defect had, so the reason is recorded where a
#: verdict reader will meet the four identical pairs:
#:
#: ``roadb_staging.pixel_asymmetry_residual`` is a **synthetic probe**. It
#: builds a blank frame at the setting's column count, places a mark, and
#: runs it through ``trapezium.unwarp``. It reads no staged pixel and takes
#: no geometry argument, so the residual is a property of COLUMN COUNT alone
#: -- the resampling quality of the unwarp path at that size, which is what
#: the G2 cells at that size consume. A g1 artifact records it as the unwarp
#: error at its resolution, not as a property of its own never-unwarped
#: pixels.
#:
#: **Identical residuals are expected. Identical ARRAYS would be the defect**
#: -- and the arrays are what answer the aliasing question: the manifest's
#: ``g2_pixels`` note, the bit-for-bit unwarp test, and the axis sheet where
#: G1's corners are visible and G2's are not.
RESIDUAL_IS_GEOMETRY_BLIND = {
    "measured": "2026-08-09, first gate run",
    "identical_pairs_expected": (
        "g1 and g2 within an aspect share the residual to full precision; "
        "identical ARRAYS, not identical residuals, would be the defect"
    ),
    "why": (
        "pixel_asymmetry_residual is a synthetic probe of the unwarp path at "
        "the setting's column count; it reads no staged pixel and takes no "
        "geometry argument"
    ),
    "what_answers_aliasing_instead": (
        "the staged arrays: the g2_pixels manifest note, the bit-for-bit "
        "unwarp test, and the axis sheet's corners"
    ),
}


#: **[MEASURED 2026-08-09] The sheet review PASSED -- and it raised a
#: hypothesis that then failed. Both are recorded, because the refuted
#: version is the one a viewer reaches for on seeing the sheet.**
#:
#: The axis sheet: same face at every setting; pad present at square and
#: absent at non-square; corners present at G1 and absent at G2 -- the
#: picture agrees with the recorded content fractions.
#:
#: **The hypothesis:** three of four axis patients appeared to have
#: compressed lips and nose at 512 and 768 square, suggesting square staging
#: distorts facial proportions. The stakes were maximal: square staging is
#: Road A's control path, so a distortion there would have touched every arm
#: including the 0.2520 baseline.
#:
#: **The measurement refuted it.** Against ``staged_v1``'s recorded geometry,
#: the staged aspect ratio matches the source to within 0.4%, max deviation
#: 0.00383 -- which is the half-pixel rounding of each content dimension,
#: not distortion. **Road A is clean.** A laptop test pins the mechanism
#: (``test_square_staging_preserves_aspect_ratio_to_pixel_rounding``).
#:
#: **What was actually seen reframes what non-square buys.** The face
#: occupies 59% of the frame at square G1 against 100% at non-square G2, so
#: the same face is drawn smaller with white around it -- and beside a
#: full-frame panel that READS as compressed. So the argument for non-square
#: is not that square distorts. It is that square spends a fixed compute
#: budget on constant pixels: ViT-B/16 has 196 patch tokens at 224 whatever
#: is in them, and at square G1 roughly 40% of them carry no patient
#: information (``NON_CONTENT_FRACTION``).
SHEET_REVIEW_PASSED = {
    "reviewed": "2026-08-09",
    "axis_sheet": (
        "same face at every setting; pad present at square, absent at "
        "non-square; corners present at G1, absent at G2 -- agrees with the "
        "recorded content fractions"
    ),
    "detail_strip": (
        "the 41 flagged patients look high quality, not soft -- which fired "
        "the review question's own alarm; see "
        "UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE"
    ),
    "hypothesis": {
        "claim": (
            "square staging compresses lips and nose at 512 and 768 -- "
            "raised on three of four axis patients"
        ),
        "stakes": (
            "square is Road A's control path; a distortion there would touch "
            "every arm including the 0.2520 baseline"
        ),
        "refuted": True,
    },
    "refutation": {
        "measured_against": "staged_v1's recorded geometry",
        "max_deviation": 0.00383,
        "result": (
            "staged aspect ratio matches the source to within 0.4% -- "
            "pixel rounding, not distortion. Road A is clean"
        ),
    },
    "reframing": {
        "not": "square does not distort; the compressed look was RELATIVE size",
        "argument": (
            "the face occupies 59% of the frame at square G1 against 100% at "
            "non-square G2, so square spends a fixed compute budget on "
            "constant pixels: ViT-B/16 has 196 patch tokens at 224 whatever "
            "is in them, and at square G1 roughly 40% carry constant white"
        ),
    },
    "why_both_recorded": (
        "the refuted reading is the one a viewer reaches for on the sheet; "
        "quote the measurement whenever the sheet is shown"
    ),
}


#: **[CORRECTED 2026-08-09, sheet review] The upsampling flag measured the
#: wrong side, and the review question caught it exactly as written:** the
#: 41 flagged patients looked sharp, and "if interpolation is not visible,
#: the upsampling flag may be measuring the wrong thing."
#:
#: The flag compared the source's SHORTER side against the target. But both
#: staging paths compute ``scale = target / max(width, height)`` -- **they
#: scale by the LONGER side** -- so content is interpolated iff the longer
#: side is under the target. A source under 768 on its shorter side and over
#: it on its longer DOWNSAMPLES, and the shorter-side flag called it
#: upsampled. R2's exact shape: ``n_upsampled`` was named for interpolation
#: and computed from a side the staging does not scale by.
#:
#: **What survives:** the 473-source counts (7 at 512, 81 at 768) were also
#: shorter-side counts, so they remain valid CEILINGS a fortiori -- a
#: longer-side-flagged source is always shorter-side-flagged too. The
#: measured frontal counts (1 at 512, 41 at 768) are shorter-side counts as
#: well; **the true interpolation counts need the longer side and are
#: cluster state, not yet measured.**
#:
#: **What it changes:** the 768 both-ways reporting plan and the
#: ``mixes_populations`` field are ceiling readings -- "may mix, pending the
#: longer-side measurement". If the true count is zero or near it, both-ways
#: reporting at 768 is unnecessary, which would remove 768's main caveat.
#:
#: **What it does not change:** the staged pixels. The flag is metadata; the
#: v1 artifacts' arrays are untouched and need no re-staging. Their manifests
#: record shorter-side flags and each manifest now names which side its
#: flags measured, so the two quantities cannot be read as one.
UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE = {
    "corrected": "2026-08-09, sheet review",
    "caught_by": (
        "the detail strip: 41 flagged patients, none soft -- the review "
        "question's own failure-mode alarm"
    ),
    "why_wrong": (
        "the flag compared the SHORTER side; both staging paths compute "
        "scale = target / max(width, height), so they scale by the LONGER "
        "side and interpolation needs longer < target"
    ),
    "ceilings_still_hold": (
        "the 473-source counts are shorter-side counts, hence a valid "
        "ceiling a fortiori: longer-side-flagged implies shorter-side-flagged"
    ),
    "true_counts": (
        "MEASURED 2026-08-09: 0 at 224, 1 at 512, 21 at 768 over the 237 "
        "frontal sources -- MEASURED_INTERPOLATION_COUNTS. The 1-at-512 / "
        "41-at-768 recorded in v1 manifests are shorter-side counts"
    ),
    "consequence": (
        "RESOLVED by the measurement: both-ways stays at 768 (21 of 237 "
        "genuinely interpolate, one in eleven) as a caveat on one cell "
        "rather than a mixed population; 512 at 1 of 237 is effectively "
        "clean. MEASURED_INTERPOLATION_COUNTS['reporting']"
    ),
    "v1_manifests": (
        "record shorter-side flags; the staged pixels are unaffected and "
        "need no re-staging. Each NEW manifest names the side its flags "
        "measured; v1 manifests predate the key and are normalised "
        "reader-side by roadb_staging.manifest_upsampling_quantity"
    ),
}


#: **[MEASURED 2026-08-09] The longer-side counts, over the 237 frontal
#: sources: 0 at 224, 1 at 512, 21 at 768.** Longer side median 2758,
#: p05 732, min 340.
#:
#: **The shorter-side flag overcounted 768 by twenty** -- the twenty sharp
#: panels the review saw were never interpolated. The true 768 rate is
#: **8.9%, not 17.3%**.
#:
#: **The reading, which matters when the resolution trend is read:** 512 at
#: 1 of 237 is effectively clean; 768 at 21 of 237 is not -- one patient in
#: eleven genuinely interpolates. So both-ways reporting STAYS at 768, but
#: as a **caveat on the trend's top point**, not a mixed population -- and
#: the middle point carries no caveat at all, which makes 512 the stronger
#: arm despite fewer pixels.
#:
#: These are the numbers ``assert_upsampling_counts`` holds a 237-frontal
#: staging to exactly; the 473 shorter-side counts remain the ceiling for
#: any other population.
MEASURED_INTERPOLATION_COUNTS = {
    "measured": "2026-08-09, longer side over the 237 frontal sources",
    "by_target": {224: 0, 512: 1, 768: 21},
    "n_frontal": 237,
    "longer_side_stats": {"median": 2758, "p05": 732, "min": 340},
    "rate_768": "8.9% -- 21 of 237, not the shorter-side flag's 17.3%",
    "overcount": (
        "the shorter-side flag overcounted 768 by twenty; those twenty are "
        "the sharp panels the review saw -- never interpolated"
    ),
    "reading": {
        224: "clean by construction -- nothing upsamples",
        512: "effectively clean -- 1 of 237",
        768: "NOT clean -- 21 of 237, one patient in eleven",
    },
    "reporting": (
        "both-ways stays at 768, recorded as a CAVEAT on one cell rather "
        "than a mixed population; 512 needs none. The ordering matters when "
        "the resolution trend is read: the trend's top point carries the "
        "caveat, its middle point does not"
    ),
    "v1_strip_missed_nothing": (
        "shorter-side flags are a superset of the interpolated set, so the "
        "reviewed strip contained every genuinely interpolated patient plus "
        "twenty sharp ones"
    ),
}

#: The staged population. The measured counts above are a property of THIS
#: population; holding any other cohort to them would be the population trap
#: again, which is why the exact gate keys on the count.
N_FRONTAL = 237


#: **[DESIGNED 2026-08-09] Road B Phase 3's seed-band design: measure the
#: extremes before fixing the band count.**
#:
#: Road A's SD 0.0137 came from one training procedure (frozen ViT probe at
#: 224 square G1); Road B has ten cells. Per-cell bands give every comparison
#: a different denominator with no shared basis; one band treats different
#: procedures as the same. The working hypothesis is **three bands, one per
#: resolution, measured at square G1** -- resolution is the axis under test
#: and geometry/aspect are held constant while measuring it.
#:
#: **But the band count is decided by measurement, not argument** -- two
#: sweeps at the extremes first, with the decision rule registered before
#: either number exists. And one reframing that lowers the stakes: under
#: PLAN §4.12.1 a band NEVER denominates a claim -- every arm reports its
#: own seed SD and a delta combines the two arms' own SDs, inherited never.
#: The Phase 3 band is the PLANNING instrument (the claimable-delta table,
#: seed counts per arm) and the gate-2 pass. A wrong band count mis-plans
#: seed counts; it cannot corrupt a claim. The inherited-band error at the
#: scale of a road is therefore blocked by the claim criterion itself --
#: what this design protects is the planning and the gate.
SEED_BAND_DESIGN = {
    "designed": "2026-08-09",
    "hypothesis": "three bands, one per resolution, measured at square G1",
    "wrong_answers": {
        "per_cell": (
            "ten denominators; cross-cell comparisons lose any shared basis"
        ),
        "one_band": (
            "treats 224 square and 768 non-square as one procedure across "
            "input size, token count and content fraction"
        ),
    },
    "not_the_denominator": (
        "PLAN 4.12.1: every arm reports its OWN seed SD, and a delta between "
        "arms combines the two arms' own SDs -- inherited, never. The Phase "
        "3 band is the planning instrument and the gate-2 pass; a wrong "
        "band count mis-plans seed counts, it cannot corrupt a claim"
    ),
    "first_measurement": {
        "configs": (
            "roadb_p3_seed_band_224_square_g1",
            "roadb_p3_seed_band_768_square_g1",
        ),
        "seeds_768": (
            "Road A's own ten -- the same seed drives the same inner-val "
            "split, so the split component is matched pairwise and the "
            "per-seed deltas against the recorded gate-2 sweep isolate the "
            "embedding's contribution"
        ),
        "seeds_224": (
            "a DISJOINT ten on Road A's own artifact: re-running Road A's "
            "seeds would reproduce 0.0137 bit-for-bit (determinism) and "
            "measure nothing. The disjoint draw is a second independent "
            "estimate of the SAME band -- the empirical noise floor the "
            "agreement test needs -- and the apparatus check, in one run"
        ),
    },
    "sensitivity": (
        "an SD from ten seeds carries ~23% sampling error, and the ratio of "
        "two ten-seed SDs is F(9,9)-distributed -- so at ten seeds each, "
        "'agree within noise' can only detect a divergence of ~2x. Forty "
        "seeds each detects ~1.4x and costs minutes (frozen probes, PLAN "
        "2.7: seed counts from the band, not the compute budget). The "
        "configs ship ten as directed; widening is one line"
    ),
    "decision_rule_preregistered": (
        "compute the F-based 95% CI on SD_768/SD_224, with the 224 "
        "replicate-vs-original ratio reported beside it as the observed "
        "noise floor. CI contains 1: the sweep cannot distinguish the bands "
        "-- ONE planning band, the LARGER of the two (conservative), and "
        "512 runs as CONFIRMATION rather than establishment. CI excludes 1: "
        "per-resolution bands, and 512 is measured in its own right. "
        "Registered before either number exists -- a window chosen after "
        "seeing the result is a forking path (phase7c.ROUND_2_IS_ROUND_1_"
        "TRUNCATED)"
    ),
    "square_g1_representativeness": (
        "OPEN -- square G1's content fraction (0.59) is the lowest of the "
        "four aspect/geometry combinations, so its band may not represent a "
        "non-square cell. Answered by one contingent sweep at a non-square "
        "cell (768 nonsquare g2, ViT-only anyway) if the non-square "
        "sub-axis needs its own planning band; claims never inherit either "
        "way, so this decides three-vs-six for PLANNING only"
    ),
    "vit_768_note": (
        "at 768 the ViT runs dynamic with its position grid interpolated "
        "14x14 -> 48x48 (factory.timm_kwargs_for, measured 2026-08-09); "
        "that is part of the procedure being banded, and the extraction "
        "report records input_size and dynamic_input so the artifact says "
        "what ran"
    ),
}


#: **[MEASURED 2026-08-09] The pre-registered rule, applied -- first at the
#: extremes, then re-run with all three points in hand: ONE planning band.**
#:
#: Extremes first: SD ratio 768/224 = 1.346, F(9,9) 95% CI ~0.67-2.71,
#: contains 1. **The replicate did its job**: 0.012436 against Road A's
#: 0.0137 is 1.10 on disjoint seeds of the SAME band -- the observed noise
#: floor -- so 1.35 across resolutions sits inside re-sampling alone.
#:
#: **Then 512 arrived with the WIDEST spread** -- 0.021425, ratio 1.72
#: against 224, near the edge of what ten seeds can see (2.01) -- and every
#: pairwise interval still contains 1 (512/224: [0.86, 3.46]; 512/768:
#: [0.64, 2.57]). So the outcome stands at one band, and the value moves:
#: the registered text said "the LARGER of the two" because conservatism was
#: the principle, and **a planning band below an observed arm SD would
#: under-plan every arm that looks like 512**. The largest of the three is
#: the faithful application; keeping 0.016739 and calling 512 an outlier
#: would be a data-dependent choice of exactly the kind the
#: pre-registration exists to block.
#:
#: The claimable-delta table re-derives from the resolved band (PLAN 4.12,
#: ``train/phase3.claimable_delta``): **0.059 at one seed, 0.027 at five,
#: 0.019 at ten.** Road B arms at five seeds resolve 0.027 -- coarser than
#: Road A's 0.017, and that is a fact about this regime, not a choice.
#:
#: Whether 512's wider spread is real or sampling is itself undecidable at
#: ten seeds; forty per point would see 1.4x. Nothing currently hangs on it
#: -- the band plans, it denominates nothing (PLAN 4.12.1).
SEED_BAND_RESOLVED = {
    "measured": "2026-08-09",
    "applied_rule": "SEED_BAND_DESIGN.decision_rule_preregistered",
    "sweeps": {
        "roadb_224_replicate": {
            "mean": 0.2319, "sd": 0.012436, "n_seeds": 10,
            "seeds": "disjoint draw on Road A's artifact",
        },
        "roadb_512": {
            "mean": 0.0857, "sd": 0.021425, "n_seeds": 10,
            "seeds": "Road A's ten, paired",
        },
        "roadb_768": {
            "mean": 0.0898, "sd": 0.016739, "n_seeds": 10,
            "seeds": "Road A's ten, paired",
        },
    },
    "pairwise_f_intervals_95": {
        "768/224": (0.67, 2.71),
        "512/224": (0.86, 3.46),
        "512/768": (0.64, 2.57),
    },
    "outcome": "one planning band",
    "planning_band": 0.021425,
    "two_point_planning_band": 0.016739,
    "three_point_application": (
        "the registered text said 'the LARGER of the two' because "
        "conservatism was the principle; a planning band below an observed "
        "arm SD would under-plan every arm that looks like 512, so the "
        "largest of the three is the faithful application. Keeping 0.016739 "
        "and calling 512 an outlier would be the data-dependent choice the "
        "pre-registration exists to block"
    ),
    "noise_floor": (
        "replicate vs Road A: 0.012436 against 0.0137 is 1.10 on disjoint "
        "seeds of the SAME band, so 1.35 across resolutions is inside "
        "re-sampling alone"
    ),
    "middle_point_note": (
        "whether 512's wider spread is real or sampling is undecidable at "
        "ten seeds (1.72 observed against a 2.01 detection edge); forty per "
        "point would see 1.4x. Nothing currently hangs on it"
    ),
    "claimable_delta": {1: 0.059, 5: 0.027, 10: 0.019},
    "planning_only": (
        "PLAN 4.12.1 stands: every arm reports its own SD; this band plans "
        "seed counts and gates, it denominates nothing"
    ),
}


#: **[OBSERVED 2026-08-09 -- NOT A FINDING. Verification gates it.]**
#:
#: 768 square G1 scored mean 0.0898 against the 224 replicate's 0.2319 --
#: 0.142 lower, an order of magnitude larger than any effect this project
#: has measured, with non-overlapping intervals. **More resolution made it
#: dramatically worse.** The framing finding is the standing lesson here:
#: verify the instrument before quoting what it shows.
#:
#: **Two explanations, and they must be separated before anything is
#: recorded as a finding:**
#:
#: * **plumbing** -- the dynamic-ViT path mis-ran on the pinned image (timm
#:   1.0.7 against the 1.0.27 the mechanics were verified on);
#: * **real** -- a frozen ImageNet ViT at 768 produces 2,304 patch tokens
#:   with position embeddings interpolated 14x14 -> 48x48, far outside the
#:   pretraining distribution, and CLS pooling over ten times more patches
#:   dilutes the signal.
#:
#: **What the run's completion already excludes**: the legacy 224-locked
#: model dies at timm's patch-embed assert on a 768 input, and guard 3
#: hash-verified the staged artifact -- so 768-sized pixels went through a
#: dynamically-sized model, or there would be no numbers at all. **What it
#: does not exclude**: a subtle interpolation defect that runs and degrades
#: -- which is exactly what the record fields exist to check.
RESOLUTION_DROP_PENDING_VERIFICATION = {
    "observed": "2026-08-09",
    # The name keeps PENDING because that is what this record WAS -- the
    # status says what it is now, the G2_WHITE_DEFECT convention.
    "status": (
        "VERIFIED 2026-08-09 -- the extraction record confirmed the "
        "instrument; the finding is scoped in "
        "BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER"
    ),
    "means": {224: 0.2319, 768: 0.0898},
    "delta": -0.1421,
    "scale": (
        "an order of magnitude larger than any effect measured in this "
        "project; the intervals do not come close to overlapping"
    ),
    "explanations": {
        "plumbing": (
            "EXCLUDED by the verification -- the dynamic path ran as built "
            "on the pinned image's timm 1.0.7"
        ),
        "real": (
            "STANDS: 2,304 patch tokens from a 224-pretrained frozen ViT, "
            "position grid interpolated to 48x48, far outside the "
            "pretraining distribution; CLS pools over ten times more patches"
        ),
    },
    "completion_already_excludes": (
        "a 224-locked model (dies at timm's patch-embed assert on 768 "
        "input) and wrong-size pixels (guard 3 hash-verified the artifact)"
    ),
    "completion_does_not_exclude": (
        "an interpolation defect that runs and silently degrades -- the "
        "class of failure that produces numbers; that is what the record "
        "fields decided"
    ),
    "verification": {
        "verified": "2026-08-09, from the run's own record",
        "input_size": [768, 768],
        "dynamic_input": True,
        "embedding_dim": 768,
        "finalized": "cleanly",
        "record_read": (
            "seed_variance.json -- a seeded sweep's summary file"
        ),
        "checklist_correction": (
            "the checklist named the wrong file: it said metrics.json, and "
            "a seeded train_cv writes seed_variance.json as its summary. "
            "The glob that found nothing was the checklist's error, not a "
            "missing write -- corrected here so the next checklist reads "
            "the right record"
        ),
    },
    "what_512_decides": (
        "DECIDED: 512 at 0.0857 sits with 768, not with 224 -- the curve "
        "is a CLIFF at the pretraining boundary, then flat"
    ),
    "resolved_into": "BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER",
}


#: **[MEASURED 2026-08-09] Branch 1's premise is refuted for the
#: frozen-transfer regime -- and the scope of that sentence is the whole of
#: its value.**
#:
#:     224   0.2319 (sd 0.012436)
#:     512   0.0857 (sd 0.021425)
#:     768   0.0898 (sd 0.016739)
#:
#: **A cliff, not a decline.** 512 and 768 are indistinguishable -- 0.0041
#: apart against arm_means_95 of 0.017 at ten seeds each -- and both sit
#: ~0.146 below 224. Performance falls off the moment the input leaves the
#: pretraining resolution and then stays flat.
#:
#: **The shape is more informative than a gradient would have been.** A
#: gradual decline would suggest something scaling with resolution. A cliff
#: followed by a plateau says the damage is done by LEAVING 224 at all, and
#: going further costs almost nothing more -- which fits the
#: out-of-distribution account precisely: interpolating a 14x14 position
#: grid to 32x32 or 48x48 is qualitatively the same violation, and the
#: magnitude barely matters.
#:
#: **What is refuted**: that a frozen ImageNet ViT can USE the recovered
#: pixels. **What stays true**: the 10.8x pixel discard is real and measured
#: (brief §2 Branch 1). The refutation is a claim about the frozen-embedding
#: regime, not about resolution in general.
#:
#: **What it does not test**: a backbone fine-tuned at 512 adapts its
#: position embeddings -- a different experiment, and one Road B's Phase 6
#: can still run. The cliff says frozen transfer breaks outside the
#: pretraining resolution; it does not say 512 is useless with training.
#:
#: **That last distinction decides whether Road B continues up the phase
#: sequence or stops here** -- which is why it is recorded before anything
#: else is built on this result.
BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER = {
    "measured": "2026-08-09, three ten-seed sweeps, paired seeds at 512/768",
    "means": {224: 0.2319, 512: 0.0857, 768: 0.0898},
    "sds": {224: 0.012436, 512: 0.021425, 768: 0.016739},
    "shape": "a cliff at the pretraining boundary, not a decline",
    "plateau": {
        "delta": 0.0041,
        "arm_means_95": 0.017,
        "note": (
            "512 and 768 indistinguishable by the project's own named "
            "quantity (PLAN 4.3: arm_means_95 at ten seeds each); the "
            "verdict is insensitive to which threshold variant is used -- "
            "0.004 clears all of them several times over"
        ),
    },
    "reading": (
        "a gradient would suggest something scaling with resolution; a "
        "cliff then a plateau says the damage is done by leaving the "
        "pretraining resolution at all -- interpolating the position grid "
        "to 32x32 or 48x48 is qualitatively the same violation, and the "
        "magnitude barely matters"
    ),
    "refuted": (
        "that a frozen ImageNet ViT can use the recovered pixels -- a claim "
        "about the frozen-embedding regime, not about resolution in general"
    ),
    "still_true": (
        "the 10.8x pixel discard is real and measured; the pixels exist, "
        "the frozen backbone cannot use them"
    ),
    "not_tested": (
        "a backbone fine-tuned at 512 adapts its position embeddings -- a "
        "different experiment, and one Phase 6 can still run. The cliff "
        "does not say 512 is useless with training"
    ),
    "decides": (
        "whether Road B continues up the phase sequence or stops here -- "
        "recorded before anything else is built on this result"
    ),
}


#: **[DECIDED 2026-08-09, the maintainer] Road B CONTINUES up the phase sequence,
#: and Phase 6 runs ALL THREE resolutions -- 224, 512 and 768. Not 512
#: alone.**
#:
#: **Why not drop 768**: 512 and 768 are indistinguishable in the frozen
#: regime -- 0.004 apart against any threshold variant -- so dropping 768
#: would be choosing on a difference the data does not support, which is the
#: error the paired audit spent three phases correcting. And the mechanism
#: gives no reason to expect them to differ once trained: if the cliff is
#: position-grid interpolation, fine-tuning repairs it at both -- a learned
#: 32x32 grid is no more out-of-distribution than a learned 48x48 one.
#:
#: **Why three points**: §5's trend argument -- a monotone trend across
#: three points is reportable where a pairwise delta is not. Collapsing to a
#: pair would give up the thing that makes the axis claimable at all.
#:
#: **Why continue at all**: the cliff refutes frozen ImageNet transfer
#: outside 224, not resolution itself. Phase 6 fine-tunes on SCUT, which
#: adapts the position embeddings during pretraining rather than carrying
#: the violation into the cleft arms -- so it tests exactly what the cliff
#: does not answer. Stopping here would report "more pixels made it worse"
#: when the measurement supports only "more pixels made frozen ImageNet
#: features worse."
#:
#: Cost is not a consideration (brief §1); 768 pretraining being heavier is
#: a fact to plan around, not an argument.
ROAD_B_CONTINUES = {
    "decided": "2026-08-09",
    "decision": "continue; Phase 6 runs 224, 512 AND 768 -- not 512 alone",
    "why_not_drop_768": (
        "512 and 768 are indistinguishable frozen (0.004 against any "
        "threshold variant); dropping 768 would choose on a difference the "
        "data does not support -- the paired-audit error. And the mechanism "
        "predicts no difference once trained: a learned 32x32 grid is no "
        "more out-of-distribution than a learned 48x48 one"
    ),
    "why_three_points": (
        "the trend design: three points are reportable where a pairwise "
        "delta is not (brief §5); a pair gives up what makes the axis "
        "claimable"
    ),
    "why_continue": (
        "the cliff refutes frozen ImageNet transfer outside 224, not "
        "resolution itself; Phase 6's SCUT fine-tuning adapts the position "
        "embeddings during pretraining, testing exactly what the cliff does "
        "not answer. Stopping would report 'more pixels made it worse' when "
        "the measurement supports only 'more pixels made frozen ImageNet "
        "features worse'"
    ),
    "cost": (
        "not a consideration (brief §1); 768 pretraining being heavier is a "
        "fact to plan around rather than an argument"
    ),
}


#: **[MEASURED 2026-08-09] Swin-B is NOT excluded from the resolution axis.
#: The square axis stays FOUR-backbone -- the opposite of the non-square
#: outcome.**
#:
#: Measured at timm 1.0.27, shape mechanics with ``pretrained=False``:
#: building ``swin_base_patch4_window7_224`` at ``img_size=512`` and ``768``
#: yields a state dict with **identical keys and identical shapes -- zero
#: mismatches** -- so the 224 checkpoint loads into the resized model with
#: NO parameter adaptation at all. Forward passes work at both sizes and the
#: embedding stays 1024-dimensional.
#:
#: **Why, and it is the interesting part**: Swin's positional machinery is
#: window-RELATIVE -- 13x13 relative-position bias tables keyed to the
#: 7-pixel window, which does not change with image size. Resizing changes
#: only the window COUNT, which carries no parameters. Unlike ViT, nothing
#: learned is interpolated at build time -- **a cleaner resolution transfer
#: than ViT's**, which is worth saying in the write-up.
#:
#: **The recorded 224-only constraint is true of FEEDING, not BUILDING**:
#: the 224-built model still asserts 224x224 exactly (re-confirmed). What
#: was never tried before is building AT the target size.
#:
#: **Costs and caveats, all named**: (a) 512/4 = 128 and its stage maps
#: (128, 64, 32, 16) are not divisible by window 7, so timm pads windows
#: internally -- feature-level padding is part of what swin-at-512 means;
#: (b) measured at timm 1.0.27, the pinned image has 1.0.7 -- the first
#: Phase 6 swin build is the confirmation, the same caveat the ViT dynamic
#: path carried and passed; (c) this measurement weakens "no rounding
#: reaches it" AS A GENERAL STATEMENT -- the NON-SQUARE decision
#: (``NON_SQUARE_IS_VIT_ONLY``) stands on its own recorded basis and is not
#: reopened here, but the maintainer should know the build-at-size fact exists
#: when that decision is next read.
SWIN_AT_RESOLUTION = {
    "measured": "2026-08-09, timm 1.0.27, pretrained=False shape mechanics",
    "result": (
        "builds at img_size 512 and 768 with state-dict keys and shapes "
        "IDENTICAL to the 224 model -- zero mismatches; the 224 checkpoint "
        "loads with no adaptation. Forward OK, embedding 1024"
    ),
    "why": (
        "positional machinery is window-relative (13x13 tables keyed to "
        "window 7); resizing changes only the parameterless window count. "
        "Unlike ViT, nothing learned is interpolated at build time -- a "
        "cleaner resolution transfer than ViT's"
    ),
    "feeding_vs_building": (
        "the 224-built model still asserts 224x224 exactly (re-confirmed); "
        "the recorded constraint was about feeding, and building AT the "
        "size was never tried before"
    ),
    "caveats": {
        "window_padding": (
            "512/4 = 128 and its stage maps are not divisible by window 7; "
            "timm pads windows internally, and that feature-level padding "
            "is part of what swin-at-512 means"
        ),
        "pinned_timm": (
            "measured at 1.0.27; the image pins 1.0.7 -- the first Phase 6 "
            "swin build confirms, the caveat the ViT dynamic path carried "
            "and passed. RESOLVED: see pinned_confirmation"
        ),
        "non_square_not_reopened": (
            "this weakens 'no rounding reaches it' as a general statement; "
            "NON_SQUARE_IS_VIT_ONLY stands on its recorded basis and is "
            "flagged, not reopened"
        ),
    },
    #: **[MEASURED 2026-08-09, the gate run on timm 1.0.7] CONFIRMED, and
    #: strongly**: parameter names identical, zero shape mismatches,
    #: pretrained tensors EQUAL -- the 224 checkpoint's learned weights load
    #: unchanged at both sizes. Not interpolated, not adapted, byte-equal.
    #:
    #: **And the buffer-not-gated decision was vindicated rather than merely
    #: unfired**: every buffer difference is an ``attn_mask`` --
    #: shifted-window masks computed from feature-map size, carrying no
    #: learning -- and ``layers.3.blocks.1.attn_mask`` differs by NAME
    #: rather than shape at both sizes, because the deepest stage's feature
    #: map at 224 is small enough that the shift is skipped entirely. Gating
    #: on buffers would have refused a pass over a quantity the claim is not
    #: about (R2) -- the exact failure the criterion was written to avoid.
    "pinned_confirmation": {
        "confirmed": "2026-08-09, gate run on timm 1.0.7, pretrained=true",
        "result": (
            "parameter names identical, zero shape mismatches, pretrained "
            "tensors EQUAL at 512 and 768; forward [1, 1024] at both. The "
            "224 checkpoint loads byte-equal -- not interpolated, not "
            "adapted"
        ),
        "buffers": (
            "every difference is an attn_mask -- shifted-window masks "
            "computed from feature-map size, no learning. "
            "layers.3.blocks.1.attn_mask differs by NAME at both sizes: "
            "the deepest stage's 224 map is small enough that the shift is "
            "skipped entirely"
        ),
        "gating_decision": (
            "VINDICATED, not merely unfired: gating on buffers would have "
            "refused a pass over a quantity the claim is not about (R2)"
        ),
        "routing_unlocked": (
            "factory.timm_kwargs_for now routes square non-224 swin to "
            "img_size; non-square swin stays refused (ViT-only)"
        ),
    },
    "consequence": "the square resolution axis stays four-backbone",
}


#: **[PRE-REGISTERED 2026-08-09, before any Phase 6 run] The mechanism
#: contrast is now measured on both architectures, and it makes a
#: prediction. Recorded in the pattern the earlier pre-registrations used:
#: the outcomes and their readings are committed before the numbers exist.**
#:
#: **The measured contrast**: ViT interpolates a LEARNED position grid at
#: build time (14x14 -> 32x32/48x48); Swin interpolates NOTHING -- its
#: positional machinery is window-relative and the 224 checkpoint loads
#: byte-equal at every size.
#:
#: **The prediction**: if the cliff's mechanism is build-time interpolation
#: of learned positional parameters, then Swin survives the resolution
#: change where frozen ViT collapsed.
#:
#: **Two regimes, named separately so the readings cannot blur (R2):**
#:
#: * **Frozen** -- the direct mechanism test, now unlocked by the routing:
#:   a frozen swin probe at 512/768 (the exact analogue of the ViT sweeps).
#:   Swin near its own 224 level supports the interpolation mechanism;
#:   **swin dropping like ViT's ~0.14 refutes the attribution** and points
#:   at causes that afflict both -- token/window count, CLS-vs-pooled
#:   dilution, texture statistics at the wrong scale.
#: * **Trained (Phase 6 proper)** -- both architectures adapt their
#:   positional machinery during SCUT pretraining, so both should recover
#:   toward 224-level performance; a backbone that stays cliffed AFTER
#:   adaptation says the damage was never positional for it.
#: **[MEASURED 2026-08-09] The masked-SCUT rebuild doubled as a determinism
#: check, and the path passed it.** The keeper rebuild's payload rollups are
#: byte-identical to the first build's -- 52273b8d... at 512 and c514727b...
#: at 768 -- across two independent runs at DIFFERENT SHAs. Only the
#: MANIFEST differs, which is what changed. 5,499 faces, two geometries, two
#: sizes: the whole staging path (SIFT-free -- crop, arrival mask, stage,
#: unwarp, all numpy) reproduces to the byte.
#: **[OBSERVED 2026-08-10 -- LAUNCHES STOPPED. ViT is not reproducible, and
#: the 224 re-run cells caught it exactly as designed.]**
#:
#:     swin_b   224  0.8719 vs Road A 0.8718  (+0.0001 -- reproduces)
#:     srgnn    224  0.8412 vs Road A 0.8433  (-0.002)
#:     agnet    224  0.8157 vs Road A 0.8244  (-0.009 -- its recorded
#:                                             ~2.3e-05 regime)
#:     vit_b16  224  0.8120 vs Road A 0.7893  (+0.023 -- an ORDER above
#:                                             every other backbone)
#:     vit_b16  512  0.7680 and 0.7412 -- same config, same SHA, different
#:                   checkpoint SHAs (b151203a / 4f45c33e)
#:
#: **What is answered from code already:** ``deterministic`` is identical to
#: Road A's by dict-equality (the lattice test); and ``dynamic_img_size`` is
#: NOT set on the 224 path -- ``timm_kwargs_for(224, 224)`` returns ``{}``,
#: pinned by test, so the 224 build call is byte-identical to Road A's.
#:
#: **Two candidate mechanisms, and ONE comparison decides:**
#:
#: * **The init is an UNDECLARED INPUT.** ``pretrained=True`` downloads
#:   ImageNet weights at run time; ``HOME=/root`` is ephemeral (PLAN 2.7),
#:   so every run re-downloads -- and guard 3 never hashes what arrives.
#:   The one input the config system does not govern, chain-3's shape: the
#:   weights that produced a result are not recoverable from the config. If
#:   the two 512 runs' ``pretrained_checkpoint.sha256`` differ FROM EACH
#:   OTHER, this is the whole story -- vit's upstream artifact changed
#:   between downloads, no nondeterminism at all -- and the fix is to cache
#:   every backbone's init as a versioned NFS artifact, declared and
#:   hashed like any other input.
#: * **Runtime position-grid resample in the training graph.** The Road B
#:   routing sent non-224 ViT through ``dynamic_img_size=True`` -- pos-embed
#:   resampled via F.interpolate EVERY forward -- and interpolate's CUDA
#:   backward is in the atomics class by this project's own isolated-op
#:   measurement, with divergence appearing ACROSS instances
#:   (``models/agnet.py`` TRAINING_DETERMINISM). Swin's routing adapts at
#:   BUILD (``img_size``), nothing in the training graph -- and swin
#:   reproduces to four decimals. If the two 512 shas MATCH each other,
#:   this is the 512 mechanism, and the fix is to route FIXED-size ViT
#:   through ``img_size`` too (resample ONCE at build -- which is what
#:   SWIN_RESOLUTION_PREDICTION's "interpolates at build time" already
#:   described; the record was ahead of the routing). ``dynamic_img_size``
#:   stays only for the mixed-shape non-square extraction path, which is
#:   forward-only.
#:
#: **The decision tree, in order:**
#:
#: 1. the two 512 runs' ``pretrained_checkpoint.sha256`` vs each other:
#:    differ -> undeclared-input drift explains 512 (and likely 224);
#:    match -> 512 is the dynamic-path atomics, fix the routing;
#: 2. the 224 run's sha vs Road A's recorded value: differ -> the +0.023 is
#:    a different init, the run itself deterministic; match -> the 224
#:    divergence needs its own hunt, because that path is byte-identical
#:    and was measured bitwise;
#: 3. either way, the init becomes a declared artifact -- both mechanisms
#:    pass through the same provenance hole.
#:
#: **The local probe cannot arbitrate and says so**: on the laptop stack
#: (torch 2.13/cu126, RTX 3060) even static ViT-224 diverges across
#: instances at 2e-06 where the pinned image measured bitwise -- only the
#: ORDERING is suggestive (swin 1.5e-08, two orders cleaner than every ViT
#: case). The image is the only environment whose determinism claims
#: transfer to the image.
VIT_NOT_REPRODUCIBLE = {
    "observed": "2026-08-10, launches stopped",
    "caught_by": (
        "the 224 re-run cells -- the free end-to-end determinism check "
        "PHASE_6_TWELVE_CELLS built in"
    ),
    "deltas_224": {
        "swin_b": 0.0001, "srgnn": -0.002, "agnet": -0.009, "vit_b16": 0.023,
    },
    "vit_512_pair": {
        "means": (0.7680, 0.7412),
        "checkpoints": ("b151203a", "4f45c33e"),
        "note": "same config, same SHA -- the trained weights differ",
    },
    "answered_from_code": {
        "deterministic": "identical to Road A by dict-equality",
        "dynamic_at_224": "NOT set -- timm_kwargs_for(224, 224) == {}",
    },
    "mechanisms": {
        "undeclared_init": (
            "pretrained weights re-download every run (ephemeral HOME) and "
            "guard 3 never hashes them -- decided by comparing the two 512 "
            "runs' pretrained_checkpoint.sha256 against each other"
        ),
        "dynamic_resample": (
            "dynamic_img_size puts pos-embed F.interpolate in the training "
            "graph every forward; its CUDA backward is atomics-class "
            "(agnet.TRAINING_DETERMINISM); swin adapts at build and "
            "reproduces. Fix: img_size for fixed-size ViT, dynamic only "
            "for mixed-shape forward-only extraction"
        ),
    },
    "decision": (
        "512-pair shas vs each other first; then 224 vs Road A's record; "
        "the init becomes a declared, hashed NFS artifact in every branch"
    ),
    "local_probe": (
        "cannot arbitrate: the laptop stack diverges across instances even "
        "on static 224 where the image is bitwise; only the ordering is "
        "suggestive (swin two orders cleaner)"
    ),
    #: **[2026-08-10] The comparison CANNOT BE MADE, and that is the
    #: answer.** ``pretrained_checkpoint.sha256`` is the run's OUTPUT -- the
    #: trained ``pretrained.npz`` -- and no field records what any run
    #: started from. The decision tree had no readable branch because the
    #: quantity it turned on was never captured: the provenance hole
    #: demonstrated rather than argued. Converted into an experiment by
    #: PRETRAINED_INIT_IS_NOW_DECLARABLE: two ViT-512 re-runs from the
    #: DECLARED init -- agreement indicts the upstream artifact for the
    #: earlier divergence, disagreement indicts dynamic_img_size's
    #: per-forward resample and indicates the img_size routing fix.
    "comparison_outcome": (
        "UNARBITRABLE AS POSED -- the sha field is the output checkpoint, "
        "nothing recorded the init. Converted to the declared-init "
        "experiment: two ViT-512 re-runs from init_vit_b16_v1; agree -> "
        "upstream drift, routing innocent; differ -> dynamic per-forward "
        "resample, img_size routing fix indicated"
    ),
}


#: **[2026-08-10] The pretrained init is now a DECLARABLE input -- closing
#: the one input guard 3 never verified.**
#:
#: ``pretrained=True`` re-downloads ImageNet weights every run (HOME is
#: ephemeral, PLAN 2.7); nothing hashed what arrived, and nothing recorded
#: it. The hole was demonstrated, not argued: when two ViT-512 runs at one
#: SHA produced different checkpoints, the comparison that would have
#: separated upstream-drift from nondeterminism could not be made, because
#: the only sha in the records is the OUTPUT's. Chain 3's shape -- the
#: weights that produced a result are not recoverable from the config.
#:
#: **The fix**: ``snapshot_pretrained_init`` freezes each backbone's init
#: into ``data/inits/init_<backbone>_v1`` -- weights, upstream identity
#: (timm ``pretrained_cfg`` / torchvision weights enum), per-file hashes --
#: and pretraining configs declare it like any other input, by the
#: ``masked_scut`` conventional-name route (task blocks stay verbatim).
#: The loader accounts for every key: missing must be exactly the head's,
#: nothing unexpected, shapes exact -- a silent partial load would be the
#: undeclared input back in a quieter form. Runs WITHOUT the declaration
#: now say so in their own metrics ("init downloaded at run time -- an
#: undeclared input"): the absence names itself.
#:
#: **Scope, stated precisely**: consumption is wired for the transformer
#: path; the graph backbones build their pretrained bodies inside their own
#: assemblies and their loading route is a RECORDED FOLLOW-UP --
#: ``task_pretrain`` refuses a graph declaration until it exists, so the
#: artifact cannot be claimed unread. Their snapshots are built NOW anyway,
#: capturing the upstream bytes before they can drift again.
#:
#: **Road A's thirty pretraining runs share this hole.** Nothing suggests
#: they are wrong -- Swin reproducing to four decimals is evidence the
#: pipeline is generally stable -- but "these results came from these
#: weights" is not currently demonstrable for ANY of them, and the write-up
#: must not claim it is. The claim available instead: "these results came
#: from this config at this SHA, with the init downloaded at run time."
PRETRAINED_INIT_IS_NOW_DECLARABLE = {
    "closed": "2026-08-10",
    "hole": (
        "pretrained=True re-downloads every run; guard 3 never hashed the "
        "bytes and no field recorded them -- demonstrated by the "
        "unarbitrable ViT comparison, not argued"
    ),
    "fix": (
        "snapshot_pretrained_init freezes data/inits/init_<backbone>_v1 "
        "(weights + upstream identity + hashes); configs declare it by the "
        "conventional-name route; the loader refuses partial loads; "
        "undeclared runs name their own absence in metrics"
    ),
    "scope": (
        "transformer consumption wired; graph loading is a recorded "
        "follow-up and task_pretrain REFUSES a graph declaration until it "
        "exists; graph snapshots built now regardless"
    ),
    "road_a": (
        "the thirty pretraining runs share the hole. Nothing suggests they "
        "are wrong -- swin reproducing to four decimals is stability "
        "evidence -- but 'these results came from these weights' is not "
        "demonstrable for any of them, and the write-up must not claim it"
    ),
    "experiment": (
        "two ViT-512 re-runs from the declared init: agree -> the earlier "
        "divergence was upstream drift and the routing is innocent; differ "
        "-> dynamic_img_size's per-forward resample, img_size routing fix"
    ),
    #: **[MEASURED 2026-08-10] All four snapshots captured -- and the
    #: upstream URL asymmetry is the sharpest fact in the table.** ViT's
    #: pretrained_cfg url is EMPTY (hub-only resolution); swin, xception and
    #: resnet50 all pin concrete release-asset URLs. The hub-only artifact
    #: is the one most able to change without notice, and ViT is the only
    #: backbone that failed to reproduce.
    "snapshots": {
        "vit_b16": {
            "rollup_prefix": "c974c782", "n_tensors": 150,
            "upstream": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "url": "EMPTY -- hub-only",
        },
        "swin_b": {
            "rollup_prefix": "934fb980", "n_tensors": 327,
            "upstream": "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            "url": "SwinTransformer release asset",
        },
        "srgnn": {
            "rollup_prefix": "2c48719c", "n_tensors": 274,
            "upstream": "legacy_xception tf_in1k",
            "url": "rwightman v0.1-cadene asset",
        },
        "agnet": {
            "rollup_prefix": "b3fdbbfe", "n_tensors": 320,
            "upstream": "torchvision resnet50 IMAGENET1K_V2",
            "url": "download.pytorch.org/models/resnet50-11ad3fa6.pth",
        },
        "asymmetry": (
            "only ViT is hub-only, and only ViT failed to reproduce -- the "
            "leading explanation for the 224 divergence, testable because "
            "the init is now declared"
        ),
    },
    #: **The next runs, with their readings committed before launch.** One
    #: subtlety the comparison must carry: the snapshot holds TODAY'S hub
    #: bytes. Under the drift hypothesis the declared-init 224 run
    #: reproduces the RECENT 0.8120, not Road A's 0.7893 -- Road A's exact
    #: bytes are unrecoverable, which is the hole's permanent cost for Road
    #: A. Landing at 0.7893 instead (strongest form: output checkpoint sha
    #: equal to Road A's recorded pretrained_checkpoint.sha256) would mean
    #: NO drift, and promote ViT-224 nondeterminism-on-identical-hardware
    #: to the open question. A dated smoking gun exists independently: the
    #: HF hub repo has revision history, and a commit between Road A's
    #: Phase 6 (2026-07-31) and Road B's runs would time-stamp the drift --
    #: a read-only metadata check, and the reason future snapshots should
    #: pin a hub REVISION, not just a name.
    "next_runs": {
        "1_vit_224_declared": {
            "reproduces_0.8120": (
                "drift-consistent: recent runs used snapshot-equivalent "
                "bytes; check the hub revision log for the dated commit"
            ),
            "reproduces_0.7893": (
                "no drift -- and ViT-224 nondeterminism on identical "
                "hardware becomes the open question"
            ),
            "third_value": "ViT-224 nondeterministic even from pinned bytes",
        },
        "2_vit_512_pair": (
            "same GPU MODEL both runs -- the cross-device lesson; agree -> "
            "routing innocent; differ -> per-forward resample, img_size fix"
        ),
        "3_relaunch_768": "whole device; nothing scientific in the OOMs",
    },
    #: **[MEASURED 2026-08-10, the first snapshot] The ViT init, captured --
    #: and its cfg carries the upstream's own statement of the constraint.**
    #: ``timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`` via timm 1.0.7,
    #: 150 tensors, 343,237,089 bytes, rollup c974c782... -- and
    #: ``fixed_input_size: True``: timm declaring this checkpoint EXPECTS
    #: 224. That is the flag ``dynamic_img_size`` overrides, and it is now
    #: in the artifact rather than implied. If the ViT-512 pair disagrees,
    #: this flag is the upstream's own statement of why.
    "vit_snapshot": {
        "upstream": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "timm": "1.0.7",
        "n_tensors": 150,
        "total_bytes": 343_237_089,
        "rollup_prefix": "c974c782",
        "fixed_input_size": (
            "True -- the upstream's own declaration that the checkpoint "
            "expects 224; the constraint dynamic_img_size overrides. If the "
            "512 pair disagrees, this is the upstream's statement of why"
        ),
    },
    #: **[MEASURED 2026-08-10, env.json] The 512 pair DISSOLVES: it was
    #: cross-device.** One run on an A40 (0.7412), one on an RTX PRO 6000
    #: Blackwell (0.7680) -- same image digest, same kernel, same base. Two
    #: runs on different hardware are NOT a determinism test at all: the
    #: decision tree lacked this branch, and any future determinism
    #: comparison must pin the GPU MODEL (env.json records it) or it cannot
    #: distinguish routing from device.
    "resolved_512": {
        "measured": "2026-08-10, env.json of both runs",
        "devices": {"0.7412": "NVIDIA A40", "0.7680": "RTX PRO 6000 Blackwell"},
        "held_constant": "image digest, kernel, base -- only the device differs",
        "verdict": (
            "the 0.027 split is A40 vs Blackwell, not upstream drift and "
            "not dynamic_img_size. The pair is not evidence about either"
        ),
        "third_branch": (
            "cross-device runs are not a determinism test; determinism "
            "comparisons pin the GPU model, recorded in env.json"
        ),
    },
    #: **[MEASURED 2026-08-10, the env.json sweep] Every training keeper run
    #: in the project is Blackwell.** The only non-Blackwell entry is
    #: p7-paired-ladder on an A40 -- bootstrap intervals, trains nothing. So
    #: Road A's thirty pretraining runs, every Phase 7 arm and all of 7B/7C
    #: are single-device, and the hardware-mixing worry CLOSES for Road A.
    "road_a_single_device": (
        "env.json sweep across all keeper runs: training is Blackwell-only; "
        "the sole A40 entry (p7-paired-ladder) computes intervals and "
        "trains nothing. Road A is not hardware-mixed"
    ),
    #: **What remains after the dissolution: the 224 divergence, alone.**
    #: 0.8120 against Road A's 0.7893 with hardware, artifact (masked_v1 at
    #: Road A's own hash), procedure (task dicts byte-equal) and build path
    #: (timm_kwargs_for(224,224) == {}) ALL held constant -- and swin
    #: reproducing to four decimals on the identical comparison. The
    #: remaining candidate is the init, and the snapshot metadata sharpened
    #: it: **ViT is HUB-ONLY (its pretrained_cfg url is empty) while the
    #: other three backbones pin concrete release-asset URLs** -- and ViT is
    #: the only backbone that failed to reproduce. A hub-resolved artifact
    #: under a stable identifier is precisely the one most able to change
    #: without notice. Testable now, because the init is declared.
    "remaining_224": {
        "delta": 0.023,
        "held_constant": (
            "hardware (Blackwell both), masked_v1 at Road A's hash, task "
            "dict byte-equal, build path identical -- and swin reproduces "
            "to four decimals on the same comparison"
        ),
        "leading_explanation": (
            "hub drift: ViT's upstream is HUB-ONLY (empty url) where the "
            "other three pin release assets, and ViT alone failed to "
            "reproduce"
        ),
        "flag_does_not_predict": (
            "both transformers declare fixed_input_size: True and swin "
            "reproduces exactly -- so the flag is not the discriminator; "
            "the live contrast is WHEN adaptation happens: swin's img_size "
            "at build vs vit's dynamic_img_size per forward"
        ),
    },
    #: The four 768 cells: OutOfMemoryError on a 27.94 GiB FRACTION after
    #: six resume attempts, header-only curves, no epochs -- nothing
    #: scientific, relaunch with a whole device (the config headers now
    #: carry the measured fact).
    "768_cells": "OOM on a GPU fraction; not diagnostic; relaunch whole-device",
}


#: **[MEASURED 2026-08-10] ViT-B/16 pretraining is NONDETERMINISTIC on the
#: pinned stack, from pinned weights, on identical hardware. The
#: pre-registered third reading fired, and this is the finding.**
#:
#:     Road A            pcc 0.7893   sel 23   ckpt 9b6d57dc...
#:     Road B first      pcc 0.8120   sel 11   ckpt e2d3dce8...
#:     Road B declared   pcc 0.8194   sel 14   ckpt 1f0015c8...
#:
#: Three runs, three checkpoints, all Blackwell; the third started from
#: hashed, DECLARED bytes and reproduced neither predecessor -- the init is
#: eliminated by measurement, the hardware by the device records, the
#: procedure by dict-equality. **The selected epochs (23, 11, 14) say the
#: trajectories diverge enough to pick materially different best
#: checkpoints** -- not small numerical drift around a common optimum. The
#: spread is 0.030 across three runs: larger than the 0.023 that started
#: the hunt, and larger than most effects this project has measured. Swin
#: does not do this -- same procedure, artifact and hardware, reproduced
#: Road A to four decimals.
#:
#: **The mechanism candidate, measured to the extent a laptop can:**
#:
#: * INSPECTION (timm): ViT's ``Attention`` has ``fused_attn=True`` -- it
#:   routes through ``F.scaled_dot_product_attention``. Swin's
#:   ``WindowAttention`` has ``fused_attn=False`` -- manual matmul+softmax
#:   (the relative-position-bias path). **Exactly the op ViT's training
#:   graph invokes that Swin's does not.**
#: * CONTROLLED CONTRAST (local CUDA, across-instance gradients): default
#:   backend 2.6e-06, memory-efficient forced 2.4e-06, **MATH forced 0.0
#:   bitwise** -- equal to swin's 0.0. The divergence switches off exactly
#:   when the fused kernel leaves the graph, everything else held.
#: * And the ``deterministic: false`` question answers MEASURED-YES: the
#:   fused backward is atomics-class, and the flag that would force a
#:   deterministic fallback (or a self-naming error) is never set -- an op
#:   that would otherwise raise runs silently non-reproducible.
#:
#: Local-stack caveat as always: the pinned image is the only environment
#: whose determinism claims transfer to the image -- which is what the
#: probe config below settles. The earlier "local probe cannot arbitrate"
#: is superseded: it could, once the right knob existed to turn.
#:
#: **The decisive pinned experiment** (``roadb_p6_det_probe_vit_224``,
#: shipped): ViT-224 from the declared init with ``deterministic: true`` --
#: through the frozen ``determinism.configure``, exactly the route the
#: config comment reserved. Two readable surfaces, committed before the
#: run: it RAISES, and the op names itself in the error; or it RUNS, and
#: two such runs must be bitwise-equal, confirming the op by elimination.
#:
#: **Fix options, flagged not chosen** -- both change ViT's procedure
#: relative to Road A, and nothing can preserve comparability with a
#: nondeterministic past: (a) build ViT with ``fused_attn`` off for
#: training (task dicts stay byte-equal; the math path measured bitwise);
#: (b) ``deterministic: true`` as policy (breaks the task-dict equality the
#: lattice asserts, and AG-Net's roi_align cannot honour it -- the
#: recorded reason the field is false). a maintainer decision, with the probe's
#: data.
VIT_PRETRAINING_IS_NONDETERMINISTIC = {
    "measured": "2026-08-10, the pre-registered third reading",
    "runs": {
        "road_a": {"pcc": 0.7893, "selected_epoch": 23, "ckpt": "9b6d57dc"},
        "road_b_first": {"pcc": 0.8120, "selected_epoch": 11, "ckpt": "e2d3dce8"},
        "road_b_declared": {"pcc": 0.8194, "selected_epoch": 14, "ckpt": "1f0015c8"},
    },
    "spread": 0.030,
    "constraints_discharged": (
        "not the init (declared bytes reproduced neither predecessor), not "
        "the hardware (all three Blackwell), not the procedure "
        "(dict-equality)"
    ),
    "trajectory_level": (
        "selected epochs 23/11/14 -- materially different best checkpoints, "
        "not numerical drift around one optimum"
    ),
    "swin_contrast": "same everything, reproduces Road A to four decimals",
    "mechanism": {
        "inspection": (
            "vit Attention fused_attn=True (F.scaled_dot_product_attention); "
            "swin WindowAttention fused_attn=False (manual matmul+softmax)"
        ),
        "controlled_contrast": (
            "across-instance grads, local CUDA: default 2.6e-06, "
            "mem-efficient forced 2.4e-06, MATH forced 0.0 bitwise == "
            "swin's 0.0 -- the divergence switches off when the fused "
            "kernel leaves the graph"
        ),
        "deterministic_false_answer": (
            "MEASURED YES: the fused backward is atomics-class and the flag "
            "that would force fallback or a self-naming error is never set "
            "-- silently non-reproducible"
        ),
    },
    "probe": {
        "config": "roadb_p6_det_probe_vit_224",
        "raises": "the op names itself in the deterministic-mode error",
        "runs": "two probe runs must be bitwise-equal; op confirmed by elimination",
    },
    "fix_options_flagged": (
        "(a) fused_attn off for ViT training builds -- task dicts stay "
        "byte-equal, math path measured bitwise; (b) deterministic: true as "
        "policy -- breaks lattice dict-equality and AG-Net cannot honour "
        "it. Either changes ViT's procedure relative to Road A; nothing "
        "preserves comparability with a nondeterministic past"
    ),
    #: **[MEASURED 2026-08-10] The probe was CONCLUSIVE -- the second
    #: surface fired.** Both runs bitwise identical: pcc 0.835346, selected
    #: epoch 11, checkpoint 58fe1c5647741b4b..., both Blackwell. It RAN
    #: rather than raising, so the fused kernel has a deterministic
    #: fallback available -- it simply is not the default. The fused
    #: attention path is convicted by elimination.
    "probe_outcome": {
        "surface": "ran; bitwise pair",
        "pcc": 0.835346,
        "selected_epoch": 11,
        "ckpt_prefix": "58fe1c5647741b4b",
        "fallback": (
            "a deterministic implementation exists and is not the default"
        ),
    },
    #: **[OBSERVED, n=3 -- recorded, not concluded] 0.8353 is a FOURTH
    #: distinct value, ABOVE the range of all three nondeterministic draws
    #: (0.7893-0.8194).** Under exchangeability a fourth draw ranks first
    #: with probability 1/4, so chance-compatible -- but if the
    #: deterministic path is systematically different rather than merely
    #: reproducible, that is a separate fact about the two paths. The
    #: unfused re-runs add data; note they are a THIRD rounding path (see
    #: numerics below), so 0.8353 is NOT their predicted value.
    "fourth_value_observation": (
        "0.8353 sits above all three fused draws; P(rank first) = 1/4 at "
        "n=3 -- chance-compatible, held open. The unfused re-runs are a "
        "different rounding path and land at their own value"
    ),
    #: **[MEASURED 2026-08-10, local CUDA] The numerics question: SAME
    #: computation, different floating-point evaluation order.** Identical
    #: seeded weights, identical input, eval mode, forward max |diff|
    #: between the three attention evaluation paths now in play:
    #:
    #:     fused-default vs fused-MATH (the probe's fallback)  7.7e-07
    #:     fused-MATH    vs unfused    (the fix)               3.3e-07
    #:     fused-default vs unfused                            9.5e-07
    #:
    #: So the unfused result SHOULD be expected to differ from the fused
    #: draws -- not because the math differs, but because in this regime a
    #: ~1e-06 per-step perturbation is exactly the scale that selects a
    #: different optimum over 30 epochs (the three fused draws differed
    #: among themselves by the same mechanism). A distinct rounding path is
    #: a distinct draw from the same training landscape; the unfused path's
    #: distinction is that its draw is REPRODUCIBLE. Only reproducibility
    #: transfers from the probe -- not the 0.8353.
    "numerics": {
        "verdict": "same computation, different fp evaluation order",
        "forward_max_diffs": {
            "fused_default_vs_fused_math": 7.7e-07,
            "fused_math_vs_unfused": 3.3e-07,
            "fused_default_vs_unfused": 9.5e-07,
        },
        "consequence": (
            "the unfused endpoint is its own value -- expected to differ "
            "from all fused draws AND from the probe's 0.8353; what "
            "transfers is reproducibility, not the number"
        ),
    },
    #: **[DECIDED 2026-08-10, the maintainer] fused_attn OFF for Road B's ViT
    #: pretraining builds** -- not deterministic:true as policy. Task dicts
    #: stay byte-equal to Road A's; the unfused path is measured bitwise
    #: locally and the deterministic route confirmed on the pinned image;
    #: no conflict with AG-Net's roi_align. Implemented in
    #: ``TorchPretrainModel.reset``: the toggle wraps the BUILD only (a
    #: construction-time flag), keyed to vit_b16, with a loud post-build
    #: refusal if it did not take -- extraction stays fused, preserving
    #: byte-compatibility with Road A's embeddings, and swin/graphs are
    #: untouched (verified through the real reset). **ViT-224 and ViT-512
    #: then re-run from the unfused path on the SAME configs** -- the fix
    #: is code-level, so the runs differ only by SHA -- because the
    #: existing results came from the fused path, and the alternative is
    #: Road B's ViT arms carrying a 0.030 spread that swin and the graphs
    #: do not.
    "decision": {
        "superseded": (
            "OVERTURNED 2026-08-11 -- the unfused pair diverged 0.035; the "
            "probe convicted the class, not the member. See "
            "UNFUSED_WAS_NOT_SUFFICIENT; this record stands as what was "
            "believed when it was written"
        ),
        "chosen": "fused_attn off for ViT pretraining builds",
        "why": (
            "task dicts stay byte-equal; the path is measured bitwise; no "
            "AG-Net conflict"
        ),
        "implementation": (
            "build-only toggle in reset, keyed to vit_b16, loud post-build "
            "refusal; extraction stays fused (Road A embedding "
            "byte-compatibility); swin and graphs untouched, verified "
            "through the real reset"
        ),
        "reruns": (
            "vit-224 and vit-512 on the SAME configs at the new SHA; the "
            "five running cells (four 768s, swin-512) are undisturbed -- "
            "the change touches only ViT's construction"
        ),
    },
}


#: **[MEASURED 2026-08-11, ten of twelve cells] The Phase 6 resolution
#: results: FLAT for the three reproducible backbones, clouds for ViT.**
#:
#:                 224       512       768
#:     swin_b     0.8719    0.8662    running
#:     srgnn      0.8412    0.8474    0.8427
#:     agnet      0.8157    0.8098    0.8074
#:     vit_b16    4 draws   3 draws   0.7340 (+1 running)
#:
#: **The trend is flat where it is readable.** Swin -0.006 at 512; AG-Net
#: -0.008 across the full 3.4x; SR-GNN non-monotone with a mid-peak at 512
#: (0.8412 -> 0.8474 -> 0.8427). All spreads sit inside a seed band, so
#: **SR-GNN's peak is recorded as a SHAPE, not a claim** -- though the
#: mechanism fits: its fixed 42x42 map is interpolated 6.0x at 224, 2.62x
#: at 512, 1.75x at 768, while SCUT sources cap at 350px, so 512 is roughly
#: where less-invention-in-the-map crosses more-invention-in-the-input.
PHASE_6_RESOLUTION_RESULTS = {
    "measured": "2026-08-11, ten of twelve; swin-768 and a vit-768 running",
    "table": {
        "swin_b": {224: 0.8719, 512: 0.8662, 768: "running"},
        "srgnn": {224: 0.8412, 512: 0.8474, 768: 0.8427},
        "agnet": {224: 0.8157, 512: 0.8098, 768: 0.8074},
        "vit_b16": {
            224: (0.8120, 0.8194, 0.8001, 0.7648),
            512: (0.7680, 0.7412, 0.7529),
            768: (0.7340,),
        },
    },
    "road_a_masked_g1": {
        "vit_b16": 0.7893, "swin_b": 0.8718, "srgnn": 0.8433, "agnet": 0.8244,
    },
    "trend": (
        "FLAT where readable: swin -0.006 at 512, agnet -0.008 across "
        "3.4x; all inside a seed band"
    ),
    "srgnn_shape_not_claim": (
        "non-monotone mid-peak at 512, inside the band -- a shape. The "
        "mechanism fits: map interpolation 6.0x/2.62x/1.75x against "
        "350px-capped input, crossing near 512"
    ),
}


#: **[OVERTURNED 2026-08-11] ``fused_attn=False`` was NOT sufficient, and
#: the decision it justified is withdrawn -- with the inference errors
#: named, because they were the record-writer's own.**
#:
#: Two UNFUSED ViT-224 runs, same declared init, same Blackwell hardware,
#: no resumes: **0.8001 and 0.7648** (checkpoints b89bebcc.../80a78469...)
#: -- a 0.035 spread, LARGER than the fused path's. Across all four 224
#: draws the spread is 0.055.
#:
#: **The two errors, precisely:**
#:
#: 1. **The probe convicted the CLASS, not the member.**
#:    ``use_deterministic_algorithms(True)`` forces fallbacks for EVERY
#:    nondeterministic op, so a bitwise pair under the flag was evidence
#:    the flag covers whatever the causes are -- not that fused attention
#:    was the sole cause. "Convicted by elimination" eliminated nothing
#:    but the flag's complement.
#: 2. **One forward-backward does not bound thirty epochs.** The local
#:    contrast (divergence off under MATH at a single step) showed the
#:    fused op was A source, not THE source; other atomics-class ops with
#:    per-step effects below that probe's floor accumulate over a
#:    trajectory.
#:
#: **Decision (the maintainer): stop chasing it.** ViT's Road B cells are
#: recorded as draws with a measured ~0.055 spread at 224 -- the treatment
#: Road A's ViT arms already carry. ``deterministic: true`` is the only
#: demonstrated route to a reproducible ViT and is NOT being taken.
#:
#: **The cost, stated plainly: Road B's ViT resolution trend is not
#: readable as a trend.** The 224 draws span 0.765-0.819 and the 512 draws
#: 0.741-0.768 -- a decline is visible, but the decline and the noise are
#: the same size. The other three backbones give trends that are both flat
#: and readable; ViT gives three clouds.
#:
#: **The unfused toggle STAYS in the build path**: it removes one measured
#: source (and its runs are among the recorded draws); reverting would
#: churn ViT's procedure a third time for no benefit. Its comment no
#: longer claims sufficiency.
UNFUSED_WAS_NOT_SUFFICIENT = {
    "overturned": "2026-08-11",
    "unfused_pair": {
        "pccs": (0.8001, 0.7648),
        "ckpts": ("b89bebcc", "80a78469"),
        "spread": 0.035,
        "note": "larger than the fused path's; all-four-224 spread 0.055",
    },
    "inference_errors": {
        "class_not_member": (
            "the deterministic flag forces fallbacks for EVERY "
            "nondeterministic op; a bitwise pair under it convicts the "
            "class, not fused attention alone"
        ),
        "one_step_no_bound": (
            "a single forward-backward contrast does not bound thirty "
            "epochs of accumulation; sub-floor per-step sources compound"
        ),
    },
    "decision": (
        "stop chasing. ViT Road B cells are draws, ~0.055 spread at 224 -- "
        "Road A's treatment. deterministic:true is the only demonstrated "
        "route and is not taken"
    ),
    "cost": (
        "ViT's Road B resolution trend is not readable as a trend: 224 "
        "spans 0.765-0.819, 512 spans 0.741-0.768 -- the decline and the "
        "noise are the same size. Three backbones give readable flat "
        "trends; ViT gives three clouds"
    ),
    "toggle_stays": (
        "one measured source removed, runs among the recorded draws; "
        "reverting would churn the procedure a third time"
    ),
}


#: **[DESIGNED 2026-08-11] Road B Phase 7's resolution pairing: MATCHED
#: PIPELINES ARE THE AXIS -- three cells per backbone, not nine.**
#:
#: The question: a Road B arm has two resolutions -- the one the backbone
#: pretrained at, and the one the cleft images are staged at -- and they
#: CAN differ mechanically (measured: swin's parameters are size-invariant;
#: vit's dynamic build keeps its 14x14 positional grid and interpolates to
#: whatever arrives; the graphs are size-agnostic). Whether they MAY differ
#: is design, and the answer is no, for three reasons:
#:
#: 1. **The Stage C lesson, which is this exact trap already recorded.**
#:    Road A's Stage C mixed a pipeline axis into a geometry comparison and
#:    "therefore measures a matched PIPELINE, not geometry" -- the arm
#:    could not attribute its own delta. A 512-pretrained backbone on
#:    768-staged cleft measures neither resolution's effect but their
#:    interaction; with per-seed BCa intervals ~0.28 wide at n=237, an
#:    interaction grid is unresolvable BY CONSTRUCTION (29-of-30), so the
#:    mismatched cells would produce numbers nothing can read.
#: 2. **The trend claim needs one variable per step.** Road B's headline is
#:    the three-point resolution trend (brief §5); matched cells move the
#:    WHOLE pipeline one step at a time. Mismatch adds a second axis and
#:    turns three points into a 3x3 grid of pairwise deltas -- exactly the
#:    shape the paired audit spent three phases withdrawing.
#: 3. **The readable backbones make the mismatch question small.** ViT --
#:    where interpolation-transfer would bite hardest -- is three clouds
#:    (UNFUSED_WAS_NOT_SUFFICIENT) and cannot read it; the backbones that
#:    could read it (swin, the graphs) just measured FLAT across the whole
#:    axis, so the expected mismatch effect sits inside the same bands.
#:
#: **The interesting cell stays available as ONE pre-registered arm, not a
#: lattice.** The 512-pretrained -> 768-staged cell asks a real question --
#: positional geometry learned on interpolated 350px faces applied to
#: genuinely detailed 2758px-median ones -- and if it earns a run it is
#: added the arms-one-at-a-time way, on a READABLE backbone (swin or
#: srgnn), with its question and reading registered first. Cost is not the
#: argument (brief §1 forbids that); attribution is.
#:
#: Arm count before extraction, therefore: THREE resolution cells per
#: backbone, matched end to end.
PHASE_7_RESOLUTION_PAIRING = {
    "designed": "2026-08-11, before any extraction was planned",
    "answer": "matched pipelines -- three cells per backbone, not nine",
    "mechanically_possible_to_differ": (
        "yes, measured: swin parameters size-invariant; vit dynamic keeps "
        "its 14x14 grid; graphs size-agnostic. The restriction is design, "
        "not mechanics"
    ),
    "reasons": {
        "stage_c_lesson": (
            "a mismatched cell measures a matched-pipeline interaction, "
            "not resolution -- the recorded Stage C trap; unresolvable by "
            "construction at n=237"
        ),
        "one_variable_per_step": (
            "the headline is the three-point trend; mismatch turns it into "
            "a 3x3 grid of pairwise deltas, the withdrawn-claim shape"
        ),
        "readability": (
            "ViT cannot read the mismatch question (three clouds); the "
            "backbones that could just measured flat, so the expected "
            "effect sits inside the bands"
        ),
    },
    "the_interesting_cell": (
        "512-pretrained -> 768-staged: interpolated-face positional "
        "geometry applied to genuinely detailed faces. Stays available as "
        "ONE pre-registered arm on a readable backbone (swin or srgnn), "
        "added arms-one-at-a-time AFTER the matched cells read -- not as a "
        "lattice. Attribution, not cost, is the argument"
    ),
}


#: **[DERIVED 2026-08-11] Phase 7's extraction set list, from the ARM LIST
#: -- Road A's ``embedding_plan`` pattern, not the cross product.**
#:
#: The arm list after PHASE_7_RESOLUTION_PAIRING: matched pipelines, square
#: G1, three resolutions, two inits -- (backbone, resolution, init) with
#: init in {imagenet, masked_g1-at-that-resolution}. **24 sets consumed;
#: 23 extractions; 1 reuse.**
#:
#: **What shares, precisely:**
#:
#: * **An imagenet set is ONE set per (backbone, resolution)** -- no
#:   checkpoint, so nothing on the masked axis multiplies it; every arm at
#:   that cell reads the same set. 12 imagenet sets total.
#: * **swin-224-masked REUSES Road A's set** -- the Road B checkpoint is
#:   bitwise-identical to Road A's (measured) and the extraction path is
#:   unchanged (fused, frozen), so re-extracting would produce identical
#:   bytes under a new hash. The strongest reuse there is: identity, not
#:   assumption. The other three 224-masked cells extract fresh -- their
#:   Road B checkpoints differ from Road A's (vit: draws; srgnn -0.002;
#:   agnet -0.009).
#: * **Storage differs by kind, not by count**: the graph backbones store
#:   FEATURE MAPS (regions applied at cleft-train time --
#:   ``embedding_plan``'s per-checkpoint rule carries over), the
#:   transformers store POOLED VECTORS.
#:
#: **Two rules the derivation forces, registered before any extraction:**
#:
#: 1. **The ViT draw-pick rule.** Every ViT cell is a cloud, so each masked
#:    set must name which draw's checkpoint it extracts from -- and picking
#:    by PCC would be selection on the outcome. The rule: **the first
#:    completed Blackwell run at the current (unfused) SHA**, per cell, by
#:    launch order. The chosen run ids are read off the run records, never
#:    off the scores.
#: 2. **Init-vintage consistency.** The resolution trend must hold the init
#:    constant across its three points, and the only init whose bytes are
#:    guaranteed constant is the declared snapshot -- Road A's imagenet
#:    extractions downloaded at run time and recorded nothing (the hole).
#:    So ALL 12 imagenet sets extract from ``init_<backbone>_v1``,
#:    including 224 -- Road A's 224-imagenet sets are NOT reused, because
#:    their weight vintage is unverifiable against the snapshots.
#:    **Pre-extraction gate: the extract task does not yet consume a
#:    declared init** -- the transformer route ports from
#:    ``pretrain.load_declared_init``; the graph route is the recorded
#:    follow-up (their bodies build inside their assemblies).
PHASE_7_EXTRACTION_PLAN = {
    "derived": "2026-08-11, from the arm list (embedding_plan's pattern)",
    "sets_consumed": 24,
    "extractions": 23,
    "reuses": 1,
    "breakdown": {
        "imagenet": "12 sets, one per (backbone, resolution), no checkpoints",
        "masked_g1": (
            "12 sets, one per cell from that cell's own checkpoint; 11 "
            "extracted, swin-224 reused from Road A by bitwise identity"
        ),
    },
    "storage": (
        "graphs: feature maps (regions applied at cleft-train time); "
        "transformers: pooled vectors"
    ),
    "vit_draw_pick_rule": (
        "first completed Blackwell run at the current unfused SHA, by "
        "launch order -- never by score; run ids read off the run records"
    ),
    "init_vintage_rule": (
        "all 12 imagenet sets extract from the declared snapshots, 224 "
        "included -- Road A's runtime-download vintage is unverifiable. "
        "GATE: the extract task must first consume pretrained_init "
        "(transformer route ports from load_declared_init; graph route is "
        "the recorded follow-up)"
    ),
    #: **[BUILT 2026-08-11] The extract-task init gate -- and building it
    #: found a SECOND gate-2-class hole in the same signature.**
    #: ``extract._load_model`` built via ``create_backbone`` WITHOUT
    #: ``input_size``: every 512/768 extraction -- all 16 of the 23 at the
    #: new resolutions -- would have died at the ViT/Swin 224 asserts after
    #: submission. Both fixed together: ``input_size`` threads from the
    #: staged images' own shape (the pretrain builder's pattern), and
    #: ``init_dir`` consumes the declared snapshot with
    #: ``load_declared_init``'s strict accounting. Refusals at the failure
    #: site: graphs (the standing follow-up, task_pretrain's rule),
    #: non-imagenet sets (declared weights the set never reads), and a
    #: task-level guard for a declaration no set in the config consumes.
    #: Undeclared imagenet extractions still run and their reports name the
    #: absence -- the pretrain convention.
    "gate_built": (
        "extract._load_model consumes init_dir with strict accounting and "
        "threads input_size -- the second gate-2 hole, found before it "
        "cost 16 of the 23 extractions. Unread declarations refused at "
        "task level; undeclared downloads name themselves in the set report"
    ),
    #: **[MEASURED 2026-08-11] The graph EXTRACTION route consumes declared
    #: inits after all -- the vintage rule holds for ALL 12 imagenet sets.**
    #: The body mappings are deterministic and were measured, then verified
    #: by strict synthetic loads: srgnn's features_only body is
    #: ``body.`` + snapshot key (274 <-> 274 exactly); agnet's Sequential
    #: body is the resnet50 child order (conv1->0, bn1->1, layer1..4->4..7;
    #: 318 = 320 minus the fc pair, which the loader asserts is the ONLY
    #: leftover). ``extract._load_graph_imagenet_init``. The TRAINING route
    #: stays the recorded follow-up -- task_pretrain still refuses.
    "graph_init_route": (
        "built for extraction: measured key mappings, strict both ways, "
        "verified by synthetic loads. Training-side consumption remains "
        "the follow-up"
    ),
    #: **[DECIDED 2026-08-11] PER-SET ARTIFACTS -- 23 configs, not one.**
    #: Road A's single-config 17-set artifact was right THERE because the
    #: ladder consumed it en bloc: one declaration, one rollup. Road B's
    #: Phase 7 consumes SELECTIVELY across resolutions -- the same fact
    #: that split the ten staging artifacts (ARTIFACT_PER_SETTING): guard 3
    #: verifies the ROLLUP, so one artifact holding 23 sets means any
    #: partial re-extraction stales every declaration in every consuming
    #: config, including arms whose own set never changed. Per-set
    #: artifacts (``roadb_emb_<backbone>_<init>_g1_<resolution>_v1``) make
    #: a partial re-extraction move exactly one hash -- and make the ViT
    #: double-extraction a clean one-set comparison. Set NAMES repeat
    #: across resolutions (``set_name`` has no resolution field); the
    #: artifact version carries it, recorded here so nobody reads two
    #: same-named sets as one.
    "grouping": (
        "per-set artifacts, 23 configs: Phase 7 consumes selectively, and "
        "one artifact would stale every declaration on any partial "
        "re-extraction -- ARTIFACT_PER_SETTING's argument, verbatim. Road "
        "A's en-bloc artifact matched its en-bloc consumption. The "
        "resolution lives in the artifact VERSION, not the set name"
    ),
    #: **[MEASURED 2026-08-11, local CUDA] ViT extraction is forward-only
    #: and measured BITWISE** -- same-instance repeat and fresh-instance
    #: rebuild both 0.0 at 224 static and 512 dynamic, eval/no_grad. The
    #: training nondeterminism lives in the fused BACKWARD (atomics);
    #: the forward has none. Local-stack caveat as always, so the pinned
    #: confirmation is PRE-REGISTERED: **the first ViT extraction runs
    #: TWICE and the two artifacts' payload rollups must be byte-equal
    #: before the batch proceeds.** If they differ, every ViT set is a draw
    #: on top of the checkpoint clouds, and the plan is re-read before any
    #: arm consumes one.
    "extraction_determinism": {
        "local": (
            "bitwise at 224 static and 512 dynamic, same-instance and "
            "fresh-instance, eval/no_grad -- the nondeterminism lives in "
            "the fused backward, and extraction has no backward"
        ),
        "pinned_confirmation": (
            "PRE-REGISTERED: the first ViT extraction runs twice; payload "
            "rollups byte-equal or the batch stops and the plan is re-read"
        ),
        #: **Both branches registered, one decision -- "the confirmation
        #: passed" and "the batch is cleared" are the same recorded fact,
        #: not a queue inference.**
        "on_pass": (
            "the remaining 22 extractions PROCEED with no further per-set "
            "double runs -- one confirmation clears the batch. The FIRST "
            "run's artifact is the set of record (launch order, the "
            "draw-pick convention); the check run writes out_version "
            "<set>_detcheck_v1, which no config ever declares, and is "
            "deleted after the byte comparison -- immutability protects "
            "artifacts results cite, and nothing cites a check copy"
        ),
        "on_fail": (
            "the batch STOPS; every ViT set would be a draw on top of the "
            "checkpoint clouds, and the plan is re-read before any arm "
            "consumes one"
        ),
        #: **[CONFIRMED 2026-08-11, the pinned image] The double-extraction
        #: is BYTE-IDENTICAL** -- values.npy md5 916b430c..., metadata.json
        #: md5 fc5311bc..., matching across both artifacts. The on_pass
        #: branch fired as registered: batch cleared, no further per-set
        #: double runs, first artifact is the set of record, check copy
        #: deleted. **The confinement, stated explicitly: the ViT
        #: nondeterminism is a TRAINING fact and does not touch extraction.**
        #: Phase 7's ViT arms inherit the checkpoint clouds and nothing
        #: else -- one layer of draws, not two.
        "confirmed": {
            "date": "2026-08-11",
            "values_md5": "916b430c57232cf35178962f667c4117",
            "metadata_md5": "fc5311bccbef90ca49458427a35b503a",
            "branch": "on_pass fired as registered; check copy deleted",
            "confinement": (
                "nondeterminism is confined to training; extraction is "
                "deterministic on the pinned image. ViT arms inherit the "
                "checkpoint clouds and nothing else -- one layer of "
                "draws, not two"
            ),
        },
    },
    #: **[MEASURED 2026-08-11] The twelve imagenet sets, built and
    #: verified**: 237 patients each; pooled vectors at 768 (ViT) and 1024
    #: (Swin); feature maps at 2048 for both graph backbones -- the
    #: dimensions the registry records for each backbone.
    "imagenet_sets_built": (
        "all twelve verified: 237 patients; pooled 768/1024 for the "
        "transformers; feature_map 2048 for the graphs"
    ),
}


#: **[DESIGNED 2026-08-12] Road B Phase 7's arm structure -- 24 arms, 8
#: trend families, 12 descriptive init contrasts. The trend is the result;
#: everything else is arranged around that.**
#:
#: **The arm list, stated rather than implied.** One arm per embedding set:
#: a seeded frozen-head fit (5-fold CV over the Phase 1 folds) on each
#: (backbone, resolution, init) cell -- 24 arms. They organise into:
#:
#: * **8 trend families** -- (backbone x init), three resolutions each.
#:   Within a family the WHOLE pipeline moves one resolution step at a
#:   time (matched, PHASE_7_RESOLUTION_PAIRING), so the family isolates
#:   resolution and nothing else. **The family's result is its SHAPE**
#:   (monotone up / monotone down / flat / peaked), pre-registered as the
#:   reportable claim per brief §5 -- a trend is reportable where a
#:   pairwise delta is not. Per-step deltas carry §4.3's two conditions
#:   and are co-reported, but the shape is the finding.
#: * **12 init contrasts** -- masked vs imagenet at each cell, isolating
#:   the value of SCUT pretraining at that resolution. DESCRIPTIVE by
#:   default: 29-of-30 says pairwise deltas rarely resolve, and these are
#:   pairwise. They exist to explain families, not to headline.
#: * **A built-in anchor, not an arm**: the three 224-imagenet arms
#:   re-measure Road A's G1 imagenet cells from re-extracted declared-init
#:   sets -- landing near Road A's values (ViT G1 0.2521) is the
#:   cross-check that the re-extraction changed provenance, not physics.
#:
#: **Seed counts and the band question -- the per-cell rule does not apply
#: as written, and here is precisely why.** The rule "each pretraining
#: cell owes its own band" governs reading against a CELL'S OWN numbers --
#: Phase 6 comparisons, where the trained backbone is the third regime. No
#: Phase 7 arm reads against a Phase 6 PCC. Phase 7 arms are FROZEN-HEAD
#: fits over precomputed embeddings -- the exact regime Road B Phase 3
#: banded, and banded ACROSS RESOLUTIONS: one band held at 224/512/768
#: (SEED_BAND_RESOLVED, 0.021425). So:
#:
#: * **transformer-head arms**: the resolved band IS the planning number.
#:   Zero new bands. Five seeds resolve 0.027; ten resolve 0.019.
#: * **graph-head arms are a different head regime** (trainable GNN
#:   machinery, BatchNorm -- the recorded third-regime note in
#:   factory.BACKBONES), so Road B needs ONE confirmation sweep, not
#:   twelve: ten seeds of the srgnn masked-512 arm, F-compared against
#:   Road A's measured graph band. The mechanism argument says it will
#:   hold -- the ROI machinery normalises every resolution to the same
#:   42x42 map (gate 3), so the graph head's inputs are
#:   resolution-invariant in shape -- but that is an argument, and the
#:   sweep is the measurement. Inside the F-interval: one graph planning
#:   band. Outside: per-resolution graph bands, and the design re-reads.
#: * **§4.12.1 stands untouched either way**: every arm reports its OWN
#:   seed SD, and every delta combines the two arms' own SDs. Bands plan;
#:   they never denominate.
#:
#: **ViT is reported DESCRIPTIVELY -- decided now, not discovered at the
#: write-up.** Its cells are checkpoint draws (0.055 spread at 224, 0.027
#: at 512), and the precision matters: **an arm's seed SD measures
#: head-fit variance GIVEN the drawn checkpoint -- the checkpoint-draw
#: spread rides on top, and the arm cannot see it.** So ViT's two families
#: make NO trend claim; each ViT number is reported with the draw caveat
#: inline (which checkpoint, per VIT_DRAW_PICKS, and the measured spread
#: it is one draw from). The other six families carry the shape claim.
#: Running is identical for all 24 -- only the reporting differs, and it
#: is decided before any arm runs.
#:
#: **The null reading, registered before any arm runs.** If the resolution
#: trend is FLAT at cleft time as it was at pretraining time, the branch's
#: answer is: **resolution does not help this pipeline -- and that is a
#: real finding, not a failure**, given the 10.8x pixel discard that
#: motivated Branch 1. The pixels exist; recovering them does not improve
#: cleft assessment through these backbones. It coheres with the frozen
#: cliff (more pixels actively hurt frozen transfer) and the flat
#: pretraining trends, and it sharpens the full-face-data argument the
#: cohort finding already makes. Registered now so it is not reached for
#: afterwards.
PHASE_7_ARM_STRUCTURE = {
    "designed": "2026-08-12",
    "arms": 24,
    "families": {
        "count": 8,
        "shape_is_the_claim": (
            "monotone up / monotone down / flat / peaked, pre-registered "
            "as the reportable result per brief §5; per-step deltas carry "
            "§4.3 and are co-reported, not headlined"
        ),
        # [DEFECT FOUND 2026-08-12] This registration names the claim and
        # its vocabulary but NO STATISTIC, and the obvious one (max-min)
        # is selected by construction. Found by trying to apply it to 22
        # arms. PHASE_7_SHAPE_STATISTIC carries the finding and the
        # narrowing; the registered text stays as written so the defect
        # is legible rather than tidied away.
        "statistic_was_never_specified": "see PHASE_7_SHAPE_STATISTIC",
        "isolates": (
            "resolution alone -- the whole matched pipeline moves one "
            "step at a time"
        ),
    },
    "init_contrasts": {
        "count": 12,
        "role": (
            "DESCRIPTIVE: masked vs imagenet at each cell, isolating SCUT "
            "pretraining's value at that resolution; they explain "
            "families, not headline"
        ),
    },
    "road_a_anchor": (
        "the 224-imagenet arms re-measure Road A's G1 cells from "
        "declared-init sets; landing near Road A's values is the "
        "provenance-not-physics cross-check"
    ),
    "seeds": {
        # [AMENDED 2026-08-12] Written as one number (5) when the arm
        # structure landed; PHASE_7_SEED_COUNTS settles it BY REGIME the
        # day the arm configs are written -- transformer 5, graph 10 --
        # and the sweep's ten seeds ARE the srgnn masked-512 arm's.
        # Amended in place, dated; the superseded value stays visible.
        "per_arm": {"transformer": 5, "graph": 10},
        "was_per_arm": 5,
        "resolves": {
            "transformer": 0.027,
            # [AMENDED 2026-08-12] Was "0.022 against Road A's graph band
            # at ten; Road B's own graph band is what the sweep
            # measures" -- the sweep measured it: 0.041452, so ten
            # resolves 0.036, not 0.022.
            "graph": 0.036,
            "graph_was": "0.022 against Road A's band, before the sweep",
        },
        "band": 0.021425,
        "rule": (
            "every arm reports its OWN seed SD (4.12.1); the band plans "
            "and never denominates"
        ),
    },
    "band_narrowing": {
        "per_cell_rule": (
            "does NOT apply as written: it governs reading against a "
            "pretraining cell's own numbers, and no Phase 7 arm does -- "
            "arms are frozen-head fits, the regime Road B Phase 3 banded "
            "ACROSS resolutions"
        ),
        "minimum_measurement": (
            "ONE graph-head confirmation sweep: ten seeds of the srgnn "
            "masked-512 arm, F-compared against Road A's graph band. The "
            "42x42 map-normalisation argument predicts it holds; the "
            "sweep measures it. Inside the interval -> one graph planning "
            "band; outside -> per-resolution graph bands and a re-read"
        ),
        "transformer_arms": "zero new bands -- SEED_BAND_RESOLVED covers them",
    },
    "vit_reporting": {
        "decided": "before any arm runs",
        "mode": "DESCRIPTIVE -- no trend claim from either ViT family",
        "second_layer": (
            "an arm's seed SD measures head-fit variance GIVEN the drawn "
            "checkpoint; the 0.055/0.027 checkpoint spread rides on top "
            "and the arm cannot see it"
        ),
        "inline_caveat": (
            "each ViT number names its checkpoint (VIT_DRAW_PICKS) and "
            "the measured spread it is one draw from"
        ),
    },
    "null_reading_registered": (
        "flat at cleft time = resolution does not help this pipeline -- a "
        "real finding given the 10.8x discard that motivated the branch, "
        "cohering with the frozen cliff and the flat pretraining trends; "
        "it sharpens the full-face-data argument. Registered before any "
        "arm runs"
    ),
}


def phase7_arms() -> list[dict]:
    """The 24 arms, as data -- one per embedding set, each naming its
    family, its regime and seed count (PHASE_7_SEED_COUNTS), and what its
    family isolates. Like ``phase6_resolution_cells``: derived, so a
    config cannot invent an arm, and the trend families are explicit
    rather than assembled at the write-up."""
    arms = []
    for entry in phase7_extraction_sets():
        family = f"{entry['backbone']}__{entry['init']}"
        regime = (
            "graph" if entry["backbone"] in ("srgnn", "agnet")
            else "transformer"
        )
        arms.append({
            "backbone": entry["backbone"],
            "resolution": entry["resolution"],
            "init": entry["init"],
            "family": family,
            "regime": regime,
            "seeds": PHASE_7_SEED_COUNTS["per_regime"][regime],
            "set_artifact": (
                "road A's swin masked-G1 224 set (bitwise reuse)"
                if entry["reuses"] else
                f"roadb_emb_{entry['backbone']}_"
                f"{'imagenet' if entry['init'] == 'imagenet' else 'masked'}"
                f"_g1_{entry['resolution']}_v1"
            ),
            "trend_claim": entry["backbone"] != "vit_b16",
            "reporting": (
                "descriptive with the draw caveat inline"
                if entry["backbone"] == "vit_b16" else
                "family shape claim, per-step deltas co-reported"
            ),
        })
    return arms


#: **[DECIDED 2026-08-12, before any arm runs] Seed counts follow the
#: arm's VARIANCE REGIME, not its claim type -- Road A's ladder rule
#: (``ladder.SEEDS_BY_KIND``), kept, with Road B's numbers behind it.**
#:
#: * **Transformer arms: FIVE seeds.** Frozen-head fits are exactly the
#:   regime Road B Phase 3 banded ACROSS resolutions (SEED_BAND_RESOLVED,
#:   0.021425): five resolve 0.027, ten 0.019. What ten would buy is that
#:   0.008 of pairwise resolution, and no registered Phase 7 reading needs
#:   it: family shapes read from three arm MEANS (SE = band/sqrt(n)), the
#:   per-step effects this pipeline has actually measured at pretraining
#:   time (0.006-0.008) resolve at NEITHER count, and the ViT arms carry a
#:   0.055/0.027 checkpoint-draw spread that more seeds cannot touch --
#:   sharpening the smaller variance component under it buys nothing.
#:   The escape is registered, not hoped: ``ladder.SEED_POOL`` nests, so
#:   extending any five-seed arm to ten later REUSES the five already run
#:   rather than replacing them.
#: * **Graph arms: TEN seeds.** Road A measured this regime at SD
#:   0.025053 with stage-D graph SDs running to 0.0567; five graph seeds
#:   would resolve only ~0.031 on the road's core observed shape (the
#:   srgnn peak is the Phase 6 trend worth having). And the confirmation
#:   sweep is ten-vs-ten BY DESIGN -- F(9,9) at ten already detects only
#:   ~2x -- so graph=10 makes the sweep BE the srgnn masked-512 arm
#:   rather than half of it.
#:
#: **Trend families vs descriptive contrasts: the answer does NOT differ,
#: and the reason is structural -- the contrasts own no arms.** The
#: twelve init contrasts read PAIRS of the same 24 arms, so there is no
#: second population to give a second count to. Claim type sets the
#: READING (shape vs delta-with-caveat) and the threshold source (both
#: arms' own SDs, 4.12.1); the count is set by the arm's regime band.
#: The observed fact stands under either count: pairwise deltas rarely
#: resolve (0.027/0.019 against plausible effect sizes), which argues for
#: honest descriptive reporting, not for more seeds.
#:
#: 12 transformer x 5 + 12 graph x 10 = **180 fits** (uniform five would
#: be 120, uniform ten 240).
PHASE_7_SEED_COUNTS = {
    "decided": "2026-08-12, before any arm runs",
    "per_regime": {"transformer": 5, "graph": 10},
    "rule_source": "ladder.SEEDS_BY_KIND -- the regime decides, kept",
    "fits": {
        "transformer": 60, "graph": 120, "total": 180,
        "uniform_five": 120, "uniform_ten": 240,
    },
    "transformer_five": (
        "the regime Road B Phase 3 banded across resolutions; ten would "
        "buy 0.027 -> 0.019 pairwise and no registered reading needs it: "
        "shapes read from means, measured per-step effects (0.006-0.008) "
        "resolve at neither count, and the ViT draw spread rides above "
        "anything seeds can sharpen"
    ),
    "graph_ten": (
        "Road A's regime SD 0.025053 with stage-D graph SDs to 0.0567; "
        "five would resolve only ~0.031 on the road's core shape, and "
        "ten makes the confirmation sweep BE the srgnn masked-512 arm"
    ),
    # [AMENDED 2026-08-12] The sweep measured Road B's own graph band at
    # 0.041452, so the numbers the graph_ten argument quoted (computed
    # against Road A's 0.025053) are superseded. The DECISION is
    # unchanged and in fact strengthened -- at the real band five would
    # resolve 0.051 and ten resolves 0.036 -- but the justification's
    # arithmetic must not keep quoting the pre-measurement figures.
    "graph_ten_amended": (
        "at the MEASURED Road B graph band (0.041452, "
        "PHASE_7_GRAPH_BAND_RESOLVED): five resolves 0.051, ten resolves "
        "0.036. The decision holds a fortiori; the '~0.031' above was "
        "computed against Road A's band before the sweep ran"
    ),
    "same_for_trend_and_descriptive": (
        "the contrasts own no arms -- they read pairs of the same 24 -- "
        "so there is no second population to count differently; claim "
        "type sets the reading, the regime band sets the count"
    ),
    "extension_escape": (
        "ladder.SEED_POOL nests: a five-seed arm's seeds are the first "
        "five of a ten-seed arm's, so extending reuses what already ran"
    ),
}


#: **[REGISTERED 2026-08-12, before the sweep runs] The graph-head band
#: sweep: both readings written down first -- the Phase 3 discipline
#: (SEED_BAND_DESIGN.decision_rule_preregistered: a window chosen after
#: seeing the result is a forking path).**
#:
#: TEN seeds of the srgnn masked-512 arm -- ``ladder.SEED_POOL``, the ten
#: Road A's graph band itself ran, so the F-comparison is like-for-like --
#: F-compared against Road A's measured graph band
#: (``graph_cleft.MEASURED_GRAPH_SEED_BAND``: SD 0.025053, ten seeds, run
#: p6-srgnn-seedband-1).
#:
#: What the seed varies is the SAME three things as Road A's band, and
#: NOT graph-layer init -- the layers warm-start from the paired
#: checkpoint, byte-identical across seeds; the seed drives classifier
#: init, batch order, and the inner-val split. Quote all three when
#: quoting either band (4.12.1: two arms can both say "seed SD" and mean
#: different things).
#:
#: THE RULE, registered before the number exists: F-based 95% CI on
#: SD_roadB/SD_roadA -- F(9,9), so the CI is the observed ratio times
#: [1/2.01, 2.01].
#:
#: * **CI contains 1** -> the sweep cannot distinguish the regimes: ONE
#:   graph planning band, the LARGER of the two SDs (the conservatism
#:   Phase 3's three-point application settled), and the graph arms
#:   PROCEED.
#: * **CI excludes 1** -> per-resolution graph bands, and the design
#:   RE-READS before any further graph arm launches. Transformer arms are
#:   not gated either way -- SEED_BAND_RESOLVED covers them, zero new
#:   bands.
#:
#: Sensitivity, stated with the rule as Phase 3 did: ten-vs-ten F(9,9)
#: detects only ~2x divergence.
#:
#: **The sweep DOUBLES AS the srgnn masked-512 arm's run**: same cell,
#: same ten seeds, task block identical by construction (the generator
#: builds both from one function). Extraction and head fits are
#: deterministic in this regime, so relaunching the arm config afterwards
#: would duplicate the sweep bit-for-bit; the sweep's run stands as that
#: arm's numbers.
PHASE_7_GRAPH_BAND_SWEEP = {
    "registered": "2026-08-12",
    "config": "roadb_p7_graph_seed_band",
    "arm": (
        "srgnn scut_masked 512 g1 native, ten seeds (ladder.SEED_POOL -- "
        "the graph band's own ten)"
    ),
    "compared_against": {
        "record": "graph_cleft.MEASURED_GRAPH_SEED_BAND",
        "sd": 0.025053,
        "n_seeds": 10,
        "run": "p6-srgnn-seedband-1",
    },
    "rule": {
        "statistic": (
            "F-based 95% CI on SD_roadB/SD_roadA, F(9,9): observed ratio "
            "x [1/2.01, 2.01]"
        ),
        "inside": (
            "one graph planning band, the LARGER of the two SDs; the "
            "graph arms proceed"
        ),
        "outside": (
            "per-resolution graph bands, and the design re-reads before "
            "any further graph arm launches; transformer arms are not "
            "gated either way (SEED_BAND_RESOLVED covers them)"
        ),
    },
    "sensitivity": "ten-vs-ten F(9,9) detects only ~2x divergence",
    "seed_varies": "classifier_init_and_batch_order_and_inner_val_split",
    "seed_does_not_vary": (
        "graph-layer init -- warm-started from the paired checkpoint, "
        "byte-identical across seeds"
    ),
    "doubles_as": (
        "the srgnn masked-512 arm's run -- same cell, same ten seeds, "
        "task block identical by construction; relaunching the arm "
        "config would duplicate it bit-for-bit, so the sweep's run "
        "stands as that arm's numbers"
    ),
    "planning_only": "4.12.1 stands: bands plan, they never denominate",
}


#: **[MEASURED 2026-08-12] The sweep ran and the pre-registered rule
#: FIRED: one graph planning band, and it is the larger -- 0.041452.**
#:
#: Road B's srgnn masked-512 arm: SD 0.041452 over ten seeds, against
#: Road A's 0.025053. Ratio 1.6546; F(9,9) 95% CI [0.823, 3.326],
#: contains 1 -- so the sweep cannot distinguish the regimes, the
#: registered "inside" branch applies, and the graph arms PROCEED.
#: Re-derived through ``train/phase3.claimable_delta``, not by hand.
#:
#: **The band nearly doubled, and the cost is stated before the arms run
#: rather than after.** At the resolved band a graph arm resolves 0.115
#: at one seed, 0.051 at five, **0.036 at ten** -- the shipped count.
#: Most effects this project has measured land at 0.02-0.08, so roughly
#: the lower half of the plausible range is unresolvable for graph arms.
#: That is known now, and it is a property of the regime rather than of
#: any result.
#:
#: **What it does NOT change.** The shape claim was pre-registered as the
#: reportable result (PHASE_7_ARM_STRUCTURE.families) and does not depend
#: on per-step deltas resolving; what mostly will not survive is a
#: per-step DELTA claim, which was already co-reported rather than
#: headlined. And 4.12.1 is untouched: this band plans, every arm still
#: reports its own SD -- the sweep's 0.041452 IS the srgnn masked-512
#: arm's own SD, since the sweep is that arm's run.
#:
#: **Escalating past ten is a design change, not a config edit.**
#: Twenty seeds would resolve 0.026 and forty 0.018, but
#: ``ladder.SEED_POOL`` holds exactly ten and ``ladder.seeds_for``
#: RAISES past it -- the 5->10 nesting escape has no 10->20 counterpart
#: without registering new seeds. Not proposed; recorded so the option's
#: real cost is visible if it is ever reached for.
PHASE_7_GRAPH_BAND_RESOLVED = {
    "measured": "2026-08-12",
    "applied_rule": "PHASE_7_GRAPH_BAND_SWEEP.rule",
    "road_b_sd": 0.041452,
    "road_a_sd": 0.025053,
    "n_seeds": 10,
    "ratio": 1.6546,
    "f_interval_95": (0.823, 3.326),
    "contains_one": True,
    "outcome": "one graph planning band, the larger",
    "planning_band": 0.041452,
    "graph_arms": "PROCEED -- the registered inside branch",
    "claimable_delta": {1: 0.115, 5: 0.051, 10: 0.036},
    "cost": (
        "most effects this project has measured are 0.02-0.08, so at ten "
        "seeds (0.036) roughly the lower half of the plausible range is "
        "unresolvable for graph arms -- known before they run"
    ),
    "does_not_change": (
        "the SHAPE claim, pre-registered as the reportable result and "
        "independent of per-step deltas resolving; what mostly will not "
        "survive is a per-step delta claim, already co-reported rather "
        "than headlined"
    ),
    "escalation": (
        "20 seeds resolve 0.026, 40 resolve 0.018 -- but ladder.SEED_POOL "
        "holds ten and seeds_for RAISES past it, so 10->20 needs new "
        "registered seeds. Not proposed; costed"
    ),
    "planning_only": (
        "4.12.1: the sweep's 0.041452 IS the srgnn masked-512 arm's own "
        "SD (the sweep is that arm's run); as a BAND it plans and "
        "denominates nothing"
    ),
}


#: **[MEASURED 2026-08-12] The two 224 anchor points are not independent
#: Road B measurements -- and they are non-independent in DIFFERENT ways,
#: which one phrase would blur.**
#:
#: **ViT imagenet 224 -- a genuine re-measurement that reproduced.** Road
#: B re-extracted this set from the DECLARED init snapshot
#: (``roadb_emb_vit_b16_imagenet_g1_224_v1``): different provenance,
#: different bytes, same physics. It returned 0.2520 against Road A's
#: recorded **0.25206** (``ladder.CLEAN_GEOMETRY_MEASUREMENTS``, mean
#: with its own five-seed SD 0.01483). So the vintage rule is confirmed:
#: agreement within ~1e-4, under 1% of that arm's own seed SD.
#:
#: **The "four decimals" phrasing needs care, and this is why.** Road A
#: presents the SAME underlying 0.25206 as 0.2520 in ``STAGE_D1_AT_G1``
#: and ``phase8.ARMS``, and as 0.2521 in three other places (this file's
#: own anchor note, ``configs/p7_extract_g1_control.yaml``, a test
#: comment). The fourth decimal therefore differs BETWEEN ROAD A'S OWN
#: RECORDS, so "0.2520 against 0.2521" compares a Road B value against a
#: rounding, not against a measurement. The defensible statement is
#: agreement within 1e-4 against 0.25206; the exact residual needs the
#: run's own metrics.json at full precision. R2 again: same name,
#: different question.
#:
#: **Swin masked 224 -- not a re-measurement at all.** The reuse arm's
#: config is IDENTICAL to ``p7_c_swin_b_scut_masked_g1.yaml`` in task and
#: inputs -- every field, every hash, verified by dict equality; only
#: ``phase`` differs (p7 vs roadb_p7). Same bytes, same procedure, same
#: five seeds, and a frozen-head fit is deterministic, so 0.1327 / sd
#: 0.0267 reproducing Road A's recorded 0.1327 / 0.0267 exactly is the
#: GUARANTEED outcome, not a coincidence and not evidence. Two arms
#: reporting one number because they are one computation.
#:
#: **Where this bites**: each family's 224 point must carry its status
#: where the family's shape is reported. ViT imagenet's is a reproduction
#: of the reference the family is read against; Swin masked's is a Road A
#: run wearing a Road B name. Neither invalidates a shape -- a shape needs
#: the points to be comparable, which they are -- but neither point is
#: independent evidence FOR Road B.
PHASE_7_ANCHOR_INDEPENDENCE = {
    "measured": "2026-08-12",
    "vit_imagenet_224": {
        "kind": "genuine re-measurement, reproduced",
        "road_b": 0.2520,
        "road_a": 0.25206,
        "road_a_source": "ladder.CLEAN_GEOMETRY_MEASUREMENTS imagenet g1",
        "road_a_own_sd": 0.01483,
        "agreement": "within ~1e-4, under 1% of the arm's own seed SD",
        "what_it_confirms": (
            "the init-vintage rule: re-extracting from the declared "
            "snapshot changed provenance, not physics"
        ),
        "rounding_caveat": (
            "Road A presents 0.25206 as 0.2520 in STAGE_D1_AT_G1 and "
            "phase8.ARMS and as 0.2521 in three other places, so 'four "
            "decimals' compares against a rounding; quote agreement "
            "within 1e-4 of 0.25206, and read the run's metrics.json for "
            "the exact residual"
        ),
    },
    "swin_imagenet_224": {
        "kind": "genuine re-measurement, reproduced -- the SECOND anchor",
        "road_b": 0.1076,
        "road_a": 0.1076,
        "road_a_source": "ladder.STAGE_D1_AT_G1 swin_b/imagenet",
        "road_a_own_sd": 0.0217,
        "agreement": "exact at the recorded 4 decimals",
        "why_it_matters": (
            "the vintage rule is confirmed on TWO independent backbones, "
            "not one -- a single agreeing point could be luck"
        ),
    },
    "vit_masked_224_is_NOT_an_anchor": {
        "road_b": 0.0846,
        "road_a": 0.0830,
        "road_a_source": "ladder.STAGE_D1_AT_G1 vit_b16/scut_masked",
        "why_not": (
            "Road B's masked 224 consumes Road B's OWN checkpoint (the "
            "unfused 224 draw, VIT_DRAW_PICKS), not Road A's -- so it "
            "was never expected to reproduce and the 0.0016 gap is not a "
            "provenance signal. It sits far inside the measured 0.055 "
            "checkpoint-draw spread, which is the only thing it says"
        ),
    },
    "swin_masked_224": {
        "kind": "not a re-measurement -- the same computation",
        "value": 0.1327,
        "sd": 0.0267,
        "road_a_record": "ladder.STAGE_D1_AT_G1 swin_b/scut_masked",
        "config_identity": (
            "task and inputs dict-equal to p7_c_swin_b_scut_masked_g1 "
            "(every field, every hash); only `phase` differs"
        ),
        "why_identical": (
            "same bytes, same procedure, same five seeds, and a "
            "frozen-head fit is deterministic -- identity is guaranteed, "
            "so it is not evidence of anything"
        ),
    },
    "reporting_rule": (
        "each family's 224 point carries its status where the family's "
        "shape is reported: ViT imagenet's is a reproduction of the "
        "reference the family is read against, Swin masked's is a Road A "
        "run under a Road B name. Shapes stay readable; neither point is "
        "independent evidence FOR Road B"
    ),
    "provenance_verdict": (
        "the init-vintage rule holds on both transformer anchors that "
        "could test it (vit 0.2520 vs 0.25206, swin 0.1076 vs 0.1076); "
        "the other two 224 points test nothing -- one is tautological, "
        "one consumes a different checkpoint"
    ),
}


#: **[OBSERVED 2026-08-12 -- NOT A TREND READING. Held deliberately.]**
#:
#: The ten runnable transformer arms are in. The values are recorded
#: because measurements are recorded; **the shapes are NOT read here**,
#: at the instruction: thirteen arms are outstanding (eleven graph
#: arms, now carrying the 0.041452 band, plus the two masked-768 cells),
#: and the graph half of the design is what most of the trend question
#: rests on.
#:
#: **Report-gate status, stated precisely.** Both imagenet families are
#: COMPLETE (three points each) and are therefore report-ELIGIBLE under
#: PHASE_7_PARTIAL_LAUNCH's rule. They are held anyway, and the reason
#: is recorded rather than left as silence: the outstanding graph arms
#: change what the whole design can say, and both families' 224 points
#: carry the anchor caveats in PHASE_7_ANCHOR_INDEPENDENCE. Both masked
#: families are incomplete (768 pending) and gate-blocked regardless.
#:
#: **Three observations that are descriptions, not readings**:
#:
#: * ViT falls sharply above 224 in both inits -- the same direction as
#:   the frozen cliff (BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER),
#:   which is where that mechanism predicted it would go. ViT claims no
#:   trend either way (vit_reporting: DESCRIPTIVE).
#: * Swin imagenet is the one family whose middle point RISES -- 0.2050
#:   at 512, the only non-224 arm approaching baseline territory. Swin is
#:   also the backbone whose resolution transfer is architecturally clean
#:   (SWIN_AT_RESOLUTION: window-relative, parameters size-invariant), so
#:   this is the cell where a real resolution effect was most plausible a
#:   priori. It deserves attention rather than dismissal -- and it is one
#:   family of eight.
#: * ViT masked 512 is -0.1002: negatively correlated, worse than
#:   predicting the training mean. Recorded flatly; it is one checkpoint
#:   draw of a nondeterministic pretraining regime and carries the draw
#:   caveat inline like every ViT number.
PHASE_7_FIRST_TRANSFORMER_ARMS = {
    "observed": "2026-08-12",
    # [SUPERSEDED 2026-08-12, same day] The graph arms landed hours
    # later. PHASE_7_TWENTY_TWO_ARMS carries all 22 with every contrast
    # computed; this record stays because its HOLD and its reasoning
    # were written before the graph half existed, and a later reader
    # should be able to see that the hold predated the full table.
    "superseded_by": "PHASE_7_TWENTY_TWO_ARMS",
    "arms_in": 10,
    "arms_outstanding": 13,
    "not_a_trend_reading": (
        "held at instruction: eleven graph arms (now at the 0.041452 "
        "band) and two masked-768 cells are outstanding"
    ),
    "values": {
        "vit_b16__imagenet": {224: 0.2520, 512: 0.0894, 768: 0.0937},
        "vit_b16__scut_masked": {224: 0.0846, 512: -0.1002, 768: None},
        "swin_b__imagenet": {224: 0.1076, 512: 0.2050, 768: 0.1537},
        "swin_b__scut_masked": {224: 0.1327, 512: 0.0301, 768: None},
    },
    "complete_families": ("vit_b16__imagenet", "swin_b__imagenet"),
    "report_eligible_but_held": (
        "both imagenet families are complete and pass the gate; held "
        "because the outstanding graph arms change what the design can "
        "say, and their 224 points carry the anchor caveats"
    ),
    "observations": {
        "vit_falls_above_224": (
            "both inits, the direction the frozen cliff predicted; ViT "
            "claims no trend either way (DESCRIPTIVE)"
        ),
        "swin_imagenet_rises_at_512": (
            "0.2050, the only non-224 arm near baseline territory, in "
            "the backbone whose resolution transfer is architecturally "
            "clean -- worth attention rather than dismissal, and one "
            "family of eight"
        ),
        "vit_masked_512_is_negative": (
            "-0.1002, worse than predicting the training mean; one draw "
            "of a nondeterministic pretraining regime, carrying the draw "
            "caveat inline"
        ),
    },
}


#: **[MEASURED 2026-08-12] Twenty-two arms, six complete families. Every
#: contrast re-derived through ``phase3.combined_claimable_delta`` rather
#: than against the planning band -- which changes three readings.**
#:
#: **Correction one: the graph declines are mostly RESOLVABLE.** Reading
#: the graph families against the 0.036 ten-seed planning band suggests
#: only srgnn-masked's drop approaches significance. But 4.12.1 forbids
#: exactly that: the band plans, and a delta is thresholded from the two
#: ARMS' OWN SDs. Re-derived per pair, three of four graph families'
#: endpoint declines clear -- agnet-masked -0.0658 against 0.0316
#: (4.1 sigma), srgnn-masked -0.0751 against 0.0388 (3.8), srgnn-imagenet
#: -0.0543 against 0.0531 (2.0, marginal). Only agnet-imagenet sits
#: inside its band, and its up-down-up is a shape drawn through noise
#: exactly as read.
#:
#: **Correction two: Swin is not one cell -- it is one FAMILY, and both
#: of its non-224 points clear.** 512 is +0.0974 against arm_means_95
#: 0.0267 (7.2 sigma) and 768 is +0.0461 against 0.0326 (2.8). The 512
#: point also exceeds ``single_run_95`` (0.0596), the conservative
#: companion -- the difference survives without averaging, which
#: ``combined_claimable_delta`` says is worth stating when true. Of the
#: fourteen non-224 cells, three sit above their own 224 and the two that
#: clear are both this family's.
#:
#: **Correction three: the multiplicity argument does not carry.** Over
#: fourteen comparisons a global null gives 0.70 expected exceedances and
#: a largest-of-fourteen around 2.2-2.7 sigma. Swin-512 is 7.2. Chance at
#: this rate does not produce it, and Bonferroni over fourteen leaves it
#: untouched. **The honest deflator is a different one entirely** -- see
#: PHASE_7_SWIN_512: seed thresholds answer "do these arms differ on THIS
#: cohort", and Road A already measured what happens when the patient
#: level is asked instead (``phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT``:
#: nine seed-claimable verdicts, 0 of 45 intervals excluding zero).
PHASE_7_TWENTY_TWO_ARMS = {
    "measured": "2026-08-12",
    "arms": 22,
    "complete_families": 6,
    "pending": ("swin_b__scut_masked 768", "vit_b16__scut_masked 768"),
    "values": {
        # family: {resolution: (pcc, own seed sd)}
        "agnet__imagenet": {
            224: (0.0612, 0.043), 512: (0.0309, 0.040), 768: (0.0929, 0.039)},
        "agnet__scut_masked": {
            224: (0.0678, 0.024), 512: (0.0410, 0.053), 768: (0.0020, 0.045)},
        "srgnn__imagenet": {
            224: (0.1719, 0.063), 512: (0.1345, 0.031), 768: (0.1176, 0.058)},
        "srgnn__scut_masked": {
            224: (0.1249, 0.028), 512: (0.1135, 0.041), 768: (0.0498, 0.056)},
        "swin_b__imagenet": {
            224: (0.1076, 0.022), 512: (0.2050, 0.021), 768: (0.1537, 0.030)},
        "swin_b__scut_masked": {224: (0.1327, 0.027), 512: (0.0301, 0.027)},
        "vit_b16__imagenet": {
            224: (0.2520, 0.015), 512: (0.0894, 0.020), 768: (0.0937, 0.022)},
        "vit_b16__scut_masked": {224: (0.0846, 0.015), 512: (-0.1002, 0.019)},
    },
    "endpoint_contrasts_224_to_768": {
        # (delta, arm_means_95, sigma) -- both points fixed by the design
        "agnet__imagenet": (+0.0317, 0.0360, +1.73),
        "agnet__scut_masked": (-0.0658, 0.0316, -4.08),
        "srgnn__imagenet": (-0.0543, 0.0531, -2.01),
        "srgnn__scut_masked": (-0.0751, 0.0388, -3.79),
        "swin_b__imagenet": (+0.0461, 0.0326, +2.77),
        "vit_b16__imagenet": (-0.1583, 0.0233, -13.29),
    },
    "band_is_not_the_denominator": (
        "4.12.1: reading the graph families against the 0.036 planning "
        "band finds one resolvable decline; thresholding each pair from "
        "the two arms' OWN SDs finds three of four"
    ),
    "cells_above_their_own_224": (
        "3 of 14 (agnet 768 +0.0317 inside its band; swin_b 512 +0.0974 "
        "and 768 +0.0461, both clearing) -- the two that clear are one "
        "family's"
    ),
    "multiplicity": (
        "14 comparisons, 0.70 expected exceedances under a global null, "
        "largest-of-14 around 2.2-2.7 sigma. Swin-512 at 7.2 is not what "
        "chance produces at this rate; Bonferroni leaves it untouched"
    ),
    "the_real_deflator": (
        "seed thresholds answer 'do these arms differ on THIS cohort'. "
        "phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT is what happened last "
        "time the patient level was asked instead"
    ),
}


#: **[DECIDED 2026-08-12] How a family SHAPE may be claimed -- the gap
#: the pre-registration left, named rather than glossed.**
#:
#: PHASE_7_ARM_STRUCTURE registered "the shape is the claim" and listed
#: the vocabulary (monotone up/down/flat/peaked). What it never specified
#: is the STATISTIC -- and the obvious one is invalid:
#:
#: * **Range (max-min) is a SELECTED statistic.** Which point is highest
#:   and which lowest are outcomes, so comparing a range to a pairwise
#:   threshold is anti-conservative -- the same winner's-curse structure
#:   that makes "the best of eighteen cells" untrustworthy, applied
#:   inside a family. Every family's range here exceeds the graph band;
#:   that fact means less than it looks.
#: * **The endpoint contrast (224 -> 768) is not selected.** Both points
#:   are fixed by the design before any number exists, so it is one
#:   pre-specified test per family, thresholded from the two arms' own
#:   SDs. That is the defensible per-family claim.
#: * **The middle point is DESCRIPTIVE.** "Peaked at 512" names which of
#:   three points came highest, which is a selection; it is reported and
#:   not claimed -- except where a mechanism named the cell in advance,
#:   which happened exactly once (SWIN_RESOLUTION_PREDICTION).
#:
#: **So shapes are reportable, and the claim is narrower than registered:
#: direction over the endpoints, tested; the interior described.** the maintainer's
#: reading is right -- at these bands a three-point shape is
#: closer to descriptive than the pre-registration assumed. The
#: correction is to say what part is testable rather than to keep the
#: word "claim" over all of it.
PHASE_7_SHAPE_STATISTIC = {
    "decided": "2026-08-12",
    #: **This is a FINDING, not only a decision** -- a defect in the
    #: project's own pre-registration, found by trying to apply it. The
    #: registration was written when no arm had run; it named the claim
    #: ("the shape") and the vocabulary (monotone/flat/peaked) and never
    #: noticed that neither implies a testable quantity. Twenty-two arms
    #: later, the question "so what number do I compare to what
    #: threshold?" has no registered answer, and the answer a reader
    #: would reach for is invalid.
    #:
    #: **Caught at the best possible moment**: after the numbers exist
    #: (so the gap is concrete) but before any shape has been reported
    #: (so nothing has to be withdrawn). The failure mode it avoids is
    #: the one Road A paid for twice -- a criterion registered, not
    #: applied, and discovered only when applying it would have changed
    #: published verdicts (ladder.CLAIMS_REST_ON_HALF_THE_CRITERION,
    #: phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT).
    #:
    #: **The generalisable lesson**: a pre-registration that names a
    #: claim must also name the STATISTIC and its threshold, or it has
    #: registered a topic rather than a test. "The shape is the claim"
    #: reads like a commitment and licenses whatever the data suggests.
    "is_a_finding": (
        "a pre-registration defect in this project's own registration, "
        "found by applying it to 22 arms -- recorded beside the arms, "
        "not only in the reasoning"
    ),
    "caught_when": (
        "after the numbers existed so the gap was concrete, before any "
        "shape was reported so nothing had to be withdrawn"
    ),
    "the_failure_it_avoids": (
        "a criterion registered but not applied until applying it would "
        "change published verdicts -- twice paid for already "
        "(ladder.CLAIMS_REST_ON_HALF_THE_CRITERION, "
        "phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT)"
    ),
    "generalisable_lesson": (
        "a pre-registration that names a claim must also name the "
        "STATISTIC and its threshold, or it has registered a TOPIC "
        "rather than a test -- 'the shape is the claim' reads like a "
        "commitment while licensing whatever the data suggests"
    ),
    "gap": (
        "PHASE_7_ARM_STRUCTURE registered 'the shape is the claim' and "
        "the shape vocabulary, but never a shape STATISTIC"
    ),
    "range_is_selected": (
        "which point is max and which is min are outcomes, so a range "
        "against a pairwise threshold is anti-conservative -- winner's "
        "curse inside the family"
    ),
    "claim_statistic": (
        "the endpoint contrast 224->768: both points fixed by the design "
        "before any number existed, thresholded from the two arms' own "
        "SDs. One pre-specified test per family"
    ),
    "interior_is_descriptive": (
        "'peaked at 512' names which of three points came highest, which "
        "is a selection -- described, not claimed, except where a "
        "mechanism named the cell in advance"
    ),
    "honest_summary": (
        "at these bands a three-point shape is closer to descriptive "
        "than the pre-registration assumed; the fix is to narrow what is "
        "claimed, not to keep 'claim' over all of it"
    ),
}


#: **[DECIDED 2026-08-12] What Swin-512 warrants: the paired BCa, which
#: needs no new runs -- and NOT more seeds, which would answer a question
#: already answered.**
#:
#: **The pre-registration covers less than the observation, and the
#: difference is the whole question.** SWIN_RESOLUTION_PREDICTION's
#: frozen-regime branch, registered 2026-08-09 before any Phase 6 run,
#: reads: supports = *"swin near its own 224 level"*, refutes = *"swin
#: dropping like ViT's ~0.14"*. So what was predicted is the ABSENCE OF A
#: COLLAPSE. That is confirmed, twice over (512 and 768 both above 224,
#: both clearing) and against a ViT that fell -0.158 in the same design.
#: **The GAIN was not predicted.** Its magnitude and its location at 512
#: are post-hoc, and "the biggest number in the table" is what selected
#: them. Both statements are true at once, and separating them is what
#: keeps the pursuit honest.
#:
#: **More seeds is therefore refused, and the first reason is not
#: selection.** The delta already clears at 7.2 sigma in seed terms and
#: even survives ``single_run_95``; more seeds sharpen an estimate that
#: is not the binding uncertainty. What IS open is whether the effect
#: survives the PATIENT level -- and seeds cannot touch that. Adding
#: seeds to the peak cell would also condition the estimate on its own
#: selection (a tighter interval around a winner's-curse-biased mean),
#: but that is the second objection, not the first.
#:
#: **The right test is the project's own pre-registered one, and it can
#: fail.** PLAN 4.3's criterion is the per-seed paired BCa
#: (``phase7b.paired_comparison``), and Road A's history is the reason to
#: take it seriously: it was promised, deferred, and when finally run it
#: withdrew all nine verdicts (0 of 45 intervals excluding zero,
#: ``phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT``). **It costs no cluster
#: time**: ``train_cv``/``train_graph_cv`` already write
#: ``seed_<n>__predictions.csv`` per arm, and both sides of every
#: contrast are OOF vectors over the same 237 patients with the same
#: folds and the same seed-driven splits -- pairing works exactly as
#: Road A's did.
#:
#: **Applied to ALL endpoint contrasts, not to Swin alone** -- running it
#: only on the interesting cell would re-introduce the selection the test
#: is meant to defend against.
#:
#: If a NEW measurement is ever wanted, the defensible one is an
#: independent cell that could refute the mechanism rather than a
#: precision increase on the observation: swin imagenet at square G2,
#: which needs no pretraining (imagenet extracts from the declared
#: snapshot) and whose staged artifacts already exist and are currently
#: unconsumed (``roadb_512_square_g2_v1``, ``roadb_768_square_g2_v1``,
#: PHASE_7_STAGED_CONSUMPTION). The architectural mechanism is
#: geometry-independent, so it predicts the effect appears there too.
#: **Not proposed for launch here** -- recorded so the option is costed
#: and its pre-registration exists before any number does.
PHASE_7_SWIN_512 = {
    "decided": "2026-08-12",
    "observed": {
        "delta_vs_own_224": 0.0974, "arm_means_95": 0.0267,
        "single_run_95": 0.0596, "sigma": 7.16,
        "family_768_also_clears": (0.0461, 0.0326, 2.77),
    },
    "what_was_predicted": (
        "SWIN_RESOLUTION_PREDICTION.frozen_regime.supports -- 'swin near "
        "its own 224 level', i.e. the ABSENCE of a collapse. Confirmed "
        "twice, against a ViT that fell -0.158 in the same design"
    ),
    "what_was_not_predicted": (
        "the GAIN -- its magnitude and its location at 512 are post-hoc, "
        "and being the largest number in the table is what selected them"
    ),
    "more_seeds": {
        "verdict": "REFUSED",
        "first_reason": (
            "it answers a question already answered: the delta clears at "
            "7.2 sigma and survives single_run_95. Seed precision is not "
            "the binding uncertainty"
        ),
        "second_reason": (
            "selecting the peak cell and measuring it harder conditions "
            "the estimate on its own selection -- a tighter interval "
            "around a winner's-curse-biased mean"
        ),
    },
    "right_test": {
        "what": "the per-seed paired BCa, phase7b.paired_comparison",
        "why": (
            "PLAN 4.3's own criterion, and the one that withdrew all "
            "nine Road A verdicts when finally applied "
            "(phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT) -- it can fail"
        ),
        "costs_no_cluster_time": (
            "train_cv/train_graph_cv already write "
            "seed_<n>__predictions.csv; both sides are OOF vectors over "
            "the same 237 patients, same folds, same seed-driven splits"
        ),
        "scope": (
            "ALL endpoint contrasts, not Swin alone -- running it only "
            "on the interesting cell re-introduces the selection it "
            "defends against"
        ),
    },
    "if_a_new_cell_is_ever_wanted": (
        "swin imagenet at square G2 -- needs no pretraining, staged "
        "artifacts already exist unconsumed, and the architectural "
        "mechanism is geometry-independent so it predicts the effect "
        "appears there too. Registered, not proposed"
    ),
    # [MEASURED 2026-08-12] The paired BCa ran. **The gain is WITHDRAWN:
    # 3.66x on condition 2, 1 of 5 on condition 1.** It failed in the
    # band where Road A's 3.62x failed and Road A's 3.83x survived --
    # the ambiguous zone, exactly where the pre-run note said prediction
    # was impossible and only the test would tell.
    #
    # **The two statements stay separate, and only one of them moved.**
    "outcome": {
        "measured": "2026-08-12",
        "the_gain": {
            "verdict": "WITHDRAWN",
            "condition_2": "3.66x",
            "condition_1": "1 of 5 intervals exclude zero",
            "reading": (
                "the one apparent gain in Road B is NOT claimable at "
                "n=237. It is the highest-scoring non-224 arm and it is "
                "not claimably above its own 224 point"
            ),
            "context": (
                "3.66x sits between Road A's 3.62x failure and its 3.83x "
                "survivor -- the band where the margin decides nothing"
            ),
        },
        "the_prediction": {
            "verdict": "STANDS, and in a sharper form than before",
            "what_it_said": (
                "frozen swin stays near its own 224 level where frozen "
                "ViT collapses (SWIN_RESOLUTION_PREDICTION.frozen_regime)"
            ),
            "what_the_criterion_returned": (
                "in ONE design at ONE standard: ViT's collapse is "
                "claimable three times over, and Swin's change is not "
                "claimable in either direction. The asymmetry the "
                "prediction named is what condition 1 produced"
            ),
            "stated_honestly": (
                "Swin's side is an UNRESOLVED NULL, not a demonstration "
                "of stability -- absence of a claimable change is not "
                "proof of no change. The prediction is supported by the "
                "contrast, not by a positive Swin result"
            ),
        },
        "why_they_must_not_merge": (
            "the prediction was registered before any number and "
            "concerned the absence of a collapse; the gain was post-hoc "
            "and concerned a magnitude. Reporting the withdrawn gain as "
            "though it damaged the prediction would discard a confirmed "
            "pre-registration; reporting the prediction as though it "
            "vindicated the gain would launder a selected maximum"
        ),
    },
    # And the replication proposed above is now WITHDRAWN as advice --
    # see PHASE_7_CLOSING.nothing_further_is_worth_running.
    "g2_replication_superseded": (
        "proposed when the open question was 'is the gain real'; the "
        "criterion has since answered that this cohort cannot resolve a "
        "gain of that size at all, so a second cell at n=237 meets the "
        "same wall. Recommended AGAINST -- PHASE_7_CLOSING"
    ),
}


#: **[READ 2026-08-12, with two arms still pending -- deliberately]**
#:
#: Road B exists because staging at 224 discards ~99% of the available
#: pixels (10.8x linear). Twenty-two arms now answer whether the models
#: can use them, and **the pre-registered null reading fires -- in a
#: STRONGER form than it was registered in.**
#:
#: Registered (PHASE_7_ARM_STRUCTURE.null_reading_registered): *flat at
#: cleft time = resolution does not help this pipeline*. Observed: not
#: flat but **declining**, and resolvably so. Eleven of fourteen non-224
#: cells sit below their own 224 point; of the six complete families,
#: four decline over the endpoints and clear their own thresholds (ViT
#: imagenet -0.158, srgnn-masked -0.075, agnet-masked -0.066,
#: srgnn-imagenet -0.054), one is inside its band, and one rises.
#:
#: **The exception is a single family, and it is the one a mechanism
#: named in advance.** Swin imagenet is above its own 224 at both higher
#: resolutions. Everything else says the pixels are not usable through
#: this pipeline; Swin says the architecture that does not interpolate
#: its positional machinery does not lose by seeing more of them.
#:
#: **What this supports.** The pixels exist and are discarded; recovering
#: them does not improve cleft assessment through frozen backbones at
#: n=237 -- which coheres with the frozen cliff
#: (BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER, measured through an
#: independent route: Phase 3's inline sweeps gave 0.0857/0.0898 at
#: 512/768 where Phase 7's precomputed ViT family gives 0.0894/0.0937)
#: and with Phase 6's flat pretraining trends. It sharpens the
#: full-face-data argument the cohort finding already makes.
#:
#: **What it does NOT support.** Not "the pixels are uninformative" --
#: these are frozen backbones, linear/graph heads, 237 patients, one
#: cohort. Not "resolution never helps" -- one family, predicted in
#: advance, says otherwise at both its points. And every figure here is a
#: seed-level statement on a cohort whose patient-level intervals
#: withdrew nine verdicts last time they were computed.
#:
#: **Written with two arms pending, on purpose.** Both are masked-768
#: transformer cells. Neither can move the reading: ViT's family is
#: descriptive by decision, and swin-masked already fell -0.103 at 512.
#: Recording now is what stops the reading from being assembled after the
#: last numbers land.
PHASE_7_PREMISE_READING = {
    "read": "2026-08-12, with two arms pending, deliberately",
    "premise": "224 staging discards ~99% of available pixels (10.8x linear)",
    "registered_null": "flat at cleft time = resolution does not help",
    "observed": "not flat but DECLINING, and resolvably so",
    "tally": (
        "11 of 14 non-224 cells below their own 224; of six complete "
        "families four decline over the endpoints and clear, one is "
        "inside its band, one rises"
    ),
    "the_exception": (
        "swin_b__imagenet, above its own 224 at BOTH higher resolutions "
        "-- the one family a mechanism named in advance"
    ),
    "supports": (
        "recovering the discarded pixels does not improve cleft "
        "assessment through frozen backbones at n=237; coheres with the "
        "frozen cliff (reproduced by an independent route: Phase 3's "
        "inline sweeps 0.0857/0.0898 against Phase 7's precomputed "
        "0.0894/0.0937) and Phase 6's flat pretraining trends"
    ),
    "does_not_support": (
        "NOT 'the pixels are uninformative' -- frozen backbones, frozen "
        "heads, 237 patients, one cohort; NOT 'resolution never helps' "
        "-- one predicted family says otherwise at both its points"
    ),
    "cohort_caveat": (
        "every figure is seed-level on a cohort whose patient-level "
        "intervals withdrew nine verdicts last time they were computed "
        "(phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT)"
    ),
    "why_now": (
        "both pending arms are masked-768 transformer cells and neither "
        "can move it: ViT's family is descriptive by decision and "
        "swin-masked already fell -0.103 at 512. Writing it now is what "
        "stops it being assembled after the last numbers land"
    ),
}


#: **[DECIDED 2026-08-12] What the Road B paired-BCa scope covers, and
#: what it deliberately does not.**
#:
#: **Covered: every pairwise RESOLUTION contrast among arms that have
#: run** -- all three per complete family (224-512, 512-768, 224-768) and
#: the one computable pair in each incomplete family. That rule is
#: uniform by construction, which is the point: applying condition 1 only
#: to the endpoint contrasts would leave the Swin interior untested, and
#: applying it only to Swin's interior would re-introduce the selection
#: the test exists to defend against (PHASE_7_SWIN_512). The set is fixed
#: by the design, not by any outcome.
#:
#: **Not covered: the twelve init contrasts.** They are DESCRIPTIVE by
#: decision (PHASE_7_ARM_STRUCTURE.init_contrasts), so they carry no
#: claims for condition 1 to withdraw. The scope is the resolution axis.
#:
#: **Not covered: the two pending masked-768 arms**, which have no
#: vectors. They enter automatically when their values are recorded --
#: the pairs derive from PHASE_7_TWENTY_TWO_ARMS, so nothing has to
#: remember to add them.
#:
#: **ViT is included and stays descriptive either way.** Running the
#: criterion over its contrasts reports intervals; it does not create a
#: claim the reporting decision already declined to make.
PAIRED_CLAIM_COVERAGE = {
    "scope": "roadb_resolution",
    "covers": (
        "every pairwise resolution contrast among arms that have run -- "
        "three per complete family, one per incomplete family"
    ),
    "why_uniform": (
        "endpoints only would leave the Swin interior untested; Swin's "
        "interior only would re-introduce the selection condition 1 "
        "exists to defend against. The set is fixed by the design"
    ),
    "excludes_init_contrasts": (
        "DESCRIPTIVE by decision, so they carry no claims for condition "
        "1 to withdraw -- this scope is the resolution axis"
    ),
    "excludes_pending": (
        "the two masked-768 arms have no vectors; they enter "
        "automatically when their values are recorded, because the pairs "
        "derive from PHASE_7_TWENTY_TWO_ARMS"
    ),
    "vit_stays_descriptive": (
        "included for uniformity; the criterion reports intervals and "
        "does not create a claim the reporting decision declined to make"
    ),
}


#: **[CORRECTED 2026-08-12, from the run listing] Every stem is its own
#: config's run directory -- the two special routings were unnecessary,
#: and the reason to drop them is the CODE SHA rather than tidiness.**
#:
#: Both cells I routed elsewhere turned out to have their own arm runs:
#:
#: * **srgnn masked 512** ran BOTH ways -- as the band sweep
#:   (``roadb_p7_graph_seed_band__30cfbfe2``) and, later, as its own arm
#:   with the other eleven graph arms (``__a54cdfae``). The configs are
#:   dict-equal (PHASE_7_GRAPH_BAND_SWEEP.doubles_as), so the two runs
#:   are the same computation at two code SHAs.
#: * **swin masked 224** has a Road B arm run (``__30cfbfe2``) beside
#:   Road A's Stage C run (``p7_c_swin_b_scut_masked_g1__4cb62c05``) --
#:   config-identical, PHASE_7_ANCHOR_INDEPENDENCE.swin_masked_224.
#:
#: **Declaring each arm's OWN run keeps every family's points at one code
#: SHA.** The graph arms all sit at a54cdfae and the transformer arms at
#: 30cfbfe2; routing srgnn-512 to the sweep would have put one point of
#: the srgnn masked family at a different SHA from its 224 and 768
#: siblings, and routing swin-224 to Road A's would have spanned
#: 4cb62c05 and 30cfbfe2 inside one contrast. This project has already
#: learned that a comparison must hold its environment fixed
#: (PRETRAINED_INIT_IS_NOW_DECLARABLE: the 512 pair dissolved into A40
#: vs Blackwell), and a code SHA is part of the environment.
#:
#: **The alternates are not waste -- they are two free duplicate pairs.**
#: Each is the same computation run twice, so comparing the two
#: prediction sets is an end-to-end determinism check on the head-fit
#: path, of exactly the shape MASKED_SCUT_REBUILD_WAS_A_DETERMINISM_CHECK
#: was. Not run here; recorded so the opportunity is visible.
PAIRED_CLAIM_DUPLICATE_RUNS = {
    "corrected": "2026-08-12, from the run listing",
    "rule": "every stem is its own config's run directory; no routing",
    "why": (
        "it keeps each family's points at ONE code SHA -- graph arms at "
        "a54cdfae, transformer arms at 30cfbfe2 -- and a code SHA is "
        "part of the environment a comparison must hold fixed"
    ),
    "duplicates": {
        "srgnn_masked_512": (
            "ran as the band sweep (30cfbfe2) AND as its own arm "
            "(a54cdfae); dict-equal configs, so the same computation "
            "twice"
        ),
        "swin_masked_224": (
            "Road B arm run (30cfbfe2) beside Road A's Stage C run "
            "(4cb62c05); config-identical"
        ),
    },
    "free_check_available": (
        "each duplicate pair is the same computation run twice, so "
        "comparing their prediction sets is an end-to-end determinism "
        "check on the head-fit path -- not run, recorded"
    ),
}


def paired_claim_pairs(scope: str = "roadb_resolution") -> list[dict]:
    """Road B's resolution pairs, derived from the recorded arms.

    Shaped exactly like ``ladder.paired_claim_pairs`` so the ONE paired
    implementation serves both roads -- ``task_paired_claims`` walks
    these, ``ladder``'s three derivations take them as an argument, and
    ``phase7b.paired_comparison`` does the arithmetic. Nothing here
    re-implements a comparison.

    Derived rather than listed: a family's pairs come from the arms that
    have recorded values, so an arm cannot be silently omitted and the
    pending cells appear the moment their numbers land.
    """
    if scope == "roadb_regioncrop":
        return _regioncrop_claim_pairs()
    if scope != "roadb_resolution":
        raise ValueError(f"unknown Road B paired-claim scope {scope!r}")

    from . import ladder
    from .train.phase3 import combined_claimable_delta

    by_family = {}
    for arm in phase7_arms():
        by_family.setdefault(arm["family"], {})[arm["resolution"]] = arm

    pairs = []
    for family in sorted(by_family):
        arms = by_family[family]
        recorded = PHASE_7_TWENTY_TWO_ARMS["values"].get(family, {})
        run = sorted(res for res in arms if res in recorded)
        for i, lo in enumerate(run):
            for hi in run[i + 1:]:
                arm_lo, arm_hi = arms[lo], arms[hi]
                seeds = list(ladder.SEED_POOL[: arm_hi["seeds"]])
                mean_lo, sd_lo = recorded[lo]
                mean_hi, sd_hi = recorded[hi]
                threshold = combined_claimable_delta(
                    sd_hi, len(seeds), sd_lo, len(seeds)
                )
                delta = mean_hi - mean_lo
                pairs.append({
                    "key": f"resolution__{family}__{lo}_to_{hi}",
                    "question": "resolution",
                    "varies": "resolution",
                    "a": _paired_claim_stem(arm_lo),
                    "b": _paired_claim_stem(arm_hi),
                    "seeds": seeds,
                    "recorded": {
                        "delta": round(delta, 6),
                        "threshold": round(threshold["arm_means_95"], 6),
                        "margin": round(
                            abs(delta) / threshold["arm_means_95"], 3
                        ),
                        "is_endpoint": lo == 224 and hi == 768,
                    },
                })
    return pairs


#: **[OBSERVED 2026-08-14 -- NOT A RESULT. The paired BCa decides.]**
#:
#:     anatomy_concat_vit   +0.2428  sd 0.0154  n 5
#:     control_whole_vit    +0.2307  sd 0.0223  n 5
#:     random_concat_vit    +0.1655  sd 0.0310  n 5
#:
#: **Anatomy beats random by +0.077 at 2.6x its combined five-seed
#: threshold** -- and that is the strictly one-factor contrast: the same
#: 22 boxes, the same sizes, the same total area, only placement differs
#: (RANDOM_ARM_NAMES_WERE_THE_DEFECT confirmed the matching survived the
#: naming fix). So if it survives, the finding is about ANATOMY, not
#: about cropping.
#:
#: **That would be the first time placement has mattered in this
#: project.** The scheme axis was null four times -- Phase 4's frozen
#: embeddings, pretraining twice, and cleft with trained graph layers --
#: and Charm (2025) reported random patch selection beating saliency,
#: entropy and gradient selection on small aesthetic datasets. The brief
#: made arm 4 mandatory precisely because "anatomy helps" and "cropping
#: helps" are otherwise indistinguishable; here they separate, in the
#: direction the literature did not predict.
#:
#: **And the shape is coherent rather than a lone number.** Anatomy
#: against the whole frame is +0.012 -- essentially nothing. So cropping
#: to anatomy is AS GOOD AS the whole face, not better, while random
#: cropping is much worse: discarding the periphery costs nothing,
#: discarding the right regions costs a lot. A single number could be
#: noise; a pattern with a null in it is harder to produce by accident.
#:
#: **Two cautions, both registered before the criterion runs.**
#:
#: * **The bar is the paired BCa, and precedent says this may withdraw.**
#:   Swin's 3.66x peak failed condition 1 at 1 of 5
#:   (PHASE_7_SWIN_512.outcome), and 2.6x is BELOW that margin. Road A's
#:   two highest margins also failed. Condition 2 does not predict
#:   condition 1, measured across 26 pairs.
#: * **0.2428 does not beat Road A's 0.2520.** It approaches it by a
#:   different route -- 22 magnified anatomy crops rather than one whole
#:   frame -- which is interesting for what it says about what the model
#:   uses, and is not a new best arm.
REGION_CROP_ARMS_OBSERVED = {
    "observed": "2026-08-14",
    "not_a_result": "the paired BCa decides; three contrasts are computable",
    "values": {
        "anatomy_concat_vit": {"pcc": 0.2428, "sd": 0.0154, "n": 5},
        "control_whole_vit": {"pcc": 0.2307, "sd": 0.0223, "n": 5},
        "random_concat_vit": {"pcc": 0.1655, "sd": 0.0310, "n": 5},
        "anatomy_concat_srgnn": {"pcc": 0.1717, "sd": 0.0222, "n": 10},
    },
    #: **[ADDED 2026-08-14] What the SR-GNN arm contributes, which is not
    #: one of the four registered contrasts.**
    #:
    #: It encodes the SAME CROP PIXELS as ``anatomy_concat_vit`` -- the
    #: same 22 boxes, cut from the same staged artifact at the same
    #: declared hash by the same deterministic path -- and lands 0.071
    #: lower. (The embedding ARTIFACTS differ, necessarily: the artifact
    #: IS the encoding. What is byte-identical is the input to the two
    #: backbones.)
    #:
    #: 0.1717 sits inside SR-GNN's range across every arm it has run on
    #: either road (0.12-0.17). **So on identical input the BACKBONE caps
    #: the result, not the representation** -- and this is a cleaner
    #: backbone comparison than any arm in either road, because elsewhere
    #: the two sides were matched by construction and here they are
    #: identical by bytes.
    "srgnn_on_identical_pixels": {
        "delta_vs_vit": -0.0711,
        # [CORRECTED 2026-08-14, by the test] 0.1717 is NOT inside
        # 0.12-0.17 -- it is at the top, and the precise fact is sharper
        # than the range was: SR-GNN's best prior arm is 0.1719 (Road A,
        # imagenet, G1, WHOLE FRAME, ladder.STAGE_D1_AT_G1). Branch 3's
        # SR-GNN on 22 magnified anatomy crops gives 0.1717 -- a
        # difference of 0.0002.
        "srgnn_best_prior": {"pcc": 0.1719, "arm": "Road A imagenet G1 whole frame"},
        "difference_from_best_prior": 0.0002,
        "reading": (
            "on byte-identical crop pixels the backbone caps the result, "
            "not the representation -- and changing the representation "
            "ENTIRELY (whole 224 frame -> 22 magnified anatomy crops) "
            "moved SR-GNN by 0.0002"
        ),
        "not_a_controlled_comparison": (
            "the 0.1719 arm differs in road, staging, aspect, geometry "
            "and input; that is what makes the agreement a CEILING "
            "observation rather than a contrast. Nothing is claimed from "
            "it -- 0.0002 is far below what either arm can resolve"
        ),
        "why_cleaner_than_any_other_backbone_pair": (
            "elsewhere the two sides were matched by construction; here "
            "the input is identical by bytes -- same boxes, same staged "
            "artifact at the same hash, same deterministic crop path"
        ),
        "artifacts_differ_necessarily": (
            "the embedding artifact IS the encoding, so the two sets are "
            "different by definition; what is identical is their input"
        ),
    },
    "anatomy_vs_random": {
        "delta": 0.0773, "margin": 2.6,
        "why_it_matters": (
            "the strictly one-factor contrast -- same boxes, sizes and "
            "total area, only placement -- so a survivor is about ANATOMY "
            "rather than about cropping"
        ),
    },
    "anatomy_vs_whole": {
        "delta": 0.0121,
        "reading": (
            "essentially nothing: cropping to anatomy is AS GOOD AS the "
            "whole face, not better"
        ),
    },
    "the_shape": (
        "discarding the periphery costs nothing; discarding the right "
        "regions costs a lot. A pattern with a null in it is harder to "
        "produce by accident than a lone number"
    ),
    "against_the_literature": (
        "Charm (2025) found random selection beating saliency, entropy "
        "and gradient selection on small aesthetic datasets, and this "
        "project's own scheme axis was null four times. Placement "
        "separating here is the opposite"
    ),
    "cautions": {
        "precedent": (
            "Swin's 3.66x failed condition 1 at 1 of 5 and Road A's two "
            "highest margins also failed; 2.6x is BELOW that margin"
        ),
        "not_a_new_best_arm": (
            "0.2428 does not beat Road A's 0.2520 -- it approaches it by "
            "a different route"
        ),
    },
    # **[MEASURED 2026-08-14] The criterion ran. All three withdrew, and
    # the caution was the right one.**
    "outcome": {
        "measured": "2026-08-14",
        "results": {
            "anatomy_vs_random": {"margin": 2.55, "excluding": "2/5",
                                  "claimable": False},
            "random_vs_whole": {"margin": 1.95, "excluding": "0/5",
                                "claimable": False},
            "crop_vs_whole": {"margin": 0.51, "excluding": "0/5",
                              "claimable": False},
        },
        "the_registered_null_fires": (
            "region-cropped input does not demonstrably improve cleft "
            "assessment at n=237 -- registered in the brief §4 before any "
            "arm ran, as the likely outcome on precedent"
        ),
        "what_survives_descriptively": (
            "anatomy crops MATCH a whole frame (0.2428 vs 0.2307, and "
            "crop_vs_whole is emphatically null at 0.51x), so discarding "
            "~60% of the image costs nothing measurable; random crops "
            "fall 0.077 short and reach 2 of 5 rather than 0 of 5 -- more "
            "than either other contrast managed, and not a claim"
        ),
        "the_shape_needs_its_null": (
            "throwing away the periphery is FREE, throwing away the right "
            "regions is NOT -- and the second half is exactly what the "
            "cohort cannot prove. The null is what makes the pair "
            "coherent rather than a lone number"
        ),
    },
}


def _regioncrop_claim_pairs() -> list[dict]:
    """Branch 3's four contrasts, in ``ladder.paired_claim_pairs`` shape.

    No recorded deltas: these are registered BEFORE any arm runs, so
    there is no condition-2 margin to carry yet. The field is omitted
    rather than filled with a placeholder -- an absent margin is
    honest, a zero one would be read as measured.
    """
    from . import ladder

    arms = {arm["key"]: arm for arm in regioncrop_arms()}
    ran = set(REGION_CROP_ARMS_OBSERVED["values"])
    pairs = []
    for contrast in REGION_CROP_CONTRASTS:
        # Only contrasts whose BOTH arms have run. graph_vs_concat waits
        # on the node path (GRAPH_NODE_PATH_IS_NOT_WIRED); it appears here
        # the moment those arms record values, without anyone editing a
        # list.
        if not {contrast["a"], contrast["b"]} <= ran:
            continue
        a, b = arms[contrast["a"]], arms[contrast["b"]]
        if a["seeds"] != b["seeds"]:
            raise ValueError(
                f"{contrast['key']}: {a['key']} runs {a['seeds']} seeds "
                f"and {b['key']} runs {b['seeds']} -- a paired contrast "
                "needs both arms on the same seeds"
            )
        pairs.append({
            "key": f"regioncrop__{contrast['key']}",
            "question": contrast["question"],
            "varies": contrast["varies"],
            "a": f"roadb_p7c_arm_{a['key']}",
            "b": f"roadb_p7c_arm_{b['key']}",
            "seeds": list(ladder.SEED_POOL[: a["seeds"]]),
        })
    return pairs


def _paired_claim_stem(arm: dict) -> str:
    """Each arm's own config stem -- no routing table.

    PAIRED_CLAIM_DUPLICATE_RUNS records why: both cells that looked like
    they needed routing have their own runs, and using them keeps every
    family at one code SHA.
    """
    short = "imagenet" if arm["init"] == "imagenet" else "masked"
    return f"roadb_p7_arm_{arm['backbone']}_{short}_{arm['resolution']}"


#: **[FOUND AND DELETED 2026-08-12] A run directory whose config stem and
#: job id disagreed -- and nothing in the apparatus checks that they
#: agree.**
#:
#: ``roadb_p7_arm_swin_b_masked_224__30cfbfe2__roadb-p7-arm-swin-b-masked
#: -768`` existed: a job launched under the **-768** job id that actually
#: ran the **224** config. Identified from the run's OWN config, which
#: declares ``embeddings_v1/swin_b__scut_masked__g1`` -- Road A's 224
#: set. Corroborated independently: swin masked-768's config still
#: carries placeholders, so it could not have run at all. Deleted, so
#: nothing declares it and no later reader takes it for a 768 result.
#:
#: **Why this shape is dangerous rather than untidy.** The run-directory
#: contract is ``<config-stem>__<sha8>__<job-id>`` (PLAN 2.4), so the
#: name carries the identity TWICE. Here the two halves disagreed, and
#: every downstream consumer reads one half or the other: a human
#: skimming a listing reads the job id, and a declared path is matched by
#: the whole string. A 224 result filed under a 768 job id would have
#: entered the record as the missing 768 point -- a plausible-looking
#: wrong comparison, in a family whose whole claim is its shape. It is
#: the ``framing`` failure again: the artifact looked like what was
#: expected, so nothing questioned it.
#:
#: **A guard is cheap and does not touch frozen code.** The check is
#: static: for every declared input path under ``runs/``, split the
#: directory name on ``__`` and require the job id to be consistent with
#: the config stem (the job ids here are the stem with underscores
#: hyphenated and the prefix dropped). That runs over shipped configs in
#: the test suite -- the moment a mistyped launch would enter the record
#: as a declaration, which is the moment it matters. ``harness.py``
#: creates run directories and is FROZEN, so the runtime side is out of
#: scope; the declaration side is where the damage would land anyway.
#: **Not built here** -- the instruction was paths and stop.
MISTYPED_LAUNCH_DELETED = {
    "found": "2026-08-12",
    "directory": (
        "roadb_p7_arm_swin_b_masked_224__30cfbfe2__"
        "roadb-p7-arm-swin-b-masked-768"
    ),
    "what_it_was": "a job under the -768 job id that ran the 224 config",
    "identified_by": (
        "the run's own config declaring embeddings_v1/"
        "swin_b__scut_masked__g1 -- Road A's 224 set"
    ),
    "corroborated_by": (
        "swin masked-768's config still carries placeholders, so it "
        "could not have run"
    ),
    "action": "deleted, so nothing declares it",
    "why_dangerous": (
        "the run-directory name carries the identity TWICE "
        "(<config>__<sha8>__<job-id>) and consumers read different "
        "halves -- a human skims the job id, a declaration matches the "
        "whole string. A 224 result filed under a 768 job id would have "
        "entered the record as the missing 768 point, in a family whose "
        "whole claim is its shape"
    ),
    "nothing_checks_agreement": (
        "no gate compares the stem half against the job-id half; this "
        "was caught by reading the run's config, not by the apparatus"
    ),
    "proposed_guard": (
        "static, over shipped configs: for every declared path under "
        "runs/, split the directory on '__' and require the job id to "
        "be consistent with the config stem. It fires when a mistyped "
        "launch would enter the record as a declaration. harness.py is "
        "frozen, so the runtime side is out of scope -- and the "
        "declaration side is where the damage lands"
    ),
    # [BUILT 2026-08-12] cleft/run_names.py + two tests in
    # test_smoke_run.py. **The rule as specified could not ship**: "job
    # id == stem with underscores hyphenated" holds for 22 of the 83 run
    # directories this repo declares and FAILS for 61, because job ids
    # legitimately abbreviate (p6-pt-swin-g1-v2), carry attempt suffixes
    # (-2, -v3) and variant markers (-unfused). A guard firing on 61
    # legitimate names gets switched off, and a switched-off guard is
    # worse than none -- so it was measured before it was written.
    "built": {
        "date": "2026-08-12",
        "where": "cleft/run_names.py, swept in test_smoke_run.py",
        "rule": (
            "a job id may DROP what the stem says and may not CONTRADICT "
            "it: per closed axis (resolution, backbone, geometry, init, "
            "scheme), values named in the job id must be named in the stem"
        ),
        "why_not_the_literal_rule": (
            "strict equality holds for 22 of 83 declared run directories "
            "and fails for 61 legitimate abbreviations -- measured first; "
            "a guard that fires on 61 gets switched off"
        ),
        "verified": (
            "0 contradictions over all 83 declared directories, and the "
            "deleted one is caught on the resolution axis (768 vs 224)"
        ),
    },
    "status": "BUILT 2026-08-12 -- the by-eye catch is now permanent",
}


#: **[MEASURED 2026-08-12] Condition 1 over all twenty resolution
#: contrasts: THREE survive, seventeen withdraw. All three survivors are
#: ViT falling from 224.**
#:
#: | contrast                      | margin | excl. | verdict   |
#: |-------------------------------|--------|-------|-----------|
#: | vit_b16 masked   224->512     |  8.57x |  5/5  | CLAIMABLE |
#: | vit_b16 imagenet 224->512     |  7.38x |  5/5  | CLAIMABLE |
#: | vit_b16 imagenet 224->768     |  6.91x |  5/5  | CLAIMABLE |
#: | swin_b  imagenet 224->512     |  3.66x |  1/5  | withdrawn |
#: | swin_b  masked   224->512     |  3.08x |  1/5  | withdrawn |
#: | 15 more                       | 2.07x-0.17x |  | withdrawn |
#:
#: **Not the project's first survivors -- the first COHERENT ones.**
#: ``ladder.LADDER_PAIRED_AUDIT`` produced one survivor in 26 on
#: 2026-08-04, and it travels with two caveats: its baseline is 0.0065 (a
#: near-zero-correlation cell) and the same comparison at the other
#: geometry gives 0.44x at 0 of 5. Road A's total across every phase was
#: one survivor in thirty. These three are one backbone, one direction,
#: three separate contrasts -- a pattern rather than a lone cell.
#:
#: **And they sit in a margin range Road A never reached.** Road A's two
#: HIGHEST margins both failed (4.69x at 2/5, 4.34x at 4/10) and its
#: survivor ranked third at 3.83x; margin does not predict condition 1.
#: These are 6.91x to 8.57x -- the effects are simply much larger than
#: anything Road A was comparing, which is why they survive a criterion
#: that withdrew almost everything else.
#:
#: **The draw spread does NOT undermine the two imagenet survivors, and
#: DOES touch the masked one.** ViT's checkpoints are draws (0.055 at
#: 224, 0.027 at 512), which looks like it should contaminate any ViT
#: result. For the imagenet contrasts it cannot: they consume the
#: DECLARED ImageNet snapshot at every resolution, so the weights are one
#: fixed artifact and resolution is the only factor. For **vit masked
#: 224->512 it does apply** -- the two arms read two different
#: pretraining cells, each its own draw, so that contrast confounds
#: resolution with the draw. Its -0.185 is ~3.4x the larger spread, so
#: the finding likely survives the confound, but it is confounded and
#: must be quoted that way. The clean pair carries the claim.
#:
#: (The confound is ViT-specific: only ViT's pretraining is
#: nondeterministic. Every other masked contrast also spans two
#: checkpoints, but those are a fixed function of their cell.)
#:
#: Margins here are the RUN's, computed from the full-precision vectors;
#: PHASE_7_TWENTY_TWO_ARMS' recorded means are 4-dp, so re-deriving from
#: them gives values differing in the second decimal (8.71 vs 8.57). The
#: run's are authoritative.
PHASE_7_PAIRED_RESULTS = {
    "measured": "2026-08-12",
    "run": "roadb_p7_paired_resolution, 20 pairs, 170 vectors",
    "survived": 3,
    "withdrawn": 17,
    "survivors": [
        {"contrast": "vit_b16__scut_masked 224->512", "margin": 8.57,
         "excluding": "5/5", "delta": -0.1848, "confounded_by_draw": True},
        {"contrast": "vit_b16__imagenet 224->512", "margin": 7.38,
         "excluding": "5/5", "delta": -0.1626, "confounded_by_draw": False},
        {"contrast": "vit_b16__imagenet 224->768", "margin": 6.91,
         "excluding": "5/5", "delta": -0.1583, "confounded_by_draw": False},
    ],
    "all_survivors_are": "ViT falling from 224 -- one backbone, one direction",
    "not_the_first_ever": (
        "ladder.LADDER_PAIRED_AUDIT had ONE survivor in 26, caveated by a "
        "0.0065 baseline and 0.44x at 0/5 at the other geometry. Road A's "
        "total was one in thirty. These three are the first COHERENT set"
    ),
    "margin_range_road_a_never_reached": (
        "Road A's two highest margins failed (4.69x at 2/5, 4.34x at "
        "4/10) and its survivor ranked third at 3.83x. These are "
        "6.91x-8.57x: the effects are much larger, not luckier"
    ),
    "draw_spread_does_not_enter_the_clean_pair": (
        "the imagenet contrasts consume the DECLARED snapshot at every "
        "resolution, so the weights are one fixed artifact and "
        "resolution is the only factor"
    ),
    "draw_spread_DOES_touch_the_masked_survivor": (
        "vit masked 224->512 reads two different pretraining cells, each "
        "its own draw, so it confounds resolution with the draw. -0.185 "
        "is ~3.4x the larger spread (0.055), so it likely survives -- "
        "but it is confounded and is quoted that way. ViT-specific: only "
        "ViT's pretraining is nondeterministic"
    ),
    "margins_are_the_runs": (
        "computed from full-precision vectors; re-deriving from the 4-dp "
        "recorded means differs in the second decimal (8.71 vs 8.57)"
    ),
}


#: **[MEASURED 2026-08-12] The fifth time seed thresholds and the cohort
#: have disagreed, and the first time the disagreement was PREDICTED.**
#:
#: Road B's graph declines read as resolvable on their own seed SDs --
#: agnet masked at 4.1 sigma, srgnn masked at 3.8, srgnn imagenet at 2.0
#: (PHASE_7_TWENTY_TWO_ARMS). All three withdrew. That is not three
#: separate results; it is one more instance of a pattern the project has
#: now met at every scale it has looked:
#:
#: 1. Phase 7B's search: 2.37x, 1 of 5 (``COHORT_CANNOT_RESOLVE``).
#: 2. Phase 7C's augmentation: nine verdicts, 0 of 45 intervals
#:    (``phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT``) -- including 3.62x.
#: 3. The 47-record audit: every claimable field computed from condition
#:    2 alone (``ladder.CLAIMS_REST_ON_HALF_THE_CRITERION``).
#: 4. The ladder audit: 25 of 26 withdrawn, and the two HIGHEST margins
#:    among the failures (``ladder.LADDER_PAIRED_AUDIT``).
#: 5. Road B's resolution axis: 17 of 20, this record.
#:
#: **What is new is that it was called in advance.** PHASE_7_SWIN_512
#: registered, before the run, that seed precision was not the binding
#: uncertainty and that Road A's history was the reason to expect the
#: criterion to bite. It bit, on the cell that was named. A pattern
#: predicted before its fifth instance is a mechanism, not a coincidence:
#: the per-seed interval's width is set by patient-level disagreement
#: between two prediction vectors, which the ratio of two seed SDs does
#: not see.
SEED_THRESHOLDS_DO_NOT_PREDICT_THE_COHORT = {
    "measured": "2026-08-12",
    "instance": 5,
    "road_b_withdrawals_that_looked_resolvable": {
        "agnet__scut_masked 224->768": "4.1 sigma on seed SDs, withdrawn",
        "srgnn__scut_masked 224->768": "3.8 sigma, withdrawn",
        "srgnn__imagenet 224->768": "2.0 sigma, withdrawn",
    },
    "prior_instances": (
        "phase 7B's search (2.37x, 1/5); phase 7C's nine (0 of 45, "
        "including 3.62x); the 47-record audit; the ladder audit (25 of "
        "26, the two highest margins among the failures)",
    ),
    "what_is_new": (
        "it was called IN ADVANCE -- PHASE_7_SWIN_512 registered before "
        "the run that seed precision was not the binding uncertainty, "
        "and named Road A's history as the reason to expect the "
        "criterion to bite. It bit on the named cell"
    ),
    "mechanism": (
        "the per-seed interval's width is set by patient-level "
        "disagreement between two prediction vectors; the ratio of two "
        "seed SDs does not see it"
    ),
    "reporting_rule": (
        "record as ONE pattern, not as N separate withdrawals -- three "
        "graph withdrawals here are one instance of a five-instance "
        "regularity"
    ),
}


#: **[MEASURED 2026-08-12] The duplicate runs were a determinism check
#: nothing was designed to produce -- and both pairs are byte-identical.**
#:
#: Fifteen prediction files, two pairs, two accidents of scheduling:
#:
#: * **SR-GNN masked-512, ten seeds**: the arm at ``a54cdfae`` against the
#:   band sweep at ``30cfbfe2``. Byte-identical across a commit range, in
#:   the noisiest regime the project has -- trainable graph layers and
#:   BatchNorm, the third regime ``factory.BACKBONES`` records.
#: * **Swin masked-224, five seeds**: Road A at ``4cb62c05`` against Road
#:   B at ``30cfbfe2``. Byte-identical across a much wider commit range
#:   AND across two different embedding artifacts that are themselves
#:   bitwise identical (PHASE_7_ANCHOR_INDEPENDENCE.swin_masked_224).
#:
#: **What it establishes, and neither was reachable any other way.**
#:
#: **1. The nondeterminism is CONFINED to pretraining.** Extraction was
#: measured bitwise earlier (PHASE_7_EXTRACTION_PLAN.extraction_
#: determinism); head fitting is measured bitwise now. So the only
#: nondeterministic step in the pipeline is ViT pretraining's fused
#: attention (UNFUSED_WAS_NOT_SUFFICIENT), and **the arms are
#: reproducible even where the checkpoints feeding them are not.** A
#: reader who knows ViT's cells are draws might reasonably doubt every
#: downstream number; this bounds the doubt to exactly one step.
#:
#: **2. The graph band is SEED variance, not run-to-run instability.**
#: 0.041452 (PHASE_7_GRAPH_BAND_RESOLVED) could in principle have been
#: one seed behaving differently on different runs -- which would make it
#: a measure of flakiness rather than of seed sensitivity, and would
#: invalidate it as a planning instrument. The duplicate settles it: one
#: seed run twice gives the same bytes, so the spread is what DIFFERENT
#: seeds do. The band means what it was quoted to mean.
DUPLICATE_RUNS_WERE_A_DETERMINISM_CHECK = {
    "measured": "2026-08-12",
    "files": 15,
    "result": "byte-identical on both pairs",
    "pairs": {
        "srgnn_masked_512": {
            "seeds": 10,
            "across": "a54cdfae (arm) vs 30cfbfe2 (sweep)",
            "regime": (
                "trainable graph layers and BatchNorm -- the noisiest "
                "regime in the project"
            ),
        },
        "swin_masked_224": {
            "seeds": 5,
            "across": "4cb62c05 (Road A) vs 30cfbfe2 (Road B)",
            "regime": (
                "two roads, a wide commit range, and two different "
                "embedding artifacts that are bitwise identical"
            ),
        },
    },
    "establishes": {
        "nondeterminism_is_confined_to_pretraining": (
            "extraction measured bitwise earlier, head fitting bitwise "
            "now -- so ViT's fused attention is the only "
            "nondeterministic step, and the ARMS are reproducible even "
            "where the checkpoints feeding them are not"
        ),
        "the_graph_band_is_seed_variance": (
            "0.041452 could have been one seed behaving differently on "
            "different runs, which would make it flakiness rather than "
            "seed sensitivity. One seed run twice gives the same bytes, "
            "so the spread is what DIFFERENT seeds do -- the band means "
            "what it was quoted to mean"
        ),
    },
    "nothing_was_designed_to_produce_it": (
        "two scheduling accidents (a cell run as both sweep and arm; a "
        "reuse arm run on both roads) turned into the check, at no cost "
        "-- the pattern MASKED_SCUT_REBUILD_WAS_A_DETERMINISM_CHECK set"
    ),
}


#: **[ESTABLISHED 2026-08-12] What Road B answers, at the strongest
#: standard the project has -- and what it does not.**
#:
#: The branch existed because staging at 224 discards ~99% of the
#: available pixels (10.8x linear). Twenty-four arms were designed,
#: twenty-two ran, twenty resolution contrasts were tested against PLAN
#: 4.3's FULL criterion. The answer has two parts and only the first is a
#: claim:
#:
#: **1. CLAIMABLE: through a frozen ViT-B/16, more pixels HURT.** Three
#: contrasts survive both conditions, all the same backbone falling from
#: its 224 pretraining resolution: imagenet -0.163 (224->512) and -0.158
#: (224->768), masked -0.185 (224->512). This is stronger than the
#: registered null, which anticipated flatness. It is not "resolution
#: does not help" -- it is "for this architecture, resolution measurably
#: harms", and it coheres with the mechanism the road already measured
#: (BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER: a learned position grid
#: interpolated at build time).
#:
#: **2. NOT CLAIMABLE: everything else, in either direction.** Seventeen
#: of twenty withdraw, including every graph decline and the single
#: apparent gain (PHASE_7_SWIN_512.outcome). So Road B establishes NO
#: case where recovering the pixels helps -- and the one cell that looked
#: like it does not survive the cohort.
#:
#: **The premise, answered:** recovering the discarded pixels does not
#: improve cleft assessment through frozen backbones at n=237, and for
#: ViT it demonstrably hurts. The pre-registered null fired in its
#: stronger form.
#:
#: **What this does NOT say**, and the write-up must not drift into: not
#: that the pixels are uninformative (frozen backbones, frozen heads, one
#: cohort of 237); not that resolution never helps (Swin's non-collapse
#: is a supported prediction, and an unresolved null is not a refutation);
#: not that the trained regime behaves this way (Phase 6 measured trained
#: arms flat, not falling -- SWIN_RESOLUTION_PREDICTION.confirmed).
#:
#: **The cohort is the finding underneath the finding.** Seventeen of
#: twenty unresolvable echoes Road A's twenty-nine of thirty
#: (``ladder.COHORT_CANNOT_RESOLVE``), now on a second road with a
#: different axis. The sharpest sentence Road B can offer a reader is not
#: about resolution at all: this cohort resolves effects of ViT-collapse
#: size and nothing smaller, and that is what a next dataset has to beat.
ROAD_B_ESTABLISHED = {
    "established": "2026-08-12",
    "premise": "224 staging discards ~99% of available pixels (10.8x linear)",
    "arms": {"designed": 24, "ran": 22, "contrasts_tested": 20},
    "claimable": {
        "finding": "through a frozen ViT-B/16, more pixels HURT",
        "contrasts": 3,
        "deltas": {
            "imagenet 224->512": -0.1626,
            "imagenet 224->768": -0.1583,
            "masked 224->512": -0.1848,
        },
        "stronger_than_registered": (
            "the null anticipated FLATNESS; the measurement is a "
            "claimable decline for one architecture"
        ),
        "mechanism_already_measured": (
            "BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER -- a learned "
            "position grid interpolated at build time"
        ),
    },
    "not_claimable": (
        "17 of 20 in either direction, including every graph decline and "
        "the single apparent gain -- so NO case where the pixels help"
    ),
    "premise_answered": (
        "recovering the discarded pixels does not improve cleft "
        "assessment through frozen backbones at n=237, and for ViT it "
        "demonstrably hurts"
    ),
    "does_not_say": (
        "NOT that the pixels are uninformative (frozen backbones, frozen "
        "heads, one cohort of 237); NOT that resolution never helps (an "
        "unresolved null is not a refutation); NOT that the TRAINED "
        "regime behaves this way -- Phase 6 measured trained arms flat"
    ),
    "the_finding_underneath": (
        "17 of 20 unresolvable echoes Road A's 29 of 30 on a second road "
        "with a different axis: this cohort resolves effects of "
        "ViT-collapse size and nothing smaller, which is what a next "
        "dataset has to beat"
    ),
}


#: **[DECIDED 2026-08-12] Phase 7 is reportable NOW, and nothing further
#: is worth running.** Both halves answered on the evidence rather than
#: on tidiness.
#:
#: **Can it be reported without the two pending arms? YES, and not
#: because they are minor -- because nothing reportable touches them.**
#:
#: * Every claimable result is ViT: two contrasts inside the COMPLETE
#:   imagenet family, and one (masked 224->512) that does not involve a
#:   768 arm at all. The headline is untouched by both absences.
#: * The families the pending arms complete are precisely the two that
#:   could not have carried a claim anyway. **vit masked** is DESCRIPTIVE
#:   by decision (PHASE_7_ARM_STRUCTURE.vit_reporting), registered before
#:   any arm ran -- its shape was never going to be claimed. **swin
#:   masked**'s only computable contrast already failed condition 1 at
#:   3.08x, 1 of 5.
#: * Under PHASE_7_SHAPE_STATISTIC the per-family claim is the endpoint
#:   contrast, so those two families simply have no claim statistic yet
#:   and are reported with two points and a stated absence. That is a
#:   smaller gap than it looks: neither could have produced a claim.
#:
#: So the report states the two cells as unrun, says what they would and
#: would not change, and does not wait. If they land, they extend a
#: descriptive family and a withdrawn contrast.
#:
#: **Is anything else worth running? NO -- and the reason is the cohort,
#: not the budget.**
#:
#: * **The two -3 pretraining cells**: optional completeness. Two of the
#:   heaviest runs in the project (768 transformers, whole GPU) to fill
#:   points that cannot become claims. Run them if the design's symmetry
#:   is wanted on paper; not for the finding.
#: * **The G2 Swin replication, WITHDRAWN as advice.** It was registered
#:   when the open question was "is the gain real", and a second
#:   operating point could have answered that. Condition 1 has since
#:   answered something stronger: this cohort cannot resolve a gain of
#:   that size AT ALL. A replication at the same n=237 meets the same
#:   wall, so it would buy a second unresolved null. Recommending it now
#:   would be chasing the cell the criterion just declined.
#: * **The free duplicate check IS worth doing, and costs nothing.** Two
#:   cells ran twice (PAIRED_CLAIM_DUPLICATE_RUNS): srgnn masked 512 as
#:   the sweep AND as its own arm, at two code SHAs; swin masked 224 on
#:   both roads. Comparing those prediction files is an end-to-end
#:   determinism check on the head-fit path ACROSS a commit range -- no
#:   GPU, no new fits, files already on disk. It is the only remaining
#:   measurement with a real chance of saying something new.
#: * **The unconsumed non-square artifacts** answer a DIFFERENT question
#:   (padding and aspect, not resolution) and would extend the road
#:   rather than close it. The pad pair is the one contrast that isolates
#:   padding cleanly, and only a ViT-collapse-sized effect would resolve
#:   -- worth naming as available, not worth holding the close for.
PHASE_7_CLOSING = {
    "decided": "2026-08-12",
    "reportable_now": True,
    "why_the_pending_arms_cannot_change_it": (
        "every claimable result is ViT -- two contrasts inside the "
        "complete imagenet family and one masked 224->512 that involves "
        "no 768 arm. The two families the pending cells complete are the "
        "two that could not have carried a claim: vit masked is "
        "DESCRIPTIVE by pre-run decision, and swin masked's only "
        "computable contrast already failed at 3.08x, 1 of 5"
    ),
    "how_to_report_them": (
        "state both cells as unrun, say what they would and would not "
        "change, and do not wait -- if they land they extend a "
        "descriptive family and a withdrawn contrast"
    ),
    "nothing_further_is_worth_running": {
        # [DECIDED 2026-08-12, the maintainer] Not "optional" any more: they
        # are NOT BEING RUN. Recorded as a decision with its reasoning
        # rather than left as a pending state, so a later reader does
        # not read two blanks as an interruption.
        "minus_3_cells": (
            "NOT BEING RUN (decided 2026-08-12) -- two of the heaviest "
            "runs in the project for points that cannot become claims; "
            "was 'optional completeness'"
        ),
        "g2_replication": (
            "WITHDRAWN as advice: registered when the question was 'is "
            "the gain real', and condition 1 has since answered that "
            "this cohort cannot resolve a gain of that size at all. A "
            "second cell at n=237 buys a second unresolved null"
        ),
        "non_square_artifacts": (
            "a DIFFERENT question (padding and aspect) that would extend "
            "the road rather than close it -- available, not blocking"
        ),
    },
    "the_one_thing_worth_doing": (
        "the free duplicate check: srgnn masked 512 ran as the sweep AND "
        "as its own arm at two code SHAs, and swin masked 224 ran on "
        "both roads. Comparing those prediction files is an end-to-end "
        "determinism check on the head-fit path across a commit range -- "
        "no GPU, no new fits, files already on disk"
    ),
}


#: **[STRUCTURE 2026-08-14] Road B Branch 3 -- region-cropped input, on
#: the magnification justification. The arm set, corrected.**
#:
#: The brief (``ROADB_REGION_CROP_BRIEF.md``, outside the repository)
#: rebuilds Branch 3 after
#: its original justification was refuted: the maps do NOT prefer the
#: border (1.065x chance, ``phase8.FRAMING_NOT_ANATOMY``). What survives
#: is magnification -- the philtral column occupies ~20 px of a 10.8x
#: downsampled face, and cropping a region to the backbone's input gives
#: that structure the full frame.
#:
#: **Two matched-pipeline defects were found in the brief, both the same
#: class: §2.3 promises arms "differing in one factor" and §3's table
#: breaks it.**
#:
#: **Defect 1 (accepted 2026-08-14): the control.** §3 nominated Road B's
#: existing 768 ViT arm, which reads ``roadb_768_square_g1_v1``, while
#: the crops come from **768 non-square G2** (§2.1, because at G1 the
#: lateral regions fall 76.4% on white). That contrast varies cropping,
#: aspect AND geometry. Fixed by adding a whole-frame ViT arm at the
#: crops' OWN staging. The square-G1 alternative is ruled out by the
#: brief's own measurement.
#:
#: **Defect 2 (FLAGGED, awaiting ratification): the combination
#: contrast.** §3 makes the graph arm SR-GNN and the concat arm ViT, so
#: "does message passing add anything" varies backbone AND combination.
#: The obvious fix -- one shared ViT crop encoder feeding both -- has a
#: cost that rules it out: SR-GNN's graph layers would then be randomly
#: initialised rather than warm-started from a paired checkpoint, and
#: ``graph_cleft.MEASURED_GRAPH_SEED_BAND`` explicitly does NOT cover
#: graph-layer initialisation. That would be a fourth regime needing its
#: own band. **So the fix is a matched partner instead**: an SR-GNN
#: crops-to-concat arm, which costs one head fit and NO new extraction
#: (it reads the same crop set as the graph arm).
#:
#: **Init is ImageNet throughout, decided for factor-cleanliness.** The
#: control is ImageNet (the crop-vs-whole contrast must match on init),
#: which forces the ViT crop arms; keeping the graph arms there too
#: makes every contrast in the branch one-factor. **The cost is
#: recorded**: Road A measured SR-GNN's imagenet arm at SD 0.0630
#: against scut_masked's 0.0211 (``ladder.STAGE_D1_AT_G1``), because
#: imagenet graph arms carry graph-layer init in their seed variance.
#: The graph arms are the noisy ones and their band is not the 0.041452
#: one -- each reports its own SD (4.12.1), as always.
REGION_CROP_STRUCTURE = {
    "designed": "2026-08-14",
    "brief": "ROADB_REGION_CROP_BRIEF.md (NOT IN REPO)",
    "justification": {
        "refuted": (
            "the border reading -- maps sit at 1.065x chance, "
            "phase8.FRAMING_NOT_ANATOMY. Must not be repeated"
        ),
        "holds": (
            "magnification: the philtral column occupies ~20 px of a "
            "10.8x downsampled face; a crop resized to the backbone "
            "input gives that structure the full frame"
        ),
    },
    "regions": {
        "source": "phase7c.PROTECTED_REGIONS",
        "names": 15,
        "boxes": 22,
        "expansion": "8 midline + 7 bilateral x 2 = 22 of 27",
        "geometry": "g2 non-square -- content fraction exactly 1.0000",
        "excluded": "glabella, medial_canthus, lateral_orbit",
    },
    "staging": {
        "artifact": "roadb_768_nonsquare_g2_v1",
        "already_built": (
            "one of the ten, hashed and residual-gate verified -- the "
            "branch needs NO new staging"
        ),
        "why_768": (
            "a region box is ~1/6 of the crop width: ~28 px from 224 "
            "staging, ~96 px from 768. Only 21 of 237 frontal sources "
            "fall below 768 on the longer side"
        ),
        "backbone_input": (
            "stays 224 -- Road B measured what leaving it costs. The "
            "magnification is in the CROP, not the input size"
        ),
    },
    "defects_found_in_the_brief": {
        "control_mismatch": (
            "ACCEPTED 2026-08-14 -- §3's control read square G1 against "
            "crops from non-square G2: three factors at once. Fixed by "
            "a whole-frame ViT arm at the crops' own staging"
        ),
        "combination_mismatch": (
            "FLAGGED, awaiting ratification -- §3 makes the graph arm "
            "SR-GNN and the concat arm ViT, so the combination contrast "
            "varies backbone too. Fixed by a matched SR-GNN concat arm "
            "(one head fit, no new extraction) rather than by sharing a "
            "ViT encoder, which would leave the graph layers randomly "
            "initialised -- a regime the measured graph band does not "
            "cover"
        ),
    },
    "init_is_imagenet_throughout": {
        "why": (
            "the control is ImageNet, so crop-vs-whole must match on "
            "init; keeping the graph arms there makes every contrast "
            "one-factor"
        ),
        "cost": (
            "Road A measured SR-GNN imagenet at SD 0.0630 against "
            "scut_masked's 0.0211 (ladder.STAGE_D1_AT_G1) -- imagenet "
            "graph arms carry graph-layer init in their seed variance"
        ),
    },
    "precompute_once": (
        "frozen backbone, head-only fit, extraction ONCE -- as every "
        "other Road B arm. §2.2's per-epoch cost sentence describes the "
        "augmentation regime's live path, which this branch does not "
        "use; corrected in the brief so nobody builds it by inference"
    ),
    "the_real_build_cost": (
        "APPNP and attention pooling are reusable; the 22-crops -> "
        "backbone -> per-crop pooled vectors -> graph nodes consumption "
        "path in FRONT of them is new. SR-GNN's existing path takes 27 "
        "regions from the FEATURE MAP at a fixed 42x42 grid, which is a "
        "different operation. Costed as a build, not a transplant"
    ),
}


#: **[MEASURED 2026-08-14] The frozen containment check is square-only,
#: and Branch 3 is the first thing to stage non-square AND map patches.**
#:
#: ``mapping.assert_inside_content`` takes ``size = staged.image.shape[0]``
#: and bounds BOTH axes by it -- correct for every image the apparatus has
#: ever seen, because Road A staged square. Measured on a landscape
#: non-square frame::
#:
#:     stage_non_square(2424x2173, 768) -> (688, 768)
#:     mapping.map_patches(...) -> MappingError: patch 12 at
#:         (596, 34, 166, 111) falls outside the 688x688 image
#:
#: x + w = 762 <= 768, the real width. **The box is correct; the check is
#: square.** The mapping ARITHMETIC (``content_box_to_pixels``) is
#: aspect-correct throughout -- it goes through the content box -- so only
#: the assertion is wrong.
#:
#: **The failure shape is the worst available.** ``scut.ar_distribution``
#: records exactly ONE landscape crop in the cohort (aspect 1.0986; 236 of
#: 237 portrait), so a crop extraction would have processed 236 patients
#: and died on the last. And the obvious "fix" -- skip the patient that
#: raises -- would silently make the cohort 236.
#:
#: ``mapping.py`` is frozen, so ``roadb_regioncrop`` does what
#: ``roadb_staging.stage_non_square`` did for the same reason: reuse the
#: frozen arithmetic, supply the containment rule the frozen one gets
#: wrong. Bounded per axis, pad checked rather than assumed.
FROZEN_CONTAINMENT_IS_SQUARE_ONLY = {
    "measured": "2026-08-14",
    "where": "geometry/mapping.py assert_inside_content",
    "what": "size = staged.image.shape[0] bounds BOTH axes",
    "reproduction": (
        "stage_non_square(2424x2173, 768) -> (688, 768); map_patches "
        "raises for patch 12 at (596, 34, 166, 111), whose x+w=762 is "
        "inside the real width 768"
    ),
    "the_arithmetic_is_fine": (
        "content_box_to_pixels goes through the content box and is "
        "aspect-correct -- only the assertion assumes square"
    ),
    "failure_shape": (
        "exactly ONE landscape crop in the cohort (aspect 1.0986, "
        "scut.ar_distribution: 236 of 237 portrait), so the run would "
        "process 236 patients and die on the last -- and 'skip the one "
        "that raises' would silently make the cohort 236"
    ),
    "fix": (
        "roadb_regioncrop.region_boxes reuses the frozen arithmetic and "
        "bounds x by the width and y by the height, the same pattern "
        "roadb_staging.stage_non_square used against frozen staging"
    ),
    "found_by": (
        "R10 -- reading the frozen API before calling it in a regime it "
        "had never run in, then measuring rather than reasoning"
    ),
}


#: **[MEASURED 2026-08-14] The magnification premise, measured through the
#: shipped crop path -- and it is stronger than the brief predicted.**
#:
#: §2.4 estimated a region box at "~96 px wide" from 768 staging, treating a
#: region as a uniform 1/6 of the crop width. The real anatomy boxes are
#: larger and vary: **median 149 px** (99 to 198) at 768 non-square G2. So
#: **every crop UPSAMPLES to the 224 backbone input**, 1.13x to 2.26x -- no
#: region is shrunk on its way in, which the 1/6 estimate did not establish.
#:
#: **What this does and does not settle.** It settles that the branch is not
#: feeding the backbone less than it had: a whole 768 frame downsamples to
#: 224 at 3.4x, while these crops arrive magnified. It does NOT settle that
#: the extra pixels carry usable signal -- that is what the sheet asks by
#: eye and what the arms measure. The premise got bigger, not true.
#:
#: The brief's 96 is amended in place with a dated block; the reasoning
#: that produced it stands and only the value was wrong.
CROP_MAGNIFICATION_MEASURED = {
    "measured": "2026-08-14",
    "through": "roadb_regioncrop.native_crop_sizes, the shipped crop path",
    "brief_predicted_px": 96,
    # **[CORRECTED 2026-08-14, by the sheet review] These are ONE
    # patient's 22 regions, not the cohort's range.** Measured on a
    # synthetic frame at aspect 0.897; a narrower patient yields smaller
    # crops throughout. The review found patient 205 at a 97 px MEDIAN --
    # below this within-patient minimum, and consistent, because it is a
    # different population. Named because "min 99" reads as a floor for
    # the cohort and is not one (R2, in this record's own field name).
    "measured_px": {"min": 99, "median": 149, "max": 198},
    "measured_px_population": (
        "one patient's 22 regions at aspect 0.897 -- NOT across patients. "
        "native_crop_sizes returns within-patient statistics"
    ),
    "across_patients": {
        "source": "CROP_SHEET_REVIEW_PASSED, measured on the real cohort",
        "median_px": {"patient_205": 97, "patient_89": 156},
        "reading": (
            "per-patient medians vary with aspect, so the cohort spans "
            "wider than one patient's spread suggests. The smallest "
            "observed median is 97 px -- still upsampling to 224 at 2.3x, "
            "so the premise holds a fortiori at the small end"
        ),
    },
    "why_the_estimate_was_low": (
        "it treated a region as a uniform 1/6 of the crop width; the real "
        "anatomy boxes are larger and vary"
    ),
    "every_crop_upsamples": {
        "to": 224, "factor": {"min": 1.13, "max": 2.26},
        "meaning": (
            "no region is shrunk on its way into the backbone, which the "
            "1/6 estimate did not establish"
        ),
    },
    "what_it_does_not_settle": (
        "that the extra pixels carry usable signal -- the sheet asks that "
        "by eye and the arms measure it. The premise got bigger, not true"
    ),
}


#: **[REVIEWED 2026-08-14 — PASSED. The gate, met.]** Brief §5 criterion 3:
#: no Branch 3 arm runs until a human has looked at the crops, because
#: whether a box labelled ``philtral_column`` contains a philtral column is
#: not something arithmetic can answer.
#:
#: **What was checked, and it holds.** The named boxes contain their
#: anatomy: philtrum boxes sit on the philtrum, and the left and right
#: philtral columns are present as SEPARATE patches -- the mirrored-pair
#: expansion doing what the 22-of-27 count says it does. Checked at both
#: ends of the size range rather than in the middle: **patient 89 at 156 px
#: median and patient 205 at 97 px**, so the verdict covers the smallest
#: crops as well as the typical ones. The magnification premise holds
#: visually at those sizes.
#:
#: **The one observation, and it is not a defect.** Some philtrum crops
#: include upper lip and some do not, varying per patient. The boxes are
#: fractions of the CONTENT BOX mapped into each patient's own pixels
#: (``mapping.py``'s whole reason for existing), so a patient whose anatomy
#: sits slightly higher or lower catches more or less of the neighbouring
#: structure. That is the mapping working, not a placement error.
#:
#: **What the mapping does NOT remove, stated because it bears on the
#: claim.** Normalising to the content box removes ASPECT-RATIO and SCALE
#: variation between patients. It cannot remove ANATOMICAL PROPORTION
#: variation, because the boxes are anchored to the frame rather than to
#: per-patient landmarks. So the crops are anatomical at the cohort's
#: average proportions, not landmark-precise per patient.
#:
#: **The consequence for a positive result, registered now.** If the
#: anatomy arm beats the random arm, the finding is *"crops at anatomical
#: positions help"* -- NOT *"landmark-precise anatomy helps"*, which this
#: design does not test. That is a weaker sentence than the branch might
#: otherwise be read as supporting, and it is weaker in a way that
#: strengthens it: it is what the measurement can carry. Recorded before
#: any arm runs so it is not negotiated afterwards.
CROP_SHEET_REVIEW_PASSED = {
    "reviewed": "2026-08-14",
    # [AMENDED 2026-08-14] This is gate ONE of two. It answered whether
    # the boxes sit on the right anatomy and could not answer whether that
    # anatomy survives the square resize -- CROP_SHEET_AS_FED_REVIEW_
    # PASSED is the partner, and criterion 3 needed both.
    "verdict": "PASSED -- gate 1 of 2; the boxes sit on the right anatomy",
    "partner_gate": "CROP_SHEET_AS_FED_REVIEW_PASSED",
    "did_not_cover": (
        "the square resize the backbone receives -- this sheet showed "
        "crops AS CUT by design, which is why the second exists"
    ),
    "checked": (
        "the named boxes contain their anatomy: philtrum boxes on the "
        "philtrum, and left and right philtral columns present as SEPARATE "
        "patches"
    ),
    "checked_at_both_extremes": {
        "patient_89": {"median_crop_width_px": 156},
        "patient_205": {"median_crop_width_px": 97},
        "why": (
            "the smallest crops, not just the typical ones -- a verdict "
            "taken in the middle of the range would not cover them"
        ),
    },
    "magnification": "holds visually at the measured sizes",
    "observation": (
        "some philtrum crops include upper lip and some do not, varying "
        "per patient"
    ),
    "explanation": (
        "NOT a placement error: the boxes are fractions of the content box "
        "mapped into each patient's own pixels, so anatomy sitting "
        "slightly higher or lower catches more or less of the neighbouring "
        "structure. That is the mapping working"
    ),
    "what_the_mapping_does_not_remove": (
        "normalising to the content box removes ASPECT-RATIO and SCALE "
        "variation; it cannot remove ANATOMICAL PROPORTION variation, "
        "because the boxes are anchored to the frame rather than to "
        "per-patient landmarks"
    ),
    "consequence_for_the_claim": (
        "if anatomy beats random, the finding is 'crops at anatomical "
        "positions help', NOT 'landmark-precise anatomy helps' -- which "
        "this design does not test. Registered before any arm runs"
    ),
}


#: **[DECIDED 2026-08-14, option B] The graph arm pools its crops to
#: vectors -- and this arm is NOT SR-GNN as published.**
#:
#: ``extract_features`` returned ``(5214, 2048, 7, 7)`` for the SR-GNN
#: crops, because graph backbones produce feature maps by a recorded
#: decision (``embeddings.KIND_FOR_BACKBONE_KIND``: the frozen boundary
#: for a graph model is its final map). The crop task had assumed one
#: vector per crop for every backbone. Three ways out, and the storage
#: question was the least of it:
#:
#: * **Store the maps** -- ``(237, 22, 2048, 7, 7)``. Flattened, that is
#:   100,352 dims per region, exactly the descriptor ``SeqSelfAttention``
#:   was built for, so the arm would be SR-GNN as published. But the
#:   matched concat partner would then need a **2.2M-dimensional head at
#:   n=237**, which is not an arm -- so graph-vs-concat reverts to two
#:   factors, the defect the partner exists to remove.
#: * **Pool to 2048** (TAKEN). Both SR-GNN arms sit on identical nodes,
#:   the combination contrast varies combination alone, no third kind, no
#:   49x storage.
#: * ViT for everything -- cleaner still, but discards the supervision
#:   backbone lineage.
#:
#: **Why B answers the question**: brief §2.3 asks whether message
#: passing over anatomy adds anything concatenation does not. That is
#: about the MECHANISM, not about ``SeqSelfAttention``'s dimensionality.
#:
#: **What B gives up, recorded so nobody reads this arm as a CleftGNN
#: replication**: pooling turns ``SeqSelfAttention`` into a smaller layer
#: (2048-wide input, not 100,352). The arm tests message passing over
#: anatomy-crop embeddings. It does NOT replicate or test SR-GNN's
#: published architecture, and no write-up may say it does.
#:
#: **The per_region size argument does not transfer.** That record
#: preferred the 95 MB feature map over ~2.6 GB of flattened descriptors
#: because the map is a SUPERSET -- regions are re-derived from it at
#: train time. Here nothing can be re-derived: each crop is an
#: independent backbone pass, which is the branch's premise. So the
#: choice rests on the head dimension, not on bytes.
GRAPH_ARM_POOLS_TO_VECTORS = {
    "decided": "2026-08-14",
    "option": "B -- pool each crop's feature map to a 2048 vector",
    "why_the_maps_appeared": (
        "graph backbones return feature maps by decision "
        "(embeddings.KIND_FOR_BACKBONE_KIND); the crop task assumed one "
        "vector per crop for every backbone"
    ),
    "why_not_store_the_maps": (
        "22 x 100,352 dims leaves the matched concat partner with a 2.2M "
        "head at n=237, which is not an arm -- graph-vs-concat would "
        "revert to two factors"
    ),
    "not_srgnn_as_published": (
        "pooling turns SeqSelfAttention into a smaller layer (2048-wide "
        "input, not 100,352). This arm tests message passing over "
        "anatomy-crop embeddings and does NOT replicate SR-GNN's "
        "published architecture -- no write-up may say it does"
    ),
    "per_region_size_argument_does_not_transfer": (
        "that record preferred the 95 MB map over ~2.6 GB of descriptors "
        "because the map is a SUPERSET regions are re-derived from. Here "
        "nothing can be re-derived -- each crop is an independent "
        "backbone pass -- so the choice rests on head dimension, not bytes"
    ),
    "pooling": "spatial mean, matching what 'pooled' already means",
}


#: **[RE-RENDERED AND RE-REVIEWED 2026-08-14] Both sheets rebuilt with
#: honest captions, and re-reviewed. Neither verdict changes.**
#:
#: The random layout's panels had been captioned through
#: ``render.abbreviate``, so they read anatomy names on non-anatomical
#: boxes (RANDOM_ARM_NAMES_WERE_THE_DEFECT). Both sheets were re-rendered
#: with each layout's own names and looked at again.
#:
#: **Why re-review rather than reason it away.** Neither gate's finding
#: depended on the captions -- gate 1 verified anatomy on the ANATOMY
#: layout, gate 2's square-resize finding is layout-independent -- so the
#: verdicts were sound. But they were sound on an artifact that asserted
#: something false, and "technically sound on misleading captions" is not
#: a state a review gate should be left in. The re-review costs one look
#: and removes the qualifier.
CROP_SHEETS_RE_REVIEWED = {
    "re_rendered": "2026-08-14",
    "why": (
        "the random layout's captions read anatomy names on "
        "non-anatomical boxes -- the artifact under review asserted "
        "something false"
    ),
    "verdicts_unchanged": (
        "gate 1 verified anatomy on the ANATOMY layout and gate 2's "
        "square-resize finding is layout-independent, so neither finding "
        "depended on the captions"
    ),
    "what_changed": (
        "the review is now clean rather than technically-sound-on-"
        "misleading-captions -- a qualifier a gate should not carry"
    ),
    "outcome": "both sheets pass on honest artifacts",
}


#: **[DECIDED 2026-08-14] Which layer reshapes, and who decides.**
#:
#: One ``region_vectors`` set is ``(N, 22, D)`` and four readings exist:
#:
#:     concat arms (3)   (N, 22*D)   flattened, C order
#:     graph arm         (N, 22, D)  22 nodes of D, as stored
#:     whole control     (N, D)      a pooled set, not region_vectors
#:
#: **The reshape lives in ``embeddings.as_consumed``** -- the module that
#: owns the artifact format, so the invariants and the readings of them
#: sit together, and there is ONE place that refuses a mismatch rather
#: than one per training task.
#:
#: **The CONSUMER declares which reading it wants**, as a config field
#: (``consume``), because inference is silent where it is wrong. "3-D
#: means flatten" would hand the graph arm a flattened vector the first
#: time someone reused the loader, and that arm would train, converge and
#: report a number. A declared reading turns that into a load-time
#: refusal.
#:
#: **Not at extraction**, which was the tempting alternative: storing
#: flattened would discard the structure the graph arm needs while saving
#: the concat arms nothing, since ``(N, 22*D)`` is a VIEW of
#: ``(N, 22, D)`` rather than a different measurement. One artifact,
#: three readings, no duplicate sets to keep in step.
#:
#: **The flatten order is part of the contract**: C order, so the result
#: is 22 contiguous blocks of D in the set's own ``regions`` order. That
#: list is what makes a flattened vector interpretable -- block *i* is
#: region ``regions[i]`` -- and it is why ``region_vectors`` refuses to be
#: written without names.
#:
#: **The control declares ``pooled`` rather than omitting the field.**
#: Absent means pooled for every Road A arm, so omission would work; but
#: an arm in a branch where three shapes are live should say which one it
#: reads, not rely on a default that means something else elsewhere.
REGION_SET_CONSUMPTION = {
    "decided": "2026-08-14",
    "where": "embeddings.as_consumed -- the module that owns the format",
    "who_decides": "the consumer, as a config field (`consume`)",
    "readings": {
        "concat": "(N, 22*D), C order",
        "nodes": "(N, 22, D) as stored",
        "pooled": "(N, D) -- the whole-frame control's own kind",
    },
    "why_declared_not_inferred": (
        "'3-D means flatten' would hand the graph arm a flattened vector "
        "and it would train, converge and report a number. A declared "
        "reading makes the mismatch a load-time refusal"
    ),
    "why_not_at_extraction": (
        "(N, 22*D) is a VIEW of (N, 22, D), not a different measurement, "
        "so storing flattened would discard what the graph arm needs and "
        "save the concat arms nothing -- and leave two sets to keep in step"
    ),
    "flatten_order": (
        "C order: 22 contiguous blocks of D in the set's own `regions` "
        "order, which is what makes a flattened vector interpretable"
    ),
    "control_declares_pooled": (
        "absent means pooled for every Road A arm, so omission would "
        "work -- but an arm in a branch where three shapes are live "
        "should say which one it reads"
    ),
}


#: **[FIXED 2026-08-14] The arms hit the stacked-tensor mismatch one layer
#: up, and the fix is in phase3 -- narrowly, and provably safe.**
#:
#: ``phase3.run`` called ``load_inputs`` unconditionally, which demands
#: ``staged_patient_<geometry>.npy``. Non-square staging can never produce
#: one (per-patient shapes differ, so ``np.stack`` raises --
#: ``roadb_staging.SHAPE_IS_FIXED_AT_STAGING``). Same mechanism as the
#: whole-frame extraction failure, one layer higher: the arms, not the
#: extraction.
#:
#: **Both questions, answered from the code rather than by preference.**
#:
#: **1. Should the arms declare staged input? YES -- and not for the
#: pixels.** Walking ``run``, ``images`` is read in exactly three places:
#: the assignment itself, the ``features_override`` row count, and
#: ``prepare_features`` (live extraction). **None is on the artifact
#: path.** So the tensor was loaded, held -- ~419 MB for a 768 arm -- and
#: discarded. But the DECLARATION earns its place twice over:
#:
#: * ``geometry.csv`` lives in that artifact and drives the row-order
#:   assertion, which is the check that actually matters and needs no
#:   pixels;
#: * provenance -- the arm's ``inputs.json`` names which staging its crops
#:   were cut from, guard-3 verified. Dropping it would break the
#:   config -> input -> staging chain and, specifically, make
#:   "both sides of crop_vs_whole read the same staging" uncheckable.
#:   That is the matched-pipeline claim the corrected control exists to
#:   make, so the declaration is provenance worth keeping.
#:
#: **2. phase3 or the configs? phase3 -- but narrower than "a change that
#: touches every arm".** ``load_inputs`` gains ``require_images``, and
#: ``run`` passes False exactly when the arm is artifact-fed and not using
#: the override. On that path the tensor is provably unread, so no number
#: can move; every other arm keeps the load. Road A's pooled arms stop
#: reading a tensor they discarded, which is strictly less work and
#: cannot change a result.
#:
#: **The skip drops a read, not a guarantee**: the geometry.csv row-order
#: assertion still runs, and the row-COUNT check is skipped only when
#: there are no rows to count. A static test walks ``run`` and asserts
#: ``images`` stays unread on the artifact path, so a future use cannot
#: silently receive ``None``.
ARTIFACT_ARMS_DO_NOT_READ_PIXELS = {
    "fixed": "2026-08-14",
    "symptom": (
        "Phase3Error: staged_patient_g2.npy does not exist -- the arms "
        "hitting the stacked-tensor mismatch one layer above extraction"
    ),
    "measured": (
        "`images` is read in exactly three places in phase3.run: its own "
        "assignment, the features_override row count, and prepare_features "
        "(live extraction). None is on the artifact path"
    ),
    "declaration_kept": (
        "geometry.csv drives the row-order assertion and needs no pixels; "
        "and the declaration is what makes 'both sides of crop_vs_whole "
        "read the same staging' checkable -- the matched-pipeline claim "
        "the corrected control exists to make"
    ),
    "fix_is_in_phase3": (
        "load_inputs gains require_images; run passes False exactly when "
        "the arm is artifact-fed and not using the override. Road A's "
        "pooled arms stop loading a tensor they discarded -- strictly "
        "less work, and no number can move because it was never read"
    ),
    "drops_a_read_not_a_guarantee": (
        "the geometry.csv row-order assertion still runs; the row-COUNT "
        "check is skipped only when there are no rows to count"
    ),
    "guarded_by": (
        "a static test walking run(), so a future use of `images` cannot "
        "silently receive None"
    ),
    "third_time_the_message_paid": (
        "the error listed what it found, so the diagnosis took one read"
    ),
}


#: **[MEASURED 2026-08-14] What condition 1 has actually done to every
#: margin either road has put to it -- and the tempting sharpening is
#: WRONG in a way worth keeping.**
#:
#: "Nothing below roughly 6x has ever passed" is the natural reading of
#: the failures, and the project's own record refutes it:
#: ``ladder.LADDER_PAIRED_AUDIT``'s survivor is **3.83x** at 5 of 5. So
#: the bar is not a threshold on the margin. The full picture:
#:
#:     margin   outcome                                    where
#:     8.57x    SURVIVED 5/5   vit masked 224->512         Road B res.
#:     7.38x    SURVIVED 5/5   vit imagenet 224->512       Road B res.
#:     6.91x    SURVIVED 5/5   vit imagenet 224->768       Road B res.
#:     4.69x    withdrew 2/5                               ladder audit
#:     4.34x    withdrew 4/10                              ladder audit
#:     3.83x    SURVIVED 5/5   Q1__swin_b                   ladder audit
#:     3.66x    withdrew 1/5   swin 224->512 (the peak)    Road B res.
#:     3.62x    withdrew 0/5   7C augmentation arm 2       phase 7C
#:     2.55x    withdrew 2/5   anatomy vs random           Branch 3
#:     2.37x    withdrew 1/5   7B search                   phase 7B
#:     1.95x    withdrew 0/5   random vs whole             Branch 3
#:     0.51x    withdrew 0/5   crop vs whole               Branch 3
#:
#: **Three regions, and the middle one is the finding.** Nothing at or
#: below 2.55x has ever passed. Everything at or above 6.91x has, three
#: for three. **Between 3.6x and 4.7x the outcome is mixed and the margin
#: does not order it**: 3.83 passed while 4.34 and 4.69 failed. That is
#: ``LADDER_PAIRED_AUDIT.margin_does_not_predict_condition_1`` confirmed
#: on four more points rather than a new rule.
#:
#: **Why the maintainer's version is nearly right anyway.** The single
#: sub-6.91x survivor is the most caveated result in the project: its
#: baseline is 0.0065 -- a near-zero correlation -- and the same
#: comparison at the other geometry gives 0.44x at 0 of 5. So no margin
#: below ~6.9x has ever produced a USABLE claim, which is the practical
#: statement; "none has ever passed" is the one to avoid, because the
#: exception exists and a reader will find it.
#:
#: **[UPDATED 2026-08-15, Phase 7D's four margins.]** Four new rows::
#:
#:     6.42x    SURVIVED 5/5   p7d patch8 vs b16            phase 7D
#:     5.32x    SURVIVED 5/5   p7d b32_512 vs patch16@512   phase 7D
#:     1.89x    withdrew 0/5   p7d mvitv2 vs swin           phase 7D
#:     1.77x    withdrew 0/5   p7d mvitv2 vs b16            phase 7D
#:
#: The 4.7x-6.91x interval above was EMPTY when the three regions were
#: drawn; 7D put two points in it and both passed, so the clean-pass
#: region now starts at 5.32x -- and the 5.32x pass is a USABLE claim on
#: real baselines, which moves the usable-claim floor down from ~6.9x.
#: The two failures sit under 2.55x, where nothing has ever passed. The
#: mixed 3.6x-4.7x middle is untouched and remains the finding.
CONDITION_1_MARGIN_STRUCTURE = {
    "measured": "2026-08-14, over every margin either road has tested",
    "updated": "2026-08-15, with Phase 7D's four margins",
    "survivors": {
        8.57: "5/5", 7.38: "5/5", 6.91: "5/5", 3.83: "5/5",
        # [2026-08-15] Phase 7D's two claims -- the first points ever to
        # land in the 4.7x-6.91x interval, both passing.
        6.42: "5/5 (p7d patch8 vs b16)",
        5.32: "5/5 (p7d b32_512 vs patch16@512)",
    },
    "withdrawals": {
        4.69: "2/5", 4.34: "4/10", 3.66: "1/5", 3.62: "0/5",
        2.55: "2/5", 2.37: "1/5", 1.95: "0/5", 0.51: "0/5",
        # [2026-08-15] Phase 7D's arm-6 pair, both 0/5.
        1.89: "0/5 (p7d mvitv2 vs swin)",
        1.77: "0/5 (p7d mvitv2 vs b16)",
    },
    "floor": (
        "nothing at or below 2.55x has ever passed -- 7D's 1.77x and "
        "1.89x at 0/5 are consistent"
    ),
    #: [UPDATED 2026-08-15] Was "everything at or above 6.91x has -- three
    #: for three". The 4.7x-6.91x interval was EMPTY when that was written;
    #: 7D put two points in it and both passed, so the clean-pass region
    #: extends DOWN.
    "ceiling": (
        "everything at or above 5.32x has passed -- five for five. "
        "[was 6.91x, three for three, when the 4.7x-6.91x interval held "
        "no data; 7D's 6.42x and 5.32x filled it, both 5/5]"
    ),
    "the_middle_is_the_finding": (
        "between 3.6x and 4.7x the outcome is MIXED and margin does not "
        "order it: 3.83 passed while 4.34 and 4.69 failed"
    ),
    "the_tempting_sharpening_is_wrong": (
        "'nothing below roughly 6x has ever passed' is refuted by the "
        "ladder audit's 3.83x survivor -- the exception exists and a "
        "reader will find it"
    ),
    "what_is_true_instead": (
        "no margin below ~6.9x has produced a USABLE claim: the one "
        "sub-6.9x survivor has a 0.0065 baseline and gives 0.44x at 0 of "
        "5 at the other geometry. [SUPERSEDED 2026-08-15: 7D's 5.32x pass "
        "is a usable claim on real baselines (0.2280 against 0.0857), so "
        "the usable-claim floor moves from ~6.9x to 5.32x. The 3.83x "
        "caveat stands unchanged]"
    ),
    "consequence": (
        "the bar is not a threshold on condition 2. A margin buys nothing "
        "below ~6.9x and everything above it, and in between the "
        "patient-level agreement decides on its own terms. [UPDATED "
        "2026-08-15: 'everything above' now starts at 5.32x; the mixed "
        "region is 3.6x-4.7x; below 2.55x nothing has passed]"
    ),
}


#: **[NOT BUILT 2026-08-14] The graph arm's node path, scoped -- and
#: refused at LOAD time so it cannot fail after a queue wait.**
#:
#: Four of the five Branch 3 arms are wired end to end: ``train_cv`` now
#: routes its artifact through ``embeddings.as_consumed`` via
#: ``PooledSource.consume``, so ``pooled`` and ``concat`` both work and
#: the D+1 parameter check compares against the width the HEAD sees
#: rather than the artifact's per-region dim.
#:
#: **The graph arm is not, and the gap is structural rather than a
#: reshape.** ``graph_cleft.run`` is built for feature maps end to end:
#:
#: * ``pack`` requires ``(N, C, H, W)`` and flattens it;
#: * ``boxes_for_arm`` computes the region boxes -- which a node set
#:   already embodies, having been cropped BEFORE the backbone;
#: * ``make_backbone`` constructs a graph head that ROI-pools that map.
#:
#: So feeding nodes needs a second construction path through the graph
#: backbone, not a reshape at the loader. Writing it unverified into a
#: 2,000-line training module while five arms are queued is the R10
#: failure this project keeps recording, so it is scoped rather than
#: guessed.
#:
#: **What it needs**: a node-source branch that skips ``boxes_for_arm``
#: and ``pack``, a ``GraphHeadBackbone`` construction that takes node
#: features directly instead of pooling them from a map, and the pairing
#: check taught that a ``region_vectors`` set is legitimate here. Then the
#: refusal in ``task_train_graph_cv`` comes out.
#:
#: **Meanwhile the refusal is at LOAD time**, the same discipline
#: ``_check_train_cv_init`` follows: the arm cannot sit in a queue and
#: fail after the wait. The other four arms are unaffected and may run.
GRAPH_NODE_PATH_IS_NOT_WIRED = {
    "recorded": "2026-08-14",
    "wired": (
        "train_cv, through PooledSource.consume -> embeddings.as_consumed; "
        "'pooled' and 'concat' both work"
    ),
    "not_wired": "train_graph_cv with consume: nodes",
    "why_not_a_reshape": (
        "pack requires (N, C, H, W); boxes_for_arm computes boxes a node "
        "set already embodies; make_backbone builds a head that ROI-pools "
        "a map. Nodes need a second construction path"
    ),
    "what_it_needs": (
        "a node-source branch skipping boxes_for_arm and pack, a "
        "GraphHeadBackbone taking node features directly, and the pairing "
        "check taught that region_vectors is legitimate here"
    ),
    "refused_at_load": (
        "task_train_graph_cv raises on consume: nodes, so the arm cannot "
        "sit in a queue and fail after the wait -- _check_train_cv_init's "
        "discipline"
    ),
    "unaffected": "the other four Branch 3 arms may run",
    "why_scoped_not_guessed": (
        "writing it unverified into a 2,000-line training module while "
        "five arms are queued is the R10 failure this project keeps "
        "recording"
    ),
}


#: **[DIAGNOSED 2026-08-14] The random arm's name collision was a NAMING
#: defect, not a construction one -- and the control was never broken.**
#:
#: ``roadb_p7c_crops_vit_b16_random`` stopped with *"region names are not
#: unique"*: ``rim_L`` and ``comm_L`` twice, ``rim_R`` and ``comm_R``
#: absent, and midline structures carrying side suffixes (``nroot_R``,
#: ``tip_L``). The natural reading is that the random layout resamples
#: which boxes it takes. **Measured, it does not:**
#:
#:     count      22 == 22
#:     bands      preserved, in order
#:     sizes      (w, h) identical box for box
#:     total area equal to 1e-12
#:     area_matches()  True
#:
#: ``random_patches`` takes the 22 anatomy boxes and moves each one,
#: exactly as designed, so the matching ``anatomy_vs_random`` depends on
#: was intact throughout.
#:
#: **What broke was the label.** ``render.abbreviate`` derives ``_L`` /
#: ``_R`` from a box's CENTRE. For anatomy that is correct and exact --
#: the 8 midline boxes sit at x=0.5 and take no suffix, the 7 bilateral
#: pairs take opposite sides. Applied to a randomly placed box it answers
#: a question that no longer exists: **laterality is an anatomical fact,
#: and a box that has been moved has none.** Two members of a pair can
#: land on the same side (the collision) and a midline structure can land
#: off-centre (the spurious suffix).
#:
#: **The fix is positional-free names for the random layout**
#: (``roadb_regioncrop.region_names``). Naming those boxes after anatomy
#: would be worse than colliding: a consumer asking which column is
#: ``philtral_column`` would get a box that is not on it, and the arm's
#: whole point is that its placement is not anatomical. The size matching
#: moves to ``size_provenance`` in the run's metrics, so what makes the
#: control a control is recorded rather than implied by a label.
#:
#: **Second consequence, found by following the same function**: the
#: random layout's SHEETS were captioned the same way, so the reviewed
#: random panels read ``philt_R`` on boxes that are not the philtrum. A
#: reviewer was being told something false by the artifact they were
#: checking. Captions now use the layout's own names.
RANDOM_ARM_NAMES_WERE_THE_DEFECT = {
    "diagnosed": "2026-08-14",
    "symptom": (
        "region names are not unique -- rim_L and comm_L twice, rim_R and "
        "comm_R absent, midline structures carrying side suffixes"
    ),
    "not_the_cause": (
        "the box set does NOT vary: count, bands, per-box (w, h) and "
        "total area are all preserved exactly, and area_matches() is True. "
        "random_patches takes the 22 anatomy boxes and moves each one"
    ),
    "the_cause": (
        "render.abbreviate derives _L/_R from a box's CENTRE. Correct for "
        "anatomy (8 midline at x=0.5 take no suffix, 7 pairs take opposite "
        "sides); meaningless once a box is moved, because laterality is an "
        "anatomical fact and a moved box has none"
    ),
    "fix": (
        "positional-free names for the random layout; the size matching "
        "recorded in size_provenance rather than implied by a label"
    ),
    "why_not_anatomy_names": (
        "a consumer asking which column is philtral_column would get a box "
        "that is not on it, and the arm's point is that its placement is "
        "not anatomical"
    ),
    "second_consequence": (
        "the random layout's SHEET captions used the same function, so "
        "reviewed random panels read anatomy names on non-anatomical "
        "boxes -- the artifact under review was telling the reviewer "
        "something false. Captions now use the layout's own names"
    ),
    "the_contrast_is_unaffected": (
        "anatomy_vs_random stays the strictly one-factor test of anatomy: "
        "count, size distribution and total area matched throughout, and "
        "only the labels changed"
    ),
}


#: **[MEASURED 2026-08-14] Every crop is squared before the backbone sees
#: it, and the first sheet could not show that.**
#:
#: ``crop_region`` calls ``resize_nearest(window, size, size)``, so a
#: non-square window becomes 224x224. Measured over the 22 anatomy boxes:
#: source aspect ratios span **0.72 to 1.54** (median 1.20), so some
#: regions are stretched and some squeezed -- a spread of about 2.1x
#: between the extremes, wider than the "about 1.2:1" first estimated.
#:
#: **It confounds no contrast**: every arm is squared identically, and the
#: whole-frame control goes through the same resize, which is part of why
#: routing it through the crop path is right. But a stretched philtral
#: column looks fine in the numbers and wrong to a clinician, and the
#: native sheet showed crops AS CUT -- so the passed review confirmed
#: anatomy and magnification and said nothing about distortion.
CROPS_ARE_SQUARE_STRETCHED = {
    "measured": "2026-08-14",
    "mechanism": "crop_region resizes every window to size x size",
    "source_aspect_ratios": {"min": 0.72, "median": 1.20, "max": 1.54},
    "confounds_nothing": (
        "every arm is squared identically and the whole-frame control "
        "goes through the same resize"
    ),
    "but": (
        "a stretched philtral column looks fine in the numbers and wrong "
        "to a clinician, and the native sheet showed crops AS CUT"
    ),
    "second_sheet": "roadb_p7c_region_crop_sheet_as_fed, mode=as_fed",
    # [REVIEWED 2026-08-14] The distortion is measured AND accepted. See
    # CROP_SHEET_AS_FED_REVIEW_PASSED: the anatomy reads correctly as
    # squares across the range, so the square resize is acceptable input
    # and the aspect-preserving alternative is not needed.
    "reviewed": "accepted -- the anatomy survives the resize",
}


#: **[REVIEWED 2026-08-14 — PASSED. The second gate.]**
#:
#: **The two sheets asked different questions and both were needed.**
#:
#: * **Sheet 1 (native, as cut)**: *do the boxes sit on the right
#:   anatomy?* Passed -- philtrum boxes on the philtrum, both philtral
#:   columns present as separate patches (CROP_SHEET_REVIEW_PASSED).
#: * **Sheet 2 (as fed, squared)**: *does that anatomy survive the
#:   transform the backbone receives?* Passed -- the crops read correctly
#:   as squares and the anatomy is still recognisable after the resize,
#:   across the measured range (source ratios 0.72 to 1.54).
#:
#: **So the square resize is acceptable input, and the aspect-preserving
#: alternative is declined.** That is a decision resting on a review, not
#: on an argument: it can be revisited only by another review, and the
#: alternative was never built.
#:
#: **Neither sheet could have answered the other's question.** The native
#: sheet showed information and hid transformation; the as-fed sheet shows
#: transformation and, by resizing everything to 224, hides how much
#: information was there to begin with. Two sheets, two questions, both
#: passed -- which is what "the review passed" now means, and it means
#: more than it did with one.
CROP_SHEET_AS_FED_REVIEW_PASSED = {
    "reviewed": "2026-08-14",
    "verdict": "PASSED -- the second gate; the four crop extractions are unblocked",
    "question": (
        "does the anatomy survive the transform the backbone receives, "
        "as distinct from sheet 1's does the box sit on the right anatomy"
    ),
    "finding": (
        "the crops read correctly as squares; the anatomy is still "
        "recognisable after the resize, across the range"
    ),
    "consequence": (
        "the square resize is acceptable input, and the aspect-preserving "
        "alternative is DECLINED -- a decision resting on a review rather "
        "than an argument, revisitable only by another review"
    ),
    "neither_sheet_substitutes": (
        "the native sheet shows information and hides transformation; the "
        "as-fed sheet shows transformation and hides how much information "
        "was there. Two questions, two sheets"
    ),
    "range_covered": "source aspect ratios 0.72 to 1.54, median 1.20",
}


#: **[RECORDED 2026-08-14] A review sheet must render what the CONSUMER
#: receives, and this is the second instance.**
#:
#: A sheet answers exactly the question its panels depict. Render a
#: different stage of the pipeline and the review passes on a question
#: nobody asked:
#:
#: 1. **The masked SCUT sheets** render masked panels at NATIVE scale
#:    beside the source-scale placement panel, specifically so the
#:    1.46x/2.19x interpolation the backbone would receive is VISIBLE.
#:    Rendering them at display scale would have hidden it.
#: 2. **The crop sheet, this branch.** It rendered crops AS CUT, which
#:    was right for the magnification question and wrong for the input
#:    question: the backbone receives them SQUARED. The review passed
#:    without anyone seeing the distortion, so "the review passed" meant
#:    less than it appeared to until the second sheet existed.
#:
#: **The rule**: when a pipeline transforms a panel between the stage
#: shown and the stage consumed, either show the consumed stage or ship a
#: second sheet that does. A sheet showing the artifact rather than the
#: input answers a different question than the one being asked.
#: **[AMENDED 2026-08-14] The rule now has all three modes, and the third
#: is the one that makes it credible.**
#:
#: A check that only ever appears when something is wrong is hard to
#: distinguish from a check that is not running. The as-fed sheet
#: **passed** -- the anatomy survives the square resize -- so the rule has
#: now been followed by design, violated with a visible cost, AND applied
#: to a confirming outcome. The confirming instance is what shows the
#: second sheet can come back clean, which is precisely what makes a
#: future failure mean something rather than look manufactured.
SHEETS_MUST_SHOW_WHAT_THE_CONSUMER_RECEIVES = {
    "recorded": "2026-08-14",
    "instances": {
        "masked_scut": (
            "PREVENTED: masked panels rendered at NATIVE scale so the "
            "1.46x/2.19x interpolation the backbone receives is visible. "
            "The rule followed by design"
        ),
        "region_crops": (
            "CAUGHT: crops rendered AS CUT -- right for magnification, "
            "wrong for the input question, because the backbone receives "
            "them SQUARED. The review passed without seeing the "
            "distortion, so the rule was violated at a visible cost"
        ),
        "region_crops_as_fed": (
            "CONFIRMED: the second sheet ran and PASSED -- the anatomy "
            "survives the resize (CROP_SHEET_AS_FED_REVIEW_PASSED). The "
            "rule applied to an outcome it did not overturn"
        ),
    },
    "rule": (
        "when a pipeline transforms a panel between the stage shown and "
        "the stage consumed, either show the consumed stage or ship a "
        "second sheet that does"
    ),
    "why_it_matters": (
        "a sheet showing the artifact rather than the input answers a "
        "different question than the one being asked, and a review that "
        "passes on it means less than it appears to"
    ),
    "why_the_confirming_case_matters": (
        "a check that only ever appears when something is wrong is hard "
        "to distinguish from a check that is not running. The as-fed "
        "sheet coming back CLEAN is what shows the second sheet can pass "
        "-- which is what makes a future failure mean something rather "
        "than look manufactured"
    ),
    "modes_seen": ("prevented", "caught", "confirmed"),
}


def regioncrop_arms() -> list[dict]:
    """The five arms, derived so a config cannot invent one.

    Five rather than the brief's four: the control is at the crops' own
    staging (defect 1) and the graph arm has a matched concat partner
    (defect 2). Every contrast the set admits varies exactly one factor.
    """
    common = {"init": "imagenet", "staging": "roadb_768_nonsquare_g2_v1"}
    return [
        # combination is None, not "pooled": a whole frame has ONE
        # feature vector, so there is no combination rule to hold
        # constant. Calling it "pooled" would make it look like a second
        # varying factor when it is an undefined one.
        {"key": "control_whole_vit", "backbone": "vit_b16", "regime": "transformer",
         "input": "whole_frame", "combination": None, "seeds": 5,
         "tests": "the control -- whole frame at the crops' own staging",
         **common},
        {"key": "anatomy_concat_vit", "backbone": "vit_b16", "regime": "transformer",
         "input": "anatomy_crops", "combination": "concat", "seeds": 5,
         "tests": "does cropping to anatomy beat a whole frame",
         **common},
        {"key": "random_concat_vit", "backbone": "vit_b16", "regime": "transformer",
         "input": "random_crops", "combination": "concat", "seeds": 5,
         "tests": "does CROPPING beat a whole frame, placement aside",
         **common},
        {"key": "anatomy_graph_srgnn", "backbone": "srgnn", "regime": "graph",
         "input": "anatomy_crops", "combination": "graph", "seeds": 10,
         "tests": "does message passing over anatomy add anything",
         **common},
        {"key": "anatomy_concat_srgnn", "backbone": "srgnn", "regime": "graph",
         "input": "anatomy_crops", "combination": "concat", "seeds": 10,
         "tests": "the graph arm's matched partner -- combination alone",
         **common},
    ]


#: **[PRE-REGISTERED 2026-08-14] The four contrasts, before any arm
#: runs, each varying ONE factor** -- enumerated the way
#: ``roadb_resolution`` was, so the set is fixed by the design rather
#: than chosen after the numbers exist.
#:
#: * **crop_vs_whole** (anatomy_concat_vit - control): the branch's
#:   question. Input varies; backbone, init, staging, combination-free
#:   head all held.
#: * **anatomy_vs_random** (anatomy_concat_vit - random_concat_vit):
#:   placement varies. **The contrast that decides what the finding is
#:   ABOUT** -- if random crops match anatomy crops, the result is about
#:   cropping, not anatomy (brief §3, Charm 2025).
#: * **random_vs_whole** (random_concat_vit - control): cropping per se
#:   against a whole frame, with placement uninformative.
#: * **graph_vs_concat** (anatomy_graph_srgnn - anatomy_concat_srgnn):
#:   combination varies, backbone and crops held. Both read the SAME
#:   crop set.
#:
#: **Deliberately NOT enumerated: any ViT-vs-SR-GNN pair.** Those vary
#: backbone and combination together, which is the defect this
#: enumeration exists to avoid. And no contrast against Road B's
#: existing 768 square-G1 arm -- that is the control mismatch, recorded.
#:
#: **[CORRECTED 2026-08-14] That exclusion note stopped being accurate
#: the moment the matched SR-GNN concat partner was added, and saying so
#: matters more than the tidiness.** ``anatomy_concat_vit`` against
#: ``anatomy_concat_srgnn`` varies BACKBONE ALONE: same input, same
#: combination, same init, same staging, same crop pixels. It is a
#: legitimate one-factor contrast and the note above says no such pair
#: exists. The note was written when the only SR-GNN arm was the graph
#: one, where it was true.
#:
#: **It is still not added, and the reason is not tidiness.** The four
#: were fixed by design before any number existed. Adding a fifth after
#: seeing that it separates by 0.071 is selection -- the exact move
#: PHASE_7_SWIN_512 refused for the Swin peak, and refusing it there
#: while allowing it here would make the principle a preference. So the
#: backbone comparison is REPORTED DESCRIPTIVELY
#: (REGION_CROP_ARMS_OBSERVED.srgnn_on_identical_pixels) and not tested
#: by condition 1. If it is ever to be tested, that is a registration
#: made before the next set of numbers, not this one.
#:
#: **[FOUND 2026-08-14, by the one-factor test] Two of the four are
#: STRICTLY one-factor and two are not, and the difference is not
#: fixable.** A whole frame yields ONE feature vector; crops yield
#: twenty-two that must be combined somehow. So the crop-vs-whole
#: comparisons change the input AND the head's input dimension together,
#: as an **indivisible** change -- there is no combination rule to hold
#: constant on the whole-frame side, because none exists there. Recorded
#: rather than papered over, and it is why the brief is right that arm 4
#: is "the control that matters": **anatomy_vs_random is the strictly
#: one-factor test of anatomy**, and graph_vs_concat is the strictly
#: one-factor test of message passing. The two indivisible contrasts
#: answer "does this input REPRESENTATION beat that one", which is the
#: branch's actual question and is still worth asking -- it just must
#: not be reported as isolating cropping from combination.
REGION_CROP_CONTRASTS = (
    {"key": "crop_vs_whole", "a": "control_whole_vit",
     "b": "anatomy_concat_vit", "varies": "input",
     "strictly_one_factor": False,
     "indivisible": (
         "input and head dimension move together -- a whole frame has "
         "no combination rule to hold constant"
     ),
     "question": "does cropping to anatomy beat a whole frame"},
    {"key": "anatomy_vs_random", "a": "random_concat_vit",
     "b": "anatomy_concat_vit", "varies": "input",
     "strictly_one_factor": True,
     "question": "is the finding about anatomy or about cropping"},
    {"key": "random_vs_whole", "a": "control_whole_vit",
     "b": "random_concat_vit", "varies": "input",
     "strictly_one_factor": False,
     "indivisible": "as crop_vs_whole",
     "question": "does cropping per se beat a whole frame"},
    {"key": "graph_vs_concat", "a": "anatomy_concat_srgnn",
     "b": "anatomy_graph_srgnn", "varies": "combination",
     "strictly_one_factor": True,
     "question": "does message passing over anatomy add anything"},
)


#: **[CLOSING 2026-08-14] Road B Branch 3 -- region-cropped input.**
#:
#: **WHAT IS CLAIMABLE: nothing.** All three computable contrasts
#: withdrew at PLAN 4.3's full criterion:
#:
#:     anatomy_vs_random   2.55x   2 of 5   withdrawn
#:     random_vs_whole     1.95x   0 of 5   withdrawn
#:     crop_vs_whole       0.51x   0 of 5   withdrawn
#:
#: **WHAT THE ARMS STILL SUPPORT, descriptively.** Anatomy crops MATCH a
#: whole frame -- 0.2428 against 0.2307, with crop_vs_whole emphatically
#: null at 0.51x -- so **discarding roughly 60% of the image costs
#: nothing measurable**. Random crops fall 0.077 short and reach 2 of 5,
#: which is more than either other contrast managed and is not a claim.
#: The shape only reads with its null in it: **throwing away the
#: periphery is free; throwing away the RIGHT regions is not -- and the
#: second half is what this cohort cannot prove.**
#:
#: **WHAT WAS NEVER BUILT: the graph node path, and with it the fourth
#: contrast.** ``graph_cleft`` builds nodes by ROI-pooling a feature map
#: and Branch 3's regions are cropped BEFORE the backbone, so feeding
#: them needs a second construction path, not a reshape
#: (GRAPH_NODE_PATH_IS_NOT_WIRED). The arm refuses at load rather than
#: failing after a queue wait. So ``graph_vs_concat`` -- whether message
#: passing over anatomy adds anything concatenation does not -- is
#: unanswered, and the SR-GNN concat arm that ran is its orphaned half.
#:
#: **WHAT THE BRANCH ESTABLISHES.** Supervision asked for manual patch
#: selection. It was built -- on the justification that survived
#: measurement rather than the one that did not, since the border reading
#: was refuted at 1.065x chance before any of this was designed -- and
#: the answer is: **anatomy-cropped input performs like a whole frame and
#: better than random crops, but not claimably so at n=237.** That is an
#: answer to the question asked, not a failure to do the work, and the
#: write-up
#: should say it in that order.
#:
#: **Two findings came out of the branch rather than out of its arms**,
#: and both are worth more than the contrasts were:
#:
#: * **The SR-GNN ceiling.** On the same crop pixels, SR-GNN scores
#:   0.1717 against its best prior arm's 0.1719 -- a WHOLE-FRAME arm on a
#:   different road, staging, aspect and geometry. Changing the input
#:   representation entirely moved it by 0.0002. The backbone caps the
#:   result, not the representation (REGION_CROP_ARMS_OBSERVED.
#:   srgnn_on_identical_pixels). A ceiling observation, not a contrast.
#: * **A one-factor contrast, refused.** ``anatomy_concat_vit`` against
#:   ``anatomy_concat_srgnn`` varies backbone alone and is not in the
#:   registered four -- because it was noticed AFTER the numbers, and
#:   adding it then is the selection PHASE_7_SWIN_512 refused for the
#:   Swin peak. Refusing it there and allowing it here would make the
#:   principle a preference (REGION_CROP_CONTRASTS' corrected note).
#:
#: **And the resolution bar is sharper than before**
#: (CONDITION_1_MARGIN_STRUCTURE): nothing at or below 2.55x has ever
#: passed, everything at or above 6.91x has, and between 3.6x and 4.7x
#: margin does not order the outcome at all.
ROAD_B_BRANCH_3_CLOSING = {
    "closed": "2026-08-14",
    "status": "CLOSED with no claimable result and two incidental findings",
    "claimable": [],
    "withdrawn": {
        "anatomy_vs_random": {"margin": 2.55, "excluding": "2/5"},
        "random_vs_whole": {"margin": 1.95, "excluding": "0/5"},
        "crop_vs_whole": {"margin": 0.51, "excluding": "0/5"},
    },
    "supported_descriptively": (
        "anatomy crops MATCH a whole frame (0.2428 vs 0.2307, "
        "crop_vs_whole null at 0.51x), so discarding ~60% of the image "
        "costs nothing measurable; random crops fall 0.077 short at 2 of 5"
    ),
    "the_shape": (
        "throwing away the periphery is free; throwing away the right "
        "regions is not -- and the second half is what the cohort cannot "
        "prove. The null is what makes it coherent"
    ),
    "never_built": {
        "what": "the graph node path, and with it graph_vs_concat",
        "why": (
            "graph_cleft builds nodes by ROI-pooling a feature map and "
            "these regions are cropped BEFORE the backbone -- a second "
            "construction path, not a reshape"
        ),
        "consequence": (
            "whether message passing over anatomy adds anything "
            "concatenation does not is UNANSWERED; the SR-GNN concat arm "
            "that ran is its orphaned half"
        ),
        "refused_at_load": "so it cannot fail after a queue wait",
    },
    "answers_the_supervision_request": (
        "supervision asked for manual patch selection; it was built on "
        "the "
        "justification that survived measurement, and anatomy-cropped "
        "input performs like a whole frame and better than random crops "
        "-- not claimably so at n=237. An answer to the question, not a "
        "failure to do the work"
    ),
    "findings_from_the_branch_not_its_arms": {
        "srgnn_ceiling": (
            "0.1717 against a best prior of 0.1719 on a whole frame -- "
            "changing the representation entirely moved the backbone by "
            "0.0002"
        ),
        "refused_fifth_contrast": (
            "anatomy_concat_vit vs anatomy_concat_srgnn varies backbone "
            "alone and was noticed AFTER the numbers, so adding it would "
            "be the selection the Swin peak was refused for"
        ),
    },
    "do_not_write_up_as": (
        "NOT 'cropping does not work' -- crop_vs_whole is null, which "
        "means anatomy crops MATCH the whole frame on 40% of the pixels; "
        "NOT 'anatomy beats random' as a claim, which is exactly what "
        "withdrew; and NOT a refutation of the magnification premise, "
        "which this cohort cannot resolve either way"
    ),
}


#: **[CLOSING 2026-08-12] Road B Phase 7, and the branch it was built to
#: answer.**
#:
#: (``PHASE_7_CLOSING`` is the run-and-report DECISION taken before the
#: duplicate check landed. This is the closing STATEMENT. Two records,
#: two questions -- named apart because this project has been bitten by
#: one name over two quantities.)
#:
#: **WHAT IS CLAIMABLE -- three contrasts, at PLAN 4.3's full criterion.**
#: All three are ViT-B/16 falling from its 224 pretraining resolution:
#:
#:     vit_b16 masked   224->512   8.57x   5/5   delta -0.185  [confounded]
#:     vit_b16 imagenet 224->512   7.38x   5/5   delta -0.163
#:     vit_b16 imagenet 224->768   6.91x   5/5   delta -0.158
#:
#: **The masked one carries a confound flag that must travel with it.**
#: Its two arms read two different pretraining cells, each a draw from a
#: nondeterministic regime, so it confounds resolution with the draw
#: (-0.185 is ~3.4x the 0.055 spread, so it likely survives, but it is
#: confounded). The two imagenet contrasts do not: they consume one
#: declared snapshot at every resolution, so resolution is the only
#: factor. **The clean pair carries the claim.**
#:
#: **WHAT IS WITHDRAWN -- seventeen of twenty, including the only gain.**
#:
#: * **Swin's peak: 3.66x on condition 2, 1 of 5 on condition 1.** The
#:   one apparent improvement in Road B is not claimable at n=237. It
#:   failed between Road A's 3.62x failure and its 3.83x survivor -- the
#:   band where the margin decides nothing.
#: * **Separately, and this must not merge with the above: the
#:   pre-registered non-collapse prediction STANDS.** It said frozen Swin
#:   would stay near its own 224 level where frozen ViT collapsed. In one
#:   design at one standard, ViT's collapse is claimable three times and
#:   Swin's change is not claimable in either direction -- the asymmetry
#:   the prediction named. Swin's side is an UNRESOLVED NULL, not a
#:   demonstration of stability. The gain was post-hoc and fell; the
#:   prediction was registered before any number and held.
#: * **The three graph declines that read as resolvable on seed SDs (4.1,
#:   3.8, 2.0 sigma) all withdrew** -- ONE instance of a five-instance
#:   pattern, not three results (SEED_THRESHOLDS_DO_NOT_PREDICT_THE_
#:   COHORT). Mechanism: the per-seed interval's width is set by
#:   patient-level disagreement between two prediction vectors, which the
#:   ratio of two seed SDs cannot see. First time it was called in
#:   advance.
#:
#: **WHAT WAS NEVER RUN, and why -- stated so the closing is honest.**
#:
#:     Swin masked-768 pretraining cell    NOT RUN
#:     ViT  masked-768 pretraining cell    NOT RUN
#:     their two extractions               NOT BUILT (no checkpoint)
#:     their two arms                      NOT RUN (no set)
#:
#: Two of twenty-four arms, and **nothing reportable touches them.** Every
#: claimable result is ViT: two inside the COMPLETE imagenet family, one
#: masked contrast involving no 768 arm. The two families they would
#: complete are precisely the two that could never have carried a claim
#: -- ViT masked is DESCRIPTIVE by a decision registered before any arm
#: ran, and Swin masked's only computable contrast already failed at
#: 3.08x, 1 of 5. They are two of the heaviest runs in the project (768
#: transformers, whole GPU) for points that cannot become claims.
#: Decided, not deferred.
#:
#: **WHAT THE ROAD ESTABLISHES.** Road B existed because staging at 224
#: discards ~99% of the available pixels (10.8x linear). The answer has
#: two parts and only the first is a claim: **through a frozen ViT-B/16,
#: more pixels measurably HURT**; and **nothing else is resolvable in
#: either direction**, so there is no case where recovering them helps.
#: The pre-registered null fired in its stronger form -- it anticipated
#: flatness and got a claimable decline.
#:
#: **And the sentence worth more than the resolution result:** seventeen
#: of twenty unresolvable, echoing Road A's twenty-nine of thirty on a
#: different axis. **This cohort resolves effects of ViT-collapse size
#: and nothing smaller, and that is what a next dataset has to beat.**
#: For scale, the comparator manuscript (brief §3) reports rater-specific
#: PCCs roughly half negative, mean ~0.14, selected by a weighted ranking
#: score, with no intervals anywhere. Road B's seventeen withdrawals are
#: not weakness relative to that; they are a stricter standard applied.
#:
#: **The determinism the road got for free**: fifteen prediction files
#: byte-identical across two scheduling accidents
#: (DUPLICATE_RUNS_WERE_A_DETERMINISM_CHECK), which confines the
#: project's nondeterminism to ViT pretraining and confirms the graph
#: band is seed variance rather than run-to-run flakiness.
ROAD_B_PHASE_7_CLOSING = {
    "closed": "2026-08-12",
    "distinct_from": (
        "PHASE_7_CLOSING is the run-and-report DECISION; this is the "
        "closing STATEMENT"
    ),
    "status": (
        "CLOSED with a claimable finding -- 22 of 24 arms run, 20 "
        "contrasts tested at the full criterion, 3 survive"
    ),
    "claimable": [
        "vit_b16 masked 224->512: 8.57x, 5/5, -0.185 [CONFOUNDED by the "
        "checkpoint draw -- the flag travels with the number]",
        "vit_b16 imagenet 224->512: 7.38x, 5/5, -0.163",
        "vit_b16 imagenet 224->768: 6.91x, 5/5, -0.158",
    ],
    "the_claim": "through a frozen ViT-B/16, more pixels measurably HURT",
    "withdrawn": {
        "count": 17,
        "swin_gain": (
            "3.66x on condition 2, 1 of 5 on condition 1 -- the only "
            "apparent improvement in Road B, not claimable at n=237"
        ),
        "swin_prediction_stands_separately": (
            "the registered non-collapse prediction is SUPPORTED by the "
            "asymmetry (ViT claimable three times, Swin not claimable "
            "either way); Swin's side is an unresolved null, not a "
            "demonstration of stability. Post-hoc gain fell; "
            "pre-registered prediction held"
        ),
        "graph_declines": (
            "4.1, 3.8 and 2.0 sigma on seed SDs, all withdrawn -- ONE "
            "instance of a five-instance pattern, not three results"
        ),
    },
    "never_run": {
        "cells": ("swin_b masked-768", "vit_b16 masked-768"),
        "downstream": "their two extractions and two arms",
        "why": (
            "nothing reportable touches them: every claimable result is "
            "ViT, two inside the complete imagenet family and one "
            "involving no 768 arm. The families they complete could "
            "never have carried a claim -- ViT masked is DESCRIPTIVE by "
            "pre-run decision, swin masked's only contrast already "
            "failed at 3.08x. Two of the heaviest runs in the project "
            "for points that cannot become claims"
        ),
        "decided_not_deferred": "2026-08-12",
    },
    "establishes": {
        "premise": "224 staging discards ~99% of available pixels",
        "answer": (
            "recovering them does not improve cleft assessment through "
            "frozen backbones at n=237, and for ViT it demonstrably hurts"
        ),
        "null_fired_stronger": (
            "the registered null anticipated FLATNESS; the measurement "
            "is a claimable decline for one architecture"
        ),
        "the_sentence": (
            "this cohort resolves effects of ViT-collapse size and "
            "nothing smaller, and that is what a next dataset has to beat"
        ),
        "comparator_context": (
            "brief §3: the comparator manuscript reports rater-specific "
            "PCCs roughly half negative, mean ~0.14, selected by a "
            "weighted ranking score, with NO intervals anywhere. Road "
            "B's seventeen withdrawals are a stricter standard applied, "
            "not weakness relative to it"
        ),
    },
    "free_determinism_result": (
        "15 prediction files byte-identical across two scheduling "
        "accidents -- confines the project's nondeterminism to ViT "
        "pretraining and confirms the graph band is seed variance"
    ),
    "do_not_write_up_as": (
        "NOT 'the pixels are uninformative' (frozen backbones, frozen "
        "heads, one cohort of 237); NOT 'resolution never helps' (an "
        "unresolved null is not a refutation, and one predicted family "
        "did not collapse); NOT anything about the TRAINED regime, where "
        "Phase 6 measured arms FLAT rather than falling; and NOT the "
        "masked survivor without its confound flag"
    ),
}


#: **[DECIDED 2026-08-12] Partial launch splits CLEANLY at the run level
#: and gates at the REPORT level -- said now so a two-point family is
#: never read as a shape.**
#:
#: Every arm is an independent frozen-head fit on its own declared set:
#: no arm reads another arm's output, and family shapes are read-time
#: aggregation over arm summaries. So the buildable arms launch without
#: waiting for the blocked ones, and nothing about their numbers is
#: provisional -- what is gated is what may be SAID, not what may run.
#:
#: The ledger, at writing:
#:
#: * **1 config fully resolved today**: the reuse arm (swin masked 224)
#:   -- Road A's set and paired checkpoint, hashes already in the repo.
#: * **22 configs awaiting ONE declaration pass**: 21 arms plus the
#:   sweep declare sets that are BUILT and verified on the cluster but
#:   never declared; one scripts/declare_inputs.py pass over the 21 set
#:   directories fills every hash. The blocker is a paste, not a build.
#: * **2 configs blocked on the -3 cells**: swin/vit masked-768 -- their
#:   checkpoints do not exist yet, so neither do their sets. Their
#:   checkpoint entries mirror the extraction configs' placeholders.
#:
#: What gates on what:
#:
#: * **swin_b__scut_masked and vit_b16__scut_masked report NOTHING until
#:   their 768 points land.** Their 224/512 arms RUN now; a family
#:   reports only complete -- a two-point family is a line, not a shape.
#:   Likewise two of the twelve init contrasts (swin-768, vit-768).
#:   The other six families and ten contrasts are report-complete once
#:   their arms run.
#: * **The eleven other graph arms wait on the SWEEP READING** (the
#:   sweep is the twelfth graph arm, run first). Transformer arms wait
#:   on nothing -- zero new bands.
PHASE_7_PARTIAL_LAUNCH = {
    "decided": "2026-08-12",
    "split": "clean at the RUN level; gated at the REPORT level",
    "why_clean": (
        "arms are independent frozen-head fits on their own declared "
        "sets; family shapes are read-time aggregation, so launching 22 "
        "of 24 wastes nothing and provisionalises nothing"
    ),
    "configs": {
        "total": 25,
        "resolved_today": 1,
        "awaiting_one_declaration_pass": 22,
        "blocked_on_minus3_cells": 2,
    },
    "the_22s_blocker": (
        "a paste, not a build: the 21 sets are built and verified but "
        "never declared; one scripts/declare_inputs.py pass fills every "
        "hash"
    ),
    # [LANDED 2026-08-12, same day] The pass ran: 359 verified, 21
    # hashed, 4 unresolved -- and the four are the two -3 cells DOUBLED
    # (each contributes a missing set and a missing checkpoint, PENDING
    # paths, no guessed directories), not new problems. The 22 entries
    # are filled (the srgnn masked-512 set fills two configs: its arm
    # and the sweep). The ledger above is the state at decision time,
    # kept; this field is what changed.
    "declaration_pass_landed": (
        "2026-08-12: 359 verified, 21 hashed, 4 unresolved = the two -3 "
        "cells doubled (set + checkpoint each). 23 of 25 configs fully "
        "resolved; the sweep is READY TO LAUNCH"
    ),
    "report_gate": (
        "a family reports only complete -- a two-point family is a line, "
        "not a shape. swin_b__scut_masked and vit_b16__scut_masked wait "
        "on their 768 points; so do two of twelve init contrasts"
    ),
    "graph_arms_wait_on": (
        "the sweep READING -- the sweep is the twelfth graph arm, run "
        "first"
    ),
    "transformer_arms_wait_on": "nothing -- zero new bands",
}


#: **[RECORDED 2026-08-11] The two ViT checkpoint picks, with their
#: rationales -- because both LOOK wrong on their face, and a later reader
#: must find the reasoning where the pick is.**
#:
#: **ViT-224: the rule coincidentally picked the higher scorer.** Four
#: candidates; filtering to the current unfused SHA (5b784842) leaves two
#: -- roadb-p6-vit-224-unfused at 0.8001 and -unfused-2 at 0.7648 -- and
#: FIRST BY LAUNCH ORDER selects the former. That it is also the higher of
#: the two is coincidence, not selection: the rule was registered before
#: the draws existed and never reads scores.
#:
#: **ViT-512: the pick is -unfused-2, and the "-2" is not a rule
#: violation.** The earlier -unfused at the same SHA OOMed on a GPU
#: fraction and produced NO checkpoint, so -unfused-2 is the first
#: COMPLETED run at the current SHA -- exactly what the rule names. A
#: reader seeing "-2" without this record would reasonably suspect the
#: rule was bent; it was not, and the OOM is the reason.
VIT_DRAW_PICKS = {
    "recorded": "2026-08-11, by the draw-pick rule",
    "vit_224": {
        "picked": "roadb-p6-vit-224-unfused",
        "pcc": 0.8001,
        "ckpt_prefix": "b89bebcc",
        "candidates_at_sha": 2,
        "note": (
            "the rule COINCIDENTALLY selects the higher scorer of the two "
            "(0.8001 over 0.7648) -- launch order, registered before the "
            "draws existed, never reads scores"
        ),
    },
    "vit_512": {
        "picked": "roadb-p6-vit-512-unfused-2",
        "ckpt_prefix": "68ffadee",
        "note": (
            "'-2' is not a rule violation: -unfused at the same SHA OOMed "
            "on a GPU fraction and produced no checkpoint, so -unfused-2 "
            "is the first COMPLETED run at the current SHA -- what the "
            "rule names"
        ),
    },
    "sha": "5b784842",
}


def phase7_extraction_sets() -> list[dict]:
    """The 24 sets, as data -- one entry per (backbone, resolution, init),
    each naming its checkpoint source, storage kind, and reuse. Like
    ``embedding_plan.required_sets``: the extraction configs must enumerate
    exactly these, asserted by test when they exist."""
    sets = []
    for backbone in PHASE_6_STRUCTURE["backbones"]:
        kind = "graph" if backbone in ("srgnn", "agnet") else "transformer"
        for resolution in RESOLUTIONS:
            # The task's own init vocabulary (embeddings.expected_variant):
            # "scut_masked" resolves to the masked_g1 checkpoint variant.
            for init in ("imagenet", "scut_masked"):
                reuses = None
                if (backbone, resolution, init) == ("swin_b", 224, "scut_masked"):
                    reuses = (
                        "road A's swin masked-G1 224 set: checkpoint "
                        "bitwise-identical (measured), extraction path "
                        "unchanged -- identity, not assumption"
                    )
                sets.append({
                    "backbone": backbone,
                    "resolution": resolution,
                    "init": init,
                    "geometry": "g1",
                    "stores": "feature_maps" if kind == "graph" else "pooled_vectors",
                    "checkpoint": (
                        None if init == "imagenet"
                        else f"the {backbone} masked-G1 {resolution} cell's "
                             "checkpoint (ViT: per the draw-pick rule)"
                    ),
                    "init_source": (
                        f"init_{backbone}_v1 (declared snapshot)"
                        if init == "imagenet" else "the checkpoint"
                    ),
                    "reuses": reuses,
                })
    return sets


#: **[RECORDED 2026-08-11] Which of the ten staged artifacts Phase 7's
#: matched design consumes: TWO -- and the other eight are recorded as
#: unconsumed rather than left implicit.**
#:
#: The matched ladder stages square G1 at three resolutions. Its three
#: staging inputs are ``staged_v1`` (Road A's 224 square, which is NOT one
#: of the ten -- the ten deliberately exclude it, TWO_TWENTY_FOUR_NON_
#: SQUARE), plus ``roadb_512_square_g1_v1`` and ``roadb_768_square_g1_v1``.
#: So: three staging inputs, two of the ten.
#:
#: **The eight unconsumed, each with why it exists and what could still
#: consume it:**
#:
#: * **512/768 square G2 (2)**: built for the geometry axis. Consuming them
#:   in a matched design needs masked-G2 pretraining cells, which were not
#:   among the twelve -- a contingent future decision, not an oversight.
#: * **non-square, all six**: the ViT-only sub-axis -- the clean pad pair
#:   at 224 (recorded in the artifacts' own manifests) and the
#:   zero-content question at G2 non-square including the eighth cell.
#:   Unconsumed by the resolution ladder BY DESIGN. **One nuance the
#:   nondeterminism finding adds**: TRAINED ViT arms are clouds, but the
#:   pad pair can run as FROZEN imagenet arms -- frozen extraction is
#:   deterministic and head fits carry only the measured linear-probe band
#:   -- so the pad question remains READABLE in the frozen regime even
#:   though ViT pretraining is not reproducible.
#:
#: Unconsumed is not waste: every artifact stands gated and hashed for the
#: registered question it was built to answer.
PHASE_7_STAGED_CONSUMPTION = {
    "recorded": "2026-08-11",
    "consumed": (
        "staged_v1 (Road A's 224 square G1 -- not one of the ten), "
        "roadb_512_square_g1_v1, roadb_768_square_g1_v1"
    ),
    "of_the_ten": "two consumed; eight unconsumed, recorded not implicit",
    "unconsumed": {
        "square_g2": (
            "512/768 square G2: geometry-axis artifacts; matched "
            "consumption needs masked-G2 pretraining cells not among the "
            "twelve -- contingent, not an oversight"
        ),
        "non_square": (
            "all six: the ViT-only sub-axis (clean pad pair, zero-content "
            "question, the eighth cell). Unconsumed by the resolution "
            "ladder BY DESIGN -- and the pad pair stays READABLE as FROZEN "
            "imagenet arms, where extraction is deterministic and only the "
            "linear-probe band applies, despite ViT pretraining being "
            "irreproducible"
        ),
    },
    "not_waste": (
        "every artifact stands gated and hashed for the registered "
        "question it was built to answer"
    ),
}


#: **[LIMITATION, RECORDED FIRST 2026-08-10] Road A's ViT arms carry the
#: same nondeterminism -- their figures are DRAWS, not fixed values.**
#:
#: ViT masked-G1 at 0.7893 is one draw from a distribution with roughly
#: 0.03 spread; so are ViT original at 0.8914 and ViT masked-G2 at 0.8306.
#: **Anything read against those figures inherits it**: the Q1 and Q2 rows
#: for ViT, and the geometry comparison that gave 0.7893 against 0.8306 --
#: a 0.041 delta against a ~0.030 nondeterminism spread, a much thinner
#: margin than it appeared.
#:
#: This goes to LIMITATIONS, not to fixes -- the apparatus is frozen at
#: phase3-freeze and the discipline holds: Road A's numbers stand as
#: measurements of what ran; what changes is the precision a reader may
#: attach to the ViT ones. Swin, SR-GNN and AG-Net are unaffected (swin
#: reproduces bitwise-adjacent; the graphs sit inside their recorded
#: regimes).
ROAD_A_VIT_ARMS_ARE_DRAWS = {
    "recorded": "2026-08-10, before anything else was built on the finding",
    "draws": {
        "vit_masked_g1": 0.7893, "vit_original": 0.8914, "vit_masked_g2": 0.8306,
    },
    "spread": "~0.030, from three same-config runs",
    "inherits": (
        "the ViT Q1 and Q2 rows, and the geometry comparison 0.7893 vs "
        "0.8306 -- 0.041 against a ~0.030 spread, thinner than it appeared"
    ),
    "discipline": (
        "a LIMITATION, not a fix: the freeze holds, the numbers stand as "
        "measurements of what ran; the precision a reader may attach to "
        "them is what changed"
    ),
    "unaffected": "swin (four-decimal reproduction), srgnn and agnet (in regime)",
}


#: **[DECIDED 2026-08-09] The twelve Phase 6 cells, and the order they
#: launch.** All four gates discharged; the configs are generated from Road
#: A's counterparts with the task block VERBATIM -- since gate 2 the builder
#: derives input size from the data, so "same procedure, different input"
#: is byte equality, asserted by test.
#:
#: **ViT at 512 launches FIRST.** It is the arm the frozen cliff was
#: measured on and the one the whole branch turns on: stable and in-band,
#: and the other eleven are the same path with different arguments; not,
#: and that is learned on one run rather than twelve. It is also the direct
#: test of ``SWIN_RESOLUTION_PREDICTION``'s trained-regime reading --
#: whether adapting the position grid repairs what interpolating it broke.
#:
#: **The 768 transformer cells are the heaviest runs in the project** --
#: ViT: 2,304 tokens with quadratic attention; Swin: padded 7-windows over
#: a 192x192 grid -- over 3,300 images for 30 fixed epochs. Costed in the
#: configs, not discovered at the queue.
#:
#: **The 224 cells re-run Road A's cells under the Road B SHA** -- same
#: artifact, same procedure -- so vit/swin/srgnn (bitwise in the image)
#: should reproduce Road A's checkpoints exactly: a free end-to-end
#: determinism check, the masked-rebuild pattern. AG-Net reproduces to
#: ~2.3e-05, its recorded property.
#:
#: **And each cell is its OWN seed regime** (PLAN 4.12: a trained
#: backbone's band is never inherited): no Road B Phase 7 arm reads
#: against a cell before that cell's band exists. Stated now so it is not
#: rediscovered at Phase 7.
PHASE_6_TWELVE_CELLS = {
    "decided": "2026-08-09, all four gates discharged",
    "cells": "4 backbones x {224, 512, 768}, masked G1, native scheme for graphs",
    "launch_first": (
        "roadb_p6_pretrain_vit_b16_masked_g1_512 -- the cliff arm; the "
        "branch turns on it, and it tests the trained-regime prediction "
        "directly"
    ),
    "heaviest": (
        "the 768 transformer cells: ViT 2,304 tokens quadratic, Swin "
        "padded 7-windows over 192x192, 3,300 images x 30 epochs -- costed "
        "in the configs"
    ),
    "224_cells": (
        "re-runs of Road A's cells under the Road B SHA; bitwise "
        "reproduction expected for vit/swin/srgnn, ~2.3e-05 for agnet -- a "
        "free end-to-end determinism check"
    ),
    "seed_bands": (
        "each cell is its own regime (PLAN 4.12); no Road B Phase 7 arm "
        "reads against a cell before that cell's band exists"
    ),
}


#: **[CORRECTED 2026-08-09, gate 2 -- another R10 instance prevented rather
#: than counted] The pretrain path never told the factory its input size.**
#: ``TorchPretrainModel.reset`` built ``create_backbone(name, ...)`` with no
#: ``input_size``, so a 512/768 pretraining run would have died at the
#: ViT/Swin 224 asserts -- after submission, with real weights loaded. The
#: fix is the extraction pattern: the training features carry their own
#: shape, and the builder consumes it; 224 keeps the historical path.
#:
#: **[MEASURED 2026-08-09, locally; the gate config confirms on the pinned
#: image]** All four backbones build through the pretrain path at
#: 224/512/768, the optimiser registers every parameter, and the trainable
#: count is IDENTICAL across sizes per backbone -- resizing adds and drops
#: nothing learned: vit_b16 85,799,425; swin_b 86,744,249; srgnn
#: 32,488,811; agnet 30,335,173.
#:
#: **The scope split, so a pass is never read as more than it is**: the
#: gate covers construction and optimiser registration. It does NOT cover
#: training stability at 512/768 with interpolated position grids -- that
#: is Phase 6's first pretraining run, an experiment with its own record.
PRETRAIN_BUILDER_THREADS_INPUT_SIZE = {
    "corrected": "2026-08-09, gate 2",
    "fault": (
        "TorchPretrainModel.reset built with no input_size; a 512/768 "
        "pretraining run dies at the ViT/Swin 224 asserts after submission"
    ),
    "fix": (
        "the training features carry their own (N, H, W, 3) shape and the "
        "builder consumes it -- the extract_embeddings pattern; 224 stays "
        "byte-identical"
    ),
    "measured": {
        "vit_b16": 85_799_425, "swin_b": 86_744_249,
        "srgnn": 32_488_811, "agnet": 30_335_173,
    },
    "parameters_size_invariant": (
        "identical counts at 224/512/768 per backbone -- resizing adds and "
        "drops nothing learned"
    ),
    "scope": {
        "covered": "construction + optimiser registration, all 12 cells",
        "not_covered": (
            "training stability at 512/768 with interpolated position "
            "grids -- Phase 6's first pretraining run, not a gate"
        ),
    },
}


MASKED_SCUT_REBUILD_WAS_A_DETERMINISM_CHECK = {
    "measured": "2026-08-09, two independent runs at different SHAs",
    "payload_rollups_identical": {512: "52273b8d...", 768: "c514727b..."},
    "what_differed": "only MANIFEST.json, which is what changed",
    "what_it_shows": (
        "the masked staging path reproduces byte-identically across runs "
        "and SHAs -- an unplanned gate-1-shaped result on a path the "
        "determinism gates never covered"
    ),
}


GRAPH_GEOMETRY_AT_RESOLUTION = {
    #: **[MEASURED 2026-08-09, local stack; the pinned gate run confirms]
    #: Gate 3: neither graph backbone is excluded from the resolution axis
    #: -- and the question "do 27 and 37 regions still land where they are
    #: specified" resolves DIFFERENTLY per backbone, which is the finding.**
    "measured": (
        "2026-08-09, torch cu126 + RTX 3060 locally; "
        "roadb_p6_graph_geometry_gate is the pinned-image confirmation"
    ),
    "srgnn": {
        "answer": (
            "YES, and input-size-INVARIANT: 27 regions at 224/512/768, the "
            "fixed-pixel box buffer untouched by input size, forward OK at "
            "every size. The feature map is upsampled to a FIXED 42x42 "
            "before any box touches it, so region geometry cannot move"
        ),
        "maps": {224: (7, 7), 512: (16, 16), 768: (24, 24)},
        "interpolation_into_42": {224: 6.0, 512: 2.62, 768: 1.75},
        "composition": (
            "the CleftGNN parallel, now measured: upsampling the map to 42 "
            "compensates for coarseness, and feeding more pixels makes the "
            "map genuinely denser -- at 224 the 42x42 is 6.0x interpolated, "
            "at 768 only 1.75x. The two mechanisms COMPOSE, exactly as the "
            "brief argued they were complementary"
        ),
    },
    "agnet": {
        "answer": (
            "YES, and resolution-COVARIANT by specification: regions are "
            "image-space (SIFT + seeded GMM, normalised boxes), so they are "
            "SPECIFIED to follow the image. Measured: 36 srs boxes -> 37 "
            "model regions at every size, all bounds in [0, 1], and "
            "placement measurably NOT identical across sizes -- the "
            "property, confirmed, not a defect. Gating on placement "
            "identity would have refused the design"
        ),
        #: **[RESOLVED 2026-08-09] The 36-vs-37 gap was neither an
        #: off-by-one nor a missing cluster -- it was the gate's own report
        #: comparing two quantities under one name (R2, self-caught on
        #: review).** ``generate_srs`` returns the SIFT+GMM boxes WITHOUT
        #: the whole image; pooling appends it. 36 = the full kappa(kappa+1)/2
        #: exactly (one missing cluster would give 28), and 36 + 1 = 37 =
        #: ``region_count()``. Measured at all three sizes.
        #:
        #: **And the law is CONDITIONAL, so Road A's "constant by
        #: construction" was too strong**: measured, keypoint-starved
        #: content collapses to 1 srs box -> 2 regions (flat and near-flat
        #: images), and the fallback constants plus the batch-padding
        #: machinery exist because counts vary per image.
        #: ``agnet.region_count``'s docstring now carries the
        #: qualification.
        "thirty_six_vs_thirty_seven": {
            "resolved": "2026-08-09",
            "was": (
                "the gate report compared generate_srs's pre-append count "
                "against the post-append law under one name -- R2, in the "
                "gate's own report"
            ),
            "not": (
                "an off-by-one (the law holds exactly: 36 + 1 = 37) and "
                "not a cluster shortfall (7 clusters would give 28 boxes, "
                "not 36)"
            ),
            "law_is_conditional": (
                "exact only on content with enough SIFT keypoints -- "
                "measured: flat content gives 1 srs box -> 2 regions. "
                "'Constant by construction' is qualified in "
                "agnet.region_count's docstring; the batch padding exists "
                "because counts vary"
            ),
        },
    },
    "local_wheel_defect": (
        "the laptop's torchvision wheel registers roi_align's CPU kernel "
        "under the CUDA dispatch key -- CPU raises NotImplementedError, "
        "CUDA rejects CUDA tensors. AG-Net's forward mechanics were "
        "verified locally with a substitute pure-PyTorch crop-resize. "
        "**DISCHARGED 2026-08-09**: the pinned-image gate run executed the "
        "REAL roi_align on CUDA -- device 'cuda', forward [2, 1] at all "
        "three sizes -- so the substituted half is now measured on the "
        "true path and the caveat is discharged rather than carried"
    ),
    "pinned_confirmation": (
        "2026-08-09: gate 3 PASSED on the pinned image -- SR-GNN 27 "
        "regions and invariant boxes at 7x7/16x16/24x24 -> 42x42 "
        "(6.0/2.62/1.75x), AG-Net normalised covariant regions with the "
        "real roi_align on CUDA at all sizes"
    ),
    "consequence": (
        "the resolution axis keeps all four backbones -- gates 1 and 3 "
        "both passed on the pinned image. Remaining: gate 2, the pretrain "
        "builder"
    ),
}


SWIN_RESOLUTION_PREDICTION = {
    "registered": "2026-08-09, before any Phase 6 run",
    "measured_contrast": (
        "ViT interpolates a learned position grid at build time; Swin "
        "interpolates nothing -- byte-equal checkpoint at every size"
    ),
    "prediction": (
        "if the cliff is build-time interpolation of learned positional "
        "parameters, Swin survives the resolution change where frozen ViT "
        "collapsed"
    ),
    "frozen_regime": {
        "test": "a frozen swin probe at 512/768, the ViT sweeps' analogue",
        "supports": "swin near its own 224 level",
        "refutes": (
            "swin dropping like ViT's ~0.14 -- the attribution fails and "
            "the cause afflicts both: token/window count, pooling dilution, "
            "or texture statistics at the wrong scale"
        ),
    },
    "trained_regime": {
        "test": "Phase 6's arms at three resolutions",
        "supports": "both architectures recover toward 224-level",
        "refutes": (
            "a backbone still cliffed after adapting its positional "
            "machinery -- the damage was never positional for it"
        ),
    },
    #: **[CONFIRMED 2026-08-11] The trained-regime reading held.**
    #: Registered 2026-08-09, before any Phase 6 run: trained arms recover
    #: toward 224-level. Measured: swin 0.8719 -> 0.8662 at 512 -- a -0.006
    #: shrug where frozen ViT fell 0.146. Adapting the positional machinery
    #: during pretraining removes the cliff, which is what the cliff being
    #: positional predicted. (The frozen-regime probe was never run -- the
    #: trained result answered the branch question first, and nothing now
    #: hangs on it.)
    "confirmed": {
        "date": "2026-08-11",
        "registered": "2026-08-09, before any Phase 6 run",
        "reading": "trained_regime.supports",
        "measured": (
            "swin 0.8719 -> 0.8662 at 512 against frozen ViT's 0.146 cliff; "
            "agnet and srgnn flat across the axis"
        ),
        "frozen_regime_status": "never run; superseded by the trained answer",
        # [CORRECTED 2026-08-12] "never run" went stale the day the
        # Phase 7 arms landed. The frozen-regime test as registered is "a
        # frozen swin probe at 512/768, the ViT sweeps' analogue" -- and
        # that is exactly what the swin imagenet arms ARE (frozen
        # backbone, declared init, head-only fit over precomputed
        # embeddings). It was run, by a route this record did not
        # anticipate, and it SUPPORTS: 0.1076 -> 0.2050 -> 0.1537, above
        # its own 224 at both sizes, where ViT fell -0.158 in the same
        # design. The registered supports-branch said "near its own 224
        # level"; the observed gain exceeds what was predicted, and
        # PHASE_7_SWIN_512 keeps those two apart.
        "frozen_regime_answered_after_all": {
            "corrected": "2026-08-12",
            "by": "the Road B Phase 7 swin imagenet arms",
            "reading": "frozen_regime.supports",
            "measured": "0.1076 -> 0.2050 -> 0.1537, both non-224 clearing",
            "against": "ViT -0.158 over the same endpoints, same design",
            "predicted_vs_observed": (
                "the branch predicted 'near its own 224 level' -- the "
                "absence of a collapse. The GAIN was not predicted; "
                "PHASE_7_SWIN_512 separates the two"
            ),
        },
    },
}


#: **[DESIGNED 2026-08-09] Road B Phase 6's structure, as data, before
#: anything is built.**
#:
#: **The axis**: 224/512/768, square, both geometries, same arm structure as
#: Road A's Phase 6 consumed selectively (brief §4). Non-square stays the
#: ViT-only sub-axis and is NOT part of the resolution claim.
#:
#: **SCUT staging at 512 and 768**: the frozen composition at a new size
#: argument -- stage, then unwarp for G2 -- exactly the reuse pattern Phase
#: 2 ran. **[MEASURED] Every SCUT image is 350x350** (the repo's own record,
#: ``scut/synthesis.py``), so at 512 ALL 5500 upsample 1.46x and at 768 ALL
#: upsample 2.19x. Not a mixed population -- a uniformly interpolated one --
#: and the consequence must be stated wherever the axis is: **SCUT at
#: 512/768 carries no texture information beyond ~350px; what
#: higher-resolution pretraining adapts is the positional geometry** (the
#: grid the cliff is about), while the cleft arms then supply genuinely
#: higher-resolution detail (median source longer side 2758). That
#: asymmetry is a design property, and supervision should see it.
#:
#: **Seed counts** come from the resolved planning band: 0.021425, so five
#: seeds resolve 0.027 (``SEED_BAND_RESOLVED``). The 4.12 caveat stands: a
#: TRAINED backbone is a third seed regime, and each pretraining cell's
#: band is a Phase 6 gate of its own.
#:
#: **Gates before any cell runs, each named now so no run discovers it**:
#:
#: 1. the swin build confirms on pinned timm 1.0.7 (``SWIN_AT_RESOLUTION``);
#: 2. the pretrain path's builder must thread ``input_size`` the way
#:    extraction now does -- extraction is confirmed at 512/768 on the
#:    pinned image (the Phase 3 sweeps ARE that confirmation), the TRAINING
#:    path has never built a non-224 model and is exactly R10's surface;
#: 3. the graph backbones' region geometry at the new staging sizes --
#:    regions map through content-box fractions so the pixel boxes scale,
#:    but the region-crop -> Xception path has never run there, and the
#:    recorded CNN danger is silently changed ROI geometry;
#: 4. SCUT staging sheets at the new sizes, reviewed before anything
#:    downstream -- the Phase 2 lesson, and Phase 2's review caught a
#:    defect every test missed.
PHASE_6_STRUCTURE = {
    "designed": "2026-08-09",
    "axis": "224/512/768, square, both geometries; non-square excluded",
    "backbones": ("vit_b16", "swin_b", "srgnn", "agnet"),
    "scut_staging": {
        "composition": "the frozen stage -> unwarp path at the new size",
        "source_size": "every SCUT image is 350x350 (scut/synthesis.py)",
        "upsampling": {512: "all 5500, 1.46x", 768: "all 5500, 2.19x"},
        "consequence": (
            "uniformly interpolated, not mixed: SCUT at 512/768 carries no "
            "texture information beyond ~350px, so higher-resolution "
            "pretraining adapts the positional geometry rather than finer "
            "detail -- while the cleft arms then supply genuine detail "
            "(median source 2758). A design property supervision should "
            "see"
        ),
        "write_up_guard": (
            "'we pretrained at 768' invites the reading that the model "
            "learned fine detail, and it did not -- it adapted positional "
            "geometry, which is exactly what the cliff is about, so the "
            "design still tests what it needs to. Say so wherever a "
            "higher-resolution pretraining is quoted, and supervision sees this "
            "BEFORE the runs"
        ),
        "artifacts": "per setting per size, ARTIFACT_PER_SETTING's pattern",
    },
    "build_order": (
        "1. the swin build gate (roadb_p6_swin_build_gate) -- the one thing "
        "that could refute the four-backbone conclusion, and a short run; "
        "2. SCUT staging at 512 and 768, then its sheets for review; "
        "3. the pretraining path with input_size threaded"
    ),
    "seed_counts": (
        "from SEED_BAND_RESOLVED: five seeds resolve 0.027. Each trained "
        "cell re-measures its own band -- a trained backbone is a third "
        "seed regime (PLAN 4.12's caveat), so the band is a Phase 6 gate"
    ),
    "gates_before_any_run": (
        "swin build on pinned timm",
        "pretrain builder threads input_size (training never built non-224)",
        "graph region geometry at the new staging sizes",
        "SCUT staging sheets reviewed before anything downstream",
    ),
}


def phase6_resolution_cells() -> list[dict]:
    """The Phase 6 resolution cells, as data -- one per backbone per
    resolution, each carrying its own gate.

    Like ``staging_settings()``: derived, so a config cannot invent a cell,
    and every cell names what must pass before it runs rather than leaving a
    run to discover it.
    """
    gates = {
        "vit_b16": (
            "extraction confirmed at this size on the pinned image (the "
            "Phase 3 sweeps); the TRAINING path has never built a non-224 "
            "model -- gate 2 of PHASE_6_STRUCTURE"
        ),
        "swin_b": (
            "builds at img_size with identical parameters "
            "(SWIN_AT_RESOLUTION); pending pinned-timm confirmation -- "
            "gate 1 of PHASE_6_STRUCTURE"
        ),
        "srgnn": (
            "region-crop -> Xception path never run at this staging size "
            "-- gate 3 of PHASE_6_STRUCTURE"
        ),
        "agnet": (
            "region-crop -> Xception path never run at this staging size "
            "-- gate 3 of PHASE_6_STRUCTURE"
        ),
    }
    cells = []
    for backbone in PHASE_6_STRUCTURE["backbones"]:
        for resolution in RESOLUTIONS:
            cells.append({
                "backbone": backbone,
                "resolution": resolution,
                "status": (
                    "road A's own regime" if resolution == CONTROL_RESOLUTION
                    else f"pending: {gates[backbone]}"
                ),
            })
    return cells


#: **[MEASURED 2026-08-08] The trapezium IS resolution-invariant, asserted
#: rather than assumed.**
#:
#: Specified as fractions (``TOP_HALF_WIDTH`` 0.301, ``BOT_HALF_WIDTH`` 0.500)
#: and evaluated on a normalised grid, so it should be -- and it is. Realised
#: half-widths from the mask:
#:
#:     224   top 0.30357   bot 0.50000
#:     512   top 0.30078   bot 0.50000
#:     768   top 0.30078   bot 0.50000
#:
#: The top error shrinks with resolution as pixel quantisation predicts; the
#: bottom is exact because 0.5 lands on a pixel boundary at every size.
TRAPEZIUM_IS_RESOLUTION_INVARIANT = {
    "measured": "2026-08-08",
    "specified": {"top": 0.301, "bot": 0.500},
    "realised": {
        224: {"top": 0.30357, "bot": 0.50000},
        512: {"top": 0.30078, "bot": 0.50000},
        768: {"top": 0.30078, "bot": 0.50000},
    },
    "tolerance_rule": "1.5 / size -- pixel quantisation on an inclusive edge",
    "why_asserted": (
        "a mask specified in fractions SHOULD be resolution-invariant, which "
        "is a hope until it is a test"
    ),
}


#: **[MEASURED 2026-08-08] Road A's 224 pixel residual. ONE source, and the
#: gate reads it from here.**
#:
#: Measured this session by unwarping a mark at a known offset and comparing
#: its recovered centre against ``unwarp_x`` -- so it is **Road A's measured
#: pixel residual, not an assumption**. It lives here rather than in a probe
#: because the monotonicity gate needs it and no Road B run produces it:
#: recomputing 224 inside every Road B run would stage Road A's artifact eight
#: times to get a number that already exists.
#:
#: ``roadb_staging.pixel_asymmetry_residual(224)`` reproduces it, and a test
#: asserts that, so the record cannot drift from the code that made it.
ROAD_A_224_PIXEL_RESIDUAL = {
    "measured": "2026-08-08",
    "provenance": "[MEASURED] -- Road A's staged geometry, not an assumption",
    "size": 224,
    "aspect": "square",
    "residual": 3.29e-03,
    "method": (
        "unwarp a mark at a known offset, compare its recovered centre "
        "against trapezium.unwarp_x"
    ),
    "why_recorded_rather_than_recomputed": (
        "the gate needs it and no Road B run produces it; recomputing 224 in "
        "every Road B run would stage Road A's artifact eight times for a "
        "number that already exists"
    ),
    "reproduced_by": "roadb_staging.pixel_asymmetry_residual(224)",
}


#: **[MEASURED 2026-08-08] The residual tracks COLUMN COUNT, not nominal size
#: -- which is why the ordering claim is within a family and not across.**
#:
#:     379 columns   2.245e-03    (non-square 512 on the PRE-rounding path)
#:     512 columns   1.401e-03
#:     768 columns   9.136e-04
#:
#: Non-square staging at 512 has roughly 379 columns, so its residual is
#: naturally LARGER than square 512's. Comparing them would report fewer
#: columns as a defect. **The ordering is about resolution within a family**
#: -- 512-square-G1 to 768-square-G1 is the comparison; 512-square-G1 to
#: 768-nonsquare-G2 is not.
#:
#: **[LABELLED 2026-08-09] 379 is a measurement of the PRE-rounding path.**
#: It was taken as 512 x 0.74 before ``SHAPE_IS_FIXED_AT_STAGING`` rounded
#: each dimension to a multiple of 16, so a production non-square 512
#: artifact records ``residual_columns`` ~384 -- the median staged width
#: after rounding -- not 379. The row stays because it is true of what it
#: measured and it is what established the within-family rule; a mismatch
#: against a production manifest is the rounding, not drift.
#:
#: **[CORRECTED 2026-08-09] "Two families have only two points" was true
#: when written and was superseded the same day** by
#: ``TWO_TWENTY_FOUR_NON_SQUARE``: the 224 non-square cells give both
#: non-square families their third point, from their OWN stagings rather
#: than Road A's square 224. All four families now run three points,
#: asserted by ``test_every_family_has_three_points_after_the_224_non_square_cells``.
RESIDUAL_FAMILIES = {
    "measured": "2026-08-08",
    "grouped_by": ("aspect", "geometry"),
    "why_within_family": (
        "the residual tracks column count: non-square 512 has ~379 columns "
        "and a larger residual than square 512, which is fewer columns and "
        "not a defect"
    ),
    "by_columns": {379: 2.245e-03, 512: 1.401e-03, 768: 9.136e-04},
    "by_columns_measured_on": (
        "[LABELLED 2026-08-09] the PRE-rounding path: 379 = 512 x 0.74 "
        "before SHAPE_IS_FIXED_AT_STAGING's multiple-of-16 rounding. A "
        "production non-square 512 manifest records residual_columns ~384; "
        "the mismatch against this row is the rounding, not drift"
    ),
    "square_families_have_three_points": (
        "224 -> 512 -> 768, because Road A staged square at 224"
    ),
    "non_square_families": (
        "[CORRECTED 2026-08-09] '512 -> 768 only' was superseded the day it "
        "was written: TWO_TWENTY_FOUR_NON_SQUARE added the 224 non-square "
        "cells, so both non-square families have three points from their "
        "own stagings"
    ),
}


#: **[DECIDED 2026-08-08] EIGHT artifacts, one per setting -- not one
#: ``roadb_v1`` with eight subdirectories. The reason is guard 3's coupling.**
#:
#: Guard 3 hash-verifies a declared input's ROLLUP. With a single artifact, a
#: Phase 6 config consuming only ``512_nonsquare_g2`` still declares the whole
#: rollup -- so re-running any one of the eight changes it, and **every config
#: declaring it fails guard 3, including ones whose own setting never
#: changed**. That means re-pasting hashes across configs for a change that
#: did not touch them, which is exactly the friction that trains people to
#: paste without reading.
#:
#: Eight artifacts means a config declares only what it consumes, and a
#: partial re-run moves only that hash.
#:
#: **This differs from masked SCUT deliberately, and the difference is
#: consumption rather than taste.** ``scut/masked.py`` holds G1 and G2
#: together and was right to: they are always built together and consumed
#: together, so the whole thing is one input and splitting it would create two
#: hashes that must move in lockstep. Road B's eight cells will be consumed
#: SELECTIVELY -- Phase 6 will not pretrain at all eight -- and the resolution
#: axis is the thing most likely to need a partial re-run.
#:
#: Recorded because a future session will see two artifacts structured
#: differently and should know it was a decision.
ARTIFACT_PER_SETTING = {
    "decided": "2026-08-08",
    "layout": "data/staged/roadb_<resolution>_<aspect>_<geometry>_v1/",
    "not": "one roadb_v1 with eight subdirectories",
    "reason": (
        "guard 3 verifies the ROLLUP, so one artifact couples all eight: "
        "re-running any cell breaks guard 3 for every config declaring it, "
        "including configs whose own setting never changed"
    ),
    "consequence": (
        "a config declares only what it consumes, and a partial re-run moves "
        "only that hash"
    ),
    "differs_from_masked_scut_because": (
        "masked SCUT's G1 and G2 are always built AND consumed together, so "
        "one input is right there. Road B's cells are consumed selectively -- "
        "Phase 6 will not pretrain at all eight -- and the resolution axis is "
        "the most likely thing to need a partial re-run"
    ),
    "per_artifact_manifest": (
        "each carries its OWN pixel residual and content fraction, so a "
        "setting's gate result travels with the setting rather than in a "
        "shared summary that a partial re-run would leave stale"
    ),
}


#: **[DECIDED 2026-08-08] Requirements carried from Road A's Phase 2. None
#: optional, and each earned its place.**
#:
#: **The asymmetry-preservation gate** for any unwarping variant. Road A
#: verified G2 to machine epsilon before use, and every G2 arm rests on it.
#:
#: **Per-patient flagging of upsampled images**, with the counts asserted:
#: 7 of 473 at 512, 81 of 473 at 768. Without the flag the 768 arm silently
#: mixes interpolated and downsampled patients.
#:
#: **A contact sheet, reviewed before anything downstream runs.** Every phase
#: where a sheet was skipped produced a defect no test caught -- and Phase 8's
#: sheet review is the reason its framing finding was caught at all, first as
#: an apparent effect and then as an artefact.
#:
#: **Parity measured, not claimed from a shared code path.** A shared
#: implementation is an argument that two things SHOULD agree; the measurement
#: is whether they do.
#:
#: **And the content fraction per setting, in the artifact.** So the geometry
#: axis is visible rather than inferred, and so the per-setting saliency
#: expectation exists where a later phase can read it rather than assume one.
PHASE_2_REQUIREMENTS = {
    "asymmetry_gate": "for any unwarping variant, to machine epsilon as Road A did",
    "flag_upsampled": "per patient, counts asserted: 7 at 512, 81 at 768 of 473",
    "contact_sheet": "reviewed before anything downstream runs",
    "parity": "measured against the recorded cleft geometry, not claimed",
    "content_fraction": "per setting, in the artifact",
}

#: **[DECIDED 2026-08-08] What non-square changes structurally, and it is not
#: only the pad.**
#:
#: With no pad, ``pad_fraction`` is zero and the content box IS the image. Two
#: downstream consumers read those fields and must behave at zero rather than
#: divide by it:
#:
#: * ``gradcam_sheet.out_of_content`` -- already handled: it returns ``None``
#:   with a reason when the expectation is zero, because there is no
#:   out-of-content region for mass to fall into. The statistic does not
#:   exist rather than evaluating to something.
#: * ``phase7c``'s protection mask -- reads the content box to place region
#:   boxes. At zero pad the box is the full frame, which is correct, but it
#:   has never been exercised there.
ZERO_PAD_CONSUMERS = {
    "gradcam_sheet.out_of_content": (
        "HANDLED -- returns None with a reason at a zero expectation"
    ),
    "phase7c protection mask": (
        "reads the content box; correct at full-frame but never exercised "
        "there. Check before any Road B augmentation arm runs"
    ),
}


#: **[MEASURED 2026-08-09] The staged artifact cannot serve all four
#: backbones at non-square. This is structural, not a parameter choice.**
#:
#:     Swin-B      asserts input == 224x224 EXACTLY; no rounding reaches it
#:     ViT-B/16    multiples of 16, dynamic sizing, 113-197 tokens
#:     Xception    accepts anything; feature map 7x4 / 7x5 / 7x6 / 7x7 by width
#:     ResNet-50   same
#:
#: The rounding constant was the wrong question. Swin is not on a divisibility
#: axis at all -- ``swin_base_patch4_window7_224`` asserts equality with its
#: declared size, so 128, 144, 160, 192 and 208 all fail identically.
#:
#: **The CNNs accepting everything is WORSE than refusing.** A 27-region
#: scheme specified as fractions of the feature map lands on 28 cells instead
#: of 49, and nothing raises. And **224x144 and 224x160 both produce a 7x5
#: map**, so two different staged inputs give one ROI geometry -- which makes
#: it clearly wrong rather than merely different.
BACKBONE_SHAPE_CONSTRAINTS = {
    "measured": "2026-08-09",
    "swin_b": "asserts input == 224x224 exactly; no rounding reaches it",
    "vit_b16": "multiples of 16, dynamic sizing, 113-197 tokens",
    "xception": "accepts anything; map 7x4 / 7x5 / 7x6 / 7x7 by width",
    "resnet50": "same as xception",
    "cnns_are_the_dangerous_case": (
        "a 27-region scheme in fractions lands on 28 cells instead of 49 and "
        "nothing raises; 224x144 and 224x160 both give 7x5, so two staged "
        "inputs share one ROI geometry"
    ),
}


#: **[DECIDED 2026-08-09] Non-square is ViT-ONLY. Square stays four-backbone.**
#:
#: Three options, and two are worse:
#:
#: * **Swin with a dynamic-size variant** -- a different model, therefore a
#:   different arm, which is a worse confound than the restriction.
#: * **Let the CNNs run** -- they would produce numbers with silently altered
#:   region geometry, which is the dangerous option rather than the
#:   permissive one.
#: * **ViT-only** -- taken. ``ARTIFACT_PER_SETTING`` already anticipated
#:   selective consumption, so nothing else changes.
#:
#: **What it costs, both named:**
#:
#: The eighth cell answers *"does zero non-content help"* on **one backbone
#: rather than four**.
#:
#: The clean pad pair is **ViT-only, at 224, where rounding distorts most**
#: (<=6.5% against ~1.9% at 768). It is the best available pad comparison on
#: one architecture -- **not one factor apart in general**, and a later reader
#: must not take that phrase literally.
#:
#: **Road B's main axis is unaffected.** 224/512/768 SQUARE remains
#: four-backbone. Non-square becomes a ViT-only sub-axis testing the pad
#: question specifically.
NON_SQUARE_IS_VIT_ONLY = {
    "decided": "2026-08-09",
    "square_cells": "four-backbone, unaffected",
    "non_square_cells": "ViT-B/16 only",
    "rejected": {
        "swin_dynamic_variant": "a different model, so a different arm -- a worse confound",
        "let_the_cnns_run": "silently altered region geometry; the dangerous option",
    },
    "costs": [
        "the eighth cell answers 'does zero non-content help' on one backbone, not four",
        "the clean pad pair is ViT-only at 224, where rounding distorts most -- "
        "the best available pad comparison on one architecture, NOT one factor "
        "apart in general",
    ],
    "main_axis_unaffected": "224/512/768 square stays four-backbone",
}


#: **[OPEN 2026-08-09] The the intent given at supervision cannot be honoured as stated,
#: and the choice goes back for ruling with the measurement.**
#:
#: The intent was to keep **each image's original proportions**, not merely to
#: avoid padding. **No backbone consumes variable shapes, and mixed shapes
#: cannot be batched** -- so per-patient proportions are not reachable.
#:
#: Two approximations, and this project should not pick between them alone:
#:
#: * **round each patient to a multiple of 16** (implemented) -- closest to
#:   the intent, distortion <=6.5% at 224 falling to ~1.9% at 768;
#: * **one common non-square shape for everyone** -- resizes each patient
#:   toward it, so the distortion is larger and varies with how far a patient
#:   sits from the common ratio.
#:
#: The first is implemented because something had to be, not because the
#: question is settled.
ORIGINAL_PROPORTIONS_ARE_NOT_REACHABLE = {
    "opened": "2026-08-09",
    "intent": "keep each image's original proportions, not merely avoid padding",
    "why_unreachable": (
        "no backbone consumes variable shapes and mixed shapes cannot be "
        "batched"
    ),
    "approximations": {
        "round_to_16": "implemented; <=6.5% at 224, ~1.9% at 768",
        "one_common_shape": "resizes each patient toward it; larger and uneven",
    },
    "status": "goes back to supervision WITH the measurement, not decided here",
}
