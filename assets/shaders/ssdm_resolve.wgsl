#define_import_path bevy_ssdm::resolve

// Discontinuity-aware bilinear sampling of Pyramid B level 0 (the resolved source-UV field).
//
// Why not just `textureSample(b0, samp, uv).xy`?
//
// At grazing angles the field stored in B0 has a sharp discontinuity at the warped
// silhouette: just inside it points at a real source rim pixel, just outside it points at
// near-identity (no seed got there from the vector pass). A linear sampler whose 2x2
// footprint straddles that boundary blends those two unrelated UVs and produces a "ghost"
// source UV that fetches from a random part of the source buffer, which surfaces as the
// rim halo / shimmer / fringe artifacts.
//
// This helper:
//   1. Computes the plain bilinear blend (cheap, gives sub-pixel quality on smooth
//      interior where it stays correct).
//   2. `textureLoad`s the 4 nearest B0 texels - no filter cross-bleed.
//   3. Measures their Chebyshev spread; if it's below the discontinuity threshold (~2
//      texels) the field is locally smooth and we use the bilinear blend directly.
//   4. Otherwise runs a cluster vote: each of the 4 samples gets a count of how many of
//      the *other* 3 lie within `THRESHOLD/2 * texel`, and the result is the bilinear-
//      weighted average over the dominant cluster. Ties (e.g. 2-2 silhouette splits) fall
//      out as the cluster with the highest bilinear weight - i.e. whichever side of the
//      silhouette the destination pixel center actually sits on, which is exactly what we
//      want for sharp warped silhouettes.
//
// `b0` is bound as `texture_2d<f32>` storing per-pixel resolved source UVs in `.xy`. The
// `samp_lin` sampler must have `Linear` mag/min filtering. `textureLoad` doesn't consult
// the sampler, so no extra binding is required.
fn sample_b0_robust(uv: vec2<f32>, b0: texture_2d<f32>, samp_lin: sampler) -> vec2<f32> {
    let dims_u = textureDimensions(b0);
    let dims_i = vec2<i32>(dims_u);
    let dims_f = vec2<f32>(dims_u);
    let texel = 1.0 / dims_f;
    let max_idx = dims_i - vec2(1, 1);

    let bilinear = textureSampleLevel(b0, samp_lin, uv, 0.0).xy;

    // 4 corner texels around the destination uv, matching the bilinear footprint exactly.
    // The +/-0.5 shift is the standard texel-center -> integer-index conversion.
    let p = uv * dims_f - 0.5;
    let pi = vec2<i32>(floor(p));
    let f = p - vec2<f32>(pi);

    let p00 = clamp(pi + vec2(0, 0), vec2(0, 0), max_idx);
    let p10 = clamp(pi + vec2(1, 0), vec2(0, 0), max_idx);
    let p01 = clamp(pi + vec2(0, 1), vec2(0, 0), max_idx);
    let p11 = clamp(pi + vec2(1, 1), vec2(0, 0), max_idx);

    let s00 = textureLoad(b0, p00, 0).xy;
    let s10 = textureLoad(b0, p10, 0).xy;
    let s01 = textureLoad(b0, p01, 0).xy;
    let s11 = textureLoad(b0, p11, 0).xy;

    // Chebyshev spread of the 4 corner samples in source-UV space.
    let lo = min(min(s00, s10), min(s01, s11));
    let hi = max(max(s00, s10), max(s01, s11));
    let spread = max(hi.x - lo.x, hi.y - lo.y);

    // 2-source-texel discontinuity threshold: comfortably above any sub-pixel noise inside
    // a converged region but well below the displacement magnitude we expect at any
    // visible warped silhouette.
    let thr = 2.0 * max(texel.x, texel.y);
    if (spread < thr) {
        return bilinear;
    }

    // Cluster vote: each corner counts neighbours within thr/2.
    let h = 0.5 * thr;
    let n00 =
        u32(distance(s00, s10) < h)
        + u32(distance(s00, s01) < h)
        + u32(distance(s00, s11) < h);
    let n10 =
        u32(distance(s10, s00) < h)
        + u32(distance(s10, s01) < h)
        + u32(distance(s10, s11) < h);
    let n01 =
        u32(distance(s01, s00) < h)
        + u32(distance(s01, s10) < h)
        + u32(distance(s01, s11) < h);
    let n11 =
        u32(distance(s11, s00) < h)
        + u32(distance(s11, s10) < h)
        + u32(distance(s11, s01) < h);
    let nmax = max(max(n00, n10), max(n01, n11));

    // Bilinear weights inside the dominant cluster. With nmax==3 (all 4 agree) this just
    // reduces to bilinear, with nmax==2 we keep the 3-of-4 majority cluster, and with
    // nmax==1 (2-2 silhouette split) we naturally pick the side with the higher bilinear
    // weight - i.e. the side the destination pixel center is closer to.
    let w00 = (1.0 - f.x) * (1.0 - f.y);
    let w10 = f.x * (1.0 - f.y);
    let w01 = (1.0 - f.x) * f.y;
    let w11 = f.x * f.y;

    var num = vec2(0.0);
    var den = 0.0;
    if (n00 == nmax) { num = num + w00 * s00; den = den + w00; }
    if (n10 == nmax) { num = num + w10 * s10; den = den + w10; }
    if (n01 == nmax) { num = num + w01 * s01; den = den + w01; }
    if (n11 == nmax) { num = num + w11 * s11; den = den + w11; }

    if (den > 1e-6) {
        return num / den;
    }
    return bilinear;
}

// Resolves the source UV at `uv` for any of the warp / gather passes. This is the
// recommended entry point for resolve passes: it does the discontinuity-aware B0 read
// (`sample_b0_robust`) followed by one Newton polish step against Pyramid A level 0.
//
// The polish matters because the refine pass builds B0 *coarse-to-fine*, and at the
// coarse levels Pyramid A is heavily averaged - any sharp local maximum in the heightmap
// (a rim spike, a crack, a chip) gets diluted into the surrounding texels and the seed
// passed to the next-finer level under-shoots that maximum. The fine-level refinement
// catches some of it back, but a residual error `delta` of a few source pixels remains,
// most visible at the warped silhouette where it shows up as the silhouette under-tracking
// the heightmap.
//
// The polish step `src_uv' = uv - V(B0(uv))` is one Newton iteration of the same fixed
// point the refine pass solves, but evaluated against the *unfiltered* per-pixel A0 (no
// coarse averaging). If the refine produced `B0(uv) = q + delta`, the polish gives
// `q - grad(V) * delta`: the residual is multiplied by `grad(V)`, which for typical
// heightmaps is well below 1, so a single step tightens convergence dramatically and
// makes the silhouette track the actual heightmap detail instead of its coarse average.
//
// `a0` must be Pyramid A level 0 (the unfiltered displacement vector field written by the
// vector pass).
fn resolve_src_uv(
    uv: vec2<f32>,
    b0: texture_2d<f32>,
    a0: texture_2d<f32>,
    samp_lin: sampler,
) -> vec2<f32> {
    let seed = sample_b0_robust(uv, b0, samp_lin);
    let v = textureSampleLevel(a0, samp_lin, seed, 0.0).xy;
    return uv - v;
}
