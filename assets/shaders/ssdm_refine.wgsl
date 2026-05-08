#import bevy_core_pipeline::fullscreen_vertex_shader::{
    FullscreenVertexOutput,
    fullscreen_vertex_shader,
}

struct SsdmRefineUniforms {
    dims: vec2<u32>,
    is_coarsest: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: SsdmRefineUniforms;
@group(0) @binding(1) var samp_a: sampler;
@group(0) @binding(2) var pyramid_a: texture_2d<f32>;
@group(0) @binding(3) var samp_b: sampler;
@group(0) @binding(4) var pyramid_b_coarse: texture_2d<f32>;

// Lobel (SSDM, Fig. 3–4): the per-level update solves the fixed point s = q − V(s),
// where q is the screen pixel UV (`center_uv`) and V is the projected forward
// displacement stored in pyramid A. We sample V at the four corners of the
// center texel cell (±half-texel) around the *seed* (the previous level's
// estimate, or q at the coarsest level), average those V samples, and write
//   bary = q − avg(V around seed).
// Anchoring at q (not at seed) is what makes the recurrence q_{n+1} = q − V(q_n)
// instead of s_{n+1} = s_n − V(s_n); the latter would re-subtract V at every
// level and drift, while this form preserves the paper's invariant that
// "no displacement ⇒ bary = current raster position" at every level and
// converges in one level under constant V. textureLoad avoids bilinear
// "mix of unrelated vectors" at exact grid corners.
fn load_a_texel(uv: vec2<f32>, w: u32, h: u32) -> vec2<f32> {
    let fw = f32(w);
    let fh = f32(h);
    let x = i32(clamp(floor(uv.x * fw), 0.0, fw - 1.0));
    let y = i32(clamp(floor(uv.y * fh), 0.0, fh - 1.0));
    return textureLoad(pyramid_a, vec2<i32>(x, y), 0).xy;
}

@fragment
fn refine_fs(iv: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let w = max(u.dims.x, 1u);
    let h = max(u.dims.y, 1u);
    let center_uv = iv.uv;

    var seed = center_uv;
    if (u.is_coarsest == 0u) {
        seed = textureSample(pyramid_b_coarse, samp_b, center_uv).xy;
    }

    let hx = 0.5 / f32(w);
    let hy = 0.5 / f32(h);
    let corners = array<vec2<f32>, 4>(
        seed + vec2(-hx, -hy),
        seed + vec2(hx, -hy),
        seed + vec2(-hx, hy),
        seed + vec2(hx, hy),
    );

    var v_acc = vec2(0.0);
    for (var i = 0u; i < 4u; i++) {
        let uv_c = clamp(corners[i], vec2(0.0), vec2(1.0));
        v_acc += load_a_texel(uv_c, w, h);
    }
    let avg_v = v_acc * 0.25;
    let bary = center_uv - avg_v;
    return vec4(bary, 0.0, 1.0);
}
