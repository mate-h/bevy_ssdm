#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_ssdm::resolve::resolve_src_uv

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var b0: texture_2d<f32>;
@group(0) @binding(2) var deferred_tex: texture_2d<u32>;
@group(0) @binding(3) var lighting_tex: texture_2d<u32>;
// Pyramid A level 0 - unfiltered per-pixel forward displacement, used by `resolve_src_uv`
// to do its inline Newton polish step on top of the B0 read.
@group(0) @binding(4) var a0: texture_2d<f32>;

struct MrtOut {
    @location(0) deferred: vec4<u32>,
    @location(1) lighting_pass_id: u32,
}

fn src_pixel(src_uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let df = vec2<f32>(dims);
    let p = src_uv * df;
    let mx = max(df - vec2<f32>(1.0), vec2<f32>(0.0));
    let c = clamp(p, vec2<f32>(0.0), mx);
    return vec2<i32>(floor(c));
}

@fragment
fn warp_mrt_fs(iv: FullscreenVertexOutput) -> MrtOut {
    // `resolve_src_uv` does discontinuity-aware B0 sampling (avoids the rim-halo bleed
    // from bilinear B0 across the silhouette discontinuity, most visible at grazing
    // angles) AND one inline Newton polish step against A0 (tightens convergence at the
    // silhouette so it tracks the actual heightmap detail rather than the coarse
    // averages walked through by the refine pass).
    let src_uv = resolve_src_uv(iv.uv, b0, a0, samp);
    let d_def = textureDimensions(deferred_tex);
    let d_l = textureDimensions(lighting_tex);
    let pd = src_pixel(src_uv, d_def);
    let pl = src_pixel(src_uv, d_l);
    var o: MrtOut;
    // Packed deferred gbuffer and the deferred_lighting_pass_id are u32-typed and not
    // filterable; nearest `textureLoad` is the only option at the source. Robust B0
    // sampling does the heavy lifting of avoiding cross-discontinuity bleed at the warp
    // input; the gbuffer fetch itself is correct as long as the resolved src_uv lands on
    // the silhouette interior.
    o.deferred = textureLoad(deferred_tex, pd, 0);
    o.lighting_pass_id = textureLoad(lighting_tex, pl, 0).x;
    return o;
}
