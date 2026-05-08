#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_ssdm::resolve::resolve_src_uv

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var b0: texture_2d<f32>;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
// Pyramid A level 0; needed by `resolve_src_uv` for its inline Newton polish step.
@group(0) @binding(3) var a0: texture_2d<f32>;

@fragment
fn warp_float_fs(iv: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Discontinuity-aware B0 + one Newton polish step against A0; see `ssdm_resolve.wgsl`.
    let src_uv = resolve_src_uv(iv.uv, b0, a0, samp);
    // Bilinear source fetch: at grazing angles the warp footprint covers many source
    // texels per destination pixel, and nearest `textureLoad` aliases the packed
    // octahedral normal and motion-vector source streams hard. `textureSampleLevel` with
    // the linear sampler bound at @binding(0) anti-aliases that, and is safe here because
    // `resolve_src_uv` already steered `src_uv` away from silhouette discontinuities so
    // the bilinear footprint stays inside one continuous surface region.
    return textureSampleLevel(src_tex, samp, src_uv, 0.0);
}
