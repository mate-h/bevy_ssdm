#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_ssdm::resolve::resolve_src_uv

@group(0) @binding(0) var samp_uv: sampler;
@group(0) @binding(1) var b0: texture_2d<f32>;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
// Pyramid A level 0; needed by `resolve_src_uv` for its inline Newton polish step. Reused
// with the `samp_uv` linear sampler at binding 0.
@group(0) @binding(4) var a0: texture_2d<f32>;

@fragment
fn depth_warp_fs(iv: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    // Discontinuity-aware B0 + one Newton polish step against A0; see `ssdm_resolve.wgsl`.
    // The polish matters extra here: the silhouette / depth boundary is the most
    // gradient-heavy region of B0, and a residual error of even a couple of source pixels
    // in the resolved `src_uv` lands the depth fetch on the wrong side of the rim.
    let src_uv = resolve_src_uv(iv.uv, b0, a0, samp_uv);
    // Depth itself is fetched with the nearest sampler (`depth_samp`). Linear filtering
    // of depth across the source rim averages surface depth with far-plane (1.0 in
    // reverse-Z) and produces non-physical depth that confuses `MainOpaquePass` ordering,
    // atmosphere `render_sky` compositing, and TAA / post effects that read depth.
    return textureSample(depth_tex, depth_samp, src_uv);
}
