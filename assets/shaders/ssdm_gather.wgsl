#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_ssdm::resolve::resolve_src_uv

struct SsdmGatherUniforms {
    enabled: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> gather_u: SsdmGatherUniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var scene: texture_2d<f32>;
@group(0) @binding(3) var b_zero: texture_2d<f32>;
// Pyramid A level 0; needed by `resolve_src_uv` for its inline Newton polish step.
@group(0) @binding(4) var a_zero: texture_2d<f32>;

@fragment
fn gather_fs(iv: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    if (gather_u.enabled == 0u) {
        return textureSample(scene, samp, iv.uv);
    }
    // Discontinuity-aware B0 + one Newton polish step against A0; see `ssdm_resolve.wgsl`.
    // Plain bilinear B0 here would smear a foreground UV with a background UV at the warp
    // boundary and fetch a random part of the lit scene; the polish then makes the gather
    // track the actual heightmap detail at the silhouette rather than its coarse average.
    let src_uv = resolve_src_uv(iv.uv, b_zero, a_zero, samp);
    return textureSample(scene, samp, src_uv);
}
