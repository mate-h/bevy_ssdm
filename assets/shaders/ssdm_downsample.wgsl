#import bevy_core_pipeline::fullscreen_vertex_shader::{
    FullscreenVertexOutput,
    fullscreen_vertex_shader,
}

struct SsdmDownUniforms {
    in_dims: vec2<u32>,
    out_dims: vec2<u32>,
}

@group(0) @binding(0) var<uniform> u: SsdmDownUniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;

@fragment
fn downsample_fs(iv: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let Ow = max(u.out_dims.x, 1u);
    let Oh = max(u.out_dims.y, 1u);
    let Iw = max(u.in_dims.x, 1u);
    let Ih = max(u.in_dims.y, 1u);
    let p = iv.uv * vec2(f32(Ow), f32(Oh));
    let ox = floor(p.x);
    let oy = floor(p.y);
    let base = vec2(ox * 2.0 + 0.5, oy * 2.0 + 0.5);
    var acc = vec4(0.0);
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let sp = base + vec2(f32(dx), f32(dy));
            let suv = sp / vec2(f32(Iw), f32(Ih));
            acc += textureSample(src, samp, suv);
        }
    }
    return acc * 0.25;
}
