#import bevy_core_pipeline::fullscreen_vertex_shader::{
    FullscreenVertexOutput,
    fullscreen_vertex_shader,
}

@group(0) @binding(0) var src: texture_2d<f32>;

@fragment
fn blit_fs(iv: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(src);
    let p = vec2<i32>(
        i32(floor(iv.uv.x * f32(dims.x))),
        i32(floor(iv.uv.y * f32(dims.y))),
    );
    return textureLoad(src, p, 0);
}
