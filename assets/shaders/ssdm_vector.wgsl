#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#import bevy_pbr::mesh_functions::{
    get_world_from_local, mesh_normal_local_to_world, mesh_position_local_to_clip,
    mesh_position_local_to_world, mesh_tangent_local_to_world,
}
#import bevy_pbr::view_transformations::{ndc_to_uv, position_world_to_ndc}

struct SsdmVectorParams {
    displacement_scale: f32,
    _pad: vec3<f32>,
}

@group(3) @binding(0) var<uniform> params: SsdmVectorParams;
@group(3) @binding(1) var disp_tex: texture_2d<f32>;
@group(3) @binding(2) var disp_sampler: sampler;
// Bound to keep `SsdmVectorMaterial`'s AsBindGroup layout intact, but the
// displacement direction follows the geometric vertex normal per the SSDM paper
// ("projected normal vector × displacement map"), not the perturbed shading
// normal. The normal map participates only in the deferred lighting pass.
@group(3) @binding(3) var normal_tex: texture_2d<f32>;
@group(3) @binding(4) var normal_sampler: sampler;

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(v.instance_index);
    var out: VertexOutput;
    out.world_position = mesh_position_local_to_world(world_from_local, vec4(v.position, 1.0));
    out.position = mesh_position_local_to_clip(world_from_local, vec4(v.position, 1.0));
    out.world_normal = mesh_normal_local_to_world(v.normal, v.instance_index);
    out.uv = v.uv;
    let world_tangent = mesh_tangent_local_to_world(world_from_local, v.tangent, v.instance_index);
    out.world_tangent = vec4(world_tangent.xyz, world_tangent.w);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let nws = normalize(in.world_normal);

    let h = textureSample(disp_tex, disp_sampler, in.uv).x;
    let disp_world = nws * (h * params.displacement_scale);

    let p0 = in.world_position.xyz;
    let ndc0 = position_world_to_ndc(p0);
    let ndc1 = position_world_to_ndc(p0 + disp_world);
    let duv = ndc_to_uv(ndc1.xy) - ndc_to_uv(ndc0.xy);
    return vec4(duv, 0.0, 1.0);
}
