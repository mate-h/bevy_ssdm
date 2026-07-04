//! Displacement vector material: rendered into pyramid A0 between deferred prepass and lighting.

use bevy::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, ColorWrites, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::material::{AlphaMode, OpaqueRendererMethod};
use bevy::shader::ShaderRef;

const SSDM_VECTOR_SHADER: &str = "shaders/ssdm_vector.wgsl";

/// GPU uniform block for [`SsdmVectorMaterial`].
#[derive(Clone, Copy, ShaderType, Default, Debug)]
pub struct SsdmVectorParams {
    pub displacement_scale: f32,
    pub _pad: Vec3,
}

/// Material that outputs a two-channel screen-space offset (RGBA16F, .xy used) for SSDM pyramid A.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct SsdmVectorMaterial {
    #[uniform(0)]
    pub params: SsdmVectorParams,
    #[texture(1)]
    #[sampler(2)]
    pub displacement: Handle<Image>,
    #[texture(3)]
    #[sampler(4)]
    pub normal_map: Handle<Image>,
}

impl Material for SsdmVectorMaterial {
    fn fragment_shader() -> ShaderRef {
        SSDM_VECTOR_SHADER.into()
    }

    fn vertex_shader() -> ShaderRef {
        SSDM_VECTOR_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn opaque_render_method(&self) -> OpaqueRendererMethod {
        OpaqueRendererMethod::Deferred
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
            Mesh::ATTRIBUTE_TANGENT.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.depth_write_enabled = Some(false);
            // Match mesh / deferred prepass (reverse-Z): Bevy uses GreaterEqual everywhere here.
            // Less/ LessEqual disagrees with that test and can drop the whole pass vs prepass depth.
            ds.depth_compare = Some(CompareFunction::GreaterEqual);
        }
        if let Some(fs) = descriptor.fragment.as_mut() {
            for target in &mut fs.targets {
                if let Some(t) = target.as_mut() {
                    t.format = crate::render_targets::SSDM_PYRAMID_FORMAT;
                    t.write_mask = ColorWrites::ALL;
                }
            }
        }
        Ok(())
    }
}
