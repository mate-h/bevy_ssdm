use crate::node::{
    init_ssdm_post_pipeline, ssdm_gbuffer_warp, ssdm_post_process, ssdm_pyramid_pass,
    ssdm_vector_pass,
};
use crate::queue::{
    extract_ssdm_vector_surfaces, queue_ssdm_vector_meshes, SsdmVectorMainEntities,
};
use crate::render_targets::plug_extract_pyramids;
use crate::settings::{SsdmSettingsPlugin, SsdmView};
use crate::vector_material::SsdmVectorMaterial;
use crate::vector_phase::{extract_ssdm_vector_phases, SsdmVector3d};
use bevy::camera::Camera3d;
use bevy::core_pipeline::{
    core_3d::main_transparent_pass_3d,
    deferred::node::late_deferred_prepass,
    schedule::{Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy::pbr::{DefaultOpaqueRendererMethod, DrawMaterial, MaterialPlugin};
use bevy::post_process::bloom::bloom;
use bevy::render::render_phase::{AddRenderCommand, BinnedRenderPhasePlugin, DrawFunctions};
use bevy::render::render_resource::TextureUsages;
use bevy::render::RenderDebugFlags;
use bevy::{
    prelude::*,
    render::{Render, RenderApp, RenderStartup, RenderSystems},
};

pub struct SsdmPlugin;

impl Plugin for SsdmPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DefaultOpaqueRendererMethod::deferred())
            .add_plugins(SsdmSettingsPlugin)
            .add_plugins(MaterialPlugin::<SsdmVectorMaterial>::default())
            .register_type::<crate::marker::SsdmVectorSurface>()
            .add_plugins(BinnedRenderPhasePlugin::<
                SsdmVector3d,
                bevy::pbr::MeshPipeline,
            >::new(RenderDebugFlags::default()))
            // The depth warp inside `ssdm_gbuffer_warp` mirrors the warped scratch depth
            // back into both `prepass.depth.texture.texture` and `view_depth_texture` so post-
            // prepass consumers (deferred lighting, atmosphere `render_sky`, `MainOpaquePass`,
            // legacy `Skybox`, …) see the same warped depth. `view_depth_texture` only declares
            // `RENDER_ATTACHMENT (+ COPY_SRC` when `DepthPrepass` is set), so we need to opt
            // into `COPY_DST` ourselves before `prepare_core_3d_depth_textures` runs in the
            // render world. Mirrors the OIT plugin's `configure_depth_texture_usages` pattern.
            .add_systems(Last, configure_ssdm_depth_texture_usages);

        plug_extract_pyramids(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SsdmVectorMainEntities>()
            .init_resource::<DrawFunctions<SsdmVector3d>>()
            .add_render_command::<SsdmVector3d, DrawMaterial>()
            .add_systems(
                bevy::render::ExtractSchedule,
                (extract_ssdm_vector_phases, extract_ssdm_vector_surfaces),
            )
            .add_systems(
                Render,
                queue_ssdm_vector_meshes.in_set(RenderSystems::QueueMeshes),
            );

        render_app
            .add_systems(RenderStartup, init_ssdm_post_pipeline)
            .add_systems(
                Core3d,
                (
                    ssdm_vector_pass,
                    ssdm_pyramid_pass,
                    ssdm_gbuffer_warp,
                )
                    .chain()
                    .after(late_deferred_prepass)
                    .before(main_transparent_pass_3d)
                    .in_set(Core3dSystems::Prepass),
            )
            .add_systems(
                Core3d,
                ssdm_post_process
                    .after(main_transparent_pass_3d)
                    .before(bloom)
                    .before(tonemapping)
                    .in_set(Core3dSystems::PostProcess),
            );
    }
}

/// Ensures `view_depth_texture` is allocated with `COPY_DST` on every camera that runs SSDM,
/// so the depth-warp pass in [`crate::node::ssdm_gbuffer_warp`] can write the warped
/// scratch depth back into the canonical view depth (the texture that `main_opaque_pass_3d`
/// binds as its depth-stencil and that the atmosphere's `render_sky` samples at binding 13).
/// Without this, `bevy_core_pipeline::core_3d::prepare_core_3d_depth_textures` allocates the
/// texture with only `RENDER_ATTACHMENT (+ COPY_SRC` if `DepthPrepass` is present), and the
/// `copy_texture_to_texture` into `view_depth_texture` would fail validation.
fn configure_ssdm_depth_texture_usages(mut cameras: Query<&mut Camera3d, With<SsdmView>>) {
    for mut camera_3d in &mut cameras {
        let mut usages = TextureUsages::from(camera_3d.depth_texture_usages);
        if usages.contains(TextureUsages::COPY_DST) {
            continue;
        }
        usages |= TextureUsages::COPY_DST;
        camera_3d.depth_texture_usages = usages.into();
    }
}
