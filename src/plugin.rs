use crate::node::{
    init_ssdm_post_pipeline, SsdmGBufferWarpPassNode, SsdmNode, SsdmPostProcessNode,
    SsdmPyramidPassNode, SsdmVectorPassNode,
};
use crate::queue::{
    extract_ssdm_vector_surfaces, queue_ssdm_vector_meshes, SsdmVectorMainEntities,
};
use crate::render_targets::plug_extract_pyramids;
use crate::settings::{SsdmSettingsPlugin, SsdmView};
use crate::vector_material::SsdmVectorMaterial;
use crate::vector_phase::{extract_ssdm_vector_phases, SsdmVector3d};
use bevy::camera::Camera3d;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::pbr::{DefaultOpaqueRendererMethod, DrawMaterial, MaterialPlugin};
use bevy::render::render_phase::{AddRenderCommand, BinnedRenderPhasePlugin, DrawFunctions};
use bevy::render::render_resource::TextureUsages;
use bevy::render::RenderDebugFlags;
use bevy::{
    prelude::*,
    render::{
        render_graph::{RenderGraph, RenderGraphError, RenderGraphExt, ViewNodeRunner},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
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
            // The depth warp inside `SsdmGBufferWarpPassNode` mirrors the warped scratch depth
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
            .add_systems(RenderStartup, patch_ssdm_render_graph_edges)
            .add_render_graph_node::<ViewNodeRunner<SsdmVectorPassNode>>(
                Core3d,
                SsdmNode::VectorPass,
            )
            .add_render_graph_node::<ViewNodeRunner<SsdmPyramidPassNode>>(Core3d, SsdmNode::Pyramid)
            .add_render_graph_node::<ViewNodeRunner<SsdmGBufferWarpPassNode>>(
                Core3d,
                SsdmNode::GBufferWarp,
            )
            .add_render_graph_node::<ViewNodeRunner<SsdmPostProcessNode>>(
                Core3d,
                SsdmNode::PostProcess,
            );
    }
}

fn patch_ssdm_render_graph_edges(mut graph: ResMut<RenderGraph>) {
    let Some(g) = graph.get_sub_graph_mut(Core3d) else {
        return;
    };

    let _ = g.remove_node_edge(Node3d::MainOpaquePass, SsdmNode::VectorPass);
    let _ = g.remove_node_edge(SsdmNode::VectorPass, Node3d::MainTransmissivePass);
    if let Err(e) = g.try_add_node_edge(Node3d::MainOpaquePass, Node3d::MainTransmissivePass) {
        match e {
            RenderGraphError::EdgeAlreadyExists(_) => {}
            _ => panic!("ssdm render graph: {e:?}"),
        }
    }

    let _ = g.remove_node_edge(Node3d::EndPrepasses, Node3d::StartMainPass);
    // Serial: vector field → pyramid B0 → warp all prepass targets before deferred lighting.
    // Note: if another plugin also inserts nodes between EndPrepasses and StartMainPass (e.g. SSAO),
    // you may need to re-order so effects that must read *unwarped* prepass run before this chain.
    g.add_node_edges((
        Node3d::EndPrepasses,
        SsdmNode::VectorPass,
        SsdmNode::Pyramid,
        SsdmNode::GBufferWarp,
        Node3d::StartMainPass,
    ));

    // The deferred lighting pass uses `DeferredLightingIdDepthTexture` (built by
    // `CopyDeferredLightingId`) as its depth-stencil with `Equal` compare, so the per-pixel
    // gate that decides whether the PBR lighting fragment runs is whatever
    // `prepass.deferred_lighting_pass_id` looked like when `CopyDeferredLightingId` ran. Bevy
    // schedules that copy in the prepass section (before `EndPrepasses`), which is *before*
    // our warp — so silhouette-extended pixels (where SSDM warps a non-zero PBR id into a
    // previously-empty pixel) would still be skipped by lighting. Move the copy to run after
    // our gbuffer warp so the gate reflects the warped lighting_pass_id.
    let _ = g.remove_node_edge(Node3d::LateDeferredPrepass, Node3d::CopyDeferredLightingId);
    let _ = g.remove_node_edge(Node3d::CopyDeferredLightingId, Node3d::EndPrepasses);
    let _ = g.try_add_node_edge(Node3d::LateDeferredPrepass, Node3d::EndPrepasses);
    let _ = g.remove_node_edge(SsdmNode::GBufferWarp, Node3d::StartMainPass);
    g.add_node_edges((
        SsdmNode::GBufferWarp,
        Node3d::CopyDeferredLightingId,
        Node3d::StartMainPass,
    ));

    let _ = g.remove_node_edge(Node3d::StartMainPassPostProcessing, Node3d::Tonemapping);
    let _ = g.remove_node_edge(Node3d::StartMainPassPostProcessing, Node3d::Bloom);
    g.add_node_edge(Node3d::StartMainPassPostProcessing, SsdmNode::PostProcess);

    let hdr_next = if g.get_node_state(Node3d::Bloom).is_ok() {
        Node3d::Bloom
    } else {
        Node3d::Tonemapping
    };
    g.add_node_edge(SsdmNode::PostProcess, hdr_next);
}

/// Ensures `view_depth_texture` is allocated with `COPY_DST` on every camera that runs SSDM,
/// so the depth-warp pass in [`crate::node::SsdmGBufferWarpPassNode`] can write the warped
/// scratch depth back into the canonical view depth (the texture that `MainOpaquePass3dNode`
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
