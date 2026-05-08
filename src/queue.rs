//! Queue [`super::vector_phase::SsdmVector3d`] draws for [`super::marker::SsdmVectorSurface`] meshes.

use crate::marker::SsdmVectorSurface;
use crate::vector_phase::SsdmVector3d;
use bevy::core_pipeline::core_3d::{Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::prelude::*;
use bevy::mesh::Mesh3d;
use bevy::pbr::{
    MainPassOpaqueDrawFunction, PreparedMaterial, RenderMaterialInstances, RenderMeshInstances,
    RenderPhaseType, SpecializedMaterialPipelineCache,
};
use bevy::platform::collections::HashSet;
use bevy::prelude::{Deref, DerefMut};
use bevy::render::{
    batching::gpu_preprocessing::GpuPreprocessingSupport, erased_render_asset::ErasedRenderAssets,
    mesh::allocator::MeshAllocator, render_phase::ViewBinnedRenderPhases, sync_world::MainEntity,
    view::visibility::RenderVisibleEntities, view::ExtractedView,
};

#[derive(Resource, Default, Deref, DerefMut)]
pub struct SsdmVectorMainEntities(pub HashSet<MainEntity>);

pub fn extract_ssdm_vector_surfaces(
    q: bevy::render::Extract<Query<Entity, With<SsdmVectorSurface>>>,
    mut out: ResMut<SsdmVectorMainEntities>,
) {
    out.0.clear();
    for e in &q {
        out.0.insert(MainEntity::from(e));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn queue_ssdm_vector_meshes(
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    render_material_instances: Res<RenderMaterialInstances>,
    render_mesh_instances: Res<RenderMeshInstances>,
    mesh_allocator: Res<MeshAllocator>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    mut ssdm_phases: ResMut<ViewBinnedRenderPhases<SsdmVector3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    vector_entities: Res<SsdmVectorMainEntities>,
    specialized_material_pipeline_cache: Res<SpecializedMaterialPipelineCache>,
) {
    for (view, visible_entities) in &views {
        let Some(ssdm_phase) = ssdm_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(view_specialized) =
            specialized_material_pipeline_cache.get(&view.retained_view_entity)
        else {
            continue;
        };

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            if !vector_entities.0.contains(visible_entity) {
                continue;
            }

            let Some((current_change_tick, pipeline_id)) =
                view_specialized.get(visible_entity).map(|(t, p)| (*t, *p))
            else {
                continue;
            };

            if ssdm_phase.validate_cached_entity(*visible_entity, current_change_tick) {
                continue;
            }

            let Some(material_instance) = render_material_instances.instances.get(visible_entity)
            else {
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                continue;
            };

            if !matches!(
                material.properties.render_phase_type,
                RenderPhaseType::Opaque
            ) {
                continue;
            }

            let Some(draw_function) = material
                .properties
                .get_draw_function(MainPassOpaqueDrawFunction)
            else {
                continue;
            };

            let (vertex_slab, index_slab) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);

            let batch_set_key = Opaque3dBatchSetKey {
                pipeline: pipeline_id,
                draw_function,
                material_bind_group_index: Some(material.binding.group.0),
                vertex_slab: vertex_slab.unwrap_or_default(),
                index_slab,
                lightmap_slab: mesh_instance.shared.lightmap_slab_index.map(|i| *i),
            };
            let bin_key = Opaque3dBinKey {
                asset_id: mesh_instance.mesh_asset_id.into(),
            };

            ssdm_phase.add(
                batch_set_key,
                bin_key,
                (*render_entity, *visible_entity),
                mesh_instance.current_uniform_index,
                bevy::render::render_phase::BinnedRenderPhaseType::mesh(
                    mesh_instance.should_batch(),
                    &gpu_preprocessing_support,
                ),
                current_change_tick,
            );
        }
    }
}
