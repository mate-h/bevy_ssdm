//! Queue [`super::vector_phase::SsdmVector3d`] draws for [`super::marker::SsdmVectorSurface`] meshes.

use crate::marker::SsdmVectorSurface;
use crate::vector_phase::SsdmVector3d;
use bevy::core_pipeline::core_3d::{Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::prelude::*;
use bevy::mesh::Mesh3d;
use bevy::pbr::{
    MainPassOpaqueDrawFunction, PendingMeshMaterialQueues, PreparedMaterial,
    RenderMaterialInstances, RenderMeshInstances, SpecializedMaterialPipelineCache,
};
use bevy::platform::collections::HashSet;
use bevy::prelude::{Deref, DerefMut, Entity};
use bevy::render::{
    batching::gpu_preprocessing::GpuPreprocessingSupport,
    camera::DirtySpecializations,
    erased_render_asset::ErasedRenderAssets,
    mesh::allocator::MeshAllocator,
    render_phase::{BinnedRenderPhaseType, ViewBinnedRenderPhases},
    sync_world::MainEntity,
    view::visibility::RenderVisibleEntities,
    view::ExtractedView,
};
use bevy::material::RenderPhaseType;

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
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_mesh_material_queues: ResMut<PendingMeshMaterialQueues>,
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

        let Some(render_visible_mesh_entities) = visible_entities.get::<Mesh3d>() else {
            continue;
        };

        let Some(view_pending_mesh_material_queues) =
            pending_mesh_material_queues.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        for &main_entity in dirty_specializations.iter_to_dequeue(
            view.retained_view_entity,
            render_visible_mesh_entities,
        ) {
            ssdm_phase.remove(main_entity);
        }

        for (render_entity, visible_entity) in dirty_specializations.iter_to_queue(
            view.retained_view_entity,
            render_visible_mesh_entities,
            &view_pending_mesh_material_queues.prev_frame,
        ) {
            if !vector_entities.0.contains(visible_entity) {
                continue;
            }

            let Some(pipeline_id) = view_specialized.get(visible_entity).copied() else {
                continue;
            };

            let Some(material_instance) = render_material_instances.instances.get(visible_entity)
            else {
                view_pending_mesh_material_queues
                    .current_frame
                    .insert((*render_entity, *visible_entity));
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                view_pending_mesh_material_queues
                    .current_frame
                    .insert((*render_entity, *visible_entity));
                continue;
            };
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                view_pending_mesh_material_queues
                    .current_frame
                    .insert((*render_entity, *visible_entity));
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

            let Some(mesh_slabs) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id()) else {
                continue;
            };

            let batch_set_key = Opaque3dBatchSetKey {
                pipeline: pipeline_id,
                draw_function,
                material_bind_group_index: Some(material.binding.group.0),
                slabs: mesh_slabs,
                lightmap_slab: mesh_instance
                    .lightmap_slab_index()
                    .map(|index| *index),
            };
            let bin_key = Opaque3dBinKey {
                asset_id: mesh_instance.mesh_asset_id().into(),
            };

            ssdm_phase.add(
                batch_set_key,
                bin_key,
                (*render_entity, *visible_entity),
                mesh_instance.current_uniform_index,
                BinnedRenderPhaseType::mesh(
                    mesh_instance.should_batch(),
                    &gpu_preprocessing_support,
                ),
            );
        }
    }
}
