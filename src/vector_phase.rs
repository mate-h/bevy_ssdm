//! Binned phase for SSDM displacement-vector pass (rendered into pyramid A level 0).

use crate::settings::SsdmView;
use bevy::camera::{Camera, Camera3d};
use bevy::core_pipeline::core_3d::{Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy::ecs::prelude::*;
use bevy::platform::collections::HashSet;
use bevy::render::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    render_phase::{
        BinnedPhaseItem, CachedRenderPipelinePhaseItem, PhaseItem, PhaseItemExtraIndex,
        ViewBinnedRenderPhases,
    },
    render_resource::CachedRenderPipelineId,
    sync_world::MainEntity,
    view::{NoIndirectDrawing, RetainedViewEntity},
    Extract,
};
use core::ops::Range;

/// Phase item mirroring deferred [`bevy_core_pipeline::core_3d::Opaque3d`] batching for the vector pass.
pub struct SsdmVector3d {
    pub batch_set_key: Opaque3dBatchSetKey,
    #[allow(dead_code)]
    pub bin_key: Opaque3dBinKey,
    pub representative_entity: (Entity, MainEntity),
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for SsdmVector3d {
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    fn draw_function(&self) -> bevy::render::render_phase::DrawFunctionId {
        self.batch_set_key.draw_function
    }

    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for SsdmVector3d {
    type BatchSetKey = Opaque3dBatchSetKey;
    type BinKey = Opaque3dBinKey;

    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        SsdmVector3d {
            batch_set_key,
            bin_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for SsdmVector3d {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}

pub fn extract_ssdm_vector_phases(
    mut phases: ResMut<ViewBinnedRenderPhases<SsdmVector3d>>,
    cameras: Extract<
        Query<(Entity, &Camera, Has<NoIndirectDrawing>), (With<Camera3d>, With<SsdmView>)>,
    >,
    mut live: Local<HashSet<RetainedViewEntity>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    live.clear();
    for (main_entity, camera, no_indirect_drawing) in &cameras {
        if !camera.is_active {
            continue;
        }
        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        let retained = RetainedViewEntity::new(main_entity.into(), None, 0);
        phases.prepare_for_new_frame(retained, gpu_preprocessing_mode);
        live.insert(retained);
    }
    phases.retain(|e, _| live.contains(e));
}
