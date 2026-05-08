//! GPU textures for SSDM pyramids (one mip level per [`Image`]) on the main world.
//! Handles are extracted to the render world via [`ExtractResource`].

use bevy::{
    asset::{Assets, Handle, RenderAssetUsages},
    core_pipeline::{
        core_3d::CORE_3D_DEPTH_FORMAT,
        deferred::{DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT},
        prepass::{
            MotionVectorPrepass, NormalPrepass, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT,
        },
    },
    image::Image,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
};

use crate::settings::{SsdmSettings, SsdmView};

pub const SSDM_PYRAMID_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Fullscreen scratch targets matching Bevy deferred / prepass formats for warping before lighting.
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct SsdmGBufferScratch {
    pub deferred: Option<Handle<Image>>,
    pub lighting_pass_id: Option<Handle<Image>>,
    pub depth: Option<Handle<Image>>,
    pub normal: Option<Handle<Image>>,
    pub motion_vectors: Option<Handle<Image>>,
    pub width: u32,
    pub height: u32,
}

fn scratch_image(width: u32, height: u32, format: TextureFormat, usage: TextureUsages) -> Image {
    let mut img = Image::new_uninit(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.usage = usage;
    img
}

/// Handles for pyramid A (offsets) and B (refined source UV), per level 0 = full resolution.
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct SsdmPyramidImages {
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub a: Vec<Handle<Image>>,
    pub b: Vec<Handle<Image>>,
}

fn make_level_image(width: u32, height: u32) -> Image {
    Image::new_target_texture(width.max(1), height.max(1), SSDM_PYRAMID_FORMAT, None)
}

/// Resizes pyramid images when the SSDM camera viewport or settings change.
pub fn prepare_ssdm_pyramid_images(
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<SsdmPyramidImages>,
    cameras: Query<(&Camera, &SsdmSettings), (With<Camera3d>, With<SsdmView>)>,
) {
    let Ok((camera, settings)) = cameras.single() else {
        return;
    };
    if settings.enabled == 0 {
        return;
    };
    let Some(phys) = camera.physical_viewport_size() else {
        return;
    };
    let width = phys.x.max(1);
    let height = phys.y.max(1);
    let max_mip = settings.pyramid_levels.clamp(2, 8);
    let chain_len = max_mip as usize;

    let needs_rebuild = !(state.width == width
        && state.height == height
        && state.levels == max_mip
        && state.a.len() == chain_len);

    if !needs_rebuild {
        return;
    }

    for h in state.a.drain(..) {
        images.remove(h.id());
    }
    for h in state.b.drain(..) {
        images.remove(h.id());
    }

    let mut w = width;
    let mut h = height;
    state.width = width;
    state.height = height;
    state.levels = max_mip;

    for _ in 0..chain_len {
        state.a.push(images.add(make_level_image(w, h)));
        state.b.push(images.add(make_level_image(w, h)));
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
}

pub fn prepare_ssdm_gbuffer_scratch(
    mut images: ResMut<Assets<Image>>,
    mut scratch: ResMut<SsdmGBufferScratch>,
    pyramid: Res<SsdmPyramidImages>,
    cameras: Query<
        (
            &Camera,
            &SsdmSettings,
            Has<NormalPrepass>,
            Has<MotionVectorPrepass>,
        ),
        (With<Camera3d>, With<SsdmView>),
    >,
) {
    let Ok((camera, settings, has_normal, has_motion)) = cameras.single() else {
        return;
    };
    if settings.enabled == 0 || pyramid.a.is_empty() {
        clear_gbuffer_scratch(&mut images, &mut scratch);
        return;
    }
    let Some(phys) = camera.physical_viewport_size() else {
        return;
    };
    let width = phys.x.max(1);
    let height = phys.y.max(1);
    let want_normal = has_normal;
    let want_motion = has_motion;
    let needs_rebuild = scratch.width != width
        || scratch.height != height
        || scratch.deferred.is_none()
        || scratch.normal.is_some() != want_normal
        || scratch.motion_vectors.is_some() != want_motion;
    if !needs_rebuild {
        return;
    }
    clear_gbuffer_scratch(&mut images, &mut scratch);
    scratch.width = width;
    scratch.height = height;

    let color_usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    scratch.deferred = Some(images.add(scratch_image(
        width,
        height,
        DEFERRED_PREPASS_FORMAT,
        color_usage,
    )));
    scratch.lighting_pass_id = Some(images.add(scratch_image(
        width,
        height,
        DEFERRED_LIGHTING_PASS_ID_FORMAT,
        color_usage,
    )));
    scratch.depth = Some(images.add(scratch_image(
        width,
        height,
        CORE_3D_DEPTH_FORMAT,
        TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST,
    )));
    if has_normal {
        scratch.normal = Some(images.add(scratch_image(
            width,
            height,
            NORMAL_PREPASS_FORMAT,
            color_usage,
        )));
    }
    if has_motion {
        scratch.motion_vectors = Some(images.add(scratch_image(
            width,
            height,
            MOTION_VECTOR_PREPASS_FORMAT,
            color_usage,
        )));
    }
}

fn clear_gbuffer_scratch(images: &mut Assets<Image>, scratch: &mut SsdmGBufferScratch) {
    for opt in [
        scratch.deferred.take(),
        scratch.lighting_pass_id.take(),
        scratch.depth.take(),
        scratch.normal.take(),
        scratch.motion_vectors.take(),
    ] {
        if let Some(h) = opt {
            images.remove(h.id());
        }
    }
    scratch.width = 0;
    scratch.height = 0;
}

pub fn plug_extract_pyramids(app: &mut App) {
    app.init_resource::<SsdmPyramidImages>()
        .init_resource::<SsdmGBufferScratch>()
        .add_plugins(ExtractResourcePlugin::<SsdmPyramidImages>::default())
        .add_plugins(ExtractResourcePlugin::<SsdmGBufferScratch>::default())
        .add_systems(
            PostUpdate,
            (
                prepare_ssdm_pyramid_images,
                prepare_ssdm_gbuffer_scratch.after(prepare_ssdm_pyramid_images),
            ),
        );
}
