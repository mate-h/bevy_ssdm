//! SSDM Core3d render pass systems.

use bevy::camera::{MainPassResolutionOverride, Viewport};
use bevy::color::LinearRgba;
use bevy::core_pipeline::{
    deferred::{
        copy_lighting_id::DeferredLightingIdDepthTexture, DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
        DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT,
    },
    prepass::{ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT},
};
use bevy::ecs::prelude::World;
use bevy::math::UVec2;
use bevy::prelude::{AssetServer, Commands, Res, ResMut, Resource};
use bevy::render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_resource::{
        encase::{ShaderType, UniformBuffer},
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferInitDescriptor,
        BufferUsages, CachedRenderPipelineId, CompareFunction, DepthBiasState, DepthStencilState,
        LoadOp, Operations, PipelineCache, RenderPassColorAttachment,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, StencilState, StoreOp,
        TextureFormat,
    },
    renderer::{RenderContext, ViewQuery},
    texture::GpuImage,
    view::{ExtractedView, ViewDepthTexture, ViewTarget},
};

use crate::render_targets::{SsdmGBufferScratch, SsdmPyramidImages};
use crate::settings::SsdmSettings;
use crate::vector_phase::SsdmVector3d;

/// HDR view-target color format used by the gather pass (matches Bevy's default HDR target).
const GATHER_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[derive(Resource)]
pub struct SsdmPostPipeline {
    pub sampler: bevy::render::render_resource::Sampler,
    pub sampler_nearest: bevy::render::render_resource::Sampler,
    /// Strong handle to `assets/shaders/ssdm_resolve.wgsl` so naga_oil keeps the module's
    /// `#define_import_path bevy_ssdm::resolve` registered for the warp shaders that import
    /// `sample_b0_robust`. Drop it and any in-flight pipeline rebuild loses the helper.
    pub _resolve_lib: bevy::asset::Handle<bevy::shader::Shader>,
    pub down_layout: BindGroupLayoutDescriptor,
    pub refine_layout: BindGroupLayoutDescriptor,
    pub gather_layout: BindGroupLayoutDescriptor,
    pub gbuffer_warp_mrt_layout: BindGroupLayoutDescriptor,
    pub prepass_depth_warp_layout: BindGroupLayoutDescriptor,
    pub blit_rgba32_layout: BindGroupLayoutDescriptor,
    pub blit_r8_layout: BindGroupLayoutDescriptor,
    pub prepass_warp_float_layout: BindGroupLayoutDescriptor,
    pub prepass_blit_float_layout: BindGroupLayoutDescriptor,
    pub down_pipeline: CachedRenderPipelineId,
    pub refine_pipeline: CachedRenderPipelineId,
    pub gather_pipeline: CachedRenderPipelineId,
    pub gbuffer_warp_mrt_pipeline: CachedRenderPipelineId,
    pub prepass_depth_warp_pipeline: CachedRenderPipelineId,
    pub blit_rgba32_pipeline: CachedRenderPipelineId,
    pub blit_r8_pipeline: CachedRenderPipelineId,
    pub prepass_warp_normal_pipeline: CachedRenderPipelineId,
    pub prepass_warp_motion_pipeline: CachedRenderPipelineId,
    pub prepass_blit_normal_pipeline: CachedRenderPipelineId,
    pub prepass_blit_motion_pipeline: CachedRenderPipelineId,
    pub copy_lighting_id_layout: BindGroupLayoutDescriptor,
    pub copy_lighting_id_pipeline: CachedRenderPipelineId,
}

#[derive(ShaderType)]
struct DownUni {
    in_dims: UVec2,
    out_dims: UVec2,
}

#[derive(ShaderType)]
struct RefineUni {
    dims: UVec2,
    is_coarsest: u32,
    _pad: u32,
}

#[derive(ShaderType)]
struct GatherUni {
    enabled: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

fn run_ssdm_pyramid_passes(world: &World, render_context: &mut RenderContext) {
    let pipeline_cache = world.resource::<PipelineCache>();
    let post = world.resource::<SsdmPostPipeline>();
    let pyramid = world.resource::<SsdmPyramidImages>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();

    let Some(down_pl) = pipeline_cache.get_render_pipeline(post.down_pipeline) else {
        return;
    };
    let Some(refine_pl) = pipeline_cache.get_render_pipeline(post.refine_pipeline) else {
        return;
    };

    if pyramid.a.len() < 2 || pyramid.b.len() < pyramid.a.len() {
        return;
    }

    for i in 1..pyramid.a.len() {
        let in_img = gpu_images.get(&pyramid.a[i - 1]).unwrap();
        let out_img = gpu_images.get(&pyramid.a[i]).unwrap();
        let du = DownUni {
            in_dims: UVec2::new(in_img.size_2d().x, in_img.size_2d().y),
            out_dims: UVec2::new(out_img.size_2d().x, out_img.size_2d().y),
        };
        let mut ubuf = UniformBuffer::new(Vec::new());
        ubuf.write(&du).unwrap();
        let buf = render_context
            .render_device()
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("ssdm_down"),
                contents: ubuf.as_ref(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
        let bg = render_context.render_device().create_bind_group(
            Some("ssdm_down_bg"),
            &pipeline_cache.get_bind_group_layout(&post.down_layout),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler_nearest,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &in_img.texture_view,
                    ),
                },
            ],
        );
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("ssdm_downsample"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &out_img.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(down_pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    let l_count = pyramid.a.len();
    for l in (0..l_count).rev() {
        let a_img = gpu_images.get(&pyramid.a[l]).unwrap();
        let is_coarsest = u32::from(l == l_count - 1);
        let ru = RefineUni {
            dims: UVec2::new(a_img.size_2d().x, a_img.size_2d().y),
            is_coarsest,
            _pad: 0,
        };
        let mut ubuf = UniformBuffer::new(Vec::new());
        ubuf.write(&ru).unwrap();
        let buf = render_context
            .render_device()
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("ssdm_refine"),
                contents: ubuf.as_ref(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
        let b_coarse_view = if l + 1 < l_count {
            &gpu_images
                .get(&pyramid.b[l + 1])
                .expect("pyramid B mip must be prepared")
                .texture_view
        } else {
            &a_img.texture_view
        };
        let bg = render_context.render_device().create_bind_group(
            Some("ssdm_refine_bg"),
            &pipeline_cache.get_bind_group_layout(&post.refine_layout),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler_nearest,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &a_img.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler,
                    ),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        b_coarse_view,
                    ),
                },
            ],
        );
        let out_img = gpu_images.get(&pyramid.b[l]).unwrap();
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("ssdm_refine"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &out_img.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(refine_pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

pub fn ssdm_vector_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &bevy::render::view::ViewUniformOffset,
        Option<&MainPassResolutionOverride>,
        Option<&SsdmSettings>,
    )>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (
        camera,
        extracted_view,
        depth,
        _view_uniform_offset,
        resolution_override,
        ssdm_settings,
    ) = view.into_inner();
    let Some(settings) = ssdm_settings else {
        return;
    };
    if settings.enabled == 0 {
        return;
    }
    let Some(phases) = world
        .get_resource::<bevy::render::render_phase::ViewBinnedRenderPhases<SsdmVector3d>>()
    else {
        return;
    };
    let Some(phase) = phases.get(&extracted_view.retained_view_entity) else {
        return;
    };
    if phase.is_empty() {
        return;
    }
    let pyramid = world.resource::<SsdmPyramidImages>();
    if pyramid.a.is_empty() {
        return;
    }
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let Some(gpu_a0) = gpu_images.get(&pyramid.a[0]) else {
        return;
    };
    let desc = RenderPassDescriptor {
        label: Some("ssdm_vector_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &gpu_a0.texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::BLACK.into()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    let mut pass = ctx.begin_tracked_render_pass(desc);

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }

    let _ = phase.render(&mut pass, world, view_entity);
}

pub fn ssdm_pyramid_pass(
    world: &World,
    view: ViewQuery<(Option<&SsdmSettings>,)>,
    mut ctx: RenderContext,
) {
    let (ssdm_settings,) = view.into_inner();
    let Some(settings) = ssdm_settings else {
        return;
    };
    if settings.enabled == 0 {
        return;
    }
    run_ssdm_pyramid_passes(world, &mut ctx);
}

pub fn ssdm_gbuffer_warp(
    world: &World,
    view: ViewQuery<(
        &ViewPrepassTextures,
        &ViewDepthTexture,
        Option<&SsdmSettings>,
        &DeferredLightingIdDepthTexture,
    )>,
    mut ctx: RenderContext,
) {
    let (prepass, view_depth, ssdm_settings, deferred_lighting_id_depth) = view.into_inner();
    let render_context = &mut ctx;
    let Some(settings) = ssdm_settings else {
        return;
    };
    if settings.enabled == 0 {
        return;
    }
    let pipeline_cache = world.resource::<PipelineCache>();
    let post = world.resource::<SsdmPostPipeline>();
    let scratch_res = world.resource::<SsdmGBufferScratch>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();

    let Some(mrt_pl) = pipeline_cache.get_render_pipeline(post.gbuffer_warp_mrt_pipeline)
    else {
        return;
    };
    let Some(depth_warp_pl) =
        pipeline_cache.get_render_pipeline(post.prepass_depth_warp_pipeline)
    else {
        return;
    };
    let Some(blit_rgba) = pipeline_cache.get_render_pipeline(post.blit_rgba32_pipeline) else {
        return;
    };
    let Some(blit_r8) = pipeline_cache.get_render_pipeline(post.blit_r8_pipeline) else {
        return;
    };
    let Some(warp_n) = pipeline_cache.get_render_pipeline(post.prepass_warp_normal_pipeline)
    else {
        return;
    };
    let Some(warp_m) = pipeline_cache.get_render_pipeline(post.prepass_warp_motion_pipeline)
    else {
        return;
    };
    let Some(blit_n_pl) = pipeline_cache.get_render_pipeline(post.prepass_blit_normal_pipeline)
    else {
        return;
    };
    let Some(blit_m_pl) = pipeline_cache.get_render_pipeline(post.prepass_blit_motion_pipeline)
    else {
        return;
    };

    let Some(def_att) = &prepass.deferred else {
        return;
    };
    let Some(lit_att) = &prepass.deferred_lighting_pass_id else {
        return;
    };

    let Some(sd) = &scratch_res.deferred else {
        return;
    };
    let Some(sl) = &scratch_res.lighting_pass_id else {
        return;
    };
    let Some(sdepth) = &scratch_res.depth else {
        return;
    };
    let gpu_sd = gpu_images.get(sd).expect("ssdm scratch deferred gpu");
    let gpu_sl = gpu_images.get(sl).expect("ssdm scratch lighting gpu");
    let gpu_sdepth = gpu_images.get(sdepth).expect("ssdm scratch depth gpu");

    let pyramid = world.resource::<SsdmPyramidImages>();
    let Some(b0_h) = pyramid.b.first() else {
        return;
    };
    // Pyramid A level 0 is the unfiltered per-pixel forward displacement vector field
    // written by the vector pass. The resolve helper reads it for the inline Newton
    // polish step (`uv - V(B0(uv))`) so the silhouette tracks the per-pixel heightmap
    // detail rather than the coarse-level averages that the refine pass walks down.
    let Some(a0_h) = pyramid.a.first() else {
        return;
    };
    let b0 = gpu_images.get(b0_h).expect("pyramid B0 gpu");
    let a0 = gpu_images.get(a0_h).expect("pyramid A0 gpu");

    let def_src = &def_att.texture.default_view;
    let lit_src = &lit_att.texture.default_view;

    let mrt_bg = render_context.render_device().create_bind_group(
        Some("ssdm_gbuffer_warp_mrt"),
        &pipeline_cache.get_bind_group_layout(&post.gbuffer_warp_mrt_layout),
        &[
            BindGroupEntry {
                binding: 0,
                resource: bevy::render::render_resource::BindingResource::Sampler(
                    &post.sampler,
                ),
            },
            BindGroupEntry {
                binding: 1,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    &b0.texture_view,
                ),
            },
            BindGroupEntry {
                binding: 2,
                resource: bevy::render::render_resource::BindingResource::TextureView(def_src),
            },
            BindGroupEntry {
                binding: 3,
                resource: bevy::render::render_resource::BindingResource::TextureView(lit_src),
            },
            BindGroupEntry {
                binding: 4,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    &a0.texture_view,
                ),
            },
        ],
    );

    let mut mrt_pass =
        render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("ssdm_gbuffer_warp_mrt"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &gpu_sd.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &gpu_sl.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
    mrt_pass.set_pipeline(mrt_pl);
    mrt_pass.set_bind_group(0, &mrt_bg, &[]);
    mrt_pass.draw(0..3, 0..1);
    drop(mrt_pass);

    if let Some(depth_att) = &prepass.depth {
        let depth_src = &depth_att.texture.default_view;
        let d_bg = render_context.render_device().create_bind_group(
            Some("ssdm_prepass_depth_warp"),
            &pipeline_cache.get_bind_group_layout(&post.prepass_depth_warp_layout),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &b0.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler_nearest,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        depth_src,
                    ),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &a0.texture_view,
                    ),
                },
            ],
        );
        let mut dpass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssdm_prepass_depth_warp"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &gpu_sdepth.texture_view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
        dpass.set_pipeline(depth_warp_pl);
        dpass.set_bind_group(0, &d_bg, &[]);
        dpass.draw(0..3, 0..1);
        drop(dpass);

        render_context.command_encoder().copy_texture_to_texture(
            gpu_sdepth.texture.as_image_copy(),
            depth_att.texture.texture.as_image_copy(),
            prepass.size,
        );

        // Mirror the warped depth into `view_depth_texture` as well. The deferred prepass
        // node renders into `view_depth_texture` and then copies it into `prepass.depth`,
        // so the two are siblings post-prepass. Post-prepass consumers split: deferred
        // lighting / `prepass_utils::prepass_depth` / SSAO read `prepass.depth` (binding
        // 20 in `mesh_view_bindings`), while `MainOpaquePass3dNode`'s depth-stencil, the
        // legacy `Skybox` draw, and the atmosphere `RenderSkyNode` (binding 13 of its own
        // bind group, set to `view_depth_texture.view()`) read `view_depth_texture`. If we
        // only updated the prepass copy, sky / forward / `MainOpaquePass` would still see
        // un-warped depth and treat the SSDM-extended silhouette as background, producing
        // a "ghost shell". `view_depth_texture` only declares `RENDER_ATTACHMENT (+ COPY_SRC`
        // when `DepthPrepass` is set); `configure_ssdm_depth_texture_usages` in `plugin.rs`
        // ORs in `COPY_DST` for SSDM cameras so this copy is valid.
        render_context.command_encoder().copy_texture_to_texture(
            gpu_sdepth.texture.as_image_copy(),
            view_depth.texture.as_image_copy(),
            prepass.size,
        );
    }

    let blit_def_bg = render_context.render_device().create_bind_group(
        Some("ssdm_blit_def"),
        &pipeline_cache.get_bind_group_layout(&post.blit_rgba32_layout),
        &[BindGroupEntry {
            binding: 0,
            resource: bevy::render::render_resource::BindingResource::TextureView(
                &gpu_sd.texture_view,
            ),
        }],
    );
    let mut bp = render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("ssdm_blit_deferred"),
            color_attachments: &[Some(def_att.get_unsampled_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    bp.set_pipeline(blit_rgba);
    bp.set_bind_group(0, &blit_def_bg, &[]);
    bp.draw(0..3, 0..1);
    drop(bp);

    let blit_lit_bg = render_context.render_device().create_bind_group(
        Some("ssdm_blit_lit"),
        &pipeline_cache.get_bind_group_layout(&post.blit_r8_layout),
        &[BindGroupEntry {
            binding: 0,
            resource: bevy::render::render_resource::BindingResource::TextureView(
                &gpu_sl.texture_view,
            ),
        }],
    );
    let mut lp = render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("ssdm_blit_lighting_id"),
            color_attachments: &[Some(lit_att.get_unsampled_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    lp.set_pipeline(blit_r8);
    lp.set_bind_group(0, &blit_lit_bg, &[]);
    lp.draw(0..3, 0..1);
    drop(lp);

    if let (Some(norm_att), Some(sn)) = (&prepass.normal, &scratch_res.normal) {
        let gpu_sn = gpu_images.get(sn).unwrap();
        let n_bg = render_context.render_device().create_bind_group(
            Some("ssdm_warp_normal"),
            &pipeline_cache.get_bind_group_layout(&post.prepass_warp_float_layout),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &b0.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &norm_att.texture.default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &a0.texture_view,
                    ),
                },
            ],
        );
        let mut np =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssdm_warp_normal_scratch"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &gpu_sn.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
        np.set_pipeline(warp_n);
        np.set_bind_group(0, &n_bg, &[]);
        np.draw(0..3, 0..1);
        drop(np);

        let blit_n_bg = render_context.render_device().create_bind_group(
            Some("ssdm_blit_normal_to_prepass"),
            &pipeline_cache.get_bind_group_layout(&post.prepass_blit_float_layout),
            &[BindGroupEntry {
                binding: 0,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    &gpu_sn.texture_view,
                ),
            }],
        );
        let mut np2 =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssdm_blit_normal_to_prepass"),
                    color_attachments: &[Some(norm_att.get_unsampled_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
        np2.set_pipeline(blit_n_pl);
        np2.set_bind_group(0, &blit_n_bg, &[]);
        np2.draw(0..3, 0..1);
    }

    if let (Some(mot_att), Some(sm)) = (&prepass.motion_vectors, &scratch_res.motion_vectors) {
        let gpu_sm = gpu_images.get(sm).unwrap();
        let m_bg = render_context.render_device().create_bind_group(
            Some("ssdm_warp_motion"),
            &pipeline_cache.get_bind_group_layout(&post.prepass_warp_float_layout),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: bevy::render::render_resource::BindingResource::Sampler(
                        &post.sampler,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &b0.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &mot_att.texture.default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &a0.texture_view,
                    ),
                },
            ],
        );
        let mut mp =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssdm_warp_motion_scratch"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &gpu_sm.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
        mp.set_pipeline(warp_m);
        mp.set_bind_group(0, &m_bg, &[]);
        mp.draw(0..3, 0..1);
        drop(mp);

        let blit_m_bg = render_context.render_device().create_bind_group(
            Some("ssdm_blit_motion_to_prepass"),
            &pipeline_cache.get_bind_group_layout(&post.prepass_blit_float_layout),
            &[BindGroupEntry {
                binding: 0,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    &gpu_sm.texture_view,
                ),
            }],
        );
        let mut mp2 =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssdm_blit_motion_to_prepass"),
                    color_attachments: &[Some(mot_att.get_unsampled_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
        mp2.set_pipeline(blit_m_pl);
        mp2.set_bind_group(0, &blit_m_bg, &[]);
        mp2.draw(0..3, 0..1);
    }

    // Bevy's built-in `copy_deferred_lighting_id` may run before this warp in the
    // prepass schedule. Re-copy after the warped lighting-pass id is written so deferred
    // lighting's depth gate matches the SSDM silhouette.
    if let Some(copy_pl) = pipeline_cache.get_render_pipeline(post.copy_lighting_id_pipeline) {
        let Some(lit_att) = &prepass.deferred_lighting_pass_id else {
            return;
        };
        let copy_bg = render_context.render_device().create_bind_group(
            Some("ssdm_copy_deferred_lighting_id"),
            &pipeline_cache.get_bind_group_layout(&post.copy_lighting_id_layout),
            &[BindGroupEntry {
                binding: 0,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    &lit_att.texture.default_view,
                ),
            }],
        );
        let mut copy_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ssdm_copy_deferred_lighting_id"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &deferred_lighting_id_depth.texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        copy_pass.set_render_pipeline(copy_pl);
        copy_pass.set_bind_group(0, &copy_bg, &[]);
        copy_pass.draw(0..3, 0..1);
    }
}

pub fn ssdm_post_process(
    world: &World,
    view: ViewQuery<(&ViewTarget, Option<&SsdmSettings>)>,
    mut ctx: RenderContext,
) {
    let (view_target, _ssdm_settings) = view.into_inner();
    let render_context = &mut ctx;
    let pipeline_cache = world.resource::<PipelineCache>();
    let post = world.resource::<SsdmPostPipeline>();

    let Some(gather_pl) = pipeline_cache.get_render_pipeline(post.gather_pipeline) else {
        return;
    };

    let pyramid = world.resource::<SsdmPyramidImages>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let b0_view = pyramid
        .b
        .first()
        .and_then(|h| gpu_images.get(h))
        .map(|gi| &gi.texture_view);
    let a0_view = pyramid
        .a
        .first()
        .and_then(|h| gpu_images.get(h))
        .map(|gi| &gi.texture_view);

    let gather_enabled = 0u32;
    let pp = view_target.post_process_write();

    let mut ubuf = UniformBuffer::new(Vec::new());
    ubuf.write(&GatherUni {
        enabled: gather_enabled,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    })
    .unwrap();
    let buf = render_context
        .render_device()
        .create_buffer_with_data(&BufferInitDescriptor {
            label: Some("ssdm_gather"),
            contents: ubuf.as_ref(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
    let bg = render_context.render_device().create_bind_group(
        Some("ssdm_gather_bg"),
        &pipeline_cache.get_bind_group_layout(&post.gather_layout),
        &[
            BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: bevy::render::render_resource::BindingResource::Sampler(&post.sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: bevy::render::render_resource::BindingResource::TextureView(pp.source),
            },
            BindGroupEntry {
                binding: 3,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    b0_view.unwrap_or(pp.source),
                ),
            },
            BindGroupEntry {
                binding: 4,
                resource: bevy::render::render_resource::BindingResource::TextureView(
                    a0_view.unwrap_or(pp.source),
                ),
            },
        ],
    );
    let mut pass = render_context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("ssdm_gather"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: pp.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(gather_pl);
    pass.set_bind_group(0, &bg, &[]);
    pass.draw(0..3, 0..1);
}

/// Builds fullscreen SSDM pipelines (runs once at render startup).
pub fn init_ssdm_post_pipeline(
    mut commands: Commands,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    fullscreen: Res<bevy::core_pipeline::FullscreenShader>,
    pipeline_cache: ResMut<bevy::render::render_resource::PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    use bevy::render::render_resource::{
        binding_types::{sampler, texture_2d, texture_depth_2d, uniform_buffer},
        ColorTargetState, ColorWrites, FilterMode, FragmentState, MultisampleState, PrimitiveState,
        RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages,
        TextureSampleType,
    };

    let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("ssdm_linear"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });

    let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("ssdm_nearest"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: bevy::render::render_resource::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let down_layout = BindGroupLayoutDescriptor::new(
        "ssdm_down_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<DownUni>(false),
                sampler(SamplerBindingType::NonFiltering),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );
    let refine_layout = BindGroupLayoutDescriptor::new(
        "ssdm_refine_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<RefineUni>(false),
                sampler(SamplerBindingType::NonFiltering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );
    // Each warp / gather layout below adds one trailing `texture_2d<f32>` binding for
    // Pyramid A level 0 (`pyramid.a[0]`). The resolve helper needs A0 to do its inline
    // Newton polish step (`uv - V(B0(uv))`) on top of the discontinuity-aware B0 sample.
    // See `assets/shaders/ssdm_resolve.wgsl::resolve_src_uv`.
    let gather_layout = BindGroupLayoutDescriptor::new(
        "ssdm_gather_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<GatherUni>(false),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    let gbuffer_warp_mrt_layout = BindGroupLayoutDescriptor::new(
        "ssdm_gbuffer_warp_mrt_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    let prepass_depth_warp_layout = BindGroupLayoutDescriptor::new(
        "ssdm_prepass_depth_warp_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::NonFiltering),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    let blit_rgba32_layout = BindGroupLayoutDescriptor::new(
        "ssdm_blit_rgba32_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Uint),),
        ),
    );

    let blit_r8_layout = BindGroupLayoutDescriptor::new(
        "ssdm_blit_r8_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Uint),),
        ),
    );

    let prepass_warp_float_layout = BindGroupLayoutDescriptor::new(
        "ssdm_prepass_warp_float_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    let prepass_blit_float_layout = BindGroupLayoutDescriptor::new(
        "ssdm_prepass_blit_float_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Float { filterable: true }),),
        ),
    );

    // `ssdm_resolve.wgsl` declares `#define_import_path bevy_ssdm::resolve` and is imported
    // (but never directly used as a pipeline) by every warp/gather shader for the
    // discontinuity-aware `sample_b0_robust` helper. Loading it via `asset_server.load` is
    // enough to register the module with naga_oil's import resolver - we just need to keep
    // the handle alive on `SsdmPostPipeline` so the asset (and therefore the module
    // registration) doesn't get garbage-collected mid-pipeline-rebuild.
    let resolve_lib = asset_server.load("shaders/ssdm_resolve.wgsl");
    let down_shader = asset_server.load("shaders/ssdm_downsample.wgsl");
    let refine_shader = asset_server.load("shaders/ssdm_refine.wgsl");
    let gather_shader = asset_server.load("shaders/ssdm_gather.wgsl");
    let gbuffer_warp_shader = asset_server.load("shaders/ssdm_gbuffer_warp.wgsl");
    let prepass_depth_warp_shader = asset_server.load("shaders/ssdm_prepass_depth_warp.wgsl");
    let blit_rgba_shader = asset_server.load("shaders/ssdm_prepass_blit_rgba32uint.wgsl");
    let blit_r8_shader = asset_server.load("shaders/ssdm_prepass_blit_r8uint.wgsl");
    let warp_float_shader = asset_server.load("shaders/ssdm_prepass_warp_float.wgsl");
    let prepass_blit_float_shader = asset_server.load("shaders/ssdm_prepass_blit_float.wgsl");

    let down_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ssdm_down".into()),
        layout: vec![down_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: down_shader,
            entry_point: Some("downsample_fs".into()),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: crate::render_targets::SSDM_PYRAMID_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let refine_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ssdm_refine".into()),
        layout: vec![refine_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: refine_shader,
            entry_point: Some("refine_fs".into()),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: crate::render_targets::SSDM_PYRAMID_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let gather_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ssdm_gather".into()),
        layout: vec![gather_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: gather_shader,
            entry_point: Some("gather_fs".into()),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: GATHER_COLOR_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let gbuffer_warp_mrt_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_gbuffer_warp_mrt".into()),
            layout: vec![gbuffer_warp_mrt_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: gbuffer_warp_shader,
                entry_point: Some("warp_mrt_fs".into()),
                shader_defs: vec![],
                targets: vec![
                    Some(ColorTargetState {
                        format: DEFERRED_PREPASS_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let prepass_depth_warp_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_prepass_depth_warp".into()),
            layout: vec![prepass_depth_warp_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: prepass_depth_warp_shader,
                entry_point: Some("depth_warp_fs".into()),
                shader_defs: vec![],
                targets: vec![],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let blit_rgba32_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ssdm_blit_rgba32".into()),
        layout: vec![blit_rgba32_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: blit_rgba_shader,
            entry_point: Some("blit_fs".into()),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: DEFERRED_PREPASS_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let blit_r8_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("ssdm_blit_r8".into()),
        layout: vec![blit_r8_layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: blit_r8_shader,
            entry_point: Some("blit_fs".into()),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let prepass_warp_normal_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_prepass_warp_normal".into()),
            layout: vec![prepass_warp_float_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: warp_float_shader.clone(),
                entry_point: Some("warp_float_fs".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: NORMAL_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let prepass_warp_motion_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_prepass_warp_motion".into()),
            layout: vec![prepass_warp_float_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: warp_float_shader,
                entry_point: Some("warp_float_fs".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: MOTION_VECTOR_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let prepass_blit_normal_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_prepass_blit_normal".into()),
            layout: vec![prepass_blit_float_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: prepass_blit_float_shader.clone(),
                entry_point: Some("blit_fs".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: NORMAL_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let prepass_blit_motion_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_prepass_blit_motion".into()),
            layout: vec![prepass_blit_float_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: prepass_blit_float_shader,
                entry_point: Some("blit_fs".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: MOTION_VECTOR_PREPASS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    let copy_lighting_id_layout = BindGroupLayoutDescriptor::new(
        "ssdm_copy_lighting_id_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Uint),),
        ),
    );
    let copy_lighting_id_shader =
        asset_server.load("shaders/ssdm_copy_deferred_lighting_id.wgsl");
    let copy_lighting_id_pipeline =
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssdm_copy_lighting_id".into()),
            layout: vec![copy_lighting_id_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: copy_lighting_id_shader,
                entry_point: Some("fragment".into()),
                shader_defs: vec![],
                targets: vec![],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            ..Default::default()
        });

    commands.insert_resource(SsdmPostPipeline {
        sampler: linear_sampler,
        sampler_nearest: nearest_sampler,
        _resolve_lib: resolve_lib,
        down_layout,
        refine_layout,
        gather_layout,
        gbuffer_warp_mrt_layout,
        prepass_depth_warp_layout,
        blit_rgba32_layout,
        blit_r8_layout,
        prepass_warp_float_layout,
        prepass_blit_float_layout,
        down_pipeline,
        refine_pipeline,
        gather_pipeline,
        gbuffer_warp_mrt_pipeline,
        prepass_depth_warp_pipeline,
        blit_rgba32_pipeline,
        blit_r8_pipeline,
        prepass_warp_normal_pipeline,
        prepass_warp_motion_pipeline,
        prepass_blit_normal_pipeline,
        prepass_blit_motion_pipeline,
        copy_lighting_id_layout,
        copy_lighting_id_pipeline,
    });
}
