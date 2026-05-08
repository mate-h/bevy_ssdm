//! Screen-space displacement mapping (SSDM) for Bevy 0.18.
//!
//! **Deferred rendering only** — adding [`SsdmPlugin`] sets [`DefaultOpaqueRendererMethod`](bevy::pbr::DefaultOpaqueRendererMethod) to
//! deferred. Use a [`DeferredPrepass`](bevy::core_pipeline::prepass::DeferredPrepass) (and typically
//! [`DepthPrepass`](bevy::core_pipeline::prepass::DepthPrepass)) on the same camera as [`SsdmView`].
//!
//! Before deferred lighting, the plugin warps the prepass **G-buffer** (packed deferred texture,
//! lighting-pass id, depth, and optional normal / motion-vector prepasses) using the SSDM UV field so
//! shading matches the displaced surface.
//!
//! See [`SsdmPlugin`] and the `marble_sphere` example.

mod marker;
mod node;
mod plugin;
mod queue;
mod render_targets;
mod settings;
mod vector_material;
mod vector_phase;

pub use marker::SsdmVectorSurface;
pub use plugin::SsdmPlugin;
pub use settings::{SsdmSettings, SsdmView};
pub use vector_material::{SsdmVectorMaterial, SsdmVectorParams};
