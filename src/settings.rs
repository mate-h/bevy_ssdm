use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, UniformComponentPlugin},
        render_resource::ShaderType,
    },
};

/// Per-camera SSDM controls (attach to the same entity as [`Camera3d`](bevy_camera::Camera3d)).
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct SsdmSettings {
    /// Master switch; when 0, gather is a no-op passthrough.
    pub enabled: u32,
    /// Number of mip levels in pyramids A and B (clamped to `2..=8` at runtime).
    ///
    /// This is the single biggest knob on silhouette extension quality. The refine pass
    /// solves the inverse fixed-point `q = uv - V(q)` coarse-to-fine; at the coarsest
    /// level the iteration's seed is `uv` itself and its 4 corner taps reach
    /// `uv ± half_texel`, where each tap then samples a Pyramid A texel that already
    /// averages over `2^(L-1) x 2^(L-1)` source pixels. So the **effective basin of
    /// attraction is roughly `±2^(pyramid_levels - 1)` source pixels**: any output pixel
    /// further than that from the unwarped silhouette has zero V in every coarse-level
    /// tap, the iteration converges to "no displacement", and the warped silhouette gets
    /// silently clipped at that radius regardless of the heightmap.
    ///
    /// At 1080p this gives roughly:
    /// - `pyramid_levels = 4` -> 8 pixel basin (default in earlier revisions; clips
    ///   silhouettes for any `displacement_scale` that produces visible parallax)
    /// - `pyramid_levels = 6` -> 32 pixel basin (good for shallow displacement)
    /// - `pyramid_levels = 8` -> 128 pixel basin (good for moderate-to-aggressive
    ///   displacement; the cost over 4 levels is negligible because the pyramid's
    ///   memory and per-frame fragment count are dominated by level 0)
    ///
    /// If you cap this at the lower end and the heightmap projects beyond the basin you
    /// will see exactly the symptom "the silhouette is no longer perfectly spherical
    /// but does not faithfully follow the actual displacement" - the iteration converges
    /// where it can but stops at the basin radius.
    pub pyramid_levels: u32,
    pub _pad: Vec2,
}

impl Default for SsdmSettings {
    fn default() -> Self {
        Self {
            enabled: 1,
            // 8 = the upper clamp; gives a +/-128px basin at 1080p which comfortably
            // exceeds the rim displacement of any reasonable PBR heightmap with
            // `displacement_scale <= 0.5`. The extra 4 levels add ~0.4% to pyramid
            // memory and a handful of microseconds to per-frame GPU time.
            pyramid_levels: 8,
            _pad: Vec2::ZERO,
        }
    }
}

pub struct SsdmSettingsPlugin;

impl Plugin for SsdmSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            bevy::render::extract_component::ExtractComponentPlugin::<SsdmSettings>::default(),
            UniformComponentPlugin::<SsdmSettings>::default(),
        ));
    }
}

/// Marker: only entities with [`Camera3d`](bevy_camera::Camera3d) + this component run the SSDM chain.
#[derive(Component, Default)]
pub struct SsdmView;
