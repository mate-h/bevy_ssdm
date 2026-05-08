//! CPU-side marker: mesh + [`super::vector_material::SsdmVectorMaterial`] participates in the vector pass.

use bevy::prelude::*;

#[derive(Component, Clone, Copy, Default, Reflect)]
#[reflect(Component)]
pub struct SsdmVectorSurface;
