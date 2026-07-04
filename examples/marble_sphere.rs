//! Marble-like PBR sphere with [`bevy_ssdm::SsdmPlugin`] screen-space displacement (deferred path).
//!
//! PBR maps and displacement are loaded from [`../assets/marble_cliff_03_1k/textures/`], matching
//! the glTF image URIs (`diff` / `arm` JPG, `nor_gl` PNG, `disp` JPG).
use bevy::anti_alias::smaa::Smaa;
use bevy::audio::AudioPlugin;
use bevy::camera::Exposure;
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{
    atmosphere::ScatteringMedium, light_consts::lux, Atmosphere, AtmosphereEnvironmentMapLight,
};
use bevy::mesh::primitives::SphereKind;
use bevy::pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};
use bevy_ssdm::{
    SsdmPlugin, SsdmSettings, SsdmVectorMaterial, SsdmVectorParams, SsdmVectorSurface, SsdmView,
};

mod marble_assets {
    pub const DIFFUSE: &str = "marble_cliff_03_1k/textures/marble_cliff_03_diff_1k.jpg";
    pub const NORMAL: &str = "marble_cliff_03_1k/textures/marble_cliff_03_nor_gl_1k.png";
    pub const ARM: &str = "marble_cliff_03_1k/textures/marble_cliff_03_arm_1k.jpg";
    pub const DISPLACEMENT: &str = "marble_cliff_03_1k/textures/marble_cliff_03_disp_1k.jpg";
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").into(),
                    ..default()
                })
                .build()
                .disable::<AudioPlugin>(),
        )
        // Opaque meshes use the deferred G-buffer path (global resource).
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins(SsdmPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (spin_camera, auto_screenshot))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_mats: ResMut<Assets<StandardMaterial>>,
    mut ssdm_mats: ResMut<Assets<SsdmVectorMaterial>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    asset_server: Res<AssetServer>,
) {
    let base_color = asset_server.load(marble_assets::DIFFUSE);
    let normal_map = asset_server.load(marble_assets::NORMAL);
    let arm = asset_server.load(marble_assets::ARM);
    let displacement = asset_server.load(marble_assets::DISPLACEMENT);

    let mut mesh = Sphere::new(1.25)
        .mesh()
        .kind(SphereKind::Uv {
            sectors: 64,
            stacks: 32,
        })
        .build();
    mesh.generate_tangents()
        .expect("UV sphere should have UVs for MikkTSpace tangents");
    let sphere_mesh = meshes.add(mesh);

    commands.spawn((
        Mesh3d(sphere_mesh.clone()),
        MeshMaterial3d(std_mats.add(StandardMaterial {
            base_color_texture: Some(base_color),
            normal_map_texture: Some(normal_map.clone()),
            metallic_roughness_texture: Some(arm),
            metallic: 0.0,
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
    ));

    commands.spawn((
        Mesh3d(sphere_mesh),
        MeshMaterial3d(ssdm_mats.add(SsdmVectorMaterial {
            params: SsdmVectorParams {
                displacement_scale: 0.35,
                ..default()
            },
            displacement,
            normal_map,
        })),
        Transform::default(),
        SsdmVectorSurface,
    ));

    commands.spawn((
        DirectionalLight {
            // Pre-atmosphere sun illuminance; the atmosphere shader filters it (see Bevy atmosphere example).
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.6, 0.0)),
    ));

    let medium = scattering_mediums.add(ScatteringMedium::default());
    commands.spawn(Atmosphere::earth(medium));

    commands.spawn((
        Camera3d::default(),
        // Bevy's deferred path is single-sample (the deferred render targets are not
        // MSAA), so hardware MSAA is not an option on this view. Post-process AA is the
        // only way to soften silhouettes here.
        Msaa::Off,
        Projection::default(),
        Transform::from_xyz(0.0, 0.0, 5.2).looking_at(Vec3::ZERO, Vec3::Y),
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AgX,
        DepthPrepass,
        DeferredPrepass,
        Bloom::default(),
        SsdmSettings::default(),
        SsdmView,
        // SMAA (Subpixel Morphological Anti-Aliasing) is the right post-process AA for
        // an SSDM view: it does sub-pixel reconstruction at luminance edges, which
        // smooths *both* the static silhouette aliasing of the deferred view *and* the
        // pixel-aligned stair-stepping that the SSDM resolve produces along the warped
        // silhouette (the deferred packed gbuffer is `Rgba32Uint` and not filterable, so
        // adjacent destination pixels along the warped rim necessarily read different
        // source texels - SMAA reconstructs a smooth boundary on top of that). Unlike
        // TAA, SMAA has no temporal accumulation so the silhouette stays sharp and
        // ghost-free as the camera spins around the sphere.
        Smaa::default(),
    ));
}

/// When `SSDM_SCREENSHOT=1`, saves a screenshot after a few frames (for CI / headless capture).
fn auto_screenshot(mut commands: Commands, time: Res<Time>, mut scheduled: Local<bool>) {
    if std::env::var("SSDM_SCREENSHOT").ok().as_deref() != Some("1") {
        return;
    }
    if *scheduled || time.elapsed_secs() < 10.0 {
        return;
    }
    *scheduled = true;
    let path = std::env::var("SSDM_SCREENSHOT_PATH")
        .unwrap_or_else(|_| "/opt/cursor/artifacts/screenshots/marble_sphere.png".to_string());
    commands.spawn(Screenshot::primary_window()).observe(
        move |screenshot: On<ScreenshotCaptured>, mut commands: Commands| {
            save_to_disk(&path)(screenshot);
            commands.write_message(AppExit::Success);
        },
    );
}

fn spin_camera(time: Res<Time>, mut q: Query<&mut Transform, With<SsdmView>>) {
    let Ok(mut tf) = q.single_mut() else {
        return;
    };
    let r = 5.2;
    let a = time.elapsed_secs() * 0.4;
    *tf = Transform::from_xyz(a.sin() * r, 0.15, a.cos() * r).looking_at(Vec3::ZERO, Vec3::Y);
}
