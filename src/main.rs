use std::f64::consts::PI;

// use mint::Vector2;
// use msdfgen::{FontExt, Bitmap, Gray, Range, MsdfGeneratorConfig, FillRule, MID_VALUE, Rgba};
// use owned_ttf_parser::{OwnedFace, Face, AsFaceRef};
use bevy::{prelude::*, asset::{AssetLoader, LoadContext, LoadedAsset}, utils::BoxedFuture, reflect::TypeUuid, render::render_resource::{Extent3d, TextureFormat, TextureDimension}, math::DVec2};

mod math;
mod edge_segments;
mod eqution_solver;
mod contour;
mod shape;
mod scanline;
mod projection;
mod bitmap;
mod rasterization;

#[derive(Debug, TypeUuid)]
#[uuid = "2e8e24a2-ba67-4d6c-96e1-2cfd18b53a62"]
struct Font {
    // face: OwnedFace
}


#[derive(Default)]
struct FontAssetLoader;

impl AssetLoader for FontAssetLoader {
    fn extensions(&self) -> &[&str] {
        &["ttf"]
    }

    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), bevy::asset::Error>> {
        Box::pin(async move {
            // info!("loading font");
            // let font = Font {
            //     face: OwnedFace::from_vec(bytes.into(), 0).unwrap()
            // };
            // load_context.set_default_asset(LoadedAsset::new(font));
            Ok(())
        })
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .init_asset_loader::<FontAssetLoader>()
        .add_asset::<Font>()
        .add_system(play_with_font)
        .run();
}

fn setup(mut commands: Commands, assets_server: Res<AssetServer>) {
    info!("setup!");
    commands.spawn(Camera2dBundle::default());
    let font: Handle<Font> = assets_server.load("fonts/FiraMono-Medium.ttf");
    commands.spawn(font);
}

fn play_with_font(
    mut events: EventReader<AssetEvent<Font>>,
    assets: Res<Assets<Font>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    time: Res<Time>,
) {
    // for e in events.iter() {
    //     info!("{e:?}");
    //     let AssetEvent::Created { handle } = e else {
    //         continue
    //     };
    //     let Some(font) = assets.get(handle) else {
    //         continue
    //     };

    //     let face = font.face.as_face_ref();
    //     let glyph = face.glyph_index('g').unwrap();
    //     let h = face.ascender() + face.descender();
    //     let units =  face.units_per_em();
    //     let line_height = face.height();
    //     let bbox = face.glyph_bounding_box(glyph).unwrap();
    //     let advance = face.glyph_hor_advance(glyph).unwrap();
    //     let mut shape = face.glyph_shape(glyph).unwrap();
    //     let start = time.elapsed().as_nanos();
    //     let width = 32;
    //     let height = 32;

    //     let bound = shape.get_bound();
    //     info!("bound: {bound:?}, height: {h:?}, units: {units:?}, box: {bbox:?}, advance: {advance:?}");
    //     let framing = bound.autoframe(width, height, Range::Px(4.0), None).unwrap();
    //     info!("framing: {framing:?}");
    //     // let fill_rule = FillRule::default();

    //     let mut bitmap = Bitmap::new(width, height);

    //     shape.edge_coloring_simple(3.0, 0);

    //     let config = MsdfGeneratorConfig::default();

    //     shape.generate_mtsdf(&mut bitmap, &framing, &config);

    //     // bitmap.convert::<Rgba<f32>>();
        
    //     let img = Image::new(
    //         Extent3d {
    //             width,
    //             height,
    //             depth_or_array_layers: 1,
    //         },
    //         TextureDimension::D2,
    //         bitmap.raw_pixels().into(),
    //         TextureFormat::Rgba32Float,
    //     );
    //     let handle = images.add(img);
    //     commands.spawn(SpriteBundle {
    //         texture: handle,
    //         sprite: Sprite {
    //             custom_size: Some(Vec2::new(300., 300.)),
    //             ..default()
    //         },
    //         ..Default::default()
    //     });




    //     let end = time.elapsed().as_nanos();
    //     info!("done in {}ns", end - start);



    // }
}