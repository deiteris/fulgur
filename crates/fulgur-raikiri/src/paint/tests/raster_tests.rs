use super::*;
use raikiri_traits::{
    DecodedImage, ImagePixelSource, IntrinsicBox, ReplacedResolver, ResolveDisposition,
    ResolvedIntrinsic, ResolverError, ResolverRequest,
};
use std::sync::Arc;

struct RedPixels(Arc<DecodedImage>);

impl ImagePixelSource for RedPixels {
    fn get_decoded(&self, _url: &url::Url) -> Option<Arc<DecodedImage>> {
        Some(Arc::clone(&self.0))
    }
}

impl ReplacedResolver for RedPixels {
    fn resolve(
        &self,
        _request: ResolverRequest<'_>,
    ) -> std::result::Result<ResolvedIntrinsic, ResolverError> {
        Ok(ResolvedIntrinsic {
            intrinsic: IntrinsicBox::new(4.0, 2.0),
            disposition: ResolveDisposition::Ok,
        })
    }
}

#[test]
fn legacy_pages_keep_raster_pixels_and_own_opacity_and_allow_no_source() {
    let pixels = RedPixels(Arc::new(DecodedImage {
        width: 4,
        height: 2,
        rgba: [255, 0, 0, 255].repeat(8),
    }));
    let resources = RenderResources::new()
        .replaced_resolver(&pixels)
        .image_pixel_source(&pixels);
    let document = raikiri_html::parse_html_with_resources(
        &b"<style>@page{size:100px 100px;margin:0}body{margin:0}div{width:20px;height:10px;background:blue}img{display:block;width:20px;height:20px;opacity:.5}</style><div></div><img src='https://images.test/image.png'>"[..],
        &resources,
    ).unwrap();
    let LayoutStatus::Completed(layout) = raikiri_html::layout(
        &document,
        PageDefaults::default(),
        LayoutConfig::default(),
        LayoutOptions::new().resources(&resources),
    )
    .unwrap() else {
        panic!("completed layout")
    };
    let page = layout.page(0).unwrap();
    for with_source in [false, true] {
        let mut pdf = krilla::Document::new();
        let mut pdf_page = pdf.start_page_with(PageSettings::from_wh(100.0, 100.0).unwrap());
        let mut surface = pdf_page.surface();
        let mut fonts = FontCache::default();
        let mut svg = svg::SvgCache::new(crate::RenderOptions::default());
        let mut raster = raster::RasterCache::for_tests(
            with_source
                .then(|| resources.image_pixel_source_ref())
                .flatten(),
        );
        paint_legacy(
            &mut surface,
            &page,
            &page.paint_order(),
            &page.text_runs(),
            &mut fonts,
            &mut svg,
            &mut raster,
            &mut crate::tagging::Tags::disabled(),
            None,
        )
        .unwrap();
        surface.finish();
        pdf_page.finish();
        let bytes = pdf.finish().unwrap();
        let parsed = lopdf::Document::load_mem(&bytes).unwrap();
        let images: Vec<_> = parsed
            .objects
            .values()
            .filter_map(|object| object.as_stream().ok())
            .filter(|stream| {
                stream
                    .dict
                    .get(b"Subtype")
                    .and_then(lopdf::Object::as_name)
                    .ok()
                    == Some(b"Image".as_slice())
            })
            .collect();
        assert_eq!(images.len(), usize::from(with_source));
        if with_source {
            assert_eq!(
                images[0].decompressed_content().unwrap(),
                [255, 0, 0].repeat(8)
            );
            assert!(
                parsed
                    .objects
                    .values()
                    .filter_map(|object| object.as_dict().ok())
                    .any(
                        |dict| dict.get(b"ca").and_then(lopdf::Object::as_float).ok() == Some(0.5)
                    )
            );
        }
        let page_id = *parsed.get_pages().values().next().unwrap();
        let content =
            lopdf::content::Content::decode(&parsed.get_page_content(page_id).unwrap()).unwrap();
        assert!(
            content
                .operations
                .iter()
                .any(|operation| operation.operator == "rg"
                    && operation.operands == vec![0.into(), 0.into(), 1.into()])
        );
        assert_eq!(
            content
                .operations
                .iter()
                .filter(|operation| operation.operator == "q")
                .count(),
            content
                .operations
                .iter()
                .filter(|operation| operation.operator == "Q")
                .count()
        );
    }
}

#[test]
fn ordered_marker_events_can_be_rendered_without_a_pixel_provider() {
    let pixels = RedPixels(Arc::new(DecodedImage {
        width: 4,
        height: 2,
        rgba: [255, 0, 0, 255].repeat(8),
    }));
    let resources = RenderResources::new().image_pixel_source(&pixels);
    let document = raikiri_html::parse_html_with_resources(
        &b"<style>@page{size:100px 100px;margin:0}body{margin:0}ul{margin:0;padding:0}li{margin-left:20px;opacity:.5;list-style-image:url(https://images.test/marker.png)}</style><ul><li></li></ul>"[..],
        &resources,
    ).unwrap();
    let LayoutStatus::Completed(layout) = raikiri_html::layout(
        &document,
        PageDefaults::default(),
        LayoutConfig::default(),
        LayoutOptions::new().resources(&resources),
    )
    .unwrap() else {
        panic!("completed layout")
    };
    let page = layout.page(0).unwrap();
    let runs = page.text_runs();
    let events = page.paint_order_for_text_runs(&runs);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, PaintEvent::MarkerImage(_)))
    );
    assert!(order::supported(&events, &runs, |_| None));
    for with_source in [false, true] {
        let mut pdf = krilla::Document::new();
        let mut pdf_page = pdf.start_page_with(PageSettings::from_wh(100.0, 100.0).unwrap());
        let mut surface = pdf_page.surface();
        let mut fonts = FontCache::default();
        let mut svg = svg::SvgCache::new(crate::RenderOptions::default());
        let mut raster = raster::RasterCache::for_tests(
            with_source
                .then(|| resources.image_pixel_source_ref())
                .flatten(),
        );
        paint_ordered(
            &mut surface,
            &page,
            &events,
            &runs,
            &mut fonts,
            &mut svg,
            &mut raster,
            &mut crate::tagging::Tags::disabled(),
            None,
        )
        .unwrap();
        surface.finish();
        pdf_page.finish();
        let parsed = lopdf::Document::load_mem(&pdf.finish().unwrap()).unwrap();
        let images = parsed
            .objects
            .values()
            .filter_map(|object| object.as_stream().ok())
            .filter(|stream| {
                stream
                    .dict
                    .get(b"Subtype")
                    .and_then(lopdf::Object::as_name)
                    .ok()
                    == Some(b"Image".as_slice())
            })
            .count();
        assert_eq!(images, usize::from(with_source));
        let page_id = *parsed.get_pages().values().next().unwrap();
        let content =
            lopdf::content::Content::decode(&parsed.get_page_content(page_id).unwrap()).unwrap();
        let saves = content
            .operations
            .iter()
            .filter(|op| op.operator == "q")
            .count();
        let restores = content
            .operations
            .iter()
            .filter(|op| op.operator == "Q")
            .count();
        assert_eq!(saves, restores);
    }
}

/// Reports a 4x2 image but hands out whatever pixels it was given.
struct SizedPixels(Option<Arc<DecodedImage>>);

impl ImagePixelSource for SizedPixels {
    fn get_decoded(&self, _url: &url::Url) -> Option<Arc<DecodedImage>> {
        self.0.clone()
    }

    fn intrinsic_size(&self, _url: &url::Url) -> Option<raikiri_traits::ImageIntrinsicSize> {
        Some(raikiri_traits::ImageIntrinsicSize {
            width: Some(4.0),
            height: Some(2.0),
            aspect_ratio: Some(2.0),
        })
    }
}

#[test]
fn background_url_skips_missing_sources_and_unusable_pixels() {
    let images = |pixels: Option<DecodedImage>, with_source: bool| {
        let source = SizedPixels(pixels.map(Arc::new));
        let resources = RenderResources::new().image_pixel_source(&source);
        let document = raikiri_html::parse_html_with_resources(
            &b"<style>@page{size:100px 100px;margin:0}body{margin:0}div{width:4px;height:2px;background-image:url(https://images.test/bg.png)}</style><div></div>"[..],
            &resources,
        )
        .unwrap();
        let LayoutStatus::Completed(layout) = raikiri_html::layout(
            &document,
            PageDefaults::default(),
            LayoutConfig::default(),
            LayoutOptions::new().resources(&resources),
        )
        .unwrap() else {
            panic!("completed layout")
        };
        let page = layout.page(0).unwrap();
        let mut pdf = krilla::Document::new();
        let mut pdf_page = pdf.start_page_with(PageSettings::from_wh(100.0, 100.0).unwrap());
        let mut surface = pdf_page.surface();
        let mut fonts = FontCache::default();
        let mut svg = svg::SvgCache::new(crate::RenderOptions::default());
        let mut raster = raster::RasterCache::for_tests(
            with_source
                .then(|| resources.image_pixel_source_ref())
                .flatten(),
        );
        paint_legacy(
            &mut surface,
            &page,
            &page.paint_order(),
            &page.text_runs(),
            &mut fonts,
            &mut svg,
            &mut raster,
            &mut crate::tagging::Tags::disabled(),
            None,
        )
        .unwrap();
        surface.finish();
        pdf_page.finish();
        let parsed = lopdf::Document::load_mem(&pdf.finish().unwrap()).unwrap();
        parsed
            .objects
            .values()
            .filter_map(|object| object.as_stream().ok())
            .filter(|stream| {
                stream
                    .dict
                    .get(b"Subtype")
                    .and_then(lopdf::Object::as_name)
                    .ok()
                    == Some(b"Image".as_slice())
            })
            .count()
    };
    let red = DecodedImage {
        width: 4,
        height: 2,
        rgba: [255, 0, 0, 255].repeat(8),
    };
    assert_eq!(images(Some(red.clone()), true), 1);
    assert_eq!(images(Some(red), false), 0);
    assert_eq!(images(None, true), 0);
    let truncated = DecodedImage {
        width: 4,
        height: 2,
        rgba: vec![0; 4],
    };
    assert_eq!(images(Some(truncated), true), 0);
}

/// Image XObjects in `bytes` and the `Do` operations of its first page.
fn images_and_draws(bytes: &[u8]) -> (Vec<Vec<u8>>, Vec<lopdf::content::Operation>) {
    let parsed = lopdf::Document::load_mem(bytes).unwrap();
    let images = parsed
        .objects
        .values()
        .filter_map(|object| object.as_stream().ok())
        .filter(|stream| {
            stream
                .dict
                .get(b"Subtype")
                .and_then(lopdf::Object::as_name)
                .ok()
                == Some(b"Image".as_slice())
        })
        .map(|stream| stream.decompressed_content().unwrap())
        .collect();
    let page_id = *parsed.get_pages().values().next().unwrap();
    let content =
        lopdf::content::Content::decode(&parsed.get_page_content(page_id).unwrap()).unwrap();
    (images, content.operations)
}

/// A page whose `<p>` has an 8x4 `::before` image, laid out with `pixels`.
fn generated_image_layout(pixels: &RedPixels, css: &str) -> raikiri_html::DocumentLayout {
    let resources = RenderResources::new()
        .replaced_resolver(pixels)
        .image_pixel_source(pixels);
    let html = format!(
        "<style>@page{{size:100px 100px;margin:0}}body{{margin:0}}p{{margin:0;font:20px/20px sans-serif}}p::before{{content:url(https://images.test/before.png)}}{css}</style><p>x</p>"
    );
    let document = raikiri_html::parse_html_with_resources(html.as_bytes(), &resources).unwrap();
    let LayoutStatus::Completed(layout) = raikiri_html::layout(
        &document,
        PageDefaults::default(),
        LayoutConfig::default(),
        LayoutOptions::new().resources(&resources),
    )
    .unwrap() else {
        panic!("completed layout")
    };
    layout
}

fn red_8x4() -> RedPixels {
    RedPixels(Arc::new(DecodedImage {
        width: 8,
        height: 4,
        rgba: [255, 0, 0, 255].repeat(32),
    }))
}

#[test]
fn ordered_generated_image_events_draw_the_content_image() {
    let pixels = red_8x4();
    let layout = generated_image_layout(&pixels, "");
    let page = layout.page(0).unwrap();
    let runs = page.text_runs();
    let events = page.paint_order_for_text_runs(&runs);
    let image = events
        .iter()
        .find_map(|event| match event {
            PaintEvent::GeneratedImage(image) => Some(*image),
            _ => None,
        })
        .expect("the ::before image has a paint event");
    assert_eq!(image.kind, raikiri_html::GeneratedKind::Before);
    assert_eq!(image.index, 0);
    assert!(order::supported(&events, &runs, |_| None));
    let source = RenderResources::new().image_pixel_source(&pixels);
    for with_source in [false, true] {
        let mut pdf = krilla::Document::new();
        let mut pdf_page = pdf.start_page_with(PageSettings::from_wh(100.0, 100.0).unwrap());
        let mut surface = pdf_page.surface();
        let mut fonts = FontCache::default();
        let mut svg = svg::SvgCache::new(crate::RenderOptions::default());
        let mut raster = raster::RasterCache::for_tests(
            with_source
                .then(|| source.image_pixel_source_ref())
                .flatten(),
        );
        paint_ordered(
            &mut surface,
            &page,
            &events,
            &runs,
            &mut fonts,
            &mut svg,
            &mut raster,
            &mut crate::tagging::Tags::disabled(),
            None,
        )
        .unwrap();
        surface.finish();
        pdf_page.finish();
        let (images, operations) = images_and_draws(&pdf.finish().unwrap());
        if with_source {
            assert_eq!(images, vec![[255, 0, 0].repeat(32)]);
            let placement = raster.generated_image(&page, image).unwrap();
            assert_eq!(
                (
                    placement.rect.x,
                    placement.rect.width,
                    placement.rect.height
                ),
                (0.0, 8.0, 4.0)
            );
            assert!(placement.rect.y > 0.0 && placement.rect.y + 4.0 <= 20.0);
        } else {
            assert!(images.is_empty());
        }
        assert_eq!(
            operations.iter().filter(|op| op.operator == "Do").count(),
            usize::from(with_source)
        );
        let saves = operations.iter().filter(|op| op.operator == "q").count();
        let restores = operations.iter().filter(|op| op.operator == "Q").count();
        assert_eq!(saves, restores);
    }
}

#[test]
fn legacy_pages_draw_generated_images_with_their_owner_opacity() {
    let pixels = red_8x4();
    let layout = generated_image_layout(&pixels, "p{opacity:.5}");
    let page = layout.page(0).unwrap();
    let source = RenderResources::new().image_pixel_source(&pixels);
    let mut pdf = krilla::Document::new();
    let mut pdf_page = pdf.start_page_with(PageSettings::from_wh(100.0, 100.0).unwrap());
    let mut surface = pdf_page.surface();
    let mut fonts = FontCache::default();
    let mut svg = svg::SvgCache::new(crate::RenderOptions::default());
    let mut raster = raster::RasterCache::for_tests(source.image_pixel_source_ref());
    paint_legacy(
        &mut surface,
        &page,
        &page.paint_order(),
        &page.text_runs(),
        &mut fonts,
        &mut svg,
        &mut raster,
        &mut crate::tagging::Tags::disabled(),
        None,
    )
    .unwrap();
    surface.finish();
    pdf_page.finish();
    let bytes = pdf.finish().unwrap();
    let (images, _) = images_and_draws(&bytes);
    assert_eq!(images, vec![[255, 0, 0].repeat(32)]);
    // The legacy painter has no opacity groups, so the image carries its
    // owner's opacity itself.
    let parsed = lopdf::Document::load_mem(&bytes).unwrap();
    assert!(
        parsed
            .objects
            .values()
            .filter_map(|object| object.as_dict().ok())
            .any(|dict| dict.get(b"ca").and_then(lopdf::Object::as_float).ok() == Some(0.5))
    );
}
