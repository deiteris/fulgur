//! The canvas background (CSS Backgrounds 3 §2.11.2).
//!
//! The background of the root element becomes the background of the canvas
//! and is painted under the page area of every page, not inside the root's
//! own box. When the root is an `<html>` element whose background is
//! transparent (`background-color: transparent` and no image), the first
//! `<body>` child propagates its background to the canvas instead. The
//! element that gave its background to the canvas does not paint it again.
//! An element with `display: none` generates no box, so it leaves the canvas
//! transparent.

use super::paint_background;
use super::raster::RasterCache;
use super::shape::{Edges, RoundedRect, Slice};
use crate::tagging::{Tags, Target};
use fulgur_core::Result;
use krilla::num::NormalizedF32;
use krilla::surface::Surface;
use krilla::tagging::ArtifactType;
use raikiri_html::computed::{
    ComputedBackgroundImage, ComputedDisplay, ComputedValues, ComputedVisualBox, CssColor,
};
use raikiri_html::{NodeId, NodeKind, Page};

/// The element whose background is the canvas background of `page`, with
/// its computed style.
pub(super) fn owner<'p>(page: &'p Page<'_>) -> Option<(NodeId, &'p ComputedValues)> {
    let html = root(page)?;
    let html_style = page.computed(html)?;
    // A root without a box leaves the canvas transparent; its body does not
    // take over.
    if !generates_box(html_style) {
        return None;
    }
    if has_background(html_style) {
        return Some((html, html_style));
    }
    let dom = page.dom();
    let body = dom
        .children(html)
        .find(|&node| is_element(page, node, "body"))?;
    page.computed(body)
        .filter(|style| has_background(style) && generates_box(style))
        .map(|style| (body, style))
}

/// The root `<html>` element of `page`.
fn root(page: &Page<'_>) -> Option<NodeId> {
    let dom = page.dom();
    dom.children(dom.root())
        .find(|&node| is_element(page, node, "html"))
}

fn is_element(page: &Page<'_>, node: NodeId, name: &str) -> bool {
    let dom = page.dom();
    dom.kind(node) == Some(NodeKind::Element) && dom.local_name(node) == Some(name)
}

fn generates_box(style: &ComputedValues) -> bool {
    style.display != ComputedDisplay::None
}

fn has_background(style: &ComputedValues) -> bool {
    style.background_color.a > 0 || !matches!(style.background_image, ComputedBackgroundImage::None)
}

/// Paint the canvas background of `page`, if any, as an artifact, and return
/// the element that gave its background to the canvas.
pub(super) fn paint_page(
    surface: &mut Surface<'_>,
    tags: &mut Tags,
    raster: &mut RasterCache<'_>,
    page: &Page<'_>,
) -> Result<Option<NodeId>> {
    let Some((owner, style)) = owner(page) else {
        return Ok(None);
    };
    tags.mark(
        surface,
        page,
        Target::Artifact(ArtifactType::Other),
        |surface| paint(surface, raster, page, style),
    )
    .map(|()| Some(owner))
}

/// The opacity of the root element's group, when it fades.
///
/// The paint order starts at the body, so the root's group, which holds the
/// canvas as well as the page content, is left to the page painter
/// (Compositing and Blending 1 §3.3.1).
pub(super) fn root_opacity(page: &Page<'_>) -> Option<NormalizedF32> {
    root(page)
        .and_then(|html| page.computed(html))
        .and_then(|style| NormalizedF32::new(style.opacity))
        .filter(|alpha| *alpha < NormalizedF32::ONE)
}

/// Paint `style`'s background over the page area of `page`.
///
/// The page area also positions the background image: the painter does not
/// keep the root element's unbroken box, which the spec uses for that.
fn paint(
    surface: &mut Surface<'_>,
    raster: &mut RasterCache<'_>,
    page: &Page<'_>,
    style: &ComputedValues,
) -> Result<()> {
    let geometry = page.geometry();
    let (page_box, margins) = (geometry.page_box, geometry.margins);
    let area = RoundedRect::rect(
        page_box.x + margins.left,
        page_box.y + margins.top,
        (page_box.width - margins.left - margins.right).max(0.0),
        (page_box.height - margins.top - margins.bottom).max(0.0),
    );
    // The page area has no borders or padding of its own: every edge is
    // treated as broken so the owner's are not subtracted.
    let open = Slice {
        top: true,
        bottom: true,
        left: true,
        right: true,
    };
    let mut style = style.clone();
    // `background-clip` has no effect on the canvas, which covers the whole
    // page area (CSS Backgrounds 3 §2.11.1). The page area has no border or
    // padding, so the other boxes already coincide with it; `text` and
    // `border-area` would otherwise paint glyphs or an empty ring.
    if matches!(
        style.background_clip,
        ComputedVisualBox::Text | ComputedVisualBox::BorderArea
    ) {
        style.background_clip = ComputedVisualBox::BorderBox;
    }
    paint_background(
        surface,
        raster,
        &area,
        Edges::default(),
        open,
        &style,
        || None,
    )
}

/// `style` without its background, for the element that gave it to the
/// canvas.
pub(super) fn without_background(style: &ComputedValues) -> ComputedValues {
    let mut style = style.clone();
    style.background_color = CssColor::TRANSPARENT;
    style.background_image = ComputedBackgroundImage::None;
    style
}
