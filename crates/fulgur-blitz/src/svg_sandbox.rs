//! Keep SVG `<image>` references from reaching the file system.
//!
//! `usvg` resolves a non-`data:` `href` on `<image>` / `<feImage>` by reading
//! the path from disk, relative to the process working directory when the
//! parser has no `resources_dir`. Blitz parses inline `<svg>` elements and
//! `<img src="*.svg">` payloads with those defaults, so a document that was
//! given no file access could still embed any file the process can read.
//!
//! An `<img>` follows the document's asset bundle and base path; an SVG
//! reference gets the same rule here: only `data:` URLs resolve. Other
//! references are removed before the SVG reaches `usvg`.

use std::borrow::Cow;

use blitz_html::HtmlDocument;
use usvg::roxmltree;

/// Parser options for SVG that Fulgur parses itself (list markers and
/// CSS backgrounds): `data:` URLs resolve, nothing is read from disk.
pub fn options() -> usvg::Options<'static> {
    usvg::Options {
        image_href_resolver: usvg::ImageHrefResolver {
            resolve_string: Box::new(|_, _| None),
            ..usvg::ImageHrefResolver::default()
        },
        ..usvg::Options::default()
    }
}

fn is_image_element(name: &str) -> bool {
    name == "image" || name == "feImage"
}

/// Whether `usvg` may be handed this `href`: a well-formed `data:` URL, or
/// nothing. `usvg` reads the string as a file path when it does not parse as
/// a data URL, so a malformed `data:` prefix (`data:/x`) is not enough.
fn is_allowed_href(href: &str) -> bool {
    let href = href.trim_start();
    href.is_empty() || data_url::DataUrl::process(href).is_ok()
}

/// Remove every `href` / `xlink:href` on an `<image>` or `<feImage>` of an
/// inline `<svg>` subtree that is not a `data:` URL.
pub fn sandbox_inline_svg(doc: &mut HtmlDocument) {
    let mut strip = Vec::new();
    for (id, node) in doc.tree().iter() {
        let Some(el) = node.element_data() else {
            continue;
        };
        if !is_image_element(&el.name.local) {
            continue;
        }
        for attr in el.attrs() {
            if &*attr.name.local == "href" && !is_allowed_href(&attr.value) {
                strip.push((id, attr.name.clone()));
            }
        }
    }
    if strip.is_empty() {
        return;
    }
    let mut mutator = doc.mutate();
    for (id, name) in strip {
        mutator.clear_attribute(id, name);
    }
}

/// Rewrite the bytes of an SVG file (plain or gzip-compressed) so that no
/// `<image>` / `<feImage>` carries a non-`data:` `href`. Returns `None` when
/// the bytes are not SVG, or when they need no change.
pub fn sandbox_svg_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    let plain: Cow<[u8]> = if bytes.starts_with(&[0x1f, 0x8b]) {
        Cow::Owned(usvg::decompress_svgz(bytes).ok()?)
    } else {
        Cow::Borrowed(bytes)
    };
    let text = std::str::from_utf8(&plain).ok()?;
    let xml = roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
    .ok()?;
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for node in xml.descendants().filter(|n| n.is_element()) {
        if !is_image_element(node.tag_name().name()) {
            continue;
        }
        for attr in node.attributes() {
            if attr.name() == "href" && !is_allowed_href(attr.value()) {
                ranges.push(attr.range_value());
            }
        }
    }
    if ranges.is_empty() && matches!(plain, Cow::Borrowed(_)) {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for r in ranges {
        out.push_str(&text[at..r.start]);
        at = r.end;
    }
    out.push_str(&text[at..]);
    Some(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::F32Units;

    const PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0xC9, 0xFE, 0x92, 0xEF, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    fn data_url() -> String {
        const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::from("data:image/png;base64,");
        for c in PNG.chunks(3) {
            let n = (c[0] as u32) << 16
                | (*c.get(1).unwrap_or(&0) as u32) << 8
                | *c.get(2).unwrap_or(&0) as u32;
            for i in 0..4 {
                if i <= c.len() {
                    s.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
                } else {
                    s.push('=');
                }
            }
        }
        s
    }

    fn count_images(group: &usvg::Group) -> usize {
        group
            .children()
            .iter()
            .map(|n| match n {
                usvg::Node::Image(_) => 1,
                usvg::Node::Group(g) => count_images(g),
                _ => 0,
            })
            .sum()
    }

    /// A PNG on disk, and the absolute path that names it.
    fn file_png() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret.png");
        std::fs::write(&path, PNG).unwrap();
        let href = path.to_string_lossy().replace('\\', "/");
        (dir, href)
    }

    fn inline_svg_images(href: &str) -> usize {
        let html = format!(
            "<!DOCTYPE html><html><body><svg width=\"20\" height=\"20\">\
             <image href=\"{href}\" width=\"20\" height=\"20\"/></svg></body></html>"
        );
        let doc = crate::blitz_adapter::parse_and_layout(
            &html,
            200.0_f32.as_px(),
            200.0_f32.as_px(),
            &[],
            true,
        );
        doc.tree()
            .iter()
            .filter_map(|(_, n)| n.element_data()?.svg_data())
            .map(|t| count_images(t.root()))
            .sum()
    }

    #[test]
    fn inline_svg_image_with_a_file_href_reads_nothing() {
        let (_dir, href) = file_png();
        assert_eq!(inline_svg_images(&href), 0);
    }

    #[test]
    fn inline_svg_image_with_a_malformed_data_href_reads_nothing() {
        assert_eq!(inline_svg_images("data:/secret.png"), 0);
        assert!(!is_allowed_href("data:/secret.png"));
        assert!(!is_allowed_href("data:image/png;base64"));
        assert!(is_allowed_href(&data_url()));
    }

    #[test]
    fn inline_svg_image_with_an_xlink_file_href_reads_nothing() {
        let (_dir, href) = file_png();
        let html = format!(
            "<!DOCTYPE html><html><body><svg width=\"20\" height=\"20\"              xmlns:xlink=\"http://www.w3.org/1999/xlink\">             <image xlink:href=\"{href}\" width=\"20\" height=\"20\"/></svg></body></html>"
        );
        let doc = crate::blitz_adapter::parse_and_layout(
            &html,
            200.0_f32.as_px(),
            200.0_f32.as_px(),
            &[],
            true,
        );
        let n: usize = doc
            .tree()
            .iter()
            .filter_map(|(_, n)| n.element_data()?.svg_data())
            .map(|t| count_images(t.root()))
            .sum();
        assert_eq!(n, 0);
    }

    #[test]
    fn inline_svg_image_with_a_data_href_still_paints() {
        assert_eq!(inline_svg_images(&data_url()), 1);
    }

    #[test]
    fn svg_file_bytes_lose_file_hrefs_and_keep_data_hrefs() {
        let (_dir, href) = file_png();
        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"20\" height=\"20\">\
             <image xlink:href=\"{href}\" width=\"20\" height=\"20\"/>\
             <image href=\"{}\" width=\"20\" height=\"20\"/></svg>",
            data_url()
        );
        let raw = usvg::Tree::from_data(svg.as_bytes(), &usvg::Options::default()).unwrap();
        assert_eq!(
            count_images(raw.root()),
            2,
            "the default parser reads the file"
        );

        let clean = sandbox_svg_bytes(svg.as_bytes()).expect("rewritten");
        let tree = usvg::Tree::from_data(&clean, &usvg::Options::default()).unwrap();
        assert_eq!(count_images(tree.root()), 1);
        assert!(sandbox_svg_bytes(&clean).is_none(), "already clean");
        assert!(sandbox_svg_bytes(b"body { color: red }").is_none());
    }

    #[test]
    fn options_resolve_data_urls_and_no_files() {
        let (_dir, href) = file_png();
        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\">\
             <image href=\"{href}\" width=\"20\" height=\"20\"/>\
             <image href=\"{}\" width=\"20\" height=\"20\"/></svg>",
            data_url()
        );
        let tree = usvg::Tree::from_data(svg.as_bytes(), &options()).unwrap();
        assert_eq!(count_images(tree.root()), 1);
    }
}
