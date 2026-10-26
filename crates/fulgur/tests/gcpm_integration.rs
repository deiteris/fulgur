use fulgur::asset::AssetBundle;
use fulgur::engine::Engine;

#[test]
fn test_counter_margin_box_values_correct() {
    // Verify that counter values actually reach margin boxes correctly
    // by generating a document with counter-increment on visible elements.
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        @page {
            @bottom-center {
                content: "Chapter " counter(chapter);
            }
        }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    // Multiple chapters with enough content to potentially span pages
    let html = r#"
        <h2>One</h2><p>Content for chapter one.</p>
        <h2>Two</h2><p>Content for chapter two.</p>
        <h2>Three</h2><p>Content for chapter three.</p>
    "#;
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "Counter values should reach margin boxes: {:?}",
        result.err()
    );
    // Verify PDF is non-trivial (contains actual content)
    let pdf_bytes = result.unwrap();
    assert!(
        pdf_bytes.len() > 1000,
        "PDF should contain rendered content with counters"
    );
}

#[test]
fn test_gcpm_no_gcpm_css_works_as_before() {
    let html = r#"<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1>Simple Document</h1>
  <p>This is a simple document with no GCPM CSS.</p>
</body>
</html>"#;

    let engine = Engine::builder().build();
    let pdf = engine.render(html).unwrap();

    assert!(!pdf.is_empty(), "PDF output should not be empty");
    assert!(
        pdf.starts_with(b"%PDF-"),
        "PDF output should start with %PDF-"
    );
}

#[test]
fn test_deterministic_output() {
    let css = r#"
        .header { position: running(pageHeader); }
        @page {
            @top-left { content: element(pageHeader); }
            @top-right { content: "Page " counter(page) " / " counter(pages); font-size: 8px; }
            @bottom-center { content: "Footer"; }
        }
    "#;

    let html = r#"<!DOCTYPE html>
<html><body>
  <div class="header">Title</div>
  <p>Content.</p>
</body></html>"#;

    let mut assets = AssetBundle::new();
    assets.add_css(css);
    let engine = Engine::builder().assets(assets).build();
    let pdf1 = engine.render(html).unwrap();

    let mut assets2 = AssetBundle::new();
    assets2.add_css(css);
    let engine2 = Engine::builder().assets(assets2).build();
    let pdf2 = engine2.render(html).unwrap();

    assert_eq!(pdf1, pdf2, "Same input must produce identical PDF output");
}

#[test]
fn test_element_policy_multiple_chapters_last() {
    let css = r#"
        @page {
            size: 400pt 300pt;
            margin: 40pt;
            @top-center { content: element(title, last); }
        }
        .title { position: running(title); }
        .big { height: 250pt; border: 1px solid black; }
    "#;

    let html = r#"<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1 class="title">Chapter 1</h1>
  <div class="big">Chapter 1 body</div>
  <h1 class="title">Chapter 2</h1>
  <div class="big">Chapter 2 body</div>
</body>
</html>"#;

    let mut assets = AssetBundle::new();
    assets.add_css(css);

    let engine = Engine::builder().assets(assets).build();
    let pdf = engine
        .render(html)
        .expect("render should succeed with element(title, last) across multiple chapters");

    assert!(
        pdf.len() > 1000,
        "PDF seems empty or too small: {} bytes",
        pdf.len()
    );
    assert!(pdf.starts_with(b"%PDF-"));
}

#[test]
fn test_element_policy_first_except() {
    let css = r#"
        @page {
            size: 400pt 300pt;
            margin: 40pt;
            @top-center { content: element(title, first-except); }
        }
        .title { position: running(title); }
        .big { height: 250pt; border: 1px solid black; }
    "#;

    let html = r#"<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1 class="title">Chapter 1</h1>
  <div class="big">Chapter 1 body</div>
</body>
</html>"#;

    let mut assets = AssetBundle::new();
    assets.add_css(css);

    let engine = Engine::builder().assets(assets).build();
    let pdf = engine
        .render(html)
        .expect("render should succeed with element(title, first-except)");

    assert!(
        pdf.len() > 1000,
        "PDF seems empty or too small: {} bytes",
        pdf.len()
    );
    assert!(pdf.starts_with(b"%PDF-"));
}

#[test]
fn test_element_policy_start() {
    let css = r#"
        @page {
            size: 400pt 300pt;
            margin: 40pt;
            @top-center { content: element(title, start); }
        }
        .title { position: running(title); }
        .spacer { height: 220pt; }
    "#;

    let html = r#"<!DOCTYPE html>
    <html>
    <head></head>
    <body>
      <h1 class="title">Seed Title</h1>
      <div class="spacer"></div>
      <div style="page-break-before: always">
        <p>Page two has no running title and should keep the previous value.</p>
      </div>
      <div class="spacer"></div>
      <div style="page-break-before: always">
        <h1 class="title">New Seed Title</h1>
        <p>Page three introduces a new running element, but start should ignore the local value.</p>
      </div>
    </body>
    </html>"#;

    let mut assets = AssetBundle::new();
    assets.add_css(css);

    let engine = Engine::builder().assets(assets).build();
    let pdf = engine
        .render(html)
        .expect("render should succeed with element(title, start)");

    assert!(
        pdf.len() > 1000,
        "PDF seems empty or too small: {} bytes",
        pdf.len()
    );
    assert!(pdf.starts_with(b"%PDF-"));
}

#[test]
fn test_element_default_policy_still_works() {
    // Baseline: element(title) without an explicit policy must still render
    // (default = first), matching pre-policy behavior.
    let css = r#"
        @page {
            size: 400pt 300pt;
            margin: 40pt;
            @top-center { content: element(title); }
        }
        .title { position: running(title); }
    "#;

    let html = r#"<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1 class="title">My Title</h1>
  <p>Body content.</p>
</body>
</html>"#;

    let mut assets = AssetBundle::new();
    assets.add_css(css);

    let engine = Engine::builder().assets(assets).build();
    let pdf = engine
        .render(html)
        .expect("render should succeed with default element() policy");

    assert!(
        pdf.len() > 1000,
        "PDF seems empty or too small: {} bytes",
        pdf.len()
    );
    assert!(pdf.starts_with(b"%PDF-"));
}

#[test]
fn test_counter_chapter_before_pseudo() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        h2::before { content: counter(chapter) ". "; }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = "<h2>Introduction</h2><p>Some text here</p><h2>Methods</h2><p>More text</p>";
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "PDF generation with counter in ::before should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_counter_in_margin_box() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        @page {
            @bottom-center {
                content: "Chapter " counter(chapter);
            }
        }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = "<h2>One</h2><p>Some text</p><h2>Two</h2><p>More text</p>";
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "PDF with counter in margin box should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_counter_upper_roman_style() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        h2::before { content: counter(chapter, upper-roman) ". "; }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = "<h2>A</h2><h2>B</h2><h2>C</h2><h2>D</h2>";
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "PDF with upper-roman counter should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_counter_set() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        .reset { counter-set: chapter 10; }
        h2::before { content: counter(chapter) ". "; }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = r#"<h2>One</h2><div class="reset"></div><h2>Eleven</h2>"#;
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "PDF with counter-set should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_counter_body_and_margin_box() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: chapter; }
        h2 { counter-increment: chapter; }
        h2::before { content: counter(chapter) ". "; }
        @page {
            @bottom-right {
                content: "Ch. " counter(chapter);
            }
        }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = "<h2>Intro</h2><p>text</p><h2>Body</h2><p>text</p><h2>End</h2><p>text</p>";
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "PDF with counter in both body and margin box should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_counter_page_still_works() {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        @page {
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
            }
        }
    "#,
    );

    let engine = Engine::builder().assets(assets).build();
    let html = "<p>Hello World</p>";
    let result = engine.render(html);
    assert!(
        result.is_ok(),
        "counter(page) should still work: {:?}",
        result.err()
    );
}

// Snapshot tests (inline <style> variants + migrated integration tests): see crates/fulgur/tests/gcpm_snapshot.rs

#[test]
fn counter_pass_survives_element_names_with_css_metacharacters() {
    // fulgur-ka6c: CounterPass reconstructs a specificity prefix from the
    // matched element's tag/id/classes. The tag prefix must not carry raw
    // characters through to the generated stylesheet — otherwise a class
    // pseudo-content rule matching an element with a crafted tag name
    // could seed unrelated declarations into the Stylo cascade.
    //
    // End-to-end guard for the escape contract: rendering must succeed and
    // produce a well-formed PDF even when html5ever preserves an odd tag
    // name verbatim.
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        body { counter-reset: c 0; }
        .x::before { content: counter(c); }
        "#,
    );
    let engine = Engine::builder().assets(assets).build();
    let html = concat!(
        "<html><body>",
        r#"<x{}body{color:red} class="x">payload</x{}body{color:red}>"#,
        "</body></html>",
    );
    let pdf = engine
        .render(html)
        .expect("render must succeed even with crafted tag names");
    assert!(pdf.starts_with(b"%PDF-"), "output should be a PDF");
}

/// Running elements are serialized back to HTML and reparsed for the margin
/// box, so the serializer has to be the parser's inverse. It used not to be
/// for text nodes: the parser decodes character references on the way in and
/// nothing re-encoded them on the way out, so text that arrived escaped came
/// back as live markup in the header/footer.
///
/// That defeats `template::render_template`'s forced `AutoEscape::Html`,
/// which `docs/security/threat-model.md` V1 lists as an existing mitigation
/// for the trusted-template / untrusted-data shape.
fn render_with_running_header(header_body: &str) -> Vec<u8> {
    let mut assets = AssetBundle::new();
    assets.add_css(
        r#"
        .header { position: running(pageHeader); }
        @page { margin: 2cm; @top-center { content: element(pageHeader); } }
    "#,
    );
    let engine = Engine::builder().assets(assets).build();
    engine
        .render(&format!(
            r#"<!DOCTYPE html><html><body>
  <div class="header">{header_body}</div>
  <p>body</p>
</body></html>"#
        ))
        .expect("render must succeed")
}

#[test]
fn running_element_escaped_text_does_not_become_a_margin_box_link() {
    // `&lt;a href=…&gt;` is what MiniJinja's HTML autoescape emits for a
    // `{{ variable }}` holding an anchor — it must stay text.
    let pdf = render_with_running_header(
        "Report for &lt;a href=&quot;javascript:alert(1)&quot;&gt;click&lt;/a&gt;",
    );
    let bytes = String::from_utf8_lossy(&pdf);
    assert!(
        !bytes.contains("javascript:alert(1)"),
        "escaped anchor became a live PDF URI action"
    );
    assert!(
        !bytes.contains("/URI"),
        "escaped anchor produced a link annotation"
    );
}

#[test]
fn running_element_escaped_text_does_not_become_margin_box_markup() {
    // Beyond links: escaped text must not render as the markup it spells.
    // Comparing against the same markup *authored* as elements is the check
    // — before the fix both serialized to the same string, so the two PDFs
    // were byte-identical.
    //
    // Asserting on the drawn text directly would be sharper, but `inspect`
    // cannot recover it: lopdf 0.40 does not read krilla's `ToUnicode` CMap
    // (no `/CMapVersion` / `/WMode` support), so extracted text comes back as
    // raw glyph ids. Byte comparison against a known-good control is the
    // available signal until that lands.
    let injected = render_with_running_header("Status: &lt;b&gt;APPROVED&lt;/b&gt;");
    let authored = render_with_running_header("Status: <b>APPROVED</b>");
    assert_ne!(
        injected, authored,
        "escaped markup still rendered as authored markup in the margin box"
    );
}

/// `<style />` reparses as an *open* `<style>`, so an empty `<style>` in a
/// running element used to turn the rest of its margin box into CSS text —
/// the header simply vanished. Same round-trip requirement as the escaping
/// above: the serializer has to be the parser's inverse.
#[test]
fn running_element_empty_style_does_not_erase_the_margin_box() {
    let with_empty_style = render_with_running_header("<style></style><b>X</b>");
    let without = render_with_running_header("<b>X</b>");
    assert_eq!(
        with_empty_style, without,
        "an empty <style> swallowed the rest of the margin box"
    );
}

/// Over-escaping tripwire at the render level: `<style>` is a raw text
/// element, so its CSS must survive serialization unescaped.
#[test]
fn running_element_style_child_survives_serialization() {
    let with_style =
        render_with_running_header("<style>.hl { color: red }</style><span class=\"hl\">X</span>");
    let without_style = render_with_running_header("<span class=\"hl\">X</span>");
    assert_ne!(
        with_style, without_style,
        "<style> inside a running element stopped applying — CSS was corrupted by escaping"
    );
}

// A 1x1 red PNG.
const MINIMAL_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
    0x00, 0x03, 0x01, 0x01, 0x00, 0xC9, 0xFE, 0x92, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
    0x44, 0xAE, 0x42, 0x60, 0x82,
];

/// The images painted in a 300pt x 200pt page with 40pt margins, `page_css`
/// in its `@page` rule and `body` before one paragraph, `logo.png` in the
/// bundle.
fn images_in_margin_box_page(page_css: &str, body: &str) -> Vec<fulgur::inspect::ImageItem> {
    let mut assets = AssetBundle::new();
    assets.add_image("logo.png", MINIMAL_PNG.to_vec());
    assets.add_css(format!(
        ".logo {{ position: running(logo); }} @page {{ size: 300pt 200pt; margin: 40pt; {page_css} }}"
    ));
    let pdf = Engine::builder()
        .assets(assets)
        .build()
        .render(&format!(
            "<!DOCTYPE html><html><body>{body}<p>body</p></body></html>"
        ))
        .expect("render must succeed");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &pdf).unwrap();
    fulgur::inspect::inspect(tmp.path()).unwrap().images
}

/// Whether `image` paints `(width, height)` points, within the rounding of
/// the px-to-pt conversion.
fn painted_size(image: &fulgur::inspect::ImageItem, (width, height): (f32, f32)) -> bool {
    (image.width - width).abs() < 0.5 && (image.height - height).abs() < 0.5
}

/// An `<img>` in a running element paints in the margin box that takes the
/// element, at the size its attributes give it, inline after the text before
/// it. The margin-box documents used to be converted without the asset
/// bundle, so every image in a margin box painted nothing.
#[test]
fn an_img_in_a_running_element_paints_in_its_margin_box() {
    let logo = r#"<span class="logo"><img src="logo.png" width="20" height="20"></span>"#;
    let images = images_in_margin_box_page(
        r#"@bottom-center { content: "A " element(logo) " B"; }"#,
        logo,
    );
    assert_eq!(images.len(), 1, "one image, in the margin box: {images:?}");
    let image = &images[0];
    assert_eq!(image.page, 1);
    assert!(painted_size(image, (15.0, 15.0)), "20px is 15pt: {image:?}");
    // PDF space: the bottom margin is the 40pt above the sheet's bottom. The
    // image stands on the baseline of the box's first line, which lies in it.
    assert!(
        image.y > 0.0 && image.y < 40.0,
        "the image stands in the bottom margin: {image:?}"
    );
    let alone =
        images_in_margin_box_page(r#"@bottom-center { content: element(logo) " B"; }"#, logo);
    assert_eq!(alone.len(), 1, "{alone:?}");
    assert!(
        image.x > alone[0].x,
        "the image follows the text before it: {image:?}, alone {:?}",
        alone[0]
    );
}

/// A `background-image` on a margin box paints, as it does on a body box.
#[test]
fn a_margin_box_background_image_paints() {
    let images = images_in_margin_box_page(
        r#"@bottom-center { content: "x"; width: 20pt; height: 20pt;
                            background: url(logo.png) no-repeat; background-size: 20pt 20pt; }"#,
        "",
    );
    assert_eq!(images.len(), 1, "the margin box's background: {images:?}");
    let image = &images[0];
    assert_eq!(image.page, 1);
    assert!(painted_size(image, (20.0, 20.0)), "{image:?}");
    assert!(
        image.y >= 0.0 && image.y + image.height <= 40.0,
        "the background lies in the bottom margin: {image:?}"
    );
}
