use super::*;

/// The `rg` color of each fill (`f`) of `operations`, in paint order.
fn fills(operations: &[lopdf::content::Operation]) -> Vec<[u8; 3]> {
    operations
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op.operator.as_str(), "f" | "f*"))
        .filter_map(|(index, _)| fill_colors(&operations[..index]).last().copied())
        .collect()
}

const PAGES: &str = "<style>@page { size: 300px 200px; margin: 20px } \
    p { margin: 0; height: 100px }</style>";

#[test]
fn body_background_fills_the_page_area_of_every_page() {
    let (_dir, path) = input(&format!(
        "{PAGES}<body style=\"margin: 0; background: rgb(0, 128, 0)\">\
         <p>one</p><p>two</p><p>three</p></body>"
    ));
    let bytes = render(&path, &Config::default()).expect("PDF bytes");
    let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
    assert_eq!(pdf.get_pages().len(), 2);
    for page in [1, 2] {
        // CSS Backgrounds 3 §2.11.2: the body background is the canvas
        // background, painted once over the page area and not again inside
        // the body's own box.
        assert_eq!(
            fills(&page_operations(&pdf, page)),
            [[0, 128, 0]],
            "page {page}"
        );
    }
}

#[test]
fn root_background_wins_over_body_background() {
    let (_dir, path) = input(&format!(
        "{PAGES}<html style=\"background: rgb(0, 0, 255)\">\
         <body style=\"margin: 0; background: rgb(0, 128, 0)\"><p>one</p></body></html>"
    ));
    let bytes = render(&path, &Config::default()).expect("PDF bytes");
    let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
    // The root's background covers the page area; the body keeps its own,
    // painted inside its box over the canvas.
    assert_eq!(fills(&page_operations(&pdf, 1)), [[0, 0, 255], [0, 128, 0]]);
}

#[test]
fn transparent_root_and_body_paint_no_canvas() {
    let (_dir, path) = input(&format!(
        "{PAGES}<body style=\"margin: 0\"><p style=\"background: rgb(0, 128, 0)\">one</p></body>"
    ));
    let bytes = render(&path, &Config::default()).expect("PDF bytes");
    let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
    assert_eq!(fills(&page_operations(&pdf, 1)), [[0, 128, 0]]);
}

#[test]
fn canvas_background_is_an_artifact_in_tagged_pdf() {
    let (_dir, path) = input(&format!(
        "{PAGES}<html lang=\"en\"><title>Canvas</title><body style=\"margin: 0; background: rgb(0, 128, 0)\">\
         <p>one</p></body></html>"
    ));
    let config = Config::builder().pdf_ua(true).build();
    let bytes = render(&path, &config).expect("a PDF/UA document");
    assert!(bytes.starts_with(b"%PDF"));
}

#[test]
fn a_display_none_owner_leaves_the_canvas_transparent() {
    // CSS Backgrounds 3 §2.11: an element that generates no box gives no
    // background to the canvas, and the body does not take over from the
    // root.
    for (html, canvas) in [
        (
            "<html style=\"display: none; background: rgb(0, 0, 255)\">\
             <body style=\"margin: 0\"><p>one</p></body></html>",
            [0, 0, 255],
        ),
        (
            "<body style=\"display: none; background: rgb(0, 128, 0)\"><p>one</p></body>",
            [0, 128, 0],
        ),
    ] {
        let (_dir, path) = input(&format!("{PAGES}{html}"));
        let bytes = render(&path, &Config::default()).expect("PDF bytes");
        let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
        for page in pdf.get_pages().keys() {
            let fills = fills(&page_operations(&pdf, *page));
            assert!(!fills.contains(&canvas), "{html}: {fills:?}");
        }
    }
}

#[test]
fn background_clip_does_not_shrink_the_canvas() {
    for clip in ["text", "border-area", "content-box"] {
        let (_dir, path) = input(&format!(
            "{PAGES}<body style=\"margin: 0; padding: 10px; border: 5px solid rgb(255, 0, 0); \
             background: rgb(0, 128, 0); background-clip: {clip}\"><p>one</p></body>"
        ));
        let bytes = render(&path, &Config::default()).expect("PDF bytes");
        let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
        let fills = fills(&page_operations(&pdf, 1));
        assert_eq!(fills.first(), Some(&[0, 128, 0]), "{clip}: {fills:?}");
    }
}

#[test]
fn the_canvas_shares_the_root_opacity_group() {
    // Compositing and Blending 1 §3.3.1: the canvas background is painted
    // into the root element's group, with the root's content, so the root's
    // opacity fades the two together.
    let (_dir, path) = input(&format!(
        "{PAGES}<html style=\"opacity: 0.5; background: rgb(0, 0, 255)\">\
         <body style=\"margin: 0\">\
         <div style=\"height: 50px; background: rgb(255, 0, 0)\"></div>\
         </body></html>"
    ));
    let bytes = render(&path, &Config::default()).expect("PDF bytes");
    let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
    let operations = page_operations(&pdf, 1);
    // The canvas and the content are drawn through one transparency group,
    // not straight onto the page.
    assert!(
        !operations.iter().any(|op| op.operator == "f")
            && operations.iter().filter(|op| op.operator == "Do").count() == 1,
        "{operations:?}"
    );
}

#[test]
fn the_canvas_goes_under_the_margin_boxes() {
    // CSS Paged Media 3 §5.1: the canvas is painted below the page-margin
    // boxes, so one that reaches into the page area stays visible.
    let (_dir, path) = input(
        "<style>@page { size: 300px 200px; margin: 20px; \
         @top-center { content: 'x'; background: rgb(255, 0, 0) } }</style>\
         <html style=\"background: rgb(0, 0, 255)\"><body></body></html>",
    );
    let bytes = render(&path, &Config::default()).expect("PDF bytes");
    let pdf = lopdf::Document::load_mem(&bytes).expect("a readable PDF");
    let fills = fills(&page_operations(&pdf, 1));
    assert_eq!(fills.first(), Some(&[0, 0, 255]), "{fills:?}");
    assert!(fills.contains(&[255, 0, 0]), "{fills:?}");
}
