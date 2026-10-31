use fulgur::{Engine, PageSize};

fn render(html: &str) -> Vec<u8> {
    let engine = Engine::builder().page_size(PageSize::A4).build();
    engine.render(html).expect("render should succeed")
}

fn render_small(html: &str) -> Vec<u8> {
    Engine::builder()
        .page_size(PageSize::custom(80.0, 80.0))
        .margin(fulgur::Margin::uniform(5.0))
        .tagged(true)
        .build()
        .render(html)
        .expect("render should succeed")
}

fn count_red_ops_in_content(bytes: &[u8]) -> usize {
    // Returning 0 here would let "no red operators on this page" hold for a
    // stream we simply failed to parse, hiding the very regression the callers
    // assert against.
    let content =
        lopdf::content::Content::decode(bytes).expect("page content stream should decode");
    content
        .operations
        .iter()
        .filter(|op| matches!(op.operator.as_str(), "rg" | "RG") && op.operands.len() == 3)
        .filter(|op| {
            let channels: Vec<f32> = op
                .operands
                .iter()
                .filter_map(|o| match o {
                    lopdf::Object::Integer(i) => Some(*i as f32),
                    lopdf::Object::Real(f) => Some(*f),
                    _ => None,
                })
                .collect();
            channels.len() == 3 && channels[0] > 0.9 && channels[1] < 0.1 && channels[2] < 0.1
        })
        .count()
}

fn count_red_ops_on_page(pdf: &[u8], page_number: u32) -> usize {
    let document = lopdf::Document::load_mem(pdf).expect("PDF should parse");
    let page_id = *document
        .get_pages()
        .get(&page_number)
        .expect("requested page should exist");
    let bytes = document.get_page_content(page_id);
    count_red_ops_in_content(&bytes)
}

fn annotation_quad_counts(document: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<usize> {
    let page = document
        .get_object(page_id)
        .expect("page object should exist")
        .as_dict()
        .expect("page object should be a dictionary");
    let Ok(annots) = page.get_deref(b"Annots", document) else {
        return Vec::new();
    };
    annots
        .as_array()
        .expect("Annots should resolve to an array")
        .iter()
        .map(|annotation| {
            let annotation = document
                .dereference(annotation)
                .expect("annotation reference should resolve")
                .1
                .as_dict()
                .expect("annotation should be a dictionary");
            assert_eq!(
                annotation
                    .get_deref(b"Subtype", document)
                    .expect("annotation should have a subtype")
                    .as_name()
                    .expect("annotation subtype should be a name"),
                b"Link"
            );
            assert_eq!(
                annotation
                    .get_deref(b"Rect", document)
                    .expect("link annotation should have a rectangle")
                    .as_array()
                    .expect("annotation rectangle should be an array")
                    .len(),
                4
            );
            let quad_points = annotation
                .get_deref(b"QuadPoints", document)
                .expect("link annotation should have quad points")
                .as_array()
                .expect("annotation quad points should be an array");
            assert_eq!(quad_points.len() % 8, 0);
            quad_points.len() / 8
        })
        .collect()
}

#[test]
fn table_simple_renders() {
    let html = r#"
    <table border="1">
        <thead><tr><th>Name</th><th>Value</th></tr></thead>
        <tbody><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></tbody>
    </table>"#;
    let pdf = render(html);
    assert!(!pdf.is_empty());
}

#[test]
fn table_no_thead_renders() {
    let html = r#"
    <table border="1">
        <tr><td>A</td><td>1</td></tr>
        <tr><td>B</td><td>2</td></tr>
    </table>"#;
    let pdf = render(html);
    assert!(!pdf.is_empty());
}

#[test]
fn table_long_with_thead_renders() {
    let mut rows = String::new();
    for i in 0..50 {
        rows.push_str(&format!(
            "<tr><td>Row {i}</td><td>Value {i}</td><td>Data {i}</td></tr>"
        ));
    }
    let html = format!(
        r#"
    <table border="1">
        <thead><tr><th>Name</th><th>Value</th><th>Data</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"#
    );
    let pdf = render(&html);
    assert!(!pdf.is_empty());
}

#[test]
fn repeated_table_header_renders_with_clip_link_and_tags() {
    let rows = (1..=30)
        .map(|row| format!("<tr><td>Row {row:02}</td></tr>"))
        .collect::<String>();
    let html = format!(
        r#"<!doctype html>
<style>
  html, body {{ margin: 0; }}
  table {{ width: 100%; overflow: hidden; border-collapse: collapse; }}
  th, td {{ height: 12px; padding: 0; }}
</style>
<table>
  <thead><tr><th><a href="https://example.com">H</a></th></tr></thead>
  <tbody>{rows}</tbody>
</table>"#
    );

    let pdf = render_small(&html);
    let document = lopdf::Document::load_mem(&pdf).expect("PDF should parse");
    assert!(
        document
            .catalog()
            .expect("PDF must have a catalog")
            .has(b"StructTreeRoot"),
        "tagged PDF catalog must contain /StructTreeRoot"
    );
    let pages = document.get_pages();
    assert!(pages.len() >= 2, "table should span at least two pages");
    for (page_number, page_id) in pages {
        assert_eq!(
            annotation_quad_counts(&document, page_id),
            vec![1],
            "page {page_number} should contain one single-quad header link annotation"
        );
    }
}

/// A nested table whose last row forces a break after it closes with a
/// zero-height fragment on the page the break opens. Without a paint
/// guard, `table_box_size` falls back to the full layout box for that
/// slice and the inner table's red frame leaks onto the page that holds
/// none of its rows.
#[test]
fn a_zero_height_slice_of_a_split_nested_table_paints_no_frame() {
    let html = r#"<!doctype html>
<html><head><style>
  html, body { margin: 0; padding: 0; }
  #lead { height: 60px; }
  table { margin: 0; border-spacing: 0; width: 100px; }
  th, td { box-sizing: border-box; padding: 0; }
  #outer-header { height: 20px; }
  #inner { background: rgb(255, 0, 0); }
  .row { height: 30px; }
  #after { height: 30px; }
</style></head><body>
  <div id="lead"></div>
  <table><thead><tr><th id="outer-header">Outer</th></tr></thead><tbody><tr><td>
    <table id="inner"><tbody>
      <tr><td class="row">A</td></tr>
      <tr><td class="row" style="break-after: page">B</td></tr>
    </tbody></table>
    <div id="after"></div>
  </td></tr></tbody></table>
</body></html>"#;

    // 200x100 CSS px content box (150x75 pt).
    let engine = Engine::builder()
        .page_size(PageSize {
            width: 150.0,
            height: 75.0,
        })
        .margin(fulgur::Margin::uniform(0.0))
        .build();
    let layout = engine.layout(html).expect("layout should succeed");
    let inner_id = layout
        .drawables
        .tables
        .iter()
        .find(|(_, table)| table.id.as_deref().is_some_and(|id| id == "inner"))
        .map(|(id, _)| *id)
        .expect("inner table drawable");
    let inner_geom = layout
        .geometry
        .get(&inner_id)
        .expect("inner table geometry");
    assert!(inner_geom.is_split(), "the inner table must be split");
    let slices: Vec<(u32, bool)> = inner_geom
        .fragments
        .iter()
        .map(|fragment| {
            (
                fragment.page_index,
                fragment.height <= fulgur::units::Px::ZERO,
            )
        })
        .collect();
    let (empty_page, _) = *slices
        .iter()
        .find(|(_, empty)| *empty)
        .unwrap_or_else(|| panic!("the inner table must record a zero-height slice: {slices:?}"));
    let (rows_page, _) = *slices
        .iter()
        .find(|(_, empty)| !*empty)
        .expect("the inner table must hold its rows somewhere");

    let pdf = engine.render(html).expect("render should succeed");
    // PDF pages are numbered from 1.
    assert_eq!(
        count_red_ops_on_page(&pdf, empty_page + 1),
        0,
        "a zero-height slice must not paint the inner table's frame"
    );
    assert!(
        count_red_ops_on_page(&pdf, rows_page + 1) > 0,
        "the inner table's frame must paint where its rows are"
    );
}
