use super::*;

const BOX_CSS: &str = "<style>@page {size:300px 200px;margin:0} body {margin:0} \
    div {width:100px;height:50px;margin:30px}</style>";

/// The `rg` fill colors and image draws (`Do`), in paint order.
fn paints(operations: &[lopdf::content::Operation]) -> Vec<String> {
    operations
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match op.operator.as_str() {
            "f" | "f*" => fill_colors(&operations[..index])
                .last()
                .map(|[r, g, b]| format!("{r},{g},{b}")),
            "Do" => Some("image".to_owned()),
            _ => None,
        })
        .collect()
}

/// Bounding box `[x0, y0, x1, y1]` in CSS px of every subpath of each
/// even-odd clip (`W*`), for the 300 x 200 px pages of [`BOX_CSS`].
fn even_odd_clip_bounds(operations: &[lopdf::content::Operation]) -> Vec<[f32; 4]> {
    let construction = ["m", "l", "c", "v", "y", "h", "re"];
    let mut bounds = Vec::new();
    for (index, operation) in operations.iter().enumerate() {
        if operation.operator != "W*" {
            continue;
        }
        let start = operations[..index]
            .iter()
            .rposition(|op| !construction.contains(&op.operator.as_str()))
            .map_or(0, |found| found + 1);
        let mut bbox = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for op in &operations[start..index] {
            let values: Vec<f32> = op
                .operands
                .iter()
                .filter_map(|value| value.as_float().ok())
                .collect();
            for point in values.as_chunks::<2>().0 {
                bbox = [
                    bbox[0].min(point[0]),
                    bbox[1].min(point[1]),
                    bbox[2].max(point[0]),
                    bbox[3].max(point[1]),
                ];
            }
        }
        let px = |pt: f32| pt / 0.75;
        bounds.push([
            px(bbox[0]),
            px(150.0 - bbox[3]),
            px(bbox[2]),
            px(150.0 - bbox[1]),
        ]);
    }
    bounds
}

fn soft_masked_images(pdf: &lopdf::Document) -> Vec<&lopdf::Stream> {
    pdf.objects
        .values()
        .filter_map(|object| object.as_stream().ok())
        .filter(|stream| {
            stream
                .dict
                .get(b"Subtype")
                .is_ok_and(|subtype| subtype.as_name().is_ok_and(|name| name == b"Image"))
                && stream.dict.has(b"SMask")
        })
        .collect()
}

#[test]
fn outer_shadows_are_below_the_background_first_declared_on_top() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='background:rgb(0,128,0);\
         box-shadow:4px 4px rgb(255,0,0), 8px 8px rgb(0,0,255)'></div>"
    ));
    assert_eq!(paints(&ops), ["0,0,255", "255,0,0", "0,128,0"]);
    // Each shadow is clipped to the outside of the border box.
    assert_eq!(count(&ops, "W*"), 2);
}

#[test]
fn outer_shadow_clip_leaves_out_the_border_box() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='box-shadow:0 0 0 10px rgb(255,0,0)'></div>"
    ));
    assert_eq!(paints(&ops), ["255,0,0"]);
    // The clip runs from past the spread shadow around the border box.
    let [x0, y0, x1, y1] = even_odd_clip_bounds(&ops)[0];
    assert!(
        x0 <= 20.0 && y0 <= 20.0 && x1 >= 140.0 && y1 >= 90.0,
        "{x0} {y0} {x1} {y1}"
    );
}

#[test]
fn current_color_shadow_uses_the_element_color() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='color:rgb(0,128,0);box-shadow:2px 2px, inset 1px 1px'></div>"
    ));
    assert_eq!(paints(&ops), ["0,128,0", "0,128,0"]);
}

#[test]
fn transparent_and_overflowing_shadows_are_skipped() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='box-shadow:2px 2px transparent, inset 1px 1px 3px transparent, \
         1e40px 1px red, 1px 1px 1e40px red'></div>"
    ));
    assert!(paints(&ops).is_empty(), "{:?}", paints(&ops));
    assert_eq!(count(&ops, "W*"), 0);
}

#[test]
fn hidden_boxes_draw_no_shadow() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='visibility:hidden;box-shadow:2px 2px red'></div>"
    ));
    assert!(paints(&ops).is_empty());
}

#[test]
fn inset_shadows_are_between_the_background_and_the_border() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='background:rgb(0,128,0);border:2px solid rgb(0,0,0);\
         box-shadow:inset 3px 3px rgb(255,0,0), inset -3px -3px rgb(0,0,255)'></div>"
    ));
    assert_eq!(
        paints(&ops),
        ["0,128,0", "0,0,255", "255,0,0", "0,0,0"],
        "background, inset shadows last declared first, then the border"
    );
}

#[test]
fn blurred_shadows_are_soft_masked_images() {
    let (pdf, ops) = operations(&format!(
        "{BOX_CSS}<div style='background:rgb(0,128,0);\
         box-shadow:2px 2px 4px rgb(255,0,0), inset 0 0 6px 2px rgb(0,0,255)'></div>"
    ));
    assert_eq!(paints(&ops), ["image", "0,128,0", "image"]);
    let images = soft_masked_images(&pdf);
    assert_eq!(images.len(), 2);
    for image in images {
        let width = image.dict.get(b"Width").unwrap().as_i64().unwrap();
        // The raster covers the 100px box plus the blur on both sides.
        assert!(width > 100, "{width}");
    }
}

#[test]
fn page_breaks_cut_the_shadow_at_the_broken_edge() {
    let (_, ops) = operations(
        "<style>@page {size:300px 200px;margin:0} body {margin:0} \
         div {width:100px;height:300px;margin-left:30px;box-shadow:0 0 0 10px rgb(255,0,0)}</style>\
         <div></div>",
    );
    assert_eq!(paints(&ops), ["255,0,0", "255,0,0"]);
    let clips = even_odd_clip_bounds(&ops);
    assert_eq!(clips.len(), 2);
    // The first page's fragment is broken at its bottom, the second's at
    // its top: neither draws shadow past the break.
    assert!((clips[0][3] - 200.0).abs() < 0.01, "{:?}", clips[0]);
    assert!(clips[0][1] < -9.0, "{:?}", clips[0]);
    assert!(clips[1][1].abs() < 0.01, "{:?}", clips[1]);
    assert!(clips[1][3] > 109.0, "{:?}", clips[1]);
}

#[test]
fn an_empty_box_casts_its_spread_shadow() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='width:0;height:0;box-shadow:0 0 0 10px rgb(255,0,0), inset 0 0 0 4px blue'></div>"
    ));
    assert_eq!(paints(&ops), ["255,0,0"]);
    let [x0, y0, x1, y1] = even_odd_clip_bounds(&ops)[0];
    assert!(
        x0 < 20.0 && y0 < 20.0 && x1 > 40.0 && y1 > 40.0,
        "{x0} {y0} {x1} {y1}"
    );
}

#[test]
fn collapsed_table_cells_have_no_outer_shadow() {
    let table = |collapse: &str| {
        let (_, ops) = operations(&format!(
            "{BOX_CSS}<table style='border-collapse:{collapse}'><tr>\
             <td style='box-shadow:3px 3px rgb(255,0,0), inset 2px 2px rgb(0,0,255)'>a</td>\
             </tr></table>"
        ));
        paints(&ops)
    };
    assert_eq!(table("collapse"), ["0,0,255"]);
    assert_eq!(table("separate"), ["255,0,0", "0,0,255"]);
}

#[test]
fn shadows_shrunk_away_or_without_a_padding_box_draw_nothing() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='box-shadow:0 0 0 -60px red, 0 0 4px -60px red'></div>\
         <div style='width:0;height:0;box-shadow:inset 0 0 0 4px red'></div>"
    ));
    assert!(paints(&ops).is_empty(), "{:?}", paints(&ops));
}

#[test]
fn the_table_decides_the_border_model_of_its_cells() {
    let cell = |table: &str, cell: &str| {
        let (_, ops) = operations(&format!(
            "{BOX_CSS}<table style='border-collapse:{table}'><tr>\
             <td style='border-collapse:{cell};box-shadow:3px 3px rgb(255,0,0)'>a</td>\
             </tr></table>"
        ));
        paints(&ops)
    };
    assert!(cell("collapse", "separate").is_empty());
    assert_eq!(cell("separate", "collapse"), ["255,0,0"]);
}

#[test]
fn collapsed_tables_have_square_shadows() {
    let table = |collapse: &str| {
        let (_, ops) = operations(&format!(
            "{BOX_CSS}<table style='border-collapse:{collapse};border-radius:20px;\
             box-shadow:0 0 0 5px rgb(255,0,0)'><tr><td>a</td></tr></table>"
        ));
        count(&ops, "c")
    };
    assert_eq!(table("collapse"), 0);
    assert!(table("separate") > 0);
}

#[test]
fn a_cell_without_a_table_element_keeps_its_shadow() {
    let (_, ops) = operations(&format!(
        "{BOX_CSS}<div style='display:table-cell;border-collapse:collapse;\
         box-shadow:3px 3px rgb(255,0,0)'>a</div>"
    ));
    assert_eq!(paints(&ops), ["255,0,0"]);
}
