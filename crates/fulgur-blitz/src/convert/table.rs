use super::*;

/// Dispatcher entry for `<table>` elements. Returns `true` when an entry
/// was inserted into `out.tables` (and any cell descendants registered).
pub(super) fn try_convert(
    doc: &BaseDocument,
    node_id: usize,
    ctx: &mut super::ConvertContext<'_>,
    depth: usize,
    out: &mut crate::drawables::Drawables,
) -> bool {
    let Some(node) = doc.get_node(node_id) else {
        return false;
    };
    let Some(elem_data) = node.element_data() else {
        return false;
    };
    if elem_data.name.local.as_ref() != "table" {
        return false;
    }
    convert_table(doc, node, ctx, depth, out);
    true
}

fn convert_table(
    doc: &BaseDocument,
    node: &Node,
    ctx: &mut ConvertContext<'_>,
    depth: usize,
    out: &mut crate::drawables::Drawables,
) {
    let (width, height) = size_in_pt(node.final_layout.size);
    let style = extract_block_style(node, ctx.assets);
    let clipping = style.has_overflow_clip();
    let (opacity, visible) = extract_opacity_visible(node);

    out.tables.insert(
        node.id,
        crate::drawables::TableEntry {
            style,
            opacity,
            visible,
            id: extract_block_id(node),
            layout_size: Some(Size { width, height }),
            width,
            cached_height: height,
            clip_descendants: Vec::new(),
        },
    );

    let mark = clipping.then(|| out.draw_mark());

    // Walk table children to recurse cells.
    let mut body_cells = Vec::new();
    for &child_id in &node.children {
        let Some(child_node) = doc.get_node(child_id) else {
            continue;
        };
        let is_thead = is_table_section(child_node, "thead");
        collect_table_cells(doc, child_id, is_thead, ctx, depth, out, &mut body_cells);
    }

    let break_border = ctx
        .column_styles
        .get(&node.id)
        .and_then(|props| props.break_border);
    if break_border == Some(crate::column_css::BreakBorder::None) {
        mark_open_bottom_cells(&body_cells, &ctx.pagination_geometry, out);
    }

    if let Some(mark) = mark {
        let descendants: Vec<usize> = out
            .drawn_since(mark)
            .into_iter()
            .filter(|&id| id != node.id)
            .collect();
        if let Some(entry) = out.tables.get_mut(&node.id) {
            entry.clip_descendants = descendants;
        }
    }
}

/// Records in `out.open_bottom_cells` the cells of `cells` (a table's body
/// and footer cells) that end a fragment of the table on a page before its
/// last: on each such page, the cells of the last row that starts there,
/// told by the top of their fragment, the lowest on the page. A cell's own
/// bottom does not tell its row: an explicit `height` shorter than the row
/// ends it higher, and a cell that cannot split runs past the page foot. A
/// repeated header is no body cell, so it keeps its border on every page.
fn mark_open_bottom_cells(
    cells: &[usize],
    geometry: &crate::pagination_layout::PaginationGeometryTable,
    out: &mut crate::drawables::Drawables,
) {
    // Cells of one row start within rounding of each other.
    const SAME_TOP_PX: f32 = 0.5;
    let placed: Vec<(usize, u32, f32)> = cells
        .iter()
        .filter_map(|&id| Some((id, geometry.get(&id)?)))
        .flat_map(|(id, geom)| {
            geom.fragments
                .iter()
                .filter(|frag| frag.height > Px::ZERO)
                .map(move |frag| (id, frag.page_index, frag.y.to_f32()))
        })
        .collect();
    let Some(last_page) = placed.iter().map(|&(_, page, _)| page).max() else {
        return;
    };
    let mut last_row_top = std::collections::BTreeMap::<u32, f32>::new();
    for &(_, page, top) in &placed {
        let row_top = last_row_top.entry(page).or_insert(top);
        *row_top = row_top.max(top);
    }
    for &(id, page, top) in &placed {
        if page < last_page
            && last_row_top
                .get(&page)
                .is_some_and(|&row_top| top >= row_top - SAME_TOP_PX)
        {
            out.open_bottom_cells.insert((id, page));
        }
    }
}

fn is_table_section(node: &Node, section_name: &str) -> bool {
    if let Some(elem) = node.element_data() {
        elem.name.local.as_ref() == section_name
    } else {
        false
    }
}

fn collect_table_cells(
    doc: &BaseDocument,
    node_id: usize,
    is_header: bool,
    ctx: &mut ConvertContext<'_>,
    depth: usize,
    out: &mut crate::drawables::Drawables,
    body_cells: &mut Vec<usize>,
) {
    if depth >= MAX_DOM_DEPTH {
        return;
    }
    let Some(node) = doc.get_node(node_id) else {
        return;
    };

    let layout_children_guard = node.layout_children.borrow();
    let effective_children = layout_children_guard.as_deref().unwrap_or(&node.children);
    for &child_id in effective_children {
        let Some(child_node) = doc.get_node(child_id) else {
            continue;
        };
        if matches!(&child_node.data, NodeData::Comment) {
            continue;
        }
        if is_non_visual_element(child_node) {
            continue;
        }

        let cw = child_node.final_layout.size.width.as_px().in_pt();
        let ch = child_node.final_layout.size.height.as_px().in_pt();

        let child_effective_is_empty = child_node
            .layout_children
            .borrow()
            .as_deref()
            .unwrap_or(&child_node.children)
            .is_empty();
        if ch == Pt::ZERO && cw == Pt::ZERO && !child_effective_is_empty {
            let child_is_header = is_header || is_table_section(child_node, "thead");
            collect_table_cells(
                doc,
                child_id,
                child_is_header,
                ctx,
                depth + 1,
                out,
                body_cells,
            );
            continue;
        }

        if ch == Pt::ZERO && cw == Pt::ZERO {
            continue;
        }

        // Actual cell — recurse via convert_node so its block / paragraph
        // entries land in the standard maps.
        convert_node(doc, child_id, ctx, depth + 1, out);
        if !is_header {
            body_cells.push(child_id);
        }
    }
}

#[cfg(test)]
mod tests {
    // --- overflow: hidden on <table> — clip_descendants path ---
    //
    // A table with `overflow: hidden` (or any non-Visible overflow) triggers
    // `has_overflow_clip()` → `clipping = true` → `draw_mark()` fires →
    // the `if let Some(mark) = mark { ... entry.clip_descendants = ... }` block
    // executes (lines 62–70 of table.rs). Without this test that entire block
    // is dead from the test runner's perspective.
    #[test]
    fn table_overflow_hidden_produces_valid_pdf() {
        let html = "<!DOCTYPE html><html><body>\
            <table style=\"overflow: hidden; width: 200px;\">\
                <tr><td style=\"width: 100px; height: 20px;\">Cell</td></tr>\
            </table>\
            </body></html>";
        let pdf = crate::engine::Engine::builder()
            .build()
            .render(html)
            .expect("render");
        assert!(pdf.starts_with(b"%PDF"));
    }

    // Targeted assertion: `clip_descendants` must be non-empty for a table
    // with `overflow: hidden` that contains real cell content.  This verifies
    // the assignment `entry.clip_descendants = descendants` (table.rs:70)
    // rather than just confirming the engine doesn't panic.
    #[test]
    fn table_overflow_hidden_populates_clip_descendants() {
        use crate::units::F32Units;
        use std::ops::DerefMut;

        let html = "<!DOCTYPE html><html><body>\
            <table style=\"overflow: hidden; width: 200px;\">\
                <tr><td style=\"width: 100px; height: 20px;\">A</td></tr>\
            </table>\
            </body></html>";

        let mut doc = crate::blitz_adapter::parse_and_layout(
            html,
            595.0_f32.as_px(),
            842.0_f32.as_px(),
            &[],
            true,
        );
        let column_styles = crate::blitz_adapter::extract_column_style_table(&doc, &[]);
        let multicol_geometry = crate::multicol_layout::run_pass(doc.deref_mut(), &column_styles);
        let pagination_geometry = crate::pagination_layout::run_pass(doc.deref_mut(), 842.0);
        let running_store = crate::gcpm::running::RunningElementStore::new();
        let mut ctx = crate::convert::ConvertContext {
            running_store: &running_store,
            assets: None,
            font_cache: Default::default(),
            string_set_by_node: Default::default(),
            counter_ops_by_node: Default::default(),
            bookmark_by_node: Default::default(),
            column_styles,
            multicol_geometry,
            pagination_geometry,
            link_cache: Default::default(),
            viewport_size_px: Some((595.0, 842.0)),
        };
        let d = crate::convert::dom_to_drawables(&doc, &mut ctx);

        assert!(!d.tables.is_empty(), "table entry must exist");
        let has_clipped = d.tables.values().any(|t| !t.clip_descendants.is_empty());
        assert!(
            has_clipped,
            "a table with overflow: hidden must have non-empty clip_descendants"
        );
    }

    // --- comment node inside <tbody> ---
    //
    // HTML5 keeps comment nodes inside their table-section parent rather than
    // foster-parenting them. `collect_table_cells` must skip them via the
    // `NodeData::Comment` guard (table.rs:104) rather than trying to lay them
    // out, so the real cell content still reaches the drawable maps.
    #[test]
    fn table_comment_in_tbody_produces_valid_pdf() {
        let html = "<!DOCTYPE html><html><body>\
            <table>\
                <tbody>\
                    <!-- comment node exercises NodeData::Comment guard -->\
                    <tr><td>Cell</td></tr>\
                </tbody>\
            </table>\
            </body></html>";
        let pdf = crate::engine::Engine::builder()
            .build()
            .render(html)
            .expect("render");
        assert!(pdf.starts_with(b"%PDF"));
    }

    // Regression: a table entry must be created even when the only content
    // before the first real row is an HTML comment.
    #[test]
    fn table_comment_before_rows_still_registers_table_entry() {
        use crate::units::F32Units;
        use std::ops::DerefMut;

        let html = "<!DOCTYPE html><html><body>\
            <table>\
                <tbody>\
                    <!-- comment before rows -->\
                    <tr><td style=\"width:50px;height:20px\">A</td></tr>\
                </tbody>\
            </table>\
            </body></html>";

        let mut doc = crate::blitz_adapter::parse_and_layout(
            html,
            595.0_f32.as_px(),
            842.0_f32.as_px(),
            &[],
            true,
        );
        let column_styles = crate::blitz_adapter::extract_column_style_table(&doc, &[]);
        let multicol_geometry = crate::multicol_layout::run_pass(doc.deref_mut(), &column_styles);
        let pagination_geometry = crate::pagination_layout::run_pass(doc.deref_mut(), 842.0);
        let running_store = crate::gcpm::running::RunningElementStore::new();
        let mut ctx = crate::convert::ConvertContext {
            running_store: &running_store,
            assets: None,
            font_cache: Default::default(),
            string_set_by_node: Default::default(),
            counter_ops_by_node: Default::default(),
            bookmark_by_node: Default::default(),
            column_styles,
            multicol_geometry,
            pagination_geometry,
            link_cache: Default::default(),
            viewport_size_px: Some((595.0, 842.0)),
        };
        let d = crate::convert::dom_to_drawables(&doc, &mut ctx);

        assert!(
            !d.tables.is_empty(),
            "table entry must be present in drawables even when tbody has a leading comment"
        );
        // Stronger assertion: the cell content must also have been converted.
        // `collect_table_cells` inserts `<td>` content via `convert_node`, which
        // produces paragraph entries for inline text. A non-empty `paragraphs`
        // map proves `collect_table_cells` walked past the comment and processed
        // the real `<tr><td>` — something `!d.tables.is_empty()` alone cannot
        // verify, since `TableEntry` is inserted before the child walk.
        assert!(
            !d.paragraphs.is_empty(),
            "cell content must be converted to paragraph drawables, \
             proving collect_table_cells walked past the comment node"
        );
    }

    // A table with `-fulgur-break-border: none` leaves the bottom border of
    // the cells that end each fragment but the last out: two cells per row,
    // two rows on page 0, one on page 1; the cells of page 0's second row
    // are open at the bottom, and none of page 1's. That row's cells end
    // unevenly, one 10 px tall (an explicit height shorter than the row),
    // the other 70 px (a cell that runs past the page foot): the row, not
    // the bottom, picks them.
    #[test]
    fn the_cells_of_the_last_row_of_each_page_but_the_last_are_open_at_the_bottom() {
        use crate::pagination_layout::{Fragment, PaginationGeometry, PaginationGeometryTable};
        use crate::units::F32Units;

        let frag = |page_index: u32, y: f32, height: f32| Fragment {
            page_index,
            x: 0.0_f32.as_px(),
            y: y.as_px(),
            width: 50.0_f32.as_px(),
            height: height.as_px(),
        };
        let mut geometry = PaginationGeometryTable::new();
        for (id, fragment) in [
            (1, frag(0, 0.0, 30.0)),
            (2, frag(0, 0.0, 30.0)),
            (3, frag(0, 30.0, 10.0)),
            (4, frag(0, 30.0, 70.0)),
            (5, frag(1, 0.0, 30.0)),
            (6, frag(1, 0.0, 30.0)),
        ] {
            geometry.insert(
                id,
                PaginationGeometry {
                    fragments: vec![fragment],
                    ..Default::default()
                },
            );
        }
        let mut out = crate::drawables::Drawables::new();

        super::mark_open_bottom_cells(&[1, 2, 3, 4, 5, 6], &geometry, &mut out);

        let open: Vec<(usize, u32)> = out.open_bottom_cells.into_iter().collect();
        assert_eq!(open, [(3, 0), (4, 0)]);
    }
}
