//! Invariants of the fragmenter over generated box trees.
//!
//! A proptest strategy generates trees of up to 30 boxes: blocks with
//! margins, padding, and borders; paragraphs of 1 to 12 lines; tables of 1
//! to 8 rows of 1 to 3 cells, with or without a `thead`; flex rows of two
//! items; images; `break-inside: avoid` and forced breaks on random boxes.
//! Pages are 100 to 400 px tall. Each case is laid out with its break
//! styles and checked against five invariants ([`Invariant`]).
//!
//! The cases are drawn from fixed seeds, one per case, so a failure names
//! its seed, and are laid out in Noto Sans with no system font, so they are
//! the same on every host. [`KNOWN_FAILURES`] lists the seeds that fail
//! today, each with the invariant it breaks and the step of the fragmenter
//! refactor expected to fix it; the test fails when a seed outside the list
//! breaks an invariant, and when a listed seed no longer breaks it, so a fix
//! removes its entries. `FRAGMENT_INVARIANT_LIST=1` prints the table the
//! seeds give today, `FRAGMENT_INVARIANT_CASES=<n>` checks more seeds than
//! the default, and `FRAGMENT_INVARIANT_SEED=<seed>` prints one case's
//! markup, its violations, and for each the smallest case that still breaks
//! it, with that case's geometry.

use super::*;
use crate::blitz_adapter;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::ops::DerefMut;

/// The number of seeds checked by default.
const CASES: u64 = 256;

/// The page width; the page height is generated.
const PAGE_WIDTH: f32 = 200.0;

/// How far a placement may stray before it counts, in px.
const EPS: f32 = 0.5;

/// The fragmenter's invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Invariant {
    /// No fragment ends past the page foot, except one of a monolithic box
    /// taller than a page.
    PastFoot,
    /// Every in-flow leaf, and every line of a paragraph, is placed exactly
    /// once.
    PlacedOnce,
    /// In block flow, document order matches (page, y) order.
    DocumentOrder,
    /// On each page, a parent's fragment encloses its children's.
    Encloses,
    /// No repeated header sits on a page without a body row.
    HeaderAlone,
}

/// The seeds that break an invariant today, by the invariant and the step
/// of the fragmenter refactor expected to fix them. The step is the one the
/// checker names for the violation (`Placed::past_cause`): `R1`, a nested
/// box's split fragments (one page cursor and one rule for a parent's
/// fragment); `R2`, the body walker's fragments of a split box and a nested
/// childless box taller than a page (one walker at every depth); `R3`, a
/// flex row crossing the foot and a header left alone above a first row
/// that places nothing (`classify` and `leading_extent`); `R5`, a table's
/// boxes crossing the foot (a table walker over rows); `R6`, a box's edge
/// (margin, border, padding) crossing the foot at a break; `none`, a
/// paragraph whose lines measure taller than its layout box, which no step
/// changes.
const KNOWN_FAILURES: &[(Invariant, &str, &[u64])] = &[
    (
        Invariant::PastFoot,
        "R1",
        &[
            22, 33, 40, 44, 46, 68, 78, 83, 95, 124, 148, 149, 199, 206, 211, 251, 253,
        ],
    ),
    (
        Invariant::PastFoot,
        "R2",
        &[
            0, 1, 3, 7, 8, 9, 10, 13, 16, 17, 22, 24, 25, 28, 30, 31, 33, 34, 38, 39, 40, 41, 43,
            44, 46, 47, 51, 54, 55, 56, 57, 58, 60, 62, 64, 65, 68, 70, 73, 75, 76, 78, 79, 83, 85,
            86, 89, 91, 95, 96, 99, 101, 102, 106, 107, 108, 110, 114, 116, 118, 124, 128, 129,
            134, 135, 136, 140, 143, 145, 148, 149, 151, 152, 154, 155, 156, 157, 158, 160, 161,
            162, 166, 168, 171, 172, 173, 180, 182, 183, 184, 186, 187, 192, 194, 195, 197, 198,
            199, 200, 201, 202, 204, 205, 206, 211, 212, 216, 218, 225, 226, 231, 232, 233, 238,
            239, 241, 244, 248, 250, 251, 253,
        ],
    ),
    (
        Invariant::PastFoot,
        "R3",
        &[
            1, 7, 13, 16, 17, 25, 33, 38, 40, 42, 43, 44, 46, 68, 76, 78, 83, 95, 101, 124, 135,
            140, 148, 149, 156, 171, 184, 192, 197, 199, 202, 206, 226, 232, 238, 239, 250, 251,
            253,
        ],
    ),
    (
        Invariant::PastFoot,
        "R5",
        &[
            3, 8, 10, 34, 43, 51, 57, 60, 78, 79, 89, 96, 107, 110, 118, 128, 152, 162, 166, 168,
            171, 180, 182, 184, 186, 192, 195, 200, 202,
        ],
    ),
    (Invariant::PastFoot, "R6", &[40, 212, 233]),
    (
        Invariant::DocumentOrder,
        "none",
        &[22, 46, 65, 68, 121, 173, 218, 227, 250],
    ),
    (Invariant::Encloses, "R2", &[3, 51, 95, 106, 202]),
    (Invariant::Encloses, "R3", &[140, 202]),
    (
        Invariant::Encloses,
        "R5",
        &[3, 65, 78, 107, 119, 128, 180, 192],
    ),
    (Invariant::HeaderAlone, "R3", &[10, 110, 166, 192]),
];

/// The style of a generated box.
#[derive(Clone, Debug, Default)]
struct BoxStyle {
    margin_top: u8,
    margin_bottom: u8,
    padding: u8,
    border: u8,
    avoid: bool,
    break_before: bool,
}

/// A generated box.
#[derive(Clone, Debug)]
enum Gen {
    /// A block of a fixed height with no children.
    Leaf { style: BoxStyle, height: u16 },
    /// A replaced box (`img`) of a fixed height.
    Image { height: u16 },
    /// A paragraph of `lines` lines of `line_height` px.
    Para {
        style: BoxStyle,
        lines: u8,
        line_height: u8,
        orphans: u8,
        widows: u8,
    },
    /// A block holding other boxes.
    Block { style: BoxStyle, children: Vec<Gen> },
    /// A table; each row's cells hold one box each.
    Table {
        style: BoxStyle,
        thead: bool,
        cell_padding: u8,
        rows: Vec<(bool, Vec<Gen>)>,
    },
    /// A flex row of two items, each holding one box.
    Flex { style: BoxStyle, items: Vec<Gen> },
}

fn style() -> impl Strategy<Value = BoxStyle> {
    (
        prop_oneof![4 => Just(0u8), 1 => 1u8..=20],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=20],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=12],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=4],
        proptest::bool::weighted(0.2),
        proptest::bool::weighted(0.08),
    )
        .prop_map(
            |(margin_top, margin_bottom, padding, border, avoid, break_before)| BoxStyle {
                margin_top,
                margin_bottom,
                padding,
                border,
                avoid,
                break_before,
            },
        )
}

/// A box with no box children.
fn leaf() -> impl Strategy<Value = Gen> {
    prop_oneof![
        4 => (style(), 5u16..=160).prop_map(|(style, height)| Gen::Leaf { style, height }),
        1 => (5u16..=120).prop_map(|height| Gen::Image { height }),
        4 => (style(), 1u8..=12, prop_oneof![Just(20u8), 16u8..=30], 1u8..=4, 1u8..=4).prop_map(
            |(style, lines, line_height, orphans, widows)| Gen::Para {
                style,
                lines,
                line_height,
                orphans,
                widows,
            }
        ),
    ]
}

/// The boxes of `nodes`, rows and cells included.
fn box_count(nodes: &[Gen]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Gen::Leaf { .. } | Gen::Image { .. } | Gen::Para { .. } => 1,
            Gen::Block { children, .. } => 1 + box_count(children),
            Gen::Table { thead, rows, .. } => {
                let header = if *thead { 1 + rows[0].1.len() } else { 0 };
                1 + header
                    + rows
                        .iter()
                        .map(|(_, cells)| 1 + cells.len() + box_count(cells))
                        .sum::<usize>()
            }
            Gen::Flex { items, .. } => 1 + items.len() + box_count(items),
        })
        .sum()
}

/// A tree of up to 30 boxes.
fn tree() -> impl Strategy<Value = Vec<Gen>> {
    let node = leaf().prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            3 => (style(), prop::collection::vec(inner.clone(), 1..=4))
                .prop_map(|(style, children)| Gen::Block { style, children }),
            2 => (
                style(),
                any::<bool>(),
                prop_oneof![Just(0u8), Just(10u8)],
                prop::collection::vec(
                    (proptest::bool::weighted(0.2), prop::collection::vec(leaf(), 1..=3)),
                    1..=8
                ),
            )
                .prop_map(|(style, thead, cell_padding, rows)| {
                    // Every row of a table has as many cells as its first.
                    let columns = rows[0].1.len();
                    let rows = rows
                        .into_iter()
                        .map(|(avoid, mut cells)| {
                            cells.resize(columns, Gen::Image { height: 20 });
                            (avoid, cells)
                        })
                        .collect();
                    Gen::Table {
                        style,
                        thead,
                        cell_padding,
                        rows,
                    }
                }),
            1 => (style(), prop::collection::vec(inner, 2..=2))
                .prop_map(|(style, items)| Gen::Flex { style, items }),
        ]
    });
    prop::collection::vec(node, 1..=5)
        .prop_filter("at most 30 boxes", |roots| box_count(roots) <= 30)
}

/// What the checker knows of each generated box.
#[derive(Clone, Debug)]
struct Info {
    id: String,
    kind: Kind,
    parent: Option<usize>,
    children: Vec<usize>,
    /// The box is a header cell or inside one: its fragments repeat.
    in_header: bool,
    /// The paragraph's line count.
    lines: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Leaf,
    Image,
    Para,
    Block,
    Table,
    Row,
    Cell,
    BodyCell,
    Flex,
    Item,
}

/// A generated case as HTML, with the boxes the checker reads.
struct Case {
    html: String,
    boxes: Vec<Info>,
    /// Each table's index, with its header cells and body cells.
    tables: Vec<(usize, Vec<usize>, Vec<usize>)>,
}

fn style_css(style: &BoxStyle) -> String {
    let mut css = String::new();
    if style.margin_top > 0 || style.margin_bottom > 0 {
        let _ = write!(
            css,
            "margin: {}px 0 {}px; ",
            style.margin_top, style.margin_bottom
        );
    }
    if style.padding > 0 {
        let _ = write!(css, "padding: {}px; ", style.padding);
    }
    if style.border > 0 {
        let _ = write!(css, "border: {}px solid; ", style.border);
    }
    if style.avoid {
        css.push_str("break-inside: avoid; ");
    }
    if style.break_before {
        css.push_str("break-before: page; ");
    }
    css
}

impl Case {
    fn new(roots: &[Gen]) -> Self {
        let mut case = Case {
            html: String::new(),
            boxes: Vec::new(),
            tables: Vec::new(),
        };
        let mut body = String::new();
        for root in roots {
            case.emit(root, None, false, &mut body);
        }
        case.html = format!(
            "<!doctype html><html><head><style>\
             html, body {{ margin: 0; padding: 0; font-family: 'Noto Sans'; }} \
             p {{ margin: 0; }} \
             table {{ border-spacing: 0; margin: 0; }} \
             th {{ padding: 0; height: 20px; box-sizing: border-box; }} \
             </style></head><body>{body}</body></html>"
        );
        case
    }

    fn add(&mut self, kind: Kind, parent: Option<usize>, in_header: bool, lines: usize) -> usize {
        let index = self.boxes.len();
        self.boxes.push(Info {
            id: format!("b{index}"),
            kind,
            parent,
            children: Vec::new(),
            in_header,
            lines,
        });
        if let Some(parent) = parent {
            self.boxes[parent].children.push(index);
        }
        index
    }

    fn emit(&mut self, node: &Gen, parent: Option<usize>, in_header: bool, out: &mut String) {
        match node {
            Gen::Leaf { style, height } => {
                let b = self.add(Kind::Leaf, parent, in_header, 0);
                let _ = write!(
                    out,
                    r#"<div id="b{b}" style="{}height: {height}px"></div>"#,
                    style_css(style)
                );
            }
            Gen::Image { height } => {
                let b = self.add(Kind::Image, parent, in_header, 0);
                let _ = write!(
                    out,
                    r#"<img id="b{b}" style="display: block; width: 10px; height: {height}px">"#
                );
            }
            Gen::Para {
                style,
                lines,
                line_height,
                orphans,
                widows,
            } => {
                let b = self.add(Kind::Para, parent, in_header, usize::from(*lines));
                let text = (1..=*lines)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join("<br>");
                let _ = write!(
                    out,
                    r#"<p id="b{b}" style="{}line-height: {line_height}px; orphans: {orphans}; widows: {widows}">{text}</p>"#,
                    style_css(style)
                );
            }
            Gen::Block { style, children } => {
                let b = self.add(Kind::Block, parent, in_header, 0);
                let _ = write!(out, r#"<div id="b{b}" style="{}">"#, style_css(style));
                for child in children {
                    self.emit(child, Some(b), in_header, out);
                }
                out.push_str("</div>");
            }
            Gen::Table {
                style,
                thead,
                cell_padding,
                rows,
            } => {
                let t = self.add(Kind::Table, parent, in_header, 0);
                let _ = write!(out, r#"<table id="b{t}" style="{}">"#, style_css(style));
                let mut header = Vec::new();
                let mut body = Vec::new();
                if *thead {
                    let r = self.add(Kind::Row, Some(t), true, 0);
                    let _ = write!(out, r#"<thead><tr id="b{r}">"#);
                    for _ in &rows[0].1 {
                        let c = self.add(Kind::Cell, Some(r), true, 0);
                        header.push(c);
                        let _ = write!(out, r#"<th id="b{c}">H</th>"#);
                    }
                    out.push_str("</tr></thead>");
                }
                out.push_str("<tbody>");
                for (avoid, cells) in rows {
                    let r = self.add(Kind::Row, Some(t), in_header, 0);
                    let row_style = if *avoid { "break-inside: avoid" } else { "" };
                    let _ = write!(out, r#"<tr id="b{r}" style="{row_style}">"#);
                    for cell in cells {
                        let c = self.add(Kind::BodyCell, Some(r), in_header, 0);
                        body.push(c);
                        let _ =
                            write!(out, r#"<td id="b{c}" style="padding: {cell_padding}px 0">"#);
                        self.emit(cell, Some(c), in_header, out);
                        out.push_str("</td>");
                    }
                    out.push_str("</tr>");
                }
                out.push_str("</tbody></table>");
                if *thead {
                    self.tables.push((t, header, body));
                }
            }
            Gen::Flex { style, items } => {
                let f = self.add(Kind::Flex, parent, in_header, 0);
                let _ = write!(
                    out,
                    r#"<div id="b{f}" style="display: flex; {}">"#,
                    style_css(style)
                );
                for item in items {
                    let i = self.add(Kind::Item, Some(f), in_header, 0);
                    let _ = write!(out, r#"<div id="b{i}" style="flex: 1 1 0">"#);
                    self.emit(item, Some(i), in_header, out);
                    out.push_str("</div>");
                }
                out.push_str("</div>");
            }
        }
    }
}

/// The node id of each generated box, by its `id`.
fn node_ids(doc: &BaseDocument, case: &Case) -> Vec<Option<usize>> {
    let mut by_id = BTreeMap::new();
    let mut stack = vec![doc.root_element().id];
    while let Some(id) = stack.pop() {
        let Some(node) = doc.get_node(id) else {
            continue;
        };
        if let Some(value) = node
            .element_data()
            .and_then(|data| data.attr(blitz_dom::LocalName::from("id")))
        {
            by_id.insert(value.to_string(), id);
        }
        stack.extend(node.children.iter().copied());
    }
    case.boxes
        .iter()
        .map(|info| by_id.get(&info.id).copied())
        .collect()
}

/// One violation: the invariant, the refactor step expected to fix it, and
/// what broke it.
type Violation = (Invariant, &'static str, String);

/// `case` laid out with its break styles on pages `page_height` px tall,
/// in Noto Sans and no system font, so the lines are the same on every host.
fn layout(case: &Case, page_height: f32) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
    let fonts = [crate::test_fonts::noto_sans_regular_ttf()];
    let (mut doc, _, _) = blitz_adapter::parse_html_with_local_resources(
        &case.html, PAGE_WIDTH, 10000, &fonts, false, None,
    );
    blitz_adapter::resolve(&mut doc);
    let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
    let geometry = super::run_pass_with_break_styles(doc.deref_mut(), page_height.as_px(), &styles);
    (doc, geometry)
}

/// The boxes whose fragments `b`'s enclose: a table's are its rows'
/// cells, as a row has no box of its own in the geometry.
fn enclosed(case: &Case, b: usize) -> Vec<usize> {
    let info = &case.boxes[b];
    match info.kind {
        Kind::Row => Vec::new(),
        Kind::Table => info
            .children
            .iter()
            .flat_map(|&row| case.boxes[row].children.iter().copied())
            .collect(),
        _ => info.children.clone(),
    }
}

/// Whether `b`, or a box around it, is of one of `kinds`.
fn within(case: &Case, b: usize, kinds: &[Kind]) -> bool {
    let mut at = Some(b);
    while let Some(k) = at {
        if kinds.contains(&case.boxes[k].kind) {
            return true;
        }
        at = case.boxes[k].parent;
    }
    false
}

/// The geometry of one laid-out case, as the checks read it.
struct Placed<'a> {
    case: &'a Case,
    page_height: f32,
    fragments: Vec<Vec<Fragment>>,
    line_boundaries: Vec<Vec<usize>>,
    layout_height: Vec<f32>,
    /// The pages on which each box's fragment ends past the page foot.
    past: Vec<BTreeSet<u32>>,
}

impl Placed<'_> {
    /// The step expected to fix `b`'s fragments past the page foot, and why.
    fn past_cause(&self, b: usize) -> (&'static str, &'static str) {
        let info = &self.case.boxes[b];
        let frags = &self.fragments[b];
        if frags.len() > 1 && info.parent.is_none() {
            return (
                "R2",
                "the body walker's fragment of a split box ends past the foot",
            );
        }
        if frags.len() > 1
            && self.past[b].contains(&frags[0].page_index)
            && (frags[0].height.to_f32() - self.layout_height[b]).abs() < EPS
        {
            return ("R1", "a split box's first fragment is its whole height");
        }
        if info.kind == Kind::Leaf
            && frags.len() == 1
            && self.layout_height[b] > self.page_height + EPS
        {
            return (
                "R2",
                "a nested childless box taller than a page is not sliced",
            );
        }
        for child in enclosed(self.case, b) {
            if !self.past[child].is_disjoint(&self.past[b]) {
                return self.past_cause(child);
            }
        }
        if within(self.case, b, &[Kind::Table]) {
            ("R5", "a table box crosses the page foot")
        } else if within(self.case, b, &[Kind::Flex, Kind::Item]) {
            ("R3", "a flex box crosses the page foot")
        } else {
            ("R6", "a box's edge crosses the page foot")
        }
    }
}

/// Lay `case` out on pages `page_height` px tall and check each invariant.
fn check(case: &Case, page_height: f32) -> Vec<Violation> {
    let (doc, geometry) = layout(case, page_height);
    let ids = node_ids(&doc, case);
    let entry = |b: usize| ids[b].and_then(|id| geometry.get(&id));
    let fragments: Vec<Vec<Fragment>> = (0..case.boxes.len())
        .map(|b| entry(b).map_or_else(Vec::new, |g| g.fragments.clone()))
        .collect();
    let line_boundaries: Vec<Vec<usize>> = (0..case.boxes.len())
        .map(|b| entry(b).map_or_else(Vec::new, |g| g.line_boundaries.clone()))
        .collect();
    let layout_height: Vec<f32> = ids
        .iter()
        .map(|id| {
            id.and_then(|id| doc.get_node(id))
                .map_or(0.0, |node| node.final_layout.size.height)
        })
        .collect();
    let past: Vec<BTreeSet<u32>> = (0..case.boxes.len())
        .map(|b| {
            // A monolithic box taller than a page may end past the foot.
            if case.boxes[b].kind == Kind::Image && layout_height[b] > page_height + EPS {
                return BTreeSet::new();
            }
            fragments[b]
                .iter()
                .filter(|f| f.y.to_f32() + f.height.to_f32() > page_height + EPS)
                .map(|f| f.page_index)
                .collect()
        })
        .collect();
    let placed = Placed {
        case,
        page_height,
        fragments,
        line_boundaries,
        layout_height,
        past,
    };
    let mut out = Vec::new();

    for (b, info) in case.boxes.iter().enumerate() {
        let frags = &placed.fragments[b];

        // No fragment ends past the page foot, except a monolithic box
        // taller than a page.
        for page in &placed.past[b] {
            let (step, why) = placed.past_cause(b);
            out.push((
                Invariant::PastFoot,
                step,
                format!(
                    "{} ({:?}) ends past the foot of page {page}: {why}",
                    info.id, info.kind
                ),
            ));
        }

        if info.in_header {
            continue;
        }

        // Every in-flow leaf, and every line of a paragraph, is placed
        // exactly once.
        match info.kind {
            Kind::Leaf | Kind::Image => {
                let pages: Vec<u32> = frags.iter().map(|f| f.page_index).collect();
                let total: f32 = frags.iter().map(|f| f.height.to_f32()).sum();
                if frags.is_empty() {
                    out.push((
                        Invariant::PlacedOnce,
                        "?",
                        format!("{} is not placed", info.id),
                    ));
                } else if pages.windows(2).any(|w| w[0] >= w[1]) {
                    out.push((
                        Invariant::PlacedOnce,
                        "?",
                        format!("{} is placed twice on a page: {pages:?}", info.id),
                    ));
                } else if (total - placed.layout_height[b]).abs() > EPS * frags.len() as f32 {
                    out.push((
                        Invariant::PlacedOnce,
                        "?",
                        format!(
                            "{} places {total} px of {} px",
                            info.id, placed.layout_height[b]
                        ),
                    ));
                }
            }
            Kind::Para => {
                let boundaries = &placed.line_boundaries[b];
                if frags.is_empty() {
                    out.push((
                        Invariant::PlacedOnce,
                        "?",
                        format!("{} is not placed", info.id),
                    ));
                } else if boundaries.is_empty() {
                    if frags.len() > 1 {
                        out.push((
                            Invariant::PlacedOnce,
                            "?",
                            format!(
                                "{} has {} fragments and no line partition",
                                info.id,
                                frags.len()
                            ),
                        ));
                    }
                } else if boundaries.len() != frags.len() + 1
                    || boundaries[0] != 0
                    || boundaries[boundaries.len() - 1] != info.lines
                    || boundaries.windows(2).any(|w| w[0] >= w[1])
                {
                    out.push((
                        Invariant::PlacedOnce,
                        "?",
                        format!(
                            "{} partitions {} lines as {boundaries:?} over {} fragments",
                            info.id,
                            info.lines,
                            frags.len()
                        ),
                    ));
                }
            }
            _ => {}
        }

        // In block flow, document order matches (page, y) order.
        let flow: Vec<usize> = if b == 0 {
            (0..case.boxes.len())
                .filter(|&k| case.boxes[k].parent.is_none())
                .collect()
        } else if matches!(info.kind, Kind::Block | Kind::BodyCell | Kind::Item) {
            info.children.clone()
        } else {
            Vec::new()
        };
        for pair in flow.windows(2) {
            let (a, c) = (pair[0], pair[1]);
            let (Some(last), Some(first)) =
                (placed.fragments[a].last(), placed.fragments[c].first())
            else {
                continue;
            };
            let a_end = last.y.to_f32() + last.height.to_f32();
            if first.page_index < last.page_index
                || first.page_index == last.page_index && first.y.to_f32() + EPS < a_end
            {
                // A paragraph's fragments, measured from the shaper's line
                // metrics, can be taller than its layout box (Noto Sans
                // needs about 22 px a line at 16 px); the box after it
                // starts at the layout's end. No step of the refactor
                // changes the line model.
                let overlap = a_end - first.y.to_f32();
                let measured: f32 = placed.fragments[a].iter().map(|f| f.height.to_f32()).sum();
                let excess = measured - placed.layout_height[a];
                let (step, why) = if first.page_index == last.page_index
                    && case.boxes[a].kind == Kind::Para
                    && excess > EPS
                    && overlap <= excess + EPS
                {
                    ("none", "a paragraph's lines end below its layout box")
                } else {
                    ("?", "a box starts above the end of the box before it")
                };
                out.push((
                    Invariant::DocumentOrder,
                    step,
                    format!(
                        "{} starts at page {} y {} before {} ends at page {} y {a_end}: {why}",
                        case.boxes[c].id,
                        first.page_index,
                        first.y.to_f32(),
                        case.boxes[a].id,
                        last.page_index
                    ),
                ));
            }
        }

        // On each page, a parent's fragment encloses its children's.
        for child in enclosed(case, b) {
            for cf in &placed.fragments[child] {
                let inside = frags.iter().any(|pf| {
                    pf.page_index == cf.page_index
                        && pf.y.to_f32() <= cf.y.to_f32() + EPS
                        && cf.y.to_f32() + cf.height.to_f32()
                            <= pf.y.to_f32() + pf.height.to_f32() + EPS
                });
                if inside {
                    continue;
                }
                let (step, why) = if !placed.past[child].is_empty() {
                    placed.past_cause(child)
                } else if within(case, child, &[Kind::Table]) {
                    ("R5", "a table box is outside its parent")
                } else if within(case, child, &[Kind::Flex, Kind::Item]) {
                    ("R3", "a flex box is outside its parent")
                } else {
                    ("?", "a box is outside its parent")
                };
                out.push((
                    Invariant::Encloses,
                    step,
                    format!(
                        "{} ({:?}) on page {} y {} h {} is outside {} ({:?}): {why}",
                        case.boxes[child].id,
                        case.boxes[child].kind,
                        cf.page_index,
                        cf.y.to_f32(),
                        cf.height.to_f32(),
                        info.id,
                        info.kind
                    ),
                ));
            }
        }
    }

    // No repeated header sits on a page without a body row.
    for (table, header, body) in &case.tables {
        let pages = |cells: &[usize]| -> BTreeSet<u32> {
            cells
                .iter()
                .flat_map(|&c| placed.fragments[c].iter().map(|f| f.page_index))
                .collect()
        };
        for page in pages(header).difference(&pages(body)) {
            out.push((
                Invariant::HeaderAlone,
                "R3",
                format!(
                    "{}'s header is alone on page {page}: its first row places nothing there",
                    case.boxes[*table].id
                ),
            ));
        }
    }
    out
}

/// A case and its page height, as one strategy.
fn case_strategy() -> impl Strategy<Value = (Vec<Gen>, u16)> {
    (tree(), 100u16..=400)
}

/// The runner of `seed`: every draw from it is the same on every host.
fn runner(seed: u64) -> TestRunner {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &bytes),
    )
}

/// The case of `seed` and its page height.
fn case_of(seed: u64) -> (Vec<Gen>, u16) {
    case_strategy()
        .new_tree(&mut runner(seed))
        .expect("a case")
        .current()
}

/// What `seed`'s case breaks: each invariant with the step expected to fix
/// it.
fn broken(seed: u64) -> BTreeSet<(Invariant, &'static str)> {
    let (roots, page) = case_of(seed);
    check(&Case::new(&roots), f32::from(page))
        .into_iter()
        .map(|(invariant, step, _)| (invariant, step))
        .collect()
}

/// `seed`'s case shrunk to the smallest one that still breaks `invariant`
/// for `step`.
fn shrink(seed: u64, invariant: Invariant, step: &str) -> (Vec<Gen>, u16) {
    let mut tree = case_strategy().new_tree(&mut runner(seed)).expect("a case");
    let fails = |value: &(Vec<Gen>, u16)| {
        check(&Case::new(&value.0), f32::from(value.1))
            .iter()
            .any(|(broken, cause, _)| *broken == invariant && *cause == step)
    };
    let mut best = tree.current();
    let mut steps = 0;
    while steps < 2000 && tree.simplify() {
        steps += 1;
        let value = tree.current();
        if fails(&value) {
            best = value;
        } else if !tree.complicate() {
            break;
        }
    }
    best
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// One seed's case, its violations, and for each invariant and step the
/// smallest case that still breaks it, with that case's geometry.
fn describe(seed: u64) -> String {
    let (roots, page) = case_of(seed);
    let case = Case::new(&roots);
    let mut text = format!("seed {seed}, page {page} px\n{}\n", case.html);
    for (invariant, step, what) in check(&case, f32::from(page)) {
        let _ = writeln!(text, "  {invariant:?} ({step}): {what}");
    }
    for (invariant, step) in broken(seed) {
        let (roots, page) = shrink(seed, invariant, step);
        let case = Case::new(&roots);
        let _ = writeln!(
            text,
            "smallest {invariant:?} ({step}), page {page} px:\n{}",
            case.html
        );
        for (broken, cause, what) in check(&case, f32::from(page)) {
            if broken == invariant && cause == step {
                let _ = writeln!(text, "  {what}");
            }
        }
        let (doc, geometry) = layout(&case, f32::from(page));
        text.push_str(&geometry_text(&geometry, &doc));
    }
    text
}

#[test]
fn generated_box_trees_keep_the_fragmenter_invariants() {
    if let Some(seed) = env_u64("FRAGMENT_INVARIANT_SEED") {
        println!("{}", describe(seed));
        return;
    }
    let cases = env_u64("FRAGMENT_INVARIANT_CASES").unwrap_or(CASES);
    let mut found = BTreeSet::new();
    for seed in 0..cases {
        for (invariant, step) in broken(seed) {
            found.insert((seed, invariant, step));
        }
    }
    if std::env::var_os("FRAGMENT_INVARIANT_LIST").is_some() {
        let mut groups: BTreeMap<(Invariant, &str), Vec<u64>> = BTreeMap::new();
        for &(seed, invariant, step) in &found {
            groups.entry((invariant, step)).or_default().push(seed);
        }
        for ((invariant, step), seeds) in groups {
            let seeds: Vec<String> = seeds.iter().map(u64::to_string).collect();
            println!(
                "    (Invariant::{invariant:?}, \"{step}\", &[{}]),",
                seeds.join(", ")
            );
        }
        return;
    }
    let known: BTreeSet<(u64, Invariant, &str)> = KNOWN_FAILURES
        .iter()
        .flat_map(|&(invariant, step, seeds)| {
            seeds.iter().map(move |&seed| (seed, invariant, step))
        })
        .filter(|(seed, _, _)| *seed < cases)
        .collect();
    let mut report = String::new();
    for (seed, invariant, step) in found.difference(&known) {
        let _ = writeln!(
            report,
            "new failure: seed {seed} breaks {invariant:?} ({step})"
        );
    }
    for (seed, invariant, step) in known.difference(&found) {
        let _ = writeln!(
            report,
            "fixed: seed {seed} keeps {invariant:?} ({step}); remove it from KNOWN_FAILURES"
        );
    }
    assert!(
        report.is_empty(),
        "{report}FRAGMENT_INVARIANT_SEED=<seed> prints a case and its smallest failing shapes"
    );
}
