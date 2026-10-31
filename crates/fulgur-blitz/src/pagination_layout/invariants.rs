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
//! the same on every host. `invariants.known.txt` beside this file is the
//! reviewed golden of today's violations: one line per seed, invariant,
//! box, and page, with a note naming the part of the fragmenter expected
//! to fix it. The test fails on any violation the golden lacks and on any
//! it lists that no longer occurs, naming each.
//! `FRAGMENT_INVARIANT_UPDATE=1` rewrites the golden and still fails,
//! listing every violation it added and removed, so no rewrite passes
//! unread. `FRAGMENT_INVARIANT_CASES=<n>` checks more seeds than the
//! default (the golden covers the default ones), and
//! `FRAGMENT_INVARIANT_SEED=<seed>` prints one case's markup, its
//! violations, and for each invariant the smallest case that still breaks
//! it, with that case's geometry.

use super::*;
use crate::blitz_adapter;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::ops::DerefMut;

/// The number of seeds checked by default, and covered by the golden.
const CASES: u64 = 256;

/// The page width; the page height is generated.
const PAGE_WIDTH: f32 = 200.0;

/// How far a placement may stray before it counts, in px.
const EPS: f32 = 0.5;

/// The golden of today's violations.
const KNOWN: &str = include_str!("invariants.known.txt");

/// The fragmenter's invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Invariant {
    /// No fragment ends past the page foot, except one of a monolithic box
    /// taller than a page.
    PastFoot,
    /// Every box is placed, once: a leaf's fragments, one a page on
    /// consecutive pages, add up to its height; a paragraph's line
    /// partition covers its lines once; a container's fragments add up to
    /// no more than its height; no fragment starts above the page top.
    PlacedOnce,
    /// In block flow, and between the rows of a table, document order
    /// matches (page, y) order.
    DocumentOrder,
    /// On each page, a parent's fragment encloses its children's.
    Encloses,
    /// No repeated header sits on a page without a body row.
    HeaderAlone,
}

/// The part of the fragmenter expected to fix a violation, as the golden's
/// note names it.
mod fix {
    /// One page cursor: one rule for a parent's fragment on each page.
    pub const PAGE_CURSOR: &str = "page cursor";
    /// One walker for the body and nested boxes.
    pub const ONE_WALKER: &str = "one walker";
    /// One measure of a box's leading unbreakable extent, lines included.
    pub const LEADING_EXTENT: &str = "leading extent";
    /// Lines measured from the line box (baseline and line height), not
    /// the shaper's selection box.
    pub const LINE_BOX: &str = "leading extent: lines measured from the line box";
    /// A table walker over row groups and rows.
    pub const TABLE_WALKER: &str = "table walker";
    /// One model of box edges (margin, border, padding) at a break.
    pub const EDGE_MODEL: &str = "edge model";
    /// No part of the fragmenter is named yet.
    pub const UNEXPLAINED: &str = "unexplained";
}

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
    /// The box has a margin, a border, or padding.
    edged: bool,
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

/// Whether `style` gives its box a margin, a border, or padding.
fn has_edges(style: &BoxStyle) -> bool {
    style.margin_top > 0 || style.margin_bottom > 0 || style.padding > 0 || style.border > 0
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
            edged: false,
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
                self.boxes[b].edged = has_edges(style);
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
                self.boxes[b].edged = has_edges(style);
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
                self.boxes[b].edged = has_edges(style);
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
                self.boxes[t].edged = has_edges(style);
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
                        self.boxes[c].edged = *cell_padding > 0;
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
                self.boxes[f].edged = has_edges(style);
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

/// One violation of an invariant by one box on one page.
#[derive(Clone, Debug)]
struct Violation {
    invariant: Invariant,
    /// The generated box's `id`.
    box_id: String,
    /// The page the violation is on; `None` for a box placed nowhere.
    page: Option<u32>,
    /// The part of the fragmenter expected to fix it ([`fix`]).
    note: &'static str,
    /// What broke, for a reader.
    what: String,
}

impl Violation {
    /// What the golden compares: the invariant, the box, and the page.
    fn key(&self) -> String {
        let page = self.page.map_or_else(|| "-".to_string(), |p| p.to_string());
        format!("{:?} {} {page}", self.invariant, self.box_id)
    }
}

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

/// `(page, y)` ordered with a tolerance: whether `a` lies before `b` by
/// more than [`EPS`].
fn before(a: (u32, f32), b: (u32, f32)) -> bool {
    a.0 < b.0 || a.0 == b.0 && a.1 + EPS < b.1
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
    /// How far `b`'s fragments add up past its layout height.
    fn excess(&self, b: usize) -> f32 {
        let total: f32 = self.fragments[b].iter().map(|f| f.height.to_f32()).sum();
        total - self.layout_height[b]
    }

    /// The part of the fragmenter expected to fix `b`'s fragments past the
    /// page foot, or adding up past its height, and why.
    fn cause(&self, b: usize) -> (&'static str, &'static str) {
        let info = &self.case.boxes[b];
        let frags = &self.fragments[b];
        if frags.len() > 1 && info.parent.is_none() {
            return (
                fix::ONE_WALKER,
                "the body walker's fragment of a split box ends past the foot",
            );
        }
        if info.kind == Kind::Para && frags.len() > 1 && self.excess(b) > EPS {
            return (
                fix::LINE_BOX,
                "a split paragraph's lines measure taller than its layout box",
            );
        }
        if frags.len() > 1 && (frags[0].height.to_f32() - self.layout_height[b]).abs() < EPS {
            return (
                fix::PAGE_CURSOR,
                "a split box's first fragment is its whole height",
            );
        }
        if info.kind == Kind::Leaf
            && frags.len() == 1
            && self.layout_height[b] > self.page_height + EPS
        {
            return (
                fix::ONE_WALKER,
                "a nested childless box taller than a page is not sliced",
            );
        }
        for child in enclosed(self.case, b) {
            if !self.past[child].is_disjoint(&self.past[b]) || self.excess(child) > EPS {
                return self.cause(child);
            }
        }
        if within(self.case, b, &[Kind::Table]) {
            (fix::TABLE_WALKER, "a table box crosses the page foot")
        } else if within(self.case, b, &[Kind::Flex, Kind::Item]) {
            (fix::LEADING_EXTENT, "a flex row crosses the page foot")
        } else if info.edged || info.children.iter().any(|&c| self.case.boxes[c].edged) {
            (fix::EDGE_MODEL, "a box's edge crosses the page foot")
        } else {
            (
                fix::PAGE_CURSOR,
                "a parent's fragment reaches past its content",
            )
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
    let mut push = |invariant, b: usize, page: Option<u32>, note, what: String| {
        out.push(Violation {
            invariant,
            box_id: case.boxes[b].id.clone(),
            page,
            note,
            what,
        });
    };

    for (b, info) in case.boxes.iter().enumerate() {
        let frags = &placed.fragments[b];

        // No fragment ends past the page foot, except a monolithic box
        // taller than a page.
        for &page in &placed.past[b] {
            let (note, why) = placed.cause(b);
            push(
                Invariant::PastFoot,
                b,
                Some(page),
                note,
                format!(
                    "{} ({:?}) ends past the page foot: {why}",
                    info.id, info.kind
                ),
            );
        }

        // No fragment starts above the page top.
        for f in frags.iter().filter(|f| f.y.to_f32() < -EPS) {
            push(
                Invariant::PlacedOnce,
                b,
                Some(f.page_index),
                fix::UNEXPLAINED,
                format!("{} starts above the page top, at {}", info.id, f.y.to_f32()),
            );
        }

        if info.in_header {
            continue;
        }

        // Every box is placed, once.
        let pages: Vec<u32> = frags.iter().map(|f| f.page_index).collect();
        if frags.is_empty() && info.kind != Kind::Row {
            push(
                Invariant::PlacedOnce,
                b,
                None,
                fix::UNEXPLAINED,
                format!("{} ({:?}) is not placed", info.id, info.kind),
            );
        }
        if matches!(info.kind, Kind::Leaf | Kind::Image | Kind::Para) {
            for pair in pages.windows(2) {
                if pair[1] <= pair[0] {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pair[1]),
                        fix::UNEXPLAINED,
                        format!("{} is placed twice or out of order: {pages:?}", info.id),
                    );
                } else if pair[1] > pair[0] + 1 {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pair[0] + 1),
                        fix::UNEXPLAINED,
                        format!("{} skips a page: {pages:?}", info.id),
                    );
                }
            }
        }
        match info.kind {
            Kind::Leaf | Kind::Image if !frags.is_empty() => {
                let excess = placed.excess(b);
                if excess.abs() > EPS * frags.len() as f32 {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pages[0]),
                        fix::UNEXPLAINED,
                        format!(
                            "{}'s fragments add up to {excess} px more than its height",
                            info.id
                        ),
                    );
                }
            }
            Kind::Para if !frags.is_empty() => {
                let boundaries = &placed.line_boundaries[b];
                let partitioned = if boundaries.is_empty() {
                    frags.len() == 1
                } else {
                    boundaries.len() == frags.len() + 1
                        && boundaries[0] == 0
                        && boundaries[boundaries.len() - 1] == info.lines
                        && boundaries.windows(2).all(|w| w[0] < w[1])
                };
                if !partitioned {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pages[0]),
                        fix::UNEXPLAINED,
                        format!(
                            "{} partitions {} lines as {boundaries:?} over {} fragments",
                            info.id,
                            info.lines,
                            frags.len()
                        ),
                    );
                }
            }
            Kind::Block | Kind::Table | Kind::BodyCell | Kind::Flex | Kind::Item
                if placed.excess(b) > EPS * frags.len().max(1) as f32 =>
            {
                let (note, why) = placed.cause(b);
                push(
                    Invariant::PlacedOnce,
                    b,
                    Some(pages[0]),
                    note,
                    format!(
                        "{} ({:?})'s fragments add up to {} px more than its height: {why}",
                        info.id,
                        info.kind,
                        placed.excess(b)
                    ),
                );
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
            let a_end = (last.page_index, last.y.to_f32() + last.height.to_f32());
            let c_start = (first.page_index, first.y.to_f32());
            if before(c_start, a_end) {
                let (note, why) = if case.boxes[a].kind == Kind::Para && placed.excess(a) > EPS {
                    (
                        fix::LINE_BOX,
                        "the paragraph before it measures its lines past its box",
                    )
                } else {
                    (fix::UNEXPLAINED, "it starts before the box before it ends")
                };
                push(
                    Invariant::DocumentOrder,
                    c,
                    Some(first.page_index),
                    note,
                    format!(
                        "{} starts at {c_start:?}, before {} ends at {a_end:?}: {why}",
                        case.boxes[c].id, case.boxes[a].id
                    ),
                );
            }
        }

        // Between the body rows of a table, document order matches (page,
        // y) order: a row starts where the row before it ends, or later.
        if info.kind == Kind::Table {
            let rows: Vec<usize> = info
                .children
                .iter()
                .copied()
                .filter(|&row| !case.boxes[row].in_header)
                .collect();
            let start = |row: usize| {
                case.boxes[row]
                    .children
                    .iter()
                    .filter_map(|&cell| placed.fragments[cell].first())
                    .map(|f| (f.page_index, f.y.to_f32()))
                    .reduce(|a, b| if before(b, a) { b } else { a })
            };
            let end = |row: usize| {
                case.boxes[row]
                    .children
                    .iter()
                    .filter_map(|&cell| placed.fragments[cell].last())
                    .map(|f| (f.page_index, f.y.to_f32() + f.height.to_f32()))
                    .reduce(|a, b| if before(a, b) { b } else { a })
            };
            for pair in rows.windows(2) {
                let (Some(a_end), Some(c_start)) = (end(pair[0]), start(pair[1])) else {
                    continue;
                };
                if before(c_start, a_end) {
                    push(
                        Invariant::DocumentOrder,
                        pair[1],
                        Some(c_start.0),
                        fix::TABLE_WALKER,
                        format!(
                            "row {} starts at {c_start:?}, before row {} ends at {a_end:?}",
                            case.boxes[pair[1]].id, case.boxes[pair[0]].id
                        ),
                    );
                }
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
                let (note, why) = if !placed.past[child].is_empty() || placed.excess(child) > EPS {
                    placed.cause(child)
                } else if within(case, child, &[Kind::Table]) {
                    (fix::TABLE_WALKER, "a table box is outside its parent")
                } else if within(case, child, &[Kind::Flex, Kind::Item]) {
                    (fix::LEADING_EXTENT, "a flex box is outside its parent")
                } else {
                    (fix::UNEXPLAINED, "a box is outside its parent")
                };
                push(
                    Invariant::Encloses,
                    child,
                    Some(cf.page_index),
                    note,
                    format!(
                        "{} ({:?}) at y {} h {} is outside {} ({:?}): {why}",
                        case.boxes[child].id,
                        case.boxes[child].kind,
                        cf.y.to_f32(),
                        cf.height.to_f32(),
                        info.id,
                        info.kind
                    ),
                );
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
        for &page in pages(header).difference(&pages(body)) {
            push(
                Invariant::HeaderAlone,
                *table,
                Some(page),
                fix::LEADING_EXTENT,
                format!(
                    "{}'s header is alone: its first row places nothing on the page",
                    case.boxes[*table].id
                ),
            );
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

/// The violations of `seed`'s case.
fn violations(seed: u64) -> Vec<Violation> {
    let (roots, page) = case_of(seed);
    check(&Case::new(&roots), f32::from(page))
}

/// `seed`'s case shrunk to the smallest one that still breaks `invariant`.
fn shrink(seed: u64, invariant: Invariant) -> (Vec<Gen>, u16) {
    let mut tree = case_strategy().new_tree(&mut runner(seed)).expect("a case");
    let fails = |value: &(Vec<Gen>, u16)| {
        check(&Case::new(&value.0), f32::from(value.1))
            .iter()
            .any(|v| v.invariant == invariant)
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

/// One seed's case, its violations, and for each invariant the smallest
/// case that still breaks it, with that case's geometry.
fn describe(seed: u64) -> String {
    let (roots, page) = case_of(seed);
    let case = Case::new(&roots);
    let found = check(&case, f32::from(page));
    let mut text = format!("seed {seed}, page {page} px\n{}\n", case.html);
    for v in &found {
        let _ = writeln!(text, "  {} ({}): {}", v.key(), v.note, v.what);
    }
    let invariants: BTreeSet<Invariant> = found.iter().map(|v| v.invariant).collect();
    for invariant in invariants {
        let (roots, page) = shrink(seed, invariant);
        let case = Case::new(&roots);
        let _ = writeln!(
            text,
            "smallest {invariant:?}, page {page} px:\n{}",
            case.html
        );
        for v in check(&case, f32::from(page)) {
            if v.invariant == invariant {
                let _ = writeln!(text, "  {} ({}): {}", v.key(), v.note, v.what);
            }
        }
        let (doc, geometry) = layout(&case, f32::from(page));
        text.push_str(&geometry_text(&geometry, &doc));
    }
    text
}

/// The golden's lines: `<seed> <invariant> <box> <page>  # <note>`, as
/// `(seed, key)` with the note.
fn golden_lines(text: &str) -> BTreeMap<(u64, String), String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (fields, note) = line.split_once('#').unwrap_or((line, ""));
            let (seed, key) = fields
                .trim()
                .split_once(' ')
                .unwrap_or_else(|| panic!("bad golden line {line:?}"));
            let seed = seed
                .parse()
                .unwrap_or_else(|_| panic!("bad seed in golden line {line:?}"));
            ((seed, key.trim().to_string()), note.trim().to_string())
        })
        .collect()
}

/// The golden of `found`.
fn golden_text(found: &BTreeMap<(u64, String), String>) -> String {
    let mut text = String::from(
        "# The fragmenter's invariant violations over the default seeds\n\
         # (`invariants.rs`): seed, invariant, box, page (`-` when the box is\n\
         # placed nowhere), then the part of the fragmenter expected to fix it.\n\
         # Rewrite with FRAGMENT_INVARIANT_UPDATE=1 and review every line.\n",
    );
    for ((seed, key), note) in found {
        let _ = writeln!(text, "{seed} {key}  # {note}");
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
    let mut found = BTreeMap::new();
    for seed in 0..cases {
        for v in violations(seed) {
            found.entry((seed, v.key())).or_insert(v.note.to_string());
        }
    }
    let known: BTreeMap<(u64, String), String> = golden_lines(KNOWN)
        .into_iter()
        .filter(|((seed, _), _)| *seed < cases)
        .collect();
    let mut report = String::new();
    for ((seed, key), note) in &found {
        if !known.contains_key(&(*seed, key.clone())) {
            let _ = writeln!(report, "added: {seed} {key}  # {note}");
        }
    }
    for ((seed, key), note) in &known {
        if !found.contains_key(&(*seed, key.clone())) {
            let _ = writeln!(report, "removed: {seed} {key}  # {note}");
        }
    }
    if std::env::var_os("FRAGMENT_INVARIANT_UPDATE").is_some() {
        assert_eq!(cases, CASES, "the golden covers the default seeds only");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/pagination_layout/invariants.known.txt"
        );
        std::fs::write(path, golden_text(&found)).expect("write the golden");
        assert!(
            report.is_empty(),
            "rewrote {path}; review each violation it added and removed:\n{report}"
        );
        return;
    }
    assert!(
        report.is_empty(),
        "the violations differ from invariants.known.txt:\n{report}\
         FRAGMENT_INVARIANT_SEED=<seed> prints a case and its smallest failing shapes"
    );
}
