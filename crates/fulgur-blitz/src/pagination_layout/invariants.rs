//! Invariants of the fragmenter over generated box trees.
//!
//! A proptest strategy generates trees of up to 30 boxes: blocks with
//! margins, padding, and borders, and `orphans` and `widows` to inherit;
//! paragraphs of 1 to 12 lines, with their own `orphans` and `widows` or
//! inherited ones; tables of 1 to 8 rows of 1 to 3 cells, with or without a
//! `thead`, a header row that skips the first page
//! (`-fulgur-header-repeat: skip-first`), a caption, a cell spanning two
//! rows, `orphans` and `widows`, and row counts (`-fulgur-table-orphans`,
//! `-fulgur-table-widows`); flex rows of two items, wrapping flex lines and
//! grids of two columns of up to four items, and flex columns of up to
//! three; floats; images; `break-inside: avoid`, and `break-before` and
//! `break-after` of `page`, `left`, `right`, or `avoid`, on random boxes,
//! table rows, row groups, and items. Pages are 100 to 400 px tall. Each
//! case is laid out with its break styles, its captions moved out of their
//! tables as the engine moves them, and checked against six invariants
//! ([`Invariant`]).
//!
//! The cases are drawn from fixed seeds, one per case, so a failure names
//! its seed, and are laid out in Noto Sans with no system font, so they are
//! the same on every host. `invariants.known.txt` beside this file is the
//! reviewed golden of today's violations: one line per seed, invariant,
//! box, and page, with a note, written by hand, stating its cause where it
//! was traced. The test fails on any violation the golden lacks and on any
//! it lists that no longer occurs, naming each.
//! `FRAGMENT_INVARIANT_UPDATE=1` rewrites the golden, keeping the note of
//! each line it keeps and writing each new line with none, and still fails,
//! listing every violation it added and removed, so no rewrite passes
//! unread. `FRAGMENT_INVARIANT_CASES=<n>` checks more seeds than the
//! default (the golden covers the default ones), and
//! `FRAGMENT_INVARIANT_SEED=<seed>` prints one case's markup, its
//! violations, and for each invariant the smallest case that still breaks
//! it, with that case's geometry. `FRAGMENT_INVARIANT_REACH=1` counts the
//! default seeds whose breaks the walker takes at an `avoid` boundary,
//! between table rows, and inside one ([`reach`]).

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
    /// no more than its height, one before its last free to run on to the
    /// page foot below its children (css-break-3 §4); no fragment starts
    /// above the page top.
    PlacedOnce,
    /// In block flow, and between the rows of a table, document order
    /// matches (page, y) order.
    DocumentOrder,
    /// On each page, a parent's fragment encloses its children's.
    Encloses,
    /// No repeated header sits on a page without a body row.
    HeaderAlone,
    /// A fragment of an item of a line (a flex or grid line's item, a
    /// table row's cell), stretched to the line, ends no lower than the
    /// line ends on its page: the lowest end of its items' content and
    /// bottom edges there, or the page foot on a page where the line breaks
    /// inside content (a box in an item goes on to a later page, or an
    /// item's content and bottom edge reach the foot).
    EndsWithLine,
}

/// A generated box's `break-before` or `break-after`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Break {
    #[default]
    Auto,
    Page,
    Left,
    Right,
    Avoid,
}

impl Break {
    /// The CSS keyword, or `None` for `auto`, which is left unwritten.
    fn keyword(self) -> Option<&'static str> {
        match self {
            Break::Auto => None,
            Break::Page => Some("page"),
            Break::Left => Some("left"),
            Break::Right => Some("right"),
            Break::Avoid => Some("avoid"),
        }
    }
}

fn break_value() -> impl Strategy<Value = Break> {
    prop_oneof![
        83 => Just(Break::Auto),
        5 => Just(Break::Page),
        1 => Just(Break::Left),
        1 => Just(Break::Right),
        10 => Just(Break::Avoid),
    ]
}

/// The style of a generated box.
#[derive(Clone, Debug, Default)]
struct BoxStyle {
    margin_top: u8,
    margin_bottom: u8,
    padding: u8,
    border: u8,
    avoid: bool,
    break_before: Break,
    break_after: Break,
}

/// `orphans` and `widows`, or nothing, so that they are inherited.
type LineCounts = Option<(u8, u8)>;

/// The breaks of a table row, a row group, or a flex or grid item.
#[derive(Clone, Copy, Debug, Default)]
struct Breaks {
    avoid: bool,
    before: Break,
    after: Break,
}

impl Breaks {
    /// The CSS of the breaks.
    fn css(self) -> String {
        style_css(&BoxStyle {
            avoid: self.avoid,
            break_before: self.before,
            break_after: self.after,
            ..BoxStyle::default()
        })
    }
}

/// How a flex or grid container lays its items out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlexMode {
    /// One flex line of all its items.
    Row,
    /// Flex lines of two items each (`flex-wrap: wrap`).
    Wrap,
    /// One item a line (`flex-direction: column`).
    Column,
    /// A grid of two columns.
    Grid,
}

impl FlexMode {
    /// The items on each line.
    fn per_line(self) -> usize {
        match self {
            FlexMode::Row => usize::MAX,
            FlexMode::Wrap | FlexMode::Grid => 2,
            FlexMode::Column => 1,
        }
    }

    /// The container's and each item's CSS.
    fn css(self) -> (&'static str, &'static str) {
        match self {
            FlexMode::Row => ("display: flex; ", "flex: 1 1 0; "),
            FlexMode::Wrap => ("display: flex; flex-wrap: wrap; ", "flex: 0 0 45%; "),
            FlexMode::Column => ("display: flex; flex-direction: column; ", ""),
            FlexMode::Grid => ("display: grid; grid-template-columns: 1fr 1fr; ", ""),
        }
    }
}

/// A table's header rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// No `thead`.
    None,
    /// A `thead` of one row, repeated on every page.
    Plain,
    /// A `thead` of a row that skips the table's first page
    /// (`-fulgur-header-repeat: skip-first`) above a row repeated on every
    /// page.
    SkipFirst,
}

impl Head {
    /// The header rows.
    fn rows(self) -> usize {
        match self {
            Head::None => 0,
            Head::Plain => 1,
            Head::SkipFirst => 2,
        }
    }
}

/// A generated box.
#[derive(Clone, Debug)]
enum Gen {
    /// A block of a fixed height with no children.
    Leaf { style: BoxStyle, height: u16 },
    /// A replaced box (`img`) of a fixed height.
    Image { height: u16 },
    /// A floated block of a fixed height, 40% of its container wide.
    Float { left: bool, height: u16 },
    /// A paragraph of `lines` lines of `line_height` px.
    Para {
        style: BoxStyle,
        lines: u8,
        line_height: u8,
        counts: LineCounts,
    },
    /// A block holding other boxes, with `orphans` and `widows` they
    /// inherit.
    Block {
        style: BoxStyle,
        counts: LineCounts,
        children: Vec<Gen>,
    },
    /// A table; each row's cells hold one box each. With `span`, the first
    /// cell of that row spans it and the next, which has one cell fewer.
    Table {
        style: BoxStyle,
        head: Head,
        caption: bool,
        counts: LineCounts,
        row_counts: LineCounts,
        cell_padding: u8,
        /// The body row group's breaks.
        body: Breaks,
        rows: Vec<(Breaks, Vec<Gen>)>,
        span: Option<usize>,
    },
    /// A flex or grid container of two to four items, each holding one
    /// box, with their breaks.
    Flex {
        style: BoxStyle,
        mode: FlexMode,
        items: Vec<(Breaks, Gen)>,
    },
}

fn style() -> BoxedStrategy<BoxStyle> {
    (
        prop_oneof![4 => Just(0u8), 1 => 1u8..=20],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=20],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=12],
        prop_oneof![4 => Just(0u8), 1 => 1u8..=4],
        proptest::bool::weighted(0.2),
        break_value(),
        break_value(),
    )
        .prop_map(
            |(margin_top, margin_bottom, padding, border, avoid, break_before, break_after)| {
                BoxStyle {
                    margin_top,
                    margin_bottom,
                    padding,
                    border,
                    avoid,
                    break_before,
                    break_after,
                }
            },
        )
        .boxed()
}

/// The breaks of a row, a row group, or an item.
fn breaks() -> BoxedStrategy<Breaks> {
    (proptest::bool::weighted(0.2), break_value(), break_value())
        .prop_map(|(avoid, before, after)| Breaks {
            avoid,
            before,
            after,
        })
        .boxed()
}

/// `orphans` and `widows` of 1 to 4, or nothing.
fn line_counts(probability: f64) -> impl Strategy<Value = LineCounts> {
    proptest::option::weighted(probability, (1u8..=4, 1u8..=4))
}

/// A box with no box children.
fn leaf() -> BoxedStrategy<Gen> {
    prop_oneof![
        4 => (style(), 5u16..=160).prop_map(|(style, height)| Gen::Leaf { style, height }),
        1 => (5u16..=120).prop_map(|height| Gen::Image { height }),
        1 => (any::<bool>(), 5u16..=160).prop_map(|(left, height)| Gen::Float { left, height }),
        4 => (style(), 1u8..=12, prop_oneof![Just(20u8), 16u8..=30], line_counts(0.7)).prop_map(
            |(style, lines, line_height, counts)| Gen::Para {
                style,
                lines,
                line_height,
                counts,
            }
        ),
    ]
    .boxed()
}

/// The boxes of `nodes`, rows, cells, and captions included.
fn box_count(nodes: &[Gen]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            Gen::Leaf { .. } | Gen::Image { .. } | Gen::Float { .. } | Gen::Para { .. } => 1,
            Gen::Block { children, .. } => 1 + box_count(children),
            Gen::Table {
                head,
                caption,
                rows,
                ..
            } => {
                let header = head.rows() * (1 + rows[0].1.len());
                1 + usize::from(*caption)
                    + header
                    + rows
                        .iter()
                        .map(|(_, cells)| 1 + cells.len() + box_count(cells))
                        .sum::<usize>()
            }
            Gen::Flex { items, .. } => {
                1 + items.len()
                    + items
                        .iter()
                        .map(|(_, item)| box_count(std::slice::from_ref(item)))
                        .sum::<usize>()
            }
        })
        .sum()
}

/// A tree of up to 30 boxes.
fn tree() -> impl Strategy<Value = Vec<Gen>> {
    let node = leaf().prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            3 => (style(), line_counts(0.2), prop::collection::vec(inner.clone(), 1..=4))
                .prop_map(|(style, counts, children)| Gen::Block {
                    style,
                    counts,
                    children
                })
                .boxed(),
            2 => (
                style(),
                prop_oneof![2 => Just(Head::None), 1 => Just(Head::Plain), 1 => Just(Head::SkipFirst)],
                proptest::bool::weighted(0.25),
                line_counts(0.2),
                line_counts(0.2),
                prop_oneof![Just(0u8), Just(10u8)],
                breaks(),
                prop::collection::vec((breaks(), prop::collection::vec(leaf(), 1..=3)), 1..=8),
                proptest::option::weighted(0.3, any::<prop::sample::Index>()),
            )
                .prop_map(
                    |(style, head, caption, counts, row_counts, cell_padding, body, rows, span)| {
                        // Every row of a table has as many cells as its
                        // first, but the row below a spanning cell, which
                        // has one fewer; a span needs a second column and a
                        // row below.
                        let columns = rows[0].1.len();
                        let span = span
                            .filter(|_| columns > 1 && rows.len() > 1)
                            .map(|row| row.index(rows.len() - 1));
                        let rows = rows
                            .into_iter()
                            .enumerate()
                            .map(|(row, (breaks, mut cells))| {
                                cells.resize(columns, Gen::Image { height: 20 });
                                if span.is_some_and(|span| span + 1 == row) {
                                    cells.remove(0);
                                }
                                (breaks, cells)
                            })
                            .collect();
                        Gen::Table {
                            style,
                            head,
                            caption,
                            counts,
                            row_counts,
                            cell_padding,
                            body,
                            rows,
                            span,
                        }
                    }
                )
                .boxed(),
            2 => (
                style(),
                prop_oneof![
                    2 => Just(FlexMode::Row),
                    1 => Just(FlexMode::Wrap),
                    1 => Just(FlexMode::Column),
                    2 => Just(FlexMode::Grid),
                ],
                prop::collection::vec((breaks(), inner), 2..=4),
            )
                .prop_map(|(style, mode, mut items)| {
                    // A flex row lays out two items, a column three.
                    let most = match mode {
                        FlexMode::Row => 2,
                        FlexMode::Column => 3,
                        FlexMode::Wrap | FlexMode::Grid => 4,
                    };
                    items.truncate(most);
                    Gen::Flex { style, mode, items }
                })
                .boxed(),
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
    /// The top padding a table cell repeats on each page it continues on.
    repeated_edge: f32,
    /// The rows a table cell spans.
    rowspan: usize,
    /// The top margin.
    margin_top: f32,
    /// The bottom margin.
    margin_bottom: f32,
    /// The bottom padding and border.
    bottom_edge: f32,
    /// The box's `break-before` and `break-after`.
    breaks: (Break, Break),
    /// A flex or grid container's items on each of its lines.
    per_line: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Leaf,
    Image,
    Float,
    Caption,
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
    if let Some(value) = style.break_before.keyword() {
        let _ = write!(css, "break-before: {value}; ");
    }
    if let Some(value) = style.break_after.keyword() {
        let _ = write!(css, "break-after: {value}; ");
    }
    css
}

/// `counts` as `orphans` and `widows` under the property names `prefix`
/// gives them.
fn counts_css(prefix: &str, counts: LineCounts) -> String {
    counts.map_or_else(String::new, |(orphans, widows)| {
        format!("{prefix}orphans: {orphans}; {prefix}widows: {widows}; ")
    })
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
             caption {{ padding: 0; }} \
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
            repeated_edge: 0.0,
            rowspan: 1,
            margin_top: 0.0,
            margin_bottom: 0.0,
            bottom_edge: 0.0,
            breaks: (Break::Auto, Break::Auto),
            per_line: usize::MAX,
        });
        if let Some(parent) = parent {
            self.boxes[parent].children.push(index);
        }
        index
    }

    /// Record the edges `style` gives the box `b`.
    fn edges(&mut self, b: usize, style: &BoxStyle) {
        let info = &mut self.boxes[b];
        info.margin_top = f32::from(style.margin_top);
        info.margin_bottom = f32::from(style.margin_bottom);
        info.bottom_edge = f32::from(style.padding) + f32::from(style.border);
        info.breaks = (style.break_before, style.break_after);
    }

    fn emit(&mut self, node: &Gen, parent: Option<usize>, in_header: bool, out: &mut String) {
        match node {
            Gen::Leaf { style, height } => {
                let b = self.add(Kind::Leaf, parent, in_header, 0);
                self.edges(b, style);
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
            Gen::Float { left, height } => {
                let b = self.add(Kind::Float, parent, in_header, 0);
                let side = if *left { "left" } else { "right" };
                let _ = write!(
                    out,
                    r#"<div id="b{b}" style="float: {side}; width: 40%; height: {height}px"></div>"#
                );
            }
            Gen::Para {
                style,
                lines,
                line_height,
                counts,
            } => {
                let b = self.add(Kind::Para, parent, in_header, usize::from(*lines));
                self.edges(b, style);
                let text = (1..=*lines)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join("<br>");
                let _ = write!(
                    out,
                    r#"<p id="b{b}" style="{}{}line-height: {line_height}px">{text}</p>"#,
                    style_css(style),
                    counts_css("", *counts)
                );
            }
            Gen::Block {
                style,
                counts,
                children,
            } => {
                let b = self.add(Kind::Block, parent, in_header, 0);
                self.edges(b, style);
                let _ = write!(
                    out,
                    r#"<div id="b{b}" style="{}{}">"#,
                    style_css(style),
                    counts_css("", *counts)
                );
                for child in children {
                    self.emit(child, Some(b), in_header, out);
                }
                out.push_str("</div>");
            }
            Gen::Table {
                style,
                head,
                caption,
                counts,
                row_counts,
                cell_padding,
                body: body_breaks,
                rows,
                span,
            } => {
                let t = self.add(Kind::Table, parent, in_header, 0);
                self.edges(t, style);
                let _ = write!(
                    out,
                    r#"<table id="b{t}" style="{}{}{}">"#,
                    style_css(style),
                    counts_css("", *counts),
                    counts_css("-fulgur-table-", *row_counts)
                );
                if *caption {
                    let c = self.add(Kind::Caption, Some(t), in_header, 0);
                    let _ = write!(out, r#"<caption id="b{c}">C</caption>"#);
                }
                let mut header = Vec::new();
                let mut body = Vec::new();
                if *head != Head::None {
                    out.push_str("<thead>");
                    for row in 0..head.rows() {
                        let r = self.add(Kind::Row, Some(t), true, 0);
                        let row_style = if *head == Head::SkipFirst && row == 0 {
                            "-fulgur-header-repeat: skip-first"
                        } else {
                            ""
                        };
                        let _ = write!(out, r#"<tr id="b{r}" style="{row_style}">"#);
                        for _ in &rows[0].1 {
                            let c = self.add(Kind::Cell, Some(r), true, 0);
                            header.push(c);
                            let _ = write!(out, r#"<th id="b{c}">H</th>"#);
                        }
                        out.push_str("</tr>");
                    }
                    out.push_str("</thead>");
                }
                let _ = write!(out, r#"<tbody style="{}">"#, body_breaks.css());
                for (row, (breaks, cells)) in rows.iter().enumerate() {
                    let r = self.add(Kind::Row, Some(t), in_header, 0);
                    self.boxes[r].breaks = (breaks.before, breaks.after);
                    let _ = write!(out, r#"<tr id="b{r}" style="{}">"#, breaks.css());
                    for (column, cell) in cells.iter().enumerate() {
                        let c = self.add(Kind::BodyCell, Some(r), in_header, 0);
                        self.boxes[c].repeated_edge = f32::from(*cell_padding);
                        self.boxes[c].bottom_edge = f32::from(*cell_padding);
                        body.push(c);
                        let rowspan = if *span == Some(row) && column == 0 {
                            self.boxes[c].rowspan = 2;
                            r#" rowspan="2""#
                        } else {
                            ""
                        };
                        let _ = write!(
                            out,
                            r#"<td id="b{c}"{rowspan} style="padding: {cell_padding}px 0">"#
                        );
                        self.emit(cell, Some(c), in_header, out);
                        out.push_str("</td>");
                    }
                    out.push_str("</tr>");
                }
                out.push_str("</tbody></table>");
                if *head != Head::None {
                    self.tables.push((t, header, body));
                }
            }
            Gen::Flex { style, mode, items } => {
                let f = self.add(Kind::Flex, parent, in_header, 0);
                self.edges(f, style);
                self.boxes[f].per_line = mode.per_line();
                let (display, item_style) = mode.css();
                let _ = write!(
                    out,
                    r#"<div id="b{f}" style="{display}{}">"#,
                    style_css(style)
                );
                for (breaks, item) in items {
                    let i = self.add(Kind::Item, Some(f), in_header, 0);
                    self.boxes[i].breaks = (breaks.before, breaks.after);
                    let _ = write!(
                        out,
                        r#"<div id="b{i}" style="{item_style}{}">"#,
                        breaks.css()
                    );
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
    /// By how many px a box's fragments add up past (or short of) its
    /// height, so that the golden fails when it grows.
    amount: Option<f32>,
    /// What broke, for a reader.
    what: String,
}

impl Violation {
    /// What the golden compares: the invariant, the box, the page, and the
    /// amount to the px, signed, where there is one.
    fn key(&self) -> String {
        let page = self.page.map_or_else(|| "-".to_string(), |p| p.to_string());
        let amount = self
            .amount
            .map_or_else(String::new, |amount| format!(" {amount:+.0}px"));
        format!("{:?} {} {page}{amount}", self.invariant, self.box_id)
    }
}

/// `case` laid out with its break styles on pages `page_height` px tall,
/// in Noto Sans and no system font, so the lines are the same on every host,
/// its captions moved out of their tables as the engine moves them.
fn layout(case: &Case, page_height: f32) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
    use blitz_adapter::DomPass as _;
    let fonts = [crate::test_fonts::noto_sans_regular_ttf()];
    let (mut doc, _, _) = blitz_adapter::parse_html_with_local_resources(
        &case.html, PAGE_WIDTH, 10000, &fonts, false, None,
    );
    blitz_adapter::CaptionRestructurePass
        .apply(&mut doc, &blitz_adapter::PassContext { font_data: &[] });
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

/// The body rows of the table `table`.
fn body_rows(case: &Case, table: usize) -> impl Iterator<Item = usize> + '_ {
    case.boxes[table]
        .children
        .iter()
        .copied()
        .filter(|&row| case.boxes[row].kind == Kind::Row && !case.boxes[row].in_header)
}

/// The lines of the table `table`'s body, as the cells of each: a row's,
/// joined with the row below when a cell of it spans that row.
fn table_lines(case: &Case, table: usize) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut spanned = false;
    for row in body_rows(case, table) {
        let cells = &case.boxes[row].children;
        match lines.last_mut() {
            Some(line) if spanned => line.extend(cells),
            _ => lines.push(cells.clone()),
        }
        spanned = cells.iter().any(|&c| case.boxes[c].rowspan > 1);
    }
    lines
}

/// The lines of the flex or grid container `flex`, as its items on each.
fn flex_lines(case: &Case, flex: usize) -> Vec<Vec<usize>> {
    let info = &case.boxes[flex];
    info.children
        .chunks(info.per_line)
        .map(<[usize]>::to_vec)
        .collect()
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
    /// Each table cell's height down to its content's end as laid out,
    /// before its row stretches it: its children's margin boxes and its
    /// bottom edge. Zero for every other box.
    natural_height: Vec<f32>,
    /// The pages on which each box's fragment ends past the page foot.
    past: Vec<BTreeSet<u32>>,
}

impl Placed<'_> {
    /// Each table and page on which its header sits without a body row
    /// ([`Invariant::HeaderAlone`]): no body cell holds content there, an
    /// empty fragment of one being none.
    fn alone_headers(&self) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        for (table, header, body) in &self.case.tables {
            let pages = |boxes: Vec<usize>| -> BTreeSet<u32> {
                boxes
                    .into_iter()
                    .flat_map(|c| self.fragments[c].iter().map(|f| f.page_index))
                    .collect()
            };
            let header_pages = pages(header.clone());
            let body_pages = pages(body.iter().flat_map(|&c| enclosed(self.case, c)).collect());
            for &page in header_pages.difference(&body_pages) {
                out.push((*table, page));
            }
        }
        out
    }

    /// Each fragment of a line's item that ends past its line's end on its
    /// page ([`Invariant::EndsWithLine`]): the item, the page, and what broke.
    /// A fragment past the page foot is left to [`Invariant::PastFoot`].
    fn past_line_end(&self) -> Vec<(usize, u32, String)> {
        let mut out = Vec::new();
        for line in self.lines() {
            let pages: BTreeSet<u32> = line
                .iter()
                .flat_map(|&c| self.fragments[c].iter().map(|f| f.page_index))
                .collect();
            let Some(&last_page) = pages.last() else {
                continue;
            };
            for &page in &pages {
                let Some(end) = line
                    .iter()
                    .filter_map(|&c| self.content_end(c, page))
                    .reduce(f32::max)
                else {
                    continue;
                };
                // The line breaks inside content on this page when a box in
                // one of its items goes on to a later page, or an item's
                // content and bottom edge reach the foot: only then may its
                // fragments run on to the foot.
                let breaks_inside = page < last_page
                    && line.iter().any(|&c| {
                        let frags = &self.fragments[c];
                        let edge = if frags.last().is_some_and(|f| f.page_index == page) {
                            0.0
                        } else {
                            self.case.boxes[c].bottom_edge
                        };
                        self.content_end(c, page)
                            .is_some_and(|end| end + edge >= self.page_height - EPS)
                            || enclosed(self.case, c).iter().any(|&child| {
                                let frags = &self.fragments[child];
                                frags.iter().any(|f| f.page_index == page)
                                    && frags.iter().any(|f| f.page_index > page)
                            })
                    });
                for &c in &line {
                    for f in self.fragments[c].iter().filter(|f| f.page_index == page) {
                        let bottom = f.y.to_f32() + f.height.to_f32();
                        let to_foot = breaks_inside && bottom >= self.page_height - EPS;
                        if bottom > end + EPS && bottom <= self.page_height + EPS && !to_foot {
                            let info = &self.case.boxes[c];
                            out.push((
                                c,
                                page,
                                format!(
                                    "{} ({:?}) ends at {bottom}, past its line's end at {end}",
                                    info.id, info.kind
                                ),
                            ));
                        }
                    }
                }
            }
        }
        out
    }

    /// The lines of boxes side by side: each flex or grid container's
    /// items, and each table's body lines ([`table_lines`]).
    fn lines(&self) -> Vec<Vec<usize>> {
        let case = self.case;
        (0..case.boxes.len())
            .flat_map(|b| match case.boxes[b].kind {
                Kind::Flex => flex_lines(case, b),
                Kind::Table => table_lines(case, b),
                _ => Vec::new(),
            })
            .collect()
    }

    /// Where the item `c`'s content ends on `page`, if it has a fragment
    /// there: below its children's there ([`Self::children_end`]), or
    /// below the top edge it repeats where it holds none, and on its last
    /// page below its bottom edge.
    fn content_end(&self, c: usize, page: u32) -> Option<f32> {
        let info = &self.case.boxes[c];
        let frags = &self.fragments[c];
        let top = frags
            .iter()
            .filter(|f| f.page_index == page)
            .map(|f| f.y.to_f32())
            .reduce(f32::min)?;
        let content = self
            .children_end(c, page)
            .map_or(top + info.repeated_edge, |end| {
                end.max(top + info.repeated_edge)
            });
        let last = frags.last().is_some_and(|f| f.page_index == page);
        Some(content + if last { info.bottom_edge } else { 0.0 })
    }

    /// Where the content of `b`'s fragment `k` ends: its fragment's end as
    /// [`Self::counted_bottom`] counts it, a paragraph's padding and border
    /// added where its fragments, measured by its lines, leave them out (its
    /// top edge on its first fragment, its bottom edge on its last), and a
    /// container's lowest child on its page below it, with the container's
    /// bottom edge on its last fragment; a float in a block that is no
    /// formatting context root may hang below the block's edge.
    fn content_bottom(&self, b: usize, k: usize) -> f32 {
        let info = &self.case.boxes[b];
        let frags = &self.fragments[b];
        let own = self.counted_bottom(b, k);
        let last = k + 1 == frags.len();
        if info.kind == Kind::Para {
            // A paragraph placed by its lines has fragments that add up to
            // its lines alone; one placed as a whole box, to its height.
            if self.excess(b).abs() <= EPS * frags.len() as f32 {
                return own;
            }
            let top = if k == 0 { info.bottom_edge } else { 0.0 };
            let bottom = if last { info.bottom_edge } else { 0.0 };
            return own + top + bottom;
        }
        let page = frags[k].page_index;
        let edge = if last { info.bottom_edge } else { 0.0 };
        let in_flow = self
            .children_end_where(b, page, |c| self.case.boxes[c].kind != Kind::Float)
            .map_or(own, |end| own.max(end + edge));
        self.children_end_where(b, page, |c| self.case.boxes[c].kind == Kind::Float)
            .map_or(in_flow, |end| in_flow.max(end))
    }

    /// The lowest end on `page` of the content of `b`'s children
    /// ([`Self::content_bottom`]), each on its last page with the bottom
    /// margin that stays inside `b`: `b`'s last in-flow child's margin
    /// collapses through a block with no bottom edge, and is counted where
    /// that margin stops ([`Self::margin_through`]).
    fn children_end(&self, b: usize, page: u32) -> Option<f32> {
        self.children_end_where(b, page, |_| true)
    }

    /// [`Self::children_end`] over the children `keep` keeps.
    fn children_end_where(&self, b: usize, page: u32, keep: impl Fn(usize) -> bool) -> Option<f32> {
        let info = &self.case.boxes[b];
        let collapses = info.kind == Kind::Block && info.bottom_edge == 0.0;
        enclosed(self.case, b)
            .into_iter()
            .filter(|&c| keep(c))
            .flat_map(|c| {
                let frags = &self.fragments[c];
                frags
                    .iter()
                    .enumerate()
                    .filter(move |(_, f)| f.page_index == page)
                    .map(move |(k, _)| (c, k, k + 1 == frags.len()))
            })
            .map(|(c, k, last)| {
                let margin = if last && !collapses {
                    self.margin_through(c)
                } else {
                    0.0
                };
                self.content_bottom(c, k) + margin
            })
            .reduce(f32::max)
    }

    /// `b`'s bottom margin, collapsed with its last in-flow child's when it
    /// is a block with no bottom padding or border.
    fn margin_through(&self, b: usize) -> f32 {
        let info = &self.case.boxes[b];
        let last = info
            .children
            .iter()
            .rev()
            .find(|&&c| self.case.boxes[c].kind != Kind::Float);
        match last {
            Some(&c) if info.kind == Kind::Block && info.bottom_edge == 0.0 => {
                info.margin_bottom.max(self.margin_through(c))
            }
            _ => info.margin_bottom,
        }
    }

    /// How far the split table cell `b`'s fragments add up past its height
    /// as laid out down to its content ([`Placed::natural_height`]; a split
    /// cell's line, not the cell, carries the row's stretch). Each fragment
    /// holding content counts down to its content ([`Self::children_end`]),
    /// the last one its bottom edge too; what of that content's margin and
    /// edge runs past the last one's end counts in the fragments after it,
    /// as far as they reach.
    fn split_cell_excess(&self, b: usize) -> f32 {
        let info = &self.case.boxes[b];
        let frags = &self.fragments[b];
        let with_content: Vec<(usize, f32, f32, f32)> = frags
            .iter()
            .enumerate()
            .filter_map(|(k, f)| {
                let (top, bottom) = (f.y.to_f32(), f.y.to_f32() + f.height.to_f32());
                let end = self.children_end(b, f.page_index)?.max(top);
                Some((k, top, bottom, end))
            })
            .collect();
        let Some(&(last, _, last_bottom, last_end)) = with_content.last() else {
            return -self.natural_height[b];
        };
        let content: f32 = with_content
            .iter()
            .map(|&(_, top, bottom, end)| end.min(bottom) - top)
            .sum();
        let edge_end = last_end + info.bottom_edge;
        let on_last = edge_end.min(last_bottom) - last_end.min(last_bottom);
        let later: f32 = frags[last + 1..].iter().map(|f| f.height.to_f32()).sum();
        let spilled = (edge_end - last_bottom).max(0.0).min(later);
        content + on_last + spilled - self.natural_height[b]
    }

    /// How far the container `b`'s fragments add up past its height. A
    /// split cell repeats its top edge by design (`fragment_table_cell`)
    /// above its content on each page it continues its content on, so its
    /// fragments add up to its height and those repeats
    /// ([`Self::split_cell_excess`]); short of that, it lost an edge or a
    /// margin.
    fn placed_excess(&self, b: usize) -> f32 {
        let info = &self.case.boxes[b];
        if info.kind == Kind::BodyCell && self.fragments[b].len() > 1 {
            let repeated = info.repeated_edge * self.content_pages(b).saturating_sub(1) as f32;
            self.split_cell_excess(b) - repeated
        } else {
            self.container_excess(b)
        }
    }

    /// The fragments of `b` whose page holds a fragment of a child of it.
    fn content_pages(&self, b: usize) -> usize {
        self.fragments[b]
            .iter()
            .filter(|f| self.children_end(b, f.page_index).is_some())
            .count()
    }

    /// How far `b`'s fragments add up past its layout height.
    fn excess(&self, b: usize) -> f32 {
        let total: f32 = self.fragments[b].iter().map(|f| f.height.to_f32()).sum();
        total - self.layout_height[b]
    }

    /// How far `b`'s fragments add up past its layout height, a container
    /// broken with a child pushed to the next page let run to the page foot
    /// (css-break-3 §4): a fragment before its last that reaches the foot
    /// counts down to the lowest of its children's fragments on its page.
    fn container_excess(&self, b: usize) -> f32 {
        let frags = &self.fragments[b];
        let total: f32 = frags
            .iter()
            .enumerate()
            .map(|(k, f)| self.counted_bottom(b, k) - f.y.to_f32())
            .sum();
        total - self.layout_height[b] - self.growth(b)
    }

    /// How much taller than their layout the lines of boxes side by side in
    /// `b`'s subtree are on the pages, its own lines' and those of the boxes
    /// in it ([`Self::line_growth`]).
    fn growth(&self, b: usize) -> f32 {
        let info = &self.case.boxes[b];
        let own = match info.kind {
            Kind::Flex => flex_lines(self.case, b)
                .iter()
                .map(|line| self.line_growth(line))
                .sum::<f32>(),
            Kind::Table => {
                let header: Vec<usize> = info
                    .children
                    .iter()
                    .copied()
                    .filter(|&row| {
                        self.case.boxes[row].kind == Kind::Row && self.case.boxes[row].in_header
                    })
                    .collect();
                table_lines(self.case, b)
                    .iter()
                    .map(|line| self.line_growth(line))
                    .sum::<f32>()
                    + self.repeated_band(b, &header)
            }
            _ => 0.0,
        };
        own + enclosed(self.case, b)
            .iter()
            .map(|&c| self.growth(c))
            .sum::<f32>()
    }

    /// The height the header rows `header` of the table `table` repeat on
    /// the pages after its first, with the table's top edge above them:
    /// each page of the table they are on carries their band once. A header
    /// placed twice on one page repeats on it once.
    fn repeated_band(&self, table: usize, header: &[usize]) -> f32 {
        let table_pages: BTreeSet<u32> =
            self.fragments[table].iter().map(|f| f.page_index).collect();
        // The band repeats from the table's top edge down
        // (`repeating_table_header`, `band_height_px`), so the edge repeats
        // with it on each page after the first a header is placed on.
        let band_pages: BTreeSet<u32> = header
            .iter()
            .flat_map(|&row| &self.case.boxes[row].children)
            .flat_map(|&c| self.fragments[c].iter().map(|f| f.page_index))
            .filter(|page| table_pages.contains(page))
            .collect();
        let edge = self.case.boxes[table].bottom_edge * band_pages.len().saturating_sub(1) as f32;
        edge + header
            .iter()
            .map(|&row| {
                let cells = &self.case.boxes[row].children;
                let pages: BTreeSet<u32> = cells
                    .iter()
                    .flat_map(|&c| self.fragments[c].iter().map(|f| f.page_index))
                    .filter(|page| table_pages.contains(page))
                    .collect();
                let height = cells
                    .iter()
                    .map(|&c| self.layout_height[c])
                    .fold(0.0_f32, f32::max);
                pages.len().saturating_sub(1) as f32 * height
            })
            .sum::<f32>()
    }

    /// How much taller than the tallest of its boxes' layouts the line of
    /// `items` is on the pages, each page's part down to the lowest of their
    /// fragments as counted there: boxes side by side that break at
    /// different points make their line taller than it was unbroken, as the
    /// line's part on each page is its tallest box's there.
    fn line_growth(&self, items: &[usize]) -> f32 {
        let pages: BTreeSet<u32> = items
            .iter()
            .flat_map(|&c| self.fragments[c].iter().map(|f| f.page_index))
            .collect();
        // Only a line whose boxes start on one page, side by side, grows
        // so: one placed after another's end is the defect this measures.
        let first_pages: BTreeSet<u32> = items
            .iter()
            .filter_map(|&c| self.fragments[c].first().map(|f| f.page_index))
            .collect();
        if pages.len() < 2 || first_pages.len() > 1 {
            return 0.0;
        }
        let paged: f32 = pages
            .iter()
            .map(|&page| {
                let on_page = || {
                    items.iter().flat_map(move |&c| {
                        self.fragments[c]
                            .iter()
                            .enumerate()
                            .filter(move |(_, f)| f.page_index == page)
                            .map(move |(k, f)| (c, k, f.y.to_f32()))
                    })
                };
                let top = on_page().map(|(_, _, y)| y).fold(f32::INFINITY, f32::min);
                let bottom = on_page()
                    .map(|(c, k, _)| self.counted_bottom(c, k))
                    .fold(top, f32::max);
                bottom - top
            })
            .sum();
        let layout = items
            .iter()
            .map(|&c| self.layout_height[c])
            .fold(0.0_f32, f32::max);
        (paged - layout).max(0.0)
    }

    /// Where `b`'s fragment `k` ends as [`Self::container_excess`] counts
    /// it: a container's fragment before its last that reaches the page
    /// foot ends at the lowest of its children's fragments on its page,
    /// each counted so in turn (css-break-3 §4). So does every fragment of
    /// a split table cell or flex item, its last too, below its bottom
    /// padding: one stretched to its line's height goes on to the line's
    /// end below its own content, and a line that broke ends lower than its
    /// layout when a box of it left a gap at a page foot. Every other
    /// fragment ends at its bottom.
    fn counted_bottom(&self, b: usize, k: usize) -> f32 {
        let frags = &self.fragments[b];
        let f = &frags[k];
        let (top, bottom) = (f.y.to_f32(), f.y.to_f32() + f.height.to_f32());
        let kind = self.case.boxes[b].kind;
        let container = matches!(
            kind,
            Kind::Block | Kind::Table | Kind::BodyCell | Kind::Flex | Kind::Item
        );
        let in_line = frags.len() > 1 && matches!(kind, Kind::BodyCell | Kind::Item);
        if !container || (!in_line && (k + 1 == frags.len() || bottom < self.page_height - EPS)) {
            return bottom;
        }
        let content = if in_line {
            // A split cell or item ends its content where its children's do,
            // their edges and its last child's bottom margin included.
            self.children_end(b, f.page_index)
                .map_or(top, |end| end.min(bottom).max(top))
        } else {
            enclosed(self.case, b)
                .iter()
                .flat_map(|&c| {
                    self.fragments[c]
                        .iter()
                        .enumerate()
                        .filter(|(_, cf)| cf.page_index == f.page_index)
                        .map(move |(ck, _)| (c, ck))
                })
                .map(|(c, ck)| self.counted_bottom(c, ck).min(bottom))
                .fold(top, f32::max)
        };
        if in_line && k + 1 == frags.len() {
            (content + self.case.boxes[b].repeated_edge).min(bottom)
        } else {
            content
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
    let mut placed = Placed {
        case,
        page_height,
        fragments,
        line_boundaries,
        layout_height,
        natural_height: vec![0.0; case.boxes.len()],
        past,
    };
    placed.natural_height = (0..case.boxes.len())
        .map(|b| {
            let info = &case.boxes[b];
            if info.kind != Kind::BodyCell {
                return 0.0;
            }
            let last = info
                .children
                .iter()
                .rev()
                .find(|&&c| case.boxes[c].kind != Kind::Float);
            // A first child that starts on a later page than the cell has
            // its top margin truncated at the break (css-break-3 §5.2).
            let first_page = placed.fragments[b].first().map(|f| f.page_index);
            let truncated = info
                .children
                .iter()
                .find(|&&c| case.boxes[c].kind != Kind::Float)
                .filter(|&&c| placed.fragments[c].first().map(|f| f.page_index) != first_page)
                .map_or(0.0, |&c| case.boxes[c].margin_top);
            info.children
                .iter()
                .filter_map(|&c| {
                    let layout = &doc.get_node(ids[c]?)?.final_layout;
                    let margin = if Some(&c) == last {
                        placed.margin_through(c)
                    } else {
                        0.0
                    };
                    Some(layout.location.y + layout.size.height + margin)
                })
                .fold(info.repeated_edge, f32::max)
                + info.bottom_edge
                - truncated
        })
        .collect();
    let mut out = Vec::new();
    let mut push = |invariant, b: usize, page: Option<u32>, amount: Option<f32>, what: String| {
        out.push(Violation {
            invariant,
            box_id: case.boxes[b].id.clone(),
            page,
            amount,
            what,
        });
    };

    for (b, info) in case.boxes.iter().enumerate() {
        let frags = &placed.fragments[b];

        // No fragment ends past the page foot, except a monolithic box
        // taller than a page.
        for &page in &placed.past[b] {
            push(
                Invariant::PastFoot,
                b,
                Some(page),
                None,
                format!("{} ({:?}) ends past the page foot", info.id, info.kind),
            );
        }

        // No fragment starts above the page top.
        for f in frags.iter().filter(|f| f.y.to_f32() < -EPS) {
            push(
                Invariant::PlacedOnce,
                b,
                Some(f.page_index),
                None,
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
                None,
                format!("{} ({:?}) is not placed", info.id, info.kind),
            );
        }
        if matches!(
            info.kind,
            Kind::Leaf | Kind::Image | Kind::Float | Kind::Caption | Kind::Para
        ) {
            for pair in pages.windows(2) {
                if pair[1] <= pair[0] {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pair[1]),
                        None,
                        format!("{} is placed twice or out of order: {pages:?}", info.id),
                    );
                } else if pair[1] > pair[0] + 1 {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pair[0] + 1),
                        None,
                        format!("{} skips a page: {pages:?}", info.id),
                    );
                }
            }
        }
        match info.kind {
            Kind::Leaf | Kind::Image | Kind::Float | Kind::Caption if !frags.is_empty() => {
                let excess = placed.excess(b);
                if excess.abs() > EPS * frags.len() as f32 {
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pages[0]),
                        Some(excess),
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
                        None,
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
                if !frags.is_empty() =>
            {
                let repeated =
                    info.repeated_edge * placed.content_pages(b).saturating_sub(1) as f32;
                let excess = placed.placed_excess(b);
                let tolerance = EPS * frags.len() as f32;
                if excess > tolerance || (info.kind == Kind::BodyCell && excess < -tolerance) {
                    // Where a split cell's last fragment ends against its
                    // content, margin, and bottom edge, for whoever reads it.
                    let last = frags.len() - 1;
                    let end = placed
                        .children_end(b, frags[last].page_index)
                        .map_or(0.0, |end| end + info.bottom_edge);
                    let bottom = frags[last].y.to_f32() + frags[last].height.to_f32();
                    push(
                        Invariant::PlacedOnce,
                        b,
                        Some(pages[0]),
                        Some(excess),
                        format!(
                            "{} ({:?})'s fragments add up to {excess} px past its height \
                             and the {repeated} px of top edge it repeats; its last \
                             fragment ends at {bottom} on page {}, its content with its \
                             bottom edge at {end}",
                            info.id, info.kind, frags[last].page_index
                        ),
                    );
                }
            }
            _ => {}
        }

        // In block flow, document order matches (page, y) order; a float
        // is out of flow, beside the boxes after it.
        let flow: Vec<usize> = if b == 0 {
            (0..case.boxes.len())
                .filter(|&k| case.boxes[k].parent.is_none())
                .collect()
        } else if matches!(info.kind, Kind::Block | Kind::BodyCell | Kind::Item) {
            info.children.clone()
        } else {
            Vec::new()
        };
        let flow: Vec<usize> = flow
            .into_iter()
            .filter(|&k| case.boxes[k].kind != Kind::Float)
            .collect();
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
                push(
                    Invariant::DocumentOrder,
                    c,
                    Some(first.page_index),
                    None,
                    format!(
                        "{} starts at {c_start:?}, before {} ends at {a_end:?}",
                        case.boxes[c].id, case.boxes[a].id
                    ),
                );
            }
        }

        // Between the body rows of a table, and the lines of a flex or grid
        // container, document order matches (page, y) order: a line starts
        // where the line before it ends, or later; a cell spanning the row
        // below ends with that row.
        let ordered: Vec<(usize, Vec<usize>)> = match info.kind {
            Kind::Table => body_rows(case, b)
                .map(|row| (row, case.boxes[row].children.clone()))
                .collect(),
            Kind::Flex => flex_lines(case, b)
                .into_iter()
                .map(|line| (line[0], line))
                .collect(),
            _ => Vec::new(),
        };
        let start = |line: &[usize]| {
            line.iter()
                .filter_map(|&c| placed.fragments[c].first())
                .map(|f| (f.page_index, f.y.to_f32()))
                .reduce(|a, b| if before(b, a) { b } else { a })
        };
        let end = |line: &[usize]| {
            line.iter()
                .filter(|&&c| case.boxes[c].rowspan == 1)
                .filter_map(|&c| placed.fragments[c].last())
                .map(|f| (f.page_index, f.y.to_f32() + f.height.to_f32()))
                .reduce(|a, b| if before(a, b) { b } else { a })
        };
        for pair in ordered.windows(2) {
            let ((a, a_line), (c, c_line)) = (&pair[0], &pair[1]);
            let (Some(a_end), Some(c_start)) = (end(a_line), start(c_line)) else {
                continue;
            };
            if before(c_start, a_end) {
                push(
                    Invariant::DocumentOrder,
                    *c,
                    Some(c_start.0),
                    None,
                    format!(
                        "the line of {} starts at {c_start:?}, before the line of {} ends at {a_end:?}",
                        case.boxes[*c].id, case.boxes[*a].id
                    ),
                );
            }
        }

        // On each page, a parent's fragment encloses its children's; a float
        // may hang out of a block that is no formatting context root.
        for child in enclosed(case, b) {
            if case.boxes[child].kind == Kind::Float && info.kind == Kind::Block {
                continue;
            }
            for cf in &placed.fragments[child] {
                // A child running past the foot is reported so already; the
                // part on its page lies inside a parent fragment that
                // reaches the foot.
                let inside = frags.iter().any(|pf| {
                    let bottom = pf.y.to_f32() + pf.height.to_f32();
                    pf.page_index == cf.page_index
                        && pf.y.to_f32() <= cf.y.to_f32() + EPS
                        && (cf.y.to_f32() + cf.height.to_f32() <= bottom + EPS
                            || bottom >= page_height - EPS)
                });
                if inside {
                    continue;
                }
                push(
                    Invariant::Encloses,
                    child,
                    Some(cf.page_index),
                    None,
                    format!(
                        "{} ({:?}) at y {} h {} is outside {} ({:?})",
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
    for (table, page) in placed.alone_headers() {
        push(
            Invariant::HeaderAlone,
            table,
            Some(page),
            None,
            format!("{}'s header is alone", case.boxes[table].id),
        );
    }

    // A line's item stretched to the line ends with it on each page.
    for (item, page, what) in placed.past_line_end() {
        push(Invariant::EndsWithLine, item, Some(page), None, what);
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

/// What one case reaches: the breaks the walker takes at its break
/// opportunities, as counted by [`reach`].
#[derive(Clone, Copy, Debug, Default)]
struct Reach {
    /// A boundary between siblings in flow, rows, or lines carries
    /// `break-before` or `break-after: avoid`.
    avoid: bool,
    /// The walker breaks at such a boundary.
    avoid_taken: bool,
    /// The walker breaks between two body rows of a table.
    row_break: bool,
    /// The walker breaks inside a body row.
    row_split: bool,
}

/// What `seed`'s case reaches: each boundary between boxes in flow (the
/// roots, a block's, a cell's, or an item's children but floats), between
/// body rows, and between the lines of a flex or grid container, and
/// whether the walker breaks there (the later starts on a later page than
/// the earlier ends on).
fn reach(seed: u64) -> Reach {
    let (roots, page) = case_of(seed);
    let case = Case::new(&roots);
    let (doc, geometry) = layout(&case, f32::from(page));
    let ids = node_ids(&doc, &case);
    let pages = |boxes: &[usize]| -> Vec<u32> {
        boxes
            .iter()
            .filter(|&&b| case.boxes[b].rowspan == 1)
            .filter_map(|&b| ids[b].and_then(|id| geometry.get(&id)))
            .flat_map(|g| g.fragments.iter().map(|f| f.page_index))
            .collect()
    };
    let flow = |boxes: Vec<usize>| -> Vec<(usize, Vec<usize>)> {
        boxes
            .into_iter()
            .filter(|&b| case.boxes[b].kind != Kind::Float)
            .map(|b| (b, vec![b]))
            .collect()
    };
    let mut sequences = vec![flow(
        (0..case.boxes.len())
            .filter(|&b| case.boxes[b].parent.is_none())
            .collect(),
    )];
    let mut reach = Reach::default();
    for (b, info) in case.boxes.iter().enumerate() {
        match info.kind {
            Kind::Block | Kind::BodyCell | Kind::Item => {
                sequences.push(flow(info.children.clone()))
            }
            Kind::Table => sequences.push(
                body_rows(&case, b)
                    .map(|row| (row, case.boxes[row].children.clone()))
                    .collect(),
            ),
            Kind::Flex => sequences.push(
                flex_lines(&case, b)
                    .into_iter()
                    .map(|line| (line[0], line))
                    .collect(),
            ),
            Kind::Row if !info.in_header => {
                let rows = pages(&info.children);
                reach.row_split |= rows.iter().min() != rows.iter().max();
            }
            _ => {}
        }
    }
    for sequence in sequences {
        for pair in sequence.windows(2) {
            let ((a, a_boxes), (c, c_boxes)) = (&pair[0], &pair[1]);
            let avoid =
                case.boxes[*a].breaks.1 == Break::Avoid || case.boxes[*c].breaks.0 == Break::Avoid;
            let taken = matches!(
                (pages(a_boxes).iter().max(), pages(c_boxes).iter().min()),
                (Some(a_end), Some(c_start)) if c_start > a_end
            );
            reach.avoid |= avoid;
            reach.avoid_taken |= avoid && taken;
            reach.row_break |= taken && case.boxes[*a].kind == Kind::Row;
        }
    }
    reach
}

/// One seed's case, its violations, and for each invariant the smallest
/// case that still breaks it, with that case's geometry.
fn describe(seed: u64) -> String {
    let (roots, page) = case_of(seed);
    let case = Case::new(&roots);
    let found = check(&case, f32::from(page));
    let mut text = format!("seed {seed}, page {page} px\n{}\n", case.html);
    for v in &found {
        let _ = writeln!(text, "  {}: {}", v.key(), v.what);
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
                let _ = writeln!(text, "  {}: {}", v.key(), v.what);
            }
        }
        let (doc, geometry) = layout(&case, f32::from(page));
        text.push_str(&geometry_text(&geometry, &doc));
    }
    text
}

/// The golden's lines: `<seed> <invariant> <box> <page>`, then
/// `  # <note>` when the line has one, as `(seed, key)` with the note.
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
         # placed nowhere), then, where it was traced, its cause, written by\n\
         # hand. Rewrite with FRAGMENT_INVARIANT_UPDATE=1, which keeps the note\n\
         # of each line it keeps, and review every line.\n",
    );
    for ((seed, key), note) in found {
        let _ = writeln!(text, "{}", golden_line(*seed, key, note));
    }
    text
}

/// One line of the golden, its note left out when it has none.
fn golden_line(seed: u64, key: &str, note: &str) -> String {
    if note.is_empty() {
        format!("{seed} {key}")
    } else {
        format!("{seed} {key}  # {note}")
    }
}

/// A 10 px wide fragment on page `page_index`, for the checks' own tests.
fn fragment(page_index: u32, y: f32, height: f32) -> Fragment {
    Fragment {
        page_index,
        x: 0.0_f32.as_px(),
        y: y.as_px(),
        width: 10.0_f32.as_px(),
        height: height.as_px(),
    }
}

/// An item stretched past its line's end on its page is flagged, on a page
/// the line continues past as on its last, and one stretched to the line's
/// end, or to the foot of a page the line continues past, is not: a flex
/// row of a 150 px leaf, broken 10 px above the foot of a 100 px page, beside
/// a 30 px one.
#[test]
fn an_item_stretched_past_its_line_s_end_is_flagged() {
    let leaf = |height| Gen::Leaf {
        style: BoxStyle::default(),
        height,
    };
    let case = Case::new(&[flex_row(vec![leaf(150), leaf(30)])]);
    // b0 the flex row, b1 and b3 its items, b2 and b4 their leaves.
    let flagged = |short_item: Vec<Fragment>| {
        let mut fragments = vec![Vec::new(); case.boxes.len()];
        fragments[0] = vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 60.0)];
        fragments[1] = vec![fragment(0, 0.0, 90.0), fragment(1, 0.0, 60.0)];
        fragments[2] = fragments[1].clone();
        fragments[3] = short_item;
        fragments[4] = vec![fragment(0, 0.0, 30.0)];
        Placed {
            case: &case,
            page_height: 100.0,
            fragments,
            line_boundaries: vec![Vec::new(); case.boxes.len()],
            layout_height: vec![150.0, 150.0, 150.0, 150.0, 30.0],
            natural_height: vec![0.0; case.boxes.len()],
            past: vec![BTreeSet::new(); case.boxes.len()],
        }
        .past_line_end()
        .into_iter()
        .map(|(item, page, _)| (item, page))
        .collect::<Vec<_>>()
    };
    let none: [(usize, u32); 0] = [];
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 90.0)]),
        none,
        "to the line's end"
    );
    assert_eq!(flagged(vec![fragment(0, 0.0, 100.0)]), none, "to the foot");
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 90.0), fragment(1, 0.0, 60.0)]),
        none,
        "on to the line's end on its last page"
    );
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 95.0)]),
        [(3, 0)],
        "past the line's end on a page it continues past"
    );
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 90.0), fragment(1, 0.0, 70.0)]),
        [(3, 1)],
        "past the line's end on its last page"
    );
}

/// The items of `case` that [`Placed::past_line_end`] flags on pages
/// `page_height` px tall with `fragments`, as `(box, page)`.
fn flagged_items(
    case: &Case,
    page_height: f32,
    fragments: Vec<Vec<Fragment>>,
) -> Vec<(usize, u32)> {
    Placed {
        case,
        page_height,
        fragments,
        line_boundaries: vec![Vec::new(); case.boxes.len()],
        layout_height: vec![0.0; case.boxes.len()],
        natural_height: vec![0.0; case.boxes.len()],
        past: vec![BTreeSet::new(); case.boxes.len()],
    }
    .past_line_end()
    .into_iter()
    .map(|(item, page, _)| (item, page))
    .collect()
}

/// A leaf of `height` px with `style`.
fn styled_leaf(style: BoxStyle, height: u16) -> Gen {
    Gen::Leaf { style, height }
}

/// A paragraph of `lines` 20 px lines with `style` and no line counts.
fn para(style: BoxStyle, lines: u8) -> Gen {
    Gen::Para {
        style,
        lines,
        line_height: 20,
        counts: None,
    }
}

/// A flex row of `items`.
fn flex_row(items: Vec<Gen>) -> Gen {
    Gen::Flex {
        style: BoxStyle::default(),
        mode: FlexMode::Row,
        items: items
            .into_iter()
            .map(|item| (Breaks::default(), item))
            .collect(),
    }
}

/// An item emptied of its content, its fragment stretched to the foot of
/// a page its line continues past, is flagged when no content of the line
/// crosses that page's break: a flex row of a 40 px leaf beside a 60 px leaf
/// moved whole to the next page on 100 px pages.
#[test]
fn an_empty_item_fragment_stretched_to_the_foot_is_flagged() {
    let case = Case::new(&[flex_row(vec![
        styled_leaf(BoxStyle::default(), 40),
        styled_leaf(BoxStyle::default(), 60),
    ])]);
    // b0 the row, b1 and b3 its items, b2 and b4 their leaves.
    let mut fragments = vec![Vec::new(); case.boxes.len()];
    fragments[1] = vec![fragment(0, 0.0, 40.0)];
    fragments[2] = vec![fragment(0, 0.0, 40.0)];
    fragments[3] = vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 60.0)];
    fragments[4] = vec![fragment(1, 0.0, 60.0)];
    assert_eq!(flagged_items(&case, 100.0, fragments), [(3, 0)]);
}

/// The row below a cell spanning it is one line with the spanning cell's
/// row: its cell may run on to the foot of a page while the spanning cell's
/// content crosses that page's break, and not once the spanning cell ends
/// above it. A table of a spanning cell and a 50 px cell over one cell, on
/// 100 px pages.
#[test]
fn a_cell_below_a_spanning_cell_breaks_with_the_spanning_cell() {
    let leaf = |height| styled_leaf(BoxStyle::default(), height);
    let case = Case::new(&[Gen::Table {
        style: BoxStyle::default(),
        head: Head::None,
        caption: false,
        counts: None,
        row_counts: None,
        cell_padding: 0,
        body: Breaks::default(),
        rows: vec![
            (Breaks::default(), vec![leaf(130), leaf(50)]),
            (Breaks::default(), vec![leaf(30)]),
        ],
        span: Some(0),
    }]);
    // b0 the table, b1 the first row, b2 the spanning cell and b3 its leaf,
    // b4 and b5 the cell beside it, b6 the second row, b7 and b8 its cell.
    let flagged = |spanning: Vec<Fragment>| {
        let mut fragments = vec![Vec::new(); case.boxes.len()];
        fragments[2] = spanning.clone();
        fragments[3] = spanning;
        fragments[4] = vec![fragment(0, 0.0, 50.0)];
        fragments[5] = vec![fragment(0, 0.0, 50.0)];
        fragments[7] = vec![fragment(0, 50.0, 50.0), fragment(1, 0.0, 30.0)];
        fragments[8] = vec![fragment(1, 0.0, 30.0)];
        flagged_items(&case, 100.0, fragments)
    };
    let none: [(usize, u32); 0] = [];
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 30.0)]),
        none,
        "the spanning cell crosses the break"
    );
    assert_eq!(
        flagged(vec![fragment(0, 0.0, 60.0)]),
        [(7, 0)],
        "the spanning cell ends above the break"
    );
}

/// A bottom margin collapses through the blocks with no bottom edge that
/// end with it, and counts once: an item holding two nested blocks that end
/// with a leaf of a 17 px bottom margin ends with its line 17 px below the
/// leaf, beside a 5 px leaf, on 100 px pages.
#[test]
fn a_bottom_margin_collapsing_through_its_blocks_counts_once() {
    let margined = BoxStyle {
        margin_bottom: 17,
        ..BoxStyle::default()
    };
    let case = Case::new(&[flex_row(vec![
        styled_leaf(BoxStyle::default(), 5),
        Gen::Block {
            style: BoxStyle::default(),
            counts: None,
            children: vec![Gen::Block {
                style: BoxStyle::default(),
                counts: None,
                children: vec![styled_leaf(margined, 50)],
            }],
        },
    ])]);
    // b0 the row, b1 an item and b2 its leaf, b3 the other item, b4 and b5
    // the blocks in it, b6 the leaf.
    let flagged = |item_end: f32| {
        let mut fragments = vec![Vec::new(); case.boxes.len()];
        fragments[1] = vec![fragment(0, 0.0, item_end)];
        fragments[2] = vec![fragment(0, 0.0, 5.0)];
        fragments[3] = vec![fragment(0, 0.0, item_end)];
        for frags in &mut fragments[4..=6] {
            *frags = vec![fragment(0, 0.0, 50.0)];
        }
        flagged_items(&case, 100.0, fragments)
    };
    let none: [(usize, u32); 0] = [];
    assert_eq!(flagged(67.0), none, "to the margin's end");
    assert_eq!(flagged(85.0), [(1, 0), (3, 0)], "past it");
}

/// A paragraph's fragments, measured by its lines, leave out its padding
/// and border, at any depth: an item holding a block that ends with a split
/// paragraph padded 12 px, with a 17 px bottom margin, ends with its line
/// 12 + 17 px below the paragraph's last line, on 150 px pages.
#[test]
fn a_nested_paragraph_s_padding_counts_in_its_line() {
    let padded = BoxStyle {
        padding: 12,
        margin_bottom: 17,
        ..BoxStyle::default()
    };
    let case = Case::new(&[flex_row(vec![
        styled_leaf(BoxStyle::default(), 5),
        Gen::Block {
            style: BoxStyle::default(),
            counts: None,
            children: vec![para(padded, 9)],
        },
    ])]);
    // b0 the row, b1 an item and b2 its leaf, b3 the other item, b4 the
    // block, b5 the paragraph: 4 lines on page 0, 5 on page 1.
    let mut fragments = vec![Vec::new(); case.boxes.len()];
    fragments[1] = vec![fragment(0, 0.0, 150.0), fragment(1, 0.0, 129.0)];
    fragments[2] = vec![fragment(0, 0.0, 5.0)];
    fragments[3] = fragments[1].clone();
    fragments[4] = vec![fragment(0, 0.0, 80.0), fragment(1, 0.0, 100.0)];
    fragments[5] = fragments[4].clone();
    let none: [(usize, u32); 0] = [];
    assert_eq!(flagged_items(&case, 150.0, fragments), none);
}

/// A split paragraph's top padding and border lie on its first page only:
/// on its last page its line ends below its last line by its bottom edge
/// alone. A flex row of a 5 px leaf beside a paragraph padded 10 px, 3 of
/// its lines on the second of 100 px pages, the items 5 px past that.
#[test]
fn a_split_paragraph_s_top_edge_stays_on_its_first_page() {
    let padded = BoxStyle {
        padding: 10,
        ..BoxStyle::default()
    };
    let case = Case::new(&[flex_row(vec![
        styled_leaf(BoxStyle::default(), 5),
        para(padded, 7),
    ])]);
    // b0 the row, b1 an item and b2 its leaf, b3 the other item, b4 the
    // paragraph: 4 lines below its top edge on page 0, 3 on page 1.
    let mut fragments = vec![Vec::new(); case.boxes.len()];
    fragments[1] = vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 75.0)];
    fragments[2] = vec![fragment(0, 0.0, 5.0)];
    fragments[3] = fragments[1].clone();
    fragments[4] = vec![fragment(0, 0.0, 80.0), fragment(1, 0.0, 60.0)];
    assert_eq!(flagged_items(&case, 100.0, fragments), [(1, 1), (3, 1)]);
}

/// A split cell repeats its top edge above its content on each page it
/// continues on by design, and ends below its bottom edge: its fragments
/// then add up to no excess, and one that ends without its bottom edge
/// falls that much short. A 150 px leaf in a cell padded 10 px, split on
/// 100 px pages.
#[test]
fn a_split_cell_counts_its_repeated_top_edge_and_its_lost_bottom_edge() {
    let case = Case::new(&[Gen::Table {
        style: BoxStyle::default(),
        head: Head::None,
        caption: false,
        counts: None,
        row_counts: None,
        cell_padding: 10,
        body: Breaks::default(),
        rows: vec![(
            Breaks::default(),
            vec![Gen::Leaf {
                style: BoxStyle::default(),
                height: 150,
            }],
        )],
        span: None,
    }]);
    // b0 the table, b1 the row, b2 the cell, b3 its leaf.
    let excess = |last_cell: f32| {
        let mut fragments = vec![Vec::new(); case.boxes.len()];
        fragments[2] = vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, last_cell)];
        fragments[3] = vec![fragment(0, 10.0, 90.0), fragment(1, 10.0, 60.0)];
        let mut natural_height = vec![0.0; case.boxes.len()];
        natural_height[2] = 170.0;
        Placed {
            case: &case,
            page_height: 100.0,
            fragments,
            line_boundaries: vec![Vec::new(); case.boxes.len()],
            layout_height: vec![170.0; case.boxes.len()],
            natural_height,
            past: vec![BTreeSet::new(); case.boxes.len()],
        }
        .placed_excess(2)
    };
    assert!(excess(80.0).abs() < EPS, "{}", excess(80.0));
    assert!((excess(70.0) + 10.0).abs() < EPS, "{}", excess(70.0));
}

/// A table's repeated header band reaches up to its top edge, which
/// repeats with it on each page after the first: a 120 px leaf in a table
/// padded 5 px under a 20 px header, split on 100 px pages, adds up to no
/// excess.
#[test]
fn a_repeated_header_band_carries_the_table_s_top_edge() {
    let case = Case::new(&[Gen::Table {
        style: BoxStyle {
            padding: 5,
            ..BoxStyle::default()
        },
        head: Head::Plain,
        caption: false,
        counts: None,
        row_counts: None,
        cell_padding: 0,
        body: Breaks::default(),
        rows: vec![(
            Breaks::default(),
            vec![Gen::Leaf {
                style: BoxStyle::default(),
                height: 120,
            }],
        )],
        span: None,
    }]);
    // b0 the table, b1 the header row, b2 its cell, b3 the body row, b4
    // its cell, b5 the leaf.
    let mut fragments = vec![Vec::new(); case.boxes.len()];
    fragments[0] = vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 75.0)];
    fragments[2] = vec![fragment(0, 5.0, 20.0), fragment(1, 5.0, 20.0)];
    fragments[4] = vec![fragment(0, 25.0, 75.0), fragment(1, 25.0, 45.0)];
    fragments[5] = fragments[4].clone();
    let excess = Placed {
        case: &case,
        page_height: 100.0,
        fragments,
        line_boundaries: vec![Vec::new(); case.boxes.len()],
        layout_height: vec![150.0, 0.0, 20.0, 0.0, 120.0, 120.0],
        natural_height: vec![0.0; case.boxes.len()],
        past: vec![BTreeSet::new(); case.boxes.len()],
    }
    .container_excess(0);
    assert!(excess.abs() < EPS, "{excess}");
}

/// A body row whose cells place nothing on a page is no body row there: a
/// header beside its empty cell is alone. A 20 px header over a cell whose
/// 60 px leaf is on the second of 100 px pages, the cell's empty fragment
/// on the first.
#[test]
fn a_header_beside_an_empty_body_cell_is_alone() {
    let case = Case::new(&[Gen::Table {
        style: BoxStyle::default(),
        head: Head::Plain,
        caption: false,
        counts: None,
        row_counts: None,
        cell_padding: 0,
        body: Breaks::default(),
        rows: vec![(
            Breaks::default(),
            vec![Gen::Leaf {
                style: BoxStyle::default(),
                height: 60,
            }],
        )],
        span: None,
    }]);
    // b0 the table, b1 the header row, b2 its cell, b3 the body row, b4
    // its cell, b5 the leaf.
    let mut fragments = vec![Vec::new(); case.boxes.len()];
    fragments[0] = vec![fragment(0, 40.0, 60.0), fragment(1, 0.0, 80.0)];
    fragments[2] = vec![fragment(0, 40.0, 20.0), fragment(1, 0.0, 20.0)];
    fragments[4] = vec![fragment(0, 60.0, 40.0), fragment(1, 20.0, 60.0)];
    fragments[5] = vec![fragment(1, 20.0, 60.0)];
    let alone = Placed {
        case: &case,
        page_height: 100.0,
        fragments,
        line_boundaries: vec![Vec::new(); case.boxes.len()],
        layout_height: vec![0.0; case.boxes.len()],
        natural_height: vec![0.0; case.boxes.len()],
        past: vec![BTreeSet::new(); case.boxes.len()],
    }
    .alone_headers();
    assert_eq!(alone, [(0, 0)]);
}

/// A table's repeated header is counted once a page, however many times it
/// is placed there: a header doubled on the table's second page leaves the
/// table's excess one band past its repeat, as a placement the checks flag.
#[test]
fn a_header_placed_twice_on_one_page_is_counted_once() {
    let case = Case::new(&[Gen::Table {
        style: BoxStyle::default(),
        head: Head::Plain,
        caption: false,
        counts: None,
        row_counts: None,
        cell_padding: 0,
        body: Breaks::default(),
        rows: vec![(
            Breaks::default(),
            vec![Gen::Leaf {
                style: BoxStyle::default(),
                height: 120,
            }],
        )],
        span: None,
    }]);
    // b0 the table, b1 the header row, b2 its cell, b3 the body row, b4
    // its cell, b5 the leaf: a 140 px table on 200 px pages.
    let placed = |header: Vec<Fragment>, table: Vec<Fragment>| {
        let mut fragments = vec![Vec::new(); case.boxes.len()];
        fragments[0] = table;
        fragments[2] = header;
        fragments[4] = vec![fragment(0, 20.0, 80.0), fragment(1, 20.0, 40.0)];
        fragments[5] = fragments[4].clone();
        let layout_height = vec![140.0, 0.0, 20.0, 0.0, 120.0, 120.0];
        Placed {
            case: &case,
            page_height: 200.0,
            fragments,
            line_boundaries: vec![Vec::new(); case.boxes.len()],
            layout_height,
            natural_height: vec![0.0; case.boxes.len()],
            past: vec![BTreeSet::new(); case.boxes.len()],
        }
        .container_excess(0)
    };
    let once = vec![fragment(0, 0.0, 20.0), fragment(1, 0.0, 20.0)];
    let twice = vec![
        fragment(0, 0.0, 20.0),
        fragment(1, 0.0, 20.0),
        fragment(1, 20.0, 20.0),
    ];
    assert!(
        placed(once, vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 60.0)]).abs() < EPS,
        "one repeat, one band"
    );
    assert!(
        placed(twice, vec![fragment(0, 0.0, 100.0), fragment(1, 0.0, 80.0)]) > EPS,
        "a doubled header excuses no second band"
    );
}

#[test]
fn generated_box_trees_keep_the_fragmenter_invariants() {
    if let Some(seed) = env_u64("FRAGMENT_INVARIANT_SEED") {
        println!("{}", describe(seed));
        return;
    }
    if std::env::var_os("FRAGMENT_INVARIANT_REACH").is_some() {
        let reached: Vec<Reach> = (0..CASES).map(reach).collect();
        let count = |f: fn(&Reach) -> bool| reached.iter().filter(|r| f(r)).count();
        println!(
            "of {CASES} seeds: an avoid boundary in {}, a break taken at one in {}, \
             a break between body rows in {}, a body row split in {}",
            count(|r| r.avoid),
            count(|r| r.avoid_taken),
            count(|r| r.row_break),
            count(|r| r.row_split)
        );
        return;
    }
    let cases = env_u64("FRAGMENT_INVARIANT_CASES").unwrap_or(CASES);
    let known: BTreeMap<(u64, String), String> = golden_lines(KNOWN)
        .into_iter()
        .filter(|((seed, _), _)| *seed < cases)
        .collect();
    // Each violation found, with the golden's note of it.
    let mut found = BTreeMap::new();
    for seed in 0..cases {
        for v in violations(seed) {
            let line = (seed, v.key());
            let note = known.get(&line).cloned().unwrap_or_default();
            found.insert(line, note);
        }
    }
    let mut report = String::new();
    for ((seed, key), note) in &found {
        if !known.contains_key(&(*seed, key.clone())) {
            let _ = writeln!(report, "added: {}", golden_line(*seed, key, note));
        }
    }
    for ((seed, key), note) in &known {
        if !found.contains_key(&(*seed, key.clone())) {
            let _ = writeln!(report, "removed: {}", golden_line(*seed, key, note));
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
