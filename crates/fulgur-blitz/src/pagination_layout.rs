//! Taffy-hooked block-level paginator (fulgur-4cbc).
//!
//! Sibling of [`crate::multicol_layout`]. The multicol module proves the
//! `LayoutPartialTree` wrapper pattern works for routing one CSS feature
//! through fulgur-owned layout while leaving the rest to `BaseDocument`;
//! this module applies the same idiom to page fragmentation.
//!
//! # Status: production-wired, observational consumer
//!
//! [`run_pass_with_break_styles`] is invoked once per render from
//! `engine.rs` after `multicol_layout::run_pass`. The production path
//! skips `taffy::compute_root_layout` and calls
//! [`fragment_pagination_root`] directly: it walks the body's direct
//! block children's existing `final_layout` — descending into Parley
//! line metrics for inline roots — and records the would-be page
//! geometry in a `PaginationGeometryTable`. Re-driving Taffy on body
//! re-stores every descendant's layout fields and introduces sub-pixel
//! floating-point drift that breaks `examples_determinism`'s byte-wise
//! PDF comparison; see [`PaginationLayoutTree::drive_taffy_root_layout`]
//! for the full root cause.
//!
//! The wrapper's `LayoutPartialTree` / `RoundTree` / `CacheTree` /
//! `TraversePartialTree` impls (which dispatch body's layout into
//! [`compute_pagination_layout`] via `taffy::compute_root_layout`) are
//! kept compile-time live as scaffolding for a future per-strip
//! constrained variant; the `taffy_driven_dispatch_matches_direct_walk`
//! test exercises them at runtime and asserts geometry parity with the
//! production direct walk.
//!
//! Today the engine drops the returned table (`let _pagination_geometry
//! = …`). Follow-up work will capture the table on `ConvertContext`
//! and wire downstream consumers (counter / string-set replacement,
//! per-page repetition redesign, …).
//!
//! # Coverage
//!
//! The wrapper is currently exercised against the body subtree only.
//! Anything nested inside body's direct children continues to use
//! Blitz's normal layout dispatch, and the fragmenter post-walks
//! `final_layout` rather than re-issuing per-strip
//! `compute_child_layout` calls. The fulgur-ik6o probe established
//! that constraining `available_space.height` does not change Taffy's
//! block-layout output — see
//! `docs/plans/2026-04-28-pagination-layout-spike.md`.
//!
//! # Features wired today
//!
//! - Block-level fragmentation against `page_height_px`
//!   ([`fragment_pagination_root`]).
//! - Inline-aware split at Parley line boundaries
//!   ([`fragment_inline_root`], reads `inline_layout_data` populated by
//!   `resolve()`).
//! - `break-before` / `break-after` / `break-inside: avoid` from the
//!   shared [`crate::column_css::ColumnStyleTable`] side-table.
//!
//! # Production extension points
//!
//! [`collect_string_set_states`] and [`implied_page_count`] are `pub`
//! for use by `render_v2` and friends. [`append_position_fixed_fragments`]
//! is wired into `engine.rs` so v2's geometry-driven dispatch can
//! repeat `position: fixed` elements on every page (`is_repeat=true`
//! on the resulting `PaginationGeometry`).

use crate::units::F32Units;
use blitz_dom::BaseDocument;
use std::collections::{BTreeMap, BTreeSet};
use style::values::specified::box_::DisplayInside;
use taffy::{
    AvailableSpace, CacheTree, LayoutPartialTree, NodeId, RoundTree, Size, TraversePartialTree,
    TraverseTree,
};

/// One placement slot recorded per (source node × page).
///
/// `x`, `y`, `width`, `height` are type-enforced CSS pixels
/// ([`crate::units::Px`]) — Taffy's native unit — and `y` is measured from
/// the page's content-box top. The convert / draw layer is responsible for
/// px→pt conversion ([`crate::units::Px::in_pt`]) before reaching Krilla.
#[derive(Clone, Debug, PartialEq)]
pub struct Fragment {
    pub page_index: u32,
    pub x: crate::units::Px,
    pub y: crate::units::Px,
    pub width: crate::units::Px,
    pub height: crate::units::Px,
}

/// Per-source-node geometry: every page on which the node has a placement.
///
/// For the block-only fragmenter the vector is normally length 1 (the node
/// fits on one page). A node taller than the page produces multiple
/// fragments — but in the current measurement-only implementation we
/// emit it as a single oversized fragment on the page where its top
/// edge lands, because we have no inline / break point information yet.
///
/// # Repeat vs. split semantics
///
/// `is_repeat = false` (default): the vector represents a *split* —
/// each fragment is one slice of the same content, so consumers
/// accumulate `frag.height` to recover where to slice paragraph lines
/// or block content.
///
/// `is_repeat = true`: the vector represents *per-page repetition* —
/// every fragment carries the full content (`width` / `height` ==
/// the full element size). Consumers must NOT slice; each fragment
/// is a complete redraw at the same coordinates. Used by
/// [`append_position_fixed_fragments`] for `position: fixed` elements
/// that repeat on every page.
#[derive(Clone, Debug, Default)]
pub struct PaginationGeometry {
    pub fragments: Vec<Fragment>,
    /// Producers: `append_position_fixed_fragments` (`position: fixed`)
    /// and `append_repeated_header_fragments` (a repeating
    /// `table-header-group` and every descendant of its cells, which
    /// includes multicol containers — see `fulgur-naj7.12`). Consumers
    /// must treat the fragments as copies of the whole node, never as
    /// slices of it.
    pub is_repeat: bool,
    /// For an inline root split at line boundaries: the line index each
    /// fragment starts at, plus a terminating total line count — so
    /// `fragments[k]` covers lines
    /// `line_boundaries[k]..line_boundaries[k + 1]`. Empty for every
    /// other producer.
    ///
    /// This exists because the split decision and its consumer were
    /// measuring in different units. [`fragment_inline_root`] chooses
    /// the partition from Parley's line metrics, which are rounded to
    /// whole pixels; `render::paragraph_lines_for_page` then
    /// *re-derived* that same partition by accumulating
    /// `ShapedLine::height`, which keeps the fractional height. The two
    /// disagree, and when they do a line belongs to no fragment at all
    /// and is dropped from the document with no later fragment to
    /// recover it from — silently, with the page count short to match.
    ///
    /// Publishing the partition the fragmenter actually chose removes
    /// the second accounting. Consumers must still tolerate an empty
    /// vector (a paragraph that was strip-sliced whole rather than
    /// split at line boundaries publishes nothing).
    pub line_boundaries: Vec<usize>,
    /// How much shorter than its layout box the node ends on its page: a
    /// table without the header rows that skip its first page
    /// (`-fulgur-header-repeat: skip-first`), and each block around it by
    /// what stayed on its page. A node that is not split paints its box at
    /// the layout height less this ([`Self::unsplit_lift_px`]). Zero for
    /// every other node.
    pub lift_px: f32,
}

impl PaginationGeometry {
    /// What a node that is not split paints its box shorter than its layout
    /// height by ([`Self::lift_px`]); zero for a split node, whose fragments
    /// carry their heights.
    pub fn unsplit_lift_px(&self) -> f32 {
        if self.is_split() { 0.0 } else { self.lift_px }
    }

    /// Whether this node's content was *split* across multiple pages —
    /// i.e. each fragment is a slice of the same content. Returns
    /// `false` when the geometry represents per-page repetition
    /// (`is_repeat == true`), because in that case every fragment
    /// carries the full content and slicers must NOT subdivide it.
    pub fn is_split(&self) -> bool {
        !self.is_repeat && self.fragments.len() > 1
    }
}

/// Side-table mapping DOM `usize` NodeIds to their pagination geometry.
///
/// `BTreeMap` for the same determinism reason as
/// [`crate::multicol_layout::MulticolGeometryTable`]: PDF byte order
/// downstream depends on iteration order.
pub type PaginationGeometryTable = BTreeMap<usize, PaginationGeometry>;

/// Taffy tree wrapper that intercepts the pagination root through
/// `compute_child_layout` and routes it through fulgur's own
/// page-stripping logic.
///
/// `page_height_px` is the height of the page content area (after the
/// engine has subtracted page-margin / `@page` margins). The wrapper
/// borrows the `BaseDocument` for one pass and is discarded; the
/// `geometry` it accumulates is drained via [`Self::take_geometry`] so
/// callers can either thread it into `ConvertContext` or drop it for
/// observational use.
pub struct PaginationLayoutTree<'a> {
    pub(crate) doc: &'a mut BaseDocument,
    pub(crate) page_height_px: f32,
    pub(crate) geometry: PaginationGeometryTable,
    /// Cached id of the `<body>` element, if any. Used as the
    /// fragmentation root for the block-only fragmenter. `None` means the
    /// document had no body and the pass becomes a no-op.
    pub(crate) body_id: Option<usize>,
    /// fulgur-k0g0: `break-before` / `break-after` / `break-inside`
    /// per node, harvested by
    /// [`crate::blitz_adapter::extract_column_style_table`]. The table
    /// is shared with `multicol_layout`, so the pagination fragmenter
    /// does not maintain its own break-style extraction. `None` means
    /// "no break properties set anywhere", which the fragmenter treats
    /// as all-`Auto`.
    pub(crate) column_styles: Option<&'a crate::column_css::ColumnStyleTable>,
    /// fulgur-s67g Phase 2.2: `position: running()` element instances
    /// registered by [`crate::blitz_adapter::RunningElementPass`].
    /// The fragmenter consults this store to skip running-named elements
    /// at any depth — they are placed into `@page` margin boxes per page,
    /// not into the flow, so they must not contribute to the cursor or
    /// page-fragment geometry. `None` (the default for unit-test entry
    /// points) means "no running mappings"; the fragmenter treats every
    /// element as in-flow.
    pub(crate) running_store: Option<&'a crate::gcpm::running::RunningElementStore>,
    /// fulgur-uebl: per-element used page-name (CSS Page 3 §5.3),
    /// resolved from the same author-facing `page` declarations the
    /// `column_styles` table carries. The fragmenter consults this when
    /// a child is iterated: if its used page-name differs from the
    /// previously-placed sibling's, an implicit forced page break is
    /// induced before the child. `None` means the document has no
    /// `page` declarations and the fragmenter skips the comparison
    /// entirely.
    pub(crate) used_page_names: Option<crate::blitz_adapter::UsedPageNameTable>,
}

/// One-shot entry: run the block-level fragmenter for `doc` against a
/// `page_height_px` page strip and return the resulting geometry table.
///
/// Intended to be called **after** `blitz_adapter::resolve()` (and after
/// `multicol_layout::run_pass` when multicol is in play) so that
/// `final_layout` reflects the post-layout positions the fragmenter
/// walks.
///
/// Calls [`fragment_pagination_root`] directly to walk body's
/// children's existing `final_layout` (populated by
/// `blitz_adapter::resolve` and `multicol_layout::run_pass`) and
/// record per-node fragments. Same direct-walk model as the production
/// entry point — see the module docs for why we skip
/// `taffy::compute_root_layout` here. The Taffy-dispatch path is
/// preserved as test-only via
/// [`PaginationLayoutTree::drive_taffy_root_layout`].
///
/// Test-only convenience for fixtures that don't need break-style
/// awareness. Production callers use [`run_pass_with_break_styles`]
/// so `break-before` / `break-after` / `break-inside` from the shared
/// `ColumnStyleTable` are honoured.
#[cfg(test)]
pub fn run_pass(doc: &mut BaseDocument, page_height_px: f32) -> PaginationGeometryTable {
    run_pass_inner(doc, page_height_px, None, None).0
}

/// fulgur-k0g0 variant: thread the document's `break-before` /
/// `break-after` / `break-inside` side-table (harvested by
/// [`crate::blitz_adapter::extract_column_style_table`]) into the
/// fragmenter. `break-before: page` and `break-after: page` force
/// page boundaries; `break-inside: avoid` defers a child that does not
/// fit the remaining strip rather than splitting it.
pub fn run_pass_with_break_styles<'a>(
    doc: &'a mut BaseDocument,
    page_height_px: crate::units::Px,
    column_styles: &'a crate::column_css::ColumnStyleTable,
) -> PaginationGeometryTable {
    run_pass_inner(doc, page_height_px.to_f32(), Some(column_styles), None).0
}

/// fulgur-s67g Phase 2.2 variant: extends
/// [`run_pass_with_break_styles`] with awareness of `position:
/// running()` element instances. Running children are skipped during
/// the body walk so they do not contribute to body cursor or page
/// fragments — per-page placement is handled by
/// [`collect_running_element_states`]. Also returns the used page-name
/// table the pass computed, which [`collect_page_facts`] reads.
pub fn run_pass_with_break_and_running<'a>(
    doc: &'a mut BaseDocument,
    page_height_px: f32,
    column_styles: &'a crate::column_css::ColumnStyleTable,
    running_store: &'a crate::gcpm::running::RunningElementStore,
) -> (
    PaginationGeometryTable,
    crate::blitz_adapter::UsedPageNameTable,
) {
    let (geometry, used_page_names) = run_pass_inner(
        doc,
        page_height_px,
        Some(column_styles),
        Some(running_store),
    );
    (geometry, used_page_names.unwrap_or_default())
}

fn run_pass_inner<'a>(
    doc: &'a mut BaseDocument,
    page_height_px: f32,
    column_styles: Option<&'a crate::column_css::ColumnStyleTable>,
    running_store: Option<&'a crate::gcpm::running::RunningElementStore>,
) -> (
    PaginationGeometryTable,
    Option<crate::blitz_adapter::UsedPageNameTable>,
) {
    // fulgur-uebl: pre-compute the used page-name table when column
    // styles are available. The walk takes one DOM pass and produces a
    // `BTreeMap` keyed by node id, matching the determinism convention
    // used by the rest of the side-tables.
    let used_page_names =
        column_styles.map(|cs| crate::blitz_adapter::compute_used_page_names(doc, cs));
    let mut tree = PaginationLayoutTree::new(doc, page_height_px);
    tree.column_styles = column_styles;
    tree.running_store = running_store;
    tree.used_page_names = used_page_names;
    if tree.body_id.is_some() && page_height_px > 0.0 {
        // Read body's children's existing `final_layout` (populated by
        // Blitz's `resolve()` and `multicol_layout::run_pass`) and
        // produce the page-fragment geometry without re-driving Taffy.
        //
        // We deliberately *skip* `drive_taffy_root_layout` (which runs
        // `taffy::compute_root_layout` through the wrapper) on the
        // production path: re-issuing layout for body forces every
        // descendant's `compute_child_layout` to re-execute, and even
        // with cache hits the round-trip introduces sub-pixel
        // floating-point drift that breaks
        // `examples_determinism`'s byte-wise comparison against
        // committed PDFs. The wrapper's `LayoutPartialTree` /
        // `RoundTree` / `CacheTree` / `TraversePartialTree` impls
        // remain in place for tests that *do* exercise the full Taffy
        // dispatch (`drive_taffy_root_layout`) and as scaffolding for
        // a future per-strip-constrained variant where re-driving
        // layout is what actually does the pagination work.
        tree.fragment_pagination_root();
    }
    (tree.take_geometry(), tree.used_page_names.take())
}

impl<'a> PaginationLayoutTree<'a> {
    pub fn new(doc: &'a mut BaseDocument, page_height_px: f32) -> Self {
        let body_id = find_body_id(doc);
        Self {
            doc,
            page_height_px,
            geometry: BTreeMap::new(),
            body_id,
            column_styles: None,
            running_store: None,
            used_page_names: None,
        }
    }

    /// Drain the accumulated per-node geometry table.
    ///
    /// Mirrors [`crate::multicol_layout::FulgurLayoutTree::take_geometry`]:
    /// uses `mem::take` so a second call returns an empty table rather than
    /// double-counting.
    pub fn take_geometry(&mut self) -> PaginationGeometryTable {
        std::mem::take(&mut self.geometry)
    }

    /// Drive `taffy::compute_root_layout(&mut self, body_id, ...)` so the
    /// wrapper's `compute_child_layout` fires on body and dispatches into
    /// [`compute_pagination_layout`].
    ///
    /// **Test-only.** Production callers (`run_pass_with_break_styles`)
    /// reach geometry via `fragment_pagination_root` directly because
    /// re-driving Taffy on body re-stores every descendant's layout
    /// fields (even on cache hits) and introduces sub-pixel
    /// floating-point drift that breaks `examples_determinism`'s
    /// byte-wise PDF comparison against committed goldens. This entry
    /// is preserved so the wrapper's `LayoutPartialTree` / `RoundTree`
    /// / `CacheTree` / `TraversePartialTree` impls keep one runtime
    /// exerciser and a future per-strip-constrained variant has a
    /// drop-in seam.
    #[cfg(test)]
    ///
    /// The available space we hand Taffy is the body's *existing* layout
    /// width and an unbounded height (`AvailableSpace::MaxContent`). We
    /// pass MaxContent rather than `page_height_px` because the
    /// fragmenter relies on the children's natural `final_layout`
    /// heights — restricting `available_space.height` here would let
    /// Taffy clip or shrink children, breaking the measurement walk.
    /// (The fulgur-ik6o spike experimented with `Definite` and
    /// established that Taffy's block layout does not consult
    /// `available_space.height` for mid-element splitting; see
    /// `docs/plans/2026-04-28-pagination-layout-spike.md`.)
    ///
    /// `compute_root_layout` resets the layout's `location` to `(0, 0)`
    /// because it treats the node as a Taffy root. Body is *not* a real
    /// root in the document tree (html is its parent), so we save and
    /// restore body's location across the call — same approach as
    /// [`crate::multicol_layout::FulgurLayoutTree::layout_multicol_subtrees`].
    fn drive_taffy_root_layout(&mut self) {
        // fulgur-uebl: production `run_pass_inner` populates
        // `used_page_names` once `column_styles` is available; the
        // test-only Taffy parity path still needs the same table or
        // any future fixture using `page:` would silently skip the
        // implicit-break logic. Lazy-fill so call sites that want the
        // baseline behaviour can still leave both `None`.
        if self.used_page_names.is_none() {
            self.used_page_names = self
                .column_styles
                .map(|cs| crate::blitz_adapter::compute_used_page_names(self.doc, cs));
        }
        let Some(body_id) = self.body_id else {
            return;
        };
        let nid = NodeId::from(body_id);
        let prior_unrounded = self.doc.get_unrounded_layout(nid);
        let prior_final = self
            .doc
            .get_node(body_id)
            .map(|n| n.final_layout)
            .unwrap_or_default();

        let avail = Size {
            width: AvailableSpace::Definite(prior_unrounded.size.width.max(1.0)),
            height: AvailableSpace::MaxContent,
        };
        taffy::compute_root_layout(self, nid, avail);

        // Restore body's full layout so downstream readers (convert,
        // paginate) see byte-identical state to Blitz's first pass —
        // examples_determinism would otherwise pick up sub-pixel
        // float-rep differences when `compute_root_layout` re-stores
        // the same logical values via `set_unrounded_layout` /
        // `set_final_layout`.
        if let Some(node) = self.doc.get_node_mut(body_id) {
            node.unrounded_layout = prior_unrounded;
            node.final_layout = prior_final;
        }
    }

    /// Walk the body's children and record fragments.
    ///
    /// The body is the root container of the one walker,
    /// [`fragment_block_subtree_inner`]: it records the body's own fragment
    /// on page 0, then walks the body's children from the top of page 0 as
    /// the walker walks any block's (`root`). Records nothing when the
    /// document has no body or the page has no height; an empty document
    /// still produces a single empty page downstream.
    pub fn fragment_pagination_root(&mut self) {
        let Some(body_id) = self.body_id else {
            return;
        };
        if self.page_height_px <= 0.0 {
            return;
        }

        let body_layout = self
            .doc
            .get_node(body_id)
            .map(|n| n.final_layout)
            .unwrap_or_default();
        let body_w = body_layout.size.width;
        let body_x = body_layout.location.x;

        // fulgur-s67g Phase 2.3 (counter parity follow-up): record
        // body itself as a fragment on page 0. body's own
        // counter-reset / string-set / bookmark declarations must fire
        // only on page 0 (GCPM string-set / counters §3), so they need
        // to be recorded on the first page and only there.
        //
        // Without this entry the fragmenter's geometry table excludes body
        // entirely; `collect_counter_states` /
        // `collect_string_set_states` / `collect_bookmark_entries`
        // miss body's ops (e.g. `tests/gcpm_integration::test_counter_set`,
        // where body carries `counter-reset: chapter`). The body
        // fragment sits ahead of every body-direct-child entry in
        // NodeId order (Blitz allocates ids depth-first during parse,
        // so `body` is smaller than its descendants), so per-page
        // walks pick up body's ops before descendants — matching the
        // document tree-walk order the collectors expect.
        self.geometry
            .entry(body_id)
            .or_default()
            .fragments
            .push(Fragment {
                page_index: 0,
                x: body_x.as_px(),
                y: 0.0_f32.as_px(),
                width: body_w.as_px(),
                height: body_layout.size.height.as_px(),
            });

        let pass = Pass::default();
        let foot_clearance = page_foot_clearance(self.doc, self.column_styles, self.page_height_px);
        let flow = Flow {
            doc: self.doc,
            column_styles: self.column_styles,
            used_page_names: self.used_page_names.as_ref(),
            running_store: self.running_store,
            pass: &pass,
            forced_before: None,
            foot_clearance,
        };
        let mut remaining_repeat_budget = crate::MAX_SUBTREE_PAGE_FRAGMENTS;
        fragment_block_subtree_inner(
            &mut self.geometry,
            flow,
            body_id,
            body_w,
            body_x,
            0,
            0.0,
            self.page_height_px - foot_clearance,
            0,
            None,
            false,
            &mut false,
            &mut remaining_repeat_budget,
            &mut 0.0,
            true,
        );
    }
}

/// The page-foot clearance, in px, of pages `page_height_px` tall: the
/// root element's `-fulgur-page-foot-clearance`, zero when it declares none
/// or one that leaves no room on the page.
fn page_foot_clearance(
    doc: &BaseDocument,
    column_styles: Option<&crate::column_css::ColumnStyleTable>,
    page_height_px: f32,
) -> f32 {
    let declared = column_styles
        .and_then(|styles| styles.get(&doc.root_element().id))
        .and_then(|props| props.page_foot_clearance)
        .map_or(0.0, |pt| pt.as_pt().in_px().to_f32());
    if declared < page_height_px {
        declared
    } else {
        0.0
    }
}

/// How far past the page foot a box may reach and still fit the page. Stylo
/// and Taffy round some lengths to whole CSS pixels (a 220pt spacer, 293.33
/// px, lays out 294 px tall), while the page height is not rounded, so a box
/// sized to fill a page measures up to a pixel taller than it.
const FIT_SLACK_PX: f32 = 1.0;

/// Whether `extent` px fit a page strip `page_height_px` tall, within
/// [`FIT_SLACK_PX`]: the fits the walk measures against a page. The strip
/// is the page less its foot clearance ([`Flow::foot_clearance`]).
fn fits(extent: f32, page_height_px: f32) -> bool {
    extent <= page_height_px + FIT_SLACK_PX
}

/// Whether `extent` px run past a page strip `page_height_px` tall by more
/// than [`FIT_SLACK_PX`]: not [`fits`], but for an extent that is no
/// number, which neither fits nor overflows.
fn overflows(extent: f32, page_height_px: f32) -> bool {
    extent > page_height_px + FIT_SLACK_PX
}

/// The page that content after a forced break starts on, given the page
/// `page_index` the break reached: one page further when the break asks for
/// a side (`left`, `right`, `recto`, `verso`) that `page_index` is not (CSS
/// Fragmentation 3 §3.1). The page skipped holds no fragment, so it prints
/// only the page box's margin boxes and still counts in `counter(page)`.
/// Page index 0 is a right page, or a left page under a right-to-left root
/// (`gcpm::page_settings::is_left_page`, the rule of the `@page :left` /
/// `:right` settings and margin boxes, CSS Paged Media §5).
fn page_on_side(
    doc: &BaseDocument,
    page_index: u32,
    side: Option<crate::draw_primitives::PageSide>,
) -> u32 {
    let Some(side) = side else {
        return page_index;
    };
    let is_right = !crate::gcpm::page_settings::is_left_page(
        page_index as usize + 1,
        crate::convert::extract_root_dir_rtl(doc),
    );
    if (side == crate::draw_primitives::PageSide::Right) == is_right {
        page_index
    } else {
        page_index + 1
    }
}

/// fulgur-ezst: true if `parent_id`'s subtree renders any VISIBLE box (a
/// descendant with positive area — both width and height > 0). Used to
/// classify a pathologically tall block as "childless" (collapsible): the
/// collapse is safe only when nothing visible would be lost.
///
/// Walks children like `record_subtree_descendants` (the `layout_children`
/// preference, short-circuiting on the first hit), but the emptiness test
/// differs deliberately: a descendant with zero AREA (EITHER dimension
/// <= 0) paints nothing, so it does not count as content — instead we
/// recurse into it, because with `overflow: visible` a zero-height /
/// zero-width box can still host visible descendants that overflow it.
/// (This is why the test is `||`, not `&&`: an empty block child lays out
/// width>0 / height==0, and `&&` would wrongly count it as content and
/// disable the collapse — a trivial DoS bypass,
/// `<div style="height:99999999px"><div></div></div>`; Codex P2 on PR #553.)
/// `record_subtree_descendants` still *records* such zero-area boxes (their
/// node ids may carry counters / ids), which is a separate question from
/// "does this paint anything". Conservative on depth: a subtree deeper than
/// `MAX_DOM_DEPTH` reads as no content, matching `record_subtree_descendants`
/// (which also stops recording there, so such content renders blank anyway).
fn subtree_has_rendered_content(doc: &BaseDocument, parent_id: usize, depth: usize) -> bool {
    if depth >= crate::MAX_DOM_DEPTH {
        return false;
    }
    let Some(parent) = doc.get_node(parent_id) else {
        return false;
    };
    let layout_children_borrow = parent.layout_children.borrow();
    let walk_children: &[usize] = layout_children_borrow
        .as_deref()
        .filter(|v| !v.is_empty())
        .unwrap_or(&parent.children);
    for &child_id in walk_children {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        let layout = child.final_layout;
        // `visibility: hidden` / `collapse` occupies layout space but paints
        // nothing (conversion skips its paint), so it does not count as
        // content — a `<div style="visibility:hidden">` must not defeat the
        // collapse (Codex P2 on PR #553).
        let invisible = {
            use style::properties::longhands::visibility::computed_value::T as Visibility;
            child
                .primary_styles()
                .is_some_and(|s| s.clone_visibility() != Visibility::Visible)
        };
        if invisible || layout.size.height <= 0.0 || layout.size.width <= 0.0 {
            // An invisible or zero-area box paints nothing itself, but may
            // host visible descendants — overflowing it (overflow: visible)
            // or flipping `visibility` back to `visible` — so recurse.
            if subtree_has_rendered_content(doc, child_id, depth + 1) {
                return true;
            }
            continue;
        }
        return true;
    }
    false
}

/// fulgur-s67g Phase 2.5: recursively record fragments for every
/// visible descendant of a body-direct child, attaching them to the
/// same `page_index` as the ancestor.
///
/// `parent_page_y` is the parent's body-relative y position on the
/// current page strip; `parent_x_in_body` is the parent's x position
/// (already pre-resolved against `body_x`). For each descendant, the
/// recorded fragment uses absolute body-relative coordinates
/// computed by adding the descendant's `final_layout.location` to
/// the parent's frame.
///
/// Skips zero-size descendants and bails at
/// [`crate::MAX_DOM_DEPTH`] to keep recursion bounded against
/// adversarial input.
///
/// Mid-element split inside a body child (a deeply nested element
/// crossing the page boundary that the parent itself did not split
/// at) is **not** modelled here — descendants land on the same page
/// as their ancestor. Closing this "block-level only" gap requires the
/// full per-strip layout pass that future fragmenter work will
/// introduce.
fn record_subtree_descendants(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    parent_id: usize,
    page_index: u32,
    parent_page_y: f32,
    parent_x_in_body: f32,
    depth: usize,
) {
    if depth >= crate::MAX_DOM_DEPTH {
        return;
    }
    let Some(parent) = doc.get_node(parent_id) else {
        return;
    };
    // Prefer Blitz's `layout_children` over the raw DOM `children` when
    // it's been computed: when a block container has mixed
    // block-level and inline-level children, Stylo synthesizes
    // anonymous block wrappers around inline-level siblings (CSS 2.1
    // §9.2.1.1). Those wrappers are real `Node` instances with their
    // own `node_id` and Taffy layout, but they live ONLY in
    // `layout_children` — the original `children` list still points
    // at the underlying inline elements (e.g. a `<span
    // display:inline-block>`).
    //
    // Without this preference v2 silently drops the inline-level
    // span: extract assigns the inner paragraph's `node_id` to the
    // anonymous wrapper (because Blitz's `is_inline_root()` flag sits
    // on the wrapper), but the fragmenter — walking `children` —
    // never visits the wrapper, so geometry has no fragment for that
    // node_id and `dispatch_fragment` skips the paragraph entirely.
    // (fulgur-bq6i: review_card_inline_block.html lost its
    // "OK Approved" rounded badge for this exact reason.)
    let layout_children_borrow = parent.layout_children.borrow();
    let walk_children: &[usize] = layout_children_borrow
        .as_deref()
        .filter(|v| !v.is_empty())
        .unwrap_or(&parent.children);
    for &child_id in walk_children {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        let layout = child.final_layout;
        let h = layout.size.height;
        let w = layout.size.width;
        // Phase 4 PR 5: zero-size containers (`<tbody>`, `<tr>`,
        // anonymous boxes) carry no paint payload but DO host visible
        // descendants (e.g. table cells) that v2 needs in geometry.
        // Skipping them entirely leaves cells out of `geometry`; the
        // dispatcher then never finds the cell node_ids and v2 emits
        // a blank table. Recurse without recording when h/w are
        // both zero so the descendant cells still register.
        if h <= 0.0 && w <= 0.0 {
            record_subtree_descendants(
                geometry,
                doc,
                child_id,
                page_index,
                parent_page_y + layout.location.y,
                parent_x_in_body + layout.location.x,
                depth + 1,
            );
            continue;
        }
        let child_x = parent_x_in_body + layout.location.x;
        let child_y = parent_page_y + layout.location.y;
        geometry
            .entry(child_id)
            .or_default()
            .fragments
            .push(Fragment {
                page_index,
                x: child_x.as_px(),
                y: child_y.as_px(),
                width: w.as_px(),
                height: h.as_px(),
            });
        record_subtree_descendants(
            geometry,
            doc,
            child_id,
            page_index,
            child_y,
            child_x,
            depth + 1,
        );
    }
}

/// fulgur-7hf5 (Phase 3.1.5c): pre-flight check for the recursion
/// gate — true if walking `parent_id`'s direct children would cross
/// a page boundary at `available_h`.
///
/// Cheaper-than-real `fragment_block_subtree` simulator: same gap /
/// OOF / whitespace skips, but no fragment emission. Returns `true`
/// on the first overflow detected. Lets the caller decide "should I
/// recurse here?" without paying the cost of recursion when recursion
/// would not actually split — distinguishing "recurse and split"
/// (within-child) from "push child whole / emit whole" (at-index or
/// no split).
///
/// `available_h` is the strip height left below the parent's entry
/// cursor on the current page. `page_h_px` lets the simulator detect
/// grandchildren taller than a full page (which would themselves
/// recurse and therefore split, regardless of `available_h`).
///
/// An inline root (a paragraph whose children are text runs and atomic
/// inlines such as `<img>`) never splits here: its inline children are not
/// break points. Their boxes sit inside line boxes, so a line holding an
/// image that overflows the strip would otherwise be cut at the image and
/// leave the line on the current page, past its edge. Line boundaries are the
/// only break points of an inline root, handled by `fragment_inline_root`;
/// a single line that does not fit moves whole to the next page.
fn would_split_block_subtree(
    doc: &BaseDocument,
    parent_id: usize,
    available_h: f32,
    page_h_px: f32,
    depth: usize,
) -> bool {
    if depth >= crate::MAX_DOM_DEPTH {
        return false;
    }
    let Some(parent) = doc.get_node(parent_id) else {
        return false;
    };
    if parent
        .element_data()
        .is_some_and(|e| e.inline_layout_data.is_some())
    {
        return false;
    }
    let mut cursor: f32 = 0.0;
    let mut prev_bottom: f32 = 0.0;
    // fulgur-yb27: walk `layout_children` so anonymous block wrappers
    // Stylo synthesizes around inline-level siblings (CSS 2.1
    // §9.2.1.1) participate in the cumulative-overflow simulation.
    // Mirrors the `fragment_block_subtree` walker switch above —
    // without this, a block whose tail anon block wrapper would
    // overflow is missed by the preflight, the recursion gate
    // returns false, and the parent falls back to a single
    // oversize fragment.
    for child_id in walk_children_of(parent) {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        if leaves_the_flow(child) {
            continue;
        }
        let layout = child.final_layout;
        let h = layout.size.height;
        if h <= 0.0 {
            continue;
        }
        let this_top = layout.location.y;
        let gap = (this_top - prev_bottom).max(0.0);
        cursor += gap;
        if cursor + h > available_h {
            return true;
        }
        if h > page_h_px {
            // Grandchild itself oversized → would recurse → would
            // split, regardless of `available_h` budget.
            return true;
        }
        cursor += h;
        prev_bottom = this_top + h;
    }
    false
}

/// True when the header band can be recorded once and cloned onto every
/// continuation page.
///
/// [`record_header_template`] snapshots the band's coordinates directly rather
/// than running it through the fragmenter, so a forced break or a page-name
/// change inside a header cell has nothing to act on and is silently dropped
/// (fulgur-naj7.7: `break-after: page` on a `<div>` inside a `<th>` rendered
/// one page where two are required). A band that fragments is also not a band
/// — "repeat this strip on each page" and "this strip itself spans pages" are
/// contradictory requests.
///
/// Rather than drop the author's break, decline to repeat: the caller falls
/// through to [`fragment_block_subtree_inner`], which honours breaks normally.
/// The table then paginates without a repeated header.
fn header_band_is_repeatable(
    flow: Flow<'_>,
    header: &RepeatingTableHeader,
    page_height_px: f32,
    depth: usize,
) -> bool {
    header.header_cell_ids.iter().all(|&cell_id| {
        // The break may sit on the header cell itself as well as below it.
        flow.doc.get_node(cell_id).is_none_or(|cell| {
            !break_before_of(flow, cell, page_height_px, depth).is_forced()
                && !break_after_of(flow, cell, page_height_px, depth).is_forced()
                && !forced_inside(flow, cell, page_height_px, depth)
        })
    })
}

/// The value of a break opportunity as it concerns pages (css-break-3 §3.1,
/// §4.4): no break asked for, a break avoided, or a forced break, to a page
/// of a side or to the next page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BreakValue {
    #[default]
    Auto,
    Avoid,
    Forced(Option<crate::draw_primitives::PageSide>),
}

impl BreakValue {
    /// The `break-before` a box declares itself.
    fn own_before(props: Option<&crate::column_css::ColumnStyleProps>) -> Self {
        use crate::draw_primitives::BreakBefore;
        match props.and_then(|props| props.break_before) {
            Some(BreakBefore::Page) => {
                Self::Forced(props.and_then(|props| props.break_before_side))
            }
            Some(BreakBefore::Avoid) => Self::Avoid,
            Some(BreakBefore::Auto) | None => Self::Auto,
        }
    }

    /// The `break-after` a box declares itself.
    fn own_after(props: Option<&crate::column_css::ColumnStyleProps>) -> Self {
        use crate::draw_primitives::BreakAfter;
        match props.and_then(|props| props.break_after) {
            Some(BreakAfter::Page) => Self::Forced(props.and_then(|props| props.break_after_side)),
            Some(BreakAfter::Avoid) => Self::Avoid,
            Some(BreakAfter::Auto) | None => Self::Auto,
        }
    }

    /// The value of the one opportunity that `self` and `later`, which comes
    /// after it in the flow, both apply to (css-break-3 §3.1): a forced
    /// break wins, and two make one break, to the side the later asks for,
    /// else to the earlier's; `avoid` beats `auto`.
    fn then(self, later: Self) -> Self {
        match (self, later) {
            (Self::Forced(side), Self::Forced(later_side)) => Self::Forced(later_side.or(side)),
            (Self::Forced(_), _) => self,
            (_, Self::Forced(_)) => later,
            (Self::Avoid, _) | (_, Self::Avoid) => Self::Avoid,
            (Self::Auto, Self::Auto) => Self::Auto,
        }
    }

    fn is_forced(self) -> bool {
        matches!(self, Self::Forced(_))
    }

    /// The page side a forced break asks for.
    fn side(self) -> Option<crate::draw_primitives::PageSide> {
        match self {
            Self::Forced(side) => side,
            Self::Auto | Self::Avoid => None,
        }
    }
}

/// Whether the walk places `child` among its parent's children: it neither
/// leaves the flow ([`leaves_the_flow`]) nor is a running element
/// ([`is_running`]).
fn is_flow_child(flow: Flow<'_>, child: &blitz_dom::Node) -> bool {
    !leaves_the_flow(child) && !is_running(flow, child)
}

/// Whether `child` takes no part in its parent's flow: white space, or
/// absolutely positioned or fixed (another pass places those).
fn leaves_the_flow(child: &blitz_dom::Node) -> bool {
    use ::style::properties::longhands::position::computed_value::T as Pos;
    crate::blitz_adapter::is_whitespace_only_text_node(child)
        || child
            .primary_styles()
            .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Absolute | Pos::Fixed))
}

/// Whether `child` takes no room in its parent's block flow: it leaves the
/// flow ([`leaves_the_flow`]), or it floats.
fn out_of_flow(child: &blitz_dom::Node) -> bool {
    leaves_the_flow(child) || crate::blitz_adapter::node_is_floating(child)
}

/// Whether `child` is a `position: running()` element, which takes no room
/// in the flow: it is placed in a page margin box.
fn is_running(flow: Flow<'_>, child: &blitz_dom::Node) -> bool {
    flow.running_store
        .is_some_and(|store| store.instance_for_node(child.id).is_some())
}

/// The children the walk iterates for `node`: its layout children, which
/// hold the anonymous blocks Stylo wraps around inline-level runs, else its
/// DOM children.
fn walk_children_of(node: &blitz_dom::Node) -> Vec<usize> {
    let layout_children = node.layout_children.borrow();
    layout_children
        .as_deref()
        .filter(|children| !children.is_empty())
        .map(Vec::from)
        .unwrap_or_else(|| node.children.clone())
}

/// Whether the walk never enters `node` for its children nor splits it, at
/// any room: a childless or replaced box, a transformed box that fits a page
/// (it paints as one shape, CSS Transforms §6.1), an atomic inline, or a box
/// in an orthogonal flow (CSS Writing Modes 4 §9). `page_foot_px` is the
/// height of the whole page ([`Flow::page_foot`]).
fn is_atomic(doc: &BaseDocument, node: &blitz_dom::Node, page_foot_px: f32) -> bool {
    let fits_page = fits(node.final_layout.size.height, page_foot_px);
    let orthogonal = node
        .parent
        .and_then(|id| doc.get_node(id))
        .is_some_and(|parent| crate::blitz_adapter::is_orthogonal_to_parent(parent, node));
    node.children.is_empty()
        || (is_transformed(node) && fits_page)
        || crate::blitz_adapter::is_atomic_inline_container_node(node)
        || orthogonal
}

/// Whether `node` is a multicol container the walk does not enter: without a
/// child that spans its columns, `multicol_layout` fragments it.
fn is_unspanned_multicol(doc: &BaseDocument, node: &blitz_dom::Node) -> bool {
    crate::blitz_adapter::is_multicol_container(node)
        && !node.children.iter().any(|&id| {
            doc.get_node(id)
                .is_some_and(crate::blitz_adapter::has_column_span_all)
        })
}

/// Whether the walk meets the boxes of `node` one by one, with class A
/// opportunities between them (css-break-3 §4.1): it is neither atomic
/// ([`is_atomic`]), nor a multicol container it does not enter, nor an
/// inline root, whose breaks lie between its lines. `page_foot_px` as
/// [`is_atomic`]'s.
fn has_flow_children(doc: &BaseDocument, node: &blitz_dom::Node, page_foot_px: f32) -> bool {
    !is_atomic(doc, node, page_foot_px)
        && !is_unspanned_multicol(doc, node)
        && (is_transformed(node)
            || node
                .element_data()
                .is_none_or(|el| el.inline_layout_data.is_none()))
}

/// The flow children of a walk over `children` of `parent`
/// ([`is_flow_child`]), as indices into `children`, in lines, as the walk
/// groups them: in a table, whose layout children are its cells, the cells
/// of one row element, in document order; in a flex or grid container a
/// child that starts above the bottom of the line before it sits in that
/// line, whatever its alignment (the lines of [`LineState`]); elsewhere one
/// must also start beside the line's first box, as a float and the block
/// after it do not. No class A opportunity lies between the boxes of a line
/// (css-break-3 §4.1); each line takes on its boxes' `break-before` and
/// `break-after`.
fn flow_lines(flow: Flow<'_>, parent: &blitz_dom::Node, children: &[usize]) -> Vec<Vec<usize>> {
    let by_extent = crate::blitz_adapter::is_flex_or_grid_container_node(parent);
    let by_row = displays(parent, DisplayInside::Table);
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut line_box = (0.0_f32, 0.0_f32);
    let mut line_row = None;
    for (index, &id) in children.iter().enumerate() {
        let Some(child) = flow.doc.get_node(id) else {
            continue;
        };
        if !is_flow_child(flow, child) {
            continue;
        }
        let layout = child.final_layout;
        let (top, bottom) = (layout.location.y, layout.location.y + layout.size.height);
        let row = by_row.then_some(child.parent).flatten();
        match lines.last_mut() {
            Some(line) if by_row && row.is_some() && row == line_row => line.push(index),
            Some(line)
                if !by_row && top < line_box.1 - 0.5 && (by_extent || top <= line_box.0 + 0.5) =>
            {
                line.push(index);
                line_box.1 = line_box.1.max(bottom);
            }
            _ => {
                lines.push(vec![index]);
                line_box = (top, bottom);
                line_row = row;
            }
        }
    }
    lines
}

/// The `break-before` or the `break-after` (`after`) of a line of
/// `children` ([`flow_lines`]): its boxes' values, combined.
fn line_break(
    flow: Flow<'_>,
    children: &[usize],
    line: &[usize],
    after: bool,
    page_height_px: f32,
    depth: usize,
) -> BreakValue {
    let boxes = line
        .iter()
        .filter_map(|&index| flow.doc.get_node(children[index]))
        .map(|child| {
            if after {
                break_after_of(flow, child, page_height_px, depth)
            } else {
                break_before_of(flow, child, page_height_px, depth)
            }
        })
        .fold(BreakValue::Auto, BreakValue::then);
    // A table row's line takes on the row's own value, and its group's at
    // the group's first or last row: they have no box of their own in the
    // layout, and no opportunity parts them from their cells (css-break-3
    // §3.1).
    let Some(row) = line
        .first()
        .and_then(|&index| flow.doc.get_node(children[index]))
        .and_then(|cell| table_row_of(flow.doc, cell))
    else {
        return boxes;
    };
    let own = |node: &blitz_dom::Node| {
        let props = flow.column_styles.and_then(|styles| styles.get(&node.id));
        if after {
            BreakValue::own_after(props)
        } else {
            BreakValue::own_before(props)
        }
    };
    let at_edge = flow
        .pass
        .row_group(flow, row)
        .is_some_and(|(group, index)| {
            if after {
                index + 1 == group.rows.len()
            } else {
                index == 0
            }
        });
    let group = row
        .parent
        .and_then(|id| flow.doc.get_node(id))
        .filter(|group| at_edge && !displays(group, DisplayInside::Table));
    let group = group.map_or(BreakValue::Auto, own);
    if after {
        boxes.then(own(row)).then(group)
    } else {
        group.then(own(row)).then(boxes)
    }
}

/// The style whose `break-before`, `break-after`, and `break-inside` the
/// walk reads for `node`: for the box the caption pass wraps around a table
/// and its caption ([`crate::blitz_adapter::captioned_table_of`]), the
/// table's, as that box stands for the table wrapper box, which holds the
/// caption in the flow: a table kept whole moves with its caption, or
/// splits under it when the two do not fit a page together, and a forced
/// break before the table goes before its caption. None for the table in
/// that box, whose breaks are the box's; else its own.
fn break_props<'a>(
    flow: Flow<'a>,
    node: &blitz_dom::Node,
) -> Option<&'a crate::column_css::ColumnStyleProps> {
    let styles = flow.column_styles?;
    if let Some(table) = crate::blitz_adapter::captioned_table_of(flow.doc, node) {
        return styles.get(&table);
    }
    let wrapped = node
        .parent
        .and_then(|id| flow.doc.get_node(id))
        .and_then(|parent| crate::blitz_adapter::captioned_table_of(flow.doc, parent))
        == Some(node.id);
    if wrapped {
        return None;
    }
    styles.get(&node.id)
}

/// The breaks of a box as its parent's walk meets it, computed together
/// once a pass and page height ([`box_breaks`]), as each reads its
/// children's.
struct BoxBreaks {
    /// Its own `break-before`, then its first line's ([`flow_lines`]),
    /// which a container takes on, as no class A opportunity parts the two
    /// (css-break-3 §3.1).
    before: BreakValue,
    /// Its last line's `break-after`, which a container takes on, then its
    /// own.
    after: BreakValue,
    /// A forced break opportunity lies inside it ([`forced_inside`]).
    forced_inside: bool,
}

/// A walk over the children of a box ([`walk_children_of`]): its lines
/// ([`flow_lines`]) and the class A opportunities before them
/// ([`opportunities`], page names compared unless
/// [`page_names_suppressed`]), computed once a pass and page height
/// ([`walk_breaks`]).
struct WalkBreaks {
    children: Vec<usize>,
    lines: Vec<Vec<usize>>,
    /// The line of each child in a line, by the child's node.
    line_of: BTreeMap<usize, usize>,
    values: Vec<BreakValue>,
}

/// The value `map` holds for `key`, computed by `compute` on first ask and
/// kept for the pass; nothing is kept when `compute` gives none. No borrow
/// of `map` is held while `compute` runs, which may ask `map` in turn.
fn memo<K: Ord, V>(
    map: &std::cell::RefCell<BTreeMap<K, std::rc::Rc<V>>>,
    key: K,
    compute: impl FnOnce() -> Option<V>,
) -> Option<std::rc::Rc<V>> {
    let cached = map.borrow().get(&key).cloned();
    if cached.is_some() {
        return cached;
    }
    let value = std::rc::Rc::new(compute()?);
    map.borrow_mut().insert(key, value.clone());
    Some(value)
}

/// The breaks of `node` on pages `page_height_px` tall ([`BoxBreaks`]).
fn box_breaks(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> std::rc::Rc<BoxBreaks> {
    let key = (node.id, page_height_px.to_bits());
    memo(&flow.pass.breaks, key, || {
        Some(compute_box_breaks(flow, node, page_height_px, depth))
    })
    .expect("a box's breaks are always computed")
}

/// [`box_breaks`], computed.
fn compute_box_breaks(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> BoxBreaks {
    let props = break_props(flow, node);
    let (own_before, own_after) = (BreakValue::own_before(props), BreakValue::own_after(props));
    if depth >= crate::MAX_DOM_DEPTH
        || !has_flow_children(flow.doc, node, flow.page_foot(page_height_px))
    {
        return BoxBreaks {
            before: own_before,
            after: own_after,
            forced_inside: false,
        };
    }
    let walk = walk_breaks(flow, node, page_height_px, depth);
    let edge = |line: Option<&Vec<usize>>, after: bool| {
        line.map(|line| line_break(flow, &walk.children, line, after, page_height_px, depth + 1))
    };
    let before = edge(walk.lines.first(), false).map_or(own_before, |first| own_before.then(first));
    let after = edge(walk.lines.last(), true).map_or(own_after, |last| last.then(own_after));
    let forced_inside = walk.values.iter().copied().any(BreakValue::is_forced)
        || walk
            .children
            .iter()
            .filter_map(|&id| flow.doc.get_node(id))
            .filter(|&child| is_flow_child(flow, child))
            .any(|child| forced_inside(flow, child, page_height_px, depth + 1));
    BoxBreaks {
        before,
        after,
        forced_inside,
    }
}

/// The walk over the children of `parent` on pages `page_height_px` tall
/// ([`WalkBreaks`]), its opportunities read at `depth` below the walk's.
fn walk_breaks(
    flow: Flow<'_>,
    parent: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> std::rc::Rc<WalkBreaks> {
    let key = (parent.id, page_height_px.to_bits());
    memo(&flow.pass.walks, key, || {
        let children = walk_children_of(parent);
        let lines = flow_lines(flow, parent, &children);
        let suppress = page_names_suppressed(flow.doc, parent);
        let values = opportunities(flow, &children, &lines, suppress, page_height_px, depth + 1);
        let line_of = lines
            .iter()
            .enumerate()
            .flat_map(|(number, line)| line.iter().map(move |&index| (index, number)))
            .map(|(index, number)| (children[index], number))
            .collect();
        Some(WalkBreaks {
            children,
            lines,
            line_of,
            values,
        })
    })
    .expect("a walk's breaks are always computed")
}

/// The `break-before` of `node` as its parent's walk meets it
/// ([`BoxBreaks::before`]).
fn break_before_of(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> BreakValue {
    box_breaks(flow, node, page_height_px, depth).before
}

/// The `break-after` of `node` as its parent's walk meets it
/// ([`BoxBreaks::after`]).
fn break_after_of(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> BreakValue {
    box_breaks(flow, node, page_height_px, depth).after
}

/// Whether the walk compares no used page names between the children of
/// `parent` (CSS Page 3 §5.3): the items of a flex or grid container, which
/// `page` does not apply to, and the children of an atomic inline or of a
/// box in an orthogonal flow, whose block flow does not paginate on its own.
fn page_names_suppressed(doc: &BaseDocument, parent: &blitz_dom::Node) -> bool {
    crate::blitz_adapter::is_flex_or_grid_container_node(parent)
        || crate::blitz_adapter::is_atomic_inline_container_node(parent)
        || parent
            .parent
            .and_then(|id| doc.get_node(id))
            .is_some_and(|gp| crate::blitz_adapter::is_orthogonal_to_parent(gp, parent))
}

/// The class A break opportunities of a walk over `children` (css-break-3
/// §4.1): the value of the one before each line of flow children
/// ([`flow_lines`]), at the line's first box, in a vector parallel to
/// `children`. Between two lines it combines the earlier's `break-after`
/// with the later's `break-before` ([`line_break`], [`BreakValue::then`]),
/// and a change of used page name between two boxes that are not floats
/// forces a break (CSS Page 3 §5.3) unless `suppress_page_names`. The first
/// line's `break-before` and the last line's `break-after` are no
/// opportunity of this walk: the parent takes them on, and its own parent's
/// walk applies them. Every other slot holds `Auto`.
fn opportunities(
    flow: Flow<'_>,
    children: &[usize],
    lines: &[Vec<usize>],
    suppress_page_names: bool,
    page_height_px: f32,
    depth: usize,
) -> Vec<BreakValue> {
    let mut values = vec![BreakValue::Auto; children.len()];
    let mut previous_after: Option<BreakValue> = None;
    let mut previous_page: Option<Option<String>> = None;
    for line in lines.iter().map(Vec::as_slice) {
        let mut value = previous_after
            .map(|after| {
                after.then(line_break(
                    flow,
                    children,
                    line,
                    false,
                    page_height_px,
                    depth,
                ))
            })
            .unwrap_or_default();
        for &index in line {
            let id = children[index];
            if flow
                .doc
                .get_node(id)
                .is_none_or(crate::blitz_adapter::node_is_floating)
            {
                continue;
            }
            let (start, end) = flow
                .used_page_names
                .and_then(|names| names.get(&id).cloned())
                .unwrap_or((None, None));
            if !suppress_page_names
                && index == line[0]
                && previous_page.as_ref().is_some_and(|page| *page != start)
            {
                value = value.then(BreakValue::Forced(None));
            }
            previous_page = Some(end);
        }
        // No opportunity lies between the rows a cell spans that are kept
        // together, and no forced one above a row a spanning cell's content
        // reaches ([`RowGroup`]).
        let (held, crossed) = flow
            .doc
            .get_node(children[line[0]])
            .and_then(|cell| table_row_of(flow.doc, cell))
            .and_then(|row| flow.pass.row_group(flow, row))
            .map_or((false, false), |(group, index)| {
                (group.spanned[index], group.crossed[index])
            });
        let dropped = if value.is_forced() { crossed } else { held };
        values[line[0]] = if dropped { BreakValue::Auto } else { value };
        previous_after = Some(line_break(
            flow,
            children,
            line,
            true,
            page_height_px,
            depth,
        ));
    }
    values
}

/// Whether a forced break opportunity lies inside `node`, between two of the
/// lines the walk meets in it at any depth ([`opportunities`]), so that the
/// walk enters it to break there even when it fits the page. The forced
/// breaks at its edges, its first line's `break-before` and its last line's
/// `break-after`, are no reason: its parent's walk applies them. No
/// opportunity lies inside an atomic box or an inline root
/// ([`has_flow_children`]).
fn forced_inside(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> bool {
    box_breaks(flow, node, page_height_px, depth).forced_inside
}

/// Whether the walk meets the children of `parent` one below the other: a
/// block container, not a flex or grid container, a table, or a table row,
/// whose boxes side by side the walk meets by their rows.
fn is_block_flow(flow: Flow<'_>, parent: &blitz_dom::Node) -> bool {
    matches!(structure(flow, parent), BoxKind::BlockFlow)
}

/// How far below the bottom of line `line` of `children` (of `parent`,
/// [`flow_lines`]) what the line keeps with it on its page reaches: the
/// bottom margins of its boxes at the page foot ([`foot_margin`]), and the
/// boxes after it that an `avoid` joins it to ([`kept_run_after_line`]).
#[allow(clippy::too_many_arguments)]
fn kept_after_line(
    flow: Flow<'_>,
    parent: &blitz_dom::Node,
    children: &[usize],
    lines: &[Vec<usize>],
    values: &[BreakValue],
    line: usize,
    page_height_px: f32,
    depth: usize,
) -> f32 {
    let margins = lines[line]
        .iter()
        .filter_map(|&index| flow.doc.get_node(children[index]))
        .map(|child| foot_margin(flow, child))
        .fold(0.0_f32, f32::max);
    margins.max(kept_run_after_line(
        flow,
        parent,
        children,
        lines,
        values,
        line,
        page_height_px,
        depth,
    ))
}

/// How far below the bottom of line `line` of `children` (of `parent`,
/// [`flow_lines`]) the boxes after it that the line keeps with it reach,
/// when the opportunity after the line is `avoid` (css-break-3 §4.4,
/// [`opportunities`] `values`): the gap, then the next line's leading unit
/// ([`leading_extent`]), or the whole next box and what it keeps with it in
/// turn when its unit is the whole box ([`kept_after_line`]). After the last
/// line, what follows the parent's bottom edge that the parent keeps with
/// it ([`kept_after_box`]). Zero when no `avoid` follows the line.
#[allow(clippy::too_many_arguments)]
fn kept_run_after_line(
    flow: Flow<'_>,
    parent: &blitz_dom::Node,
    children: &[usize],
    lines: &[Vec<usize>],
    values: &[BreakValue],
    line: usize,
    page_height_px: f32,
    depth: usize,
) -> f32 {
    let doc = flow.doc;
    if depth >= crate::MAX_DOM_DEPTH {
        return 0.0;
    }
    let bottom = lines[line]
        .iter()
        .filter_map(|&index| doc.get_node(children[index]))
        .map(|child| child.final_layout.location.y + child.final_layout.size.height)
        .fold(0.0_f32, f32::max);
    let Some(next_line) = lines.get(line + 1) else {
        let after = kept_after_box(flow, parent, page_height_px, depth + 1);
        return if after > 0.0 {
            (parent.final_layout.size.height - bottom).max(0.0) + after
        } else {
            0.0
        };
    };
    if values[next_line[0]] != BreakValue::Avoid {
        return 0.0;
    }
    let next: Vec<&blitz_dom::Node> = next_line
        .iter()
        .filter_map(|&index| doc.get_node(children[index]))
        .collect();
    let Some(&first) = next.first() else {
        return 0.0;
    };
    let top = next
        .iter()
        .map(|node| node.final_layout.location.y)
        .fold(f32::INFINITY, f32::min);
    let height = line_bottom(&next) - top;
    let gap = (top - bottom).max(0.0);
    let lead = line_leading(flow, &next, top, page_height_px, page_height_px, depth + 1)
        .unwrap_or_else(|| leading_extent(flow, first, page_height_px, page_height_px, depth + 1));
    let kept = if lead.extent >= height - 0.5 {
        height
            + kept_after_line(
                flow,
                parent,
                children,
                lines,
                values,
                line + 1,
                page_height_px,
                depth + 1,
            )
    } else {
        lead.extent
    };
    gap + kept
}

/// How far below the bottom edge of `node` the boxes after it that it keeps
/// with it reach ([`kept_after_line`]), in its parent's block flow: zero
/// for the body, and in a parent the walk does not meet in block flow.
fn kept_after_box(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    page_height_px: f32,
    depth: usize,
) -> f32 {
    let doc = flow.doc;
    let is_body = |node: &blitz_dom::Node| {
        node.element_data()
            .is_some_and(|el| el.name.local.as_ref() == "body")
    };
    if depth >= crate::MAX_DOM_DEPTH || is_body(node) {
        return 0.0;
    }
    let Some(parent) = node
        .layout_parent
        .get()
        .or(node.parent)
        .and_then(|id| doc.get_node(id))
    else {
        return 0.0;
    };
    if !is_block_flow(flow, parent) {
        return 0.0;
    }
    let walk = walk_breaks(flow, parent, page_height_px, depth);
    let Some(&line) = walk.line_of.get(&node.id) else {
        return 0.0;
    };
    kept_after_line(
        flow,
        parent,
        &walk.children,
        &walk.lines,
        &walk.values,
        line,
        page_height_px,
        depth + 1,
    )
}

/// The top margin of `node` that adjoins its top edge and that a break
/// before it, `forced` or not, keeps: its own `margin-top`, collapsed (CSS
/// 2.1 §8.3.1) with those of the first in-flow boxes whose margins the
/// layout collapsed through its top, down through empty boxes: a first child
/// the layout placed at its parent's content edge (unrounded, as a hairline
/// edge rounds to zero) gave the parent its margin; one placed below it keeps
/// its margin inside. Each box's margin counts only where its `margin-break`
/// keeps it ([`keeps_margin`]). The largest positive margin and the most
/// negative add up, each read from the computed style. Taffy's
/// `final_layout.margin` is no measure of it: Blitz folds a floated first
/// child's margin into its parent's there.
fn adjoining_top_margin(flow: Flow<'_>, node: &blitz_dom::Node, forced: bool) -> f32 {
    let doc = flow.doc;
    let own = |node: &blitz_dom::Node| {
        if !keeps_margin(flow, node, forced) || is_table_internal(node) {
            return 0.0;
        }
        own_margin(doc, node, false)
    };
    let (mut positive, mut negative) = (0.0_f32, 0.0_f32);
    let mut add = |margin: f32| {
        positive = positive.max(margin);
        negative = negative.min(margin);
    };
    let mut current = node;
    add(own(current));
    for _ in 0..crate::MAX_DOM_DEPTH {
        let edge = current.unrounded_layout.border.top + current.unrounded_layout.padding.top;
        let mut next = None;
        for id in walk_children_of(current) {
            let Some(child) = doc.get_node(id) else {
                continue;
            };
            if out_of_flow(child) {
                continue;
            }
            if child.element_data().is_none() || child.unrounded_layout.location.y > edge + 0.01 {
                break;
            }
            add(own(child));
            let layout = child.unrounded_layout;
            let empty = layout.size.height <= 0.0
                && layout.border.top <= 0.0
                && layout.padding.top <= 0.0
                && layout.border.bottom <= 0.0
                && layout.padding.bottom <= 0.0;
            if !empty {
                next = Some(child);
                break;
            }
        }
        let Some(child) = next else {
            break;
        };
        current = child;
    }
    positive + negative
}

/// The computed `margin-top` of `node`, or its `margin-bottom` when
/// `bottom`, in px: a percentage resolves against the width of its
/// containing block's content box, and `auto` is zero.
fn own_margin(doc: &BaseDocument, node: &blitz_dom::Node, bottom: bool) -> f32 {
    use ::style::values::computed::Length;
    use ::style::values::generics::length::GenericMargin;
    let basis = node
        .layout_parent
        .get()
        .or(node.parent)
        .and_then(|id| doc.get_node(id))
        .map_or(0.0, |parent| {
            let layout = parent.final_layout;
            layout.size.width
                - layout.padding.left
                - layout.padding.right
                - layout.border.left
                - layout.border.right
        });
    node.primary_styles().map_or(0.0, |styles| {
        let margins = styles.get_margin();
        let margin = if bottom {
            &margins.margin_bottom
        } else {
            &margins.margin_top
        };
        match margin {
            GenericMargin::LengthPercentage(lp) => lp.resolve(Length::new(basis)).px(),
            _ => 0.0,
        }
    })
}

/// What the bottom margin of `node` keeps below its bottom edge where the
/// box ends near the page foot, by its `-fulgur-page-foot-margin`
/// ([`crate::column_css::PageFootMargin`]): under `page`, the part of the
/// margin beyond the page-foot clearance, which the walk's page strip
/// leaves free already, so that the margin box ends by the page foot; under
/// `clearance`, the whole margin, so that it ends above the clearance;
/// otherwise nothing, as css-break-3 §5.2 truncates the margin at the break.
/// A table's row or cell has no margin.
fn foot_margin(flow: Flow<'_>, node: &blitz_dom::Node) -> f32 {
    use crate::column_css::PageFootMargin;
    let declared = flow
        .column_styles
        .and_then(|styles| styles.get(&node.id))
        .and_then(|props| props.page_foot_margin);
    let margin = || {
        if is_table_internal(node) {
            0.0
        } else {
            own_margin(flow.doc, node, true).max(0.0)
        }
    };
    match declared {
        None | Some(PageFootMargin::None) => 0.0,
        Some(PageFootMargin::Page) => (margin() - flow.foot_clearance).max(0.0),
        Some(PageFootMargin::Clearance) => margin(),
    }
}

/// Whether `node` is a table's row, row group, column, column group, or cell,
/// which CSS gives no margin (CSS 2.1 §17.5, §17.2), whatever its
/// `margin-top`.
fn is_table_internal(node: &blitz_dom::Node) -> bool {
    node.primary_styles().is_some_and(|styles| {
        matches!(
            styles.clone_display().inside(),
            DisplayInside::TableRowGroup
                | DisplayInside::TableHeaderGroup
                | DisplayInside::TableFooterGroup
                | DisplayInside::TableRow
                | DisplayInside::TableColumn
                | DisplayInside::TableColumnGroup
                | DisplayInside::TableCell
        )
    })
}

/// Whether a break before `node`, `forced` or not, keeps the node's own top
/// margin, by its `margin-break` (css-break-4 §5.2): `auto` keeps it after a
/// forced break only, `keep` after any break, `discard` after none.
fn keeps_margin(flow: Flow<'_>, node: &blitz_dom::Node, forced: bool) -> bool {
    use crate::column_css::MarginBreak;
    // The box's own, not its break values' box ([`break_props`]): a table's
    // margin is the table's, inside the box around it and its caption.
    let own = flow.column_styles.and_then(|styles| styles.get(&node.id));
    match own.and_then(|props| props.margin_break) {
        None | Some(MarginBreak::Auto) => forced,
        Some(MarginBreak::Keep) => true,
        Some(MarginBreak::Discard) => false,
    }
}

/// The part of a top `margin` that an unforced break keeps above a unit
/// `unit` px tall, which moves to the top of the next page to start it
/// whole: as much as still leaves the unit its page, so a box that fits a
/// page never runs past its foot for its margin.
fn kept_margin(margin: f32, unit: f32, page_height_px: f32) -> f32 {
    margin.max(0.0).min((page_height_px - unit).max(0.0))
}

/// Moves `child`, whose top in its parent's layout is `top`, to the top of
/// the next page at an unforced break ([`PageCursor::move_before`]), and
/// returns its y there: below the top margin a break keeps
/// ([`adjoining_top_margin`], zero unless a `margin-break` is `keep`), as
/// far as the leading `unit` of the child still fits below it
/// ([`kept_margin`]).
fn move_on(
    cursor: &mut PageCursor,
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    child: &blitz_dom::Node,
    top: f32,
    unit: f32,
) -> f32 {
    let margin = kept_margin(
        adjoining_top_margin(flow, child, false),
        unit,
        cursor.page_height_px,
    );
    cursor.move_before(geometry, top - margin, false);
    margin
}

/// Whether `node`, which the layout gives no height, holds a line all the
/// same: text of only white space that CSS does not collapse, such as
/// no-break spaces (only spaces, tabs, and line breaks collapse). Blitz lays
/// such a line out with no height, so the walk counts it as the content of
/// its page, as a browser's line box would be: the forced breaks around it
/// are taken ([`PageCursor::at_page_top`]).
fn holds_a_dropped_line(node: &blitz_dom::Node) -> bool {
    node.element_data()
        .is_some_and(|el| el.inline_layout_data.is_some())
        && {
            let text = node.text_content();
            !text.is_empty()
                && text.chars().all(char::is_whitespace)
                && text.chars().any(|c| !c.is_ascii_whitespace())
        }
}

/// The height `node` takes on the page it starts on, which every fit the
/// walk measures reads: its layout height, less the header rows below it
/// that skip their table's first page ([`first_page_lift`]).
fn extent(flow: Flow<'_>, node: &blitz_dom::Node, depth: usize) -> f32 {
    node.final_layout.size.height - first_page_lift(flow, node, depth)
}

/// The height of the header rows below `node` that skip their table's
/// first page (`-fulgur-header-repeat: skip-first`), which
/// [`fragment_repeating_table`] leaves off that page and the boxes after
/// them lift into: a table's own, else its children's, added in block flow
/// and the largest of boxes side by side.
fn first_page_lift(flow: Flow<'_>, node: &blitz_dom::Node, depth: usize) -> f32 {
    let Flow {
        doc, column_styles, ..
    } = flow;
    if depth >= crate::MAX_DOM_DEPTH || !skipped_header_row_below(doc, node.id, column_styles, 0) {
        return 0.0;
    }
    if let Some(header) = repeating_table_header(doc, node.id, depth + 1, column_styles) {
        return header.first_page_lift_px;
    }
    let lifts = in_flow_children(flow, node)
        .into_iter()
        .filter_map(|id| doc.get_node(id))
        .map(|child| first_page_lift(flow, child, depth + 1));
    if crate::blitz_adapter::is_flex_or_grid_container_node(node) {
        lifts.fold(0.0, f32::max)
    } else {
        lifts.sum()
    }
}

/// Whether a header row that skips its table's first page
/// (`-fulgur-header-repeat: skip-first`) lies below `node`: only the table's
/// own fragmentation ([`fragment_repeating_table`]) leaves it off that page,
/// so the walk enters a box holding one.
fn skipped_header_row_below(
    doc: &BaseDocument,
    node_id: usize,
    column_styles: Option<&crate::column_css::ColumnStyleTable>,
    depth: usize,
) -> bool {
    if depth >= crate::MAX_DOM_DEPTH {
        return false;
    }
    let Some(node) = doc.get_node(node_id) else {
        return false;
    };
    node.children.iter().any(|&child_id| {
        !doc.get_node(child_id).is_some_and(out_of_flow)
            && (column_styles
                .and_then(|styles| styles.get(&child_id))
                .is_some_and(|props| {
                    props.header_repeat == Some(crate::column_css::HeaderRepeat::SkipFirst)
                })
                || skipped_header_row_below(doc, child_id, column_styles, depth + 1))
    })
}

/// A line of boxes side by side that the walk places as one unit
/// (`BoxKind::Parallel`): a table row's cells, or a flex or grid line's
/// items ([`flow_lines`]). Whether it starts on this page is decided once,
/// before its first box ([`line_moves_on`]); then every box starts on the
/// line's first page at the line's top (fulgur-ysms), each continues as far
/// as its content reaches, and the walk goes on below the box that ends
/// last ([`close_line`]).
struct LineState {
    /// The line's number among the walk's children ([`flow_lines`]).
    number: usize,
    /// The cursor where the line starts, which each of its boxes restores.
    start: PageMark,
    started_on_occupied_strip: bool,
    max_end_page: u32,
    max_end_cursor_y: f32,
    /// A box placed content on `max_end_page`.
    max_end_has_content: bool,
    /// The bottom of the line's own boxes in their parent's layout: a box
    /// that reaches it is stretched to the line's height. A cell that
    /// spans rows reaches below its row and is not one.
    bottom: f32,
    /// The boxes placed so far.
    items: Vec<usize>,
}

impl LineState {
    /// Records where a box of the line ended, at `y` on `page`, when it is
    /// past every other, and whether the line placed content on its last
    /// page.
    fn reach(&mut self, page: u32, y: f32, has_content: bool) {
        if page > self.max_end_page {
            self.max_end_page = page;
            self.max_end_cursor_y = y;
            self.max_end_has_content = has_content;
        } else if page == self.max_end_page {
            self.max_end_cursor_y = self.max_end_cursor_y.max(y);
            self.max_end_has_content |= has_content;
        }
    }
}

/// Where a table cell that spans rows ended, which ends the last row it
/// spans rather than its own ([`reach_line`]).
struct SpanEnd {
    /// The cell.
    id: usize,
    /// The cells of the rows it spans that the walk closed so far, but
    /// those that span rows themselves ([`stretch_span`]).
    cells: Vec<usize>,
    /// The line of the last row the cell spans ([`flow_lines`]).
    last_line: usize,
    page: u32,
    y: f32,
    has_content: bool,
}

/// The rows the table cell `node` spans by its `rowspan`, the one place it
/// is read: 1 for a node that is no cell, or a value that is no positive
/// count. Blitz ends a span with its row group ([`span_end`]).
fn rowspan_of(node: &blitz_dom::Node) -> usize {
    if !displays(node, DisplayInside::TableCell) {
        return 1;
    }
    node.element_data()
        .and_then(|el| crate::blitz_adapter::get_attr(el, "rowspan"))
        .and_then(|value| value.parse::<u16>().ok())
        .map_or(1, |rowspan| usize::from(rowspan.max(1)))
}

/// The rows of the row group `row` belongs to, and its index among them.
fn group_rows<'a>(
    doc: &'a BaseDocument,
    row: &blitz_dom::Node,
) -> Option<(Vec<&'a blitz_dom::Node>, usize)> {
    let group = doc.get_node(row.parent?)?;
    let rows: Vec<&blitz_dom::Node> = group
        .children
        .iter()
        .filter_map(|&id| doc.get_node(id))
        .filter(|&node| displays(node, DisplayInside::TableRow))
        .collect();
    let index = rows.iter().position(|node| node.id == row.id)?;
    Some((rows, index))
}

/// The index of the last of `rows`, a row group's, that the rows from
/// `from` reach by their cells' `rowspan`: a cell spans its own row and the
/// ones below it its count gives, and a cell of a row reached so spans on
/// in turn (staggered spans chain). A span ends with its group. `from` when
/// no cell spans rows.
fn span_end(doc: &BaseDocument, rows: &[&blitz_dom::Node], from: usize) -> usize {
    let mut end = from;
    let mut at = from;
    while at <= end && at < rows.len() {
        for cell in row_cells(doc, rows[at]) {
            end = end.max(at + rowspan_of(cell) - 1);
        }
        at += 1;
    }
    end.min(rows.len().saturating_sub(1))
}

/// Records where the box `child` of the open line ended, at the cursor
/// ([`LineState::reach`]): a cell that spans rows ends the last row it
/// spans, not its own, as the rows below it are placed beside it
/// ([`SpanEnd`]).
fn reach_line(
    line: &mut Option<LineState>,
    spans: &mut Vec<SpanEnd>,
    child: &blitz_dom::Node,
    cursor: &PageCursor,
) {
    let Some(open) = line.as_mut() else {
        return;
    };
    match rowspan_of(child) {
        1 => open.reach(cursor.page_index, cursor.cursor_y, cursor.page_has_content),
        rowspan => spans.push(SpanEnd {
            id: child.id,
            cells: Vec::new(),
            last_line: open.number + rowspan - 1,
            page: cursor.page_index,
            y: cursor.cursor_y,
            has_content: cursor.page_has_content,
        }),
    }
}

/// The table row of the cell `node`, or `None` for a node that is not a
/// cell of a row.
fn table_row_of<'a>(doc: &'a BaseDocument, node: &blitz_dom::Node) -> Option<&'a blitz_dom::Node> {
    if !displays(node, DisplayInside::TableCell) {
        return None;
    }
    doc.get_node(node.parent?)
        .filter(|&row| displays(row, DisplayInside::TableRow))
}

/// The bottom of the line of `items` in their parent's layout ([`LineState`]):
/// of a table row, its cells' that span no rows, else of every box.
fn line_bottom(items: &[&blitz_dom::Node]) -> f32 {
    let bottom =
        |node: &&blitz_dom::Node| node.final_layout.location.y + node.final_layout.size.height;
    items
        .iter()
        .filter(|item| rowspan_of(item) == 1)
        .map(bottom)
        .reduce(f32::max)
        .or_else(|| items.iter().map(bottom).reduce(f32::min))
        .unwrap_or(0.0)
}

/// The leading unit of the line of boxes side by side `items`, whose
/// layout top is `top`, with `room` px left below it ([`leading_extent`]):
/// a table row's, else, for more than one box, their shortest
/// ([`tracks_leading`]), the line placing content when it fits the room
/// whole. `None` for a line of one box that is no cell.
fn line_leading(
    flow: Flow<'_>,
    items: &[&blitz_dom::Node],
    top: f32,
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Option<Leading> {
    let &first = items.first()?;
    if let Some(row) = table_row_of(flow.doc, first) {
        return Some(leading_extent(flow, row, room, page_height_px, 0));
    }
    if items.len() < 2 {
        return None;
    }
    let ids: Vec<usize> = items.iter().map(|item| item.id).collect();
    let lead = tracks_leading(flow, &ids, top, room, page_height_px, depth)?;
    Some(Leading {
        places: lead.places || line_bottom(items) - top <= room,
        ..lead
    })
}

/// Whether the line of `items`, whose layout top is `top`, moves whole to
/// the next page before its first box, starting `line_y` down a page of
/// `page_height_px` ([`LineState`]).
///
/// - A table row moves when none of its cells places content in the room
///   left ([`leading_extent`]): a cell has no break point between its top
///   edge and its first content. A row kept whole moves when it does not
///   fit. At the top of a page, below its table's top edge, a row moves only
///   when its unit fits a whole page and is not held together by `avoid`
///   alone (css-break-3 §4.4). A row a cell above spans down to moves with
///   that cell's row only ([`spanned_from_above`]), and the table's first
///   row does not move at the top of its fragment.
/// - A flex or grid line after content in its container moves when none of
///   its items places content and its unit fits a page, or when an item
///   placed whole that is no slice of a box does not fit the room but fits
///   a page; at the top of a page not when `avoid` alone holds it.
#[allow(clippy::too_many_arguments)]
fn line_moves_on(
    flow: Flow<'_>,
    items: &[&blitz_dom::Node],
    top: f32,
    line_y: f32,
    after_content: bool,
    at_page_top: bool,
    page_height_px: f32,
    depth: usize,
) -> bool {
    let Flow { doc, .. } = flow;
    let Some(&first) = items.first() else {
        return false;
    };
    let room = page_height_px - line_y;
    if let Some(row) = table_row_of(doc, first) {
        let movable = kept_table_row_bottom(flow, first, page_height_px).is_some()
            || !spanned_from_above(flow, row);
        if !movable || line_y <= 0.0 {
            return false;
        }
        let lead = leading_extent(flow, row, room, page_height_px, 0);
        return if at_page_top {
            lead.moves_on(flow.page_foot(page_height_px)) && !lead.relaxes
        } else {
            !lead.places
        };
    }
    if !after_content {
        return false;
    }
    let ids: Vec<usize> = items.iter().map(|item| item.id).collect();
    let Some(lead) = tracks_leading(flow, &ids, top, room, page_height_px, depth) else {
        return false;
    };
    let lead = Leading {
        places: lead.places || line_bottom(items) - top <= room,
        ..lead
    };
    let monolithic_overflow = !at_page_top
        && items.iter().any(|item| {
            let offset = item.final_layout.location.y - top;
            let height = item.final_layout.size.height;
            matches!(
                classify(flow, item, page_height_px),
                BoxKind::Monolithic(Whole::Atomic)
            ) && !slices_at_a_page_top(item)
                && offset + height > room
                && flow.fits_page(offset + height, page_height_px)
        });
    monolithic_overflow
        || (lead.moves_on(flow.page_foot(page_height_px)) && !(at_page_top && lead.relaxes))
}

/// Ends the line `line` of a walk: each of its boxes stretched to the
/// line's height ([`stretch_line_items`]), the cursor moved below the box
/// that ends last, on its page, the cells that span rows down to it among
/// them (`spans`, all that are left when it is the walk's `last` line),
/// each stretched to the rows it spans ([`stretch_span`]).
/// Returns the y at which the next line starts when the line crossed pages,
/// or a spanning cell's content ended it below its own boxes: right below
/// it, the gap above the next line dropped, as below a box that crossed.
fn close_line(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    mut line: LineState,
    spans: &mut Vec<SpanEnd>,
    last: bool,
    cursor: &mut PageCursor,
) -> Option<f32> {
    cursor.close_row();
    let cells: Vec<usize> = line
        .items
        .iter()
        .copied()
        .filter(|&id| doc.get_node(id).is_some_and(|node| rowspan_of(node) == 1))
        .collect();
    let own_end = (line.max_end_page, line.max_end_cursor_y);
    let mut ended = Vec::new();
    spans.retain_mut(|span| {
        span.cells.extend(&cells);
        let ends = last || span.last_line <= line.number;
        if ends {
            line.reach(span.page, span.y, span.has_content);
            ended.push((span.id, std::mem::take(&mut span.cells)));
        }
        !ends
    });
    stretch_line_items(geometry, doc, &line, cursor.page_height_px);
    for (id, cells) in ended {
        stretch_span(geometry, doc, id, &cells, &line, cursor.page_height_px);
    }
    let next = cursor.after_line(&line);
    // A spanning cell whose content ends past its rows, on their last page
    // too, as when its content moved to the next page, ends them there: the
    // next line starts below it, not where the layout puts it.
    let past = line.max_end_page > own_end.0 || line.max_end_cursor_y > own_end.1 + 0.5;
    next.or(past.then_some(line.max_end_cursor_y))
}

/// Gives each box of a line that crossed pages, stretched to the line's
/// height by a taller neighbour (it reaches [`LineState::bottom`]), but a
/// cell that spans rows ([`stretch_span`]), a
/// fragment on every page down to the line's end: cut at the foot of each
/// page before the line's last, where a box placed whole ran past it, and
/// on the last page reaching the line's end, as an empty fragment where its
/// own content ended before. A replaced or transformed box is not cut.
fn stretch_line_items(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    line: &LineState,
    page_height_px: f32,
) {
    let (end_page, end_y) = (line.max_end_page, line.max_end_cursor_y);
    if end_page <= line.start.page_index {
        return;
    }
    for &id in &line.items {
        let Some(node) = doc.get_node(id) else {
            continue;
        };
        let layout = node.final_layout;
        if layout.location.y + layout.size.height < line.bottom - 0.5
            || rowspan_of(node) > 1
            || is_transformed(node)
            || is_replaced(node)
        {
            continue;
        }
        stretch_box(geometry, node, end_page, |page| {
            if page < end_page {
                page_height_px
            } else {
                end_y
            }
        });
    }
}

/// Gives the cell `id` that spans rows, which started on a page before the
/// end of `line`, the last row it spans, a fragment on every page down to
/// it: on each page ending with the
/// lowest of the `cells` of its rows there, or of its own content, cut
/// where the cell placed whole ran past them, and on the last page reaching
/// the line's end.
fn stretch_span(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    id: usize,
    cells: &[usize],
    line: &LineState,
    page_height_px: f32,
) {
    let (end_page, end_y) = (line.max_end_page, line.max_end_cursor_y);
    let Some(node) = doc.get_node(id) else {
        return;
    };
    let Some(first_page) = geometry
        .get(&id)
        .and_then(|entry| entry.fragments.first())
        .map(|fragment| fragment.page_index)
    else {
        return;
    };
    if first_page >= end_page || is_transformed(node) {
        return;
    }
    let content = walk_children_of(node);
    let ends: BTreeMap<u32, f32> = (first_page..end_page)
        .map(|page| {
            let end = cells
                .iter()
                .chain(&content)
                .filter_map(|id| geometry.get(id))
                .flat_map(|entry| &entry.fragments)
                .filter(|fragment| fragment.page_index == page)
                .map(|fragment| fragment.y.to_f32() + fragment.height.to_f32())
                .reduce(f32::max);
            (page, end.unwrap_or(page_height_px))
        })
        .collect();
    stretch_box(geometry, node, end_page, |page| {
        ends.get(&page).copied().unwrap_or(end_y)
    });
}

/// Gives the box `node` a fragment on every page from its first down to
/// `end_page`, each ending at `bottom` of its page: before the last cut
/// there, on the last at least there, as an empty fragment where its own
/// content ended before. The lines of an inline root are on its first
/// fragments; the added ones hold none (`line_boundaries`).
fn stretch_box(
    geometry: &mut PaginationGeometryTable,
    node: &blitz_dom::Node,
    end_page: u32,
    bottom: impl Fn(u32) -> f32,
) {
    let Some(entry) = geometry.get_mut(&node.id) else {
        return;
    };
    let Some(first) = entry.fragments.first().cloned() else {
        return;
    };
    let lines = entry.line_boundaries.last().copied().or_else(|| {
        node.element_data()
            .is_some_and(|el| el.inline_layout_data.is_some())
            .then(|| collect_inline_line_metrics(node).len())
    });
    let before = entry.fragments.len();
    for page in first.page_index..=end_page {
        let bottom = bottom(page);
        if let Some(fragment) = entry
            .fragments
            .iter_mut()
            .rev()
            .find(|fragment| fragment.page_index == page)
        {
            let top = fragment.y.to_f32();
            let height = if page < end_page {
                bottom - top
            } else {
                fragment.height.to_f32().max(bottom - top)
            };
            fragment.height = height.max(0.0).as_px();
        } else {
            entry.fragments.push(Fragment {
                page_index: page,
                x: first.x,
                y: 0.0_f32.as_px(),
                width: first.width,
                height: bottom.max(0.0).as_px(),
            });
        }
    }
    let added = entry.fragments.len() - before;
    if added > 0 {
        entry.fragments.sort_by_key(|fragment| fragment.page_index);
        if let Some(lines) = lines {
            if entry.line_boundaries.is_empty() && before == 1 {
                entry.line_boundaries = vec![0, lines];
            }
            if entry.line_boundaries.len() == before + 1 {
                entry
                    .line_boundaries
                    .extend(std::iter::repeat_n(lines, added));
            }
        }
    }
}

/// Whether `node` is a replaced element, which paints as one image.
fn is_replaced(node: &blitz_dom::Node) -> bool {
    const REPLACED: [&str; 9] = [
        "img", "svg", "canvas", "video", "iframe", "object", "embed", "input", "textarea",
    ];
    node.element_data()
        .is_some_and(|el| REPLACED.contains(&el.name.local.as_ref()))
}

/// How tall a walker makes the fragment of its parent on the page it
/// closes, from the fragment's top (`PageCursor::page_start_y`).
#[derive(Clone, Copy, Debug)]
enum ParentHeight {
    /// Down to the lowest point placed on the page (`PageCursor::cursor_y`).
    Placed,
    /// Down to the foot of the page strip.
    ToFoot,
    /// A height the walker measured: a child that crossed the page.
    Measured(f32),
}

/// The box a nested walk fills, whose fragments [`PageCursor`] closes.
#[derive(Debug)]
struct ParentSink {
    id: usize,
    x: f32,
    width: f32,
    /// While a flex or grid row is open, the pages its cells already closed
    /// the parent on when they moved on: cells that leave the same page
    /// close it once.
    row_pages: Option<BTreeSet<u32>>,
}

/// The parts of a [`PageCursor`] a flex or grid row restores for each of
/// its cells, so that each starts where the row's first cell did.
#[derive(Clone, Copy, Debug)]
struct PageMark {
    page_index: u32,
    cursor_y: f32,
    page_start_y: f32,
    page_taffy_origin: f32,
    entered_below_content: bool,
    page_has_content: bool,
}

/// A walker's place in the page sequence: the page it fills, the lowest
/// point placed on it, and, for a nested walk, its parent's fragment on
/// that page. The walker moves only through it: before a box
/// ([`Self::move_before`], [`Self::break_before`]) and after a child that
/// crossed pages ([`Self::after_crossing`]), each by [`Self::advance`]; every
/// fragment of the parent goes through [`Self::close_parent`] or
/// [`Self::close_parent_once`]; and it answers whether a break point
/// precedes a box ([`Self::break_point_before`]).
#[derive(Debug)]
struct PageCursor {
    page_index: u32,
    /// The lowest point placed on the page, in page-strip coordinates.
    cursor_y: f32,
    /// Where the parent's fragment on this page starts.
    page_start_y: f32,
    /// The parent-relative layout y placed at `page_start_y` (fulgur-kv0r):
    /// a child's page y is `page_start_y + (location.y - page_taffy_origin)`.
    page_taffy_origin: f32,
    /// The page the walk entered its parent on.
    entry_page: u32,
    /// The walk entered on a page holding content above it, and has not
    /// placed its first child yet (the first row's cells, side by side,
    /// each start so): the parent's fragment on a page a child crosses is
    /// then measured from what its children placed there.
    entered_below_content: bool,
    /// An in-flow box with height is placed on this page, at any depth:
    /// the page is no longer empty ([`Self::at_page_top`]).
    page_has_content: bool,
    /// The height of the page strip the walk fills.
    page_height_px: f32,
    /// The page's foot, below the strip's by the page-foot clearance
    /// ([`Flow::page_foot`]): where the parent's fragment on a page a child
    /// crossed ends, as a child taller than a page is sliced there.
    page_foot_px: f32,
    parent: Option<ParentSink>,
}

impl PageCursor {
    /// A cursor at `cursor_y` on `page_index`, closing no parent.
    fn new(page_index: u32, cursor_y: f32, page_height_px: f32) -> Self {
        Self {
            page_index,
            cursor_y,
            page_start_y: cursor_y,
            page_taffy_origin: 0.0,
            entry_page: page_index,
            entered_below_content: false,
            page_has_content: false,
            page_height_px,
            page_foot_px: page_height_px,
            parent: None,
        }
    }

    /// Nothing is placed on this page yet: no in-flow box with height, at
    /// any depth, above the walk's place (the boxes beside it, a flex or
    /// grid row's items or a table row's cells, do not count). A box's
    /// padding or border, or a margin, is no content. A break here would
    /// leave the page with nothing but such edges, so a paragraph splits
    /// within relaxed `orphans` and `widows` rather than move whole
    /// (css-break-3 §4.4), and a side break only moves on to its side.
    /// Whether a break point precedes a box at all is the walker's own
    /// question: it still moves a box that does not fit below another, or
    /// below a gap or edge, as before.
    fn at_page_top(&self) -> bool {
        !self.page_has_content
    }

    /// Records that a box of `height` is placed on this page.
    fn place(&mut self, height: f32) {
        self.page_has_content |= height > 0.0;
    }

    /// Moves to the top of the next page.
    fn advance(&mut self) {
        self.page_index += 1;
        self.cursor_y = 0.0;
        self.page_start_y = 0.0;
        self.page_has_content = false;
    }

    /// Moves on to the page of `side` when this page is not on it (a side
    /// break, [`page_on_side`]).
    fn move_to_side(&mut self, doc: &BaseDocument, side: Option<crate::draw_primitives::PageSide>) {
        self.page_index = page_on_side(doc, self.page_index, side);
    }

    /// A forced break: moves to the next page, then on to `side`'s.
    fn break_to(&mut self, doc: &BaseDocument, side: Option<crate::draw_primitives::PageSide>) {
        self.advance();
        self.move_to_side(doc, side);
    }

    /// A forced break before a box whose top in the parent's layout is
    /// `top`: closes the parent's fragment at what is placed, breaks to the
    /// page of `side` ([`Self::break_to`]), and places the box's top at the
    /// new page's top.
    fn break_before(
        &mut self,
        geometry: &mut PaginationGeometryTable,
        doc: &BaseDocument,
        side: Option<crate::draw_primitives::PageSide>,
        top: f32,
    ) {
        self.close_parent(geometry, ParentHeight::Placed);
        self.break_to(doc, side);
        self.page_taffy_origin = top;
    }

    /// Moves a box whose top in the parent's layout is `top` to the top of
    /// the next page: closes the parent's fragment on this page at what is
    /// placed, unless `leaves_nothing` (a table's first row leaves no
    /// fragment of the table behind), and places the box's top at the new
    /// page's top, as the first the walk places there.
    fn move_before(
        &mut self,
        geometry: &mut PaginationGeometryTable,
        top: f32,
        leaves_nothing: bool,
    ) {
        if !leaves_nothing {
            self.close_parent_once(geometry, ParentHeight::Placed);
        }
        self.advance();
        self.page_taffy_origin = top;
        self.entered_below_content = false;
    }

    /// A child that started on this page ended on the later `end_page`, at
    /// `end_y`: closes the parent's fragment on this page `height` tall when
    /// positive (growing the one a box beside it closed), gives the parent
    /// a full fragment on each page between, and goes on below the child,
    /// the parent's fragment on `end_page` starting at the page top.
    fn after_crossing(
        &mut self,
        geometry: &mut PaginationGeometryTable,
        end_page: u32,
        end_y: f32,
        height: f32,
    ) {
        if height > 0.0 && !self.close_parent_once(geometry, ParentHeight::Measured(height)) {
            self.grow_parent(geometry, height);
        }
        self.fill_parent_pages(geometry, end_page);
        self.page_index = end_page;
        self.cursor_y = end_y;
        self.page_start_y = 0.0;
    }

    /// Goes on below a paragraph split or a box sliced from this page, which
    /// ended at `end_y` on `end_page`: past the pages it crossed
    /// ([`Self::after_crossing`], the parent's fragment on this page
    /// reaching the foot), or on this page, below it, or in a line of
    /// boxes side by side (`in_line`) below the lowest of them, so that a
    /// shorter box followed by `break-after: page` closes the parent at the
    /// line's true extent (codex P2). Returns whether it crossed pages.
    fn after_box(
        &mut self,
        geometry: &mut PaginationGeometryTable,
        end_page: u32,
        end_y: f32,
        in_line: bool,
    ) -> bool {
        if end_page > self.page_index {
            let height = self.parent_height(ParentHeight::ToFoot);
            self.after_crossing(geometry, end_page, end_y, height);
            return true;
        }
        self.cursor_y = if in_line {
            self.cursor_y.max(end_y)
        } else {
            end_y
        };
        false
    }

    /// Goes on below the line `line` of boxes side by side, which ends
    /// below the box that ends last, on its page ([`close_line`]). Returns
    /// the y at which the next line starts when the line crossed pages.
    fn after_line(&mut self, line: &LineState) -> Option<f32> {
        self.page_index = line.max_end_page;
        self.cursor_y = line.max_end_cursor_y;
        self.page_has_content = line.max_end_has_content;
        if line.max_end_page > line.start.page_index {
            self.page_start_y = 0.0;
            Some(line.max_end_cursor_y)
        } else {
            self.page_start_y = line.start.page_start_y;
            None
        }
    }

    /// Whether a break point precedes a box placed at `y` on this page in
    /// the parent's fragment (css-break-3 §4.1): a sibling or a gap lies
    /// above it there, or the walk entered its parent below content on the
    /// page it entered on. With none, a break before the box would leave
    /// only the parent's top edge behind.
    fn break_point_before(&self, y: f32) -> bool {
        y > self.page_start_y || (self.entered_below_content && self.page_index == self.entry_page)
    }

    /// Whether the walk has placed anything of the parent on this page.
    fn holds_placed(&self) -> bool {
        self.cursor_y > self.page_start_y
    }

    fn mark(&self) -> PageMark {
        PageMark {
            page_index: self.page_index,
            cursor_y: self.cursor_y,
            page_start_y: self.page_start_y,
            page_taffy_origin: self.page_taffy_origin,
            entered_below_content: self.entered_below_content,
            page_has_content: self.page_has_content,
        }
    }

    fn restore(&mut self, mark: PageMark) {
        self.page_index = mark.page_index;
        self.cursor_y = mark.cursor_y;
        self.page_start_y = mark.page_start_y;
        self.page_taffy_origin = mark.page_taffy_origin;
        self.entered_below_content = mark.entered_below_content;
        self.page_has_content = mark.page_has_content;
    }

    /// Opens a flex or grid row: until [`Self::close_row`], the parent is
    /// closed once per page by [`Self::close_parent_once`].
    fn open_row(&mut self) {
        if let Some(parent) = &mut self.parent {
            parent.row_pages = Some(BTreeSet::new());
        }
    }

    fn close_row(&mut self) {
        if let Some(parent) = &mut self.parent {
            parent.row_pages = None;
        }
    }

    fn parent_height(&self, height: ParentHeight) -> f32 {
        match height {
            ParentHeight::Placed => self.cursor_y - self.page_start_y,
            ParentHeight::ToFoot => (self.page_foot_px - self.page_start_y).max(0.0),
            ParentHeight::Measured(height) => height,
        }
    }

    fn push_parent(
        &self,
        geometry: &mut PaginationGeometryTable,
        page_index: u32,
        y: f32,
        height: f32,
    ) {
        let Some(parent) = &self.parent else {
            return;
        };
        let fragments = &mut geometry.entry(parent.id).or_default().fragments;
        // The lines of a table or a flex or grid container close the
        // parent on a page each time one of them crosses it: its fragment
        // there is one, reaching the lowest of them.
        if let Some(fragment) = fragments
            .iter_mut()
            .rev()
            .find(|fragment| fragment.page_index == page_index)
        {
            let top = fragment.y.to_f32().min(y);
            let bottom = (fragment.y.to_f32() + fragment.height.to_f32()).max(y + height);
            fragment.y = top.as_px();
            fragment.height = (bottom - top).as_px();
            return;
        }
        fragments.push(Fragment {
            page_index,
            x: parent.x.as_px(),
            y: y.as_px(),
            width: parent.width.as_px(),
            height: height.as_px(),
        });
    }

    /// Closes the parent's fragment on this page.
    fn close_parent(&self, geometry: &mut PaginationGeometryTable, height: ParentHeight) {
        self.push_parent(
            geometry,
            self.page_index,
            self.page_start_y,
            self.parent_height(height),
        );
    }

    /// Closes the parent's fragment on this page unless a cell of the open
    /// row already did; returns whether it did now.
    fn close_parent_once(
        &mut self,
        geometry: &mut PaginationGeometryTable,
        height: ParentHeight,
    ) -> bool {
        let page_index = self.page_index;
        let first = self
            .parent
            .as_mut()
            .and_then(|parent| parent.row_pages.as_mut())
            .is_none_or(|pages| pages.insert(page_index));
        if first {
            self.close_parent(geometry, height);
        }
        first
    }

    /// Grows the parent's fragment on this page, closed already by a cell
    /// of the open row, to `height` at least.
    fn grow_parent(&self, geometry: &mut PaginationGeometryTable, height: f32) {
        let Some(parent) = &self.parent else {
            return;
        };
        if let Some(fragment) = geometry
            .entry(parent.id)
            .or_default()
            .fragments
            .iter_mut()
            .find(|fragment| fragment.page_index == self.page_index)
        {
            fragment.height = fragment.height.to_f32().max(height).as_px();
        }
    }

    /// Gives the parent a full-page fragment on each page after this one
    /// and before `end_page`, which a child crossed, once per open row.
    fn fill_parent_pages(&mut self, geometry: &mut PaginationGeometryTable, end_page: u32) {
        for page_index in (self.page_index + 1)..end_page {
            let first = self
                .parent
                .as_mut()
                .and_then(|parent| parent.row_pages.as_mut())
                .is_none_or(|pages| pages.insert(page_index));
            if first {
                self.push_parent(geometry, page_index, 0.0, self.page_foot_px);
            }
        }
    }
}

/// The bottom edge, in its table's coordinates, of the table row that `cell`
/// belongs to when that row is kept whole and fits a strip of
/// `page_height_px`. `None` otherwise: a row taller than a page still has to
/// split.
///
/// A row has no box of its own in Blitz's layout of a table, a grid of its
/// cells, so its cells carry its `avoid`, and it spans their extent. A row is
/// kept whole when it carries `break-inside: avoid` (css-break-3 §3.3) or
/// holds a cell that spans rows: such a cell reaches down through the rows
/// it spans ([`span_end`]), so they move with it and the cell is never cut
/// at a page foot apart from its rows; a cell of a spanned row that spans
/// further takes its own rows along too.
fn kept_table_row_bottom(
    flow: Flow<'_>,
    cell: &blitz_dom::Node,
    page_height_px: f32,
) -> Option<f32> {
    let Flow {
        doc, column_styles, ..
    } = flow;
    let row = doc.get_node(cell.parent?)?;
    if !displays(row, DisplayInside::TableRow) {
        return None;
    }
    let avoid = column_styles
        .and_then(|styles| styles.get(&row.id))
        .is_some_and(|props| {
            matches!(
                props.break_inside,
                Some(crate::draw_primitives::BreakInside::Avoid)
            )
        });
    let (group, index) = flow.pass.row_group(flow, row)?;
    let end = group.reach[index];
    if end == index && !avoid {
        return None;
    }
    let (top, bottom) = group.rows[index..=end]
        .iter()
        .filter_map(|&id| doc.get_node(id))
        .flat_map(|row| row_cells(doc, row))
        .map(|node| {
            let layout = node.final_layout;
            (layout.location.y, layout.location.y + layout.size.height)
        })
        .reduce(|(top, bottom), (y, end)| (top.min(y), bottom.max(end)))?;
    flow.fits_page(bottom - top, page_height_px)
        .then_some(bottom)
}

/// What a box is to the fragmenter: whether the walk places it whole,
/// splits it between its lines, or enters it for its children, and where
/// its leading unit ends ([`leading_extent`]). Every branch of the walk
/// asks this one question.
#[derive(Debug)]
enum BoxKind {
    /// Placed whole, never split between its children or its lines.
    Monolithic(Whole),
    /// An inline root, split between its `lines` within its `limits`.
    InlineRoot {
        lines: Vec<(f32, f32)>,
        limits: LineLimits,
    },
    /// A block container whose children lie one below the other.
    BlockFlow,
    /// Boxes side by side: the items of a flex or grid container's first
    /// line, or the cells of a table row. `top` is the box's own top in
    /// their coordinates: zero for a container, the row's top in its table
    /// for a row, which has no box of its own.
    Parallel { tracks: Vec<usize>, top: f32 },
    /// A table: `table` itself, or the box the caption pass wraps around
    /// it and its caption ([`crate::blitz_adapter::captioned_table_of`]).
    Table { table: usize },
}

/// Why a box is placed whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Whole {
    /// Never entered: a replaced or childless box, a transformed box that
    /// fits a page (it paints as one shape, CSS Transforms §6.1), a
    /// multicol container without a spanning child (`multicol_layout`
    /// fragments it), an atomic inline, or a box in an orthogonal flow.
    Atomic,
    /// Kept by `break-inside: avoid`, its own or its table row's
    /// ([`kept_table_row_bottom`]), and fitting a page: entered only for a
    /// forced break or a skipped header row below it, which `avoid` does
    /// not concern.
    Kept,
}

/// What `node` is to the fragmenter on pages `page_height_px` tall
/// ([`BoxKind`]).
fn classify(flow: Flow<'_>, node: &blitz_dom::Node, page_height_px: f32) -> BoxKind {
    let Flow { doc, .. } = flow;
    // An atomic inline or a box in an orthogonal flow is monolithic (CSS
    // Writing Modes 4 §9), which the walk did not decide before `classify`:
    // an orthogonal inline root was split between its lines. Neither reaches
    // the walk today: a walked child is a block box (an inline-level one is
    // wrapped in an anonymous block, a flex or grid item blockified), and the
    // engine's style system computes every `writing-mode` as `horizontal-tb`,
    // so no test can build one.
    if is_atomic(doc, node, flow.page_foot(page_height_px)) {
        return BoxKind::Monolithic(Whole::Atomic);
    }
    let avoid = !displays(node, DisplayInside::TableRow)
        && break_props(flow, node).is_some_and(|props| {
            matches!(
                props.break_inside,
                Some(crate::draw_primitives::BreakInside::Avoid)
            )
        });
    if (avoid && flow.fits_page(extent(flow, node, 0), page_height_px))
        || kept_table_row_bottom(flow, node, page_height_px).is_some()
    {
        return BoxKind::Monolithic(Whole::Kept);
    }
    structure(flow, node)
}

/// What `node`, entered, holds ([`BoxKind`]), whether it is kept whole or
/// not: lines, columns, a table, boxes side by side, or block flow.
fn structure(flow: Flow<'_>, node: &blitz_dom::Node) -> BoxKind {
    let Flow {
        doc, column_styles, ..
    } = flow;
    // A row has no box of its own: its cells carry its `avoid`.
    if displays(node, DisplayInside::TableRow) {
        let tracks: Vec<usize> = row_cells(doc, node).map(|cell| cell.id).collect();
        let top = row_cells(doc, node)
            .map(|cell| cell.final_layout.location.y)
            .reduce(f32::min)
            .unwrap_or(0.0);
        return BoxKind::Parallel { tracks, top };
    }
    if !is_transformed(node)
        && node
            .element_data()
            .is_some_and(|el| el.inline_layout_data.is_some())
    {
        return BoxKind::InlineRoot {
            lines: collect_inline_line_metrics(node),
            limits: line_limits(doc, column_styles, node.id),
        };
    }
    if is_unspanned_multicol(doc, node) {
        return BoxKind::Monolithic(Whole::Atomic);
    }
    if displays(node, DisplayInside::Table) {
        return BoxKind::Table { table: node.id };
    }
    if let Some(table) = crate::blitz_adapter::captioned_table_of(doc, node) {
        return BoxKind::Table { table };
    }
    if crate::blitz_adapter::is_flex_or_grid_container_node(node) {
        let items = in_flow_children(flow, node);
        let first_bottom = items
            .first()
            .and_then(|&id| doc.get_node(id))
            .map_or(0.0, |first| {
                first.final_layout.location.y + first.final_layout.size.height
            });
        let tracks = items
            .into_iter()
            .filter(|&id| {
                doc.get_node(id)
                    .is_some_and(|item| item.final_layout.location.y < first_bottom - 0.5)
            })
            .collect();
        return BoxKind::Parallel { tracks, top: 0.0 };
    }
    BoxKind::BlockFlow
}

/// Whether `node` is transformed: it paints as one shape (CSS Transforms
/// §6.1), so it is never split while it fits a page, nor sliced.
fn is_transformed(node: &blitz_dom::Node) -> bool {
    node.primary_styles()
        .is_some_and(|s| !s.get_box().transform.0.is_empty())
}

/// The children of `node` in its block flow ([`out_of_flow`]) with
/// height, in layout order: the boxes that lead it
/// ([`leading_of`]) and form its first line ([`structure`]). A running
/// element the layout gives a height is one of them.
fn in_flow_children(flow: Flow<'_>, node: &blitz_dom::Node) -> Vec<usize> {
    walk_children_of(node)
        .into_iter()
        .filter(|&id| {
            flow.doc
                .get_node(id)
                .is_some_and(|child| !out_of_flow(child) && child.final_layout.size.height > 0.0)
        })
        .collect()
}

/// Whether the walk enters `node`, of `kind`, for its children with
/// `room` px left on the page, rather than place it whole: it has
/// children and is not atomic, and a forced break, a page-name change, or
/// a skipped header row lies below it, or (unless it is kept whole) its
/// children would split in the room ([`would_split_block_subtree`]).
fn enters(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    kind: &BoxKind,
    room: f32,
    page_height_px: f32,
) -> bool {
    if node.children.is_empty() || matches!(kind, BoxKind::Monolithic(Whole::Atomic)) {
        return false;
    }
    forced_inside(flow, node, page_height_px, 0)
        || skipped_header_row_below(flow.doc, node.id, flow.column_styles, 0)
        || (!matches!(kind, BoxKind::Monolithic(Whole::Kept))
            && would_split_block_subtree(flow.doc, node.id, room, page_height_px, 0))
}

/// The leading unit of a box: the part from its top with no break
/// opportunity inside, which a break before the box would carry to the
/// next page ([`leading_extent`]).
#[derive(Clone, Copy, Debug)]
struct Leading {
    /// Where the unit ends, below the box's top.
    extent: f32,
    /// The box places content in the room it was asked for: the unit, or
    /// the whole box, fits it.
    places: bool,
    /// Only `avoid` holds the unit together, or it is a childless box that
    /// slices: at the top of a page, below a top edge, the walk breaks
    /// inside it rather than move it on (css-break-3 §4.4): a box kept
    /// whole is entered, a kept row co-splits its cells, a paragraph
    /// relaxes its `orphans` and `widows`, and a childless box is sliced
    /// from where it stands.
    relaxes: bool,
}

impl Leading {
    /// A unit `extent` tall placed whole, in `room` px.
    fn whole(extent: f32, room: f32) -> Self {
        Self {
            extent,
            places: extent <= room,
            relaxes: false,
        }
    }

    /// The same unit, held together only by `avoid` ([`Self::relaxes`]).
    fn relaxed(self) -> Self {
        Self {
            relaxes: true,
            ..self
        }
    }

    /// Whether the box this leads moves to the next page whole instead of
    /// starting here: nothing of it places in the room, and its unit fits
    /// an empty page, `page_foot_px` tall ([`Flow::page_foot`]). A unit
    /// taller than a page has to start somewhere, and moving it would leave
    /// a blank page behind.
    fn moves_on(&self, page_foot_px: f32) -> bool {
        !self.places && fits(self.extent, page_foot_px)
    }
}

/// The leading unit of `node` on pages `page_height_px` tall, with `room`
/// px left below its top: what the walk cannot part from the box's start.
/// css-break-3 §4.1 has no break point between a box's start and its
/// first child: breaks are between siblings, which is why §3.1 propagates
/// a first child's `break-before` to its container. Padding and border lie
/// outside the content edge, so they are none either; only a gap between
/// the content edge and the first child's margin edge (class C) is one.
///
/// - A box placed whole ([`Whole`], or not entered, [`enters`]) leads with
///   itself.
/// - An inline root of several lines leads with the lines its `orphans`
///   and `widows` cannot leave at a page foot ([`inline_leading`]).
/// - A block container leads with its top edge and its first in-flow
///   child's unit, or with its top edge alone before a gap.
/// - A flex or grid container, or a table row, places content when one of
///   its first line's boxes does, and its unit ends with the shortest.
/// - A table leads with its top edge, its repeated header, and its first
///   row; below its caption, in the box around both.
///
/// The box places content in the room when its unit fits it, or the whole
/// box does.
fn leading_extent(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Leading {
    let kind = classify(flow, node, page_height_px);
    leading_of(flow, node, &kind, room, page_height_px, depth)
}

/// [`leading_extent`] of `node`, of `kind`.
fn leading_of(
    flow: Flow<'_>,
    node: &blitz_dom::Node,
    kind: &BoxKind,
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Leading {
    let Flow { doc, .. } = flow;
    let height = extent(flow, node, depth);
    let whole = Leading::whole(height, room);
    if depth >= crate::MAX_DOM_DEPTH {
        return whole;
    }
    let within = |inner: Leading, offset: f32| Leading {
        extent: offset + inner.extent,
        places: height <= room || inner.places,
        relaxes: inner.relaxes,
    };
    match kind {
        // A kept box is measured through its structure only when a forced
        // break or a page-name change below it splits it. `enters` also
        // enters one for a skipped header row, which leaves it unsplit: its
        // unit is still the whole box, as the move of a kept box entered so
        // needs.
        BoxKind::Monolithic(Whole::Kept) if forced_inside(flow, node, page_height_px, 0) => {
            let kind = structure(flow, node);
            leading_of(flow, node, &kind, room, page_height_px, depth)
        }
        BoxKind::Monolithic(Whole::Kept) => whole.relaxed(),
        BoxKind::Monolithic(Whole::Atomic) if slices_at_a_page_top(node) => whole.relaxed(),
        BoxKind::Monolithic(Whole::Atomic) => whole,
        BoxKind::InlineRoot { lines, .. } if lines.len() <= 1 => whole,
        BoxKind::InlineRoot { lines, limits } => {
            let lead = inline_leading(lines, *limits, top_inset(node), room);
            Leading {
                places: lead.places || height <= room,
                relaxes: true,
                ..lead
            }
        }
        BoxKind::Parallel { tracks, top } => {
            let row = displays(node, DisplayInside::TableRow);
            if row
                && let Some(bottom) = tracks
                    .first()
                    .and_then(|&id| doc.get_node(id))
                    .and_then(|cell| kept_table_row_bottom(flow, cell, page_height_px))
            {
                return Leading::whole(bottom - top, room).relaxed();
            }
            let Some(lead) = tracks_leading(flow, tracks, *top, room, page_height_px, depth) else {
                return whole;
            };
            Leading {
                places: lead.places || (!row && height <= room),
                ..lead
            }
        }
        BoxKind::Table { table, .. } => {
            if !enters(flow, node, kind, room, page_height_px) {
                return whole;
            }
            let Some(table) = doc.get_node(*table) else {
                return whole;
            };
            let offset = if table.id == node.id {
                0.0
            } else {
                table.final_layout.location.y
            };
            let inner = table_leading(flow, table, room - offset, page_height_px, depth + 1);
            within(inner, offset)
        }
        _ => {
            if !enters(flow, node, kind, room, page_height_px) {
                return whole;
            }
            let Some(first) = in_flow_children(flow, node)
                .first()
                .and_then(|&id| doc.get_node(id))
            else {
                return whole;
            };
            let layout = first.final_layout;
            let content_top = node.final_layout.border.top + node.final_layout.padding.top;
            if layout.location.y - layout.margin.top - content_top > 0.5 {
                return Leading {
                    extent: content_top,
                    places: height <= room || content_top <= room,
                    relaxes: false,
                };
            }
            let inner = leading_extent(
                flow,
                first,
                room - layout.location.y,
                page_height_px,
                depth + 1,
            );
            within(inner, layout.location.y)
        }
    }
}

/// The leading unit of boxes side by side, `tracks`, whose line's top is
/// `top` in their parent's layout, with `room` px left below it: it places
/// content when one of the boxes does, and ends with the shortest of their
/// units, as no box moves alone ([`LineState`]). `None` for no box.
fn tracks_leading(
    flow: Flow<'_>,
    tracks: &[usize],
    top: f32,
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Option<Leading> {
    tracks
        .iter()
        .filter_map(|&id| flow.doc.get_node(id))
        .map(|track| {
            let offset = track.final_layout.location.y - top;
            let inner = leading_extent(flow, track, room - offset, page_height_px, depth + 1);
            Leading {
                extent: offset + inner.extent,
                ..inner
            }
        })
        .reduce(|a, b| Leading {
            extent: a.extent.min(b.extent),
            places: a.places || b.places,
            relaxes: a.relaxes && b.relaxes,
        })
}

/// The leading unit of an inline root of `lines`, with `room` px left
/// below its top: the first lines its `limits` cannot leave
/// at a page foot, the first `orphans` when `widows` lines remain after
/// them, else every line ([`split_point`], by which
/// [`fragment_inline_root`] splits it, so the two cannot disagree). The
/// lines are measured from the first, below the root's top border and
/// padding, `inset` ([`top_inset`]), which each of its fragments paints
/// above its lines.
fn inline_leading(lines: &[(f32, f32)], limits: LineLimits, inset: f32, room: f32) -> Leading {
    let Some(&(first_top, _)) = lines.first() else {
        return Leading::whole(0.0, room);
    };
    let unit_lines = (1..lines.len())
        .find(|&fit| split_point(fit, 0, lines.len(), limits).is_some())
        .unwrap_or(lines.len());
    let extent = inset + lines[unit_lines - 1].1 - first_top;
    Leading::whole(extent, room)
}

/// The top border and padding of `node`, above its content: the renderer
/// paints an inline root's lines that far below the top of each of its
/// fragments, while its line boxes are measured from its content's top
/// ([`collect_inline_line_metrics`]).
fn top_inset(node: &blitz_dom::Node) -> f32 {
    let layout = node.final_layout;
    layout.border.top + layout.padding.top
}

/// The leading unit of `table` ([`leading_extent`]): its top edge and its
/// first row, or, under a header it repeats, the header as its first page
/// carries it and the first body row below it, measured on the pages the
/// header leaves the rows ([`fragment_repeating_table`]).
fn table_leading(
    flow: Flow<'_>,
    table: &blitz_dom::Node,
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Leading {
    let Flow {
        doc, column_styles, ..
    } = flow;
    let row_of = |cell: &blitz_dom::Node| {
        cell.parent
            .and_then(|id| doc.get_node(id))
            .filter(|&row| displays(row, DisplayInside::TableRow))
    };
    let header = (depth < crate::MAX_DOM_DEPTH)
        .then(|| repeating_table_header(doc, table.id, depth + 1, column_styles))
        .flatten()
        .filter(|header| {
            header.band_height_px < page_height_px
                && header_band_is_repeatable(flow, header, page_height_px, depth + 1)
        });
    if let Some(header) = header {
        let band = header.band_height_px - header.first_page_lift_px;
        let Some(row) = header
            .body_cell_ids
            .first()
            .and_then(|&id| doc.get_node(id))
            .and_then(row_of)
        else {
            return Leading::whole(band, room);
        };
        let rows_page = page_height_px - header.band_height_px;
        let inner = leading_extent(flow, row, room - band, rows_page, depth + 1);
        let inner = orphan_rows(
            flow,
            table,
            &header.body_cell_ids,
            room - band,
            rows_page,
            depth,
        )
        .unwrap_or(inner);
        return Leading {
            extent: band + inner.extent,
            ..inner
        };
    }
    let Some(row) = in_flow_children(flow, table)
        .first()
        .and_then(|&id| doc.get_node(id))
        .and_then(row_of)
    else {
        return Leading::whole(table.final_layout.size.height, room);
    };
    let BoxKind::Parallel { top, .. } = classify(flow, row, page_height_px) else {
        return Leading::whole(table.final_layout.size.height, room);
    };
    let inner = leading_extent(flow, row, room - top, page_height_px, depth + 1);
    let inner = orphan_rows(
        flow,
        table,
        &walk_children_of(table),
        room - top,
        page_height_px,
        depth,
    )
    .unwrap_or(inner);
    Leading {
        extent: top + inner.extent,
        ..inner
    }
}

/// The leading unit of the rows of `table` among `cells`, its body's, from
/// the first row's top with `room` px left below it, when the table's
/// row `orphans` keep more than one row start together ([`table_line_limits`]):
/// its first row `orphans` rows, the last of them by its own leading unit, or
/// all its rows when it has no more. The unit is held together by
/// row `orphans` alone, so it is relaxed at a page top, and it is no taller
/// than a page: a table whose rows that fit a page are fewer than its
/// row `orphans` moves on whole once, and breaks where it stands there.
fn orphan_rows(
    flow: Flow<'_>,
    table: &blitz_dom::Node,
    cells: &[usize],
    room: f32,
    page_height_px: f32,
    depth: usize,
) -> Option<Leading> {
    let orphans = table_line_limits(flow.column_styles, table)?.orphans;
    if orphans < 2 || depth >= crate::MAX_DOM_DEPTH {
        return None;
    }
    let rows: Vec<Vec<&blitz_dom::Node>> = flow_lines(flow, table, cells)
        .iter()
        .map(|line| {
            line.iter()
                .filter_map(|&index| flow.doc.get_node(cells[index]))
                .collect()
        })
        .collect();
    let top_of = |row: &[&blitz_dom::Node]| {
        row.iter()
            .map(|cell| cell.final_layout.location.y)
            .fold(f32::INFINITY, f32::min)
    };
    let first_top = top_of(rows.first()?);
    let extent = match rows.get(orphans - 1) {
        Some(last) => {
            let top = top_of(last) - first_top;
            let lead = line_leading(
                flow,
                last,
                top_of(last),
                room - top,
                page_height_px,
                depth + 1,
            )?;
            top + lead.extent
        }
        None => line_bottom(rows.last()?) - first_top,
    };
    let extent = extent.min(page_height_px);
    Some(Leading {
        extent,
        places: extent <= room,
        relaxes: true,
    })
}

/// Whether `node` is a childless box the walk slices from where it stands
/// at the top of a page ([`Leading::relaxes`]): not replaced, as an image
/// is, nor transformed, which paints as one shape.
fn slices_at_a_page_top(node: &blitz_dom::Node) -> bool {
    node.children.is_empty()
        && !is_transformed(node)
        && node.element_data().is_some()
        && !is_replaced(node)
}

/// Whether a cell of an earlier row of `row`'s group spans down to it and
/// keeps it with the row above ([`RowGroup::spanned`]).
fn spanned_from_above(flow: Flow<'_>, row: &blitz_dom::Node) -> bool {
    flow.pass
        .row_group(flow, row)
        .is_some_and(|(group, index)| group.spanned[index])
}

/// Whether `node`'s display has the inner type `inside`.
fn displays(node: &blitz_dom::Node, inside: DisplayInside) -> bool {
    node.primary_styles()
        .is_some_and(|s| s.clone_display().inside() == inside)
}

/// The table cells of `row`.
fn row_cells<'a>(
    doc: &'a BaseDocument,
    row: &'a blitz_dom::Node,
) -> impl Iterator<Item = &'a blitz_dom::Node> {
    row.children
        .iter()
        .filter_map(|&id| doc.get_node(id))
        .filter(|&node| displays(node, DisplayInside::TableCell))
}

#[derive(Debug)]
struct RepeatingTableHeader {
    table_id: usize,
    header_cell_ids: Vec<usize>,
    body_cell_ids: Vec<usize>,
    body_origin_px: f32,
    band_height_px: f32,
    /// The header cells placed on the table's first page: every one but
    /// those of the rows at the top of the group that skip it
    /// (`-fulgur-header-repeat: skip-first`), which print only where the
    /// header repeats.
    first_page_cell_ids: Vec<usize>,
    /// How much shorter the band is on the table's first page: from the top
    /// of the skipped rows to the top of the first row that prints there (or
    /// of the body, when every header row skips). Zero when no row skips.
    first_page_lift_px: f32,
}

fn collect_repeating_table_cells(
    doc: &BaseDocument,
    node_id: usize,
    in_header: bool,
    depth: usize,
    header_claimed: &mut bool,
    header_cell_ids: &mut Vec<usize>,
    body_cell_ids: &mut Vec<usize>,
) {
    if depth >= crate::MAX_DOM_DEPTH {
        return;
    }
    let Some(node) = doc.get_node(node_id) else {
        return;
    };
    let Some(display) = node.primary_styles().map(|s| s.clone_display().inside()) else {
        return;
    };
    let next_in_header = match display {
        // css-tables-3: "If a table owns multiple `display: table-header-group`
        // boxes, only the first is treated as a header; the others are treated
        // as if they had `display: table-row-group`."
        //
        // Promoting every header group made the band span from the table's top
        // down to the *last* one, so a header group placed after a `tbody`
        // produced a band taller than the rows above it and blew the page box
        // (fulgur-naj7.5). Children are visited in document order, so the first
        // group to reach this arm is the one the spec designates.
        DisplayInside::TableHeaderGroup => {
            if *header_claimed {
                false
            } else {
                *header_claimed = true;
                true
            }
        }
        DisplayInside::TableRowGroup
        | DisplayInside::TableFooterGroup
        | DisplayInside::TableRow
        | DisplayInside::Contents => in_header,
        DisplayInside::TableCell => {
            if in_header {
                header_cell_ids.push(node_id);
            } else {
                body_cell_ids.push(node_id);
            }
            return;
        }
        DisplayInside::Table => return,
        _ => return,
    };
    for &child_id in &node.children {
        collect_repeating_table_cells(
            doc,
            child_id,
            next_in_header,
            depth + 1,
            header_claimed,
            header_cell_ids,
            body_cell_ids,
        );
    }
}

fn repeating_table_header(
    doc: &BaseDocument,
    table_id: usize,
    child_depth: usize,
    column_styles: Option<&crate::column_css::ColumnStyleTable>,
) -> Option<RepeatingTableHeader> {
    if child_depth >= crate::MAX_DOM_DEPTH {
        return None;
    }
    let table = doc.get_node(table_id)?;
    let display = table.primary_styles().map(|s| s.clone_display().inside())?;
    if display != DisplayInside::Table {
        return None;
    }

    let mut header_cell_ids = Vec::new();
    let mut body_cell_ids = Vec::new();
    // Threaded across siblings so only the first `table-header-group` in
    // document order becomes the header — see `collect_repeating_table_cells`.
    let mut header_claimed = false;
    for &child_id in &table.children {
        collect_repeating_table_cells(
            doc,
            child_id,
            false,
            child_depth,
            &mut header_claimed,
            &mut header_cell_ids,
            &mut body_cell_ids,
        );
    }
    if header_cell_ids.is_empty() || body_cell_ids.is_empty() {
        return None;
    }

    let header_top_px = header_cell_ids
        .iter()
        .filter_map(|&id| doc.get_node(id).map(|n| n.final_layout.location.y))
        .reduce(f32::min)?;
    let header_bottom_px = header_cell_ids
        .iter()
        .filter_map(|&id| {
            doc.get_node(id)
                .map(|n| n.final_layout.location.y + n.final_layout.size.height)
        })
        .reduce(f32::max)?;
    let body_origin_px = body_cell_ids
        .iter()
        .filter_map(|&id| doc.get_node(id).map(|n| n.final_layout.location.y))
        .reduce(f32::min)?;
    if !header_top_px.is_finite() || !header_bottom_px.is_finite() || !body_origin_px.is_finite() {
        return None;
    }

    // `band_height_px` below measures the strip from the table's top edge down
    // to where the body starts, which is only the header's own box when the
    // header actually sits at the top of the table.
    //
    // CSS 2.1 §17.5.1 says a `table-header-group` is "always displayed before
    // all other rows and row groups" regardless of its source position, but
    // that reordering happens in layout and the upstream engine does not
    // implement it — a header group written after a `tbody` keeps its
    // in-flow offset here. Reserving a band that spans the rows above it
    // then produces a fragment taller than the page box (fulgur-naj7.5:
    // measured [(0, h=120), (1, h=100), (2, h=100)] on a 100px page).
    //
    // Since fulgur consumes layout rather than producing it, it cannot honour
    // the reordering on its own. Decline to repeat instead of emitting broken
    // geometry; the table still paginates, just without a repeated header.
    if header_top_px > body_origin_px {
        return None;
    }

    let band_height_px = header_bottom_px.max(body_origin_px);
    if band_height_px <= 0.0 {
        return None;
    }

    // A row that skips the first page (`-fulgur-header-repeat: skip-first`)
    // is left out there only above every row that prints: lifting the rest
    // of the table by its height leaves no gap. One lower down prints.
    let skips_first = |cell_id: usize| {
        doc.get_node(cell_id)
            .and_then(|cell| cell.parent)
            .and_then(|row_id| column_styles?.get(&row_id))
            .is_some_and(|props| {
                props.header_repeat == Some(crate::column_css::HeaderRepeat::SkipFirst)
            })
    };
    let cell_top = |cell_id: usize| doc.get_node(cell_id).map(|n| n.final_layout.location.y);
    let first_shown_top_px = header_cell_ids
        .iter()
        .filter(|&&id| !skips_first(id))
        .filter_map(|&id| cell_top(id))
        .fold(body_origin_px, f32::min);
    let first_page_cell_ids: Vec<usize> = header_cell_ids
        .iter()
        .copied()
        .filter(|&id| {
            !(skips_first(id) && cell_top(id).is_some_and(|top| top < first_shown_top_px - 0.5))
        })
        .collect();
    let first_page_lift_px = if first_page_cell_ids.len() < header_cell_ids.len() {
        (first_shown_top_px - header_top_px).max(0.0)
    } else {
        0.0
    };

    Some(RepeatingTableHeader {
        table_id,
        header_cell_ids,
        body_cell_ids,
        body_origin_px,
        band_height_px,
        first_page_cell_ids,
        first_page_lift_px,
    })
}

/// What every walk of the fragmenter reads: the laid-out document, the
/// side tables of its break properties and used page names, and the
/// `position: running()` elements, which the walk takes out of the flow.
#[derive(Clone, Copy)]
struct Flow<'a> {
    doc: &'a BaseDocument,
    column_styles: Option<&'a crate::column_css::ColumnStyleTable>,
    used_page_names: Option<&'a crate::blitz_adapter::UsedPageNameTable>,
    running_store: Option<&'a crate::gcpm::running::RunningElementStore>,
    pass: &'a Pass,
    /// The first cell of a table row before which the walk takes a forced
    /// break, for its table's row `widows` ([`fragment_table_rows`]).
    forced_before: Option<usize>,
    /// How far above the page foot a fragment ends, at least, where a break
    /// point precedes it, in px (`-fulgur-page-foot-clearance` on the root
    /// element, [`page_foot_clearance`]). The walk fills a page strip this
    /// much shorter than the page, so that every fit it measures leaves it
    /// free; only a box taller than the page itself is sliced at the page
    /// foot ([`Flow::page_foot`]).
    foot_clearance: f32,
}

impl Flow<'_> {
    /// The foot of the page whose strip, above the page-foot clearance, is
    /// `page_height_px` tall: where the walk slices a box taller than the
    /// page.
    fn page_foot(&self, page_height_px: f32) -> f32 {
        page_height_px + self.foot_clearance
    }

    /// Whether `extent` px fit a whole page whose strip is `page_height_px`
    /// tall ([`fits`] to its foot): what a box needs to move to the next page
    /// whole, where it stands at the top and may end in the page-foot
    /// clearance, rather than split as a box taller than a page.
    fn fits_page(&self, extent: f32, page_height_px: f32) -> bool {
        fits(extent, self.page_foot(page_height_px))
    }
}

/// What a pass computes once and its walks read: the breaks of its boxes
/// ([`box_breaks`]) and of the walks over their children ([`walk_breaks`]),
/// and the rows of its tables' row groups.
#[derive(Default)]
struct Pass {
    /// The breaks of each box asked about, by node and page height.
    breaks: std::cell::RefCell<BTreeMap<(usize, u32), std::rc::Rc<BoxBreaks>>>,
    /// The walk over each box's children asked about, by node and page
    /// height.
    walks: std::cell::RefCell<BTreeMap<(usize, u32), std::rc::Rc<WalkBreaks>>>,
    /// The rows of each row group asked about, by the group's node.
    rows: std::cell::RefCell<BTreeMap<usize, std::rc::Rc<RowGroup>>>,
}

impl Pass {
    /// The row group the table row `row` belongs to, and its index in it,
    /// read once a group ([`group_rows`]).
    fn row_group(
        &self,
        flow: Flow<'_>,
        row: &blitz_dom::Node,
    ) -> Option<(std::rc::Rc<RowGroup>, usize)> {
        let group = memo(&self.rows, row.parent?, || {
            group_rows(flow.doc, row).map(|(rows, _)| RowGroup::new(flow, &rows))
        })?;
        let index = *group.index.get(&row.id)?;
        Some((group, index))
    }
}

/// The rows of one row group, as its table's walk and break queries ask
/// about them row by row ([`Pass::row_group`]).
struct RowGroup {
    rows: Vec<usize>,
    /// Each row's index in `rows`, by its node.
    index: BTreeMap<usize, usize>,
    /// The last row the cells from each row reach by `rowspan`
    /// ([`span_end`]) in a run of rows kept together ([`Self::spanned`]);
    /// the row itself elsewhere.
    reach: Vec<usize>,
    /// Whether a cell of an earlier row spans down to each row and the rows
    /// it spans are kept together: under `-fulgur-span-break-inside:
    /// avoid-unsplittable` on the table, when no row of the run the spans
    /// chain ([`span_end`]) holds a cell of two lines or more
    /// ([`row_splits`]). Then no opportunity lies between it and the row
    /// above, so it does not move alone ([`line_moves_on`]). Elsewhere the
    /// rows a cell spans break as other rows do: css-tables-3 §6.1 calls a
    /// row whose spanning cells span a later row freely fragmentable.
    spanned: Vec<bool>,
    /// Whether a cell of an earlier row spans down to each row with its
    /// content ([`span_content_bottom`]): the walk does not cut a spanning
    /// cell's content at a forced break, so no forced break is taken before
    /// the row ([`opportunities`]).
    crossed: Vec<bool>,
}

/// The bottom of the content of the table cell `cell`, in its table's
/// coordinates as its layout box is: its last in-flow child's bottom, or
/// its last line's, else its content box top.
fn span_content_bottom(doc: &BaseDocument, cell: &blitz_dom::Node) -> f32 {
    let layout = cell.final_layout;
    let top = layout.border.top + layout.padding.top;
    let lines = collect_inline_line_metrics(cell)
        .last()
        .map(|&(_, bottom)| top + bottom);
    let children = walk_children_of(cell)
        .into_iter()
        .filter_map(|id| doc.get_node(id))
        .filter(|child| !leaves_the_flow(child))
        .map(|child| child.final_layout.location.y + child.final_layout.size.height)
        .reduce(f32::max);
    layout.location.y + lines.or(children).unwrap_or(top)
}

/// The lines of `node`'s content, counted as far as two: an inline root's
/// lines, a block container's children's, and one for a childless box with
/// a height, which places as one piece.
fn content_lines(doc: &BaseDocument, node: &blitz_dom::Node, depth: usize) -> usize {
    let lines = collect_inline_line_metrics(node).len();
    if lines > 0 || depth >= crate::MAX_DOM_DEPTH {
        return lines;
    }
    let mut count = 0;
    for child in walk_children_of(node)
        .into_iter()
        .filter_map(|id| doc.get_node(id))
        .filter(|child| !out_of_flow(child))
    {
        count += if child.children.is_empty() {
            usize::from(child.final_layout.size.height > 0.0)
        } else {
            content_lines(doc, child, depth + 1)
        };
        if count >= 2 {
            break;
        }
    }
    count
}

/// Whether the table row `row` may split: one of its cells that spans no
/// rows counts two lines or more ([`content_lines`]).
fn row_splits(doc: &BaseDocument, row: &blitz_dom::Node) -> bool {
    row_cells(doc, row).any(|cell| rowspan_of(cell) == 1 && content_lines(doc, cell, 0) >= 2)
}

/// The `-fulgur-span-break-inside` of the table the row group `rows`
/// belongs to.
fn span_break_inside(
    flow: Flow<'_>,
    rows: &[&blitz_dom::Node],
) -> crate::column_css::SpanBreakInside {
    let group = rows.first().and_then(|row| flow.doc.get_node(row.parent?));
    let table = group.and_then(|group| {
        if displays(group, DisplayInside::Table) {
            Some(group)
        } else {
            flow.doc.get_node(group.parent?)
        }
    });
    table
        .and_then(|table| flow.column_styles?.get(&table.id)?.span_break_inside)
        .unwrap_or_default()
}

impl RowGroup {
    fn new(flow: Flow<'_>, rows: &[&blitz_dom::Node]) -> Self {
        let doc = flow.doc;
        let span_ends: Vec<usize> = (0..rows.len())
            .map(|from| span_end(doc, rows, from))
            .collect();
        let keeps =
            span_break_inside(flow, rows) == crate::column_css::SpanBreakInside::AvoidUnsplittable;
        let mut spanned = vec![false; rows.len()];
        let mut reach: Vec<usize> = (0..rows.len()).collect();
        let mut start = 0;
        while start < rows.len() {
            let end = span_ends[start];
            if keeps && end > start && !rows[start..=end].iter().any(|row| row_splits(doc, row)) {
                for flag in &mut spanned[start + 1..=end] {
                    *flag = true;
                }
                reach[start..=end].copy_from_slice(&span_ends[start..=end]);
            }
            start = end + 1;
        }
        let top = |row: &blitz_dom::Node| {
            row_cells(doc, row)
                .map(|cell| cell.final_layout.location.y)
                .reduce(f32::min)
        };
        let crossed = (0..rows.len())
            .map(|at| {
                (0..at).filter(|&from| span_ends[from] >= at).any(|from| {
                    row_cells(doc, rows[from]).any(|cell| {
                        from + rowspan_of(cell) > at
                            && top(rows[at])
                                .is_none_or(|top| span_content_bottom(doc, cell) > top + 0.5)
                    })
                })
            })
            .collect();
        Self {
            rows: rows.iter().map(|row| row.id).collect(),
            index: rows
                .iter()
                .enumerate()
                .map(|(index, row)| (row.id, index))
                .collect(),
            reach,
            spanned,
            crossed,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fragment_block_subtree(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    parent_id: usize,
    parent_w: f32,
    parent_x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
) -> (u32, f32) {
    let widows = flow
        .doc
        .get_node(parent_id)
        .filter(|&node| displays(node, DisplayInside::Table))
        .and_then(|table| table_line_limits(flow.column_styles, table))
        .map_or(1, |limits| limits.widows);
    if widows < 2 {
        return fragment_block_box(
            geometry,
            flow,
            parent_id,
            parent_w,
            parent_x_in_body,
            page_in,
            cursor_in,
            page_height_px,
            depth,
            entered_below_content,
            page_has_content,
            remaining_repeat_budget,
            lifted,
        );
    }
    fragment_table_rows(
        geometry,
        flow,
        parent_id,
        parent_w,
        parent_x_in_body,
        page_in,
        cursor_in,
        page_height_px,
        depth,
        entered_below_content,
        page_has_content,
        remaining_repeat_budget,
        lifted,
        widows,
    )
}

/// Fragments the table `table_id` whose row `widows`
/// (`-fulgur-table-widows`) keep that many rows on its
/// last page ([`table_line_limits`]): it is walked once, and when its last
/// page holds fewer rows (a row split between pages counting as one), a
/// second time with a forced break before the row that leaves `widows` on
/// it ([`widows_row`]), when those rows fit a page and an earlier row
/// stays before them. Each walk writes to a table of its own, merged once
/// the table is placed.
#[allow(clippy::too_many_arguments)]
fn fragment_table_rows(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    table_id: usize,
    width: f32,
    x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
    widows: usize,
) -> (u32, f32) {
    let before = (*page_has_content, *remaining_repeat_budget, *lifted);
    let mut walk = |geometry: &mut PaginationGeometryTable, flow: Flow<'_>| {
        (*page_has_content, *remaining_repeat_budget, *lifted) = before;
        fragment_block_box(
            geometry,
            flow,
            table_id,
            width,
            x_in_body,
            page_in,
            cursor_in,
            page_height_px,
            depth,
            entered_below_content,
            page_has_content,
            remaining_repeat_budget,
            lifted,
        )
    };
    let mut rows = PaginationGeometryTable::new();
    let mut end = walk(&mut rows, flow);
    if let Some(cell) = widows_row(&rows, flow, table_id, widows, page_height_px, depth) {
        rows.clear();
        end = walk(
            &mut rows,
            Flow {
                forced_before: Some(cell),
                ..flow
            },
        );
    }
    for (node_id, source) in rows {
        let target = geometry.entry(node_id).or_default();
        if target.fragments.is_empty() {
            target.line_boundaries = source.line_boundaries;
        } else {
            target.line_boundaries.clear();
        }
        target.fragments.extend(source.fragments);
        target.is_repeat |= source.is_repeat;
        target.lift_px += source.lift_px;
    }
    end
}

/// The first cell of the row of `table_id` before which a forced break
/// leaves `widows` rows on the table's last page, as `rows` place them
/// ([`fragment_table_rows`]); `None` when that page holds as many already,
/// when the table ends on its first page, when no earlier row would stay
/// before the break, or when those rows do not fit a page.
fn widows_row(
    rows: &PaginationGeometryTable,
    flow: Flow<'_>,
    table_id: usize,
    widows: usize,
    page_height_px: f32,
    depth: usize,
) -> Option<usize> {
    let doc = flow.doc;
    let table = doc.get_node(table_id)?;
    let header = (depth < crate::MAX_DOM_DEPTH)
        .then(|| repeating_table_header(doc, table_id, depth + 1, flow.column_styles))
        .flatten();
    let (cells, band) = match &header {
        Some(header) => (header.body_cell_ids.clone(), header.band_height_px),
        None => (walk_children_of(table), 0.0),
    };
    let lines = flow_lines(flow, table, &cells);
    let pages = |line: &[usize]| -> Vec<u32> {
        line.iter()
            .filter_map(|&index| rows.get(&cells[index]))
            .flat_map(|geometry| geometry.fragments.iter().map(|f| f.page_index))
            .collect()
    };
    let first_page = lines
        .first()
        .and_then(|line| pages(line).into_iter().min())?;
    let end_page = lines
        .iter()
        .filter_map(|line| pages(line).into_iter().max())
        .max()?;
    let on_last = lines
        .iter()
        .filter(|line| pages(line).contains(&end_page))
        .count();
    if end_page == first_page || on_last >= widows {
        return None;
    }
    let from = lines.len().checked_sub(widows).filter(|&from| from > 0)?;
    let boxes = |line: &[usize]| -> Vec<&blitz_dom::Node> {
        line.iter()
            .filter_map(|&index| doc.get_node(cells[index]))
            .collect()
    };
    let top = boxes(&lines[from])
        .iter()
        .map(|cell| cell.final_layout.location.y)
        .fold(f32::INFINITY, f32::min);
    let bottom = line_bottom(&boxes(lines.last()?));
    flow.fits_page(bottom - top, page_height_px - band)
        .then(|| cells[lines[from][0]])
}

/// [`fragment_block_subtree`] of a box whose rows need no second walk.
#[allow(clippy::too_many_arguments)]
fn fragment_block_box(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    parent_id: usize,
    parent_w: f32,
    parent_x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
) -> (u32, f32) {
    let Flow {
        doc, column_styles, ..
    } = flow;
    if depth < crate::MAX_DOM_DEPTH
        && let Some(header) = repeating_table_header(doc, parent_id, depth + 1, column_styles)
        && header_band_is_repeatable(flow, &header, page_height_px, depth + 1)
    {
        return fragment_repeating_table(
            geometry,
            flow,
            &header,
            parent_w,
            parent_x_in_body,
            page_in,
            cursor_in,
            page_height_px,
            depth,
            entered_below_content,
            page_has_content,
            remaining_repeat_budget,
            lifted,
        );
    }
    if let Some(content_top) = table_cell_content_top(doc, parent_id) {
        return fragment_table_cell(
            geometry,
            flow,
            parent_id,
            parent_w,
            parent_x_in_body,
            page_in,
            cursor_in,
            page_height_px,
            depth,
            entered_below_content,
            page_has_content,
            remaining_repeat_budget,
            lifted,
            content_top,
        );
    }
    fragment_block_subtree_inner(
        geometry,
        flow,
        parent_id,
        parent_w,
        parent_x_in_body,
        page_in,
        cursor_in,
        page_height_px,
        depth,
        None,
        entered_below_content,
        page_has_content,
        remaining_repeat_budget,
        lifted,
        false,
    )
}

/// The offset of a table cell's content box from its border box top (its
/// top border and padding), or `None` for a node that is not a table cell
/// or has neither.
fn table_cell_content_top(doc: &BaseDocument, node_id: usize) -> Option<f32> {
    let node = doc.get_node(node_id)?;
    let is_cell = node
        .primary_styles()
        .is_some_and(|s| s.clone_display().inside() == DisplayInside::TableCell);
    let layout = node.final_layout;
    let top = layout.border.top + layout.padding.top;
    (is_cell && top > 0.0).then_some(top)
}

/// Fragment a table cell whose content starts `content_top` below its top
/// edge, keeping that top border and padding on each page the cell
/// continues on, as `box-decoration-break: clone` would and as print
/// tables commonly do: the borders of every fragment of a split cell are
/// painted already, and without the repeat its continued content sat
/// against its top edge, or against a repeated header.
///
/// The cell is fragmented on a strip `content_top` shorter, whose top is
/// `content_top` below each page's top, so its content restarts that far
/// down on every continued page and the edge takes room from the page.
/// Its first page is unchanged: the strip ends at the page foot, and the
/// content starts where it did. On every later page the cell's own
/// fragment starts at the page top and holds the edge.
#[allow(clippy::too_many_arguments)]
fn fragment_table_cell(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    cell_id: usize,
    cell_w: f32,
    cell_x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
    content_top: f32,
) -> (u32, f32) {
    let mut cell_geometry = PaginationGeometryTable::new();
    let (page, cursor) = fragment_block_subtree_inner(
        &mut cell_geometry,
        flow,
        cell_id,
        cell_w,
        cell_x_in_body,
        page_in,
        cursor_in - content_top,
        page_height_px - content_top,
        depth,
        None,
        entered_below_content,
        page_has_content,
        remaining_repeat_budget,
        lifted,
        false,
    );
    // The walk ends at the content's bottom. A cell that ends on its first
    // page is placed by its layout box, but one that continued ends there
    // too: its last fragment, and the row, end below its bottom padding and
    // border, sliced at the page foot.
    let end = if page > page_in {
        let layout = flow.doc.get_node(cell_id).map(|node| node.final_layout);
        let bottom_edge = layout.map_or(0.0, |l| l.padding.bottom + l.border.bottom);
        (cursor + content_top + bottom_edge).min(page_height_px.max(cursor + content_top))
    } else {
        cursor + content_top
    };
    for (node_id, mut source) in cell_geometry {
        for fragment in &mut source.fragments {
            if node_id == cell_id && fragment.page_index > page_in {
                fragment.height = (fragment.height.to_f32() + content_top).as_px();
            } else {
                fragment.y = (fragment.y.to_f32() + content_top).as_px();
            }
        }
        if node_id == cell_id
            && page > page_in
            && let Some(last) = source
                .fragments
                .iter_mut()
                .rev()
                .find(|fragment| fragment.page_index == page)
        {
            last.height = (end - last.y.to_f32()).max(last.height.to_f32()).as_px();
        }
        let target = geometry.entry(node_id).or_default();
        if target.fragments.is_empty() {
            target.line_boundaries = source.line_boundaries;
        } else {
            target.line_boundaries.clear();
        }
        target.fragments.extend(source.fragments);
        target.is_repeat |= source.is_repeat;
        target.lift_px += source.lift_px;
    }
    (page, end)
}

fn fragmented_descendant_page_extent(
    geometry: &PaginationGeometryTable,
    doc: &BaseDocument,
    node_id: usize,
    page_index: u32,
    page_start_y: f32,
    depth: usize,
) -> f32 {
    if depth >= crate::MAX_DOM_DEPTH {
        return 0.0;
    }
    let Some(node) = doc.get_node(node_id) else {
        return 0.0;
    };
    let layout_children = node.layout_children.borrow();
    let children = layout_children
        .as_deref()
        .filter(|children| !children.is_empty())
        .unwrap_or(&node.children);
    children.iter().fold(0.0_f32, |extent, &child_id| {
        let child_extent = geometry
            .get(&child_id)
            .into_iter()
            .flat_map(|entry| &entry.fragments)
            .filter(|fragment| fragment.page_index == page_index)
            .map(|fragment| fragment.y.to_f32() + fragment.height.to_f32() - page_start_y)
            .fold(0.0_f32, f32::max);
        extent
            .max(child_extent)
            .max(fragmented_descendant_page_extent(
                geometry,
                doc,
                child_id,
                page_index,
                page_start_y,
                depth + 1,
            ))
    })
}

fn record_header_template(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    header_cell_ids: &[usize],
    page_index: u32,
    table_x: f32,
    table_y: f32,
    depth: usize,
) {
    for &cell_id in header_cell_ids {
        let Some(cell) = doc.get_node(cell_id) else {
            continue;
        };
        let layout = cell.final_layout;
        let cell_x = table_x + layout.location.x;
        let cell_y = table_y + layout.location.y;
        geometry
            .entry(cell_id)
            .or_default()
            .fragments
            .push(Fragment {
                page_index,
                x: cell_x.as_px(),
                y: cell_y.as_px(),
                width: layout.size.width.as_px(),
                height: layout.size.height.as_px(),
            });
        record_subtree_descendants(
            geometry,
            doc,
            cell_id,
            page_index,
            cell_y,
            cell_x,
            depth + 1,
        );
    }
}

/// Places the header on the table's pages: `first_page` (recorded on the
/// first of `table_pages`) as it is, and `template` (recorded on the same
/// page at `first_table_top`) copied onto each later page. The two differ
/// when header rows skip the first page.
fn append_repeated_header_fragments(
    geometry: &mut PaginationGeometryTable,
    first_page: &PaginationGeometryTable,
    template: &PaginationGeometryTable,
    table_pages: &[(u32, f32)],
    first_table_top: f32,
    remaining_repeat_budget: &mut usize,
) {
    let repeats = table_pages.len() > 1;
    for (&node_id, source) in first_page {
        let target = geometry.entry(node_id).or_default();
        target.is_repeat |= source.is_repeat || repeats;
        target.fragments.extend(source.fragments.iter().cloned());
    }
    if !repeats {
        return;
    }
    for (&node_id, source) in template {
        geometry.entry(node_id).or_default().is_repeat |= source.is_repeat || repeats;
    }

    let fragments_per_header = template.values().map(|g| g.fragments.len()).sum::<usize>();
    for &(page_index, table_top) in &table_pages[1..] {
        if *remaining_repeat_budget < fragments_per_header {
            log::warn!(
                "repeating table header fragment budget exhausted; skipping remaining pages"
            );
            return;
        }
        for (&node_id, source) in template {
            let target = geometry.entry(node_id).or_default();
            target
                .fragments
                .extend(source.fragments.iter().map(|fragment| {
                    let mut repeated = fragment.clone();
                    repeated.page_index = page_index;
                    repeated.y = (table_top + fragment.y.to_f32() - first_table_top).as_px();
                    repeated
                }));
        }
        *remaining_repeat_budget -= fragments_per_header;
    }
}

#[allow(clippy::too_many_arguments)]
fn fragment_repeating_table(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    header: &RepeatingTableHeader,
    parent_w: f32,
    parent_x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
) -> (u32, f32) {
    let doc = flow.doc;
    if header.band_height_px >= page_height_px {
        log::warn!(
            "repeating table header band ({:.2}px) is not smaller than the page ({page_height_px:.2}px); using whole-table fallback",
            header.band_height_px
        );
        return fragment_block_subtree_inner(
            geometry,
            flow,
            header.table_id,
            parent_w,
            parent_x_in_body,
            page_in,
            cursor_in,
            page_height_px,
            depth,
            None,
            entered_below_content,
            page_has_content,
            remaining_repeat_budget,
            lifted,
            false,
        );
    }

    // Starting the table here requires room for the band AND at least one body
    // row. Testing the band alone produced a page holding nothing but a
    // repeated header, with every body row pushed to the next page
    // (fulgur-naj7.6: measured table = [(0, y=60, h=20), (1, y=0, h=80)] with
    // both rows on page 1). An orphaned header row is never useful output, and
    // css-tables-3 likewise bounds repetition to cases where it does not eat
    // the page.
    // Whether the first body row places content below the band in `room`
    // px from the table's top, as the walk below and its row move measure
    // it ([`table_leading`]).
    let body_page_height = page_height_px - header.band_height_px;
    let table = doc.get_node(header.table_id);
    let first_row_places = |room: f32| {
        table.is_none_or(|table| table_leading(flow, table, room, page_height_px, depth).places)
    };
    // The band the table's first page carries, without the rows that skip it.
    let lift = header.first_page_lift_px;
    let first_band = header.band_height_px - lift;
    let band_overflows = cursor_in + first_band > page_height_px;
    // The band fits but its first body row does not: starting here would emit
    // a header with nothing under it. Only worth moving if the pair actually
    // fits on an empty page — an oversized first row has to start somewhere,
    // and pushing it would just leave a blank page behind.
    let orphaned_header = !band_overflows
        && !first_row_places(page_height_px - cursor_in)
        && first_row_places(page_height_px);
    // The first body row's `break-before`, which its walk below leaves to
    // the table, as the first line's of any walk: forced after content on
    // the page, the table starts on the next page, its header with it,
    // rather than leave the header alone above the break.
    let first_row_break = table.map_or(BreakValue::Auto, |table| {
        flow_lines(flow, table, &header.body_cell_ids)
            .first()
            .map_or(BreakValue::Auto, |line| {
                line_break(
                    flow,
                    &header.body_cell_ids,
                    line,
                    false,
                    page_height_px,
                    depth + 1,
                )
            })
    });
    let (first_page, first_table_top) = if first_row_break.is_forced() && *page_has_content {
        (page_on_side(doc, page_in + 1, first_row_break.side()), 0.0)
    } else if band_overflows || orphaned_header {
        (page_in + 1, 0.0)
    } else {
        (page_in, cursor_in)
    };
    // Where the table would start were the skipped rows on its first page:
    // the body and the header rows below them are placed from here, so they
    // lift by the rows' height (above the page top when the table starts
    // there).
    let full_band_top = first_table_top - lift;
    let mut header_template = PaginationGeometryTable::new();
    record_header_template(
        &mut header_template,
        doc,
        &header.header_cell_ids,
        first_page,
        parent_x_in_body,
        first_table_top,
        depth,
    );
    let mut first_page_template = PaginationGeometryTable::new();
    if lift > 0.0 {
        record_header_template(
            &mut first_page_template,
            doc,
            &header.first_page_cell_ids,
            first_page,
            parent_x_in_body,
            full_band_top,
            depth,
        );
    }

    let mut body_geometry = PaginationGeometryTable::new();
    // A table moved on starts on an empty page; the band above the rows is
    // no content for them: a first row's paragraph under it on an otherwise
    // empty page splits there rather than leave the band alone.
    *page_has_content &= first_page == page_in;
    let (body_end_page, body_end_cursor) = fragment_block_subtree_inner(
        &mut body_geometry,
        flow,
        header.table_id,
        parent_w,
        parent_x_in_body,
        first_page,
        full_band_top,
        body_page_height,
        depth,
        Some(header),
        first_table_top > 0.0,
        page_has_content,
        remaining_repeat_budget,
        // The rows place their cells side by side and apply no lift.
        &mut 0.0,
        false,
    );

    // Pages that actually carry body content. A forced break after the last
    // body block advances the fragmenter and closes the table with a
    // zero-height fragment on the following page; adding the band to that would
    // promote it into a real page holding nothing but a cloned header, and
    // shift everything after the table down by the phantom band.
    let body_content_pages: std::collections::BTreeSet<u32> = body_geometry
        .iter()
        .filter(|(node_id, _)| **node_id != header.table_id)
        .flat_map(|(_, geom)| {
            // Distinguish a box placed on this page from the empty
            // continuation the fragmenter opens to close a node that started
            // earlier. A zero-height fragment counts as content only when the
            // node has nothing on an earlier page — then it was placed here
            // and can still paint (border, background, box-shadow). Node
            // identity is the wrong axis for this: cells and their wrappers
            // both produce continuations, and both can also be genuinely
            // placed zero-height boxes.
            let first_page = geom
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .min();
            geom.fragments
                .iter()
                .filter(move |fragment| {
                    fragment.height > crate::units::Px::ZERO
                        || Some(fragment.page_index) == first_page
                })
                .map(|fragment| fragment.page_index)
        })
        .collect();

    let mut table_pages = BTreeMap::<u32, f32>::new();
    for (node_id, mut source) in body_geometry {
        let pages_before_sort: Vec<u32> = source
            .fragments
            .iter()
            .map(|fragment| fragment.page_index)
            .collect();
        source.fragments.sort_by_key(|fragment| fragment.page_index);
        if node_id == header.table_id {
            // Keep a zero-height slice only where body content shares the page
            // (the occupied-strip case); drop it when the table has nothing
            // left to show there.
            source.fragments.retain(|fragment| {
                fragment.height > crate::units::Px::ZERO
                    || body_content_pages.contains(&fragment.page_index)
            });
            for fragment in &mut source.fragments {
                // The first page's fragment starts at `full_band_top`; the
                // table itself starts below the rows that skip it.
                let lifted = if fragment.page_index == first_page {
                    lift
                } else {
                    0.0
                };
                fragment.y = (fragment.y.to_f32() + lifted).as_px();
                fragment.height =
                    (fragment.height.to_f32() + header.band_height_px - lifted).as_px();
                let table_top = fragment.y.to_f32();
                table_pages
                    .entry(fragment.page_index)
                    .and_modify(|top| *top = top.min(table_top))
                    .or_insert(table_top);
            }
        } else {
            // Drop the continuations that live only on pages the table just
            // gave up. `draw_block_inner_paint` paints a zero-height fragment
            // of a split block at its full `layout_size`, so a styled cell or
            // wrapper left behind would still show — at the offset of a header
            // that is no longer there, possibly over the following sibling.
            source
                .fragments
                .retain(|fragment| body_content_pages.contains(&fragment.page_index));
            for fragment in &mut source.fragments {
                fragment.y = (fragment.y.to_f32() + header.band_height_px).as_px();
            }
        }
        let target = geometry.entry(node_id).or_default();
        target.is_repeat |= source.is_repeat;
        // `line_boundaries` indexes `fragments` by position, so it survives
        // only when the list it was published for is the one that is kept:
        // nothing before it in the target, and no fragment reordered or
        // dropped above. Otherwise it stays empty and the renderer
        // re-derives the partition.
        let fragments_unchanged = target.fragments.is_empty()
            && source
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .eq(pages_before_sort.iter().copied());
        if fragments_unchanged {
            target.line_boundaries = std::mem::take(&mut source.line_boundaries);
        }
        target.fragments.extend(source.fragments);
    }

    table_pages.entry(first_page).or_insert_with(|| {
        geometry
            .entry(header.table_id)
            .or_default()
            .fragments
            .insert(
                0,
                Fragment {
                    page_index: first_page,
                    x: parent_x_in_body.as_px(),
                    y: first_table_top.as_px(),
                    width: parent_w.as_px(),
                    height: first_band.as_px(),
                },
            );
        first_table_top
    });
    let table_pages = table_pages.into_iter().collect::<Vec<_>>();
    append_repeated_header_fragments(
        geometry,
        if lift > 0.0 {
            &first_page_template
        } else {
            &header_template
        },
        &header_template,
        &table_pages,
        first_table_top,
        remaining_repeat_budget,
    );

    // The band only occupies space on pages that kept a table fragment. A
    // trailing forced break can leave the end page with none (see the `retain`
    // above), and adding the band here would push the table's next sibling down
    // by a header that was never drawn.
    let end_page_carries_band = table_pages.iter().any(|&(page, _)| page == body_end_page);
    let end_cursor = if end_page_carries_band {
        body_end_cursor + header.band_height_px
    } else {
        body_end_cursor
    };
    // A table that ends on its first page ends higher than its layout box by
    // the skipped rows; what follows it on that page moves up with it.
    // Such a table ends at its own bottom, its border included, less the
    // skipped rows, as one placed whole would: the body's cursor stops at
    // its last row.
    let end_cursor = if body_end_page == first_page && lift > 0.0 {
        geometry.entry(header.table_id).or_default().lift_px = lift;
        *lifted += lift;
        let table_height = doc
            .get_node(header.table_id)
            .map_or(0.0, |table| table.final_layout.size.height);
        end_cursor.max(first_table_top + table_height - lift)
    } else {
        end_cursor
    };
    (body_end_page, end_cursor)
}

/// fulgur-g9e3.1: split a block across pages by walking its children and
/// emitting per-page fragments for both the block itself and its children.
/// This is the fragmenter's one walker: the body is its root container
/// (`root`, from [`PaginationLayoutTree::fragment_pagination_root`]), and
/// every child it enters is walked by it again.
///
/// For each in-flow child, if it does not fit in the remaining strip, the
/// walk moves to the next page and places it there. A child whose subtree
/// would split (`would_split_block_subtree`), or holds a forced break or a
/// page-name change, is entered from the current cursor
/// (`fragment_block_subtree`), so it splits in place. A multi-line inline
/// root splits between its lines (`fragment_inline_root`), by its
/// `orphans` and `widows`; its line metrics do not include its own padding.
/// fulgur-a36m (Phase 3.1.5b): `break-before: page` and `break-after: page`
/// on a child force a page boundary.
///
/// The parent's fragment on each page it spans is closed through the
/// cursor ([`PageCursor::close_parent`]); the collectors
/// (`collect_string_set_states`, `collect_counter_states`,
/// `collect_bookmark_entries`) read these per-page fragments. The root has
/// none: its one fragment, on page 0, is its layout box, recorded before its
/// walk. The root differs in one more way only: a forced break before its
/// first child is the document's start, and makes no page.
///
/// A child taller than a page that the walk does not enter is sliced, one
/// fragment a page, where it lies in block flow ([`slices_where_it_is`]).
///
/// Skips out-of-flow and whitespace-text children. Bails at
/// [`crate::MAX_DOM_DEPTH`] — any nodes below that depth go unrecorded
/// (matches `record_subtree_descendants`).
///
/// With `header`, the walk places the rows of a table below the header it
/// repeats ([`fragment_repeating_table`]): it meets the table's body cells
/// only, from the body's top, and compares no page names between them.
///
/// Returns `(final_page_index, final_cursor_y)`: the page and y where
/// the parent's last child finished. The caller resumes its outer
/// cursor from these values.
#[allow(clippy::too_many_arguments)]
fn fragment_block_subtree_inner(
    geometry: &mut PaginationGeometryTable,
    flow: Flow<'_>,
    parent_id: usize,
    parent_w: f32,
    parent_x_in_body: f32,
    page_in: u32,
    cursor_in: f32,
    page_height_px: f32,
    depth: usize,
    header: Option<&RepeatingTableHeader>,
    entered_below_content: bool,
    page_has_content: &mut bool,
    remaining_repeat_budget: &mut usize,
    lifted: &mut f32,
    root: bool,
) -> (u32, f32) {
    let Flow { doc, .. } = flow;
    // The parent's fragment on each page starts at the cursor's
    // `page_start_y`: the walk closes one and starts the next each time it
    // crosses a page boundary. fulgur-kv0r: `page_taffy_origin` is the
    // parent-relative y of the first in-flow child on the current page
    // strip. Taffy's `layout.location.y` is in the parent's full coordinate
    // system (same value across page splits); each child's page-local y
    // becomes `page_start_y + (this_top_in_parent - page_taffy_origin)`,
    // which gives:
    // - block siblings: sequential placement (same as cursor-advance)
    // - grid / flex parallel siblings: same y (Taffy reports same
    //   `location.y` for cards in the same row, so the offset
    //   collapses to the row's first y).
    let mut cursor = PageCursor {
        page_index: page_in,
        cursor_y: cursor_in,
        page_start_y: cursor_in,
        page_taffy_origin: header.map_or(0.0, |header| header.body_origin_px),
        entry_page: page_in,
        entered_below_content,
        page_has_content: *page_has_content,
        page_height_px,
        page_foot_px: flow.page_foot(page_height_px),
        // The root's one fragment is recorded before its walk.
        parent: (!root).then_some(ParentSink {
            id: parent_id,
            x: parent_x_in_body,
            width: parent_w,
            row_pages: None,
        }),
    };
    if depth >= crate::MAX_DOM_DEPTH {
        // Bailed: emit a single whole-fragment for the parent at its
        // entry coordinates so geometry still has an entry for it.
        let h = doc
            .get_node(parent_id)
            .map(|n| n.final_layout.size.height)
            .unwrap_or(0.0);
        cursor.close_parent(geometry, ParentHeight::Measured(h));
        *page_has_content |= h > 0.0;
        return (page_in, cursor_in + h);
    }
    let Some(parent) = doc.get_node(parent_id) else {
        return (page_in, cursor_in);
    };
    // The lift of the children that ended on their page, which moved the
    // siblings after them up: the parent ends that much higher too.
    let mut own_lift = 0.0_f32;
    let mut origin_pending_target_y: Option<f32> = None;
    // The line of boxes side by side the walk is placing ([`LineState`]),
    // and where the cells that span rows below it ended ([`SpanEnd`]).
    let mut line: Option<LineState> = None;
    let mut spans: Vec<SpanEnd> = Vec::new();
    // Outside a table row or a flex or grid line (whose [`LineState`] does
    // the same): the line
    // of the previous in-flow child with height ([`flow_lines`]), and the
    // page and `page_has_content` where the children side by side with it
    // started. Blitz lays a table out as a grid of its cells, so the cells
    // of a row are such children; a child beside another is placed below
    // what was on the page when the first of them started, not below its
    // neighbour.
    let mut prev_line: Option<usize> = None;
    let mut beside_start = (cursor.page_index, cursor.page_has_content);
    // fulgur-uebl: flex / grid containers establish a flex/grid
    // formatting context where children are not class A break points
    // (CSS Fragmentation 3 §3.2). The `page` property doesn't apply to
    // flex / grid items, so we suppress the implicit-forced-break
    // comparison among them. Atomic inline containers (`inline-block`,
    // `inline-flex`, `inline-grid`) are similarly opaque from a
    // pagination perspective — their internal block flow does not
    // paginate independently, so sibling comparison among their
    // children would just produce spurious breaks. Orthogonal-flow
    // containers (writing-mode different from their own parent) are
    // also treated atomically per CSS Writing Modes 4 §9. Inner
    // block-level descendants in their own BFC still get the
    // comparison via deeper recursion.
    let parent_is_orthogonal = parent
        .parent
        .and_then(|gp_id| doc.get_node(gp_id))
        .is_some_and(|gp| crate::blitz_adapter::is_orthogonal_to_parent(gp, parent));
    let allow_same_row_rebase =
        header.is_some() || crate::blitz_adapter::is_flex_or_grid_container_node(parent);
    let suppress_page_check = allow_same_row_rebase
        || crate::blitz_adapter::is_atomic_inline_container_node(parent)
        || parent_is_orthogonal;
    // The children lie side by side in lines the walk places as units
    // ([`LineState`]): a table's cells in its rows, or a flex or grid
    // container's items in its lines.
    let parallel = allow_same_row_rebase || displays(parent, DisplayInside::Table);

    // fulgur-yb27: prefer `layout_children` over raw `children` —
    // same rationale as `record_subtree_descendants`. When a block
    // container has mixed block-level and inline-level children, Stylo
    // synthesizes anonymous block wrappers around the inline-level
    // siblings (CSS 2.1 §9.2.1.1). Those wrappers carry their own Taffy
    // layout and `node_id`, but they live ONLY in `layout_children`.
    // Walking raw `children` would re-visit the underlying inline runs
    // without their Taffy layout (so they'd land at the parent's origin
    // instead of after their predecessor block), skip the per-anon-block
    // break decision, and leave the wrapper without geometry, so
    // `dispatch_fragment` would skip its paragraph (fulgur-bq6i: a body
    // with `<label>`, `<fieldset>`, then `<select>` lost its text).
    //
    // Cross-page recursion correctness depends on fulgur-oc51's
    // parent-fragment push above — flipping this walk to
    // `layout_children` without that fix would lose the parent's
    // pre-recursion-page fragment in mo-006/008 (flex/grid + tall
    // monolithic + trailing inline text).
    let walk_children = header.map_or_else(
        || walk_children_of(parent),
        |header| header.body_cell_ids.clone(),
    );
    // The class A break opportunity before each child: the value of the
    // previous sibling's `break-after` and this one's `break-before`, and of a
    // change of used page name between them (fulgur-uebl, CSS Page 3 §5.3).
    // The lines of the children, by the index of each line's first child,
    // and whether the walk meets them in block flow, where `avoid` between
    // two lines keeps them together ([`kept_after_line`]); a table's rows and
    // a flex or grid container's lines are another walk's.
    let (lines, mut opportunities) = if header.is_some() {
        let lines = flow_lines(flow, parent, &walk_children);
        let values = opportunities(
            flow,
            &walk_children,
            &lines,
            suppress_page_check,
            page_height_px,
            depth + 1,
        );
        (lines, values)
    } else {
        let walk = walk_breaks(flow, parent, page_height_px, depth);
        (walk.lines.clone(), walk.values.clone())
    };
    if let Some(index) = flow
        .forced_before
        .and_then(|cell| walk_children.iter().position(|&id| id == cell))
    {
        opportunities[index] = opportunities[index].then(BreakValue::Forced(None));
    }
    let mut line_at = vec![None; walk_children.len()];
    // The line each flow child belongs to.
    let mut line_of = vec![None; walk_children.len()];
    for (number, line) in lines.iter().enumerate() {
        line_at[line[0]] = Some(number);
        for &index in line {
            line_of[index] = Some(number);
        }
    }
    let block_flow = header.is_none() && !allow_same_row_rebase && is_block_flow(flow, parent);
    for (index, (&child_id, &opportunity)) in walk_children.iter().zip(&opportunities).enumerate() {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        // CSS 2.1 §10.6.4 / §9.6: out-of-flow elements (`position:
        // absolute` / `position: fixed`) do not contribute to their
        // containing block's normal-flow height, so the fragmenter must not
        // advance cursors for them. Abs/fixed children are handled by
        // separate passes (`append_position_fixed_fragments` and the abs
        // positioning pipeline in `render`). White space is skipped too.
        if leaves_the_flow(child) {
            continue;
        }
        let layout = child.final_layout;
        // fulgur-s67g Phase 2.2: `position: running()` named elements are
        // removed from the flow and placed into `@page` margin boxes per
        // page; they take no room, at any depth. Phase 3.4 follow-up (PR
        // #296 Devin): a zero-height fragment at the cursor still enters
        // the running node into `geometry`, on the page on which its source
        // position lands — what `collect_running_element_states` needs to
        // map running instances to their per-page state. The engine's
        // stylesheet gives a running element `display: none`; one the
        // layout still gives a height (an inline `position: running()`)
        // gives that room back: what follows it moves up by its height, as
        // the siblings after a lifted table do.
        if is_running(flow, child) {
            let height = layout.size.height;
            if height.is_finite() && height > 0.0 && !allow_same_row_rebase {
                cursor.page_taffy_origin += height;
                own_lift += height;
            }
            if child.element_data().is_some() {
                geometry
                    .entry(child_id)
                    .or_default()
                    .fragments
                    .push(Fragment {
                        page_index: cursor.page_index,
                        x: (parent_x_in_body + layout.location.x).as_px(),
                        y: cursor.cursor_y.as_px(),
                        width: 0.0_f32.as_px(),
                        height: 0.0_f32.as_px(),
                    });
            }
            continue;
        }
        // fulgur-2m6w: defend against a non-finite Taffy height (`+inf` /
        // `NaN`, or an `f32::MAX` height that overflows to `+inf` after one
        // `cursor_y += child_h`). A `NaN` height slips past the slice gate
        // (NaN compares false) and would poison `cursor_y` / `page_start_y`
        // for the parent and every following sibling. Treat it as zero
        // height so it falls into the `child_h <= 0.0` branch, where the
        // node still enters geometry.
        let child_h = if layout.size.height.is_finite() {
            layout.size.height
        } else {
            0.0
        };
        let child_w = if layout.size.width > 0.0 {
            layout.size.width
        } else {
            parent_w
        };

        // Compute Taffy parent-relative top early — both the zero-
        // height path below and the non-zero path further down use
        // it (and break-before / break-after rebases the
        // `page_taffy_origin` against it on page advance).
        let this_top_in_parent = layout.location.y;

        // The line of boxes side by side this child belongs to, in a table
        // or a flex or grid container ([`LineState`]). Must run before
        // `origin_pending_target_y` is consumed: the line before it ends
        // here, and sets it when it crossed pages. A later box of the open
        // line starts where the line does, on its first page at its top,
        // whatever its neighbours before it placed (fulgur-ysms).
        let line_number = if parallel { line_of[index] } else { None };
        if let Some(open) = line.take_if(|open| Some(open.number) != line_number) {
            if let Some(target_y) = close_line(geometry, doc, open, &mut spans, false, &mut cursor)
            {
                origin_pending_target_y = Some(target_y);
            }
        }
        if let Some(open) = &line {
            cursor.restore(open.start);
            origin_pending_target_y = None;
        }
        let in_line = line_number.is_some();

        if let Some(target_y) = origin_pending_target_y.take() {
            cursor.page_taffy_origin = this_top_in_parent - (target_y - cursor.page_start_y);
        }

        if child_h <= 0.0 {
            // Phase 2.3 fix: zero-height **element** nodes still
            // need to enter geometry so their counter / string-set
            // / bookmark markers participate in the parity walks.
            //
            // Zero-height children skip the inter-child gap, so
            // break-before can fire here without first folding gap into
            // cursor_y. fulgur-p3uf (Phase 3.1.5a): a bare `<img>` with no
            // explicit dimensions or a pseudo-only `<div>` arrives here
            // after Blitz's intrinsic-size collapse, and its
            // `break-before: page` still takes effect
            // (`tests/pseudo_only_break_before.rs`).
            if opportunity.is_forced() && !cursor.at_page_top() {
                // Zero-height break-before: this child IS the first
                // on the new page — apply origin rebase eagerly.
                cursor.break_before(geometry, doc, opportunity.side(), this_top_in_parent);
                origin_pending_target_y = None;
            } else if !suppress_page_check && cursor.page_index > 0 && cursor.at_page_top() {
                cursor.move_to_side(doc, opportunity.side());
            }
            if child.element_data().is_some() {
                geometry
                    .entry(child_id)
                    .or_default()
                    .fragments
                    .push(Fragment {
                        page_index: cursor.page_index,
                        x: (parent_x_in_body + layout.location.x).as_px(),
                        y: cursor.cursor_y.as_px(),
                        width: child_w.as_px(),
                        height: 0.0_f32.as_px(),
                    });
            }
            // A line of only no-break spaces is content, which CSS lays out as
            // a line box and the layout gives no height ([`holds_a_dropped_line`]).
            if holds_a_dropped_line(child) {
                cursor.page_has_content = true;
            }
            continue;
        }

        if !parallel {
            let beside = prev_line.is_some() && prev_line == line_of[index];
            if beside {
                cursor.page_has_content = cursor.page_index == beside_start.0 && beside_start.1;
            } else {
                beside_start = (cursor.page_index, cursor.page_has_content);
            }
            prev_line = line_of[index];
        }

        // fulgur-kv0r: place the child at its Taffy-reported parent-
        // relative y, offset by the parent's start on the current
        // page (`page_start_y`) and rebased against
        // `page_taffy_origin` so the first child on each page strip
        // lands at `page_start_y` regardless of its absolute parent
        // y. For grid / flex parallel siblings (same `location.y`),
        // this places them at the same page-local y; for sequential
        // block flow, it matches Taffy's stacked positions exactly.
        let mut child_page_y =
            cursor.page_start_y + (this_top_in_parent - cursor.page_taffy_origin);

        // A forced break before the child, when content precedes it on the
        // page: the parent's fragment ends at the last content, the gap above
        // the child not included. With nothing placed on the page yet, here
        // or above, a break makes no empty page: a side break only moves on
        // to its side.
        if opportunity.is_forced() && !cursor.at_page_top() {
            // The breaking child is the first in-flow child on the new page
            // strip. css-break-3 §5.2 truncates the margins before a forced
            // break and keeps those after it: the child starts its own top
            // margin below the page top ([`adjoining_top_margin`]), the gap
            // to the box before it dropped.
            let margin = adjoining_top_margin(flow, child, true).max(0.0);
            cursor.break_before(
                geometry,
                doc,
                opportunity.side(),
                this_top_in_parent - margin,
            );
            child_page_y = margin;
        } else if !suppress_page_check && cursor.page_index > 0 && cursor.at_page_top() {
            cursor.move_to_side(doc, opportunity.side());
        }
        // The first box of a line opens it, after any forced break before
        // it: the line moves to the next page whole when it starts here with
        // no box of it placing content ([`line_moves_on`]), then each of its
        // boxes starts where it does. No box of a line moves alone.
        if let Some(number) = line_number
            && line.is_none()
        {
            let items: Vec<&blitz_dom::Node> = lines[number]
                .iter()
                .filter_map(|&i| doc.get_node(walk_children[i]))
                .collect();
            let top = items
                .iter()
                .map(|item| item.final_layout.location.y)
                .fold(this_top_in_parent, f32::min);
            let line_y = child_page_y - (this_top_in_parent - top);
            let after_content = cursor.break_point_before(line_y);
            // A line followed by an `avoid` moves to the next page with the
            // run the opportunity joins it to ([`kept_after_line`]) when the
            // run does not fit the rest of the page but fits a page, as a
            // box placed whole that starts a run in block flow does. A line
            // inside a run (an `avoid` before it) follows the run's first.
            let height = line_bottom(&items) - top;
            let kept = if opportunity == BreakValue::Avoid {
                0.0
            } else {
                kept_after_line(
                    flow,
                    parent,
                    &walk_children,
                    &lines,
                    &opportunities,
                    number,
                    page_height_px,
                    depth + 1,
                )
            };
            let run_moves = kept > 0.0
                && after_content
                && !cursor.at_page_top()
                && line_y + height + kept > page_height_px
                && flow.fits_page(height + kept, page_height_px);
            if run_moves
                || line_moves_on(
                    flow,
                    &items,
                    top,
                    line_y,
                    after_content,
                    cursor.at_page_top(),
                    page_height_px,
                    depth + 1,
                )
            {
                // A table's first row leaves nothing of the table behind.
                let leaves_nothing =
                    displays(parent, DisplayInside::Table) && !cursor.holds_placed();
                // The line keeps the highest top margin its boxes keep above
                // it, as far as the line still fits its page ([`kept_margin`]).
                let above = items
                    .iter()
                    .map(|item| {
                        adjoining_top_margin(flow, item, false)
                            - (item.final_layout.location.y - top)
                    })
                    .fold(0.0, f32::max);
                let unit = height + if run_moves { kept } else { 0.0 };
                let margin = kept_margin(above, unit, page_height_px);
                cursor.move_before(geometry, top - margin, leaves_nothing);
                child_page_y = this_top_in_parent - top + margin;
            }
            line = Some(LineState {
                number,
                start: cursor.mark(),
                started_on_occupied_strip: cursor.entered_below_content,
                max_end_page: cursor.page_index,
                max_end_cursor_y: cursor.cursor_y,
                max_end_has_content: cursor.page_has_content,
                bottom: line_bottom(&items),
                items: Vec::new(),
            });
            cursor.open_row();
        }
        if let Some(open) = line.as_mut() {
            open.items.push(child_id);
        }
        // Update the cursor only when the child's bottom advances
        // past it. For block flow this matches cursor advancing by
        // `gap + child_h`; for grid parallel siblings the cursor
        // tracks the row's max bottom (so break-before / overflow
        // checks see the full row height).
        cursor.cursor_y = cursor.cursor_y.max(child_page_y);

        // (Strip-overflow page cut moved below the recursion gate as
        // part of fulgur-7hf5 — see the `if cursor_y > page_start_y
        // && cursor_y + child_h > page_height_px` block after the
        // gate. The gate must run from the **current** cursor so an
        // in-place split produces a `WithinChild`-shaped result on
        // the current strip, not a pre-advanced fresh page.)

        let child_x_in_body = parent_x_in_body + layout.location.x;

        // What the child is to the walk: placed whole, split between its
        // lines, or entered for its children ([`classify`]). A cell of a
        // row kept whole places whole too, and the row's extent, not the
        // cell's, decides whether it moves on.
        let kind = classify(flow, child, page_height_px);
        // A line followed by an `avoid` keeps the boxes after it that the
        // opportunity joins it to ([`kept_after_line`], css-break-3 §4.4
        // rule 1). When the run does not fit the rest of the page, the break
        // goes at the last opportunity before it that is not `avoid`: a box
        // the walk places whole that starts the run moves to the next page
        // when the run fits one, and a block container, a table, or a flex
        // or grid container is entered, so that the break goes inside it,
        // before the last of its boxes or lines that starts the run there.
        // A box inside a run (an `avoid` before it) follows
        // the run's first box. With nothing placed on the page above, or a
        // run taller than a page, `avoid` is relaxed (rule 4): the walk
        // breaks where the content reaches the foot.
        let mut enter_for_kept_run = false;
        // What an inline root's last lines keep below them on their page: the
        // root's bottom edge and the run after it.
        let mut inline_kept_after = 0.0_f32;
        if block_flow
            && opportunity != BreakValue::Avoid
            && let Some(line) = line_at[index]
        {
            let kept = kept_after_line(
                flow,
                parent,
                &walk_children,
                &lines,
                &opportunities,
                line,
                page_height_px,
                depth + 1,
            );
            let run = extent(flow, child, depth + 1) + kept;
            if kept > 0.0 && child_page_y + run > page_height_px {
                match &kind {
                    BoxKind::BlockFlow | BoxKind::Table { .. } | BoxKind::Parallel { .. } => {
                        enter_for_kept_run = true;
                    }
                    // A paragraph keeps only its trailing unit, its last
                    // `widows` lines, with the run: `avoid` forbids the break
                    // after it, not those between its lines
                    // ([`fragment_inline_root`]).
                    BoxKind::InlineRoot { lines, limits } if lines.len() > 1 => {
                        let below =
                            (child_h - top_inset(child) - lines[lines.len() - 1].1).max(0.0) + kept;
                        let first = lines.len().saturating_sub(limits.widows.max(1));
                        let trailing = lines[lines.len() - 1].1 - lines[first].0;
                        if flow.fits_page(trailing + below, page_height_px) {
                            inline_kept_after = below;
                        }
                    }
                    BoxKind::Monolithic(_) | BoxKind::InlineRoot { .. } => {
                        if !cursor.at_page_top() && flow.fits_page(run, page_height_px) {
                            child_page_y = move_on(
                                &mut cursor,
                                geometry,
                                flow,
                                child,
                                this_top_in_parent,
                                run,
                            );
                        }
                    }
                }
            }
        }
        // fulgur-2s7p.1: an inline root splits at line boundaries. Without
        // the line split a multi-line inline root falls through to the
        // block-child gate below, which can only push the *whole* child to
        // the next page or emit it as one oversized fragment, and a first
        // in-flow child whose own content overflows the remaining strip
        // never gets a chance to fragment (P0 silent content loss), because
        // `child_page_y > page_start_y` is false for it and
        // `would_split_block_subtree` only sees zero-height text nodes as
        // its DOM children. One kept whole or transformed is monolithic
        // ([`classify`]), and a child taller than a page is not kept whole:
        // it splits between its lines instead of being placed as one
        // fragment that runs past the page foot.
        if let BoxKind::InlineRoot {
            lines: line_metrics,
            limits,
        } = &kind
            && line_metrics.len() > 1
        {
            // Split-before fallback (CSS Fragmentation §4.2, class A
            // break point): if the paragraph crosses the foot of the
            // remaining strip, cannot split there within its `orphans` /
            // `widows` (its leading lines, [`inline_leading`]), and
            // something is already on this page, push to a fresh page
            // before splitting. When this container's first in-flow child
            // is itself this paragraph (`cursor_y == page_start_y`, the
            // fulgur-2s7p.1 repro shape), this is skipped and the split
            // begins right where the container currently sits.
            let limits = *limits;
            // codex P1: compare against `child_page_y` (this child's
            // own row-local start), not `cursor_y` — for a later
            // grid/flex row cell, `cursor_y` is the previous cell's row
            // bottom, which can overflow the page even when this cell
            // would fit fine at its own row-local y. The leading
            // `holds_placed` guard answers whether anything of
            // this container precedes the paragraph on this page (the
            // fulgur-2s7p.1 first-child exemption), regardless of row
            // position; `at_page_top` whether the page holds any content,
            // which a move whole would leave it without. A box of a line
            // never moves alone: its line moved or stayed already.
            if !in_line
                && cursor.holds_placed()
                && !cursor.at_page_top()
                && !inline_leading(
                    line_metrics,
                    limits,
                    top_inset(child),
                    page_height_px - child_page_y,
                )
                .places
            {
                // The parent's fragment ends at the content placed on the
                // page (plain block flow, e.g. a `break-inside:avoid` block
                // pushed to a fresh page): reaching the page foot paints
                // the parent's background far past its content and pushes
                // whatever comes after it onto a spurious extra page
                // (confirmed regression in examples/break-inside via PR
                // #741). The paragraph's leading lines start the next page.
                let lead = inline_leading(line_metrics, limits, top_inset(child), page_height_px);
                let unit = lead.extent + line_metrics.first().map_or(0.0, |line| line.0);
                child_page_y =
                    move_on(&mut cursor, geometry, flow, child, this_top_in_parent, unit);
            }
            // codex P1: start from `child_page_y` (this child's own
            // rebased page-local y), not the parent's running
            // `cursor_y` — for block flow the two are equal, but for
            // grid / flex parallel siblings `cursor_y` still holds the
            // previous sibling's row bottom while `child_page_y` is
            // this cell's row-local start (fulgur-u0p0, mirrored from
            // the recursion branch below).
            let mut paragraph = PageCursor::new(cursor.page_index, child_page_y, page_height_px);
            paragraph.page_has_content = cursor.page_has_content;
            let split_frag_count = fragment_inline_root(
                geometry,
                child_id,
                child_x_in_body,
                child_w,
                &mut paragraph,
                line_metrics,
                limits,
                top_inset(child),
                inline_kept_after,
            );
            let (new_page_index, new_cursor_y) = (paragraph.page_index, paragraph.cursor_y);
            cursor.page_has_content = paragraph.page_has_content;
            record_inline_root_descendants(geometry, doc, child_id, line_metrics, split_frag_count);
            // Same shape as the recursion branch's fulgur-oc51 fix
            // below: without a parent fragment per page the split
            // crossed, the parent's own background / border would
            // disappear from every page but the last.
            if cursor.after_box(geometry, new_page_index, new_cursor_y, in_line) {
                origin_pending_target_y = Some(new_cursor_y);
            }
            cursor.entered_below_content = false;
            reach_line(&mut line, &mut spans, child, &cursor);
            continue;
        }

        // fulgur-7hf5 (Phase 3.1.5c): unified recursion gate — recurse
        // whenever the child's subtree would split (in-place,
        // truly-oversized, or forced-break-below). The recursion
        // enters from the current cursor so an in-place split
        // produces a `WithinChild`-shaped result on the current page
        // strip and a tail on the next.
        //
        // `would_split_block_subtree` returns `false` when all the
        // child's grandchildren fit the available strip — protects
        // against the "parent CSS height > children sum" case where
        // recursion would emit a parent fragment shorter than
        // expected.
        // A box of a line measures from its own place: the boxes beside it
        // placed before it do not take its room.
        let placed_to = if in_line {
            child_page_y
        } else {
            cursor.cursor_y
        };
        let available_strip = (page_height_px - placed_to).max(0.0);
        // fulgur-7hf5: a multicol container is not entered unless a child
        // spans its columns ([`classify`]), a box kept whole only for a
        // forced break or a page-name change below it, which are not the
        // unforced break `avoid` concerns itself with (fulgur-2s7p.5), and a
        // transformed box only when it is taller than a page.
        // At the top of a page, below a top edge, a box kept whole that does
        // not fit the room left is entered: `avoid` is relaxed (css-break-3
        // §4.4), and the box splits between its children, rather than move
        // and leave the edge alone on the page. No break point precedes it,
        // so the room reaches the page foot, below the page-foot clearance.
        let relaxes_kept = cursor.at_page_top()
            && matches!(kind, BoxKind::Monolithic(Whole::Kept))
            && !child.children.is_empty()
            && overflows(
                child_page_y + extent(flow, child, depth + 1),
                flow.page_foot(page_height_px),
            );
        let needs_recursion = enter_for_kept_run
            || relaxes_kept
            || enters(flow, child, &kind, available_strip, page_height_px);
        // A box entered after content in its container moves whole to the
        // next page first, and is entered there, when nothing of it places in
        // the rest of the page: its leading unit does not fit
        // ([`leading_extent`]) but fits a whole page, as a box not entered
        // moves (the strip-overflow cut below). At the top of a page, below a
        // padded top edge only, it moves too: css-break-3 §4.1 has no break
        // point there, so the unit fits only the next page. A box of a
        // line moves with its line instead ([`line_moves_on`]).
        if needs_recursion && !in_line && cursor.break_point_before(child_page_y) {
            let lead = leading_of(
                flow,
                child,
                &kind,
                page_height_px - child_page_y,
                page_height_px,
                depth + 1,
            );
            if lead.moves_on(flow.page_foot(page_height_px))
                && !(cursor.at_page_top() && lead.relaxes)
            {
                child_page_y = move_on(
                    &mut cursor,
                    geometry,
                    flow,
                    child,
                    this_top_in_parent,
                    lead.extent,
                );
            }
        }
        if needs_recursion {
            let occupied_before_recursion = cursor.entered_below_content;
            let row_started_on_occupied_strip = line.as_ref().is_some_and(|open| {
                open.started_on_occupied_strip && open.start.page_index == cursor.page_index
            });
            let pre_recursion_page = cursor.page_index;

            // fulgur-u0p0: enter recursion from `child_page_y` (the rebased
            // page-local y of THIS child) instead of the parent's running
            // `cursor_y`. For block flow, `child_page_y == cursor_y` after
            // the cursor_y update above, so this is a no-op. For grid / flex
            // parallel siblings (same Taffy `location.y` as a previous
            // sibling on this page), `cursor_y` still holds the previous
            // sibling's bottom — passing it would stack the parallel cell
            // below the previous one instead of beside it. Using
            // `child_page_y` keeps each cell's recursion strip aligned to
            // the row's y on the current page.
            let mut child_lift = 0.0;
            // A box of a line that starts below content on its page is
            // entered so: its first content that does not fit goes on to
            // the next page, as the line stays for the boxes beside it; a
            // cell that spans rows too, which ends the last row it spans,
            // whose rows are placed beside it ([`SpanEnd`]).
            let below_content = cursor.entered_below_content
                || line.as_ref().is_some_and(|open| {
                    open.start.page_has_content && open.start.page_index == cursor.page_index
                });
            let (np, nc) = fragment_block_subtree(
                geometry,
                flow,
                child_id,
                child_w,
                child_x_in_body,
                cursor.page_index,
                child_page_y,
                page_height_px,
                depth + 1,
                below_content,
                &mut cursor.page_has_content,
                remaining_repeat_budget,
                &mut child_lift,
            );
            let occupied_page_extent = if occupied_before_recursion || row_started_on_occupied_strip
            {
                fragmented_descendant_page_extent(
                    geometry,
                    doc,
                    child_id,
                    pre_recursion_page,
                    cursor.page_start_y,
                    depth + 1,
                )
                .max(0.0)
            } else {
                0.0
            };
            // The row's `start.entered_below_content` records whether the
            // page was already occupied when the ROW began, and every cell of
            // that row restores from it (see the `allow_same_row_rebase`
            // branch). It is a per-row constant, so a child consuming the
            // strip must not clear it: doing so left the row's second and
            // later cells believing they started on an empty page, skipping
            // the occupied-strip page-break check and painting past the page
            // bottom (fulgur-naj7.3: measured right-tall = [(0, y=80, h=30)],
            // bottom 110 on a 100px page, while the row's left cell had
            // already moved to page 1).
            cursor.entered_below_content = false;
            // If the recursion crossed a boundary, the parent's
            // current-page fragment must restart at y=0 on the new
            // page. Defensive `nc < page_start_y` guards against
            // backward cursor returns (impossible in normal flow).
            let crossed = np != pre_recursion_page || nc < cursor.page_start_y;
            if np > pre_recursion_page {
                // fulgur-oc51: emit parent fragments for every
                // page span the recursion crossed. Without this,
                // only the trailing close at the end of this
                // function emits a parent fragment (on the *last*
                // page), so the parent's background / borders
                // disappear from the previous and intermediate
                // pages. Pre-recursion overflow close (no-
                // recursion branch, line ~1713) covers the
                // analogous case for non-recursive children, but
                // there is no equivalent push when the page
                // advance happens *inside* the recursion.
                //
                // The parent's content extended to the page
                // bottom on every previous page (otherwise the
                // recursion would not have advanced past that
                // page), so the previous-page fragment spans
                // `[page_start_y, page_height_px]` and any
                // intermediate page is a full strip. It ends at the
                // page foot, not with the crossing child's whole
                // height: a box taller than a page in the child is
                // sliced, and a parent fragment that took its height
                // painted the child's background of every page on
                // the first (WPT css-page monolithic-overflow-006,
                // -008).
                let prev_height = if occupied_before_recursion || row_started_on_occupied_strip {
                    occupied_page_extent
                } else {
                    cursor.parent_height(ParentHeight::ToFoot)
                };
                cursor.after_crossing(geometry, np, nc, prev_height);
            } else {
                // fulgur-u0p0: when the recursion stayed on the same page,
                // keep the larger of the parent's existing `cursor_y` (row
                // max bottom from a previous parallel sibling) and the
                // recursion's returned `nc`.
                cursor.cursor_y = cursor.cursor_y.max(nc);
            }
            // A child that ended on this page higher than its layout box (a
            // table without its skipped header rows) takes the siblings
            // after it up with it, here and, through `lifted`, in the
            // parent's own siblings. Cells side by side in a row are not
            // moved: the row's height is not this child's alone.
            if np == pre_recursion_page && child_lift > 0.0 && line.is_none() {
                cursor.page_taffy_origin += child_lift;
                own_lift += child_lift;
            }
            if crossed {
                cursor.page_start_y = 0.0;
                origin_pending_target_y = Some(cursor.cursor_y);
            }

            reach_line(&mut line, &mut spans, child, &cursor);
            continue;
        }

        // No recursion — apply the strip-overflow page cut for
        // children that don't split (non-splittable, or splittable
        // but all grandchildren fit the available strip — the
        // parent-CSS-height-vs-children-sum case stays here).
        // Use `child_page_y + child_h` (the actual placement bottom)
        // rather than `cursor_y + child_h` so a parallel sibling
        // returning to a smaller page-local y is checked correctly.
        //
        // At the top of a page, below a top edge or a margin only
        // ([`PageCursor::at_page_top`]), a box moves only when it fits a
        // whole page, so fits only the next one: one taller stays, as moving
        // would gain nothing. A box of a line moves with its line, never
        // alone ([`line_moves_on`]).
        // A childless box at the top of a page that does not fit the room
        // left below a top edge is sliced from where it stands instead
        // (css-break-3 §4.4), in a table cell too. Slices are cut at the
        // page foot, below the page-foot clearance: a box that a break
        // cannot move ends there whole when it fits the page.
        let page_foot = flow.page_foot(page_height_px);
        let slices_at_top = cursor.at_page_top()
            && overflows(child_page_y + child_h, page_foot)
            && slices_at_a_page_top(child)
            && slices_where_it_is(doc, child, true);
        // A box of a line placed whole that crosses the page foot, neither
        // replaced nor transformed, is sliced where it stands, as the boxes
        // beside it go on from the line's start, not after it; one
        // stretched to the line's height also reaches the line's end when
        // the line closes ([`stretch_line_items`]).
        let slices_in_line = in_line
            && overflows(child_page_y + child_h, page_foot)
            && child_page_y < page_foot
            && !is_transformed(child)
            && !is_replaced(child)
            && slices_where_it_is(doc, child, true);
        if !in_line
            && !slices_at_top
            && cursor.break_point_before(child_page_y)
            && (!cursor.at_page_top() || flow.fits_page(child_h, page_height_px))
            && child_page_y + child_h > page_height_px
        {
            // Forced to a fresh page: the current child lands at its top,
            // and sequential siblings continue from this point.
            child_page_y = move_on(
                &mut cursor,
                geometry,
                flow,
                child,
                this_top_in_parent,
                child_h,
            );
        }
        cursor.entered_below_content = false;

        // fulgur-sbw2: a child whose CSS-resolved height alone
        // exceeds `page_height_px` (e.g. `<div height:300vh>`)
        // must be sliced across pages. The recursion gate above
        // returns false for this shape because the child's *own*
        // children fit the strip (the gate measures descendant
        // overflow, not the parent's intrinsic height), so we
        // would otherwise emit a single oversized fragment on the
        // current page and stop. Emit one fragment per page strip
        // with page-local y — `PaginationGeometry::is_split()`
        // flips to `true` automatically once `fragments.len() > 1`,
        // and render.rs picks the per-slice height accordingly.
        // Descendants are recorded against the *first* slice only;
        // exact mid-element pagination of nested content is still
        // future work (see `record_subtree_descendants` notes).
        //
        // Tolerance ([`fits`]): without the slack a 220pt spacer that
        // lays out 294 px tall on a 293.33 px strip is sliced into two
        // pages (gcpm_snapshot tests regress: spacer + page-break pair
        // becomes 2 + 1 instead of 1 + 1 per section), while truly
        // oversized content (`300vh ≈ 880 px on a 293-px strip`) is
        // still sliced.
        //
        // Atomic children opt out: a transformed subtree paints
        // as a single atomic box (CSS Transforms §6.1) — it must
        // never split across pages because the rotation /
        // skew / matrix is applied to a single shape, not
        // per-slice. `contain: size` is NOT excluded: a
        // `<div contain:size height:350vh>` still spans four
        // pages visually (WPT `monolithic-overflow-022-print`),
        // it's only the descendant content that's atomic.
        //
        // Nor is a box in a table cell or a flex or grid item, or one that
        // overflows an ancestor clipping it ([`slices_where_it_is`]), or a
        // first child entered below the page foot, which has no slice there.
        if slices_at_top
            || slices_in_line
            || (!is_transformed(child)
                && overflows(child_h, page_foot)
                && child_page_y < page_foot
                && slices_where_it_is(doc, child, true))
        {
            // The slices are placed from the child's own place on the page,
            // and the walk then goes on from the last one, as after an
            // inline root that split.
            let mut slices = PageCursor::new(cursor.page_index, child_page_y, page_foot);
            slices.page_has_content = cursor.page_has_content;
            let first_slice_h = (page_foot - child_page_y).min(child_h);
            geometry
                .entry(child_id)
                .or_default()
                .fragments
                .push(Fragment {
                    page_index: slices.page_index,
                    x: child_x_in_body.as_px(),
                    y: slices.cursor_y.as_px(),
                    width: child_w.as_px(),
                    height: first_slice_h.as_px(),
                });
            slices.place(first_slice_h);
            record_subtree_descendants(
                geometry,
                doc,
                child_id,
                slices.page_index,
                slices.cursor_y,
                child_x_in_body,
                depth + 1,
            );
            let mut remaining = child_h - first_slice_h;
            let mut last_slice_h = first_slice_h;
            // fulgur-ezst: a CHILDLESS block whose slicing would exceed
            // the cap is a pathological amplifier — `<div
            // style="height:99999999px">` is a web-only spacer/overflow
            // idiom that prints nothing but blank pages. Collapse it to
            // its single first slice: emit only that slice AND bound the
            // space it occupies (resume following content on the next
            // page) so the document does not balloon. Background / border
            // presence does NOT gate this — nobody authors a
            // >MAX_PAGES-tall filled band on purpose. A content-bearing
            // block, or a childless band that fits within the cap, is not
            // collapsed and takes the truncate-and-warn path below
            // unchanged.
            //
            // fulgur-c8re (security): a replaced element (`<img>` /
            // `<svg>`) is NOT special-cased out of this collapse —
            // painting or not. It only reaches this branch when it is
            // taller than `MAX_PAGES` pages (~10M px), a range no
            // legitimate single image occupies, so clipping it to one page
            // loses nothing real — even a *resolved* image, because a 1×1
            // bitmap stretched with `height:99999999px` is the same
            // amplifier as an unresolved `src`. Gating on tag name alone
            // (the removed `is_replaced_content`) let such a node amplify a
            // few bytes of HTML into ~`MAX_PAGES` blank pages (a validated
            // high-severity DoS).
            let collapse_childless = page_foot > 0.0
                && (remaining / page_foot).ceil() > crate::MAX_PAGES as f32
                && !subtree_has_rendered_content(doc, child_id, 0);
            if collapse_childless {
                // fulgur-c8re (Codex P1 on PR #575): bound the OCCUPIED
                // SPACE, not just the fragment pushes. `implied_page_count`
                // reads the max fragment index across ALL nodes, so if the
                // slice loop advanced `page_index` to `MAX_PAGES` a trailing
                // in-flow sibling (`<div huge></div><p>after</p>`) would be
                // stranded on a deep page and re-inflate the PDF to
                // ~`MAX_PAGES` blank pages. The first slice already emitted
                // above fills the current strip, so resume following content
                // on the next page.
                log::warn!(
                    "pagination: collapsed a childless block of height \
                     {child_h}px (slicing would exceed the {}-page limit) \
                     to a single page (fulgur-ezst)",
                    crate::MAX_PAGES,
                );
                slices.advance();
            } else {
                // fulgur-2m6w: cap the per-page-strip slicing at
                // `MAX_PAGES`. `child_h` is attacker-controlled CSS
                // (`height` / `vh`), so without this bound a few bytes of
                // HTML (`<div style="height:99999999px">`) generate ~10^5
                // fragments — and a non-finite height would never reduce
                // `remaining`, looping forever. The `page_index` ceiling is
                // load-bearing on its own (it stops the loop even when
                // `remaining` is `+inf`); content past the cap is truncated.
                while remaining > 0.0 && slices.page_index < crate::MAX_PAGES {
                    slices.advance();
                    last_slice_h = remaining.min(page_foot);
                    geometry
                        .entry(child_id)
                        .or_default()
                        .fragments
                        .push(Fragment {
                            page_index: slices.page_index,
                            x: child_x_in_body.as_px(),
                            y: 0.0_f32.as_px(),
                            width: child_w.as_px(),
                            height: last_slice_h.as_px(),
                        });
                    slices.place(last_slice_h);
                    remaining -= last_slice_h;
                }
                if remaining > 0.0 {
                    log::warn!(
                        "pagination: block height {child_h}px exceeds the \
                         {}-page limit; truncating remaining content to \
                         bound rendering (fulgur-2m6w)",
                        crate::MAX_PAGES,
                    );
                }
                slices.cursor_y = if child_h - first_slice_h > 0.0 {
                    last_slice_h
                } else {
                    slices.cursor_y + first_slice_h
                };
            }
            let (new_page_index, new_cursor_y) = (slices.page_index, slices.cursor_y);
            cursor.page_has_content = slices.page_has_content;
            // The parent's fragment reaches the foot of every page the
            // slices cross, as around an inline root that split.
            if cursor.after_box(geometry, new_page_index, new_cursor_y, in_line) {
                origin_pending_target_y = Some(new_cursor_y);
            }
            reach_line(&mut line, &mut spans, child, &cursor);
            continue;
        }

        // Child fits the strip (or is an atomic oversized leaf that
        // simply overflows below the page bottom — with a single
        // in-flow child there is no valid interior split point, so
        // the leaf emits whole). Emit its fragment and recurse into
        // descendants on the same page.
        geometry
            .entry(child_id)
            .or_default()
            .fragments
            .push(Fragment {
                page_index: cursor.page_index,
                x: child_x_in_body.as_px(),
                y: child_page_y.as_px(),
                width: child_w.as_px(),
                height: child_h.as_px(),
            });
        record_subtree_descendants(
            geometry,
            doc,
            child_id,
            cursor.page_index,
            child_page_y,
            child_x_in_body,
            depth + 1,
        );
        // Track the lowest point reached on this page so the
        // overflow / break-before checks above see the full row's
        // bottom for grid / flex parents (parallel siblings update
        // `cursor_y` to `max(cursor_y, child_page_y + child_h)` —
        // the per-row max bottom).
        cursor.cursor_y = cursor.cursor_y.max(child_page_y + child_h);
        cursor.place(child_h);
        reach_line(&mut line, &mut spans, child, &cursor);
    }

    // The last line ends the walk.
    if let Some(open) = line.take() {
        close_line(geometry, doc, open, &mut spans, true, &mut cursor);
    }

    // A parent entered only to lift a table on its page (it ends on the
    // page it started on) ends at its own bottom, its padding and border
    // included, less the lift, as it would were it placed whole: the
    // cursor alone stops at its last child.
    if own_lift > 0.0 && cursor.page_index == page_in {
        cursor.cursor_y = cursor
            .cursor_y
            .max(cursor.page_start_y + parent.final_layout.size.height - cursor.page_taffy_origin);
    }

    // Close the parent's fragment for the final page span. Always
    // emit at least one fragment so the parent is represented in
    // geometry — `collect_counter_states` and friends look up nodes
    // by id, and a missing entry would silently bypass the parity
    // gate via the early `counter_ops_by_node ⊄ geometry` check in
    // `render.rs`. Height may be 0 when every child was skipped
    // (whitespace / OOF / running) — that's intentional and matches
    // the zero-height-element path. The root closes no fragment.
    cursor.close_parent(geometry, ParentHeight::Placed);
    // The root's one fragment is its layout box, recorded before its walk.
    if own_lift > 0.0 && !root {
        geometry.entry(parent_id).or_default().lift_px = own_lift;
        *lifted += own_lift;
    }

    *page_has_content = cursor.page_has_content;
    (cursor.page_index, cursor.cursor_y)
}

/// fulgur-p55h: the `(top, bottom)` of each line box of a node's Parley
/// `inline_layout_data`, if any, in the paragraph's local coordinates, in
/// CSS pixels: each line `line_height` tall, from the bottom of the line
/// before it, the boxes Parley stacks to the layout's height and the
/// renderer slices a split paragraph by. Parley's `min_coord` and
/// `max_coord` are the selection box, which clamps a negative half-leading
/// to zero: with a `line-height` below the font's ascent and descent, each
/// line measured more than its line height, and a split paragraph ran past
/// its layout box. Returns an empty vec for non-inline-root nodes (block /
/// text / element with no inline children) so callers can branch on
/// `metrics.len() > 1` to decide between line-aware and block paths.
fn collect_inline_line_metrics(node: &blitz_dom::Node) -> Vec<(f32, f32)> {
    let Some(elem) = node.element_data() else {
        return Vec::new();
    };
    let Some(text_layout) = elem.inline_layout_data.as_deref() else {
        return Vec::new();
    };
    let mut bottom = 0.0_f32;
    text_layout
        .layout
        .lines()
        .map(|line| {
            let top = bottom;
            bottom = top + line.metrics().line_height;
            (top, bottom)
        })
        .collect()
}

/// The `orphans` and `widows` of an inline root, as line counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LineLimits {
    orphans: usize,
    widows: usize,
}

impl Default for LineLimits {
    /// The initial values (CSS Fragmentation 3 §4.4): 2 and 2.
    fn default() -> Self {
        Self {
            orphans: 2,
            widows: 2,
        }
    }
}

/// The `orphans` and `widows` of the inline root `node_id`. Both properties
/// are inherited, so each is the value its nearest element declares, from
/// the node up through its ancestors (an `inherit` passes on to the
/// parent), else the initial 2.
fn line_limits(
    doc: &BaseDocument,
    column_styles: Option<&crate::column_css::ColumnStyleTable>,
    node_id: usize,
) -> LineLimits {
    use crate::column_css::LineCount;

    let mut orphans = None;
    let mut widows = None;
    let mut next = Some(node_id);
    while let Some(id) = next {
        if orphans.is_some() && widows.is_some() {
            break;
        }
        if let Some(props) = column_styles.and_then(|table| table.get(&id)) {
            if let (None, Some(LineCount::Lines(count))) = (orphans, props.orphans) {
                orphans = Some(count);
            }
            if let (None, Some(LineCount::Lines(count))) = (widows, props.widows) {
                widows = Some(count);
            }
        }
        next = doc.get_node(id).and_then(|node| node.parent);
    }
    let initial = LineLimits::default();
    let lines = |count: Option<u32>, initial: usize| {
        count.map_or(initial, |count| {
            usize::try_from(count).unwrap_or(usize::MAX)
        })
    };
    LineLimits {
        orphans: lines(orphans, initial.orphans),
        widows: lines(widows, initial.widows),
    }
}

/// A table's row counts, `-fulgur-table-orphans` and `-fulgur-table-widows`,
/// as declared on it: at least `orphans` rows start on the page where the
/// table first breaks, else it moves to the next page whole
/// ([`table_leading`]), and at least `widows` rows, a row split between
/// pages counting as one, are on its last page ([`fragment_table_rows`]).
/// css-break-3 §3.3 gives `orphans` and `widows` to block containers that
/// hold lines only, which a table's cells inherit as such: these are an
/// extension of their own. `None` for a table that declares neither; a
/// count it leaves out is 1, no constraint.
fn table_line_limits(
    column_styles: Option<&crate::column_css::ColumnStyleTable>,
    table: &blitz_dom::Node,
) -> Option<LineLimits> {
    let props = column_styles?.get(&table.id)?;
    let count =
        |count: Option<u32>| count.map(|count| usize::try_from(count).unwrap_or(usize::MAX));
    let (orphans, widows) = (count(props.table_orphans), count(props.table_widows));
    (orphans.is_some() || widows.is_some()).then(|| LineLimits {
        orphans: orphans.unwrap_or(1),
        widows: widows.unwrap_or(1),
    })
}

/// The line a fragment of an inline root breaks before, when lines
/// `[start, fit)` of its `total` fit the page: after the last that fits,
/// pulled back so the remainder keeps `limits.widows` lines, or `None`
/// when that leaves the fragment fewer than `limits.orphans` lines or none
/// (CSS Fragmentation §4.4). [`fragment_inline_root`] splits by it, and
/// [`inline_leading`] asks it, so the two cannot disagree.
fn split_point(fit: usize, start: usize, total: usize, limits: LineLimits) -> Option<usize> {
    let split = fit.min(total.saturating_sub(limits.widows));
    (split > start && split >= start + limits.orphans).then_some(split)
}

/// fulgur-p55h: split a multi-line inline root across page boundaries
/// at line edges, append one Fragment per page span to the geometry
/// table, and return the number of fragments emitted. The paragraph starts
/// at `cursor`, which ends below its last line.
///
/// Walks lines, tracks the first line of the current fragment in
/// `fragment_start_idx`, and breaks when a line's bottom, in
/// paragraph-local coords, would pass `page_height_px`. A fragment is
/// only ever closed after lines that fit, so none extends past the page
/// strip (CSS Fragmentation §3.3); the one exception is a single line
/// taller than an empty page, which is emitted alone so the walk makes
/// progress.
///
/// fulgur-s67g Phase 2.1 (widow / orphan): the break goes after the last
/// line that fits, pulled back so the remainder keeps `limits.widows` lines
/// (CSS Fragmentation §4.4). If that leaves fewer than `limits.orphans`
/// lines in the fragment:
///
/// - when the fragment starts below the top of a page, the whole
///   fragment moves to the next page and is evaluated again there. This
///   is also what happens when not even its first line fits;
/// - at the top of a page moving gains nothing, so the minimums are
///   relaxed and the break goes after the lines that fit.
///
/// `limits` are the paragraph's `orphans` and `widows` ([`line_limits`]).
/// `kept_after` is what its last line keeps below it on its page, its
/// bottom edge and the boxes an `avoid` after it joins it to: the last line
/// fits only with that room left, so the paragraph splits before its last
/// `widows` lines rather than leave them parted from the run.
///
/// Once the next page would be `MAX_PAGES` the walk stops breaking and
/// the rest of the paragraph accumulates into the final fragment.
///
/// Output:
///
/// - On a single-page paragraph (no overflow), one Fragment is appended
///   covering all lines. `cursor_y` advances by the paragraph's natural
///   height.
/// - On a multi-page paragraph, one Fragment per page is appended. The
///   final `cursor_y` is the height consumed on the last page (lines
///   ending on a partial page leave room for a following sibling).
/// - `line_boundaries` records the first line of each fragment and ends
///   with the total line count.
///
/// The renderer paints each fragment's lines `inset` below its top, below
/// the root's top border and padding ([`top_inset`]), so a line fits where
/// it ends on the page that much lower.
#[allow(clippy::too_many_arguments)]
fn fragment_inline_root(
    geometry: &mut PaginationGeometryTable,
    child_id: usize,
    paragraph_x: f32,
    width: f32,
    cursor: &mut PageCursor,
    line_metrics: &[(f32, f32)],
    limits: LineLimits,
    inset: f32,
    kept_after: f32,
) -> usize {
    if line_metrics.is_empty() {
        return 0;
    }

    // The cursor's `cursor_y` is the top of the fragment being built.
    let page_height_px = cursor.page_height_px;
    let mut fragment_start_idx: usize = 0;
    let mut emitted = 0usize;
    let total_lines = line_metrics.len();

    let mut i = 0usize;
    while i < total_lines {
        let frag_top_local = line_metrics[fragment_start_idx].0;
        let projected_bottom_in_body =
            cursor.cursor_y + inset + (line_metrics[i].1 - frag_top_local);

        // codex P1: also refuse to break once the next fragment would land
        // on `MAX_PAGES` — the same bound the oversized-block re-slice loop
        // and the absolute-positioning path enforce. Without this, an
        // attacker-controlled narrow `word-break: break-all` paragraph with
        // pathologically many lines has no cap here, and `page_index` —
        // which feeds `implied_page_count`, driving per-page allocation and
        // rendering — can grow unbounded with the line count instead of
        // being clamped.
        let room = if i + 1 == total_lines {
            page_height_px - kept_after
        } else {
            page_height_px
        };
        if projected_bottom_in_body <= room || cursor.page_index + 1 >= crate::MAX_PAGES {
            i += 1;
            continue;
        }

        let at_page_top = cursor.at_page_top();
        if i == fragment_start_idx {
            // Not even the first line of this fragment fits.
            if at_page_top {
                // A line taller than an empty page: nothing to gain by
                // moving, so keep it and let the next line close the
                // fragment.
                i += 1;
            } else {
                cursor.advance();
            }
            continue;
        }

        // Lines [fragment_start_idx, i) fit. Keep `widows` lines for the
        // remainder.
        let split = if let Some(split) = split_point(i, fragment_start_idx, total_lines, limits) {
            split
        } else if at_page_top {
            // css-break-3 §4.4: the minimums cannot be met and moving gains
            // nothing, so relax them and split after the lines that fit.
            i
        } else {
            // Break before the fragment instead, and evaluate its lines
            // again on the fresh page.
            cursor.advance();
            continue;
        };

        let frag_h = line_metrics[split - 1].1 - frag_top_local;
        let frag = Fragment {
            page_index: cursor.page_index,
            x: paragraph_x.as_px(),
            y: cursor.cursor_y.as_px(),
            width: width.as_px(),
            height: frag_h.as_px(),
        };
        let entry = geometry.entry(child_id).or_default();
        entry.fragments.push(frag);
        // Publish the partition rather than leaving the consumer to
        // rediscover it from fragment heights — see
        // `PaginationGeometry::line_boundaries`.
        entry.line_boundaries.push(fragment_start_idx);
        emitted += 1;
        cursor.place(frag_h);

        cursor.advance();
        fragment_start_idx = split;
        // A pulled-back split puts lines the loop already passed into the
        // new fragment, so measure from its first line.
        i = split;
    }

    // Final fragment covers lines [fragment_start_idx, end).
    let frag_top_local = line_metrics[fragment_start_idx].0;
    let last_bottom_local = line_metrics.last().expect("non-empty checked above").1;
    let frag_h = last_bottom_local - frag_top_local;
    let frag = Fragment {
        page_index: cursor.page_index,
        x: paragraph_x.as_px(),
        y: cursor.cursor_y.as_px(),
        width: width.as_px(),
        height: frag_h.as_px(),
    };
    let entry = geometry.entry(child_id).or_default();
    entry.fragments.push(frag);
    entry.line_boundaries.push(fragment_start_idx);
    // Terminator: `fragments[k]` covers
    // `line_boundaries[k]..line_boundaries[k + 1]`, so the vector is one
    // longer than the fragment list and its last element is the total
    // line count. A consumer can check that last element against the
    // line vector it holds to prove both sides mean the same lines.
    entry.line_boundaries.push(total_lines);
    emitted += 1;

    cursor.cursor_y += frag_h;
    cursor.place(frag_h);
    emitted
}

/// Record geometry for the atomic inlines (`<img>`, `inline-block`) and
/// inline elements under a multi-line inline root, after
/// [`fragment_inline_root`] split it into `frag_count` fragments.
///
/// A single-line inline root takes the whole-block path, whose
/// [`record_subtree_descendants`] gives every descendant a fragment. The
/// line-aware path emitted fragments for the root only, so an `<img>` inside
/// a paragraph of several lines had no geometry and the renderer skipped it.
/// Each descendant lands on the fragment whose lines hold its vertical
/// centre, at its offset from that fragment's first line.
fn record_inline_root_descendants(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    root_id: usize,
    line_metrics: &[(f32, f32)],
    frag_count: usize,
) {
    let Some(root_frags) = geometry.get(&root_id).map(|g| {
        let from = g.fragments.len().saturating_sub(frag_count);
        g.fragments[from..].to_vec()
    }) else {
        return;
    };
    // Paragraph-local `(top, bottom)` of the lines each fragment holds.
    let mut spans: Vec<(f32, f32)> = Vec::with_capacity(root_frags.len());
    let mut next_line = 0usize;
    for frag in &root_frags {
        let Some(&(top, _)) = line_metrics.get(next_line) else {
            break;
        };
        let bottom = top + frag.height.to_f32();
        spans.push((top, bottom));
        next_line += line_metrics[next_line..]
            .iter()
            .take_while(|l| l.0 < bottom - 0.01)
            .count()
            .max(1);
    }
    if spans.is_empty() {
        return;
    }
    walk_inline_root_descendants(geometry, doc, root_id, 0.0, 0.0, &root_frags, &spans, 0);
}

#[allow(clippy::too_many_arguments)]
fn walk_inline_root_descendants(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    parent_id: usize,
    parent_local_x: f32,
    parent_local_y: f32,
    root_frags: &[Fragment],
    spans: &[(f32, f32)],
    depth: usize,
) {
    if depth >= crate::MAX_DOM_DEPTH {
        return;
    }
    let Some(parent) = doc.get_node(parent_id) else {
        return;
    };
    let layout_children_borrow = parent.layout_children.borrow();
    let walk_children: &[usize] = layout_children_borrow
        .as_deref()
        .filter(|v| !v.is_empty())
        .unwrap_or(&parent.children);
    for &child_id in walk_children {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        let layout = child.final_layout;
        let (w, h) = (layout.size.width, layout.size.height);
        let local_x = parent_local_x + layout.location.x;
        let local_y = parent_local_y + layout.location.y;
        if h <= 0.0 && w <= 0.0 {
            walk_inline_root_descendants(
                geometry,
                doc,
                child_id,
                local_x,
                local_y,
                root_frags,
                spans,
                depth + 1,
            );
            continue;
        }
        let centre = local_y + h / 2.0;
        let k = spans
            .iter()
            .position(|&(_, bottom)| centre < bottom)
            .unwrap_or(spans.len() - 1);
        let frag = &root_frags[k];
        let page_x = frag.x.to_f32() + local_x;
        let page_y = frag.y.to_f32() + (local_y - spans[k].0);
        geometry
            .entry(child_id)
            .or_default()
            .fragments
            .push(Fragment {
                page_index: frag.page_index,
                x: page_x.as_px(),
                y: page_y.as_px(),
                width: w.as_px(),
                height: h.as_px(),
            });
        record_subtree_descendants(
            geometry,
            doc,
            child_id,
            frag.page_index,
            page_y,
            page_x,
            depth + 1,
        );
    }
}

/// Per-page state for a named string emitted by `string-set:`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StringSetPageState {
    /// Value at start of page (carried from previous page's `last`).
    pub start: Option<String>,
    /// First value set on this page.
    pub first: Option<String>,
    /// Last value set on this page.
    pub last: Option<String>,
}

/// Per-page state for running element instances of a given name.
#[derive(Debug, Clone, Default)]
pub struct PageRunningState {
    /// Instance IDs of running elements whose source position falls on
    /// this page, in source order.
    pub instance_ids: Vec<usize>,
}

/// fulgur-6tco: walk the geometry table page-by-page to thread
/// `string-set` state across pages.
///
/// For each page index 0..max_page:
///
/// 1. Initialise per-name `start` from the previous page's `last`
///    (the carry).
/// 2. For each node id whose **first** fragment lands on this page,
///    apply its `(name, value)` markers in NodeId order — records
///    `first` (only set once per page per name) and updates `last`
///    plus the carry for subsequent pages.
///
/// Markers fire only on a node's first appearance: when an inline
/// root spans two pages, its second-page fragment does **not** re-
/// emit the marker.
///
/// Source-order assumption: `geometry` is a `BTreeMap<usize, ..>` so
/// iteration is by ascending NodeId. For body's direct children that
/// matches DOM source order, since Blitz allocates ids sequentially
/// during parse. Nested string-set declarations (markers attached to
/// a `<span>` inside a `<p>`) are not in the fragmenter's geometry
/// table today and so are silently dropped — same scope limitation as
/// `fragment_pagination_root` itself.
pub fn collect_string_set_states(
    geometry: &PaginationGeometryTable,
    string_set_by_node: &BTreeMap<usize, Vec<(String, String)>>,
) -> Vec<BTreeMap<String, StringSetPageState>> {
    let max_page = geometry
        .values()
        .flat_map(|g| g.fragments.iter())
        .map(|f| f.page_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    // For each page, the list of nodes whose first fragment lands
    // there, in NodeId (≈ source) order.
    let mut nodes_per_page: Vec<Vec<usize>> = vec![Vec::new(); max_page as usize];
    for (&node_id, geom) in geometry {
        if let Some(first_frag) = geom.fragments.first()
            && (first_frag.page_index as usize) < nodes_per_page.len()
        {
            nodes_per_page[first_frag.page_index as usize].push(node_id);
        }
    }

    let mut result: Vec<BTreeMap<String, StringSetPageState>> =
        Vec::with_capacity(nodes_per_page.len());
    let mut carry: BTreeMap<String, String> = BTreeMap::new();

    for nodes in &nodes_per_page {
        let mut page_state: BTreeMap<String, StringSetPageState> = BTreeMap::new();
        for (name, value) in &carry {
            page_state.entry(name.clone()).or_default().start = Some(value.clone());
        }
        for node_id in nodes {
            let Some(entries) = string_set_by_node.get(node_id) else {
                continue;
            };
            for (name, value) in entries {
                let state = page_state.entry(name.clone()).or_default();
                if state.first.is_none() {
                    state.first = Some(value.clone());
                }
                state.last = Some(value.clone());
                carry.insert(name.clone(), value.clone());
            }
        }
        result.push(page_state);
    }

    result
}

/// Walk the geometry table page-by-page and emit the running element
/// instances whose first fragment lands on each page.
///
/// Each `instance_id` is adopted only once — on the page where its
/// node's first fragment lands. This matches the source-order policy
/// the margin-box renderer uses with `resolve_element_policy` to pick
/// the right instance for `first` / `last` / `first-except`.
pub fn collect_running_element_states(
    geometry: &PaginationGeometryTable,
    running_store: &crate::gcpm::running::RunningElementStore,
) -> Vec<BTreeMap<String, PageRunningState>> {
    let max_page = geometry
        .values()
        .flat_map(|g| g.fragments.iter())
        .map(|f| f.page_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    let mut result: Vec<BTreeMap<String, PageRunningState>> =
        vec![BTreeMap::new(); max_page as usize];

    for (&node_id, geom) in geometry {
        let Some(first_frag) = geom.fragments.first() else {
            continue;
        };
        let page_idx = first_frag.page_index as usize;
        if page_idx >= result.len() {
            continue;
        }
        let Some(instance_id) = running_store.instance_for_node(node_id) else {
            continue;
        };
        let Some(name) = running_store.name_of(instance_id) else {
            continue;
        };
        result[page_idx]
            .entry(name.to_string())
            .or_default()
            .instance_ids
            .push(instance_id);
    }

    result
}

/// fulgur-s67g Phase 2.3: walk the geometry table page-by-page and
/// replay counter operations in document order, returning the
/// cumulative counter snapshot at the end of each page.
///
/// Each page first applies the page context (css-page-3, page-based
/// counters): the
/// `page_ops` of the `@page` rules (resets, then increments, then
/// sets) and the implicit `counter-increment: page 1` unless `page_ops`
/// increment `page` themselves. Then the elements whose first fragment
/// lands on the page apply theirs, so an element's `counter-reset: page
/// N` or `counter-set: page N` numbers the page it starts on and the
/// pages after it count on from there. The `page` entry of each
/// snapshot is the number that page's margin boxes print.
///
/// Same source-order assumption as
/// [`collect_string_set_states`]: the per-node counter ops are
/// applied in the order they appear in the body's children list,
/// approximated by `BTreeMap<NodeId, _>` iteration. Nested counter
/// declarations on descendants of body's direct children are not in
/// the fragmenter's geometry today and are silently dropped — same
/// scope limitation as `fragment_pagination_root` itself.
pub fn collect_counter_states(
    geometry: &PaginationGeometryTable,
    counter_ops_by_node: &BTreeMap<usize, Vec<crate::gcpm::CounterOp>>,
    page_ops: &[crate::gcpm::CounterOp],
) -> Vec<BTreeMap<String, i32>> {
    use crate::gcpm::CounterOp;
    use crate::gcpm::counter::CounterState;

    let max_page = geometry
        .values()
        .flat_map(|g| g.fragments.iter())
        .map(|f| f.page_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    // For each page, the list of nodes whose first fragment lands
    // there, in NodeId (≈ source) order.
    let mut nodes_per_page: Vec<Vec<usize>> = vec![Vec::new(); max_page as usize];
    for (&node_id, geom) in geometry {
        if let Some(first_frag) = geom.fragments.first()
            && (first_frag.page_index as usize) < nodes_per_page.len()
        {
            nodes_per_page[first_frag.page_index as usize].push(node_id);
        }
    }

    let increments_page = page_ops
        .iter()
        .any(|op| matches!(op, CounterOp::Increment { name, .. } if name == "page"));
    let mut state = CounterState::new();
    let mut result: Vec<BTreeMap<String, i32>> = Vec::with_capacity(nodes_per_page.len());

    for nodes in &nodes_per_page {
        for op in page_ops {
            if let CounterOp::Reset { name, value } = op {
                state.reset(name, *value);
            }
        }
        if !increments_page {
            state.increment("page", 1);
        }
        for op in page_ops {
            if let CounterOp::Increment { name, value } = op {
                state.increment(name, *value);
            }
        }
        for op in page_ops {
            if let CounterOp::Set { name, value } = op {
                state.set(name, *value);
            }
        }
        for node_id in nodes {
            let Some(ops) = counter_ops_by_node.get(node_id) else {
                continue;
            };
            for op in ops {
                match op {
                    CounterOp::Reset { name, value } => state.reset(name, *value),
                    CounterOp::Increment { name, value } => state.increment(name, *value),
                    CounterOp::Set { name, value } => state.set(name, *value),
                }
            }
        }
        result.push(state.snapshot());
    }

    result
}

/// What a margin box's `@page` selector reads of one page, besides its
/// number (`gcpm::margin_box::page_selector_specificity`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageFacts {
    /// The page name (css-page-3 §5.3), which selects `@page <name>` rules.
    pub name: Option<String>,
    /// No in-flow fragment lands on the page: it is a page a forced side
    /// break (`break-before: right`) leaves empty, which `@page :blank`
    /// selects.
    pub blank: bool,
}

/// The [`PageFacts`] of each page. A page's name is the start page name
/// (`used_page_names`, from `blitz_adapter::compute_used_page_names`) of
/// the first node, in NodeId (≈ source) order, whose first fragment lands
/// on it. A page where no node starts keeps the name of the page before
/// it: it continues that page's content, or it is a blank page a side
/// break leaves, which counts with the pages before it (see
/// [`collect_counter_states`]); it does not take the name of the page the
/// break leads to. A page is blank when no fragment lands on it at all.
/// Repeated fragments (`position: fixed`, a repeated table header) start
/// nothing and leave no page blank.
pub fn collect_page_facts(
    geometry: &PaginationGeometryTable,
    used_page_names: &crate::blitz_adapter::UsedPageNameTable,
) -> Vec<PageFacts> {
    let pages = implied_page_count(geometry) as usize;
    let mut started: Vec<Option<Option<String>>> = vec![None; pages];
    let mut blank = vec![true; pages];
    for (node_id, geom) in geometry {
        if geom.is_repeat {
            continue;
        }
        for fragment in &geom.fragments {
            if let Some(page) = blank.get_mut(fragment.page_index as usize) {
                *page = false;
            }
        }
        if let Some(first) = geom.fragments.first()
            && let Some(slot) = started.get_mut(first.page_index as usize)
            && slot.is_none()
            && let Some((start, _)) = used_page_names.get(node_id)
        {
            *slot = Some(start.clone());
        }
    }
    let mut carry = None;
    started
        .into_iter()
        .zip(blank)
        .map(|(start, blank)| {
            if let Some(name) = start {
                carry = name;
            }
            PageFacts {
                name: carry.clone(),
                blank,
            }
        })
        .collect()
}

/// fulgur-jkl5: enumerate `position: fixed` elements and emit one
/// fragment per page so downstream rendering can repeat them on every
/// page (Chrome-compatible behaviour for paged media — see WPT
/// fixedpos-* family).
///
/// fulgur-rpvu: wired into the v2 production path. v2's
/// `dispatch_fragment` loop iterates `Fragment`s per (node_id, page),
/// so emitting one Fragment per page for each `position: fixed`
/// element produces the expected per-page repetition naturally. The
/// resulting `PaginationGeometry.is_repeat` is set to `true` so
/// consumers know each fragment carries the *full* content rather
/// than a slice (paragraph-line / block-height slicing must be
/// suppressed for repeat fragments). Both fixed-element paths
/// (v1 and v2) produce equivalent output.
///
/// `total_pages` is the document's resolved page count, typically
/// computed from `PaginationGeometryTable`'s max `page_index + 1` after
/// `run_pass*` has run. `0` is normalised to `1` so even an empty
/// document gets a valid fragment for any fixed element on it.
///
/// The fragment's `(x, y, width, height)` come from each fixed
/// element's existing `final_layout` — same coordinate frame the
/// non-paginated convert path already uses. **This function relies on
/// `blitz_adapter::relayout_position_fixed` (added in fulgur-tbxs,
/// branch `feat/fixedpos-viewport-cb`) having run beforehand** so
/// that `final_layout` reflects viewport-CB resolution rather than
/// the inherited (often wrong) abs-position layout. The fragmenter branch
/// does not yet include `relayout_position_fixed`; once both land on
/// `main` this function picks up the corrected positions automatically.
///
/// The emitted fragments are appended to `geometry` (typically the
/// table returned by `run_pass`) so a single side-table carries both
/// the body-fragmentation geometry and the fixed-element repetition.
/// Convert-side consumers (`convert::positioned.rs`) iterate the
/// node's `Vec<Fragment>` to place one copy of the element per page.
pub fn append_position_fixed_fragments(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    total_pages: u32,
    viewport_w_px: f32,
    viewport_h_px: f32,
) {
    use ::style::properties::longhands::position::computed_value::T as Pos;

    let pages = total_pages.max(1);
    let body_offset_xy = body_origin_in_px(doc);
    let mut fixed_ids: Vec<usize> = Vec::new();
    let root_id = doc.root_element().id;
    walk_for_position_fixed(doc, root_id, &mut fixed_ids, 0);

    // Aggregate budget across all fixed roots and their descendants: each
    // node emits one fragment per page, so the retained total is
    // O((roots + descendants) × pages). `pages` is bounded by MAX_PAGES, but
    // the node axis is not — cap the total to keep the pass bounded on
    // untrusted input (`crate::MAX_SUBTREE_PAGE_FRAGMENTS`).
    let mut emitted: usize = 0;

    for id in fixed_ids {
        let Some(node) = doc.get_node(id) else {
            continue;
        };
        // Re-check style here even though `walk_for_position_fixed`
        // already filtered — guards against nodes whose style was
        // mutated between the walk and this read (defensive only,
        // single-threaded code path).
        let is_fixed = node
            .primary_styles()
            .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Fixed));
        if !is_fixed {
            continue;
        }
        let layout = node.final_layout;
        let (w, h) = (layout.size.width, layout.size.height);
        // fulgur-a8m5: Taffy's `compute_root_layout` (used by
        // `relayout_position_fixed`) does not resolve `bottom` / `right`
        // insets when the absolute element is the root of a layout
        // subtree — it places the element at (0, 0) regardless. The
        // CSS 2.1 §9.4 viewport CB needs explicit inset resolution
        // here, otherwise WPT fixedpos-001/002/008 render `bottom: 0`
        // fixed elements at the top of every page.
        let (resolved_x, resolved_y) =
            resolve_viewport_cb_location(node, w, h, viewport_w_px, viewport_h_px)
                .unwrap_or((layout.location.x, layout.location.y));
        // Render adds `body_offset_pt.y` to every fragment's y to
        // account for the html→body offset (collapsed margins from
        // in-flow body-direct children, etc.). Fixed elements are
        // viewport-anchored, not body-anchored, so subtract that
        // offset here so the dispatch path produces a viewport-
        // relative y in PDF coordinates. Without this compensation,
        // documents that mix in-flow content with `position: fixed`
        // (WPT fixedpos-008) shift the fixed text by the in-flow
        // div's margin-top.
        let (x, y) = (resolved_x - body_offset_xy.0, resolved_y - body_offset_xy.1);

        let entry = geometry.entry(id).or_default();
        // Replace any prior placements (e.g. if the fixed element was
        // also walked by `fragment_pagination_root` and emitted as a
        // single fragment). Per-page repetition is the canonical
        // representation for fixed content.
        entry.fragments.clear();
        entry.is_repeat = true;
        for page_index in 0..pages {
            if emitted >= crate::MAX_SUBTREE_PAGE_FRAGMENTS {
                break;
            }
            entry.fragments.push(Fragment {
                page_index,
                x: x.as_px(),
                y: y.as_px(),
                width: w.as_px(),
                height: h.as_px(),
            });
            emitted += 1;
        }

        // fulgur-4m16: emit per-page repeated fragments for every
        // in-flow descendant of the fixed root. v2 dispatch is
        // geometry-driven and reads fragments per `node_id`, so a
        // fixed root with a sized block-element child (e.g. WPT
        // fixedpos-009 `<div style="position:fixed; bottom:0; right:0">
        // <div class="pencil" style="width:36px; height:36px;
        // background:black; mask-image:..."></div></div>`) needs an
        // entry for the pencil child or v2 never reaches it. The
        // existing root-only fragment carries inline text rendering
        // (fixedpos-001 / 008 ref pattern) but not block descendants.
        // Always walk the subtree (even once the budget is spent): the walk
        // clears every descendant's entry, which is required because this
        // fixed pass runs a second time after the absolute pass and must not
        // leave stale fragments from the first pass behind (Codex review). The
        // per-node push is itself bounded by the `emitted` budget.
        record_fixed_subtree_descendants(geometry, doc, id, (x, y), pages, &mut emitted);
    }

    // Don't allocate empty entries for nodes without fragments.
    geometry.retain(|_, geom| !geom.fragments.is_empty());
}

/// fulgur-4m16: walk every in-flow descendant of a `position: fixed`
/// root and emit one fragment per page (`is_repeat = true`) at the
/// descendant's offset within the fixed subtree, anchored to the
/// root's already-resolved viewport-CB position.
///
/// Mirrors [`record_subtree_fragments_at_offset`] (used by
/// `append_position_absolute_body_direct_fragments`) except:
///   - fragments repeat on every page instead of landing on a single
///     y-derived page,
///   - the caller passes the root's body-relative stored (x, y) so we
///     don't re-resolve viewport-CB here (the root's `final_layout`
///     can lack end-side inset resolution — see fulgur-a8m5),
///   - body-offset compensation already happened on the caller's `(x, y)`,
///     so descendants just add their subtree offset on top.
///
/// Skips out-of-flow descendants (handled by their own pass) and
/// whitespace-only text nodes (matches fragmenter behavior).
fn record_fixed_subtree_descendants(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    fixed_root_id: usize,
    root_stored_xy: (f32, f32),
    pages: u32,
    emitted: &mut usize,
) {
    use ::style::properties::longhands::position::computed_value::T as Pos;

    #[allow(clippy::too_many_arguments)]
    fn walk(
        geometry: &mut PaginationGeometryTable,
        doc: &BaseDocument,
        node_id: usize,
        offset_in_subtree: (f32, f32),
        root_stored_xy: (f32, f32),
        pages: u32,
        depth: usize,
        emitted: &mut usize,
    ) {
        if depth >= crate::MAX_DOM_DEPTH {
            return;
        }
        let Some(node) = doc.get_node(node_id) else {
            return;
        };
        let stored_x = root_stored_xy.0 + offset_in_subtree.0;
        let stored_y = root_stored_xy.1 + offset_in_subtree.1;
        let w = node.final_layout.size.width;
        let h = node.final_layout.size.height;

        let entry = geometry.entry(node_id).or_default();
        entry.fragments.clear();
        entry.is_repeat = true;
        // Per-page repeat fragments, bounded by the aggregate `emitted` budget
        // (the hard cap on O(descendants × pages)). Use `break`, not `return`,
        // and do not skip zero-area nodes: the walk must visit and clear every
        // descendant (the fixed pass runs twice — before and after the absolute
        // pass — so a later node can hold stale fragments from the first pass),
        // and a zero-size wrapper's fragment can still scope a visible child via
        // transform / opacity dispatch, which reads the wrapper's geometry
        // (Codex review).
        for page_index in 0..pages.max(1) {
            if *emitted >= crate::MAX_SUBTREE_PAGE_FRAGMENTS {
                break;
            }
            entry.fragments.push(Fragment {
                page_index,
                x: stored_x.as_px(),
                y: stored_y.as_px(),
                width: w.as_px(),
                height: h.as_px(),
            });
            *emitted += 1;
        }

        let children: Vec<usize> = {
            let layout_borrow = node.layout_children.borrow();
            if let Some(lc) = layout_borrow.as_deref()
                && !lc.is_empty()
            {
                lc.to_vec()
            } else {
                node.children.clone()
            }
        };
        for child_id in children {
            let Some(child) = doc.get_node(child_id) else {
                continue;
            };
            let is_oof = child.primary_styles().is_some_and(|s| {
                matches!(s.get_box().clone_position(), Pos::Absolute | Pos::Fixed)
            });
            if is_oof {
                continue;
            }
            if crate::blitz_adapter::is_whitespace_only_text_node(child) {
                continue;
            }
            let child_offset = (
                offset_in_subtree.0 + child.final_layout.location.x,
                offset_in_subtree.1 + child.final_layout.location.y,
            );
            walk(
                geometry,
                doc,
                child_id,
                child_offset,
                root_stored_xy,
                pages,
                depth + 1,
                emitted,
            );
        }
    }

    let Some(root) = doc.get_node(fixed_root_id) else {
        return;
    };
    let children: Vec<usize> = {
        let layout_borrow = root.layout_children.borrow();
        if let Some(lc) = layout_borrow.as_deref()
            && !lc.is_empty()
        {
            lc.to_vec()
        } else {
            root.children.clone()
        }
    };
    for child_id in children {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        let is_oof = child
            .primary_styles()
            .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Absolute | Pos::Fixed));
        if is_oof {
            continue;
        }
        if crate::blitz_adapter::is_whitespace_only_text_node(child) {
            continue;
        }
        let child_offset = (child.final_layout.location.x, child.final_layout.location.y);
        walk(
            geometry,
            doc,
            child_id,
            child_offset,
            root_stored_xy,
            pages,
            1,
            emitted,
        );
    }
}

/// fulgur-a8m5: emit a Fragment for every body-direct
/// `position: absolute` element whose effective containing block falls
/// back to the viewport (when body's box collapses to zero because all
/// of its children are out-of-flow — see CSS 2.1 §10.1.5 and the
/// matching `viewport_size_px` body-zero fallback in
/// `convert::positioned::resolve_cb_for_absolute`).
///
/// `fragment_pagination_root` skips out-of-flow children unconditionally,
/// so without this pass `<body><div style="position:absolute; bottom:0">…</div></body>`
/// never reaches `pagination_geometry` and the v2 dispatch loop drops
/// the element entirely (WPT `fixedpos-00{1,2,8}` ref-side breakage).
///
/// Each visited in-flow node emits fragments for every page intersected
/// by its resolved y range; off-page elements (e.g. `bottom: -100vh` in
/// a single-page document) are dropped because no page can paint them.
pub fn append_position_absolute_body_direct_fragments(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    total_pages: u32,
    viewport_w_px: f32,
    viewport_h_px: f32,
    running_store: Option<&crate::gcpm::running::RunningElementStore>,
) {
    use ::style::properties::longhands::position::computed_value::T as Pos;

    let pages = total_pages.max(1);
    let body_id = match find_body_id(doc) {
        Some(id) => id,
        None => return,
    };
    let body = match doc.get_node(body_id) {
        Some(n) => n,
        None => return,
    };
    // Per CSS 2.1 §10.1.5, the containing block for `position: absolute`
    // children of `<body>` (a static-position element) falls through to
    // the initial containing block (the viewport) regardless of body's
    // own size. The fragmenter unconditionally skips out-of-flow
    // children (`fragment_pagination_root` `continue` for `Pos::Absolute`),
    // so this pass runs for every body-direct abs — not just when body
    // collapses to zero.
    let body_offset_xy = body_origin_in_px(doc);

    // Aggregate budget across all body-direct absolute subtrees: each node is
    // recorded once per intersected page, so many page-spanning absolutes are
    // O(nodes × pages). Cap the retained total (shared with the fixed pass via
    // `crate::MAX_SUBTREE_PAGE_FRAGMENTS`).
    let mut emitted: usize = 0;

    let body_children = body.children.clone();
    let body_has_in_flow_content = body_children.iter().any(|&child_id| {
        let Some(child) = doc.get_node(child_id) else {
            return false;
        };
        if crate::blitz_adapter::is_whitespace_only_text_node(child) {
            return false;
        }
        if running_store.is_some_and(|s| s.instance_for_node(child_id).is_some()) {
            return false;
        }
        !is_out_of_flow_positioned(child) && !crate::blitz_adapter::node_is_floating(child)
    });
    for child_id in body_children {
        let Some(child) = doc.get_node(child_id) else {
            continue;
        };
        let is_abs_only = child
            .primary_styles()
            .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Absolute));
        if !is_abs_only {
            continue;
        }
        let layout = child.final_layout;
        let (w, h) = (layout.size.width, layout.size.height);
        let (resolved_x, resolved_y) =
            resolve_viewport_cb_location(child, w, h, viewport_w_px, viewport_h_px)
                .unwrap_or((layout.location.x, layout.location.y));
        let page_stride_px = if uses_bottom_without_top(child) {
            viewport_h_px.round()
        } else {
            viewport_h_px
        };

        // Walk the subtree and emit fragments for every page each
        // in-flow node intersects (block, paragraph, anonymous wrapper).
        // Without this the dispatch loop drops descendants whose node_id
        // is not the root abs id (anonymous block wrappers around mixed
        // text/element content — fixedpos-002 / fixedpos-008 ref-side
        // pattern). Out-of-flow descendants are skipped because they
        // are handled by their own pass. The render path adds
        // `body_offset_pt` to every emitted fragment's y, so we pass
        // the body offset down and let the walker subtract it from the
        // stored y while keeping page assignment based on the un-
        // compensated viewport-anchored y.
        record_subtree_fragments_at_offset(
            geometry,
            doc,
            child_id,
            (resolved_x, resolved_y),
            body_offset_xy,
            viewport_h_px,
            page_stride_px,
            pages,
            !body_has_in_flow_content,
            &mut emitted,
            running_store,
        );
    }

    geometry.retain(|_, geom| !geom.fragments.is_empty());
}

/// Walk a body-direct out-of-flow subtree and emit Fragments on every
/// intersected page for each in-flow node (the subtree root + every
/// block / paragraph / anonymous wrapper inside it). Each fragment's
/// body-relative location = the subtree root's resolved viewport-CB
/// location plus the node's accumulated `final_layout.location` offset
/// from that root. A size-contained box taller than a page is sliced into
/// one fragment a page where block flow slices it, and is otherwise kept on
/// its first page ([`is_monolithic_in_absolute_subtree`]). The pages its
/// slices reach join the budget of the whole subtree, past the flow's.
///
/// Skips out-of-flow descendants (their own pass handles them) and
/// whitespace-only text nodes (mirrors `fragment_pagination_root`).
#[allow(clippy::too_many_arguments)]
fn record_subtree_fragments_at_offset(
    geometry: &mut PaginationGeometryTable,
    doc: &BaseDocument,
    subtree_root_id: usize,
    root_xy_for_paging: (f32, f32),
    body_offset: (f32, f32),
    page_h_px: f32,
    page_stride_px: f32,
    total_pages: u32,
    may_extend_pages: bool,
    emitted: &mut usize,
    running_store: Option<&crate::gcpm::running::RunningElementStore>,
) {
    #[allow(clippy::too_many_arguments)]
    fn walk(
        geometry: &mut PaginationGeometryTable,
        doc: &BaseDocument,
        node_id: usize,
        offset_in_subtree: (f32, f32),
        root_xy_for_paging: (f32, f32),
        body_offset: (f32, f32),
        page_h_px: f32,
        page_stride_px: f32,
        total_pages: u32,
        may_extend_pages: bool,
        // fulgur-xa9q: true once we are at or below an overflow-clipping
        // ancestor (`clips_overflow`: any non-`visible` overflow). Inside such
        // a subtree the "START beyond budget extends" exception is suppressed
        // for descendants, so clipped overflow cannot generate pages
        // (page-background-003 cat box, which is `overflow:clip`). `contain:
        // size` alone is NOT a clip and does not set this boundary.
        containment_boundary: bool,
        // Subtree-offset and border-box size of the nearest positioned
        // ancestor of `node_id` — the containing block that `node_id`'s
        // out-of-flow children resolve their explicit insets against (CSS
        // 2.1 §10.1.4). The body-direct abs at the subtree root is itself
        // positioned, so it overrides these on the first frame.
        cb_anchor: (f32, f32),
        cb_size: (f32, f32),
        depth: usize,
        emitted: &mut usize,
        sliced: &mut SlicedBoxes,
        running_store: Option<&crate::gcpm::running::RunningElementStore>,
    ) {
        if depth >= crate::MAX_DOM_DEPTH {
            return;
        }
        let Some(node) = doc.get_node(node_id) else {
            return;
        };
        // Prefer `layout_children` so anonymous block wrappers Stylo
        // synthesizes around mixed inline/block content are visited
        // (CSS 2.1 §9.2.1.1 — see `fragment_pagination_root` for the
        // same idiom).
        let children: Vec<usize> = {
            let layout_borrow = node.layout_children.borrow();
            if let Some(lc) = layout_borrow.as_deref()
                && !lc.is_empty()
            {
                lc.to_vec()
            } else {
                node.children.clone()
            }
        };
        // The containing block for THIS node's out-of-flow children is
        // `node` itself when `node` is positioned (non-static), otherwise the
        // inherited nearest-positioned ancestor (`cb_anchor`/`cb_size`). A
        // `position: static` element does not establish a CB, so an abs child
        // beneath it anchors to the same ancestor its static parent does.
        //
        // The CB is the positioned ancestor's *padding box* (CSS 2.1
        // §10.1.4): the anchor backs in by its top/left border and the size
        // drops both borders. `offset_in_subtree` is `node`'s border-box
        // origin, so add the border to reach the padding edge.
        let node_positioned = {
            use ::style::properties::longhands::position::computed_value::T as Pos;
            node.primary_styles()
                .is_some_and(|s| !matches!(s.get_box().clone_position(), Pos::Static))
        };
        let (child_cb_anchor, child_cb_size) = if node_positioned {
            let border = node.final_layout.border;
            (
                (
                    offset_in_subtree.0 + border.left,
                    offset_in_subtree.1 + border.top,
                ),
                (
                    node.final_layout.size.width - border.left - border.right,
                    node.final_layout.size.height - border.top - border.bottom,
                ),
            )
        } else {
            (cb_anchor, cb_size)
        };
        // Page assignment is based on the un-compensated viewport-CB
        // resolved position (the actual paint location). Storage is
        // body-relative because the dispatch path adds `body_offset_pt`
        // back at draw time.
        let final_y_for_paging = root_xy_for_paging.1 + offset_in_subtree.1;
        let stored_x = root_xy_for_paging.0 + offset_in_subtree.0 - body_offset.0;
        let w = node.final_layout.size.width;
        let h = node.final_layout.size.height;
        let is_monolithic = is_monolithic_in_absolute_subtree(doc, node, page_h_px);
        let is_sliced = has_contain_size(node) && !is_monolithic;
        let monolithic_adjust: f32 = children
            .iter()
            .filter_map(|child_id| doc.get_node(*child_id))
            .filter(|child| {
                !is_out_of_flow_positioned(child)
                    && is_monolithic_in_absolute_subtree(doc, child, page_h_px)
            })
            .map(|child| (child.final_layout.size.height - page_h_px).max(0.0))
            .sum();
        let h_for_paging = (h - monolithic_adjust).max(0.0);
        let mut descendant_total_pages = total_pages;

        if final_y_for_paging.is_finite()
            && h.is_finite()
            && h_for_paging.is_finite()
            && page_h_px > 0.0
            && h_for_paging > 0.0
        {
            // Stylo computes `Nvh` against the viewport snapshot taken
            // at parse time, which can differ from `page_h_px` by a
            // sub-px amount (the @page resolution uses the resolved
            // content area; Stylo's computed `100vh` rounds elsewhere).
            // Without tolerance, `top: 100vh` on a 1-page render
            // becomes `final_y = 971.0` against `page_h_px = 971.34`,
            // which floor()s to page 0 and renders the off-page text
            // on page 1 (WPT fixedpos-008 ref-side). Snap final_y
            // toward integer multiples of page_h before paging.
            let start_ratio = final_y_for_paging / page_h_px;
            let start_round = start_ratio.round();
            // fulgur-xa9q: the Stylo `Nvh` sub-px residual vs `page_h_px` scales
            // with the `vh` MULTIPLE summed into `final_y_for_paging` — Stylo's
            // per-`vh`-unit rounding error is multiplied by the vh count, so a
            // `top:400vh` lands ~1.35px short of `4*page_h` and a `top:500vh`
            // further, regardless of nesting depth. A flat `1e-3` tolerance
            // misses this and mis-pages the element onto the page boundary so
            // its line splits and clips (fixedpos-005 "fifth", fixedpos-008
            // page 6). Scale the tolerance by the rounded page (≈ the vh
            // multiple), capped at 1% of the page (~half a line) so extreme
            // page counts cannot grow the window unbounded. NOTE: depth-scaling
            // was tried and regresses fixedpos-008 — the residual tracks the vh
            // multiple, not nesting depth (Codex review on PR #498).
            let snap_tol = (1e-3 * start_round.abs().max(1.0)).min(0.01);
            let start_is_snapped = (start_ratio - start_round).abs() < snap_tol;
            let snapped_start_ratio = if start_is_snapped {
                start_round
            } else {
                start_ratio
            };
            let bottom_y_for_paging = final_y_for_paging + h_for_paging;
            let bottom_ratio = bottom_y_for_paging / page_h_px;
            let mut last_page_f =
                if bottom_y_for_paging.is_infinite() && bottom_y_for_paging.is_sign_positive() {
                    total_pages.saturating_sub(1) as f32
                } else if (bottom_ratio - bottom_ratio.round()).abs() < 1e-6 {
                    bottom_ratio.round() - 1.0
                } else {
                    bottom_ratio.floor()
                };
            let first_page_f = snapped_start_ratio.floor().max(0.0);
            if is_monolithic {
                last_page_f = first_page_f;
            }
            // fulgur-xa9q: the start snap can advance `first_page_f` onto the
            // next page while `last_page_f` still floors from the UNSNAPPED
            // bottom; for a box shorter than the corrected residual that would
            // make `first_page_f > last_page_f` and drop the fragment entirely
            // (Codex review). A box must appear on at least its (snapped) start
            // page, so never let the last page precede the first.
            last_page_f = last_page_f.max(first_page_f);
            if is_sliced && last_page_f.is_finite() {
                let pages = (last_page_f.max(0.0) as u32).min(crate::MAX_PAGES);
                sliced.pages = sliced.pages.max(pages.saturating_add(1));
                sliced.boxes.push(node_id);
            }
            // fulgur-xa9q: an abs whose START is at/beyond the existing in-flow
            // page budget extends the page count even with in-flow content
            // present (Chrome-compatible) — UNLESS it sits inside a containment
            // /clip boundary, where overflow is invisible and must not paginate.
            // A tall abs anchored WITHIN the budget stays clamped (it has an
            // in-budget page to clip onto) so short-flow layouts do not grow.
            let node_may_extend =
                may_extend_pages || (first_page_f >= total_pages as f32 && !containment_boundary);
            if first_page_f.is_finite()
                && last_page_f.is_finite()
                && first_page_f <= last_page_f
                && (node_may_extend || first_page_f < total_pages as f32)
            {
                // fulgur-2m6w: clamp the emitted page index to `MAX_PAGES`,
                // but ONLY on the page-EXTENSION path. `first_page_f` /
                // `last_page_f` derive from the abs element's viewport-CB
                // resolved Y, which is attacker-controlled CSS
                // (`position:absolute; top:99999999px`). When the element
                // extends the page count (`node_may_extend`) an unclamped
                // value lands at page ~10^5, `descendant_total_pages`
                // propagates that to `implied_page_count`, and `render_v2`
                // allocates + renders every intervening page — the same
                // small-input DoS the body-direct slice cap blocks. This
                // `walk` runs for the abs root AND nested abs descendants,
                // so one clamp bounds both.
                //
                // The NON-extending branch is already bounded by
                // `total_pages` (the in-flow page count, itself capped /
                // input-proportional), so it must stay UNCLAMPED: clamping
                // `first_page` while `last_page` keeps `min(.., total_pages-1)`
                // would emit a spurious fragment run from `MAX_PAGES` through
                // the real (in-budget) page when `total_pages > MAX_PAGES`
                // from many ordinary in-flow pages (Codex review on PR #501).
                let (first_page, last_page) = if node_may_extend {
                    (
                        (first_page_f as u32).min(crate::MAX_PAGES),
                        (last_page_f as u32).min(crate::MAX_PAGES),
                    )
                } else {
                    (
                        first_page_f as u32,
                        (last_page_f as u32).min(total_pages.saturating_sub(1)),
                    )
                };
                // fulgur-xa9q: page ASSIGNMENT snaps the start ratio to the page
                // grid (`first_page_f` via `snapped_start_ratio`), but the stored
                // Y must use the SAME snapped origin. Otherwise a `top: 100vh`
                // abs lands ~1px off the in-flow fragmenter's exact page-top
                // placement (the Stylo `100vh` vs `page_h_px` sub-px residual),
                // producing the fixedpos-005/006/008 page-2 pixel diff once the
                // page count matches. Snap only the paging origin, and only for
                // top-anchored subtrees (`page_stride_px == page_h_px`); the
                // bottom-only repeat idiom uses a rounded stride and must keep
                // its raw origin.
                let paging_origin_y = if page_stride_px == page_h_px && start_is_snapped {
                    snapped_start_ratio * page_h_px
                } else {
                    final_y_for_paging
                };
                let entry = geometry.entry(node_id).or_default();
                entry.fragments.clear();
                entry.line_boundaries.clear();
                entry.is_repeat = false;
                for page_index in first_page..=last_page {
                    // Aggregate budget: each node emits one fragment per
                    // intersected page, so a subtree of many page-spanning
                    // absolutes is O(nodes × pages). Stop past the cap
                    // (`crate::MAX_SUBTREE_PAGE_FRAGMENTS`) — shared with the
                    // fixed pass. Use `break`, not `return`: the walk must keep
                    // visiting (and clearing) the rest of the subtree so no
                    // stale fragments survive from an earlier pass (Codex
                    // review).
                    if *emitted >= crate::MAX_SUBTREE_PAGE_FRAGMENTS {
                        break;
                    }
                    let is_monolithic_continuation =
                        monolithic_adjust > 0.0 && page_index > first_page;
                    let page_top = (page_index as f32) * page_stride_px;
                    let (stored_y, stored_h) = if is_monolithic_continuation {
                        let consumed = (page_index - first_page) as f32 * page_h_px;
                        (
                            -body_offset.1,
                            (h_for_paging - consumed).clamp(0.0, page_h_px),
                        )
                    } else {
                        (paging_origin_y - page_top - body_offset.1, h)
                    };
                    entry.fragments.push(Fragment {
                        page_index,
                        x: stored_x.as_px(),
                        y: stored_y.as_px(),
                        width: w.as_px(),
                        height: stored_h.as_px(),
                    });
                    *emitted += 1;
                }
                descendant_total_pages = descendant_total_pages.max(last_page.saturating_add(1));
                sliced.reached = sliced.reached.max(last_page.saturating_add(1));
            }
        }

        let mut monolithic_y_adjust = 0.0;
        // Bottom edge, in `node`-local px, of the in-flow children walked so
        // far. A `display: none` running child has a zero-size box at the
        // parent's origin, so its source position is this cursor rather than
        // `final_layout.location`. Margin collapsing is not modelled.
        let mut flow_bottom = 0.0_f32;
        for child_id in children {
            let Some(child) = doc.get_node(child_id) else {
                continue;
            };
            // fulgur: `position: running(name)` children are rewritten to
            // `display: none` by `gcpm::parser::parse_gcpm`'s cleaned_css
            // (they are removed from normal flow and repainted only via
            // their `@page` margin box). A `display: none` node collapses
            // to a zero-size Taffy box, so the generic below (height-gated
            // on `h_for_paging > 0.0`) never records a geometry entry for
            // it — unlike `fragment_pagination_root`'s in-flow body walk,
            // which has the equivalent carve-out at the cursor level. Without
            // this check here, a running element nested inside a
            // `position: absolute` / body-direct-abs subtree never lands in
            // `geometry`, so `collect_running_element_states` can't find its
            // NodeId and the margin box it should populate renders empty
            // (fulgur `--css` + hidden-ancestor running-element bug).
            // Mirror the main walk: record a zero-size marker at the
            // running child's current subtree position and skip recursing
            // into it (it has no flow content of its own to lay out).
            if running_store.is_some_and(|s| s.instance_for_node(child_id).is_some()) {
                if child.element_data().is_some() && page_h_px > 0.0 {
                    // `max` keeps the real position of a running element an
                    // author re-displayed (it then has a box of its own).
                    let child_y_for_paging = root_xy_for_paging.1
                        + offset_in_subtree.1
                        + child.final_layout.location.y.max(flow_bottom)
                        - monolithic_y_adjust;
                    let page_index = (child_y_for_paging / page_h_px).max(0.0).floor() as u32;
                    let page_index = page_index.min(descendant_total_pages.saturating_sub(1));
                    let stored_x =
                        root_xy_for_paging.0 + offset_in_subtree.0 + child.final_layout.location.x
                            - body_offset.0;
                    let stored_y =
                        child_y_for_paging - (page_index as f32) * page_h_px - body_offset.1;
                    let entry = geometry.entry(child_id).or_default();
                    entry.fragments.clear();
                    entry.fragments.push(Fragment {
                        page_index,
                        x: stored_x.as_px(),
                        y: stored_y.as_px(),
                        width: 0.0_f32.as_px(),
                        height: 0.0_f32.as_px(),
                    });
                }
                continue;
            }
            // Out-of-flow descendants:
            //   - `position: fixed` is a repeat element handled by
            //     `append_position_fixed_fragments`; skip it here so it
            //     does not contribute to extent (it would otherwise be
            //     double-counted).
            //   - `position: absolute` (fulgur-puml #1): a nested abs's
            //     height/offset must still drive the page count. Resolve
            //     its location against ITS containing block (the nearest
            //     positioned ancestor = the current `node`'s box) via
            //     `resolve_viewport_cb_location`, fold that CB-relative
            //     (x, y) into the accumulated `offset_in_subtree`, and
            //     recurse. `root_xy_for_paging` stays the body-direct
            //     abs root anchor; viewport-relative (`vh`) insets are
            //     CB-independent, so the CB dims only affect percentage
            //     insets.
            {
                use ::style::properties::longhands::position::computed_value::T as Pos;
                match child.primary_styles().map(|s| s.get_box().clone_position()) {
                    // Fixed is handled by its own pass; skip here.
                    Some(Pos::Fixed) => continue,
                    Some(Pos::Absolute) => {
                        let (child_w, child_h) = (
                            child.final_layout.size.width,
                            child.final_layout.size.height,
                        );
                        // Resolve against the nearest positioned ancestor's
                        // padding box (`child_cb_*`), NOT the immediate DOM
                        // parent — a `position: static` parent does not
                        // establish a CB (CSS 2.1 §10.1.4).
                        let (rel_x, rel_y) = resolve_viewport_cb_location(
                            child,
                            child_w,
                            child_h,
                            child_cb_size.0,
                            child_cb_size.1,
                        )
                        .unwrap_or((child.final_layout.location.x, child.final_layout.location.y));
                        // Per-axis anchor: an explicit inset is relative to the
                        // CB origin (`child_cb_anchor`); an `auto` inset keeps
                        // the static position, i.e. the flow offset relative to
                        // the immediate parent (`offset_in_subtree`). For a
                        // directly-nested abs the immediate parent IS the
                        // positioned ancestor, so both bases coincide and this
                        // is a no-op; they diverge only across a static
                        // intermediate.
                        let (explicit_x, explicit_y) = explicit_inset_axes(child);
                        let base_x = if explicit_x {
                            child_cb_anchor.0
                        } else {
                            offset_in_subtree.0
                        };
                        let base_y = if explicit_y {
                            child_cb_anchor.1
                        } else {
                            offset_in_subtree.1
                        };
                        let nested_offset = (base_x + rel_x, base_y + rel_y);
                        walk(
                            geometry,
                            doc,
                            child_id,
                            nested_offset,
                            root_xy_for_paging,
                            body_offset,
                            page_h_px,
                            page_stride_px,
                            descendant_total_pages,
                            may_extend_pages,
                            containment_boundary || clips_overflow(node),
                            child_cb_anchor,
                            child_cb_size,
                            depth + 1,
                            emitted,
                            sliced,
                            running_store,
                        );
                        continue;
                    }
                    // In-flow (or unstyled): fall through to the normal path below.
                    _ => {}
                }
            }
            // Skip whitespace-only text (matches fragmenter).
            if crate::blitz_adapter::is_whitespace_only_text_node(child) {
                continue;
            }
            let child_offset = (
                offset_in_subtree.0 + child.final_layout.location.x,
                offset_in_subtree.1 + child.final_layout.location.y - monolithic_y_adjust,
            );
            walk(
                geometry,
                doc,
                child_id,
                child_offset,
                root_xy_for_paging,
                body_offset,
                page_h_px,
                page_stride_px,
                descendant_total_pages,
                may_extend_pages,
                containment_boundary || clips_overflow(node),
                child_cb_anchor,
                child_cb_size,
                depth + 1,
                emitted,
                sliced,
                running_store,
            );
            flow_bottom =
                flow_bottom.max(child.final_layout.location.y + child.final_layout.size.height);
            if is_monolithic_in_absolute_subtree(doc, child, page_h_px) {
                monolithic_y_adjust += (child.final_layout.size.height - page_h_px).max(0.0);
            }
        }
    }

    let emitted_before = *emitted;
    let mut sliced = SlicedBoxes::default();
    let walk_within = |geometry: &mut PaginationGeometryTable,
                       emitted: &mut usize,
                       total_pages: u32,
                       sliced: &mut SlicedBoxes| {
        walk(
            geometry,
            doc,
            subtree_root_id,
            (0.0, 0.0),
            root_xy_for_paging,
            body_offset,
            page_h_px,
            page_stride_px,
            total_pages,
            may_extend_pages,
            // fulgur-xa9q: the subtree root starts outside any containment boundary.
            false,
            // Initial CB: the subtree root is the body-direct abs (positioned),
            // so it overrides these on the first frame — seed with the root's own
            // anchor/size for correctness if that ever changes.
            (0.0, 0.0),
            (0.0, 0.0),
            0,
            emitted,
            sliced,
            running_store,
        )
    };
    walk_within(geometry, emitted, total_pages, &mut sliced);
    // The slices of a box block flow would slice take pages past the flow's,
    // as block flow's slices do, and the subtree around them reaches those
    // pages too: walk it again with them in the budget, and with the pages
    // the boxes past the flow's pages reached, which now start within it.
    // Each node then emits on at least the pages it did, so every entry of
    // the first walk is rewritten.
    if !may_extend_pages && sliced.pages > total_pages {
        *emitted = emitted_before;
        let budget = sliced.pages.max(sliced.reached);
        sliced = SlicedBoxes::default();
        walk_within(geometry, emitted, budget, &mut sliced);
    }
    // A sliced box, and every box around it up to the subtree's root, has
    // one fragment a page that ends at the page foot and starts at the page
    // top after the first, as in block flow; the walk stored each page's
    // fragment as the whole box moved up by the pages before it.
    let mut clipped = BTreeSet::new();
    for id in sliced.boxes {
        let mut next = Some(id);
        while let Some(node_id) = next.filter(|&node_id| clipped.insert(node_id)) {
            if let Some(entry) = geometry.get_mut(&node_id) {
                for fragment in &mut entry.fragments {
                    let top = fragment.y.to_f32() + body_offset.1;
                    let bottom = top + fragment.height.to_f32();
                    let (top, bottom) = (top.max(0.0), bottom.min(page_h_px));
                    fragment.y = (top - body_offset.1).as_px();
                    fragment.height = (bottom - top).max(0.0).as_px();
                }
            }
            next = (node_id != subtree_root_id)
                .then(|| doc.get_node(node_id))
                .flatten()
                .and_then(|node| node.layout_parent.get().or(node.parent));
        }
    }
}

/// The boxes the walk of an absolute subtree sliced
/// ([`is_monolithic_in_absolute_subtree`]), the number of pages their
/// slices reach, budget or not, and the number of pages the fragments the
/// walk emitted reach.
#[derive(Default)]
struct SlicedBoxes {
    pages: u32,
    reached: u32,
    boxes: Vec<usize>,
}

/// Which axes of an out-of-flow element carry an explicit (length /
/// percentage) inset, as `(x, y)`. An explicit inset is resolved against the
/// containing block; an `auto` inset falls back to the static (flow) position.
/// Mirrors the per-axis idiom in [`resolve_viewport_cb_location`].
fn explicit_inset_axes(node: &blitz_dom::Node) -> (bool, bool) {
    use ::style::values::generics::position::GenericInset;

    fn is_length_percentage(inset: &::style::values::computed::position::Inset) -> bool {
        matches!(inset, GenericInset::LengthPercentage(_))
    }

    let Some(styles) = node.primary_styles() else {
        return (false, false);
    };
    let pos = styles.get_position();
    (
        is_length_percentage(&pos.left) || is_length_percentage(&pos.right),
        is_length_percentage(&pos.top) || is_length_percentage(&pos.bottom),
    )
}

fn is_out_of_flow_positioned(node: &blitz_dom::Node) -> bool {
    use ::style::properties::longhands::position::computed_value::T as Pos;

    node.primary_styles()
        .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Absolute | Pos::Fixed))
}

/// Whether a size-contained box in an absolute subtree is monolithic there:
/// one fragment on the page it starts on, and the content after it in its
/// parent moved up by its excess over a page. A box taller than a page that
/// block flow slices where it is ([`slices_where_it_is`]) is sliced into one
/// fragment a page instead, with what follows it after its last slice.
fn is_monolithic_in_absolute_subtree(
    doc: &BaseDocument,
    node: &blitz_dom::Node,
    page_h_px: f32,
) -> bool {
    has_contain_size(node)
        && !(overflows(node.final_layout.size.height, page_h_px)
            && slices_where_it_is(doc, node, false))
}

fn has_contain_size(node: &blitz_dom::Node) -> bool {
    node.primary_styles().is_some_and(|s| {
        s.get_box()
            .clone_contain()
            .contains(::style::values::computed::box_::Contain::SIZE)
    })
}

/// Whether `node` establishes a clip context that hides its overflowing
/// descendants from painting — used as the page-extension boundary
/// (fulgur-xa9q): a descendant whose paint overflow is clipped here cannot
/// generate pages even when it lands past the in-flow budget.
///
/// Two conditions, both required:
///   1. `overflow` is not `visible` on some axis (`hidden`/`clip`/`scroll`/
///      `auto` all clip; mirrors `convert::style::overflow`).
///   2. the box actually establishes a clip context — a block-level box or an
///      atomic inline / BFC root. A non-replaced `display: inline` box ignores
///      `overflow` and the renderer pushes no clip scope for it, so an inline
///      wrapper like `<span style="overflow:hidden">` must NOT act as a
///      boundary (Codex review on PR #498).
///
/// NOTE: `contain: size` is deliberately NOT treated as a clip — it sizes the
/// box without its contents but leaves overflow visible, so a
/// `contain:size; overflow:visible` box must still let descendants extend.
fn clips_overflow(node: &blitz_dom::Node) -> bool {
    use ::style::values::computed::Overflow as Ov;
    use ::style::values::specified::box_::{DisplayInside, DisplayOutside};
    node.primary_styles().is_some_and(|s| {
        let clips = s.clone_overflow_x() != Ov::Visible || s.clone_overflow_y() != Ov::Visible;
        if !clips {
            return false;
        }
        let display = s.clone_display();
        display.outside() == DisplayOutside::Block || display.inside() != DisplayInside::Flow
    })
}

/// Whether `item`, a child of the flex or grid container `container`,
/// shares its row with another item: an in-flow sibling whose layout
/// extent overlaps its own vertically. A flex column, or a grid of one
/// column, stacks its items, as block flow does.
fn item_shares_its_row(
    doc: &BaseDocument,
    container: &blitz_dom::Node,
    item: &blitz_dom::Node,
) -> bool {
    let (top, bottom) = (
        item.final_layout.location.y,
        item.final_layout.location.y + item.final_layout.size.height,
    );
    let layout_children = container.layout_children.borrow();
    let children = layout_children
        .as_deref()
        .filter(|children| !children.is_empty())
        .unwrap_or(&container.children);
    children
        .iter()
        .filter(|&&id| id != item.id)
        .filter_map(|&id| doc.get_node(id))
        .filter(|sibling| !out_of_flow(sibling))
        .any(|sibling| {
            let layout = sibling.final_layout;
            layout.location.y < bottom - 0.5 && layout.location.y + layout.size.height > top + 0.5
        })
}

/// Whether a box taller than a page that a walk does not enter may be
/// sliced where it is, at any depth. Not past the bottom of an ancestor
/// below the body that clips its overflow ([`clips_overflow`]): what is
/// past the clip paints nothing, so it takes no page. `in_lines` is the
/// walker's: it places a table row's cells and a flex or grid line's items
/// side by side as a unit ([`LineState`]), so a box in a cell or an item
/// slices as in block flow. The absolute-positioning pass places no lines:
/// there a box in a table cell, or in a flex or grid item that shares its
/// row ([`item_shares_its_row`]), is not sliced, as the items of a row
/// share its height and one that continued on the next page would part
/// from those beside it. Rows and cells count up to the root of the
/// absolute or fixed subtree the box is in, which may be the box itself;
/// clips up to the body.
fn slices_where_it_is(doc: &BaseDocument, node: &blitz_dom::Node, in_lines: bool) -> bool {
    // A cell that spans rows ties them together, which the walker does not
    // place as one unit yet: nothing in it is sliced.
    let tied = |cell: &blitz_dom::Node| {
        displays(cell, DisplayInside::TableCell) && (!in_lines || rowspan_of(cell) > 1)
    };
    if tied(node) {
        return false;
    }
    // The rows and cells are those of the flow the box is in, which ends at
    // the root of an absolute or fixed subtree: that root is no item or cell
    // of its parent. The clips go on up to the body.
    let mut in_flow = !is_out_of_flow_positioned(node);
    let mut bottom = node.final_layout.location.y + node.final_layout.size.height;
    let mut below = node;
    let mut next = node.parent;
    while let Some(ancestor) = next.and_then(|id| doc.get_node(id)) {
        let is_body = ancestor
            .element_data()
            .is_some_and(|el| el.name.local.as_ref() == "body");
        let layout = ancestor.final_layout;
        if (in_flow
            && (tied(ancestor)
                || (!in_lines
                    && crate::blitz_adapter::is_flex_or_grid_container_node(ancestor)
                    && item_shares_its_row(doc, ancestor, below))))
            || (!is_body && clips_overflow(ancestor) && bottom > layout.size.height + 0.5)
        {
            return false;
        }
        if is_body {
            return true;
        }
        in_flow &= !is_out_of_flow_positioned(ancestor);
        bottom += layout.location.y;
        below = ancestor;
        next = ancestor.parent;
    }
    true
}

/// CSS-px (x, y) of `<body>`'s top-left in its containing block (html).
/// Mirrors `convert::extract_body_offset_pt` but stays in CSS px so
/// pagination_layout doesn't need to round-trip through pt. The render
/// path adds `drawables.body_offset_pt` to every fragment's y when
/// dispatching, so viewport-anchored fragments must subtract this
/// offset to keep the dispatched y page-relative.
fn body_origin_in_px(doc: &BaseDocument) -> (f32, f32) {
    let Some(body_id) = find_body_id(doc) else {
        return (0.0, 0.0);
    };
    let Some(body) = doc.get_node(body_id) else {
        return (0.0, 0.0);
    };
    (body.final_layout.location.x, body.final_layout.location.y)
}

/// Resolve a viewport-CB-anchored absolute/fixed element's CSS px
/// (x, y) using its computed `top` / `left` / `right` / `bottom`
/// insets. Mirrors CSS 2.1 §10.3.7 / §10.6.4 over-constrained
/// resolution: start-side (top / left) wins when both sides are set,
/// end-side (bottom / right) only fires when the start-side is `auto`.
///
/// Returns `None` when no inset is set on either axis (caller should
/// keep Taffy's `final_layout.location` as the static-position
/// fallback).
fn resolve_viewport_cb_location(
    node: &blitz_dom::Node,
    el_w_px: f32,
    el_h_px: f32,
    cb_w_px: f32,
    cb_h_px: f32,
) -> Option<(f32, f32)> {
    use ::style::values::computed::Length;
    use ::style::values::generics::position::GenericInset;

    fn resolve(inset: &::style::values::computed::position::Inset, basis_px: f32) -> Option<f32> {
        match inset {
            GenericInset::LengthPercentage(lp) => Some(lp.resolve(Length::new(basis_px)).px()),
            _ => None,
        }
    }

    let styles = node.primary_styles()?;
    let pos = styles.get_position();
    let left = resolve(&pos.left, cb_w_px);
    let top = resolve(&pos.top, cb_h_px);
    let right = resolve(&pos.right, cb_w_px);
    let bottom = resolve(&pos.bottom, cb_h_px);
    if left.is_none() && top.is_none() && right.is_none() && bottom.is_none() {
        return None;
    }
    // Per-axis resolution: when both insets on an axis are `auto`, fall
    // back to Taffy's static position (`final_layout.location`). Caller
    // unwraps the returned tuple against that fallback, so we surface
    // only the axes that have an explicit inset.
    // End-side (right / bottom) anchoring positions the element's *margin
    // box* edge against the CB edge, so the border-box origin must back off
    // by the used end-side margin (CSS 2.1 §10.3.7 / §10.6.4). Taffy resolves
    // these into `final_layout.margin`; without subtracting them an abs with
    // `bottom:0; margin-bottom:2em` collapses onto `bottom:0`. This is a
    // distinct end-side-margin bug from nested-abs pagination — see the
    // `abs_bottom_margin_offsets_above_sibling` regression test. NOTE: this
    // helper resolves `position: fixed` elements too (all three callers), so
    // the margin term corrects fixed end-anchoring as well as abs. The margin
    // is subtracted unrounded between the two rounded terms; fixedpos-004's
    // pixel-exact reftest is the validator for this rounding choice.
    let margin = node.final_layout.margin;
    let x = if let Some(l) = left {
        l
    } else if let Some(r) = right {
        // To mirror Taffy's internal end-side layout, collapse viewport
        // edge before subtracting element width.
        (cb_w_px - r).round() - margin.right - el_w_px.round()
    } else {
        node.final_layout.location.x
    };
    let y = if let Some(t) = top {
        t
    } else if let Some(b) = bottom {
        // Same order as the in-tree flow for `bottom: 0` style anchors:
        // first round viewport location, then subtract rounded element
        // height. This keeps fixed/abs reference paths aligned to one
        // px boundary when cb height / element size are sub-pixel.
        (cb_h_px - b).round() - margin.bottom - el_h_px.round()
    } else {
        node.final_layout.location.y
    };
    Some((x, y))
}

fn uses_bottom_without_top(node: &blitz_dom::Node) -> bool {
    use ::style::values::generics::position::GenericInset;

    fn is_length_percentage(inset: &::style::values::computed::position::Inset) -> bool {
        matches!(inset, GenericInset::LengthPercentage(_))
    }

    let Some(styles) = node.primary_styles() else {
        return false;
    };
    let pos = styles.get_position();
    !is_length_percentage(&pos.top) && is_length_percentage(&pos.bottom)
}

/// Recursive walker that collects every node id whose computed
/// `position` is `fixed`. Mirrors the helper of the same shape in
/// `blitz_adapter::relayout_position_fixed`. Visits raw `node.children`
/// rather than `layout_children` because the latter may be invalidated
/// by the time this runs, and pseudo-elements (`::before` / `::after`)
/// live in `node.before` / `node.after` outside the children vec.
///
/// Used by [`append_position_fixed_fragments`].
fn walk_for_position_fixed(doc: &BaseDocument, node_id: usize, out: &mut Vec<usize>, depth: usize) {
    use ::style::properties::longhands::position::computed_value::T as Pos;

    if depth >= crate::MAX_DOM_DEPTH {
        return;
    }
    let Some(node) = doc.get_node(node_id) else {
        return;
    };
    let is_fixed = node
        .primary_styles()
        .is_some_and(|s| matches!(s.get_box().clone_position(), Pos::Fixed));
    if is_fixed {
        out.push(node_id);
    }
    for &child_id in &node.children {
        walk_for_position_fixed(doc, child_id, out, depth + 1);
    }
    // Pseudo-elements: a `::before { position: fixed }` would
    // otherwise be missed by the children-only walk. The `before` /
    // `after` slots live directly on `Node`, not on `ElementData`.
    if let Some(pseudo_id) = node.before {
        walk_for_position_fixed(doc, pseudo_id, out, depth + 1);
    }
    if let Some(pseudo_id) = node.after {
        walk_for_position_fixed(doc, pseudo_id, out, depth + 1);
    }
}

/// fulgur-jkl5: total page count implied by a geometry table.
///
/// Returns `max(page_index) + 1` if the table has any fragments, else
/// `1` (at least one page is always implied).
pub fn implied_page_count(geometry: &PaginationGeometryTable) -> u32 {
    geometry
        .values()
        .flat_map(|g| g.fragments.iter())
        .map(|f| f.page_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

/// The geometry table as text: one line per node and fragment, nodes in
/// tree order and each node's fragments in their vector order, so two runs
/// of the fragmenter can be compared line by line.
///
/// Each line holds the node's path from the root (each step is the tag
/// name, `#id` when the element has one, and the index in its parent's
/// children), then the fragment's page, `x`, `y`, `width`, and `height`
/// rounded to 0.01 px, `repeat` when the geometry is a per-page copy, every
/// entry of the node's `line_boundaries` when it has any, and the node's
/// `lift_px` when it is not zero. A node with no fragment is one line
/// with `none` in place of the fragment.
///
/// Anonymous blocks are `::anon`, text nodes `#text`, and generated
/// boxes `::before` and `::after`; a node outside its parent's DOM
/// children takes its index in the parent's layout children.
pub fn geometry_text(geometry: &PaginationGeometryTable, doc: &BaseDocument) -> String {
    let mut nodes: Vec<(Vec<(usize, String)>, &PaginationGeometry)> = geometry
        .iter()
        .map(|(&id, g)| (geometry_path(doc, id), g))
        .collect();
    nodes.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::new();
    for (path, g) in nodes {
        let path: String = path
            .iter()
            .map(|(_, step)| step.as_str())
            .collect::<Vec<_>>()
            .join("/");
        let mut node = String::new();
        if g.is_repeat {
            node.push_str(" repeat");
        }
        if !g.line_boundaries.is_empty() {
            let lines: Vec<String> = g.line_boundaries.iter().map(usize::to_string).collect();
            node.push_str(&format!(" lines={}", lines.join(",")));
        }
        if g.lift_px != 0.0 {
            node.push_str(&format!(" lift={}", geometry_number(g.lift_px)));
        }
        if g.fragments.is_empty() {
            out.push_str(&format!("{path} none{node}\n"));
        }
        for f in &g.fragments {
            out.push_str(&format!(
                "{path} page={} x={} y={} w={} h={}{node}\n",
                f.page_index,
                geometry_number(f.x.to_f32()),
                geometry_number(f.y.to_f32()),
                geometry_number(f.width.to_f32()),
                geometry_number(f.height.to_f32()),
            ));
        }
    }
    out
}

/// A length rounded to 0.01 px, without a negative zero.
fn geometry_number(v: f32) -> String {
    let rounded = (f64::from(v) * 100.0).round() / 100.0;
    if rounded == 0.0 {
        "0.00".to_string()
    } else {
        format!("{rounded:.2}")
    }
}

/// The steps from the root to `id`: each the sort key (the index under the
/// parent, with the generated boxes around the children) and its text.
fn geometry_path(doc: &BaseDocument, id: usize) -> Vec<(usize, String)> {
    let mut steps = Vec::new();
    let mut current = id;
    while let Some(node) = doc.get_node(current) {
        let parent_id = node.parent.or_else(|| node.layout_parent.get());
        let Some(parent) = parent_id.and_then(|p| doc.get_node(p)) else {
            break;
        };
        let (key, step) = if parent.before == Some(current) {
            (0, "::before".to_string())
        } else if parent.after == Some(current) {
            (usize::MAX, "::after".to_string())
        } else {
            let index = parent
                .children
                .iter()
                .position(|&c| c == current)
                .or_else(|| {
                    parent
                        .layout_children
                        .borrow()
                        .as_ref()
                        .and_then(|children| children.iter().position(|&c| c == current))
                })
                .unwrap_or(usize::MAX - 1);
            (index + 1, format!("{}[{index}]", geometry_label(node)))
        };
        steps.push((key, step));
        current = parent.id;
    }
    steps.reverse();
    steps
}

/// The label of one path step: the tag name and `#id`, `::anon`, or `#text`.
fn geometry_label(node: &blitz_dom::Node) -> String {
    if node.is_anonymous() {
        return "::anon".to_string();
    }
    if node.is_text_node() {
        return "#text".to_string();
    }
    match node.element_data() {
        Some(data) => match data.attr(blitz_dom::LocalName::from("id")) {
            Some(id) => format!("{}#{id}", data.name.local),
            None => data.name.local.to_string(),
        },
        None => "#node".to_string(),
    }
}

/// Locate the `<body>` element id by walking the html root's children.
///
/// Prefers the first child whose tag name is `body`. Falls back to
/// `None` when the document has no `<body>` (e.g. fragments parsed
/// outside a normal document context). Spec-pure HTML5 parsing always
/// synthesizes a `<body>`, but tests and library callers can pass
/// arbitrary fragments so we do not rely on its presence.
fn find_body_id(doc: &BaseDocument) -> Option<usize> {
    let root_id = doc.root_element().id;
    let root = doc.get_node(root_id)?;
    for child_id in &root.children {
        let Some(child) = doc.get_node(*child_id) else {
            continue;
        };
        if let Some(elem) = child.element_data()
            && elem.name.local.as_ref() == "body"
        {
            return Some(*child_id);
        }
    }
    None
}

// ── Trait delegation to BaseDocument (mirrors multicol_layout) ────────────
//
// These trait impls are not exercised by the current measurement-only
// `fragment_pagination_root` walk — they are scaffolding for the next
// iteration that will call `taffy::compute_root_layout(self, body_id, ...)`
// to drive the fragmenter through Taffy's normal dispatch. Keeping the
// shapes here so the upgrade is a localized change.

impl TraversePartialTree for PaginationLayoutTree<'_> {
    type ChildIter<'a>
        = <BaseDocument as TraversePartialTree>::ChildIter<'a>
    where
        Self: 'a;

    fn child_ids(&self, node_id: NodeId) -> Self::ChildIter<'_> {
        self.doc.child_ids(node_id)
    }

    fn child_count(&self, node_id: NodeId) -> usize {
        self.doc.child_count(node_id)
    }

    fn get_child_id(&self, node_id: NodeId, index: usize) -> NodeId {
        self.doc.get_child_id(node_id, index)
    }
}

impl TraverseTree for PaginationLayoutTree<'_> {}

impl CacheTree for PaginationLayoutTree<'_> {
    fn cache_get(
        &self,
        node_id: NodeId,
        known_dimensions: Size<Option<f32>>,
        available_space: Size<AvailableSpace>,
        run_mode: taffy::RunMode,
    ) -> Option<taffy::LayoutOutput> {
        self.doc
            .cache_get(node_id, known_dimensions, available_space, run_mode)
    }

    fn cache_store(
        &mut self,
        node_id: NodeId,
        known_dimensions: Size<Option<f32>>,
        available_space: Size<AvailableSpace>,
        run_mode: taffy::RunMode,
        layout_output: taffy::LayoutOutput,
    ) {
        self.doc.cache_store(
            node_id,
            known_dimensions,
            available_space,
            run_mode,
            layout_output,
        );
    }

    fn cache_clear(&mut self, node_id: NodeId) {
        self.doc.cache_clear(node_id);
    }
}

impl LayoutPartialTree for PaginationLayoutTree<'_> {
    type CoreContainerStyle<'a>
        = &'a taffy::Style<style::Atom>
    where
        Self: 'a;

    type CustomIdent = style::Atom;

    fn get_core_container_style(&self, node_id: NodeId) -> Self::CoreContainerStyle<'_> {
        self.doc.get_core_container_style(node_id)
    }

    fn set_unrounded_layout(&mut self, node_id: NodeId, layout: &taffy::Layout) {
        self.doc.set_unrounded_layout(node_id, layout);
    }

    fn resolve_calc_value(&self, calc_ptr: *const (), parent_size: f32) -> f32 {
        self.doc.resolve_calc_value(calc_ptr, parent_size)
    }

    fn compute_child_layout(
        &mut self,
        node_id: NodeId,
        inputs: taffy::tree::LayoutInput,
    ) -> taffy::LayoutOutput {
        if Some(usize::from(node_id)) == self.body_id {
            return compute_pagination_layout(self, node_id, inputs);
        }
        // Everything else delegates to BaseDocument's normal dispatch.
        self.doc.compute_child_layout(node_id, inputs)
    }
}

impl RoundTree for PaginationLayoutTree<'_> {
    fn get_unrounded_layout(&self, node_id: NodeId) -> taffy::Layout {
        self.doc.get_unrounded_layout(node_id)
    }

    fn set_final_layout(&mut self, node_id: NodeId, layout: &taffy::Layout) {
        self.doc.set_final_layout(node_id, layout);
    }
}

/// Custom layout dispatch for the body (the fragmenter's fragmentation root).
///
/// Mirrors the structure of [`crate::multicol_layout::compute_multicol_layout`]:
/// the wrapper's `compute_child_layout` fires for body, delegates the
/// real layout to `BaseDocument` (so children's `final_layout` is
/// populated correctly), then post-walks body's direct children and
/// records fragments in the geometry side-table.
///
/// In the next iteration this is where per-strip available_space
/// constraint and child-by-child re-layout will live. For the current
/// fragmenter it's a thin shim that proves the dispatch path works.
fn compute_pagination_layout(
    tree: &mut PaginationLayoutTree<'_>,
    body_id: NodeId,
    inputs: taffy::tree::LayoutInput,
) -> taffy::LayoutOutput {
    // Delegate the actual layout work to BaseDocument so children get
    // their normal natural sizes. The output is body's full natural
    // height — that height is what `convert::dom_to_drawables` already
    // expects to read from `final_layout`.
    let output = tree.doc.compute_child_layout(body_id, inputs);

    // Now post-walk to populate the geometry side-table.
    tree.fragment_pagination_root();

    output
}

#[cfg(test)]
mod invariants;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blitz_adapter;
    use std::ops::DerefMut;
    use std::sync::Arc;

    /// Parse helper for the fragmenter's tests.
    ///
    /// We deliberately don't accept a viewport height: `blitz_adapter::parse`
    /// uses a hardcoded viewport_h internally, and the fragmenter's strip slicing
    /// is driven by the `page_height_px` argument to `run_pass` rather than
    /// by the viewport. The fixtures pass viewport_w only.
    fn parse(html: &str, viewport_w: f32) -> blitz_html::HtmlDocument {
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, viewport_w, &fonts);
        blitz_adapter::resolve(&mut doc);
        doc
    }

    /// Splits `lines` as [`super::fragment_inline_root`] does from
    /// `cursor_y` on `page_index`, and returns the page and the y the
    /// paragraph ends at with the number of fragments it emitted.
    #[allow(clippy::too_many_arguments)]
    fn split_inline_root(
        geometry: &mut PaginationGeometryTable,
        child_id: usize,
        paragraph_x: f32,
        width: f32,
        cursor_y: f32,
        page_index: u32,
        page_height_px: f32,
        lines: &[(f32, f32)],
        limits: LineLimits,
    ) -> (u32, f32, usize) {
        let mut cursor = PageCursor::new(page_index, cursor_y, page_height_px);
        // A paragraph that starts below the top of its page follows content.
        cursor.page_has_content = cursor_y > 0.0;
        let emitted = fragment_inline_root(
            geometry,
            child_id,
            paragraph_x,
            width,
            &mut cursor,
            lines,
            limits,
            0.0,
            0.0,
        );
        (cursor.page_index, cursor.cursor_y, emitted)
    }

    #[test]
    fn empty_document_emits_only_body_fragment() {
        let mut doc = parse("<html><body></body></html>", 600.0);
        let table = run_pass(&mut doc, 800.0);
        // Phase 2.3 fix: body itself is now recorded so its own
        // counter / string-set / bookmark ops are visible to the
        // parity walks. Empty body → just the body fragment.
        assert_eq!(table.len(), 1, "expected only body fragment, got {table:?}");
    }

    #[test]
    fn html_only_input_still_paginates_synthesized_body() {
        // html5ever synthesizes `<body>` for any HTML input, so
        // `find_body_id` always succeeds in the parse pipeline. The
        // synthesized body has no children — the geometry table
        // still contains the body fragment itself (Phase 2.3 fix)
        // but no child entries.
        let mut doc = parse("<html></html>", 600.0);
        let tree = PaginationLayoutTree::new(&mut doc, 800.0);
        assert!(tree.body_id.is_some(), "html5ever should synthesize a body");
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(table.len(), 1, "expected only body fragment, got {table:?}");
    }

    /// fulgur-s67g Phase 2.5: nested descendants must be recorded
    /// in the geometry table on the same page as their ancestor, so
    /// bookmark / counter / string-set markers attached to deeply
    /// nested DOM elements participate in parity assertions.
    #[test]
    fn nested_descendants_inherit_parent_page() {
        let html = r#"
            <html><body>
              <div style="height: 600px">
                <h2 style="height: 30px">Section 1</h2>
              </div>
              <div style="height: 600px">
                <h2 style="height: 30px">Section 2</h2>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);

        // Two outer divs split across two pages (600 + 600 > 800),
        // each carrying a nested h2. Geometry should contain both
        // outer divs AND both inner h2s — four entries total — with
        // the h2 sharing its parent's page_index.
        assert!(
            table.len() >= 4,
            "expected at least 4 entries (2 divs + 2 h2s), got {}",
            table.len(),
        );
        let h2_pages: Vec<u32> = table
            .values()
            .filter(|g| {
                g.fragments
                    .iter()
                    .any(|f| (f.height.to_f32() - 30.0).abs() < 0.5)
            })
            .map(|g| g.fragments[0].page_index)
            .collect();
        assert_eq!(h2_pages.len(), 2, "expected 2 h2 entries, got {h2_pages:?}");
        // Pages of the h2s should match those of their containing divs:
        // first div on page 0 → first h2 on page 0; second div on page
        // 1 → second h2 on page 1.
        assert_eq!(h2_pages, vec![0, 1]);
    }

    /// fulgur-2map.5: directly exercise `fragment_block_subtree`'s
    /// `depth >= MAX_DOM_DEPTH` guard (pagination_layout.rs ~1539-1557).
    ///
    /// The guard is the FIRST statement of the function, so calling it
    /// with `depth = crate::MAX_DOM_DEPTH` trips the bail immediately —
    /// no deep HTML and no big-stack thread required (the prior
    /// render_smoke coverage needed a 600-deep `<div>` chain plus a
    /// 256 MB stack just to recurse far enough to reach this arm). This
    /// also tracks the constant itself: if `MAX_DOM_DEPTH` ever changes,
    /// the test still enters the guard at exactly the limit.
    #[test]
    fn fragment_block_subtree_at_depth_limit_bails_with_whole_fragment() {
        let doc = parse(
            r#"<html><body style="margin:0"><div id="d" style="height:50px"></div></body></html>"#,
            600.0,
        );
        let parent_id = find_by_id(&doc, "d").expect("div#d should exist");
        // The bail copies this node's laid-out height into the fragment.
        let node_h = doc
            .get_node(parent_id)
            .expect("div node")
            .final_layout
            .size
            .height;
        assert!(
            (node_h - 50.0).abs() < 1.0,
            "div should lay out ~50px tall, got {node_h}"
        );

        let mut geom = PaginationGeometryTable::new();
        let mut remaining_repeat_budget = crate::MAX_SUBTREE_PAGE_FRAGMENTS;
        let (page_out, cursor_out) = fragment_block_subtree(
            &mut geom,
            Flow {
                doc: &doc,
                column_styles: None,
                used_page_names: None,
                running_store: None,
                pass: &Pass::default(),
                forced_before: None,
                foot_clearance: 0.0,
            },
            parent_id,
            600.0,                // parent_w
            0.0,                  // parent_x_in_body
            0,                    // page_in
            0.0,                  // cursor_in
            800.0,                // page_height_px
            crate::MAX_DOM_DEPTH, // depth → trips the guard immediately
            false,
            &mut false,
            &mut remaining_repeat_budget,
            &mut 0.0,
        );

        // The bail pushes exactly ONE whole fragment for parent_id at its
        // entry coordinates, then returns (page_in, cursor_in + height).
        let entry = geom
            .get(&parent_id)
            .expect("bail must emit a geometry entry for parent_id");
        // Hot-path-bind `.len()` so the assert message doesn't introduce a
        // failure-only region that codecov marks uncovered (see the P1c
        // assert-arg artifact note in CLAUDE.md / units migration memory).
        let n_frags = entry.fragments.len();
        assert_eq!(
            n_frags, 1,
            "bail emits a single whole fragment, got {n_frags}"
        );
        let frag = &entry.fragments[0];
        assert_eq!(frag.page_index, 0, "fragment stays on the entry page");
        let fx = frag.x.to_f32();
        assert_eq!(fx, 0.0, "fragment x == parent_x_in_body, got {fx}");
        let fy = frag.y.to_f32();
        assert_eq!(fy, 0.0, "fragment y == cursor_in, got {fy}");
        let fw = frag.width.to_f32();
        assert_eq!(fw, 600.0, "fragment width == parent_w, got {fw}");
        let fh = frag.height.to_f32();
        assert!(
            (fh - node_h).abs() < 1.0,
            "fragment height == node layout height: {fh} vs {node_h}"
        );

        assert_eq!(page_out, 0, "bail returns the entry page unchanged");
        assert!(
            (cursor_out - node_h).abs() < 1.0,
            "cursor advances by the node height: {cursor_out} vs {node_h}"
        );
        assert_eq!(remaining_repeat_budget, crate::MAX_SUBTREE_PAGE_FRAGMENTS);
    }

    #[test]
    fn table_at_depth_limit_uses_whole_fragment_fallback() {
        let doc = parse(
            r#"
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  table { margin: 0; border-spacing: 0; }
                  th, td { box-sizing: border-box; height: 20px; padding: 0; }
                </style></head><body>
                  <table id="t">
                    <thead><tr><th id="h">Header</th></tr></thead>
                    <tbody><tr><td>Body</td></tr></tbody>
                  </table>
                </body></html>
            "#,
            200.0,
        );
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let table_height = doc.get_node(table_id).unwrap().final_layout.size.height;
        let mut geometry = PaginationGeometryTable::new();
        let mut remaining_repeat_budget = crate::MAX_SUBTREE_PAGE_FRAGMENTS;

        let (page_out, cursor_out) = fragment_block_subtree(
            &mut geometry,
            Flow {
                doc: &doc,
                column_styles: None,
                used_page_names: None,
                running_store: None,
                pass: &Pass::default(),
                forced_before: None,
                foot_clearance: 0.0,
            },
            table_id,
            200.0,
            0.0,
            0,
            0.0,
            100.0,
            crate::MAX_DOM_DEPTH,
            false,
            &mut false,
            &mut remaining_repeat_budget,
            &mut 0.0,
        );

        assert_eq!(page_out, 0);
        assert!((cursor_out - table_height).abs() < 0.5);
        assert_eq!(geometry[&table_id].fragments.len(), 1);
        assert!(!geometry.contains_key(&header_id));
        assert_eq!(remaining_repeat_budget, crate::MAX_SUBTREE_PAGE_FRAGMENTS);
    }

    #[test]
    fn three_short_blocks_fit_one_page() {
        // Each block is 200px tall; page is 800px → all three fit on
        // page 0.
        let html = r#"
            <html><body>
              <div style="height: 200px"></div>
              <div style="height: 200px"></div>
              <div style="height: 200px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        // Phase 2.3 fix: body itself is recorded too, so total = 4
        // (body + 3 child divs). All on page 0.
        assert_eq!(table.len(), 4, "expected 4 entries, got {}", table.len());
        for (id, geom) in &table {
            assert_eq!(
                geom.fragments.len(),
                1,
                "node {id} should have a single fragment"
            );
            assert_eq!(geom.fragments[0].page_index, 0);
        }
    }

    /// A one-line paragraph whose line box holds an `<img>` and does not fit
    /// the rest of the page moves whole to the next page, like a text line.
    #[test]
    fn line_box_with_an_inline_image_that_does_not_fit_moves_to_the_next_page() {
        let html = r#"
            <html><body style="margin:0">
              <div style="height: 770px"></div>
              <p id="img" style="margin:0">S: <img width="100" height="50"> end</p>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        let id = find_by_id(&doc, "img").expect("paragraph");
        let frags = &table[&id].fragments;
        assert_eq!(frags.len(), 1, "{frags:?}");
        assert_eq!(frags[0].page_index, 1, "{frags:?}");
        assert!(
            frags[0].y.to_f32().abs() < 0.5,
            "starts at the page top: {frags:?}"
        );
    }

    /// An `<img>` in a paragraph of several lines has geometry on the page
    /// and at the offset of the line that holds it, also when the paragraph
    /// splits across pages.
    #[test]
    fn inline_image_in_a_split_paragraph_follows_its_line() {
        let html = r#"
            <html><body style="margin:0;font-size:20px;line-height:20px">
              <p id="p" style="margin:0;width:100px">aa bb cc dd ee ff gg hh ii jj kk ll mm <img id="im" width="10" height="20"> nn oo pp qq</p>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 50.0);
        let p = &table[&find_by_id(&doc, "p").expect("paragraph")].fragments;
        let im = &table[&find_by_id(&doc, "im").expect("image")].fragments;
        // Two 20px lines fit a 50px page, so the 5-line paragraph takes
        // three fragments: 2, 2, and a 1-line tail (widows relaxed at the
        // top of a page, where the minimums cannot be met).
        assert_eq!(p.len(), 3, "the paragraph splits: {p:?}");
        assert_eq!(im.len(), 1, "{im:?}");
        let frag = p
            .iter()
            .find(|f| f.page_index == im[0].page_index)
            .expect("the image is on a page the paragraph occupies");
        let (top, bottom) = (frag.y.to_f32(), (frag.y + frag.height).to_f32());
        let y = im[0].y.to_f32();
        assert!(
            y >= top - 0.5 && y + 20.0 <= bottom + 0.5,
            "image y {y} lies inside its fragment {top}..{bottom}"
        );
    }

    #[test]
    fn oversize_block_run_breaks_to_next_page() {
        // Block 1 is 600px, block 2 is 400px. Page strip is 800px.
        // Block 1 fits on page 0 (cursor 0 → 600). Block 2 starts at
        // y=600, would end at y=1000 > 800 → break to page 1.
        let html = r#"
            <html><body>
              <div style="height: 600px"></div>
              <div style="height: 400px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        // Phase 2.3 fix: body + 2 children = 3 entries.
        // Body fragment is page 0; children are 0, 1.
        assert_eq!(table.len(), 3);
        let pages: Vec<u32> = table.values().map(|g| g.fragments[0].page_index).collect();
        assert_eq!(
            pages,
            vec![0, 0, 1],
            "body page 0, first child page 0, second child page 1, got {pages:?}"
        );
    }

    /// Find the first element node carrying `id="<id>"`.
    fn find_by_id(doc: &blitz_dom::BaseDocument, id: &str) -> Option<usize> {
        fn walk(doc: &blitz_dom::BaseDocument, node_id: usize, target: &str) -> Option<usize> {
            let node = doc.get_node(node_id)?;
            if let Some(ed) = node.element_data()
                && let Some(attr_id) = ed.attrs().iter().find(|a| a.name.local.as_ref() == "id")
                && attr_id.value.as_str() == target
            {
                return Some(node_id);
            }
            for &child in &node.children {
                if let Some(found) = walk(doc, child, target) {
                    return Some(found);
                }
            }
            None
        }
        walk(doc, doc.root_element().id, id)
    }

    /// css-tables-3: "If a table owns multiple `display: table-header-group`
    /// boxes, only the first is treated as a header; the others are treated as
    /// if they had `display: table-row-group`."
    ///
    /// Supersedes `repeating_table_metadata_uses_computed_display_and_full_band`
    /// (PR #710), which asserted that `tbody.repeat` — the *second* header
    /// group here — also contributed a header cell.
    #[test]
    fn repeating_table_metadata_uses_only_the_first_header_group() {
        let doc = parse(
            r#"
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  table { border-spacing: 0 6px; width: 120px; }
                  th, td { box-sizing: border-box; height: 20px; padding: 0; }
                  tbody.repeat { display: table-header-group; }
                </style></head><body>
                  <table id="t">
                    <thead>
                      <tr><th id="h1">H1</th></tr>
                      <tr><th id="h1b">H1b</th></tr>
                    </thead>
                    <tbody class="repeat"><tr><td id="h2">H2</td></tr></tbody>
                    <tbody><tr><td id="b1">B1</td></tr></tbody>
                    <tfoot><tr><td id="foot">Foot</td></tr></tfoot>
                  </table>
                </body></html>
            "#,
            200.0,
        );
        let table_id = find_by_id(&doc, "t").unwrap();
        let h1 = find_by_id(&doc, "h1").unwrap();
        let h1b = find_by_id(&doc, "h1b").unwrap();
        let h2 = find_by_id(&doc, "h2").unwrap();
        let b1 = find_by_id(&doc, "b1").unwrap();
        let foot = find_by_id(&doc, "foot").unwrap();
        // The body now starts at the demoted second header group, not at `b1`.
        let body_y = doc.get_node(h2).unwrap().final_layout.location.y;

        let metadata = repeating_table_header(&doc, table_id, 1, None).unwrap();

        assert_eq!(metadata.table_id, table_id);
        // Only `<thead>`'s cells are headers; `tbody.repeat` is demoted.
        assert_eq!(metadata.header_cell_ids, vec![h1, h1b]);
        assert!(metadata.body_cell_ids.contains(&h2));
        assert!(metadata.body_cell_ids.contains(&b1));
        assert!(metadata.body_cell_ids.contains(&foot));
        assert!((metadata.body_origin_px - body_y).abs() < 0.5);
        assert!((metadata.band_height_px - body_y).abs() < 0.5);
    }

    #[test]
    fn repeating_table_metadata_honors_table_row_group_opt_out() {
        let doc = parse(
            r#"
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  table { border-spacing: 0; }
                  thead { display: table-row-group; }
                  th, td { height: 20px; padding: 0; }
                </style></head><body>
                  <table id="t">
                    <thead><tr><th>Header</th></tr></thead>
                    <tbody><tr><td>Body</td></tr></tbody>
                  </table>
                </body></html>
            "#,
            200.0,
        );
        let table_id = find_by_id(&doc, "t").unwrap();

        assert!(repeating_table_header(&doc, table_id, 1, None).is_none());
    }

    #[test]
    fn repeating_table_metadata_rejects_non_table_parent() {
        let doc = parse(
            r#"
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  .header { display: table-header-group; }
                  .row { display: table-row; }
                  .cell { display: table-cell; height: 20px; }
                </style></head><body>
                  <div id="not-table">
                    <div class="header"><div class="row"><div class="cell">Header</div></div></div>
                    <div class="row"><div class="cell">Body</div></div>
                  </div>
                </body></html>
            "#,
            200.0,
        );
        let parent_id = find_by_id(&doc, "not-table").unwrap();

        assert!(repeating_table_header(&doc, parent_id, 1, None).is_none());
    }

    #[test]
    fn repeating_table_metadata_honors_caller_depth() {
        let doc = parse(
            r#"
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  table { border-spacing: 0; }
                  th, td { height: 20px; padding: 0; }
                </style></head><body>
                  <table id="t">
                    <thead><tr><th>Header</th></tr></thead>
                    <tbody><tr><td>Body</td></tr></tbody>
                  </table>
                </body></html>
            "#,
            200.0,
        );
        let table_id = find_by_id(&doc, "t").unwrap();

        assert!(repeating_table_header(&doc, table_id, crate::MAX_DOM_DEPTH, None).is_none());
    }

    #[test]
    fn table_header_repeats_and_reserves_body_space() {
        let html = r#"
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { border-spacing: 0 6px; width: 100px; }
              th, td { box-sizing: border-box; height: 30px; padding: 0; }
            </style></head><body>
              <table id="table">
                <thead><tr><th id="header"><div id="header-block">HEADER</div></th></tr></thead>
                <tbody>
                  <tr><td id="row-1">1</td></tr>
                  <tr><td id="row-2">2</td></tr>
                  <tr><td id="row-3">3</td></tr>
                  <tr><td id="row-4">4</td></tr>
                  <tr><td id="row-5">5</td></tr>
                </tbody>
              </table>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "table").unwrap();
        let header_id = find_by_id(&doc, "header").unwrap();
        let header_block_id = find_by_id(&doc, "header-block").unwrap();
        let first_row_id = find_by_id(&doc, "row-1").unwrap();
        let last_row_id = find_by_id(&doc, "row-5").unwrap();
        let body_origin = doc.get_node(first_row_id).unwrap().final_layout.location.y;

        let geometry = run_pass(&mut doc, 90.0);
        let table_pages: Vec<u32> = geometry[&table_id]
            .fragments
            .iter()
            .map(|f| f.page_index)
            .collect();
        let header = &geometry[&header_id];
        let header_pages: Vec<u32> = header.fragments.iter().map(|f| f.page_index).collect();
        let header_block = &geometry[&header_block_id];

        assert!(
            header.is_repeat,
            "header geometry must represent full redraws"
        );
        assert_eq!(
            header_pages, table_pages,
            "header must cover every table page"
        );
        assert!(
            header_block.is_repeat,
            "header descendants must redraw in full"
        );
        assert_eq!(
            header_block
                .fragments
                .iter()
                .map(|f| f.page_index)
                .collect::<Vec<_>>(),
            table_pages,
            "header descendants must cover every table page"
        );
        assert!(geometry[&last_row_id].fragments.last().unwrap().page_index > 0);

        for (&node_id, node_geometry) in &geometry {
            if node_id == header_id || node_id == table_id {
                continue;
            }
            for fragment in node_geometry.fragments.iter().filter(|f| f.page_index > 0) {
                if [
                    first_row_id,
                    find_by_id(&doc, "row-2").unwrap(),
                    find_by_id(&doc, "row-3").unwrap(),
                    find_by_id(&doc, "row-4").unwrap(),
                    last_row_id,
                ]
                .contains(&node_id)
                {
                    let table_top = geometry[&table_id]
                        .fragments
                        .iter()
                        .find(|f| f.page_index == fragment.page_index)
                        .unwrap()
                        .y
                        .to_f32();
                    assert!(
                        fragment.y.to_f32() + 0.5 >= table_top + body_origin,
                        "body node {node_id} overlaps repeated header: {fragment:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn table_header_row_group_opt_out_does_not_repeat() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              thead { display: table-row-group; }
              th, td { box-sizing: border-box; height: 30px; padding: 0; }
            </style></head><body>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead><tbody>
                <tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr>
                <tr><td>4</td></tr><tr><td id="last">5</td></tr>
              </tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let last_id = find_by_id(&doc, "last").unwrap();

        let geometry = run_pass(&mut doc, 90.0);
        let header = &geometry[&header_id];

        assert!(!header.is_repeat);
        assert_eq!(header.fragments.len(), 1);
        assert!(geometry[&table_id].fragments.len() > 1);
        assert!(geometry[&last_id].fragments.last().unwrap().page_index > 0);
    }

    #[test]
    fn repeated_header_reservation_can_increase_page_count() {
        fn pages(extra_css: &str) -> u32 {
            let html = r#"<!doctype html>
                <html><head><style>
                  html, body { margin: 0; padding: 0; }
                  table { margin: 0; border-spacing: 0; width: 100px; }
                  th, td { box-sizing: border-box; height: 30px; padding: 0; }
                  /* extra css */
                </style></head><body>
                  <table><thead><tr><th>Header</th></tr></thead><tbody>
                    <tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr>
                    <tr><td>4</td></tr><tr><td>5</td></tr>
                  </tbody></table>
                </body></html>"#
                .replace("/* extra css */", extra_css);
            let mut doc = parse(&html, 200.0);
            implied_page_count(&run_pass(&mut doc, 90.0))
        }

        let repeated = pages("");
        let opt_out = pages("thead { display: table-row-group; }");

        assert_eq!(repeated, 3);
        assert_eq!(opt_out, 2);
        assert!(repeated > opt_out);
    }

    /// Supersedes `non_thead_and_multiple_header_groups_repeat_full_band`
    /// (PR #710), which asserted that *every* `table-header-group` repeats.
    /// css-tables-3 gives that status to the first one only; `tbody.repeat`
    /// below is the second and must paginate as ordinary body content.
    #[test]
    fn only_the_first_header_group_repeats_per_css_tables_3() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; height: 20px; padding: 0; }
              tbody.repeat { display: table-header-group; }
            </style></head><body>
              <table id="t">
                <thead><tr><th id="h1">H1</th></tr></thead>
                <tbody class="repeat"><tr><td id="h2">H2</td></tr></tbody>
                <tbody>
                  <tr><td id="b1">1</td></tr><tr><td>2</td></tr>
                  <tr><td>3</td></tr><tr><td id="b4">4</td></tr><tr><td>5</td></tr>
                </tbody>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let h1_id = find_by_id(&doc, "h1").unwrap();
        let h2_id = find_by_id(&doc, "h2").unwrap();
        let b4_id = find_by_id(&doc, "b4").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let table_pages = geometry[&table_id]
            .fragments
            .iter()
            .map(|fragment| fragment.page_index)
            .collect::<Vec<_>>();

        assert!(geometry[&h1_id].is_repeat, "the first header group repeats");
        assert_eq!(
            geometry[&h1_id]
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .collect::<Vec<_>>(),
            table_pages
        );
        assert!(
            !geometry[&h2_id].is_repeat,
            "the second header group is demoted to table-row-group and must not repeat"
        );
        let continuation = geometry[&b4_id]
            .fragments
            .iter()
            .filter(|fragment| fragment.page_index > table_pages[0])
            .collect::<Vec<_>>();
        assert!(!continuation.is_empty());
        // The reserved band is one 20px header row (`<thead>`), not two:
        // the demoted `tbody.repeat` paginates as body content now, so
        // continuation rows clear 20px rather than the former 40px.
        assert!(
            continuation
                .iter()
                .all(|fragment| fragment.y.to_f32() >= 19.5)
        );
    }

    #[test]
    fn multiple_rows_in_one_header_group_repeat_full_band() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; height: 20px; padding: 0; }
            </style></head><body>
              <table id="t">
                <thead>
                  <tr><th id="h1">H1</th></tr>
                  <tr><th id="h2">H2</th></tr>
                </thead>
                <tbody>
                  <tr><td>1</td></tr><tr><td>2</td></tr>
                  <tr><td>3</td></tr><tr><td id="b4">4</td></tr>
                </tbody>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let h1_id = find_by_id(&doc, "h1").unwrap();
        let h2_id = find_by_id(&doc, "h2").unwrap();
        let b4_id = find_by_id(&doc, "b4").unwrap();

        let geometry = run_pass(&mut doc, 80.0);
        let table_pages = geometry[&table_id]
            .fragments
            .iter()
            .map(|fragment| fragment.page_index)
            .collect::<Vec<_>>();

        for header_id in [h1_id, h2_id] {
            assert!(geometry[&header_id].is_repeat);
            assert_eq!(
                geometry[&header_id]
                    .fragments
                    .iter()
                    .map(|fragment| fragment.page_index)
                    .collect::<Vec<_>>(),
                table_pages
            );
        }
        let continuation = geometry[&b4_id]
            .fragments
            .iter()
            .filter(|fragment| fragment.page_index > table_pages[0])
            .collect::<Vec<_>>();
        assert!(!continuation.is_empty());
        assert!(
            continuation
                .iter()
                .all(|fragment| fragment.y.to_f32() >= 39.5)
        );
    }

    #[test]
    fn single_page_table_does_not_create_repeat_geometry() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; height: 20px; padding: 0; }
            </style></head><body>
              <table><thead><tr><th id="h">Header</th></tr></thead>
                <tbody><tr><td>Body</td></tr></tbody>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let header_id = find_by_id(&doc, "h").unwrap();

        let geometry = run_pass(&mut doc, 800.0);
        let header = &geometry[&header_id];

        assert!(!header.is_repeat);
        assert_eq!(header.fragments.len(), 1);
    }

    /// A table of 20 px rows whose header group holds `head` (the rows
    /// `note`, which skips the first page, and `h`, in some order) and
    /// `body_rows` body rows `b1`, `b2`, ...
    fn header_table(attrs: &str, head: &str, body_rows: usize) -> String {
        let rows = (1..=body_rows)
            .map(|n| format!(r#"<tr><td id="b{n}">{n}</td></tr>"#))
            .collect::<String>();
        format!(r#"<table id="t"{attrs}><thead>{head}</thead><tbody>{rows}</tbody></table>"#)
    }

    /// The note row that skips the first page, above the heading row.
    const NOTE_THEN_H: &str = r#"<tr style="-fulgur-header-repeat: skip-first"><th id="note">continued</th></tr><tr><th id="h">H</th></tr>"#;

    /// `body` laid out with the document's break styles on pages
    /// `page_height` px tall, with 20 px cells, captions, and `div`s, its
    /// captions moved out of their tables as the engine moves them.
    fn header_pass(
        body: &str,
        page_height: f32,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        use crate::blitz_adapter::DomPass as _;
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              th, td {{ box-sizing: border-box; height: 20px; padding: 0; }}
              caption {{ height: 20px; padding: 0; }}
              .s {{ height: 20px; }}
            </style></head><body>{body}</body></html>"#
        );
        let mut doc = blitz_adapter::parse(&html, 200.0, &[]);
        blitz_adapter::CaptionRestructurePass
            .apply(&mut doc, &blitz_adapter::PassContext { font_data: &[] });
        blitz_adapter::resolve(&mut doc);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), page_height.as_px(), &styles);
        (doc, geometry)
    }

    /// The skip-first table of `body_rows` rows, then a 20 px block, on 80 px
    /// pages.
    fn skip_first_header_pass(
        body_rows: usize,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let table = header_table("", NOTE_THEN_H, body_rows);
        header_pass(&format!(r#"{table}<div id="after" class="s"></div>"#), 80.0)
    }

    /// `(page, y)` of each fragment of `id`.
    fn placements(
        doc: &blitz_dom::BaseDocument,
        geometry: &PaginationGeometryTable,
        id: &str,
    ) -> Vec<(u32, f32)> {
        let node_id = find_by_id(doc, id).unwrap();
        geometry.get(&node_id).map_or_else(Vec::new, |g| {
            g.fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32()))
                .collect()
        })
    }

    /// The rows that skip a table's first page take no room there: a block
    /// kept whole around a one-page table with such a row stays where it
    /// fits once the row is left off (`extent`). Its layout height still
    /// counted the row, 40 + 60 px on an 80 px page, so it moved to the next
    /// page for nothing.
    #[test]
    fn a_kept_block_around_a_one_page_skip_first_table_stays_when_it_fits_after_the_lift() {
        let table = header_table("", NOTE_THEN_H, 1);
        let (doc, geometry) = header_pass(
            &format!(
                r#"<div class="s" style="height: 40px"></div><div id="k" style="break-inside: avoid">{table}</div>"#
            ),
            80.0,
        );
        assert_eq!(placements(&doc, &geometry, "k"), [(0, 40.0)]);
    }

    /// A header row with `-fulgur-header-repeat: skip-first` is left off the
    /// table's first page and placed above the other header rows on each page
    /// the header repeats on. On the first page the rows below it lift into
    /// its place, and the body fills the room it leaves: a 20 px band there
    /// holds three 20 px rows on an 80 px page, the 40 px band of the later
    /// pages two.
    #[test]
    fn a_header_row_that_skips_the_first_page_is_placed_where_the_header_repeats() {
        let (doc, geometry) = skip_first_header_pass(6);

        assert_eq!(placements(&doc, &geometry, "note"), [(1, 0.0), (2, 0.0)]);
        assert_eq!(
            placements(&doc, &geometry, "h"),
            [(0, 0.0), (1, 20.0), (2, 20.0)]
        );
        assert!(geometry[&find_by_id(&doc, "note").unwrap()].is_repeat);
        assert_eq!(placements(&doc, &geometry, "b1"), [(0, 20.0)]);
        assert_eq!(placements(&doc, &geometry, "b3"), [(0, 60.0)]);
        assert_eq!(placements(&doc, &geometry, "b4"), [(1, 40.0)]);
        assert_eq!(placements(&doc, &geometry, "b6"), [(2, 40.0)]);
        let table = &geometry[&find_by_id(&doc, "t").unwrap()].fragments[0];
        assert_eq!((table.page_index, table.y.to_f32()), (0, 0.0));
        assert_eq!(placements(&doc, &geometry, "after"), [(2, 60.0)]);
    }

    /// A table on one page never places a `skip-first` header row: the rows
    /// below it lift into its place, the block after the table follows the
    /// last row, and the table's frame is that much shorter than its layout.
    #[test]
    fn a_header_row_that_skips_the_first_page_is_not_placed_on_a_one_page_table() {
        let (doc, geometry) = skip_first_header_pass(1);

        assert!(placements(&doc, &geometry, "note").is_empty());
        assert_eq!(placements(&doc, &geometry, "h"), [(0, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "b1"), [(0, 20.0)]);
        assert_eq!(placements(&doc, &geometry, "after"), [(0, 40.0)]);
        let table = &geometry[&find_by_id(&doc, "t").unwrap()];
        assert_eq!(table.unsplit_lift_px(), 20.0);
    }

    /// A one-page table nested in blocks takes what follows it up with it,
    /// in its own block and in each block around it: the siblings are
    /// placed by their layout offsets, which count the skipped row. Each
    /// block around it ends that much higher, at its own bottom (padding
    /// included), and paints its box that much shorter; a block split over
    /// pages carries its heights in its fragments.
    #[test]
    fn a_nested_one_page_table_takes_what_follows_it_up_by_the_skipped_row() {
        let table = header_table("", NOTE_THEN_H, 1);
        let (doc, geometry) = header_pass(
            &format!(
                r#"<div id="outer" style="padding-bottom: 10px"><div id="inner">{table}<div id="in" class="s"></div></div><div id="out" class="s"></div></div><div id="after" class="s"></div>"#
            ),
            200.0,
        );

        assert!(placements(&doc, &geometry, "note").is_empty());
        assert_eq!(placements(&doc, &geometry, "b1"), [(0, 20.0)]);
        assert_eq!(placements(&doc, &geometry, "in"), [(0, 40.0)]);
        assert_eq!(placements(&doc, &geometry, "out"), [(0, 60.0)]);
        // The outer block ends at its padding's bottom, less the row.
        assert_eq!(placements(&doc, &geometry, "after"), [(0, 90.0)]);
        for block in ["inner", "outer"] {
            let block = &geometry[&find_by_id(&doc, block).unwrap()];
            assert_eq!(block.unsplit_lift_px(), 20.0, "{block:?}");
        }
        let fragment = geometry[&find_by_id(&doc, "outer").unwrap()].fragments[0].clone();
        let split = PaginationGeometry {
            fragments: vec![fragment.clone(), fragment],
            lift_px: 20.0,
            ..Default::default()
        };
        assert_eq!(split.unsplit_lift_px(), 0.0);
    }

    /// A table kept whole (`break-inside: avoid`), on itself or on a block
    /// around it, at the body level or nested, that does not fit the rest of
    /// the page moves whole to the next page, where it starts without its
    /// skipped row, as it would move without one.
    #[test]
    fn a_table_kept_whole_with_a_skipped_header_row_moves_whole() {
        let kept = r#" style="break-inside: avoid""#;
        let spacer = r#"<div style="height: 100px"></div>"#;
        for body in [
            format!("{spacer}{}", header_table(kept, NOTE_THEN_H, 5)),
            format!(
                "{spacer}<div{kept}>{}</div>",
                header_table("", NOTE_THEN_H, 5)
            ),
            format!("<div>{spacer}{}</div>", header_table(kept, NOTE_THEN_H, 5)),
        ] {
            let (doc, geometry) = header_pass(&body, 200.0);

            assert_eq!(placements(&doc, &geometry, "h"), [(1, 0.0)], "{body}");
            assert_eq!(placements(&doc, &geometry, "b5"), [(1, 100.0)], "{body}");
            assert!(placements(&doc, &geometry, "note").is_empty(), "{body}");
        }
    }

    /// A table's break properties are its caption's too, since the caption
    /// lies outside the table's box, in the box around both. A table kept
    /// whole (`break-inside: avoid`) that does not fit the rest of the page
    /// moves to the next page with its caption, above or below it, at the
    /// body level or nested, with a skipped header row or without; one that
    /// fits a page but not with its caption, or that is taller than a page,
    /// splits where it stands, under its caption. A forced break before the
    /// table goes before its caption.
    #[test]
    fn a_table_kept_whole_moves_with_its_caption() {
        // A table of `attrs` with the caption `c` of `caption` style, the
        // header rows `head`, and `body_rows` 20 px rows.
        let table = |attrs: &str, caption: &str, head: &str, body_rows: usize| {
            header_table(attrs, head, body_rows).replacen(
                "<thead>",
                &format!(r#"<caption id="c" style="{caption}">C</caption><thead>"#),
                1,
            )
        };
        let kept = r#" style="break-inside: avoid""#;
        let h = r#"<tr><th id="h">H</th></tr>"#;
        let spacer = r#"<div style="height: 100px"></div>"#;
        let at = |body: &str, ids: [&str; 2]| {
            let (doc, geometry) = header_pass(body, 200.0);
            ids.map(|id| placements(&doc, &geometry, id))
        };
        for body in [
            format!("{spacer}{}", table(kept, "", h, 5)),
            format!("<div>{spacer}{}</div>", table(kept, "", h, 5)),
            format!("{spacer}{}", table(kept, "", NOTE_THEN_H, 5)),
            format!("<div>{spacer}{}</div>", table(kept, "", NOTE_THEN_H, 5)),
        ] {
            assert_eq!(
                at(&body, ["c", "h"]),
                [vec![(1, 0.0)], vec![(1, 20.0)]],
                "{body}"
            );
        }

        let bottom = format!("{spacer}{}", table(kept, "caption-side: bottom", h, 5));
        assert_eq!(at(&bottom, ["h", "c"]), [vec![(1, 0.0)], vec![(1, 120.0)]]);

        // A 180 px table under a 40 px caption: together taller than a page.
        let window = format!("{spacer}{}", table(kept, "height: 40px", h, 8));
        assert_eq!(
            at(&window, ["c", "h"]),
            [vec![(0, 100.0)], vec![(0, 140.0), (1, 0.0)]]
        );

        let long = format!("{spacer}{}", table(kept, "", h, 10));
        assert_eq!(
            at(&long, ["c", "h"]),
            [vec![(0, 100.0)], vec![(0, 120.0), (1, 0.0)]]
        );

        let forced = format!(
            r#"<div><div class="s"></div>{}</div>"#,
            table(r#" style="break-before: page""#, "", h, 1)
        );
        assert_eq!(at(&forced, ["c", "h"]), [vec![(1, 0.0)], vec![(1, 20.0)]]);
    }

    /// The walk reads a captioned table's break properties for the box
    /// around the table and its caption, which stands for the table wrapper
    /// box: the side table keeps them on the table, none on that box, and a
    /// table kept whole still moves with its caption, a forced break before
    /// it still goes before the caption
    /// (`a_table_kept_whole_moves_with_its_caption`).
    #[test]
    fn a_caption_moves_with_a_table_kept_whole_without_moved_properties() {
        use crate::blitz_adapter::DomPass as _;
        let html = |style: &str| {
            format!(
                r#"<!doctype html><html><head><style>
                html, body {{ margin: 0; padding: 0; }}
                table {{ margin: 0; border-spacing: 0; width: 100px; }}
                th, td {{ box-sizing: border-box; height: 20px; padding: 0; }}
                caption {{ height: 20px; padding: 0; }}
                </style></head><body><div style="height: 100px"></div>
                <table id="t" style="{style}"><caption id="c">C</caption><thead><tr><th id="h">H</th></tr></thead>
                <tbody>{}</tbody></table></body></html>"#,
                (1..=5)
                    .map(|n| format!(r#"<tr><td id="b{n}">{n}</td></tr>"#))
                    .collect::<String>()
            )
        };
        for (style, at) in [
            ("break-inside: avoid", [(1, 0.0), (1, 20.0)]),
            ("break-before: page", [(1, 0.0), (1, 20.0)]),
        ] {
            let mut doc = blitz_adapter::parse(&html(style), 200.0, &[]);
            blitz_adapter::CaptionRestructurePass
                .apply(&mut doc, &blitz_adapter::PassContext { font_data: &[] });
            blitz_adapter::resolve(&mut doc);
            let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
            let table = find_by_id(&doc, "t").unwrap();
            let wrapper = doc.get_node(table).and_then(|node| node.parent).unwrap();
            let breaks = |id: usize| {
                styles.get(&id).map_or((false, false), |props| {
                    (props.break_inside.is_some(), props.break_before.is_some())
                })
            };
            assert_eq!(breaks(wrapper), (false, false), "{style}");
            assert_ne!(breaks(table), (false, false), "{style}");
            let geometry =
                super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &styles);
            assert_eq!(
                ["c", "h"].map(|id| placements(&doc, &geometry, id)[0]),
                at,
                "{style}"
            );
        }
    }

    /// A `skip-first` row below a header row that prints is placed on the
    /// first page too: leaving it out would leave a gap.
    #[test]
    fn a_skip_first_row_below_a_row_that_prints_is_placed_on_the_first_page() {
        let head = r#"<tr><th id="h">H</th></tr><tr style="-fulgur-header-repeat: skip-first"><th id="note">continued</th></tr>"#;
        let (doc, geometry) = header_pass(&header_table("", head, 1), 200.0);

        assert_eq!(placements(&doc, &geometry, "h"), [(0, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "note"), [(0, 20.0)]);
        assert_eq!(placements(&doc, &geometry, "b1"), [(0, 40.0)]);
    }

    /// A table that starts mid-page lifts its first page's rows from where it
    /// starts, and its later pages carry the whole band.
    #[test]
    fn a_table_starting_mid_page_skips_the_row_on_its_first_page() {
        let table = header_table("", NOTE_THEN_H, 6);
        let (doc, geometry) =
            header_pass(&format!(r#"<div style="height: 30px"></div>{table}"#), 80.0);

        assert_eq!(
            placements(&doc, &geometry, "h"),
            [(0, 30.0), (1, 20.0), (2, 20.0), (3, 20.0)]
        );
        assert_eq!(
            placements(&doc, &geometry, "note"),
            [(1, 0.0), (2, 0.0), (3, 0.0)]
        );
        assert_eq!(placements(&doc, &geometry, "b1"), [(0, 50.0)]);
        assert_eq!(placements(&doc, &geometry, "b2"), [(1, 40.0)]);
        assert_eq!(placements(&doc, &geometry, "b6"), [(3, 40.0)]);
    }

    #[test]
    fn over_page_height_table_header_uses_generic_bounded_fallback() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 120px; }
              td { height: 20px; }
            </style></head><body>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead>
                <tbody><tr><td id="body-cell">Body</td></tr></tbody>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let body_id = find_by_id(&doc, "body-cell").unwrap();

        let geometry = run_pass(&mut doc, 100.0);

        assert!(!geometry[&header_id].is_repeat);
        assert_eq!(implied_page_count(&geometry), 2);
        assert_eq!(geometry[&table_id].fragments.len(), 2);
        assert!(!geometry[&body_id].fragments.is_empty());
        assert_eq!(geometry[&body_id].fragments.last().unwrap().page_index, 1);
    }

    #[test]
    fn nested_multipage_table_keeps_inner_repeat_geometry() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; }
              #outer { width: 300px; }
              #outer > thead > tr > th { box-sizing: border-box; height: 20px; padding: 0; }
              #outer > tbody > tr > td { box-sizing: border-box; height: 180px; padding: 0; }
              #inner { width: 300px; }
              #inner th, #inner td { box-sizing: border-box; height: 20px; padding: 0; }
            </style></head><body>
              <table id="outer">
                <thead><tr><th id="outer-h">Outer</th></tr></thead>
                <tbody><tr><td><table id="inner">
                  <thead><tr><th id="inner-h">Inner</th></tr></thead><tbody>
                    <tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr>
                    <tr><td>4</td></tr><tr><td>5</td></tr><tr><td>6</td></tr>
                    <tr><td>7</td></tr><tr><td>8</td></tr>
                  </tbody>
                </table></td></tr></tbody>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 300.0);
        let outer_id = find_by_id(&doc, "outer").unwrap();
        let outer_header_id = find_by_id(&doc, "outer-h").unwrap();
        let inner_id = find_by_id(&doc, "inner").unwrap();
        let inner_header_id = find_by_id(&doc, "inner-h").unwrap();

        let geometry = run_pass(&mut doc, 80.0);
        let outer_pages = geometry[&outer_id]
            .fragments
            .iter()
            .map(|fragment| fragment.page_index)
            .collect::<Vec<_>>();
        let inner_pages = geometry[&inner_id]
            .fragments
            .iter()
            .map(|fragment| fragment.page_index)
            .collect::<Vec<_>>();

        assert!(outer_pages.len() > 1);
        assert!(inner_pages.len() > 1);
        assert!(geometry[&outer_header_id].is_repeat);
        assert!(geometry[&inner_header_id].is_repeat);
        assert_eq!(
            geometry[&outer_header_id]
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .collect::<Vec<_>>(),
            outer_pages
        );
        assert_eq!(
            geometry[&inner_header_id]
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .collect::<Vec<_>>(),
            inner_pages
        );

        for (inner_table, inner_header) in geometry[&inner_id]
            .fragments
            .iter()
            .zip(&geometry[&inner_header_id].fragments)
        {
            let outer_table = geometry[&outer_id]
                .fragments
                .iter()
                .find(|fragment| fragment.page_index == inner_table.page_index)
                .unwrap();
            let outer_header = geometry[&outer_header_id]
                .fragments
                .iter()
                .find(|fragment| fragment.page_index == inner_table.page_index)
                .unwrap();
            assert!(
                (outer_header.y.to_f32() - outer_table.y.to_f32()).abs() < 0.5,
                "outer header must stay at the table top"
            );
            assert!(
                (inner_table.y.to_f32() - outer_table.y.to_f32() - 20.0).abs() < 0.5,
                "inner table must stay below the outer header band"
            );
            assert!(
                (inner_header.y.to_f32() - inner_table.y.to_f32()).abs() < 0.5,
                "inner header must stay at the inner table top"
            );
        }
    }

    #[test]
    fn table_cells_in_one_row_co_split_from_same_cursor() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; table-layout: fixed; width: 200px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 20px; }
              td { height: 120px; }
              #left > div { height: 30px; }
              #right > div { height: 20px; }
            </style></head><body>
              <table><thead><tr><th colspan="2">Header</th></tr></thead><tbody><tr>
                <td id="left"><div>1</div><div>2</div><div>3</div><div>4</div></td>
                <td id="right"><div>R</div></td>
              </tr></tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let left_id = find_by_id(&doc, "left").unwrap();
        let right_id = find_by_id(&doc, "right").unwrap();

        let geometry = run_pass(&mut doc, 70.0);
        let left = &geometry[&left_id].fragments;
        let right = &geometry[&right_id].fragments;

        assert!(left.len() > 1);
        assert_eq!(right[0].page_index, left[0].page_index);
        assert!((right[0].y.to_f32() - left[0].y.to_f32()).abs() < 0.5);
    }

    /// The cells of a row that splits all start on the row's first page at
    /// its top, without an explicit height to make them equal: the second
    /// cell, one 30 px paragraph beside the first's eight, is placed beside
    /// the first paragraph, not after the first cell ends three pages on.
    /// It is stretched to the row's height, so it is cut at each page foot
    /// and continues, empty, to the row's end.
    #[test]
    fn the_cells_of_a_row_split_at_the_page_foot_start_side_by_side() {
        let paragraphs = (1..=8).map(|n| format!("<p>{n}</p>")).collect::<String>();
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 200px; }}
              td {{ padding: 0; }}
              p {{ margin: 0; height: 30px; }}
            </style></head><body>
              <table><tbody><tr>
                <td id="left">{paragraphs}</td>
                <td id="right"><p id="short">S</p></td>
              </tr><tr><td id="after"><div style="height: 10px"></div></td><td></td></tr>
              </tbody></table>
            </body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let geometry = run_pass(&mut doc, 70.0);

        let left = fragments_of(&doc, &geometry, "left");
        assert_eq!(left.first().map(|f| f.0), Some(0));
        assert_eq!(left.last().map(|f| f.0), Some(3), "{left:?}");
        assert_eq!(fragments_of(&doc, &geometry, "short"), [(0, 0.0, 30.0)]);
        assert_eq!(
            fragments_of(&doc, &geometry, "right"),
            [
                (0, 0.0, 70.0),
                (1, 0.0, 70.0),
                (2, 0.0, 70.0),
                (3, 0.0, 60.0)
            ],
            "the stretched cell is cut at each foot and runs to the row's end"
        );
        assert_eq!(placements(&doc, &geometry, "after"), [(3, 60.0)]);
    }

    /// A flex item stretched to its line's height by a taller neighbour,
    /// whose own content fits the page, is cut at the page foot and goes on
    /// as an empty fragment to the line's end, rather than run past the
    /// foot as one fragment of the line's height. The line is 207 px on
    /// 194 px pages: its first item splits, its 112 px block moving to the
    /// next page, and the second item's paragraph fits the first.
    #[test]
    fn a_stretched_flex_item_whose_content_fits_is_cut_at_the_page_foot() {
        let html = r#"<!doctype html><html><head><style>
            html, body { margin: 0; padding: 0; font-family: 'Noto Sans'; }
            p { margin: 0; }
            </style></head><body><div id="row" style="display: flex">
            <div id="tall" style="flex: 1 1 0"><div><div style="height: 91px"></div>
            <div id="block" style="border: 2px solid; height: 112px"></div></div></div>
            <div id="short" style="flex: 1 1 0"><p id="lines" style="padding: 2px;
            border: 4px solid; line-height: 20px">1<br>2<br>3<br>4<br>5</p></div>
            </div></body></html>"#;
        let fonts = [crate::test_fonts::noto_sans_regular_ttf()];
        let (mut doc, _, _) =
            blitz_adapter::parse_html_with_local_resources(html, 200.0, 10000, &fonts, false, None);
        blitz_adapter::resolve(&mut doc);
        let geometry = run_pass(doc.deref_mut(), 194.0);

        assert_eq!(placements(&doc, &geometry, "block"), [(1, 0.0)]);
        assert_eq!(fragments_of(&doc, &geometry, "lines"), [(0, 0.0, 112.0)]);
        assert_eq!(
            fragments_of(&doc, &geometry, "short"),
            [(0, 0.0, 194.0), (1, 0.0, 116.0)],
            "cut at the foot, then on to the line's end"
        );
        assert_eq!(
            fragments_of(&doc, &geometry, "row"),
            [(0, 0.0, 194.0), (1, 0.0, 116.0)]
        );
    }

    /// `body` laid out with its break styles on 100 px pages, its table
    /// cells 20 px tall.
    fn row_break_pass(body: &str) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              td {{ box-sizing: border-box; height: 20px; padding: 0; }}
              .s {{ height: 50px; }}
            </style></head><body>{body}</body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
        (doc, geometry)
    }

    /// A table row's own `break-before` and `break-after`, and its row
    /// group's at its first or last row, are the opportunity between it and
    /// the row next to it: the row has no box of its own in the layout,
    /// whose cells the walk meets.
    #[test]
    fn a_rows_and_its_groups_breaks_are_the_opportunities_between_rows() {
        let at = |body: &str| {
            let (doc, geometry) = row_break_pass(body);
            ["a", "b", "c"].map(|id| placements(&doc, &geometry, id))
        };
        let rows = |b: &str, c: &str| {
            format!(
                r#"<table><tbody><tr><td id="a">a</td></tr><tr{b}><td id="b">b</td></tr>
                <tr{c}><td id="c">c</td></tr></tbody></table>"#
            )
        };
        assert_eq!(
            at(&rows(r#" style="break-before: page""#, "")),
            [vec![(0, 0.0)], vec![(1, 0.0)], vec![(1, 20.0)]]
        );
        assert_eq!(
            at(&rows(r#" style="break-after: page""#, "")),
            [vec![(0, 0.0)], vec![(0, 20.0)], vec![(1, 0.0)]]
        );
        let groups = r#"<table><tbody><tr><td id="a">a</td></tr><tr><td id="b">b</td></tr></tbody>
            <tbody style="break-before: page"><tr><td id="c">c</td></tr></tbody></table>"#;
        assert_eq!(
            at(groups),
            [vec![(0, 0.0)], vec![(0, 20.0)], vec![(1, 0.0)]]
        );
    }

    /// A row with `break-after: avoid` before a row that does not fit the
    /// rest of the page moves to the next page with it, as the boxes of a
    /// run in block flow do; so does a flex line before the line it avoids
    /// breaking from.
    #[test]
    fn a_row_or_a_flex_line_moves_with_the_line_it_avoids_breaking_from() {
        let rows = r#"<div class="s"></div><table><tbody><tr><td id="a">a</td></tr>
            <tr style="break-after: avoid"><td id="b">b</td></tr><tr><td id="c">c</td></tr>
            </tbody></table>"#;
        let (doc, geometry) = row_break_pass(rows);
        assert_eq!(
            ["a", "b", "c"].map(|id| placements(&doc, &geometry, id)),
            [vec![(0, 50.0)], vec![(1, 0.0)], vec![(1, 20.0)]]
        );

        let lines = r#"<div class="s"></div><div style="display: flex; flex-wrap: wrap">
            <div id="a" style="width: 100%; height: 20px"></div>
            <div id="b" style="width: 100%; height: 20px; break-after: avoid"></div>
            <div id="c" style="width: 100%; height: 20px"></div></div>"#;
        let (doc, geometry) = row_break_pass(lines);
        assert_eq!(
            ["a", "b", "c"].map(|id| placements(&doc, &geometry, id)),
            [vec![(0, 50.0)], vec![(1, 0.0)], vec![(1, 20.0)]]
        );
    }

    /// A table's last row moves to the next page with the block after the
    /// table that it avoids breaking from, which does not fit the rest of
    /// the page: the walk enters the table for the run, as it enters a
    /// block, and breaks before the row.
    #[test]
    fn a_tables_last_row_moves_with_the_block_it_avoids_breaking_from() {
        let body = r#"<div class="s"></div><table><tbody><tr><td id="a">a</td></tr>
            <tr><td id="b">b</td></tr></tbody></table>
            <div id="c" style="height: 20px; break-before: avoid"></div>"#;
        let (doc, geometry) = row_break_pass(body);
        assert_eq!(
            ["a", "b", "c"].map(|id| placements(&doc, &geometry, id)),
            [vec![(0, 50.0)], vec![(1, 0.0)], vec![(1, 20.0)]]
        );
    }

    /// A forced break between the rows a cell spans is taken below the
    /// spanning cell's content (`s`, one 20 px line).
    /// One that the content crosses (a 30 px block over 20 px rows) is not,
    /// as the walk does not cut a spanning cell's content at a forced break.
    #[test]
    fn a_forced_break_between_the_rows_a_cell_spans_is_taken_below_its_content() {
        let at = |content: &str| {
            let body = format!(
                r#"<table><tbody><tr><td id="a">a</td><td rowspan="2">{content}</td></tr>
                <tr style="break-before: page"><td id="b">b</td></tr>
                <tr style="break-before: page"><td id="c">c</td></tr></tbody></table>"#
            );
            let (doc, geometry) = row_break_pass(&body);
            ["a", "b", "c"].map(|id| placements(&doc, &geometry, id))
        };
        assert_eq!(
            [at("s"), at(r#"<div style="height: 30px"></div>"#)],
            [
                [vec![(0, 0.0)], vec![(1, 0.0)], vec![(2, 0.0)]],
                [vec![(0, 0.0)], vec![(0, 20.0)], vec![(1, 0.0)]],
            ]
        );
    }

    /// A table after a `spacer` px block, on 100 px pages, under the table
    /// style `style`, of one row group: a cell `s` of `span` 20 px lines
    /// that spans a row for each of `rows`, beside a cell `a`, `b`, ... of
    /// that many 20 px lines. Cells take `orphans: 1; widows: 1`, as text
    /// in a cell splits anywhere.
    fn span_group_pass(
        style: &str,
        rows: &[usize],
        span: usize,
        spacer: u32,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let lines = |n: usize, text: &str| vec![text; n].join("<br>");
        let cells: Vec<String> = rows
            .iter()
            .zip(["a", "b", "c", "d"])
            .map(|(&n, id)| format!(r#"<td id="{id}">{}</td>"#, lines(n, id)))
            .collect();
        let body = cells
            .iter()
            .enumerate()
            .map(|(k, cell)| {
                let s = (k == 0).then(|| {
                    format!(
                        r#"<td id="s" rowspan="{}">{}</td>"#,
                        rows.len(),
                        lines(span, "s")
                    )
                });
                format!("<tr>{}{cell}</tr>", s.unwrap_or_default())
            })
            .collect::<String>();
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              td {{ padding: 0; line-height: 20px; orphans: 1; widows: 1; vertical-align: top; }}
            </style></head><body><div style="height: {spacer}px"></div>
            <table style="{style}"><tbody>{body}</tbody></table></body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
        (doc, geometry)
    }

    const KEEP_UNSPLITTABLE: &str = "-fulgur-span-break-inside: avoid-unsplittable";

    /// Rows of one line each that a one-line cell spans, from y 70: freely
    /// fragmentable by default (css-tables-3 §6.1 keeps unfragmented only
    /// rows that no cell spans past), they break between them,
    /// and the spanning cell's box goes on beside `b`; under
    /// `avoid-unsplittable` they move to the next page together, though the
    /// spanning cell's content fits above the break.
    #[test]
    fn rows_of_one_line_that_a_cell_spans_break_apart_unless_kept() {
        let at = |style| {
            let (doc, geometry) = span_group_pass(style, &[1, 1], 1, 70);
            (
                ["a", "b"].map(|id| placements(&doc, &geometry, id)),
                placements(&doc, &geometry, "s"),
            )
        };
        assert_eq!(
            [at(""), at(KEEP_UNSPLITTABLE)],
            [
                ([vec![(0, 70.0)], vec![(1, 0.0)]], vec![(0, 70.0), (1, 0.0)]),
                ([vec![(1, 0.0)], vec![(1, 20.0)]], vec![(1, 0.0)]),
            ]
        );
    }

    /// Rows of one line each that a cell of four lines spans, grown to 40 px
    /// each, from y 50: by default they break between them, and the
    /// spanning cell's lines with them, two on each page; under
    /// `avoid-unsplittable` they move to the next page together.
    #[test]
    fn a_tall_cell_spanning_rows_of_one_line_breaks_by_lines_unless_kept() {
        let at = |style| {
            let (doc, geometry) = span_group_pass(style, &[1, 1], 4, 50);
            let s = find_by_id(&doc, "s").unwrap();
            (
                ["a", "b"].map(|id| placements(&doc, &geometry, id)),
                line_partition(&geometry, s),
            )
        };
        assert_eq!(
            [at(""), at(KEEP_UNSPLITTABLE)],
            [
                (
                    [vec![(0, 50.0)], vec![(1, 0.0)]],
                    (vec![0, 1], vec![0, 2, 4])
                ),
                ([vec![(1, 0.0)], vec![(1, 40.0)]], (vec![1], vec![])),
            ]
        );
    }

    /// Rows of two lines that a cell of three lines spans, from y 50, break
    /// between them under `avoid-unsplittable` too: a row that can split
    /// between its lines leaves its rows freely fragmentable. The spanning
    /// cell breaks by its lines at the same place, two on the first page.
    #[test]
    fn rows_of_two_lines_that_a_cell_spans_break_between_them_though_kept() {
        let (doc, geometry) = span_group_pass(KEEP_UNSPLITTABLE, &[2, 2], 3, 50);
        let s = find_by_id(&doc, "s").unwrap();
        assert_eq!(
            (
                ["a", "b"].map(|id| placements(&doc, &geometry, id)),
                line_partition(&geometry, s),
            ),
            (
                [vec![(0, 50.0)], vec![(1, 0.0)]],
                (vec![0, 1], vec![0, 2, 3])
            )
        );
    }

    /// Rows of two lines that a cell of three lines spans, from y 70, break
    /// inside the first row under `avoid-unsplittable`: one line of `a` and
    /// of `s` stays on the first page.
    #[test]
    fn rows_of_two_lines_that_a_cell_spans_break_inside_a_row_though_kept() {
        let (doc, geometry) = span_group_pass(KEEP_UNSPLITTABLE, &[2, 2], 3, 70);
        let (a, s) = (
            find_by_id(&doc, "a").unwrap(),
            find_by_id(&doc, "s").unwrap(),
        );
        assert_eq!(
            (line_partition(&geometry, a), line_partition(&geometry, s)),
            ((vec![0, 1], vec![0, 1, 2]), (vec![0, 1], vec![0, 1, 3]))
        );
    }

    /// A table of `rows` rows `r1`, `r2`, ... of 80 px cells, under the
    /// table style `style`, after a `spacer` px block, on pages `page` px
    /// tall.
    fn counted_rows_pass(
        style: &str,
        rows: usize,
        spacer: u32,
        page: f32,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let rows = (1..=rows)
            .map(|n| format!(r#"<tr><td id="r{n}">{n}</td></tr>"#))
            .collect::<String>();
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              td {{ box-sizing: border-box; height: 80px; padding: 0; }}
            </style></head><body><div style="height: {spacer}px"></div>
            <table style="{style}"><tbody>{rows}</tbody></table></body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry = super::run_pass_with_break_styles(doc.deref_mut(), page.as_px(), &styles);
        (doc, geometry)
    }

    /// A table's `-fulgur-table-widows` counts rows: at least that many are
    /// on its last page. Sixty 80 px rows after a 100 px block on 400 px
    /// pages fill three rows of the first page and five of each after it,
    /// leaving rows 59 and 60 on the last; under `-fulgur-table-widows: 5`
    /// it opens on row 56, rows 54 and 55 ending the page before. The CSS
    /// `widows` of the table moves no row.
    #[test]
    fn a_tables_widows_count_rows() {
        let opening = |style: &str| {
            let (doc, geometry) = counted_rows_pass(style, 60, 100, 400.0);
            ["r55", "r56", "r59"].map(|id| placements(&doc, &geometry, id))
        };
        assert_eq!(
            opening(""),
            [vec![(11, 80.0)], vec![(11, 160.0)], vec![(12, 0.0)]]
        );
        assert_eq!(
            opening("-fulgur-table-widows: 5"),
            [vec![(11, 80.0)], vec![(12, 0.0)], vec![(12, 240.0)]]
        );
        assert_eq!(
            opening("widows: 5"),
            [vec![(11, 80.0)], vec![(11, 160.0)], vec![(12, 0.0)]]
        );
    }

    /// A table's `-fulgur-table-orphans` counts rows: at least that many
    /// start on the page where it first breaks, else it moves to the next
    /// page whole. After a 150 px block on 300 px pages, one 80 px row fits:
    /// a count of 2 moves the table, as two rows need 160 px; 9, more rows
    /// than a page holds, moves it too, and it breaks where the next page
    /// ends. The CSS `orphans` of the table moves no row.
    #[test]
    fn a_tables_orphans_move_it_whole() {
        let first = |style: &str| {
            let (doc, geometry) = counted_rows_pass(style, 6, 150, 300.0);
            ["r1", "r4", "r5"].map(|id| placements(&doc, &geometry, id))
        };
        assert_eq!(
            first(""),
            [vec![(0, 150.0)], vec![(1, 160.0)], vec![(2, 0.0)]]
        );
        assert_eq!(
            first("-fulgur-table-orphans: 2"),
            [vec![(1, 0.0)], vec![(2, 0.0)], vec![(2, 80.0)]]
        );
        assert_eq!(
            first("-fulgur-table-orphans: 9"),
            [vec![(1, 0.0)], vec![(2, 0.0)], vec![(2, 80.0)]]
        );
        assert_eq!(
            first("orphans: 2"),
            [vec![(0, 150.0)], vec![(1, 160.0)], vec![(2, 0.0)]]
        );
    }

    /// A table's `orphans` and `widows` are inherited by its cells'
    /// paragraphs as line counts, as CSS has it (css-break-3 §3.3); its row
    /// counts are not. Six 20 px lines with three of them fitting the page
    /// split 3 + 3 under the initial counts and under
    /// `-fulgur-table-widows: 4`, and 2 + 4 under the table's `widows: 4`.
    #[test]
    fn a_tables_widows_are_its_cells_line_counts_and_its_row_counts_are_not() {
        let partition = |style: &str| {
            let html = format!(
                r#"<!doctype html><html><head><style>
                html, body {{ margin: 0; padding: 0; font-family: 'Noto Sans'; }}
                table {{ border-spacing: 0; }} td {{ padding: 0; }}
                p {{ margin: 0; line-height: 20px; }}
                </style></head><body><div style="height: 40px"></div>
                <table style="{style}"><tr><td><p id="p">1<br>2<br>3<br>4<br>5<br>6</p></td></tr></table>
                </body></html>"#
            );
            let fonts = [crate::test_fonts::noto_sans_regular_ttf()];
            let (mut doc, _, _) = blitz_adapter::parse_html_with_local_resources(
                &html, 200.0, 10000, &fonts, false, None,
            );
            blitz_adapter::resolve(&mut doc);
            let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geometry =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
            let p = find_by_id(&doc, "p").unwrap();
            line_partition(&geometry, p)
        };
        assert_eq!(partition(""), (vec![0, 1], vec![0, 3, 6]));
        assert_eq!(partition("widows: 4"), (vec![0, 1], vec![0, 2, 6]));
        assert_eq!(
            partition("-fulgur-table-widows: 4"),
            (vec![0, 1], vec![0, 3, 6])
        );
    }

    /// A table on 100 px pages whose cells are padded 10 px above and below,
    /// under `head`, after the rows `before`: a row of the cell `cell`
    /// holding ten 20 px blocks `d1` to `d10`, then, when `side`, the cell
    /// `side` holding `e1` to `e10`; then a 20 px block `after`. `css` comes
    /// last.
    fn padded_cell_pass(
        head: &str,
        before: &str,
        side: bool,
        css: &str,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let blocks = |prefix: &str| {
            (1..=10)
                .map(|n| format!(r#"<div id="{prefix}{n}">{n}</div>"#))
                .collect::<String>()
        };
        let side = if side {
            format!(r#"<td id="side">{}</td>"#, blocks("e"))
        } else {
            String::new()
        };
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              th {{ box-sizing: border-box; height: 20px; padding: 0; }}
              td {{ padding: 10px 0; }}
              td > div, #after {{ height: 20px; }}
              {css}
            </style></head><body>
              <table>{head}<tbody>{before}<tr><td id="cell">{}</td>{side}</tr></tbody></table>
              <div id="after"></div>
            </body></html>"#,
            blocks("d")
        );
        let mut doc = parse(&html, 200.0);
        let geometry = run_pass(&mut doc, 100.0);
        (doc, geometry)
    }

    /// The header row of [`padded_cell_pass`]'s tables that have one.
    const PADDED_HEAD: &str = r#"<thead><tr><th id="h">H</th></tr></thead>"#;

    /// A cell split by a page break keeps its top padding on each page it
    /// continues on, as `box-decoration-break: clone` would: its content
    /// restarts below the padding, and the padding takes room from the
    /// page. The cell's fragment there starts at the page top, so its
    /// background covers the padding. Four blocks fit each page below the
    /// padding: `d5` and `d9` open pages 1 and 2 at y 10 (at 0 without the
    /// repeat, five blocks a page). Its last fragment ends below its bottom
    /// padding, and so does the table: `after` follows at y 60, not against
    /// `d10`'s bottom at 50.
    #[test]
    fn a_continued_table_cell_repeats_its_top_padding() {
        let (doc, geometry) = padded_cell_pass("", "", false, "");

        assert_eq!(placements(&doc, &geometry, "d4"), [(0, 70.0)]);
        assert_eq!(placements(&doc, &geometry, "d5"), [(1, 10.0)]);
        assert_eq!(placements(&doc, &geometry, "d8"), [(1, 70.0)]);
        assert_eq!(placements(&doc, &geometry, "d9"), [(2, 10.0)]);
        assert_eq!(placements(&doc, &geometry, "d10"), [(2, 30.0)]);
        assert_eq!(
            placements(&doc, &geometry, "cell"),
            [(0, 0.0), (1, 0.0), (2, 0.0)],
            "each fragment of the cell starts at its page top"
        );
        let cell = &geometry[&find_by_id(&doc, "cell").unwrap()].fragments;
        assert_eq!(cell.last().unwrap().height.to_f32(), 60.0);
        assert_eq!(placements(&doc, &geometry, "after"), [(2, 60.0)]);
    }

    /// A cell's top border repeats with its padding: its content restarts
    /// 15 px down for a 5 px border and 10 px of padding.
    #[test]
    fn a_continued_table_cell_repeats_its_top_border() {
        let (doc, geometry) =
            padded_cell_pass("", "", false, "#cell { border-top: 5px solid black; }");

        assert_eq!(placements(&doc, &geometry, "d1"), [(0, 15.0)]);
        assert_eq!(placements(&doc, &geometry, "d4"), [(0, 75.0)]);
        assert_eq!(placements(&doc, &geometry, "d5"), [(1, 15.0)]);
        assert_eq!(placements(&doc, &geometry, "d9"), [(2, 15.0)]);
        assert_eq!(placements(&doc, &geometry, "after"), [(2, 65.0)]);
    }

    /// Under a repeated header a continued cell starts at the header's
    /// bottom and its content one padding lower; each cell of the row
    /// repeats its own padding.
    #[test]
    fn a_continued_table_cell_repeats_its_top_padding_below_a_repeated_header() {
        let (doc, geometry) = padded_cell_pass(PADDED_HEAD, "", false, "");

        assert_eq!(placements(&doc, &geometry, "d3"), [(0, 70.0)]);
        assert_eq!(placements(&doc, &geometry, "d4"), [(1, 30.0)]);
        assert_eq!(
            placements(&doc, &geometry, "cell")[..2],
            [(0, 20.0), (1, 20.0)]
        );

        let (doc, geometry) = padded_cell_pass(PADDED_HEAD, "", true, "#side { padding: 20px 0; }");
        assert_eq!(placements(&doc, &geometry, "d4"), [(1, 30.0)]);
        assert_eq!(placements(&doc, &geometry, "e3"), [(0, 80.0)]);
        assert_eq!(placements(&doc, &geometry, "e4"), [(1, 40.0)]);
        assert_eq!(
            placements(&doc, &geometry, "side")[..2],
            [(0, 20.0), (1, 20.0)]
        );
    }

    /// A row none of whose cells can place content in the rest of the page
    /// moves to the next page whole, leaving no empty padded box behind: a
    /// row below a 75 px row has 25 px, and its cell's first block ends
    /// 30 px below its top.
    #[test]
    fn a_row_whose_first_content_does_not_fit_moves_whole() {
        let before =
            r#"<tr><td style="padding: 0"><div id="first" style="height: 75px"></div></td></tr>"#;
        let (doc, geometry) = padded_cell_pass("", before, false, "");

        assert_eq!(placements(&doc, &geometry, "first"), [(0, 0.0)]);
        assert_eq!(
            placements(&doc, &geometry, "cell"),
            [(1, 0.0), (2, 0.0), (3, 0.0)]
        );
        assert_eq!(placements(&doc, &geometry, "d1"), [(1, 10.0)]);
    }

    /// A paragraph `k` of two 20 px lines kept whole by `break-inside:
    /// avoid`, in a `div` `wrap` that opens a cell `cell` padded 10 px above
    /// and below: the row, after a row holding a block `first` of `first`
    /// px when not empty, with `lead` before `wrap` in the cell, laid out
    /// with the break styles on 100 px pages.
    fn kept_in_padded_cell_pass(
        first: &str,
        lead: &str,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let before = if first.is_empty() {
            String::new()
        } else {
            format!(
                r#"<tr><td style="padding: 0"><div id="first" style="height: {first}px"></div></td></tr>"#
            )
        };
        let html = format!(
            r#"<!doctype html>
            <html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ margin: 0; border-spacing: 0; width: 100px; }}
              td {{ padding: 10px 0; }}
              p {{ margin: 0; line-height: 20px; }}
            </style></head><body>
              <table><tbody>{before}<tr><td id="cell">{lead}<div id="wrap">
                <p id="k" style="break-inside: avoid">K1<br>K2</p>
              </div></td></tr></tbody></table>
            </body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
        (doc, geometry)
    }

    /// A paragraph kept whole in a `div` that opens a padded cell.
    ///
    /// At the page foot, with 40 px left under a 60 px row, the cell's
    /// first content (the padding and the 40 px paragraph, 50 px) does not
    /// fit, so the row moves whole, the cell's padding with it
    /// (`row_places_nothing_in`): the padding is no content of its own, and
    /// moving the `div` alone inside the cell, at the padding's break point,
    /// would leave the empty padded cell at the foot. At the page top the
    /// row stays, the paragraph below the padding. After a 60 px block in
    /// the cell, the paragraph crosses the foot and moves to the next page
    /// with its `div`, the block before it in the cell giving the break
    /// point, and starts below the cell's repeated padding.
    #[test]
    fn a_kept_paragraph_in_a_div_opening_a_padded_cell() {
        let (doc, geometry) = kept_in_padded_cell_pass("60", "");
        assert_eq!(placements(&doc, &geometry, "cell"), [(1, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 10.0)]);

        let (doc, geometry) = kept_in_padded_cell_pass("", "");
        assert_eq!(placements(&doc, &geometry, "cell"), [(0, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "k"), [(0, 10.0)]);

        let (doc, geometry) =
            kept_in_padded_cell_pass("", r#"<div id="lead" style="height: 60px"></div>"#);
        assert_eq!(placements(&doc, &geometry, "lead"), [(0, 10.0)]);
        assert_eq!(placements(&doc, &geometry, "wrap"), [(1, 10.0)]);
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 10.0)]);
        assert_eq!(placements(&doc, &geometry, "cell"), [(0, 0.0), (1, 0.0)]);
    }

    /// A table whose first row moves to the next page takes its caption and
    /// its top edge along: a captioned table at y 150 of a 200 px page,
    /// whose first row's cell opens with a 50 px block against the 30 px
    /// left under the 20 px caption, starts on the next page, caption first;
    /// and a table with a 10 px top border leaves no fragment of it behind.
    #[test]
    fn a_table_moves_with_its_caption_and_top_edge_when_its_first_row_moves() {
        let row =
            r#"<tr><td id="cell" style="height: auto"><div style="height: 50px"></div></td></tr>"#;
        let (doc, geometry) = header_pass(
            &format!(
                r#"<div style="height: 150px"></div><table><caption id="cap">C</caption>{row}</table>"#
            ),
            200.0,
        );
        assert_eq!(placements(&doc, &geometry, "cap"), [(1, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "cell"), [(1, 20.0)]);

        let (doc, geometry) = header_pass(
            &format!(
                r#"<div style="height: 150px"></div><table id="t" style="border-top: 10px solid">{row}</table>"#
            ),
            200.0,
        );
        assert_eq!(placements(&doc, &geometry, "t"), [(1, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "cell"), [(1, 10.0)]);
    }

    /// A table whose first body row places nothing under its header in the
    /// rest of the page starts on the next page, header and all, as the row
    /// would move there whole: the check reads the row's first content as
    /// the row move does, here the 10 px padding and two lines, the
    /// paragraph's `orphans`, of a 15-line paragraph, 50 px against the
    /// 30 px left under a 20 px header at y 150. Measuring the one-box cell
    /// whole (320 px, more than a page) started the table and left the
    /// header alone at the foot.
    #[test]
    fn a_header_whose_first_row_places_nothing_moves_with_it() {
        let lines = (1..=15)
            .map(|n| format!("L{n}"))
            .collect::<Vec<_>>()
            .join("<br>");
        let body = format!(
            r#"<div style="height: 150px"></div><table><thead><tr><th id="h">H</th></tr></thead>
            <tbody><tr><td style="padding: 10px 0; height: auto">
            <p id="p" style="margin: 0; line-height: 20px; orphans: 2">{lines}</p>
            </td></tr></tbody></table>"#
        );
        let (doc, geometry) = header_pass(&body, 200.0);

        assert_eq!(placements(&doc, &geometry, "h")[0], (1, 0.0));
        assert_eq!(placements(&doc, &geometry, "p")[0], (1, 30.0));
    }

    #[test]
    fn table_starting_mid_page_uses_remaining_first_page_capacity() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 20px; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; height: 20px; padding: 0; }
            </style></head><body>
              <div id="lead"></div>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead><tbody>
                <tr><td id="row1">1</td></tr><tr><td>2</td></tr>
                <tr><td id="row3">3</td></tr><tr><td>4</td></tr>
              </tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let row1_id = find_by_id(&doc, "row1").unwrap();
        let row3_id = find_by_id(&doc, "row3").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let headers = &geometry[&header_id].fragments;

        assert!((headers[0].y.to_f32() - 20.0).abs() < 0.5);
        assert_eq!(geometry[&row1_id].fragments[0].page_index, 0);
        assert_eq!(headers[1].page_index, 1);
        assert!(headers[1].y.to_f32().abs() < 0.5);
        let later = geometry[&row3_id].fragments.last().unwrap();
        let table_top = geometry[&table_id]
            .fragments
            .iter()
            .find(|fragment| fragment.page_index == later.page_index)
            .unwrap()
            .y
            .to_f32();
        assert!(later.y.to_f32() >= table_top + 19.5);
    }

    /// Supersedes `table_moves_first_body_row_when_only_header_fits_remaining_space`
    /// (PR #710), which expected the header on pages `[0, 1]` — page 0 carrying
    /// the header with no row under it. The whole table now starts on page 1
    /// instead (fulgur-naj7.6).
    #[test]
    fn table_starts_on_next_page_when_only_the_header_would_fit() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 60px; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 20px; }
              td, td > div { height: 30px; }
            </style></head><body>
              <div id="lead"></div>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead><tbody>
                <tr><td><div id="row1">1</div></td></tr>
              </tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let row_id = find_by_id(&doc, "row1").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let row = &geometry[&row_id].fragments[0];

        assert_eq!(row.page_index, 1);
        assert!((row.y.to_f32() - 20.0).abs() < 0.5);
        assert!(row.y.to_f32() + row.height.to_f32() <= 100.0);
        // No orphaned header on page 0 — the table begins on page 1.
        assert_eq!(
            geometry[&header_id]
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert!(
            geometry[&table_id]
                .fragments
                .iter()
                .all(|fragment| fragment.y.to_f32() + fragment.height.to_f32() <= 100.0),
            "table fragments must stay within the page: {:?}",
            geometry[&table_id].fragments
        );
    }

    #[test]
    fn table_does_not_insert_header_only_page_for_oversized_first_row_at_page_top() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 20px; }
              #row1 { height: 90px; }
              #row2 { height: 20px; }
            </style></head><body>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead><tbody>
                <tr><td id="row1">1</td></tr>
                <tr><td id="row2">2</td></tr>
              </tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();
        let row1_id = find_by_id(&doc, "row1").unwrap();
        let row2_id = find_by_id(&doc, "row2").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let row1 = &geometry[&row1_id].fragments[0];
        let row2 = &geometry[&row2_id].fragments[0];

        // The 90 px row is sliced at the foot of its 80 px strip and ends
        // 10 px down the next page, under the repeated header.
        assert_eq!(row1.page_index, 0);
        assert!((row1.y.to_f32() - 20.0).abs() < 0.5);
        assert_eq!(geometry[&row1_id].fragments.len(), 2);
        assert_eq!(row2.page_index, 1);
        assert!((row2.y.to_f32() - 30.0).abs() < 0.5);
        assert_eq!(
            geometry[&header_id]
                .fragments
                .iter()
                .map(|fragment| fragment.page_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let first_table = geometry[&table_id]
            .fragments
            .iter()
            .find(|fragment| fragment.page_index == 0)
            .unwrap();
        assert!(
            first_table.height.to_f32() > 20.5,
            "first page must include the oversized body row, not a header-only band: {first_table:?}"
        );
    }

    #[test]
    fn table_moves_all_nested_cells_in_first_body_row_from_occupied_strip() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 60px; }
              table { margin: 0; border-spacing: 0; table-layout: fixed; width: 100px; }
              th, td { box-sizing: border-box; height: 30px; padding: 0; }
              th { height: 20px; }
              td > div { height: 30px; }
            </style></head><body>
              <div id="lead"></div>
              <table><thead><tr><th colspan="2">Header</th></tr></thead><tbody><tr>
                <td><div id="left">L</div></td><td><div id="right">R</div></td>
              </tr></tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let left_id = find_by_id(&doc, "left").unwrap();
        let right_id = find_by_id(&doc, "right").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let left = &geometry[&left_id].fragments[0];
        let right = &geometry[&right_id].fragments[0];

        assert_eq!(left.page_index, 1);
        assert_eq!(right.page_index, 1);
        assert!((left.y.to_f32() - 20.0).abs() < 0.5);
        assert!((right.y.to_f32() - left.y.to_f32()).abs() < 0.5);
    }

    #[test]
    fn table_keeps_nested_body_extent_that_fits_below_mid_page_header() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 60px; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 20px; }
              #short { height: 10px; }
              #tall { height: 30px; }
            </style></head><body>
              <div id="lead"></div>
              <table id="t"><thead><tr><th>Header</th></tr></thead><tbody><tr><td>
                <div id="short">short</div><div id="tall">tall</div>
              </td></tr></tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let short_id = find_by_id(&doc, "short").unwrap();
        let tall_id = find_by_id(&doc, "tall").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        assert_eq!(geometry[&short_id].fragments[0].page_index, 0);
        assert_eq!(geometry[&tall_id].fragments[0].page_index, 1);
        let first_table = geometry[&table_id]
            .fragments
            .iter()
            .find(|fragment| fragment.page_index == 0)
            .unwrap();
        assert!((first_table.height.to_f32() - 30.0).abs() < 0.5);
    }

    #[test]
    fn table_uses_max_partial_extent_across_first_body_row_cells() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 60px; }
              table { margin: 0; border-spacing: 0; table-layout: fixed; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              th { height: 20px; }
              #left-short { height: 5px; }
              #right-short { height: 15px; }
              .tall { height: 30px; }
            </style></head><body>
              <div id="lead"></div>
              <table id="t"><thead><tr><th colspan="2">Header</th></tr></thead><tbody><tr>
                <td><div id="left-short"></div><div class="tall"></div></td>
                <td><div id="right-short"></div><div class="tall"></div></td>
              </tr></tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let first_table = geometry[&table_id]
            .fragments
            .iter()
            .find(|fragment| fragment.page_index == 0)
            .unwrap();

        assert!(
            (first_table.height.to_f32() - 35.0).abs() < 0.5,
            "expected header plus max partial row extent: {first_table:?}"
        );
    }

    #[test]
    fn nested_oversized_header_fallback_honors_occupied_outer_strip() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 60px; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; padding: 0; }
              #outer-header { height: 20px; }
              #inner-header { height: 80px; }
              #inner-body { height: 30px; }
            </style></head><body>
              <div id="lead"></div>
              <table><thead><tr><th id="outer-header">Outer</th></tr></thead><tbody><tr><td>
                <table id="inner"><thead><tr><th id="inner-header">Inner</th></tr></thead>
                  <tbody><tr><td id="inner-body">Body</td></tr></tbody></table>
              </td></tr></tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let inner_id = find_by_id(&doc, "inner").unwrap();
        let inner_header_id = find_by_id(&doc, "inner-header").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        assert_eq!(geometry[&inner_header_id].fragments[0].page_index, 1);
        assert!(
            geometry[&inner_id]
                .fragments
                .iter()
                .filter(|fragment| fragment.page_index == 0)
                .all(|fragment| fragment.height.to_f32() <= 0.0)
        );
    }

    #[test]
    fn table_moves_to_fresh_page_when_header_does_not_fit() {
        let html = r#"<!doctype html>
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              #lead { height: 90px; }
              table { margin: 0; border-spacing: 0; width: 100px; }
              th, td { box-sizing: border-box; height: 20px; padding: 0; }
            </style></head><body>
              <div id="lead"></div>
              <table id="t"><thead><tr><th id="h">Header</th></tr></thead><tbody>
                <tr><td>1</td></tr><tr><td>2</td></tr>
                <tr><td>3</td></tr><tr><td>4</td></tr>
              </tbody></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_id = find_by_id(&doc, "t").unwrap();
        let header_id = find_by_id(&doc, "h").unwrap();

        let geometry = run_pass(&mut doc, 100.0);
        let table = &geometry[&table_id].fragments[0];
        let header = &geometry[&header_id].fragments[0];

        assert_eq!(table.page_index, 1);
        assert!(table.y.to_f32().abs() < 0.5);
        assert_eq!(header.page_index, 1);
        assert!(header.y.to_f32().abs() < 0.5);
    }

    #[test]
    fn repeated_header_clone_budget_is_shared_and_page_atomic() {
        let mut template = PaginationGeometryTable::new();
        template.entry(7).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 10.0_f32.as_px(),
            width: 50.0_f32.as_px(),
            height: 20.0_f32.as_px(),
        });
        template.entry(8).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 12.0_f32.as_px(),
            width: 30.0_f32.as_px(),
            height: 10.0_f32.as_px(),
        });
        let table_pages = [(0, 10.0), (1, 0.0)];
        let mut first = PaginationGeometryTable::new();
        let mut second = PaginationGeometryTable::new();
        let mut remaining_repeat_budget = 3;

        append_repeated_header_fragments(
            &mut first,
            &template,
            &template,
            &table_pages,
            10.0,
            &mut remaining_repeat_budget,
        );
        append_repeated_header_fragments(
            &mut second,
            &template,
            &template,
            &table_pages,
            10.0,
            &mut remaining_repeat_budget,
        );

        for node_id in [7, 8] {
            assert!(first[&node_id].is_repeat);
            assert_eq!(first[&node_id].fragments.len(), 2);
            assert_eq!(first[&node_id].fragments[1].page_index, 1);
            assert!(second[&node_id].is_repeat);
            assert_eq!(second[&node_id].fragments.len(), 1);
            assert_eq!(second[&node_id].fragments[0].page_index, 0);
        }
        assert_eq!(remaining_repeat_budget, 1);
    }

    /// fulgur-ezst: a tiny input with a pathologically tall CSS height on a
    /// CHILDLESS block prints only blank pages, so instead of slicing it to
    /// the `MAX_PAGES` cap (~10k blank pages) the fragmenter collapses it to
    /// a single page. `height: 99999999px` would otherwise slice into
    /// ~125 000 fragments — the small-input DoS.
    #[test]
    fn pathological_childless_tall_block_collapses() {
        let html = r#"<html><body><div style="height: 99999999px"></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(
            implied_page_count(&table),
            1,
            "childless pathological height must collapse to a single page",
        );
    }

    /// fulgur-ezst: Stylo/Taffy clamps an absurd `<length>` to `f32::MAX`
    /// (≈3.4e38), the worst-case finite input. A childless block that tall
    /// also collapses to one page (the ceiling-bounded slice loop still runs
    /// its counter math but pushes no fragment).
    #[test]
    fn f32_max_childless_height_collapses() {
        let html = r#"<html><body><div style="height: 1e39px"></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(
            implied_page_count(&table),
            1,
            "childless f32::MAX height must collapse to a single page",
        );
    }

    /// fulgur-ezst: background presence does not gate the collapse — a
    /// childless filled band this tall is still a pathological amplifier.
    #[test]
    fn childless_tall_block_with_background_collapses() {
        let html =
            r#"<html><body><div style="height: 99999999px; background: red"></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(implied_page_count(&table), 1);
    }

    /// fulgur-ezst: a tall block WITH rendered content is not childless, so
    /// it is NOT collapsed — it still clamps to ~MAX_PAGES (truncate-and-
    /// warn). Also guards routing: a huge parent with a small child must
    /// reach the slice loop, not the recursion branch (the recursion gate
    /// measures descendant overflow, not the parent's intrinsic height).
    #[test]
    fn content_bearing_tall_block_is_page_capped() {
        let html = r#"<html><body><div style="height: 99999999px"><p>x</p></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        let pages = implied_page_count(&table);
        assert!(
            pages > 1_000,
            "content-bearing block must take the cap path, not collapse; got {pages}",
        );
        assert!(
            pages <= crate::MAX_PAGES + 1,
            "content-bearing block must stay clamped to ~MAX_PAGES ({}); got {pages}",
            crate::MAX_PAGES,
        );
    }

    /// fulgur-ezst: a childless band that fits WITHIN the cap renders its
    /// full page count — the collapse must not over-fire on ordinary
    /// multi-page spacers. `height: 4000px` on an 800px strip = 5 pages.
    #[test]
    fn childless_subcap_band_renders_full() {
        let html = r#"<html><body><div style="height: 4000px"></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(
            implied_page_count(&table),
            5,
            "a sub-cap childless band must render fully, not collapse",
        );
    }

    /// fulgur-ezst (Codex P2 on PR #553): an empty *block* child lays out
    /// with positive width but zero height, so a `height <= 0 && width <= 0`
    /// emptiness test would wrongly count it as content and disable the
    /// collapse — a trivial DoS bypass. `subtree_has_rendered_content` treats
    /// any zero-AREA descendant as blank (recursing for overflow), so a tall
    /// block whose only child is an empty block still collapses to one page.
    #[test]
    fn childless_tall_block_with_empty_block_child_collapses() {
        let html = r#"<html><body><div style="height: 99999999px"><div></div></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(
            implied_page_count(&table),
            1,
            "an empty (zero-area) block child must not defeat the collapse",
        );
    }

    /// fulgur-ezst: the emptiness recursion must still find VISIBLE content
    /// nested under a zero-area wrapper — a zero-height (overflow:visible)
    /// wrapper holding a real `<p>` is content, so the tall block is NOT
    /// childless and stays on the cap path. Guards the `||` fix against
    /// over-collapsing (dropping visibly-overflowing descendants).
    #[test]
    fn tall_block_with_content_under_zero_height_wrapper_is_not_collapsed() {
        let html = r#"<html><body><div style="height: 99999999px"><div style="height: 0"><p>x</p></div></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        let pages = implied_page_count(&table);
        assert!(
            pages > 1_000,
            "visible content under a zero-height wrapper must block the collapse; got {pages}",
        );
        assert!(
            pages <= crate::MAX_PAGES + 1,
            "still clamped to ~MAX_PAGES ({}); got {pages}",
            crate::MAX_PAGES,
        );
    }

    /// fulgur-ezst: the defensive guards of `subtree_has_rendered_content`
    /// read as "no content" (false) without panicking — a nonexistent node
    /// id (the `get_node` None guard) and a call already at the recursion
    /// ceiling (the `MAX_DOM_DEPTH` guard).
    #[test]
    fn subtree_has_rendered_content_guards() {
        use std::ops::DerefMut;
        let html = r#"<html><body><div id="x">hi</div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let x = find_by_id(doc.deref_mut(), "x").expect("div#x");
        assert!(!subtree_has_rendered_content(
            doc.deref_mut(),
            usize::MAX,
            0
        ));
        assert!(!subtree_has_rendered_content(
            doc.deref_mut(),
            x,
            crate::MAX_DOM_DEPTH
        ));
    }

    /// fulgur-c8re (security): a pathologically tall CHILDLESS replaced
    /// element (`<img>` / `<svg>`) that paints nothing must collapse like any
    /// blank spacer. "Paints nothing" covers the common offline-first case of
    /// an unresolved `src` (no matching `AssetBundle` entry), a
    /// `visibility:hidden` image, an undecodable format, and an empty `<svg>`.
    ///
    /// The predecessor `is_replaced_content` exception (Codex P2 on PR #553 /
    /// fulgur-ezst) gated the collapse on tag name alone, so such a
    /// non-painting node disabled the collapse and amplified a few bytes of
    /// HTML into ~`MAX_PAGES` blank pages (a validated high-severity DoS). It
    /// was removed: a replaced element only reaches this branch when it is
    /// taller than `MAX_PAGES` pages (~10M px), a range no legitimate single
    /// image occupies, so clipping it to one page loses no real content.
    #[test]
    fn replaced_tall_childless_block_collapses() {
        for tag in [
            // Unresolved `src`: no `AssetBundle` on this test path, so the
            // image paints nothing — yet it is still a pathological amplifier.
            r#"<img src="missing.png" style="display:block;width:10px;height:99999999px">"#,
            // `visibility:hidden`: occupies layout, paints nothing.
            r#"<img src="x.png" style="display:block;visibility:hidden;width:10px;height:99999999px">"#,
            // Empty `<svg>`: no drawable content.
            r#"<svg style="display:block;width:10px;height:99999999px"></svg>"#,
        ] {
            let html = format!(r#"<html><body>{tag}</body></html>"#);
            let mut doc = parse(&html, 600.0);
            let table = run_pass(&mut doc, 800.0);
            assert_eq!(
                implied_page_count(&table),
                1,
                "a non-painting tall replaced element must collapse, not \
                 amplify into blank pages: {tag}",
            );
        }
    }

    /// fulgur-ezst (Codex P2 on PR #553): a `visibility:hidden` descendant
    /// occupies layout space but paints nothing, so it must not defeat the
    /// collapse — a tall block whose only child is invisible still collapses.
    #[test]
    fn invisible_only_child_collapses() {
        let html = r#"<html><body><div style="height:99999999px"><div style="visibility:hidden;height:1px"></div></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert_eq!(
            implied_page_count(&table),
            1,
            "a visibility:hidden-only descendant must not defeat the collapse",
        );
    }

    /// fulgur-ezst: the visibility skip must still recurse — a
    /// `visibility:visible` descendant under a `visibility:hidden` wrapper is
    /// real content, so the tall block is not childless and stays capped.
    #[test]
    fn visible_child_under_hidden_wrapper_blocks_collapse() {
        let html = r#"<html><body><div style="height:99999999px"><div style="visibility:hidden"><p style="visibility:visible">x</p></div></div></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        let pages = implied_page_count(&table);
        assert!(
            pages > 1_000,
            "visible content under a hidden wrapper must block the collapse; got {pages}",
        );
    }

    /// fulgur-c8re (security, Codex P1 on PR #575): the childless collapse must
    /// bound the SPACE the pathological element occupies, not just skip its
    /// per-page fragment pushes. The predecessor behaviour left the slice loop
    /// advancing `page_index` all the way to `MAX_PAGES`, so a following in-flow
    /// sibling was stranded on a deep page and `implied_page_count` — which
    /// reads the max fragment index across ALL nodes — re-inflated the document
    /// to ~`MAX_PAGES` blank pages (`<div huge></div><p>after</p>`), defeating
    /// the collapse. The element now consumes only its single first slice and
    /// following content reflows onto the next page. Asserts: the tall div
    /// contributes exactly one fragment (collapse fired) AND the `<p>` sibling
    /// lands on a bounded page, so the whole document stays a handful of pages.
    #[test]
    fn childless_collapse_reflows_following_sibling() {
        use std::ops::DerefMut;
        let html = r#"<html><body><div id="tall" style="height: 99999999px"></div><p id="after">after</p></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        let tall_id = find_by_id(doc.deref_mut(), "tall").expect("div#tall");
        let after_id = find_by_id(doc.deref_mut(), "after").expect("p#after");
        let tall_frags = table
            .get(&tall_id)
            .expect("div#tall must appear in geometry")
            .fragments
            .len();
        assert_eq!(
            tall_frags, 1,
            "collapsed childless div must contribute exactly one fragment, got {tall_frags}",
        );
        let after_page = table
            .get(&after_id)
            .expect("p#after must appear in geometry")
            .fragments[0]
            .page_index;
        assert!(
            after_page <= 1,
            "collapsed spacer must reflow its trailing sibling onto a bounded page \
             (<= 1), not strand it ~MAX_PAGES deep; got {after_page}",
        );
        assert!(
            implied_page_count(&table) <= 2,
            "a childless spacer followed by content must stay a handful of pages; got {}",
            implied_page_count(&table),
        );
    }

    /// fulgur-c8re (security, Codex P1 on PR #575): the replaced-element
    /// collapse must bound trailing content too. A non-painting tall image
    /// (`<img src="missing.png" height:99999999px>`) followed by a `<p>` must
    /// not amplify into ~`MAX_PAGES` blank pages — the same trailing-sibling
    /// vector as `childless_collapse_reflows_following_sibling`, but exercising
    /// the replaced-element path that the `is_replaced_content` removal opened.
    #[test]
    fn replaced_tall_collapse_bounds_trailing_sibling() {
        let html = r#"<html><body><img src="missing.png" style="display:block;width:10px;height:99999999px"><p>after</p></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        assert!(
            implied_page_count(&table) <= 2,
            "a non-painting tall replaced element followed by content must \
             collapse without amplifying; got {} pages",
            implied_page_count(&table),
        );
    }

    /// fulgur-2m6w: a non-finite Taffy height (`+inf` / `NaN`) is treated
    /// as zero so it can never reach the slicing loop (where `+inf` would
    /// make `remaining -= last_slice_h` loop forever) nor poison the
    /// `cursor_y` advance. CSS clamps to `f32::MAX` and cannot produce a
    /// non-finite height, so inject one directly into the resolved layout.
    /// The node still enters geometry (via the `child_h <= 0.0` branch) and
    /// the document stays single-page.
    #[test]
    fn non_finite_height_treated_as_zero() {
        use std::ops::DerefMut;
        for bad in [f32::INFINITY, f32::NAN] {
            let html = r#"<html><body><div id="x">hi</div></body></html>"#;
            let mut doc = parse(html, 600.0);
            let id = find_by_id(doc.deref_mut(), "x").expect("div#x");
            doc.deref_mut()
                .get_node_mut(id)
                .expect("div#x")
                .final_layout
                .size
                .height = bad;
            let table = run_pass(&mut doc, 800.0);
            assert!(
                implied_page_count(&table) == 1,
                "non-finite height {bad} must not paginate; got {} pages",
                implied_page_count(&table),
            );
            assert!(
                table.contains_key(&id),
                "non-finite-height node must still be recorded in geometry",
            );
        }
    }

    /// fulgur-i5a: `overflow: hidden` is a pure visual effect that must
    /// not affect pagination. A 50px-tall parent containing a 1200px-tall
    /// child stays single-page even though the child would otherwise
    /// extend past the 800px page strip. The clip is applied at draw
    /// time (`render.rs` push/pop_clip_path), and pagination follows the
    /// parent's layout box, not the child's overflowing intrinsic height.
    #[test]
    fn overflow_hidden_does_not_split_oversize_child() {
        let html = r#"
            <html><body>
              <div style="height: 50px; overflow: hidden">
                <div style="height: 1200px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);

        // body + outer (50px clip) + inner (1200px overflowing child)
        // = 3 entries; every entry has a single fragment on page 0.
        assert_eq!(table.len(), 3, "expected body + outer + inner = 3 entries");
        for (id, geom) in &table {
            assert_eq!(
                geom.fragments.len(),
                1,
                "node {id} should not split under overflow:hidden parent"
            );
            assert_eq!(geom.fragments[0].page_index, 0);
        }
    }

    /// Phase 4 prerequisite repro: confirm `string-set` carry semantic
    /// across page break.
    #[test]
    fn string_set_carry_across_page_break() {
        use crate::blitz_adapter;
        use crate::gcpm::parser::parse_gcpm;
        use std::ops::DerefMut;
        use std::sync::Arc;

        let html = r#"<!DOCTYPE html>
<html>
<head><style>
@page { size: A4; margin: 60pt; }
h2 { string-set: chapter-title content(text); }
.box { padding: 60pt 0; }
</style></head>
<body>
<h2>Introduction</h2>
<div class="box">f1</div>
<div class="box">f2</div>
<div class="box">f3</div>
<div class="box">f4</div>
<div class="box">f5</div>
<div class="box">f6</div>
<div class="box">f7</div>
<div class="box">f8</div>
<h2 style="page-break-before:always">Background</h2>
</body></html>"#;
        let css = "h2 { string-set: chapter-title content(text); }";
        let gcpm = parse_gcpm(css);
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, 600.0, &fonts);
        let pass = blitz_adapter::StringSetPass::new(gcpm.string_set_mappings.clone());
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };
        blitz_adapter::apply_single_pass(&pass, &mut doc, &pass_ctx);
        let store = pass.into_store();
        blitz_adapter::resolve(&mut doc);
        let column_styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            run_pass_with_break_styles(doc.deref_mut(), 720.0_f32.as_pt().in_px(), &column_styles);

        let mut by_node: std::collections::BTreeMap<usize, Vec<(String, String)>> =
            std::collections::BTreeMap::new();
        for entry in store.entries() {
            by_node
                .entry(entry.node_id)
                .or_default()
                .push((entry.name.clone(), entry.value.clone()));
        }
        let states = collect_string_set_states(&geometry, &by_node);
        assert!(
            states.len() >= 2,
            "must span at least 2 pages, got {}",
            states.len()
        );
        let p0 = states[0]
            .get("chapter-title")
            .expect("page 0 must have chapter-title state");
        assert_eq!(p0.first.as_deref(), Some("Introduction"), "page 0 first");
        let p1 = states[1]
            .get("chapter-title")
            .expect("page 1 must have chapter-title state (carry)");
        assert_eq!(
            p1.start.as_deref(),
            Some("Introduction"),
            "page 1 start (carry from page 0 last)"
        );
    }

    /// Phase 3.4 follow-up (PR #296 Devin): regression for the
    /// fragmenter's running-element handling. `fragment_pagination_root`
    /// must record a zero-height fragment for every
    /// `position: running()` element so the running NodeId appears in
    /// geometry; without this, the downstream collect walk returns
    /// all-empty maps and `content: element(name)` in margin boxes
    /// silently produces nothing. Drive the engine pipeline through
    /// `Engine::render_html` and inspect the geometry table built by
    /// the same fragmenter pass.
    #[test]
    fn running_element_node_lands_in_geometry_with_zero_height() {
        use crate::blitz_adapter;
        use crate::gcpm::parser::parse_gcpm;
        use std::ops::DerefMut;
        use std::sync::Arc;

        let css = ".header { position: running(pageHeader); }";
        let html = r#"<!DOCTYPE html>
<html><head><style>.header { position: running(pageHeader); }</style></head>
<body>
<div class="header">Doc Header</div>
<p>Body.</p>
</body></html>"#;

        let gcpm = parse_gcpm(css);
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, 600.0, &fonts);
        let pass = blitz_adapter::RunningElementPass::new(gcpm.running_mappings.clone());
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };
        blitz_adapter::apply_single_pass(&pass, &mut doc, &pass_ctx);
        let store = pass.into_running_store();
        blitz_adapter::resolve(&mut doc);
        let column_styles = blitz_adapter::extract_column_style_table(&doc, &[]);

        let (geometry, _) = run_pass_with_break_and_running(
            doc.deref_mut(),
            800.0_f32.as_pt().in_px().to_f32(),
            &column_styles,
            &store,
        );

        // The running element's NodeId must exist in geometry on page 0
        // with a zero-height fragment.
        let mut found_running_node = None;
        for (&node_id, geom) in &geometry {
            if store.instance_for_node(node_id).is_some() {
                found_running_node = Some((node_id, geom.fragments.clone()));
                break;
            }
        }
        let (node_id, fragments) =
            found_running_node.expect("running element NodeId must appear in geometry table");
        assert_eq!(fragments.len(), 1, "single zero-height fragment");
        assert_eq!(fragments[0].page_index, 0);
        assert_eq!(
            fragments[0].height.to_f32(),
            0.0,
            "running fragment must not advance the cursor"
        );

        // collect_running_element_states must surface the instance.
        let states = collect_running_element_states(&geometry, &store);
        let entry = states[0]
            .get("pageHeader")
            .expect("pageHeader entry must appear in page 0 state");
        assert_eq!(
            entry.instance_ids,
            vec![store.instance_for_node(node_id).unwrap()]
        );
    }

    /// `body` laid out on pages `page_height` px tall with its
    /// `.header { position: running(pageHeader) }` elements registered as
    /// running, which the layout still gives their height.
    fn running_header_pass(
        body: &str,
        page_height: f32,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        use crate::gcpm::parser::parse_gcpm;
        use std::sync::Arc;

        let css = ".header { position: running(pageHeader); }";
        let html = format!(
            "<!DOCTYPE html><html><head><style>{css} div {{ margin: 0 }}</style></head>\
             <body style=\"margin: 0\">{body}</body></html>"
        );
        let gcpm = parse_gcpm(css);
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(&html, 600.0, &fonts);
        let pass = blitz_adapter::RunningElementPass::new(gcpm.running_mappings.clone());
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };
        blitz_adapter::apply_single_pass(&pass, &mut doc, &pass_ctx);
        let store = pass.into_running_store();
        blitz_adapter::resolve(&mut doc);
        let column_styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let (geometry, _) =
            run_pass_with_break_and_running(doc.deref_mut(), page_height, &column_styles, &store);
        (doc, geometry)
    }

    /// A `position: running()` element is out of the flow at every depth:
    /// in a block the walker enters, the sibling after it sits where it
    /// would without it, and the running element is placed with no height.
    #[test]
    fn a_nested_running_element_takes_no_room() {
        let (doc, geometry) = running_header_pass(
            r#"<div id="wrap">
                 <div id="header" class="header" style="height: 30px">Header</div>
                 <div id="a" style="height: 60px"></div>
                 <div id="b" style="height: 60px"></div>
               </div>"#,
            100.0,
        );
        let at = |id: &str| {
            geometry[&find_by_id(&doc, id).unwrap()]
                .fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                .collect::<Vec<_>>()
        };
        assert_eq!(at("header"), [(0, 0.0, 0.0)]);
        assert_eq!(at("a"), [(0, 0.0, 60.0)]);
        assert_eq!(at("b"), [(1, 0.0, 60.0)]);
    }

    /// Regression for the `--css` + hidden-ancestor running-element bug
    /// (FULGUR_CSS_FLAG_RUNNING_ELEMENT_BUG.md): `gcpm::parser::parse_gcpm`
    /// rewrites `position: running(name)` to `display: none` in its
    /// `cleaned_css` output (the real DOM copy must not also paint in
    /// normal flow — only its `@page` margin-box copy should). When the
    /// running element is nested inside a `position: absolute` ancestor
    /// (the common "absolute + invisible wrapper" header/footer idiom),
    /// its now-zero-size Taffy box used to fall through
    /// `record_subtree_fragments_at_offset`'s height gate
    /// (`h_for_paging > 0.0`) with no running-element carve-out — unlike
    /// `fragment_pagination_root`'s in-flow body walk, which records a
    /// zero-height marker for running children unconditionally. The
    /// element's NodeId never reached `geometry`, so
    /// `collect_running_element_states` silently produced nothing and the
    /// margin box rendered empty.
    #[test]
    fn running_element_nested_in_absolute_subtree_lands_in_geometry_when_display_none() {
        use crate::blitz_adapter;
        use crate::gcpm::parser::parse_gcpm;
        use std::ops::DerefMut;
        use std::sync::Arc;

        let css = r#"
            @page { @top-center { content: element(top-center); } }
            .absolute { position: absolute; }
            .invisible { visibility: hidden; }
            #top-center { position: running(top-center); }
        "#;
        let gcpm = parse_gcpm(css);
        assert!(
            gcpm.cleaned_css.contains("display: none"),
            "sanity: parse_gcpm must rewrite position:running to display:none \
             in cleaned_css; got {:?}",
            gcpm.cleaned_css
        );

        let html = r#"<!DOCTYPE html>
<html><body>
  <div class="absolute invisible">
    <div id="top-center">HEADER</div>
  </div>
  <p>Body content.</p>
</body></html>"#;

        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, 600.0, &fonts);
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };

        // Mirrors `Engine::layout_to_drawables`: cleaned_css (the
        // display:none rewrite) is injected via `InjectCssPass` — this is
        // exactly what happens for AssetBundle / `--css`-sourced CSS.
        let inject = blitz_adapter::InjectCssPass {
            css: gcpm.cleaned_css.clone(),
        };
        blitz_adapter::apply_single_pass(&inject, &mut doc, &pass_ctx);

        let running_pass = blitz_adapter::RunningElementPass::new(gcpm.running_mappings.clone());
        blitz_adapter::apply_single_pass(&running_pass, &mut doc, &pass_ctx);
        let store = running_pass.into_running_store();

        blitz_adapter::resolve(&mut doc);

        let mut geometry = PaginationGeometryTable::new();
        append_position_absolute_body_direct_fragments(
            &mut geometry,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            Some(&store),
        );

        let found = geometry
            .keys()
            .any(|&node_id| store.instance_for_node(node_id).is_some());
        assert!(
            found,
            "running element nested in a position:absolute subtree must land \
             in geometry even when display:none collapses its layout box; \
             geometry keys={:?}",
            geometry.keys().collect::<Vec<_>>()
        );

        let states = collect_running_element_states(&geometry, &store);
        let entry = states[0]
            .get("top-center")
            .expect("top-center running instance must be recorded for page 0");
        assert_eq!(entry.instance_ids.len(), 1);
    }

    /// Parse `html`, inject the `display: none` rewrite of `css`, collect
    /// running elements, and resolve layout — the setup shared by the
    /// absolute-subtree running-element tests.
    fn absolute_running_fixture(
        html: &str,
        css: &str,
    ) -> (
        blitz_html::HtmlDocument,
        crate::gcpm::running::RunningElementStore,
    ) {
        use crate::blitz_adapter;
        use crate::gcpm::parser::parse_gcpm;
        use std::sync::Arc;

        let gcpm = parse_gcpm(css);
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, 600.0, &fonts);
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };
        let inject = blitz_adapter::InjectCssPass {
            css: gcpm.cleaned_css.clone(),
        };
        blitz_adapter::apply_single_pass(&inject, &mut doc, &pass_ctx);
        let running_pass = blitz_adapter::RunningElementPass::new(gcpm.running_mappings.clone());
        blitz_adapter::apply_single_pass(&running_pass, &mut doc, &pass_ctx);
        let store = running_pass.into_running_store();
        blitz_adapter::resolve(&mut doc);
        (doc, store)
    }

    const ABSOLUTE_RUNNING_CSS: &str = r#"
        .absolute { position: absolute; }
        #late { position: running(late); }
    "#;

    /// A `display: none` running element has a zero-size box at its
    /// parent's origin, so `final_layout.location.y` is 0 rather than its
    /// source position. Inside an absolute wrapper the marker must follow
    /// the flow cursor past the preceding in-flow sibling (1000px, taller
    /// than the 800px page), landing on page 1 — not on the wrapper's start
    /// page — and the page budget must grow to hold it.
    #[test]
    fn running_element_in_absolute_subtree_is_placed_after_preceding_flow() {
        use std::ops::DerefMut;

        let html = r#"<!DOCTYPE html>
<html><body>
  <div class="absolute">
    <div style="height: 1000px">tall</div>
    <div id="late">LATE</div>
  </div>
</body></html>"#;
        let (mut doc, store) = absolute_running_fixture(html, ABSOLUTE_RUNNING_CSS);

        let mut geometry = PaginationGeometryTable::new();
        append_position_absolute_body_direct_fragments(
            &mut geometry,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            Some(&store),
        );

        let fragments: Vec<_> = geometry
            .iter()
            .filter(|(id, _)| store.instance_for_node(**id).is_some())
            .flat_map(|(_, g)| g.fragments.iter())
            .collect();
        assert_eq!(fragments.len(), 1, "one marker for the running element");
        assert_eq!(
            fragments[0].page_index, 1,
            "marker sits where the flow cursor lands, not at the wrapper start"
        );
    }

    /// The marker replaces, not extends, whatever an earlier pass stored for
    /// the running node, like every other fragment this walk records.
    #[test]
    fn running_element_marker_in_absolute_subtree_replaces_stale_fragments() {
        use std::ops::DerefMut;

        let html = r#"<!DOCTYPE html>
<html><body>
  <div class="absolute"><div id="late">LATE</div></div>
</body></html>"#;
        let (mut doc, store) = absolute_running_fixture(html, ABSOLUTE_RUNNING_CSS);

        let mut geometry = PaginationGeometryTable::new();
        for _ in 0..2 {
            append_position_absolute_body_direct_fragments(
                &mut geometry,
                doc.deref_mut(),
                1,
                600.0,
                800.0,
                Some(&store),
            );
        }

        let counts: Vec<usize> = geometry
            .iter()
            .filter(|(id, _)| store.instance_for_node(**id).is_some())
            .map(|(_, g)| g.fragments.len())
            .collect();
        assert_eq!(counts, vec![1], "a second pass must not stack markers");
    }

    /// fulgur-6tco: synthesize a geometry table + string_set_by_node
    /// map and verify `collect_string_set_states` produces the expected
    /// per-page `(start, first, last)` shape.
    #[test]
    fn string_set_state_carries_across_pages() {
        use super::StringSetPageState;

        // Three nodes: A on page 0, B on page 0, C on page 1.
        // A sets header="a", B sets header="b" (so first/last on page 0
        // differ), C sets nothing — page 1 inherits "b" via carry.
        let mut geom = PaginationGeometryTable::new();
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        geom.entry(20).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 50.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        geom.entry(30).or_default().fragments.push(Fragment {
            page_index: 1,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });

        let mut markers: BTreeMap<usize, Vec<(String, String)>> = BTreeMap::new();
        markers.insert(10, vec![("header".into(), "a".into())]);
        markers.insert(20, vec![("header".into(), "b".into())]);

        let states = super::collect_string_set_states(&geom, &markers);
        assert_eq!(states.len(), 2);

        // Page 0: no carry (first page), first set by A, last updated by B.
        let p0 = &states[0]["header"];
        assert_eq!(
            *p0,
            StringSetPageState {
                start: None,
                first: Some("a".into()),
                last: Some("b".into()),
            }
        );
        // Page 1: carry from p0.last ("b"). C sets nothing → first/last stay None.
        let p1 = &states[1]["header"];
        assert_eq!(
            *p1,
            StringSetPageState {
                start: Some("b".into()),
                first: None,
                last: None,
            }
        );
    }

    #[test]
    fn string_set_first_appearance_only_for_split_paragraph() {
        // A node spans two pages (inline-aware split). Markers fire
        // only on the first appearance.
        use super::StringSetPageState;

        let mut geom = PaginationGeometryTable::new();
        geom.entry(42).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 800.0_f32.as_px(),
        });
        geom.entry(42).or_default().fragments.push(Fragment {
            page_index: 1,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 200.0_f32.as_px(),
        });

        let mut markers: BTreeMap<usize, Vec<(String, String)>> = BTreeMap::new();
        markers.insert(42, vec![("title".into(), "intro".into())]);

        let states = super::collect_string_set_states(&geom, &markers);
        assert_eq!(states.len(), 2);
        assert_eq!(
            states[0]["title"],
            StringSetPageState {
                start: None,
                first: Some("intro".into()),
                last: Some("intro".into()),
            }
        );
        assert_eq!(
            states[1]["title"],
            StringSetPageState {
                start: Some("intro".into()),
                first: None,
                last: None,
            }
        );
    }

    #[test]
    fn string_set_states_empty_geometry_returns_one_empty_page() {
        // "Always at least one page" convention: even an empty
        // geometry yields a single empty per-page state so downstream
        // consumers can index by page without special-casing zero.
        let geom = PaginationGeometryTable::new();
        let markers = BTreeMap::new();
        let states = super::collect_string_set_states(&geom, &markers);
        assert_eq!(states.len(), 1);
        assert!(states[0].is_empty());
    }

    /// fulgur-jkl5: `position: fixed` element should emit one
    /// fragment per page so downstream rendering can repeat it.
    #[test]
    fn position_fixed_repeats_per_page() {
        let html = r#"
            <html><body>
              <div style="height: 600px"></div>
              <div style="height: 600px"></div>
              <div style="position: fixed; top: 10px; left: 20px;
                          width: 100px; height: 50px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);

        let mut geom = super::run_pass(doc.deref_mut(), 800.0);
        let pages_before = super::implied_page_count(&geom);
        assert!(
            pages_before >= 2,
            "two 600px blocks on 800px page should split → {pages_before} pages",
        );

        super::append_position_fixed_fragments(
            &mut geom,
            doc.deref_mut(),
            pages_before,
            600.0,
            800.0,
        );

        // The fixed div should now appear in `geom` with one fragment
        // per page. We don't know its NodeId statically, so locate it
        // by the per-fragment width = 100.0.
        let fixed_entries: Vec<_> = geom
            .iter()
            .filter(|(_, g)| {
                g.fragments.iter().any(|f| {
                    (f.width.to_f32() - 100.0).abs() < 0.5 && (f.height.to_f32() - 50.0).abs() < 0.5
                })
            })
            .collect();
        assert_eq!(
            fixed_entries.len(),
            1,
            "exactly one fixed element entry expected, got {}",
            fixed_entries.len()
        );
        let (_, fixed_geom) = fixed_entries[0];
        assert_eq!(
            fixed_geom.fragments.len() as u32,
            pages_before,
            "fixed element should have one fragment per page",
        );
        let pages_seen: Vec<u32> = fixed_geom.fragments.iter().map(|f| f.page_index).collect();
        assert_eq!(pages_seen, (0..pages_before).collect::<Vec<_>>());
    }

    #[test]
    fn position_fixed_with_no_pages_normalises_to_one_page() {
        // append_position_fixed_fragments(geom, doc, 0) should still
        // emit exactly one fragment per fixed element (the "always
        // at least one page" convention applied to fixed
        // repetition).
        let html = r#"
            <html><body>
              <div style="position: fixed; top: 0; left: 0;
                          width: 50px; height: 30px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_fixed_fragments(&mut geom, doc.deref_mut(), 0, 600.0, 800.0);
        assert_eq!(geom.len(), 1);
        let (_, g) = geom.iter().next().unwrap();
        assert_eq!(g.fragments.len(), 1);
        assert_eq!(g.fragments[0].page_index, 0);
    }

    /// fulgur-a8m5: `append_position_fixed_fragments` must resolve
    /// `bottom: 0` against the viewport CB. Taffy's `compute_root_layout`
    /// (used by `relayout_position_fixed`) does not honour end-side
    /// insets when the absolute element IS the layout-tree root, so
    /// `final_layout.location.y` stays at 0 even for `bottom: 0`. v2's
    /// dispatch reads `pagination_geometry` directly, so without inset
    /// resolution here, WPT fixedpos-001 / fixedpos-002 / fixedpos-008
    /// render their `bottom: 0` fixed text at the top of every page
    /// instead of the bottom.
    #[test]
    fn position_fixed_bottom_zero_resolves_against_viewport() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: fixed; bottom: 0; height: 30px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        crate::blitz_adapter::relayout_position_fixed(&mut doc, 600.0, 800.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_fixed_fragments(&mut geom, doc.deref_mut(), 1, 600.0, 800.0);

        // Locate the fixed div fragment by its 30px height.
        let entries: Vec<_> = geom
            .iter()
            .filter(|(_, g)| {
                g.fragments
                    .iter()
                    .any(|f| (f.height.to_f32() - 30.0).abs() < 0.5)
            })
            .collect();
        assert_eq!(entries.len(), 1, "exactly one fixed entry");
        let (_, g) = entries[0];
        assert_eq!(g.fragments.len(), 1);
        let frag = &g.fragments[0];
        // viewport_h_px=800, height=30 → bottom edge sits at 800 → top at 770.
        // body has zero height (no in-flow content), so body_offset_xy=(0,0).
        let frag_y = frag.y.to_f32();
        assert!(
            (frag_y - 770.0).abs() < 1.0,
            "bottom:0 fixed should resolve to y=770 (viewport_h - height); got {frag_y}",
        );
    }

    /// fulgur-orcx: `position: fixed` uses `bottom` anchoring against
    /// viewport-anchored CB. When both values are sub-pixel, resolving
    /// via the exact computed formula and rounding order used by Taffy
    /// keeps fixed root and abs-in-relative reference paths aligned.
    /// Without this, fixed-path can land 0.5px lower and fail
    /// WPT fixedpos-009-print by halo edge.
    #[test]
    fn position_fixed_bottom_zero_rounds_like_taffy_with_fractional_viewport_height() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: fixed; right: 0; bottom: 0; width: 36px; height: 36px">
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        crate::blitz_adapter::relayout_position_fixed(&mut doc, 600.0, 800.6);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_fixed_fragments(&mut geom, doc.deref_mut(), 1, 600.0, 800.6);

        let entries: Vec<_> = geom
            .iter()
            .filter(|(_, g)| {
                g.fragments
                    .iter()
                    .any(|f| (f.width.to_f32() - 36.0).abs() < 0.5)
            })
            .collect();
        assert_eq!(entries.len(), 1, "expected one fixed entry");
        let frag = entries[0].1.fragments.first().unwrap();
        assert_eq!(frag.page_index, 0, "expected fixed fragment only on page 0");
        // With Taffy-like rounding: round(cb_h - b) - round(h) => round(800.6)-36 = 765.
        let frag_y = frag.y.to_f32();
        assert!(
            (frag_y - 765.0).abs() < 0.25,
            "fractional fixed bottom anchor should resolve to y≈765; got {frag_y}",
        );
    }

    #[test]
    fn position_absolute_body_direct_bottom_viewport_page_stride_keeps_page_local_y_stable() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; bottom: -971px; height: 19px">x</div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            3,
            600.0,
            971.338_87,
            None,
        );

        let frag = geom
            .values()
            .flat_map(|g| &g.fragments)
            .find(|f| f.page_index == 1 && (f.height.to_f32() - 19.0).abs() < 0.5)
            .expect("absolute bottom:-viewport fragment should land on page 1");
        let frag_y = frag.y.to_f32();
        assert!(
            (frag_y - 952.0).abs() < 0.01,
            "absolute ref fragment should keep the same page-local bottom anchor as fixed; got {frag_y}",
        );
    }

    /// fulgur-a8m5: body's collapsed-margin offset (e.g. an in-flow
    /// child with `margin-top:4em`) appears in
    /// `drawables.body_offset_pt`, which the v2 dispatch path adds to
    /// every fragment's y. Viewport-anchored fixed elements must
    /// subtract that offset at storage time so the dispatched y lands
    /// at the page-relative position the CSS asks for. Locks the math
    /// PDF y = margin_top_pt + body_offset_pt + (frag.y_pt) — for
    /// `top:0` we want PDF y = margin_top_pt, so frag.y_px must equal
    /// `-body_offset_y_px`.
    #[test]
    fn position_fixed_top_zero_compensates_for_body_offset() {
        // The in-flow div pushes body's content area down by ~4em.
        let html = r#"
            <html><body>
              <div style="margin-top:4em">x</div>
              <div style="position: fixed; top: 0; height: 30px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        crate::blitz_adapter::relayout_position_fixed(&mut doc, 600.0, 800.0);
        let body_y_px = super::body_origin_in_px(doc.deref_mut()).1;
        // Sanity: body_offset must be non-zero, otherwise the test is
        // not exercising compensation at all.
        assert!(
            body_y_px > 0.5,
            "test assumes body has a non-zero offset; got {body_y_px}"
        );
        let mut geom = PaginationGeometryTable::new();
        super::append_position_fixed_fragments(&mut geom, doc.deref_mut(), 1, 600.0, 800.0);

        let entries: Vec<_> = geom
            .iter()
            .filter(|(_, g)| {
                g.fragments
                    .iter()
                    .any(|f| (f.height.to_f32() - 30.0).abs() < 0.5)
            })
            .collect();
        assert_eq!(entries.len(), 1, "exactly one fixed entry");
        let frag = entries[0].1.fragments.first().unwrap();
        // top:0 → resolved_y=0 → stored_y = 0 - body_y_px = -body_y_px.
        let frag_y = frag.y.to_f32();
        assert!(
            (frag_y - (-body_y_px)).abs() < 0.5,
            "top:0 fixed frag.y must be -body_offset (={}); got {}",
            -body_y_px,
            frag_y
        );
    }

    /// fulgur-4m16: when a `position: fixed` root has a sized
    /// block-element child, the child must also receive per-page
    /// repeated fragments. v2 dispatch reads
    /// `pagination_geometry[node_id]` and never recurses into a fixed
    /// root's subtree, so without per-descendant fragments the child
    /// is never drawn (WPT fixedpos-009: a `<div class="pencil"
    /// style="width:36px; height:36px">` inside the fixed root never
    /// renders, leaving every page blank where the pencil should be).
    #[test]
    fn position_fixed_emits_fragments_for_block_descendants() {
        // Use shrink-to-fit-friendly width / height on the fixed root
        // so Taffy gives it a definite size; otherwise an unsized fixed
        // root inherits available_space (600 wide), which obscures the
        // root-vs-child position relationship the test pins. The bug
        // being tested (descendants missing from geometry) is
        // independent of root sizing.
        let html = r#"
            <html><body style="margin:0">
              <div style="position: fixed; bottom: 0; width: 36px; height: 36px">
                <div style="width: 36px; height: 36px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        crate::blitz_adapter::relayout_position_fixed(&mut doc, 600.0, 800.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_fixed_fragments(&mut geom, doc.deref_mut(), 2, 600.0, 800.0);

        // Both root and its 36×36 child must appear in `geom`. The
        // root's fragments come from the existing inset-resolution
        // path; the child's fragments come from the new descendants
        // walker (fulgur-4m16). Without that walker, only one entry
        // (the root) shows up.
        let entries: Vec<_> = geom
            .iter()
            .filter(|(_, g)| {
                g.fragments.iter().any(|f| {
                    (f.width.to_f32() - 36.0).abs() < 0.5 && (f.height.to_f32() - 36.0).abs() < 0.5
                })
            })
            .collect();
        assert_eq!(
            entries.len(),
            2,
            "expected 2 entries (fixed root + child), got {} — fulgur-4m16: \
             without record_fixed_subtree_descendants the child entry is missing",
            entries.len(),
        );

        // Both must be `is_repeat = true` with one fragment per page.
        for (_, g) in &entries {
            assert!(g.is_repeat, "entry must be is_repeat=true");
            assert_eq!(g.fragments.len(), 2, "entry: one fragment per page");
            let pages_seen: Vec<u32> = g.fragments.iter().map(|f| f.page_index).collect();
            assert_eq!(pages_seen, vec![0u32, 1u32]);
        }

        // Both entries' first fragments must agree on (x, y) because
        // the child's `final_layout.location` inside the root is
        // (0, 0) (no padding/margin/border on the root). A divergence
        // here would mean the descendants walker is computing offsets
        // wrong.
        let f0 = &entries[0].1.fragments[0];
        let f1 = &entries[1].1.fragments[0];
        let f0x = f0.x.to_f32();
        let f0y = f0.y.to_f32();
        let f1x = f1.x.to_f32();
        let f1y = f1.y.to_f32();
        assert!(
            (f0x - f1x).abs() < 0.5 && (f0y - f1y).abs() < 0.5,
            "root and child must share (x, y); got root=({f0x},{f0y}) child=({f1x},{f1y})",
        );

        // Pin the y coordinate: bottom:0 with height=36 in an 800px
        // viewport places the box top at y=764. body is empty so
        // body_offset_xy=(0,0).
        assert!(
            (f0y - 764.0).abs() < 1.0,
            "bottom:0 fixed (h=36) must resolve to y=764 (viewport_h - h); got {f0y}",
        );
    }

    /// fulgur-a8m5: body-direct `position: absolute` with `bottom: 0`
    /// should land at the bottom of page 0 with its descendant Paragraph
    /// fragments at the same position. The fragmenter unconditionally
    /// skips out-of-flow children, so this pass is the only thing that
    /// puts these abs body-direct nodes into `pagination_geometry` for
    /// v2 dispatch (WPT fixedpos-001 ref, fixedpos-008 ref).
    #[test]
    fn position_absolute_body_direct_bottom_zero_lands_on_page_zero() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; bottom: 0; height: 30px">x</div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            None,
        );

        // Locate the abs div by height=30.
        let mut found = None;
        for g in geom.values() {
            for f in &g.fragments {
                if (f.height.to_f32() - 30.0).abs() < 0.5 {
                    found = Some(f.clone());
                }
            }
        }
        let frag = found.expect("abs body-direct fragment for bottom:0 must be emitted");
        assert_eq!(frag.page_index, 0);
        // Same math as the fixed case: body has zero height, so
        // body_offset compensation is a no-op and viewport-CB
        // resolution gives y = 800 - 30 = 770.
        let frag_y = frag.y.to_f32();
        assert!(
            (frag_y - 770.0).abs() < 1.0,
            "abs body-direct bottom:0 should land at y=770; got {frag_y}",
        );
    }

    /// fulgur-z4zc: a body-direct `position:absolute` subtree whose
    /// viewport-CB y range crosses page boundaries must emit geometry on
    /// every intersected page. The main fragmenter skips absolute OOF
    /// children, so `append_position_absolute_body_direct_fragments` is
    /// responsible for recording these page-local placements.
    #[test]
    fn position_absolute_body_direct_overflow_emits_fragments_on_each_page() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; top: 0; width: 100px; height: 1800px">
                <div style="width: 50px; height: 1800px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            None,
        );

        let mut tall_entries: Vec<Vec<u32>> = geom
            .values()
            .filter(|g| {
                g.fragments
                    .iter()
                    .any(|f| (f.height.to_f32() - 1800.0).abs() < 0.5)
            })
            .map(|g| {
                let mut pages: Vec<u32> = g.fragments.iter().map(|f| f.page_index).collect();
                pages.sort_unstable();
                pages
            })
            .collect();
        tall_entries.sort();

        assert_eq!(
            tall_entries,
            vec![vec![0, 1, 2], vec![0, 1, 2]],
            "expected absolute root and in-flow child to emit fragments on pages 0, 1, and 2; got {tall_entries:?}"
        );
    }

    /// fulgur-2m6w: a body-direct `position:absolute` node positioned far
    /// beyond the page budget (`top: 99999999px`) must not extend the page
    /// count without bound. The abs path emits one fragment per intersected
    /// page and `descendant_total_pages` feeds `implied_page_count`, so an
    /// unclamped page index from a tiny box makes `render_v2` allocate and
    /// render ~10^5 pages — the same small-input DoS the body-direct slice
    /// cap blocks, reached via the absolute-positioning path (Codex review
    /// on PR #501). Without the clamp this input lands at page ~125000;
    /// page indices are clamped to `MAX_PAGES`.
    #[test]
    fn position_absolute_far_offset_is_page_capped() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; top: 99999999px; width: 10px; height: 10px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            None,
        );
        // The fragment must still be emitted (not silently dropped) — just
        // pinned to the cap page rather than page ~125000.
        let max_page = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .map(|f| f.page_index)
            .max()
            .expect("abs fragment must be emitted");
        assert!(
            max_page <= crate::MAX_PAGES,
            "abs page index must be clamped to MAX_PAGES ({}), got {max_page}",
            crate::MAX_PAGES,
        );
    }

    /// fulgur-2m6w (Codex review on PR #501): the `MAX_PAGES` clamp must
    /// apply ONLY to the page-extension path. When `total_pages` legitimately
    /// exceeds `MAX_PAGES` (many ordinary in-flow pages — input-proportional,
    /// not amplified), a NON-extending absolute element starting at an
    /// in-budget page above the cap must emit a single fragment at its real
    /// page, not a spurious run from `MAX_PAGES` through the real page.
    #[test]
    fn position_absolute_in_budget_start_above_cap_not_clamped() {
        let page_h = 800.0_f32;
        let start_page = crate::MAX_PAGES + 20_000; // > MAX_PAGES
        let total_pages = start_page + 5; // in-budget
        let top = page_h * start_page as f32;
        // In-flow `<p>` makes `body_has_in_flow_content` true, so
        // `may_extend_pages` is false → the non-extending branch.
        let html = format!(
            r#"<html><body style="margin:0">
                 <p>in-flow</p>
                 <div style="position:absolute; top:{top}px; width:10px; height:10px"></div>
               </body></html>"#
        );
        let mut doc = parse(&html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            total_pages,
            600.0,
            page_h,
            None,
        );
        // Isolate the 10px-wide abs node's fragments.
        let abs_pages: Vec<u32> = geom
            .values()
            .filter(|g| {
                g.fragments
                    .iter()
                    .any(|f| (f.width.to_f32() - 10.0).abs() < 0.5)
            })
            .flat_map(|g| g.fragments.iter().map(|f| f.page_index))
            .collect();
        assert_eq!(
            abs_pages,
            vec![start_page],
            "non-extending in-budget abs must emit only at its real page \
             (no spurious clamp run), got {abs_pages:?}",
        );
    }

    /// `html` through the absolute-positioning pass alone, on pages
    /// 600 x 100 px, of which the flow laid out `total_pages`.
    fn absolute_pass(
        html: &str,
        total_pages: u32,
    ) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            total_pages,
            600.0,
            100.0,
            None,
        );
        (doc, geom)
    }

    /// The page, y, and height of each fragment of the element `#id`.
    fn fragments_of(
        doc: &blitz_dom::BaseDocument,
        geom: &PaginationGeometryTable,
        id: &str,
    ) -> Vec<(u32, f32, f32)> {
        geom.get(&find_by_id(doc, id).unwrap())
            .map(|g| {
                g.fragments
                    .iter()
                    .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The pages of the fragments of the element `#id`.
    fn pages_of(
        doc: &blitz_dom::BaseDocument,
        geom: &PaginationGeometryTable,
        id: &str,
    ) -> Vec<u32> {
        fragments_of(doc, geom, id).iter().map(|f| f.0).collect()
    }

    /// A size-contained box taller than a page in an absolute subtree is
    /// sliced into one fragment a page, as block flow slices it, and what
    /// follows it lands after its last slice, extending the page count
    /// (WPT `monolithic-overflow-013-print`).
    #[test]
    fn a_tall_box_in_an_absolute_subtree_is_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <div style="position: absolute; top: 0; width: 100%">
                <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
                <p id="after" style="margin: 0">after</p>
              </div>
            </body></html>"#,
            1,
        );
        assert_eq!(
            fragments_of(&doc, &geom, "tall"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)]
        );
        let after: Vec<(u32, f32)> = fragments_of(&doc, &geom, "after")
            .iter()
            .map(|f| (f.0, f.1))
            .collect();
        assert_eq!(after, [(2, 50.0)]);
    }

    /// The boxes around a sliced box in an absolute subtree have one
    /// fragment a page that ends at the page foot and starts at the page
    /// top after the first, as block flow's do, so none paints into the
    /// page margins.
    #[test]
    fn the_boxes_around_a_sliced_box_in_an_absolute_subtree_end_at_each_page_foot() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <div id="root" style="position: absolute; top: 0; width: 100%">
                <div id="wrapper">
                  <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
                </div>
                <p style="margin: 0">after</p>
              </div>
            </body></html>"#,
            1,
        );
        let root_h = doc
            .get_node(find_by_id(&doc, "root").unwrap())
            .unwrap()
            .final_layout
            .size
            .height;
        assert_eq!(
            fragments_of(&doc, &geom, "root"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, root_h - 200.0)]
        );
        assert_eq!(
            fragments_of(&doc, &geom, "wrapper"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)]
        );
    }

    /// The slices of a tall box in an absolute subtree take pages past the
    /// flow's, as block flow's do, and the subtree around them reaches
    /// those pages too: none is left blank. A box after them that extends
    /// the pages further keeps the pages it reaches.
    #[test]
    fn a_sliced_box_in_an_absolute_subtree_extends_the_pages_of_the_flow() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <p style="margin: 0">in flow</p>
              <div id="root" style="position: absolute; top: 0; width: 100%">
                <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
                <div id="plain" style="width: 50px; height: 300px"></div>
              </div>
            </body></html>"#,
            1,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0, 1, 2]);
        assert_eq!(pages_of(&doc, &geom, "plain"), [2, 3, 4, 5]);
        assert_eq!(pages_of(&doc, &geom, "root"), [0, 1, 2, 3, 4, 5]);
    }

    /// A table cell is no place block flow slices a box, in an absolute
    /// subtree too: the tall box keeps one fragment on its first page, and
    /// what follows it in the cell moves up by its excess over a page.
    #[test]
    fn a_tall_box_in_a_table_cell_of_an_absolute_subtree_is_not_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <div style="position: absolute; top: 0; width: 100%">
                <table style="border-spacing: 0"><tr><td style="padding: 0">
                  <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
                  <div id="after" style="width: 50px; height: 10px"></div>
                </td></tr></table>
              </div>
            </body></html>"#,
            1,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0]);
        assert_eq!(fragments_of(&doc, &geom, "after"), [(1, 0.0, 10.0)]);
    }

    /// The root of an absolute subtree is no item of a flex body, so an
    /// in-flow item beside it does not keep its tall box whole.
    #[test]
    fn a_tall_box_in_an_absolute_child_of_a_flex_body_is_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0; display: flex">
              <div style="width: 50px; height: 50px"></div>
              <div style="position: absolute; top: 0; left: 100px; width: 50px">
                <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
              </div>
            </body></html>"#,
            3,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0, 1, 2]);
    }

    /// A tall root of an absolute subtree is no item of a flex body either.
    #[test]
    fn a_tall_absolute_child_of_a_flex_body_is_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0; display: flex">
              <div style="width: 50px; height: 50px"></div>
              <div id="tall" style="position: absolute; top: 0; left: 100px;
                  contain: size; width: 50px; height: 250px"></div>
            </body></html>"#,
            3,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0, 1, 2]);
    }

    /// A tall absolute box nested in a flex row of an absolute subtree is
    /// no item of that row.
    #[test]
    fn a_tall_absolute_box_in_a_flex_row_is_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <div style="position: absolute; top: 0; width: 200px; display: flex">
                <div style="width: 50px; height: 50px"></div>
                <div id="tall" style="position: absolute; top: 0; left: 100px;
                    contain: size; width: 50px; height: 250px"></div>
              </div>
            </body></html>"#,
            1,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0, 1, 2]);
    }

    /// A clip above a nested absolute box still hides what is past it: the
    /// tall box in the nested box is not sliced past an outer absolute box
    /// that clips it.
    #[test]
    fn a_tall_box_past_a_clip_above_a_nested_absolute_box_is_not_sliced() {
        let (doc, geom) = absolute_pass(
            r#"<html><body style="margin: 0">
              <div style="position: absolute; top: 0; width: 100px;
                  height: 100px; overflow: hidden">
                <div style="position: absolute; top: 0; width: 50px">
                  <div id="tall" style="contain: size; width: 50px; height: 250px"></div>
                </div>
              </div>
            </body></html>"#,
            1,
        );
        assert_eq!(pages_of(&doc, &geom, "tall"), [0]);
    }

    #[test]
    fn position_fixed_repeats_on_pages_added_by_body_direct_absolute() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: fixed; top: 0; width: 10px; height: 20px"></div>
              <div style="position: absolute; top: 0; width: 100px; height: 1200px"></div>
            </body></html>
        "#;
        let engine = crate::Engine::builder().build();
        let (_, geom) = engine.build_drawables_and_geometry_for_testing_no_gcpm(html);

        let fixed_pages = geom
            .values()
            .find(|g| {
                g.is_repeat
                    && g.fragments.iter().any(|f| {
                        (f.width.to_f32() - 10.0).abs() < 0.5
                            && (f.height.to_f32() - 20.0).abs() < 0.5
                    })
            })
            .map(|g| {
                let mut pages: Vec<u32> = g.fragments.iter().map(|f| f.page_index).collect();
                pages.sort_unstable();
                pages
            });

        assert_eq!(fixed_pages, Some(vec![0, 1]));
    }

    #[test]
    fn fixed_subtree_fragment_total_is_bounded() {
        // Security regression: a `position: fixed` root and every in-flow
        // descendant each emit one fragment per page → O(nodes × pages).
        // The aggregate MAX_SUBTREE_PAGE_FRAGMENTS budget caps the retained
        // total regardless of how many descendants the subtree has.
        let mut kids = String::new();
        for _ in 0..20_000 {
            kids.push_str(r#"<div style="height:1px">x</div>"#);
        }
        let html = format!(
            r#"<html><body style="margin:0">
              <div style="position: fixed; top:0; width:50px; height:50px">{kids}</div>
              <div style="height:100000px"></div>
            </body></html>"#
        );
        let engine = crate::Engine::builder().build();
        let (_, geom) = engine.build_drawables_and_geometry_for_testing_no_gcpm(&html);
        let total: usize = geom
            .values()
            .filter(|g| g.is_repeat)
            .map(|g| g.fragments.len())
            .sum();
        assert!(
            total <= crate::MAX_SUBTREE_PAGE_FRAGMENTS,
            "fixed fragment total must be bounded, got {total}"
        );
    }

    #[test]
    fn absolute_subtree_fragment_total_is_bounded() {
        // Security regression (same class as fixed-subtree fragments): the
        // body-direct absolute pass records, for every node in each absolute
        // subtree, one fragment per page the node intersects
        // (`first_page..=last_page`, bounded per-node by MAX_PAGES). Many
        // page-spanning absolute elements amplify to O(nodes × pages). The
        // aggregate MAX_SUBTREE_PAGE_FRAGMENTS budget caps the retained total.
        let mut sibs = String::new();
        for _ in 0..15_000 {
            sibs.push_str(
                r#"<div style="position:absolute; top:0; width:10px; height:200000px"></div>"#,
            );
        }
        let html = format!(r#"<html><body style="margin:0">{sibs}</body></html>"#);
        let engine = crate::Engine::builder().build();
        let (_, geom) = engine.build_drawables_and_geometry_for_testing_no_gcpm(&html);
        let total: usize = geom.values().map(|g| g.fragments.len()).sum();
        // The absolute pass is capped at MAX_SUBTREE_PAGE_FRAGMENTS; the sum
        // also counts incidental non-absolute fragments (root/body/in-flow),
        // which are themselves bounded by MAX_PAGES (a single in-flow flow).
        // Uncapped this scenario emits ~3M fragments, so the budget clearly
        // fired.
        assert!(
            total <= crate::MAX_SUBTREE_PAGE_FRAGMENTS + crate::MAX_PAGES as usize,
            "absolute fragment total must be bounded, got {total}"
        );
    }

    #[test]
    fn position_absolute_body_direct_tiny_overflow_reaches_next_page() {
        let html = r#"
            <html><body style="margin:0">
              <div id="tiny" style="position: absolute; top: 0; width: 100px; height: 801px"></div>
              <div id="exact" style="position: absolute; top: 0; width: 100px; height: 800px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        // Reuses the module-level `find_by_id` helper (see the
        // pagination-cap tests) instead of redefining it here.
        let tiny_id = find_by_id(doc.deref_mut(), "tiny").expect("tiny abs node");
        doc.deref_mut()
            .get_node_mut(tiny_id)
            .expect("tiny abs node")
            .final_layout
            .size
            .height = 800.1;
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            2,
            600.0,
            800.0,
            None,
        );

        let mut tiny_overflow_pages = None;
        let mut exact_boundary_pages = None;
        for g in geom.values() {
            if g.fragments
                .iter()
                .any(|f| (f.height.to_f32() - 800.1).abs() < 0.05)
            {
                let mut pages: Vec<u32> = g.fragments.iter().map(|f| f.page_index).collect();
                pages.sort_unstable();
                tiny_overflow_pages = Some(pages);
            }
            if g.fragments
                .iter()
                .any(|f| (f.height.to_f32() - 800.0).abs() < 0.05)
            {
                let mut pages: Vec<u32> = g.fragments.iter().map(|f| f.page_index).collect();
                pages.sort_unstable();
                exact_boundary_pages = Some(pages);
            }
        }

        assert_eq!(tiny_overflow_pages, Some(vec![0, 1]));
        assert_eq!(exact_boundary_pages, Some(vec![0]));
    }

    #[test]
    fn position_absolute_body_direct_height_overflow_extends_to_last_page() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; top: 0; width: 100px; height: 1px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let abs_id = find_node_by_local_name(&doc, "div").expect("abs div");
        doc.deref_mut()
            .get_node_mut(abs_id)
            .expect("abs div")
            .final_layout
            .size
            .height = f32::MAX;
        let mut geom = PaginationGeometryTable::new();
        super::record_subtree_fragments_at_offset(
            &mut geom,
            doc.deref_mut(),
            abs_id,
            (0.0, f32::MAX),
            (0.0, 0.0),
            f32::MAX,
            f32::MAX,
            3,
            true,
            &mut 0,
            None,
        );

        let pages: Vec<u32> = geom
            .get(&abs_id)
            .expect("abs div geometry")
            .fragments
            .iter()
            .map(|f| f.page_index)
            .collect();
        assert_eq!(pages, vec![1, 2]);
    }

    /// Exercises `PaginationLayoutTree`'s `LayoutPartialTree` /
    /// `RoundTree` / `CacheTree` / `TraversePartialTree` impls
    /// at runtime by routing body's layout through
    /// `taffy::compute_root_layout`. Production reaches geometry via
    /// `fragment_pagination_root` directly (see the docstring on
    /// `drive_taffy_root_layout` for why), so this test is the only
    /// runtime user of those trait impls — without it, `cargo build`
    /// would still type-check the impls but no code path would actually
    /// invoke them. Asserts the geometry the Taffy-driven path produces
    /// matches the direct walk used in production.
    ///
    /// Both sides feed the same `ColumnStyleTable` so the parity check
    /// covers the break-style-aware code path that production wires
    /// through `run_pass_with_break_styles`. The fixture sets
    /// `break-before: page` on the middle child so the geometry differs
    /// from the style-unaware case (without the table all three blocks
    /// pack onto page 0; with it, the middle block opens page 1).
    #[test]
    fn taffy_driven_dispatch_matches_direct_walk() {
        let html = r#"
            <html><body>
              <div style="height: 200px"></div>
              <div style="break-before: page; height: 200px"></div>
              <div style="height: 200px"></div>
            </body></html>
        "#;

        let direct_geom = {
            let mut doc = parse(html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table)
        };

        let taffy_geom = {
            let mut doc = parse(html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let mut tree = PaginationLayoutTree::new(doc.deref_mut(), 800.0);
            tree.column_styles = Some(&table);
            tree.drive_taffy_root_layout();
            tree.take_geometry()
        };

        // Sanity: the break-* branch actually fired — page_index 1
        // appears at least once in the direct geometry.
        assert!(
            direct_geom
                .values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "expected break-before: page to push a child onto page 1, got {direct_geom:?}"
        );

        assert_eq!(direct_geom.len(), taffy_geom.len());
        for (id, direct) in &direct_geom {
            let taffy = taffy_geom.get(id).expect("same node id in both passes");
            assert_eq!(direct.fragments, taffy.fragments, "node {id}");
        }
    }

    /// fulgur-kv0r: parallel siblings in a grid / flex parent should
    /// share the same page-local y when they share Taffy's
    /// `layout.location.y`. Pre-fix, `fragment_block_subtree`
    /// advanced `cursor_y` after each child via `cursor_y += child_h`,
    /// so card 2 (Taffy y=0) was recorded at y=200 (= card 1's
    /// height) in geometry. Post-fix the loop reads
    /// `child_page_y = page_start_y + (this_top_in_parent - page_taffy_origin)`
    /// directly from Taffy, and updates `cursor_y` only as a row's
    /// max bottom for break / overflow checks.
    #[test]
    fn fragment_block_subtree_grid_parallel_siblings_share_page_y() {
        // Two cells in a 2-column grid row, each 100px tall and
        // 100px wide so the grid container distinguishes them by x.
        // Pre-fix: card 2 placed at y=100 (cursor-advanced after
        // card 1). Post-fix: card 2 placed at y=0 (Taffy `location.y`).
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                <div style="height: 100px; width: 100px"></div>
                <div style="height: 100px; width: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);

        // Filter to fragments whose width is exactly the cell width
        // (100) — that's only the two cards. Grid container has
        // width 200 (two columns), html / body have viewport width.
        let card_y: Vec<f32> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                f.page_index == 0
                    && (f.height.to_f32() - 100.0).abs() < 0.5
                    && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| f.y.to_f32())
            .collect();
        assert_eq!(
            card_y.len(),
            2,
            "expected two grid cells (100×100) on page 0, got {card_y:?}"
        );
        for y in &card_y {
            assert!(
                y.abs() < 0.5,
                "grid parallel siblings must share y=0, got {y} (pre-fix: card 2 at y=100 due \
                 to cursor-advance)",
            );
        }
    }

    #[test]
    fn fragment_block_subtree_following_block_continues_after_split_child_tail() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
                <h2 style="height: 30px; width: 200px; margin: 0">after grid</h2>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut candidates: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.height.to_f32() - 30.0).abs() < 0.5)
            .collect();
        candidates.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        let h2 = candidates
            .first()
            .expect("expected the trailing h2 to land on page 1");
        let h2_y = h2.y.to_f32();
        assert!(
            h2_y >= 100.0,
            "block sibling after split grid must continue after the grid tail; got y={h2_y} (pre-fix: y=0 overlaps the tail)",
        );
    }

    /// fulgur-2m6w: a nested child with a non-finite Taffy height must not
    /// poison `cursor_y` inside the recursive splitter. The grid pattern
    /// routes the section through `fragment_block_subtree` (250px content vs
    /// 150px strip); a `+inf` height injected into one cell is sanitized to
    /// zero so the recursion neither panics nor produces a non-finite /
    /// unbounded page count, and the node is still recorded in geometry.
    #[test]
    fn fragment_block_subtree_non_finite_child_height_is_sanitized() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                  <div id="bad" style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let bad = find_by_id(doc.deref_mut(), "bad").expect("div#bad");
        doc.deref_mut()
            .get_node_mut(bad)
            .expect("div#bad")
            .final_layout
            .size
            .height = f32::INFINITY;
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 150.0_f32.as_px(), &table);
        // Load-bearing assertion: without the guard the injected `+inf`
        // reaches a Fragment height and poisons `cursor_y`, so emitted
        // fragments carry non-finite `y` / `height`. The guard zeroes it,
        // keeping every emitted coordinate finite.
        for (id, g) in geom.iter() {
            for f in &g.fragments {
                let fy = f.y.to_f32();
                let fh = f.height.to_f32();
                assert!(
                    fy.is_finite() && fh.is_finite(),
                    "node {id}: non-finite fragment y={fy} height={fh} leaked through \
                     fragment_block_subtree",
                );
            }
        }
        assert!(
            geom.contains_key(&bad),
            "nested non-finite-height node must still be recorded",
        );
    }

    #[test]
    fn fragment_block_subtree_following_block_continues_after_split_flex_tail() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: flex; flex-wrap: wrap; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
                <h2 style="height: 30px; width: 200px; margin: 0">after flex</h2>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut candidates: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.height.to_f32() - 30.0).abs() < 0.5)
            .collect();
        candidates.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        let h2 = candidates
            .first()
            .expect("expected the trailing h2 to land on page 1");
        let h2_y = h2.y.to_f32();
        assert!(
            h2_y >= 100.0,
            "block sibling after split flex must continue after the flex tail; got y={h2_y} (pre-fix: y=0 overlaps the tail)",
        );
    }

    #[test]
    fn fragment_block_subtree_grid_later_row_parallel_siblings_share_page_y() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut cells: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 100.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        cells.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let page_one_cells: Vec<_> = cells.iter().filter(|(p, _, _)| *p == 1).collect();
        assert_eq!(
            page_one_cells.len(),
            2,
            "expected the second grid row's two cells on page 1, got {cells:?}"
        );
        assert!(
            (page_one_cells[0].2 - page_one_cells[1].2).abs() < 0.5,
            "parallel cells in the same later grid row must share y; got {cells:?}"
        );
    }

    #[test]
    fn fragment_block_subtree_flex_later_row_parallel_siblings_share_page_y() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: flex; flex-wrap: wrap; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut cells: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 100.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        cells.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let page_one_cells: Vec<_> = cells.iter().filter(|(p, _, _)| *p == 1).collect();
        assert_eq!(
            page_one_cells.len(),
            2,
            "expected the second flex row's two cells on page 1, got {cells:?}"
        );
        assert!(
            (page_one_cells[0].2 - page_one_cells[1].2).abs() < 0.5,
            "parallel cells in the same later flex row must share y; got {cells:?}"
        );
    }

    #[test]
    fn fragment_block_subtree_grid_later_row_preserves_parallel_sibling_offset() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: grid; grid-template-columns: 100px 100px; align-items: start; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 80px; width: 100px; margin-top: 20px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut page_one_cells: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.width.to_f32() - 100.0).abs() < 0.5)
            .map(|f| (f.height.to_f32(), f.x.to_f32(), f.y.to_f32()))
            .collect();
        page_one_cells.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The line moves whole: its top at the page top, the grid cell 20 px
        // below it still 20 px below.
        assert_eq!(
            page_one_cells,
            [(80.0, 0.0, 20.0), (100.0, 100.0, 0.0)],
            "same-row sibling must preserve its cross-axis offset relative to the line"
        );
    }

    #[test]
    fn fragment_block_subtree_flex_later_row_preserves_parallel_sibling_offset() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 220px;">
                <div style="height: 100px; width: 200px"></div>
                <div style="display: flex; flex-wrap: wrap; align-items: flex-start; width: 200px;">
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 100px; width: 100px"></div>
                  <div style="height: 80px; width: 100px; margin-top: 20px"></div>
                  <div style="height: 100px; width: 100px"></div>
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 250.0_f32.as_px(), &table);

        let mut page_one_cells: Vec<_> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.width.to_f32() - 100.0).abs() < 0.5)
            .map(|f| (f.height.to_f32(), f.x.to_f32(), f.y.to_f32()))
            .collect();
        page_one_cells.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The line moves whole: its top at the page top, the flex item 20 px
        // below it still 20 px below.
        assert_eq!(
            page_one_cells,
            [(80.0, 0.0, 20.0), (100.0, 100.0, 0.0)],
            "same-row sibling must preserve its cross-axis offset relative to the line"
        );
    }

    #[test]
    fn fragment_block_subtree_grid_row_recursive_cells_share_page_y() {
        // 2 cell の grid row で、両 cell が inner content を持ち
        // recursion 経路を通る (`needs_recursion=true`) case。両 cell
        // が page 内に収まるサイズなので page boundary は跨がない。
        //
        // pre-fix: 右 cell の recursion が parent の `cursor_y`
        // (= 左 cell の bottom = 200) を引きずり、右 column の inner
        // div が y=200 (block flow stacking) に積まれる。
        //
        // post-fix: 両 cell とも row top y=0 から開始し、各 inner div
        // が y=0..50, y=50..100 に並ぶ。
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                <div style="width: 100px">
                  <div style="height: 50px; width: 100px"></div>
                  <div style="height: 50px; width: 100px"></div>
                </div>
                <div style="width: 100px">
                  <div style="height: 50px; width: 100px"></div>
                  <div style="height: 50px; width: 100px"></div>
                </div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);

        // 100×50 の inner div fragment 4 個を集める。
        let mut inner: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.width.to_f32() - 100.0).abs() < 0.5 && (f.height.to_f32() - 50.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        inner.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            inner.len(),
            4,
            "expected 4 inner divs (2 per column × 2 columns), got inner={inner:?}"
        );

        // すべて page 0 に収まる
        assert!(
            inner.iter().all(|(p, _, _)| *p == 0),
            "all 4 inner divs must be on page 0, inner={inner:?}"
        );

        // 左 column (x=0): y=0 と y=50 の 2 個
        let left_ys: Vec<f32> = inner
            .iter()
            .filter(|(_, x, _)| x.abs() < 0.5)
            .map(|(_, _, y)| *y)
            .collect();
        let approx_eq = |a: &[f32], b: &[f32]| -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.5)
        };
        assert!(
            approx_eq(&left_ys, &[0.0, 50.0]),
            "left column y, inner={inner:?}"
        );

        // 右 column (x=100): y=0 と y=50 の 2 個 (block flow になっていれば
        // y=100, y=150 になる)
        let right_ys: Vec<f32> = inner
            .iter()
            .filter(|(_, x, _)| (x - 100.0).abs() < 0.5)
            .map(|(_, _, y)| *y)
            .collect();
        assert!(
            approx_eq(&right_ys, &[0.0, 50.0]),
            "right column y must match left column (parallel siblings, not stacked), inner={inner:?}"
        );
    }

    #[test]
    fn fragment_block_subtree_flex_row_recursive_cells_share_page_y() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="display: flex; width: 200px;">
                <div style="width: 100px; flex: 0 0 100px">
                  <div style="height: 50px; width: 100px"></div>
                  <div style="height: 50px; width: 100px"></div>
                </div>
                <div style="width: 100px; flex: 0 0 100px">
                  <div style="height: 50px; width: 100px"></div>
                  <div style="height: 50px; width: 100px"></div>
                </div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);

        let mut inner: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.width.to_f32() - 100.0).abs() < 0.5 && (f.height.to_f32() - 50.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        inner.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            inner.len(),
            4,
            "expected 4 inner divs (2 per column × 2 columns), got inner={inner:?}"
        );
        assert!(
            inner.iter().all(|(p, _, _)| *p == 0),
            "all 4 inner divs must be on page 0, inner={inner:?}"
        );

        let left_ys: Vec<f32> = inner
            .iter()
            .filter(|(_, x, _)| x.abs() < 0.5)
            .map(|(_, _, y)| *y)
            .collect();
        let right_ys: Vec<f32> = inner
            .iter()
            .filter(|(_, x, _)| (x - 100.0).abs() < 0.5)
            .map(|(_, _, y)| *y)
            .collect();
        let approx_eq = |a: &[f32], b: &[f32]| -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.5)
        };
        assert!(
            approx_eq(&left_ys, &[0.0, 50.0]),
            "left column y, inner={inner:?}"
        );
        assert!(
            approx_eq(&right_ys, &[0.0, 50.0]),
            "right column y must match left column (parallel siblings, not stacked), inner={inner:?}"
        );
    }

    /// fulgur-916y: a multicol container with a `column-span: all`
    /// child whose subtree exceeds one page must split across pages
    /// in the partition path. Pre-fix, the multicol gate
    /// (`!is_multicol`) blocked recursion, so the whole multicol
    /// container ended up as a single fragment regardless of
    /// overflow — fragmenter reported 1 page. Post-fix, the gate
    /// admits multicol containers that have a span:all child, so
    /// `fragment_block_subtree` recurses into the span subtree and
    /// splits it across pages via the regular block-flow logic.
    ///
    /// Pins `implied_page_count(geometry) >= 2` for the
    /// `multicol_span_all` integration fixture's HTML rendered with
    /// the fragmenter's strip height set small enough that the
    /// span:all section overflows page 0.
    #[test]
    fn fragment_pagination_root_recurses_into_multicol_with_span_all() {
        let mut long = String::new();
        for _ in 0..40 {
            long.push_str(
                "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                 Sed do eiusmod tempor incididunt ut labore et dolore magna \
                 aliqua. Ut enim ad minim veniam, quis nostrud exercitation.</p>",
            );
        }
        let html = format!(
            r#"<!doctype html><html><head><style>
                body {{ margin: 10pt; font-size: 10pt; }}
                .mc {{ column-count: 2; column-gap: 10pt; }}
                .span {{ column-span: all; }}
            </style></head><body>
              <div class="mc">
                <p>before column content.</p>
                <section class="span">{long}</section>
                <p>after column content.</p>
              </div>
            </body></html>"#,
            long = long
        );

        // 600 viewport, 400 page strip (small enough to overflow).
        let mut doc = parse(&html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 400.0_f32.as_px(), &table);
        let pages = super::implied_page_count(&geom);
        assert!(
            pages >= 2,
            "expected multicol with span:all overflow to split into >=2 pages, got {pages}",
        );
    }

    /// fulgur-916y below the body: a multicol container with a
    /// `column-span: all` child is entered at every depth, so a nested one
    /// whose spanning subtree overflows splits across pages between its
    /// paragraphs, rather than being sliced with all of them on its first
    /// slice.
    #[test]
    fn a_nested_multicol_container_with_a_column_span_all_child_splits() {
        let mut long = String::new();
        for _ in 0..40 {
            long.push_str(
                "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                 Sed do eiusmod tempor incididunt ut labore et dolore magna \
                 aliqua. Ut enim ad minim veniam, quis nostrud exercitation.</p>",
            );
        }
        let html = format!(
            r#"<!doctype html><html><head><style>
                body {{ margin: 10pt; font-size: 10pt; }}
                .mc {{ column-count: 2; column-gap: 10pt; }}
                .span {{ column-span: all; }}
            </style></head><body>
              <div><div class="mc">
                <p>before column content.</p>
                <section class="span">{long}<p id="last">last.</p></section>
                <p>after column content.</p>
              </div></div>
            </body></html>"#
        );

        let mut doc = parse(&html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 400.0_f32.as_px(), &table);
        let last = &geom[&find_by_id(&doc, "last").unwrap()].fragments;
        assert!(
            last.iter().all(|f| f.page_index >= 1),
            "the spanning subtree's last paragraph must be on a later page: {last:?}",
        );
    }

    /// fulgur-2s7p.1: a first in-flow child whose subtree contains a
    /// nested multi-line `<p>` (wrapped in one or more plain `<div>`s)
    /// must have its lines split across the page boundary, the same
    /// way a body-direct paragraph already does — it must not fall
    /// through to the block-child gate (`child_page_y > page_start_y`,
    /// this function's strip-overflow check) that can only push a
    /// *whole* child to the next page or emit it as one oversized
    /// fragment. Repro shape matches the reported bug: a sibling
    /// occupies part of page 0, so the nested `<p>` — the parent's
    /// first and only in-flow child — is itself mid-page and overflows
    /// the remaining strip. Pre-fix, `child_page_y == page_start_y` for
    /// this child (it's first), the strip-overflow gate never fires,
    /// and the entire paragraph (including the ~30 words that don't
    /// fit) is emitted as a single fragment overflowing the page
    /// bottom — silently lost once rendering clips to the page.
    ///
    /// Uses `margin: 150px 0` rather than the originally reported
    /// `padding: 150px 0` — see the identically-shaped end-to-end test
    /// in `crates/fulgur/tests/render_smoke.rs`
    /// (`fulgur_2s7p1_nested_first_child_overflow_continues_across_page`)
    /// for why: `padding-top` on inline roots is a separate,
    /// pre-existing, already documented Blitz limitation, verified to
    /// affect body-direct paragraphs identically on unmodified `main`,
    /// not something this fix touches.
    #[test]
    fn nested_inline_root_splits_at_line_boundary_when_first_child_overflows() {
        let mut words = String::new();
        for i in 0..120 {
            words.push_str(&format!("W{i:04} "));
        }
        let html = format!(
            r#"<!doctype html><html><body style="margin:0; font-size:14px">
                <div style="height:100px"></div>
                <div><div><p style="margin:150px 0; padding:0; line-height:20px">{words}</p></div></div>
            </body></html>"#
        );
        let mut doc = parse(&html, 500.0);

        fn find_p(doc: &blitz_dom::BaseDocument, node_id: usize) -> Option<usize> {
            let node = doc.get_node(node_id)?;
            if node
                .element_data()
                .is_some_and(|e| e.name.local.as_ref() == "p")
            {
                return Some(node_id);
            }
            for &child in &node.children {
                if let Some(found) = find_p(doc, child) {
                    return Some(found);
                }
            }
            None
        }
        let body_id = find_body_id(&doc).expect("body must exist");
        let p_id = find_p(&doc, body_id).expect("fixture must contain a <p>");

        let table = run_pass(doc.deref_mut(), 400.0);
        let p_geom = table.get(&p_id).expect("<p> must be recorded in geometry");

        assert!(
            p_geom.fragments.len() >= 2,
            "expected the nested paragraph to split across >=2 pages, got {:?}",
            p_geom.fragments
        );
        for frag in &p_geom.fragments {
            assert!(
                frag.y.to_f32() + frag.height.to_f32() <= 400.0 + 0.5,
                "fragment must not overflow the page strip: {frag:?}"
            );
        }
        let pages: std::collections::BTreeSet<u32> =
            p_geom.fragments.iter().map(|f| f.page_index).collect();
        assert_eq!(
            pages.len(),
            p_geom.fragments.len(),
            "each fragment should land on a distinct page: {:?}",
            p_geom.fragments
        );
    }

    /// fulgur-2s7p.1 (inverse-failure guard, epic `fulgur-2s7p` design
    /// note #2): the new inline-root split branch's own split-before
    /// pre-check reads `cursor_y > page_start_y` — the same style of
    /// proxy the epic's landmine list warns must not be misread as
    /// "real content already placed on this page". A parent's own
    /// `padding-top` alone (no preceding sibling) also advances
    /// `cursor_y` past `page_start_y` for its first child, exactly like
    /// real content would. If the pre-check fired on that basis alone
    /// regardless of whether the paragraph actually overflows, it would
    /// spuriously push a fresh page (and an extra parent fragment) for
    /// content that fits fine. Here the padding (50px) plus the
    /// paragraph's natural height comfortably fit one page strip
    /// (400px) — assert exactly one fragment for both the paragraph and
    /// its wrapping div, i.e. no spurious extra fragment was emitted.
    #[test]
    fn nested_inline_root_padding_top_alone_does_not_spawn_spurious_fragment() {
        let html = r#"<!doctype html><html><body style="margin:0; font-size:14px">
            <div id="wrap" style="padding-top:50px">
                <p style="margin:0; line-height:20px">one two three four five</p>
            </div>
        </body></html>"#;
        let mut doc = parse(html, 500.0);
        let wrap_id = find_by_id(&doc, "wrap").expect("div#wrap");

        fn find_p(doc: &blitz_dom::BaseDocument, node_id: usize) -> Option<usize> {
            let node = doc.get_node(node_id)?;
            if node
                .element_data()
                .is_some_and(|e| e.name.local.as_ref() == "p")
            {
                return Some(node_id);
            }
            for &child in &node.children {
                if let Some(found) = find_p(doc, child) {
                    return Some(found);
                }
            }
            None
        }
        let p_id = find_p(&doc, wrap_id).expect("fixture must contain a <p>");

        let table = run_pass(doc.deref_mut(), 400.0);
        let p_geom = table.get(&p_id).expect("<p> must be recorded in geometry");
        assert_eq!(
            p_geom.fragments.len(),
            1,
            "padding-top alone must not force a page split: {:?}",
            p_geom.fragments
        );
        let wrap_geom = table
            .get(&wrap_id)
            .expect("div#wrap must be recorded in geometry");
        assert_eq!(
            wrap_geom.fragments.len(),
            1,
            "padding-top alone must not spawn a spurious/ghost parent fragment: {:?}",
            wrap_geom.fragments
        );
    }

    /// fulgur-2s7p.1 (epic `fulgur-2s7p` design note #4 — break-before
    /// / break-after symmetry): forcing the same break point via
    /// `break-after: page` on the multi-line nested paragraph vs
    /// `break-before: page` on its sibling must close the wrapping
    /// div's page-0 fragment to the *same* height either way. The new
    /// inline-root split branch has its own `break_after_page` handling
    /// (mirroring the pre-existing block-child branches' pattern);
    /// `break-before` on the sibling is handled by this function's
    /// shared upstream check (unrelated to this fix). If cursor-derived
    /// height and full-remaining-strip-derived height ever disagreed
    /// between the two paths, the same logical break would produce a
    /// different-looking parent background/border depending on which
    /// side declared the break — this pins that they agree.
    #[test]
    fn nested_inline_root_break_after_vs_break_before_same_parent_height() {
        let make_html = |p1_style: &str, p2_style: &str| {
            format!(
                r#"<!doctype html><html><body style="margin:0; font-size:14px">
                    <div><div id="parent">
                        <p id="p1" style="{p1_style} margin:0; line-height:20px">one two three four five six seven eight nine ten eleven twelve</p>
                        <p id="p2" style="{p2_style} margin:0; line-height:20px">short tail</p>
                    </div></div>
                </body></html>"#
            )
        };
        let html_after = make_html("break-after:page;", "");
        let html_before = make_html("", "break-before:page;");

        let run = |html: &str| -> (usize, f32) {
            // Narrow viewport forces p1's 12 words to wrap across
            // multiple lines, so the assertion exercises the new
            // inline-root split branch's `break_after_page` handling
            // (only reached when `line_metrics.len() > 1`), not the
            // pre-existing single-line block-child path.
            let mut doc = parse(html, 150.0);
            let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
            let parent_id = find_by_id(&doc, "parent").expect("div#parent");
            let p1_id = find_by_id(&doc, "p1").expect("p#p1");
            let lines = super::collect_inline_line_metrics(doc.get_node(p1_id).expect("p1 node"));
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 400.0_f32.as_px(), &table_cs);
            let parent_geom = geom.get(&parent_id).expect("div#parent must be recorded");
            let page0 = parent_geom
                .fragments
                .iter()
                .find(|f| f.page_index == 0)
                .expect("page 0 fragment for div#parent");
            (lines.len(), page0.height.to_f32())
        };

        let (lines_after, h_after) = run(&html_after);
        let (lines_before, h_before) = run(&html_before);
        assert!(
            lines_after > 1 && lines_before > 1,
            "fixture must wrap p1 to >1 line to exercise the new branch: after={lines_after} before={lines_before}"
        );
        assert!(
            (h_after - h_before).abs() < 0.5,
            "break-after vs break-before must close the parent fragment to the same height: after={h_after} before={h_before}"
        );
    }

    /// codex P1 review on PR #741 (fulgur-2s7p.1 follow-up): the new
    /// inline-root split branch's split-before pre-check advances
    /// `page_index` / `page_start_y` to a fresh page but, pre-fix, never
    /// rebased `page_taffy_origin` to the splitting child's own top.
    /// When the paragraph then fits entirely on that fresh page (no
    /// further split — `new_page_index == pre_split_page`), the
    /// `new_page_index > pre_split_page` rebase-pending path below never
    /// runs either, so `page_taffy_origin` stays stale. The FOLLOWING
    /// sibling then computes `child_page_y` from that stale origin and
    /// lands far past where the paragraph actually ended instead of
    /// immediately after it.
    ///
    /// Fixture: A (90px) pushes `cursor_y` to 90 on a 100px strip, then
    /// a narrow multi-line `<p>` (one word per line, comfortably under
    /// 100px total) doesn't fit the remaining 10px — triggering
    /// split-before — but fits whole on the fresh page. `div#after`
    /// must land immediately after the paragraph's single fragment on
    /// that same fresh page.
    #[test]
    fn nested_inline_root_split_before_rebases_taffy_origin_for_sibling() {
        let html = r#"<!doctype html><html><body style="margin:0; font-size:12px">
            <div>
                <div style="height:90px"></div>
                <p style="margin:0; padding:0; line-height:20px; width:20px">one two three four</p>
                <div id="after" style="height:20px"></div>
            </div>
        </body></html>"#;
        let mut doc = parse(html, 200.0);
        let after_id = find_by_id(&doc, "after").expect("div#after");

        fn find_p(doc: &blitz_dom::BaseDocument, node_id: usize) -> Option<usize> {
            let node = doc.get_node(node_id)?;
            if node
                .element_data()
                .is_some_and(|e| e.name.local.as_ref() == "p")
            {
                return Some(node_id);
            }
            for &child in &node.children {
                if let Some(found) = find_p(doc, child) {
                    return Some(found);
                }
            }
            None
        }
        let body_id = find_body_id(&doc).expect("body must exist");
        let p_id = find_p(&doc, body_id).expect("fixture must contain a <p>");

        let table = run_pass(doc.deref_mut(), 100.0);

        let p_geom = table.get(&p_id).expect("<p> must be recorded in geometry");
        assert_eq!(
            p_geom.fragments.len(),
            1,
            "fixture must fit the paragraph on one fresh page after split-before \
             (this reproduces the `new_page_index == pre_split_page` case where \
             the rebase-pending path is skipped): {:?}",
            p_geom.fragments
        );
        let p_frag = &p_geom.fragments[0];
        assert_eq!(
            p_frag.page_index, 1,
            "expected the split-before pre-check to push the paragraph to page 1: {p_frag:?}"
        );

        let after_geom = table
            .get(&after_id)
            .expect("div#after must be recorded in geometry");
        assert_eq!(after_geom.fragments.len(), 1, "{:?}", after_geom.fragments);
        let after_frag = &after_geom.fragments[0];

        assert_eq!(
            after_frag.page_index, p_frag.page_index,
            "div#after must land on the same page the paragraph continued to \
             (stale page_taffy_origin bug pushes it to a later page): \
             p_frag={p_frag:?} after_frag={after_frag:?}"
        );
        assert!(
            (after_frag.y.to_f32() - (p_frag.y.to_f32() + p_frag.height.to_f32())).abs() < 1.0,
            "div#after must immediately follow the paragraph's fragment — a stale \
             page_taffy_origin would place it far past the page bottom instead: \
             p_frag={p_frag:?} after_frag={after_frag:?}"
        );
    }

    /// codex P1 + CodeRabbit review on PR #741 (fulgur-2s7p.1
    /// follow-up): when the new inline-root split branch's multi-line
    /// `<p>` is a grid/flex row's parallel sibling (not the row's first
    /// cell), `fragment_inline_root` was started from the parent's
    /// running `cursor_y` (the previous cell's row-bottom) instead of
    /// this cell's own row-local `child_page_y`, stacking the paragraph
    /// below the previous cell instead of beside it at the row's y.
    ///
    /// Fixture: 3-column grid row on a 100px strip, 70px into the page
    /// (30px available on the current strip). Column A (5px) advances
    /// `cursor_y` to 75 while the row itself starts at row-local y=70.
    /// Column B is a narrow 2-line `<p>` (one word per line, tight
    /// 5px `line-height` — small enough that it fits the page from
    /// *either* candidate start (70 or 75), isolating the
    /// start-position bug from split-before / page-crossing behaviour
    /// (covered separately by
    /// `grid_row_multiline_paragraph_cosplits_with_later_row_siblings`
    /// below). Column C (20px) exists only so the body-level preflight
    /// (`would_split_block_subtree`, which sums direct children's
    /// heights) sees enough total height to force recursion into the
    /// grid at all — without it the grid is emitted as one atomic
    /// fragment via `record_subtree_descendants`, never reaching the
    /// per-child fragmenter loop this bug lives in.
    #[test]
    fn grid_row_inline_split_starts_at_row_local_y_not_prior_cell_bottom() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div style="height: 70px"></div>
              <div style="display: grid; grid-template-columns: 60px 60px 60px; width: 180px;">
                <div id="col_a" style="height: 5px; width: 60px"></div>
                <p id="col_b" style="margin:0; padding:0; line-height:5px; width:20px">one two</p>
                <div id="col_c" style="height: 20px; width: 60px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let col_b_id = find_by_id(&doc, "col_b").expect("p#col_b");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table_cs);

        let col_b_geom = geom.get(&col_b_id).expect("p#col_b must be recorded");
        assert_eq!(
            col_b_geom.fragments.len(),
            1,
            "fixture must fit col_b on the current page from either candidate \
             start (isolating the start-position bug from page-crossing): {:?}",
            col_b_geom.fragments
        );
        let frag = &col_b_geom.fragments[0];
        assert_eq!(frag.page_index, 0, "{frag:?}");
        assert!(
            (frag.y.to_f32() - 70.0).abs() < 1.0,
            "col_b must start at the row-local y=70 (col_a's row-top), not \
             col_a's bottom y=75 (the `cursor_y`-vs-`child_page_y` start-arg \
             bug): frag={frag:?}"
        );
    }

    /// CodeRabbit review on PR #741 (fulgur-2s7p.1 follow-up): when the
    /// new inline-root split branch's multi-line `<p>` is a grid/flex
    /// row's FIRST cell and crosses a page boundary, the later cells in
    /// the same row must still start where the row does (fulgur-ysms; since
    /// the walk places a line as one unit, every box of a [`LineState`]
    /// restores its start): before the fix they landed only on the page the
    /// paragraph's tail ended up on, losing their earlier-page row content
    /// entirely.
    ///
    /// Fixture: 3-column grid row on a 100px strip, 50px into the page.
    /// Column 1 is a narrow multi-line `<p>` (one word per line, 20px
    /// lines) tall enough to cross the page boundary — as the row's
    /// first cell, `cursor_y == page_start_y` so split-before doesn't
    /// fire and the split runs in place. The row starts far enough up
    /// the page that two lines fit, so the orphan minimum is met and the
    /// paragraph splits rather than moving whole to the next page. Columns
    /// 2 and 3 must still land on page 0 at the row-local y (50).
    #[test]
    fn grid_row_multiline_paragraph_cosplits_with_later_row_siblings() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div style="height: 50px"></div>
              <div style="display: grid; grid-template-columns: 60px 60px 60px; width: 180px;">
                <p id="col_p" style="margin:0; padding:0; line-height:20px; width:20px">one two three four five</p>
                <div id="col_b" style="height: 10px; width: 60px"></div>
                <div id="col_c" style="height: 10px; width: 60px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let col_p_id = find_by_id(&doc, "col_p").expect("p#col_p");
        let col_b_id = find_by_id(&doc, "col_b").expect("div#col_b");
        let col_c_id = find_by_id(&doc, "col_c").expect("div#col_c");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table_cs);

        let col_p_geom = geom.get(&col_p_id).expect("p#col_p must be recorded");
        assert!(
            col_p_geom.fragments.len() >= 2,
            "fixture must cross the page boundary to exercise the co-split path: {:?}",
            col_p_geom.fragments
        );

        for (id, geom_entry) in [("col_b", col_b_id), ("col_c", col_c_id)] {
            let cell_geom = geom
                .get(&geom_entry)
                .unwrap_or_else(|| panic!("{id} must be recorded in geometry"));
            let page0 = cell_geom
                .fragments
                .iter()
                .find(|f| f.page_index == 0)
                .unwrap_or_else(|| {
                    panic!(
                        "{id} must co-split back to page 0 at the row-local y once \
                     col_p crossed a page boundary — but has no page-0 fragment \
                     at all (lost its earlier-page row content): {:?}",
                        cell_geom.fragments
                    )
                });
            assert!(
                (page0.y.to_f32() - 50.0).abs() < 1.0,
                "{id} must land at the row-local y=50 on page 0: page0={page0:?}"
            );
        }
    }

    /// codex P1 review on PR #741 (second round, commit 84654dc):
    /// when the split-before pre-check alone pushes a grid/flex row's
    /// multi-line `<p>` to a fresh page — and the paragraph then fits
    /// *whole* on that fresh page, so `fragment_inline_root` itself
    /// never advances past `pre_split_page` — the row was still never
    /// marked as crossed (the row-bookkeeping was gated on
    /// `new_page_index > pre_split_page`, which is false here even
    /// though the row DID cross a page, via the split-before advance
    /// itself). A later same-row cell then never restored to the
    /// row-start state and lost its earlier-page presence entirely.
    ///
    /// Fixture: 2-column grid, two rows. Row 1 (two plain 20px cells)
    /// advances `cursor_y` past the container's own entry point so
    /// row 2's first cell — the multi-line `<p>` — does not qualify
    /// for the "first in-flow child" split-before exemption. The `<p>`
    /// is sized to fit whole on the fresh page after split-before
    /// fires (isolating this from the multi-page-crossing case the
    /// sibling test above covers). Row 2's second cell must land on
    /// page 0 at the row-local y, not on the fresh page split-before
    /// pushed the paragraph to.
    #[test]
    fn grid_row_split_before_only_crossing_still_cosplits_with_row_sibling() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div style="height: 70px"></div>
              <div id="grid2" style="display: grid; grid-template-columns: 60px 60px; width: 120px;">
                <div style="height: 20px; width: 60px"></div>
                <div style="height: 20px; width: 60px"></div>
                <p id="col_p2" style="margin:0; padding:0; line-height:20px; width:20px">one two three four five</p>
                <div id="col_b2" style="height: 10px; width: 60px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let grid2_id = find_by_id(&doc, "grid2").expect("div#grid2");
        let col_p2_id = find_by_id(&doc, "col_p2").expect("p#col_p2");
        let col_b2_id = find_by_id(&doc, "col_b2").expect("div#col_b2");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table_cs);

        let col_p2_geom = geom.get(&col_p2_id).expect("p#col_p2 must be recorded");
        assert_eq!(
            col_p2_geom.fragments.len(),
            1,
            "fixture must fit col_p2 on one fresh page after split-before \
             (isolating the split-before-only crossing from a further \
             mid-paragraph split): {:?}",
            col_p2_geom.fragments
        );
        assert_eq!(
            col_p2_geom.fragments[0].page_index, 1,
            "expected split-before to push col_p2 to page 1: {:?}",
            col_p2_geom.fragments[0]
        );

        let col_b2_geom = geom.get(&col_b2_id).expect("div#col_b2 must be recorded");
        let page0 = col_b2_geom
            .fragments
            .iter()
            .find(|f| f.page_index == 0)
            .unwrap_or_else(|| {
                panic!(
                    "col_b2 must co-split back to page 0 at the row-local y once \
                     col_p2's split-before pushed it to page 1 — but has no \
                     page-0 fragment at all: {:?}",
                    col_b2_geom.fragments
                )
            });
        assert!(
            (page0.y.to_f32() - 90.0).abs() < 1.0,
            "col_b2 must land at the row-local y=90 on page 0: page0={page0:?}"
        );

        // coderabbit review on PR #741: the grid's own page-0 parent
        // fragment (its background/border strip) must cover the full
        // remaining page-0 strip (page_start_y=70 through the page
        // bottom at 100), not just `cursor_y - page_start_y` (90 - 70 =
        // 20) at the moment split-before fired for col_p2. col_b2
        // restores back to page 0 (row-local y=90..100) *after* that
        // push, so a `cursor_y`-bounded parent fragment would stop at
        // row 1's bottom (90) and leave col_b2's paint region
        // (90..100) uncovered by the grid's own background.
        let grid2_geom = geom.get(&grid2_id).expect("div#grid2 must be recorded");
        let grid2_page0 = grid2_geom
            .fragments
            .iter()
            .find(|f| f.page_index == 0)
            .unwrap_or_else(|| {
                panic!(
                    "grid2 must have a page-0 parent fragment: {:?}",
                    grid2_geom.fragments
                )
            });
        assert!(
            (grid2_page0.height.to_f32() - 30.0).abs() < 1.0,
            "grid2's page-0 fragment must span the full remaining strip \
             (70..100, height 30) to cover col_b2's restored paint \
             region at row-local y=90..100, not stop at row 1's bottom \
             (90, height 20): grid2_page0={grid2_page0:?}"
        );
    }

    /// codex P1 review on PR #741 (second round, commit 84654dc): the
    /// split-before overflow check compared `cursor_y + para_total_h`
    /// against `page_height_px`. For a grid/flex row's later cell,
    /// `cursor_y` is the row's running max bottom from earlier cells,
    /// not this cell's own row-local start (`child_page_y`) — so a
    /// tall preceding cell can make the check see overflow even though
    /// this paragraph would fit fine at its own position.
    ///
    /// Fixture: 2-column grid row, 70px into a 100px strip. Column A
    /// (25px) pushes `cursor_y` to 95. Column B is a narrow 2-line
    /// `<p>` whose own row-local start (`child_page_y` = 70) plus its
    /// height fits comfortably under 100, but `cursor_y` (95) plus that
    /// same height overflows — the false-positive split-before this
    /// finding describes. Column B must land on page 0 (no split-before
    /// push at all), not page 1.
    #[test]
    fn grid_row_split_before_check_uses_row_local_y_not_prior_cell_bottom() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div style="height: 70px"></div>
              <div style="display: grid; grid-template-columns: 60px 60px; width: 120px;">
                <div id="col_a" style="height: 25px; width: 60px"></div>
                <p id="col_b" style="margin:0; padding:0; line-height:10px; width:20px">one two</p>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let col_b_id = find_by_id(&doc, "col_b").expect("p#col_b");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table_cs);

        let col_b_geom = geom.get(&col_b_id).expect("p#col_b must be recorded");
        assert_eq!(
            col_b_geom.fragments.len(),
            1,
            "col_b must fit whole at its row-local position, needing no split \
             at all: {:?}",
            col_b_geom.fragments
        );
        assert_eq!(
            col_b_geom.fragments[0].page_index, 0,
            "col_b fits at row-local y=70 on page 0 — a split-before push driven \
             by col_a's row-max cursor_y (95) instead of col_b's own row-local \
             y (70) would wrongly land it on page 1 instead: {:?}",
            col_b_geom.fragments[0]
        );
    }

    /// codex P2 review on PR #741 (second round, commit 84654dc): after
    /// the line-split branch returns without crossing a page
    /// (`new_page_index == pre_split_page`), `cursor_y` was set
    /// unconditionally to `new_cursor_y` — the split paragraph's own
    /// bottom. For a grid/flex row where an earlier cell is taller
    /// than this paragraph, that shrinks `cursor_y` below the row's
    /// true max extent (mirrors the recursion branch's fulgur-u0p0
    /// `cursor_y.max(nc)` handling, which this branch skipped).
    ///
    /// Fixture: 2-column grid row. Column A (60px) establishes the
    /// row's true bottom. Column B is a short 2-line `<p>` with
    /// `break-after: page` whose own bottom is well under 60, and an item
    /// of a second row follows it, so the break lies inside the grid. The
    /// grid's own parent fragment on page 0 — closed by column B's
    /// `break-after: page` — must span the row's true height (60), not
    /// column B's shorter one.
    #[test]
    fn grid_row_same_page_split_keeps_row_max_cursor_for_break_after() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div id="grid" style="display: grid; grid-template-columns: 60px 60px; width: 120px; align-items: start">
                <div id="col_a" style="height: 60px; width: 60px"></div>
                <p id="col_b" style="margin:0; padding:0; line-height:10px; width:20px; break-after:page">one two</p>
                <div style="height: 10px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let grid_id = find_by_id(&doc, "grid").expect("div#grid");
        let col_b_id = find_by_id(&doc, "col_b").expect("p#col_b");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table_cs);

        let col_b_geom = geom.get(&col_b_id).expect("p#col_b must be recorded");
        assert_eq!(
            col_b_geom.fragments.len(),
            1,
            "col_b must fit whole on the same page as its row-local start \
             (isolating the same-page cursor_y handling from a crossing): {:?}",
            col_b_geom.fragments
        );
        assert_eq!(
            col_b_geom.fragments[0].page_index, 0,
            "{:?}",
            col_b_geom.fragments[0]
        );
        assert!(
            col_b_geom.fragments[0].height.to_f32() < 59.0,
            "fixture must make col_b's own bottom shorter than col_a's 60px \
             row height: {:?}",
            col_b_geom.fragments[0]
        );

        let grid_geom = geom.get(&grid_id).expect("div#grid must be recorded");
        let page0 = grid_geom
            .fragments
            .iter()
            .find(|f| f.page_index == 0)
            .expect("div#grid must have a page-0 fragment");
        assert!(
            (page0.height.to_f32() - 60.0).abs() < 1.0,
            "break-after: page on col_b must close the grid's page-0 fragment \
             at the row's true max bottom (60, from col_a) — shrinking to \
             col_b's own shorter bottom would clip col_a's background/border: \
             page0={page0:?}"
        );
    }

    /// CodeRabbit review on PR #741 (third round, commit 32aa763):
    /// Being in a row alone doesn't distinguish a later cell of the row
    /// *already on the current page* from the row's *first* cell, which
    /// could itself trigger split-before onto a fresh page (a box of a
    /// [`LineState`] no longer moves alone).
    /// In the latter case `cursor_y` still holds the *previous row's*
    /// bottom on the *old* page — not a same-page previous-cell bottom
    /// — so `.max(new_cursor_y)` against it is exactly as stale as the
    /// no-row case the surrounding fix (32aa763) was written to guard
    /// against.
    ///
    /// Fixture: row 1 is two 90px cells (`cursor_y` = 90 on page 0).
    /// Row 2's first cell is a 2-line `<p>` (`line-height:20px`, wraps
    /// to 40px total) with `break-after:page` — short enough that
    /// 90 + 40 > 100 forces split-before, but 40 alone fits the fresh
    /// page whole (`new_page_index == pre_split_page`). The `<p>`'s own
    /// `break-after:page` then closes the grid's page-1 fragment right
    /// after it: that close must use the paragraph's actual bottom on
    /// page 1 (40), not the stale row-1 bottom (90) carried over from
    /// the old page.
    #[test]
    fn grid_row_first_cell_split_before_does_not_inherit_prior_row_cursor() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; font-size: 12px">
              <div id="grid3" style="display: grid; grid-template-columns: 60px 60px; width: 120px;">
                <div style="height: 90px; width: 60px"></div>
                <div style="height: 90px; width: 60px"></div>
                <p id="col_p3" style="margin:0; padding:0; line-height:20px; width:20px; break-after:page">one two</p>
                <div id="col_b3" style="height: 10px; width: 60px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let col_p3_id = find_by_id(&doc, "col_p3").expect("p#col_p3");
        let grid3_id = find_by_id(&doc, "grid3").expect("div#grid3");

        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table_cs);

        let col_p3_geom = geom.get(&col_p3_id).expect("p#col_p3 must be recorded");
        assert_eq!(
            col_p3_geom.fragments.len(),
            1,
            "fixture must fit col_p3 whole on the fresh page after \
             split-before (isolating the same-page cursor_y bug from a \
             further mid-paragraph split): {:?}",
            col_p3_geom.fragments
        );
        let p3_frag = &col_p3_geom.fragments[0];
        assert_eq!(
            p3_frag.page_index, 1,
            "expected split-before to push col_p3 to page 1: {p3_frag:?}"
        );
        assert!(
            (p3_frag.height.to_f32() - 40.0).abs() < 1.0,
            "fixture must make col_p3's own page-1 bottom (40, two 20px \
             lines) shorter than row 1's 90px bottom: {p3_frag:?}"
        );

        let grid3_geom = geom.get(&grid3_id).expect("div#grid3 must be recorded");
        let page1 = grid3_geom
            .fragments
            .iter()
            .find(|f| f.page_index == 1)
            .expect("div#grid3 must have a page-1 fragment");
        assert!(
            (page1.height.to_f32() - 40.0).abs() < 1.0,
            "col_p3's break-after:page must close the grid's page-1 \
             fragment at col_p3's own bottom (40) — inheriting row 1's \
             stale 90px bottom from the old page would overstate it: \
             page1={page1:?}"
        );
    }

    /// codex P2 review on PR #741: a nested multi-line `<p>` with a CSS
    /// transform paints as a single atomic box (CSS Transforms §6.1) —
    /// it must not be split into independently rendered per-page
    /// fragments (the rotation / skew / matrix would then apply to each
    /// slice separately instead of the whole box). Before the new
    /// inline-root split branch existed, such a child fell through to
    /// the oversized-child / recursion path, which already excludes
    /// transformed subtrees (see `has_transform` in
    /// `fragment_pagination_root`) — this pins that the new branch
    /// preserves the same exclusion instead of splitting it.
    #[test]
    fn nested_inline_root_with_transform_stays_atomic() {
        let html = r#"<!doctype html><html><body style="margin:0; font-size:12px">
            <div>
                <div style="height:90px"></div>
                <p style="margin:0; padding:0; line-height:20px; width:20px; transform:rotate(1deg)">one two three four five six seven eight</p>
            </div>
        </body></html>"#;
        let mut doc = parse(html, 200.0);

        fn find_p(doc: &blitz_dom::BaseDocument, node_id: usize) -> Option<usize> {
            let node = doc.get_node(node_id)?;
            if node
                .element_data()
                .is_some_and(|e| e.name.local.as_ref() == "p")
            {
                return Some(node_id);
            }
            for &child in &node.children {
                if let Some(found) = find_p(doc, child) {
                    return Some(found);
                }
            }
            None
        }
        let body_id = find_body_id(&doc).expect("body must exist");
        let p_id = find_p(&doc, body_id).expect("fixture must contain a <p>");

        let table = run_pass(doc.deref_mut(), 100.0);
        let p_geom = table.get(&p_id).expect("<p> must be recorded in geometry");
        assert_eq!(
            p_geom.fragments.len(),
            1,
            "a transformed multi-line paragraph must stay atomic (one fragment), \
             not be split at line boundaries: {:?}",
            p_geom.fragments
        );
    }

    /// A transformed block is monolithic at every depth: one holding several
    /// paragraphs that crosses the page foot moves whole to the next page,
    /// at the body level and nested, instead of splitting between them.
    #[test]
    fn a_transformed_block_of_paragraphs_moves_whole_at_the_page_foot() {
        for (open, close) in [("", ""), ("<div>", "</div>")] {
            let html = format!(
                r#"<!doctype html><html><body style="margin:0; font-size:12px">{open}
                <div style="height:50px"></div>
                <div id="t" style="transform:rotate(1deg)">
                  <p style="margin:0; line-height:20px">a</p>
                  <p style="margin:0; line-height:20px">b</p>
                  <p style="margin:0; line-height:20px">c</p>
                </div>{close}
            </body></html>"#
            );
            let mut doc = parse(&html, 200.0);
            let t = find_by_id(&doc, "t").unwrap();
            let table = run_pass(doc.deref_mut(), 100.0);
            let pages = table[&t]
                .fragments
                .iter()
                .map(|f| f.page_index)
                .collect::<Vec<_>>();
            assert_eq!(pages, [1], "{open}");
        }
    }

    /// A transformed block taller than a page is not kept whole: it splits
    /// between its children, so none runs past the page foot.
    #[test]
    fn a_transformed_block_taller_than_a_page_splits_between_its_children() {
        let mut rows = String::new();
        for _ in 0..8 {
            rows.push_str(r#"<p style="margin:0; line-height:20px">a</p>"#);
        }
        let html = format!(
            r#"<!doctype html><html><body style="margin:0; font-size:12px">
                <div id="t" style="transform:translateX(1px)">{rows}</div>
            </body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let t = find_by_id(&doc, "t").unwrap();
        let table = run_pass(doc.deref_mut(), 100.0);
        let pages = table[&t]
            .fragments
            .iter()
            .map(|f| f.page_index)
            .collect::<Vec<_>>();
        assert_eq!(pages, [0, 1]);
    }

    /// A transformed paragraph is as atomic at the body level as nested:
    /// crossing the page foot, it moves whole to the next page instead of
    /// splitting between its lines.
    #[test]
    fn a_transformed_paragraph_is_not_split_at_the_body_level() {
        let html = r#"<!doctype html><html><body style="margin:0; font-size:12px">
            <div style="height:50px"></div>
            <p id="p" style="margin:0; padding:0; line-height:20px; width:20px; transform:rotate(1deg)">one two three four</p>
        </body></html>"#;
        let mut doc = parse(html, 200.0);
        let p_id = find_by_id(&doc, "p").unwrap();
        let table = run_pass(doc.deref_mut(), 100.0);
        let pages = table[&p_id]
            .fragments
            .iter()
            .map(|f| f.page_index)
            .collect::<Vec<_>>();
        assert_eq!(pages, [1]);
    }

    /// Devin Review on PR #285 (fulgur-a36m Phase 3.1.5b):
    /// `fragment_block_subtree` had `break-before: page` firing BEFORE
    /// the inter-child gap was folded into `cursor_y`, so the gap was
    /// re-applied AFTER the break-before reset — placing the child at
    /// `y=gap` on the new page instead of `y=0`. The body-level
    /// `fragment_pagination_root` had the correct ordering. This test
    /// pins B's y-coordinate on the new page and would catch the
    /// pre-fix value (gap≈20, was 26.6 in CSS px after Stylo's pt→px).
    ///
    /// Setup: outer wrapper triggers recursion via
    /// `forced_inside`. Inside, A (h=100) at y=0 and B
    /// (h=100) at y=120 with `break-before: page`. The `margin-top:
    /// 20px` on B creates a 20px gap that the bug would leak through.
    ///
    /// css-break-3 §5.2 truncates the margins before a forced break and
    /// keeps those after it: B keeps its own 20px top margin on page 1,
    /// below the page top, where the walker placed it at y=0.
    #[test]
    fn fragment_block_subtree_break_before_after_gap_keeps_the_childs_top_margin() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div id="outer" style="margin: 0; padding: 0">
                <div id="a" style="height: 100px; margin: 0"></div>
                <div id="b" style="margin-top: 20px; break-before: page; height: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);

        // Find every fragment with height ≈ 100 on page 1; B is the
        // only such fragment (outer's page-1 fragment height is the
        // total parent strip, which equals 100 after the fix because
        // only B sits on page 1; outer's page-0 fragment carries
        // A + gap = 120; A is on page 0).
        let b_on_page1: Vec<&Fragment> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.height.to_f32() - 100.0).abs() < 0.5)
            .collect();
        assert!(
            !b_on_page1.is_empty(),
            "expected B fragment on page 1, geom={geom:?}"
        );
        for f in &b_on_page1 {
            let fy = f.y.to_f32();
            assert!(
                (fy - 20.0).abs() < 0.5,
                "B should land at its 20px top margin on the new page (a forced \
                 break keeps the margin after it), but got y={fy}. frag={f:?}",
            );
        }
    }

    /// A first child's `break-before` is its container's (css-break-3 §3.1):
    /// the break goes before the container, which starts the next page,
    /// with or without a top edge. Without the edge the break was dropped,
    /// as no content preceded the child in its container; with one, it
    /// broke inside the container and left its edge alone on the page.
    #[test]
    fn a_first_childs_break_before_breaks_before_its_container() {
        for edge in ["", "padding-top: 10px"] {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0">
                  <div style="height: 50px"></div>
                  <div id="c" style="{edge}">
                    <div id="b" style="break-before: page; height: 20px"></div>
                  </div>
                </body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            let top = if edge.is_empty() { 0.0 } else { 10.0 };
            assert_eq!(
                (placements(&doc, &geom, "c"), placements(&doc, &geom, "b")),
                (vec![(1, 0.0)], vec![(1, top)]),
                "{edge}"
            );
        }
    }

    /// A forced break keeps the top margin of the box after it (css-break-3
    /// §5.2), after a `break-before` and after a `break-after`, at the body
    /// level and nested: the box starts its own 10px margin below the page
    /// top, the 30px gap to the box before the break dropped; and a margin
    /// that collapses through the top of the box's first child is the box's
    /// (15px). The walker placed the box at the page top.
    #[test]
    fn a_child_after_a_forced_break_keeps_its_top_margin() {
        let page_y = |open: &str, close: &str, before: &str, after: &str| {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0">{open}
                  <div style="height: 40px; margin-bottom: 30px; {after}"></div>
                  <div id="b" style="height: 20px; margin-top: 10px; {before}"></div>
                {close}</body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            placements(&doc, &geom, "b")
        };
        for (open, close) in [("", ""), ("<div>", "</div>")] {
            assert_eq!(
                page_y(open, close, "break-before: page", ""),
                [(1, 10.0)],
                "before {open}"
            );
            assert_eq!(
                page_y(open, close, "", "break-after: page"),
                [(1, 10.0)],
                "after {open}"
            );
        }
        // The layout decides whether the first child's margin collapses
        // through the section's top: it does in a `flow-root` too, which
        // Blitz lays out as block flow, and does not below a hairline of
        // padding, where the margin stays inside.
        for (section, expected) in [
            ("", (vec![(1, 15.0)], vec![(1, 15.0)])),
            ("display: flow-root", (vec![(1, 15.0)], vec![(1, 15.0)])),
            ("padding-top: 0.01pt", (vec![(1, 0.0)], vec![(1, 15.0)])),
        ] {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0">
                  <div style="height: 40px"></div>
                  <section id="s" style="break-before: page; {section}">
                    <div id="h" style="height: 20px; margin-top: 15px"></div>
                  </section>
                </body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            assert_eq!(
                (placements(&doc, &geom, "s"), placements(&doc, &geom, "h")),
                expected,
                "{section}"
            );
        }
    }

    /// The top margin of the box an unforced break puts at the top of the next
    /// page, by its `margin-break` (css-break-4 §5.2): `auto` truncates it to
    /// zero, as does `discard`, and `keep` keeps it below the page top. So for
    /// a box placed whole, a paragraph that moves before it splits, a box
    /// entered that moves before its first child, and a box kept whole that
    /// moves with the run its `break-after: avoid` starts; and a margin that
    /// collapses through the top of a box from its first child (15px) is kept
    /// when each box it collapses from keeps its own.
    #[test]
    fn margin_break_decides_the_top_margin_after_an_unforced_break() {
        let page_y = |body: &str, id: &str| {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0; line-height: 20px">{body}</body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            placements(&doc, &geom, id)
        };
        for (value, kept) in [("", 0.0), ("auto", 0.0), ("keep", 10.0), ("discard", 0.0)] {
            let mb = format!("margin-break: {value}");
            let cases = [
                (
                    "placed whole",
                    format!(
                        r#"<div style="height: 70px"></div><div id="b" style="height: 40px; margin-top: 10px; {mb}"></div>"#
                    ),
                ),
                (
                    "paragraph",
                    format!(
                        r#"<div style="height: 70px"></div><p id="b" style="margin: 10px 0 0; {mb}">a<br>b<br>c</p>"#
                    ),
                ),
                (
                    "entered",
                    format!(
                        r#"<div style="height: 70px"></div><div id="b" style="margin-top: 10px; {mb}"><div style="height: 30px"></div><div style="height: 30px"></div></div>"#
                    ),
                ),
                (
                    "kept run",
                    format!(
                        r#"<div style="height: 50px"></div><div id="b" style="height: 20px; margin-top: 10px; break-inside: avoid; break-after: avoid; {mb}"></div><div style="height: 30px"></div>"#
                    ),
                ),
            ];
            for (case, body) in cases {
                assert_eq!(page_y(&body, "b"), [(1, kept)], "{case}, {value}");
            }
            let collapsed = format!(
                r#"<div style="height: 70px"></div><section id="s" style="{mb}"><div id="h" style="height: 40px; margin-top: 15px; {mb}"></div></section>"#
            );
            let kept = kept * 1.5;
            assert_eq!(page_y(&collapsed, "s"), [(1, kept)], "section, {value}");
            assert_eq!(page_y(&collapsed, "h"), [(1, kept)], "its child, {value}");
        }
    }

    /// `margin-break` is each box's own, where margins collapse at a break:
    /// a section that keeps its 5px under a heading that does not keeps 5px,
    /// and a heading that keeps its 15px under a section that does not keeps
    /// 15px; a table below its bottom caption keeps its own, though the walk
    /// reads the table's break values on the box around it and its caption.
    #[test]
    fn margin_break_is_each_collapsed_boxs_own() {
        let page_y = |body: &str, id: &str| {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0; line-height: 20px">{body}</body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            placements(&doc, &geom, id)
        };
        let nested = |section: &str, heading: &str| {
            page_y(
                &format!(
                    r#"<div style="height: 70px"></div><section id="s" style="margin-top: 5px; margin-break: {section}"><div style="height: 40px; margin-top: 15px; margin-break: {heading}"></div></section>"#
                ),
                "s",
            )
        };
        assert_eq!(
            nested("keep", "auto"),
            [(1, 5.0)],
            "the section keeps its own"
        );
        assert_eq!(
            nested("auto", "keep"),
            [(1, 15.0)],
            "the heading keeps its own"
        );
        for (value, kept) in [("auto", 0.0), ("keep", 10.0)] {
            use crate::blitz_adapter::DomPass as _;
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0; line-height: 20px"><div style="height: 70px"></div><table id="t" style="margin-top: 10px; margin-break: {value}; caption-side: bottom; border-spacing: 0"><caption>c</caption><tr><td style="height: 30px; padding: 0"></td></tr></table></body></html>"#
            );
            let mut doc = blitz_adapter::parse(&html, 600.0, &[]);
            blitz_adapter::CaptionRestructurePass
                .apply(&mut doc, &blitz_adapter::PassContext { font_data: &[] });
            blitz_adapter::resolve(&mut doc);
            let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
            assert_eq!(
                placements(&doc, &geom, "t"),
                [(1, kept)],
                "a table above its caption, {value}"
            );
        }
    }

    /// A margin a break keeps is kept as far as the box after it still fits a
    /// page below it: a box kept whole 95px tall keeps 5px of its 10px, not
    /// running past the foot; one that fits with its margin keeps it whole.
    #[test]
    fn a_kept_margin_leaves_the_box_after_the_break_its_page() {
        let page_y = |height: u32| {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0"><div style="height: 50px"></div><div id="b" style="height: {height}px; margin-top: 10px; margin-break: keep; break-inside: avoid"></div></body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            placements(&doc, &geom, "b")
        };
        assert_eq!(page_y(95), [(1, 5.0)], "5px short of a page");
        assert_eq!(page_y(80), [(1, 10.0)], "room for the margin");
    }

    /// A line of a flex container that moves to the next page keeps the top
    /// margin of an item that keeps its own, as a block keeps it.
    #[test]
    fn a_flex_line_that_moves_keeps_its_items_kept_margin() {
        for (value, kept) in [("auto", 0.0), ("keep", 10.0)] {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0"><div style="display: flex; flex-wrap: wrap"><div style="width: 100%; height: 60px"></div><div id="i" style="width: 100%; height: 40px; margin-top: 10px; margin-break: {value}"></div></div></body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            assert_eq!(placements(&doc, &geom, "i"), [(1, kept)], "{value}");
        }
    }

    /// A table row that moves to the next page keeps no margin of its cells:
    /// CSS gives a table's rows and cells none, whatever their `margin-top`.
    #[test]
    fn a_table_row_that_moves_keeps_no_cell_margin() {
        let html = r#"<html><body style="margin: 0; padding: 0"><table style="border-spacing: 0"><tr><td style="height: 60px; padding: 0"></td></tr><tr style="margin-top: 8px; margin-break: keep"><td id="c" style="height: 50px; padding: 0; margin-top: 6px; margin-break: keep"></td></tr></table></body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        assert_eq!(placements(&doc, &geom, "c"), [(1, 0.0)]);
    }

    /// The top margin of the box after a forced break, by its `margin-break`
    /// (css-break-4 §5.2): `auto` and `keep` keep it below the page top,
    /// `discard` truncates it to zero, after a `break-before` and after a
    /// `break-after`.
    #[test]
    fn margin_break_decides_the_top_margin_after_a_forced_break() {
        for (value, kept) in [("auto", 10.0), ("keep", 10.0), ("discard", 0.0)] {
            for (before, after) in [("break-before: page", ""), ("", "break-after: page")] {
                let html = format!(
                    r#"<html><body style="margin: 0; padding: 0">
                      <div style="height: 40px; margin-bottom: 30px; {after}"></div>
                      <div id="b" style="height: 20px; margin-top: 10px; margin-break: {value}; {before}"></div>
                    </body></html>"#
                );
                let mut doc = parse(&html, 600.0);
                let table = blitz_adapter::extract_column_style_table(&doc, &[]);
                let geom =
                    super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
                assert_eq!(
                    placements(&doc, &geom, "b"),
                    [(1, kept)],
                    "{value} {before}{after}"
                );
            }
        }
    }

    /// The fragments of `body` on 100px pages, under `root`'s style, of the
    /// element `id`.
    fn foot_placements(root: &str, body: &str, id: &str) -> Vec<(u32, f32)> {
        let html = format!(
            r#"<html style="{root}"><body style="margin: 0; padding: 0; line-height: 20px">{body}</body></html>"#
        );
        let mut doc = parse(&html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        placements(&doc, &geom, id)
    }

    /// With `-fulgur-page-foot-clearance` on the root, a fragment that a break
    /// point precedes ends that far above the page foot or higher, at any
    /// depth: a box placed whole, a paragraph's line, and a table row. One
    /// that would end closer moves to the next page: with 16px on a 100px
    /// page, a box ending at 84 stays and one ending at 85 moves. Without the
    /// property the page foot is the limit.
    #[test]
    fn a_fragment_ends_above_the_page_foot_clearance() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let pages = |root: &str, body: &str, id: &str| -> Vec<u32> {
            foot_placements(root, body, id)
                .iter()
                .map(|&(page, _)| page)
                .collect()
        };
        let boxes = |filler: u32| {
            format!(
                r#"<div style="height: {filler}px"></div><div id="b" style="height: 10px"></div>"#
            )
        };
        assert_eq!(pages(clearance, &boxes(74), "b"), [0], "ends at 84");
        assert_eq!(pages(clearance, &boxes(75), "b"), [1], "ends at 85");
        assert_eq!(pages("", &boxes(90), "b"), [0], "no clearance");
        let nested = |filler: u32| {
            format!(
                r#"<section><div style="height: {filler}px"></div><div><div id="b" style="height: 10px"></div></div></section>"#
            )
        };
        assert_eq!(pages(clearance, &nested(74), "b"), [0], "nested, at 84");
        assert_eq!(pages(clearance, &nested(75), "b"), [1], "nested, at 85");
        let lines = r#"<div style="height: 30px"></div><p id="p" style="margin: 0; orphans: 1; widows: 1">a<br>b<br>c</p>"#;
        assert_eq!(
            foot_placements(clearance, lines, "p"),
            [(0, 30.0), (1, 0.0)],
            "the line ending at 90 moves"
        );
        assert_eq!(foot_placements("", lines, "p"), [(0, 30.0)]);
        let rows = r#"<table style="border-spacing: 0"><tr><td style="height: 70px; padding: 0"></td></tr><tr><td id="c" style="height: 20px; padding: 0"></td></tr></table>"#;
        assert_eq!(pages(clearance, rows, "c"), [1], "the row ending at 90");
        assert_eq!(pages("", rows, "c"), [0]);
    }

    /// The page-foot clearance holds only where a break point precedes the
    /// fragment: a box at the top of a page taller than the room above the
    /// clearance stays there whole, neither moved nor sliced, and what
    /// follows it moves on; a box taller than the page is sliced at the page
    /// foot, as without the property.
    #[test]
    fn a_box_at_a_page_top_may_end_in_the_page_foot_clearance() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let body =
            r#"<div id="t" style="height: 95px"></div><div id="n" style="height: 5px"></div>"#;
        assert_eq!(foot_placements(clearance, body, "t"), [(0, 0.0)]);
        assert_eq!(foot_placements(clearance, body, "n"), [(1, 0.0)]);
        let tall = r#"<div id="t" style="height: 150px"></div>"#;
        assert_eq!(
            foot_placements(clearance, tall, "t"),
            foot_placements("", tall, "t"),
            "sliced at the page foot"
        );
    }

    /// The `(page, y, height)` of each fragment of the element `id` of
    /// `body` on 100px pages, under `root`'s style.
    fn foot_fragments(root: &str, body: &str, id: &str) -> Vec<(u32, f32, f32)> {
        let html = format!(
            r#"<html style="{root}"><body style="margin: 0; padding: 0; line-height: 20px">{body}</body></html>"#
        );
        let mut doc = parse(&html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        let node_id = find_by_id(&doc, id).unwrap();
        geom.get(&node_id).map_or_else(Vec::new, |g| {
            g.fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                .collect()
        })
    }

    /// A box that fits the page but not the room above the page-foot
    /// clearance (90px: 16px off a 100px page leaves 84) still fits a page:
    /// after content it moves whole to the next page and ends there in the
    /// clearance, as at any page top, instead of splitting as a box taller
    /// than a page would: a box kept whole by `break-inside: avoid`, a table
    /// row kept whole, a transformed box, and a block whose leading unit,
    /// its first child, is that tall (its 5px second child goes on to the
    /// page after, leaving no empty fragment of the block behind).
    #[test]
    fn a_box_only_taller_than_the_room_above_the_clearance_moves_whole() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let halves = r#"<div style="height: 45px"></div><div style="height: 45px"></div>"#;
        let moved = vec![(1, 0.0)];
        let cases = [
            (
                "kept whole",
                format!(
                    r#"<div style="height: 20px"></div><div id="b" style="break-inside: avoid">{halves}</div>"#
                ),
            ),
            (
                "a kept row",
                format!(
                    r#"<div style="height: 20px"></div><table style="border-spacing: 0"><tr style="break-inside: avoid"><td id="b" style="padding: 0">{halves}</td></tr></table>"#
                ),
            ),
            (
                "transformed",
                format!(
                    r#"<div style="height: 20px"></div><div id="b" style="transform: translateX(1px)">{halves}</div>"#
                ),
            ),
            (
                "a leading unit",
                r#"<div style="height: 20px"></div><div id="b"><div style="height: 90px"></div><div style="height: 5px"></div></div>"#
                    .to_string(),
            ),
        ];
        let placed = |body: &str| -> Vec<(u32, f32)> {
            foot_fragments(clearance, body, "b")
                .iter()
                .map(|&(page, y, _)| (page, y))
                .collect()
        };
        assert_eq!(
            cases.map(|(case, body)| (case, placed(&body))),
            [
                ("kept whole", moved.clone()),
                ("a kept row", moved.clone()),
                ("transformed", moved),
                ("a leading unit", vec![(1, 0.0), (2, 0.0)]),
            ]
        );
    }

    /// A parent's fragment on a page that its child crossed reaches the page
    /// foot, below the page-foot clearance, where the child's slice ends: a
    /// 250px box in a section is cut at 100 on each page, and so is the
    /// section.
    #[test]
    fn a_parent_reaches_the_page_foot_where_its_child_is_sliced() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let body = r#"<section id="s"><div id="t" style="height: 250px"></div></section>"#;
        let ends = |id: &str| -> Vec<f32> {
            foot_fragments(clearance, body, id)
                .iter()
                .map(|&(_, y, height)| y + height)
                .collect()
        };
        assert_eq!(ends("t"), [100.0, 100.0, 50.0], "the slices");
        assert_eq!(ends("s"), ends("t"), "the section");
    }

    /// Under `-fulgur-page-foot-margin: page`, a paragraph's margin box ends
    /// by the page foot, its bottom margin taking the page-foot clearance
    /// in: with 16px on a 100px page, a 32px margin lets the paragraph end
    /// at 68, not 69, and a 12px margin, inside the clearance, leaves the
    /// limit at 84. Of a paragraph of several lines, the last line moves.
    /// Without the property the margin is not counted.
    #[test]
    fn a_paragraph_margin_must_fit_above_the_page_foot() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let page = "-fulgur-page-foot-margin: page";
        let page_of = |filler: u32, margin: u32, foot: &str| {
            let body = format!(
                r#"<div style="height: {filler}px"></div><p id="p" style="margin: 0 0 {margin}px; {foot}">a</p>"#
            );
            foot_placements(clearance, &body, "p")[0].0
        };
        assert_eq!(
            [page_of(48, 32, page), page_of(49, 32, page)],
            [0, 1],
            "a 32px margin"
        );
        assert_eq!(
            [page_of(64, 12, page), page_of(65, 12, page)],
            [0, 1],
            "a 12px margin"
        );
        assert_eq!(page_of(64, 32, ""), 0, "not counted");
        let lines = format!(
            r#"<div style="height: 9px"></div><p id="p" style="margin: 0 0 32px; orphans: 1; widows: 1; {page}">a<br>b<br>c</p>"#
        );
        assert_eq!(
            foot_placements(clearance, &lines, "p"),
            [(0, 9.0), (1, 0.0)],
            "the last line, ending at 69"
        );
    }

    /// Under `-fulgur-page-foot-margin: clearance`, a list's margin box ends
    /// above the page-foot clearance, its bottom margin adding to it: with
    /// 16px on a 100px page and a 12px margin, its last item ends at 72 at
    /// most, and moves alone when it would end at 73. Without the property
    /// the margin is not counted.
    #[test]
    fn a_list_margin_adds_to_the_page_foot_clearance() {
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let list = |filler: u32, items: u32, foot: &str| {
            let item = r#"<li style="height: 20px"></li>"#.repeat(items as usize - 1);
            format!(
                r#"<div style="height: {filler}px"></div><ul style="margin: 0 0 12px; padding: 0; list-style: none; {foot}">{item}<li id="i" style="height: 20px"></li></ul>"#
            )
        };
        let foot = "-fulgur-page-foot-margin: clearance";
        let pages = |body: &str| foot_placements(clearance, body, "i")[0].0;
        assert_eq!(
            [pages(&list(52, 1, foot)), pages(&list(53, 1, foot))],
            [0, 1]
        );
        assert_eq!(pages(&list(33, 2, foot)), 1, "the second item, at 73");
        assert_eq!(pages(&list(32, 2, foot)), 0, "the second item, at 72");
        assert_eq!(pages(&list(64, 1, "")), 0, "not counted");
    }

    /// The lines of a box that holds its text directly, a table cell here,
    /// are painted below its top padding on each of its fragments, so they
    /// fit a page only with that padding: three 20px lines in a cell with
    /// 6px of padding, starting at 20, end at 46, 66, and 86, and the third
    /// goes to the next page under a 16px page-foot clearance on a 100px
    /// page, as the lines of a paragraph in a cell do; at 36 with no
    /// clearance, the third would end at 102, past the page foot.
    #[test]
    fn a_cells_own_lines_fit_the_page_below_its_top_padding() {
        let lines_on_first_page = |root: &str, filler: u32, cell: &str| {
            let html = format!(
                r#"<html style="{root}"><body style="margin: 0; padding: 0; line-height: 20px"><div style="height: {filler}px"></div><table style="border-spacing: 0"><tr>{cell}</tr></table></body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            let id = find_by_id(&doc, "c").unwrap();
            geom[&id].line_boundaries.get(1).copied()
        };
        let clearance = "-fulgur-page-foot-clearance: 16px";
        let own = r#"<td id="c" style="padding: 6px 0; orphans: 1; widows: 1">a<br>b<br>c</td>"#;
        let paragraph = r#"<td style="padding: 6px 0"><p id="c" style="margin: 0; orphans: 1; widows: 1">a<br>b<br>c</p></td>"#;
        assert_eq!(
            lines_on_first_page(clearance, 20, paragraph),
            Some(2),
            "the paragraph's control"
        );
        assert_eq!(
            lines_on_first_page(clearance, 20, own),
            Some(2),
            "own lines"
        );
        assert_eq!(
            lines_on_first_page("", 36, own),
            Some(2),
            "at the page foot"
        );
    }

    /// A forced break is taken only where content precedes it on the page
    /// (css-break-3 §3.1 and css-page-3 §5 leave a break at the start of a
    /// page that holds nothing without effect, and the WPT references let such
    /// breaks coalesce): a `break-after`, a `break-before`, and a change of
    /// page name alike. A box with no height is no content: two empty blocks
    /// that break after themselves make one page; an empty box of its own page
    /// name between two others breaks before itself, after the box of content,
    /// and not after itself (WPT `css-page/page-name-display-none-child`); and
    /// boxes of zero height with their own page names, between two boxes of
    /// content, share the second's page (WPT `css-page/page-name-zero-height-001`:
    /// three pages). A line of only no-break spaces is content
    /// ([`holds_a_dropped_line`]): two such blocks breaking after themselves
    /// make two pages (WPT `css-break/overflowed-abs-pos-with-percentage-height`'s
    /// reference).
    #[test]
    fn a_forced_break_with_nothing_on_the_page_is_not_taken() {
        let pages = |body: &str| {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0; line-height: 20px">{body}</body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            super::implied_page_count(&geom)
        };
        assert_eq!(
            pages(r#"<div style="break-after: page"></div><div style="break-after: page"></div>"#),
            1,
            "two empty breaking blocks"
        );
        assert_eq!(
            pages(
                r#"<div style="page: a; height: 20px"></div><div style="page: c"><div style="display: none">c</div></div><div style="page: b; height: 20px"></div>"#
            ),
            2,
            "an empty box between two page names"
        );
        assert_eq!(
            pages(
                r#"<div style="page: a">a</div><div style="page: b; height: 0">b</div><div style="page: c; height: 0">c</div><div style="page: e">e</div><div style="page: f">f</div>"#
            ),
            3,
            "zero-height boxes of their own page names"
        );
        assert_eq!(
            pages(
                r#"<div style="break-after: page">&nbsp;</div><div style="break-after: page">&nbsp;</div>"#
            ),
            2,
            "lines of no-break spaces"
        );
    }

    /// A `break-after` and the next box's `break-before`, forced, are one
    /// break at the one opportunity between them (css-break-3 §3.1), to the
    /// side the later asks for: after `break-after: left` on page 1 (a left
    /// page), `break-before: right` takes page 2, at the body level and from
    /// a container's last child, where each side took a page of its own
    /// (page 4).
    #[test]
    fn adjacent_side_breaks_merge_into_one() {
        for (open, close) in [("", ""), ("<div>", "</div>")] {
            let html = format!(
                r#"<html><body style="margin: 0; padding: 0">
                  <div style="height: 20px; break-after: page"></div>
                  {open}<div style="height: 20px; break-after: left"></div>{close}
                  <div id="r" style="height: 20px; break-before: right"></div>
                </body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
            assert_eq!(placements(&doc, &geom, "r"), [(2, 0.0)], "{open}");
        }
    }

    /// `break-after: avoid` keeps a heading with the box after it
    /// (css-break-3 §4.4): a heading that fits the page foot, before a block
    /// that does not, moves to the next page with it, in either spelling
    /// and from a rule, and so does a run of two headings that avoid a
    /// break after each. A heading that is the last child of a section
    /// avoids the break after the section too: the section is entered and
    /// the break goes before the heading inside it. The walk read no
    /// `avoid` and broke after the heading.
    #[test]
    fn a_heading_with_break_after_avoid_moves_with_its_successor() {
        let page = |css: &str, body: &str| {
            let html = format!(
                r#"<html><head><style>body {{ margin: 0; padding: 0 }}
                  h1, h2 {{ margin: 0; height: 20px }} p {{ margin: 0; height: 60px }}
                  {css}</style></head><body>{body}</body></html>"#
            );
            let mut doc = parse(&html, 600.0);
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table);
            ["h", "n"].map(|id| placements(&doc, &geom, id))
        };
        let moved = [vec![(1, 0.0)], vec![(1, 20.0)]];
        for style in ["break-after: avoid", "page-break-after: avoid"] {
            let body = format!(
                r#"<div style="height: 150px"></div><h2 id="h" style="{style}">H</h2><p id="n">N</p>"#
            );
            assert_eq!(page("", &body), moved, "{style}");
        }
        let flat = r#"<div style="height: 150px"></div><h2 id="h">H</h2><p id="n">N</p>"#;
        assert_eq!(
            page("", flat),
            [vec![(0, 150.0)], vec![(1, 0.0)]],
            "no avoid"
        );
        assert_eq!(
            page("h2 { break-after: avoid-page }", flat),
            moved,
            "a rule"
        );
        let run = r#"<div style="height: 130px"></div><h1 id="h">H</h1><h2>H</h2><p id="n">N</p>"#;
        assert_eq!(
            page("h1, h2 { break-after: avoid }", run),
            [vec![(1, 0.0)], vec![(1, 40.0)]],
            "a run"
        );
        let nested = r#"<section><div style="height: 150px"></div><h2 id="h">H</h2></section><p id="n">N</p>"#;
        assert_eq!(page("h2 { break-after: avoid }", nested), moved, "nested");
    }

    /// The items of a flex line are one line for its breaks whatever their
    /// alignment: a tall item with `break-after: page` and a shorter one
    /// centred beside it have no opportunity between them, so the break goes
    /// after the container, and the short item stays beside the tall one.
    /// The breaks grouped the centred item, which starts below the line's
    /// top, into a line of its own: the break fell between the two items,
    /// inside the row the walk places side by side, and was lost.
    #[test]
    fn a_centred_flex_item_shares_its_line_with_a_tall_item_that_breaks_after_it() {
        let html = r#"<html><body style="margin: 0; padding: 0">
              <div id="row" style="display: flex; align-items: center">
                <div id="a" style="width: 50px; height: 60px; break-after: page"></div>
                <div id="b" style="width: 50px; height: 20px"></div>
              </div>
              <div id="n" style="height: 10px"></div>
            </body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        assert_eq!(
            (placements(&doc, &geom, "b"), placements(&doc, &geom, "n")),
            (vec![(0, 20.0)], vec![(1, 0.0)])
        );
    }

    /// A paragraph that avoids a break after it keeps only its last
    /// `widows` lines with the block after it (css-break-3 §4.4 forbids the
    /// break after it, not those between its lines): eight 20 px lines from
    /// y 100 of a 300 px page, before a 60 px block that does not fit, leave
    /// six lines there and take two to the next page with the block. The
    /// paragraph moved whole and left 200 px of the page blank.
    #[test]
    fn a_paragraph_avoiding_a_break_after_it_keeps_only_its_last_lines_with_its_successor() {
        let html = r#"<html><body style="margin: 0; padding: 0; line-height: 20px">
              <div style="height: 100px"></div>
              <p id="p" style="margin: 0; break-after: avoid">1<br>2<br>3<br>4<br>5<br>6<br>7<br>8</p>
              <div id="n" style="height: 60px"></div>
            </body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 300.0_f32.as_px(), &table);
        assert_eq!(
            (placements(&doc, &geom, "p"), placements(&doc, &geom, "n")),
            (vec![(0, 100.0), (1, 0.0)], vec![(1, 40.0)])
        );
        assert_eq!(
            geom[&find_by_id(&doc, "p").unwrap()].line_boundaries,
            [0, 6, 8]
        );
    }

    /// No break opportunity lies inside an atomic inline (css-break-3 §4.1):
    /// a `break-before` in an `inline-block` does not make the walk enter a
    /// box kept whole around it, which moves whole to the next page when it
    /// crosses the foot. The walk entered it for the break, so it split
    /// between its children where it stood.
    #[test]
    fn a_break_inside_an_inline_block_does_not_split_its_kept_ancestor() {
        let html = r#"<html><body style="margin: 0; padding: 0; line-height: 20px">
              <div style="height: 50px"></div>
              <div id="k" style="break-inside: avoid">
                <div style="height: 40px"></div>
                <span style="display: inline-block; width: 50px">
                  <span style="display: block; break-before: page">x</span>
                </span>
              </div>
            </body></html>"#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        assert_eq!(placements(&doc, &geom, "k"), [(1, 0.0)]);
    }

    /// The body is walked as any block: a flex body's items sit side by side
    /// at their layout position, as a nested flex container's do, instead of
    /// one below the other.
    #[test]
    fn a_flex_body_places_its_items_side_by_side() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; display: flex">
              <div id="a" style="flex: 1; height: 50px"></div>
              <div id="b" style="flex: 1; height: 50px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let geom = run_pass(doc.deref_mut(), 100.0);
        let at = |id: &str| {
            geom[&find_by_id(&doc, id).unwrap()]
                .fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32()))
                .collect::<Vec<_>>()
        };
        assert_eq!((at("a"), at("b")), (vec![(0, 0.0)], vec![(0, 0.0)]));
    }

    /// A grid of one column stacks its items: a box taller than a page in
    /// an item alone in its row is sliced, as in block flow (WPT
    /// `css-page/monolithic-overflow-008`).
    #[test]
    fn a_tall_box_in_the_item_of_a_one_column_grid_is_sliced() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="display: grid">
                <div><div id="a" style="height: 250px"></div></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let geom = run_pass(doc.deref_mut(), 100.0);
        let a = &geom[&find_by_id(&doc, "a").unwrap()].fragments;
        assert_eq!(a.len(), 3, "{a:?}");
    }

    /// A flex column places its items one below the other, as block flow
    /// does: an item taller than a page is sliced, and what follows it
    /// starts after its last slice (WPT `css-page/monolithic-overflow-005`).
    #[test]
    fn a_tall_item_of_a_flex_column_is_sliced() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="display: flex; flex-flow: column">
                <div id="a" style="height: 250px"></div>
                <div id="b" style="height: 20px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let geom = run_pass(doc.deref_mut(), 100.0);
        let at = |id: &str| {
            geom[&find_by_id(&doc, id).unwrap()]
                .fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            (at("a"), at("b")),
            (
                vec![(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)],
                vec![(2, 50.0, 20.0)]
            )
        );
    }

    /// A flex body's item taller than a page is sliced, as a nested flex
    /// container's is: the walk places the line's items side by side from
    /// its start, so the item beside it stays on the first page.
    #[test]
    fn a_tall_item_of_a_flex_body_is_sliced() {
        let html = r#"
            <html><body style="margin: 0; padding: 0; display: flex">
              <div id="a" style="flex: 1; height: 250px"></div>
              <div id="b" style="flex: 1; height: 50px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let geom = run_pass(doc.deref_mut(), 100.0);
        assert_eq!(
            fragments_of(&doc, &geom, "a"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)]
        );
        assert_eq!(fragments_of(&doc, &geom, "b"), [(0, 0.0, 50.0)]);
    }

    /// A forced break keeps the top margin of the child after it at the body
    /// level (css-break-3 §5.2), whatever the child's height: one that fills
    /// the page with its margin starts below the page top too, by its own
    /// margin, not by the gap to the box before the break.
    #[test]
    fn a_page_filling_child_after_a_forced_break_keeps_its_top_margin() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 40px; margin: 0"></div>
              <div style="height: 90px; margin-top: 10px; break-before: page"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);

        let second_on_page1: Vec<&Fragment> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| f.page_index == 1 && (f.height.to_f32() - 90.0).abs() < 0.5)
            .collect();
        assert_eq!(
            second_on_page1.len(),
            1,
            "expected only the second child on page 1, geom={geom:?}"
        );
        assert!(
            (second_on_page1[0].y.to_f32() - 10.0).abs() < 0.5,
            "a child after a forced break keeps its top margin; geom={geom:?}"
        );
    }

    #[test]
    fn implied_page_count_is_one_for_empty_geometry() {
        let geom = PaginationGeometryTable::new();
        assert_eq!(super::implied_page_count(&geom), 1);
    }

    #[test]
    fn implied_page_count_uses_max_index_plus_one() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(1).or_default().fragments.push(Fragment {
            page_index: 2,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 1.0_f32.as_px(),
            height: 1.0_f32.as_px(),
        });
        assert_eq!(super::implied_page_count(&geom), 3);
    }

    /// `geometry_text` writes one line per node and fragment, in tree order
    /// and then in the fragments' vector order: the path of tag names, `#id`,
    /// and child indices, the box rounded to 0.01 px, then `repeat`, every
    /// line boundary, and the lift.
    #[test]
    fn geometry_text_writes_one_line_per_fragment_in_tree_order() {
        let doc = parse(
            r#"<html><body style="margin:0"><div id="a"><p>x<br>y</p></div></body></html>"#,
            200.0,
        );
        let a = find_by_id(&doc, "a").expect("div#a");
        let p = doc.get_node(a).expect("div").children[0];
        let fragment = |page_index: u32, y: f32| Fragment {
            page_index,
            x: 0.0_f32.as_px(),
            y: y.as_px(),
            width: 200.0_f32.as_px(),
            height: 20.0_f32.as_px(),
        };
        let mut geom = PaginationGeometryTable::new();
        geom.insert(
            p,
            PaginationGeometry {
                fragments: vec![fragment(1, 0.0), fragment(0, 80.004)],
                line_boundaries: vec![0, 1, 2],
                ..Default::default()
            },
        );
        geom.insert(
            a,
            PaginationGeometry {
                fragments: vec![fragment(0, 79.996)],
                is_repeat: true,
                lift_px: 2.5,
                ..Default::default()
            },
        );
        assert_eq!(
            geometry_text(&geom, &doc),
            "html[0]/body[1]/div#a[0] page=0 x=0.00 y=80.00 w=200.00 h=20.00 repeat lift=2.50\n\
             html[0]/body[1]/div#a[0]/p[0] page=1 x=0.00 y=0.00 w=200.00 h=20.00 lines=0,1,2\n\
             html[0]/body[1]/div#a[0]/p[0] page=0 x=0.00 y=80.00 w=200.00 h=20.00 lines=0,1,2\n"
        );
    }

    /// Five pages: node 1 (`a`) starts on page 1 and runs on to page 2;
    /// page 3 holds only the repeated node 4 (`x`, on every page); nodes 5
    /// (`c`) and 7 (`d`) start on page 4; node 8 (unnamed) on page 5.
    #[test]
    fn collect_page_facts_names_each_page_by_its_first_start_and_finds_the_blank_ones() {
        let fragment = |page_index: u32| Fragment {
            page_index,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 1.0_f32.as_px(),
            height: 1.0_f32.as_px(),
        };
        let mut geom = PaginationGeometryTable::new();
        let mut names = crate::blitz_adapter::UsedPageNameTable::new();
        for (node, start, pages) in [
            (8, None, vec![4]),
            (7, Some("d"), vec![3]),
            (5, Some("c"), vec![3]),
            (1, Some("a"), vec![0, 1]),
        ] {
            geom.entry(node).or_default().fragments = pages.into_iter().map(fragment).collect();
            names.insert(node, (start.map(str::to_string), None));
        }
        let repeat = geom.entry(4).or_default();
        repeat.is_repeat = true;
        repeat.fragments = (0..5).map(fragment).collect();
        names.insert(4, (Some("x".to_string()), None));

        let facts = |name: Option<&str>, blank| super::PageFacts {
            name: name.map(str::to_string),
            blank,
        };
        assert_eq!(
            super::collect_page_facts(&geom, &names),
            [
                facts(Some("a"), false),
                facts(Some("a"), false),
                facts(Some("a"), true),
                facts(Some("c"), false),
                facts(None, false),
            ]
        );
    }

    /// fulgur-s67g Phase 2.1: a 3-line paragraph at the top of a page
    /// that overflows the strip after line 2 cannot split between
    /// line 2 and the final line without a 1-line tail, below the
    /// widows = 2 minimum, and pulling the split back would leave a
    /// 1-line head, below orphans = 2. Moving the paragraph gains
    /// nothing at the top of a page, so css-break-3 §4.4 relaxes the
    /// minimums: the split goes after the two lines that fit.
    #[test]
    fn widow_minimum_is_relaxed_at_the_top_of_a_page() {
        let mut geom = PaginationGeometryTable::new();
        // Each line 75px; cumulative bottoms at 75, 150, 225.
        // Page strip = 200, so naturally we'd split at line 2 (bottom
        // 225 > 200), leaving 1 line in the tail — widow violated.
        let lines = vec![(0.0, 75.0), (75.0, 150.0), (150.0, 225.0)];
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            /*child_id=*/ 1,
            /*paragraph_x=*/ 0.0,
            /*width=*/ 100.0,
            /*initial_cursor_y=*/ 0.0,
            /*initial_page_index=*/ 0,
            /*page_height_px=*/ 200.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 2, "relaxed split → 2 fragments");
        assert_eq!(new_page, 1);
        assert!(
            (new_cursor - 75.0).abs() < 0.01,
            "cursor is the one-line tail on page 1, got {new_cursor}",
        );
        let frags = &geom.get(&1).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].page_index, 0);
        assert!((frags[0].height.to_f32() - 150.0).abs() < 0.01);
        assert_eq!(frags[1].page_index, 1);
    }

    /// fulgur-s67g Phase 2.1: a 4-line paragraph splittable at line 2
    /// (first 2 lines on page 0, last 2 on page 1) honours both
    /// orphans = 2 and widows = 2.
    #[test]
    fn widow_orphan_minimum_allows_balanced_split() {
        let mut geom = PaginationGeometryTable::new();
        // Each line 75px; bottoms at 75, 150, 225, 300.
        // Page strip = 200 → natural split at line 2 (bottom 225 > 200).
        // first_size = 2 ≥ orphans, remaining_size = 2 ≥ widows. Split OK.
        let lines = vec![(0.0, 75.0), (75.0, 150.0), (150.0, 225.0), (225.0, 300.0)];
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            0.0,
            0,
            200.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 2, "valid split → 2 fragments");
        assert_eq!(new_page, 1);
        let frags = &geom.get(&1).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].page_index, 0);
        assert_eq!(frags[1].page_index, 1);
        // First fragment: lines 0-1 (height = 150).
        assert!((frags[0].height.to_f32() - 150.0).abs() < 0.01);
        // Second fragment: lines 2-3 (height = 150 in para-local).
        assert!((frags[1].height.to_f32() - 150.0).abs() < 0.01);
        // cursor_y on page 1 = paragraph_top_in_body (0.0) + 150 = 150.
        assert!((new_cursor - 150.0).abs() < 0.01);
    }

    /// fulgur-s67g Phase 2.1: orphan violation. A 3-line paragraph
    /// at the top of a page with overflow at line 1 (only line 0
    /// fits) would put just 1 line in the first fragment — below
    /// orphans = 2. Moving gains nothing at the top of a page, so the
    /// minimum is relaxed (css-break-3 §4.4) and each line takes the
    /// page it fits on, rather than an oversized fragment being
    /// emitted into the 100px strip.
    #[test]
    fn orphan_minimum_is_relaxed_when_one_line_fits_at_the_top_of_a_page() {
        let mut geom = PaginationGeometryTable::new();
        // Lines 75px; bottoms at 75, 150, 225.
        // Page strip = 100 → natural split at line 1 (bottom 150 > 100).
        // first_size = 1 < orphans=2, but the paragraph is at the page top.
        let lines = vec![(0.0, 75.0), (75.0, 150.0), (150.0, 225.0)];
        let (new_page, _new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            0.0,
            0,
            100.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 3, "relaxed → one line per fragment");
        assert_eq!(new_page, 2);
        let frags = &geom.get(&1).unwrap().fragments;
        assert_eq!(frags.len(), 3);
        for (page, frag) in frags.iter().enumerate() {
            assert_eq!(frag.page_index, page as u32);
            assert!((frag.height.to_f32() - 75.0).abs() < 0.01);
        }
    }

    #[test]
    fn taller_than_page_block_emits_single_oversize_fragment() {
        // 1000px block on a 800px page. Block-only fragmenter emits it whole
        // on the page where its top lands, with the full height — true
        // split is the next iteration's job.
        let html = r#"
            <html><body>
              <div style="height: 1000px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = run_pass(&mut doc, 800.0);
        // Phase 2.3 fix: body + 1 oversized child = 2 entries.
        assert_eq!(table.len(), 2);
        // The oversized child is the entry whose height ≈ 1000.
        let oversize = table
            .values()
            .find(|g| (g.fragments[0].height.to_f32() - 1000.0).abs() < 1.0)
            .expect("oversized child fragment");
        assert_eq!(oversize.fragments.len(), 1);
        assert_eq!(oversize.fragments[0].page_index, 0);
    }

    /// fulgur-yb27: `fragment_block_subtree` must walk `layout_children`
    /// so anonymous block wrappers Stylo synthesizes around inline-
    /// level siblings (CSS 2.1 §9.2.1.1) are visited. Without this,
    /// a tail inline string after a tall block sibling never
    /// fragments to the next page — the block consumes page 1, the
    /// trailing inline run is treated as a zero-height text node
    /// (`final_layout` defaults), and pagination terminates with a
    /// single fragment.
    #[test]
    fn fragment_block_subtree_walks_layout_children_for_anon_block_synthesis() {
        // Outer wrapper > [tall block sibling, trailing inline text].
        // Stylo wraps the trailing text in an anon block whose Taffy
        // location.y is past the page boundary.
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="width: 200px;">
                <div style="height: 300px; width: 50px; background:hotpink;"></div>
                trailing tail
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table);
        // Page count must be ≥ 2 — without yb27 the trailing inline
        // text never reaches a new page (single fragment, page 0 only).
        let max_page = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .map(|f| f.page_index)
            .max()
            .unwrap_or(0);
        assert!(
            max_page >= 1,
            "yb27: anon block from mixed inline/block siblings must \
             paginate to a new page; max_page={max_page}",
        );
    }

    /// fulgur-oc51: when `fragment_block_subtree`'s recursion advances
    /// `page_index`, the parent's pre-recursion-page span must be
    /// recorded as a fragment. Without this, a tall nested subtree
    /// that crosses pages inside the recursion lifts the parent's
    /// only fragment to the *last* page (line 1799 close), leaving
    /// page 1 with no parent paint at all (background/borders gone).
    #[test]
    fn fragment_block_subtree_emits_parent_fragment_when_recursion_crosses_page() {
        // Two-deep nesting: outer (with background) > inner > [tall
        // child, trailing inline]. Inner's recursion will cross the
        // page boundary; outer (the <section>) must still get a
        // fragment on page 0.
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <section style="width: 200px;">
                <div style="width: 200px;">
                  <div style="height: 300px; width: 50px; background:hotpink;"></div>
                  tail
                </div>
              </section>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        // Locate the <section> node id explicitly so the assertion
        // targets the specific block that should keep a page-0
        // fragment — without this, a wide body fragment on page 0
        // would satisfy a generic "any wide page-0 fragment" check
        // even when the section itself has no page-0 entry.
        let section_id =
            find_node_by_local_name(&doc, "section").expect("fixture must contain a <section>");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table);
        let section_geom = geom.get(&section_id).unwrap_or_else(|| {
            panic!("oc51: <section> (node_id={section_id}) missing from geometry; geom={geom:?}")
        });
        let has_page_0 = section_geom.fragments.iter().any(|f| f.page_index == 0);
        assert!(
            has_page_0,
            "oc51: <section> (node_id={section_id}) must keep a page-0 \
             fragment when its nested recursion crosses a page boundary; \
             section_fragments={:?}",
            section_geom.fragments,
        );
        // Must also reach page 1+ — otherwise the test isn't actually
        // exercising the recursion-cross-page path.
        let max_page = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .map(|f| f.page_index)
            .max()
            .unwrap_or(0);
        assert!(
            max_page >= 1,
            "oc51: fixture must paginate to ≥ 2 pages to exercise the \
             recursion-cross-page path; max_page={max_page}",
        );
    }

    /// Locate the first node in the document whose element local name
    /// matches `tag`. Used by tests that need a node_id reference for
    /// a specific HTML element without depending on Stylo's internal
    /// node numbering.
    fn find_node_by_local_name(doc: &blitz_html::HtmlDocument, tag: &str) -> Option<usize> {
        use std::ops::Deref;
        let base: &blitz_dom::BaseDocument = doc.deref();
        let root_id = base.root_element().id;
        fn walk(base: &blitz_dom::BaseDocument, id: usize, tag: &str) -> Option<usize> {
            let node = base.get_node(id)?;
            if let Some(elem) = node.element_data()
                && elem.name.local.as_ref() == tag
            {
                return Some(id);
            }
            for &child_id in &node.children {
                if let Some(found) = walk(base, child_id, tag) {
                    return Some(found);
                }
            }
            None
        }
        walk(base, root_id, tag)
    }

    /// fulgur-puml: `explicit_inset_axes` must classify each axis as
    /// explicit (length/percentage inset present) vs auto, since the nested-
    /// abs base selection depends on it. A silent regression here would
    /// mis-anchor abs descendants across a static intermediate.
    #[test]
    fn explicit_inset_axes_classifies_per_axis() {
        use std::ops::Deref;
        // (fragment, (x_explicit, y_explicit))
        let cases: &[(&str, (bool, bool))] = &[
            ("<div style=\"position:absolute\"></div>", (false, false)),
            (
                "<div style=\"position:absolute; top:10px\"></div>",
                (false, true),
            ),
            (
                "<div style=\"position:absolute; right:0\"></div>",
                (true, false),
            ),
            (
                "<div style=\"position:absolute; left:0; right:0\"></div>",
                (true, false),
            ),
            (
                "<div style=\"position:absolute; top:0; left:0\"></div>",
                (true, true),
            ),
        ];
        for (frag, expected) in cases {
            let html = format!("<html><body>{frag}</body></html>");
            let doc = parse(&html, 600.0);
            let id = find_node_by_local_name(&doc, "div").expect("div present");
            let node = doc.deref().get_node(id).expect("node present");
            assert_eq!(explicit_inset_axes(node), *expected, "fragment: {frag}");
        }
    }

    /// coderabbit: body containing only a `position: running()` element
    /// plus a tall `position: absolute` div must treat the body as
    /// having no in-flow content so `may_extend_pages = true`.
    /// Without the running_store guard, the running child increments
    /// `body_has_in_flow_content` and truncates the abs subtree.
    #[test]
    fn position_absolute_body_direct_running_only_body_extends_pages() {
        use crate::blitz_adapter;
        use crate::gcpm::parser::parse_gcpm;
        use std::sync::Arc;

        let css = ".header { position: running(pageHeader); }";
        let html = r#"<!DOCTYPE html>
<html><head></head>
<body style="margin:0">
<div class="header">Doc Header</div>
<div style="position: absolute; top: 0; width: 100px; height: 1800px">x</div>
</body></html>"#;

        let gcpm = parse_gcpm(css);
        let fonts: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut doc = blitz_adapter::parse(html, 600.0, &fonts);
        let pass = blitz_adapter::RunningElementPass::new(gcpm.running_mappings.clone());
        let pass_ctx = blitz_adapter::PassContext { font_data: &fonts };
        blitz_adapter::apply_single_pass(&pass, &mut doc, &pass_ctx);
        let store = pass.into_running_store();
        blitz_adapter::resolve(&mut doc);

        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            Some(&store),
        );

        let max_page = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| (f.height.to_f32() - 1800.0).abs() < 0.5)
            .map(|f| f.page_index)
            .max();
        assert!(
            max_page.is_some_and(|p| p >= 2),
            "tall abs div in running-only body should extend to page 2; max_page={max_page:?}"
        );
    }

    /// coderabbit: abs subtree starting beyond the current page budget
    /// must still emit fragments when `may_extend_pages` is true.
    /// Regression for the condition `first_page_f < total_pages` that
    /// blocked fragment emission even when the absolute pass is
    /// responsible for extending the page count.
    #[test]
    fn position_absolute_body_direct_beyond_page_budget_extends_pages() {
        let html = r#"
            <html><body style="margin:0">
              <div style="position: absolute; top: 1600px; width: 100px; height: 100px">x</div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let mut geom = PaginationGeometryTable::new();
        super::append_position_absolute_body_direct_fragments(
            &mut geom,
            doc.deref_mut(),
            1,
            600.0,
            800.0,
            None,
        );
        let max_page = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .map(|f| f.page_index)
            .max();
        assert!(
            max_page.is_some_and(|p| p >= 2),
            "abs div at top:1600px with 800px pages should land on page 2; max_page={max_page:?}"
        );
    }

    #[test]
    fn grid_row_leaf_cells_move_together_across_page_boundary() {
        // 2-col grid, each leaf cell 60px tall.
        // spacer 80px pushes grid to y=80 on a 100px page.
        // pre-fix: cell 2 pushed to page 1 at y=0 (whole cell) while cell 1
        // stayed. Neither cell places anything in the 20 px left, so the
        // row moves to page 1 whole, both cells at its top.
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 80px"></div>
              <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                <div style="height: 60px; width: 100px"></div>
                <div style="height: 60px; width: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 400.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);

        // collect all 60px-tall, 100px-wide fragments (the two leaf cells)
        let mut frags: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 60.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        frags.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            frags,
            [(1, 0.0, 0.0), (1, 100.0, 0.0)],
            "both leaf cells move to page 1, side by side at its top"
        );
    }

    #[test]
    fn flex_row_leaf_cells_move_together_across_page_boundary() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 80px"></div>
              <div style="display: flex; width: 200px;">
                <div style="height: 60px; width: 100px; flex: 0 0 100px"></div>
                <div style="height: 60px; width: 100px; flex: 0 0 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 400.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);

        let mut frags: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 60.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        frags.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(
            frags,
            [(1, 0.0, 0.0), (1, 100.0, 0.0)],
            "both leaf cells move to page 1, side by side at its top"
        );
    }

    #[test]
    fn grid_row_recursive_cells_cosplit_across_page_boundary() {
        // 2-col grid, each cell has 2 inner divs (40px each) = 80px total.
        // spacer 50px, page_height 100px → grid starts at y=50.
        // Row bottom = 50 + 80 = 130 > 100 → crosses page boundary.
        //
        // pre-fix: cell 2 recursion starts at page 1 cursor=0 (because
        //   cell 1's recursion advanced page_index to 1) → both inner
        //   divs of cell 2 land entirely on page 1.
        // post-fix: both cells start recursion at page 0 cursor=50 →
        //   the first inner div (40px) of each stays on page 0 (y=50..90)
        //   and the second moves to page 1.
        //   Specifically: both columns' first inner divs must appear on page 0.
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 50px"></div>
              <div style="display: grid; grid-template-columns: 100px 100px; width: 200px;">
                <div style="width: 100px">
                  <div style="height: 40px; width: 100px"></div>
                  <div style="height: 40px; width: 100px"></div>
                </div>
                <div style="width: 100px">
                  <div style="height: 40px; width: 100px"></div>
                  <div style="height: 40px; width: 100px"></div>
                </div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 400.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);

        // inner divs: 40px tall, 100px wide
        let mut inner: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 40.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        inner.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Each column has 2 inner divs (40px each).
        // After the fix, both columns' FIRST inner div must appear on page 0
        // (at y≈50, since the grid row starts there).
        // Pre-fix: only cell 1's first inner div is on page 0; cell 2's
        // recursion starts from page 1 so its inner divs are at y=0,40 on page 1.
        let _first_divs_page0: Vec<_> = inner
            .iter()
            .filter(|(p, _, y)| *p == 0 && *y > 40.0)
            .collect();
        // The critical assertion: the RIGHT column's first inner div must also
        // appear on page 0 (at x=100, y≈50). Pre-fix: cell 2's recursion starts
        // from page 1, so its inner divs land at x=100 only on page 1.
        let right_col_page0: Vec<_> = inner
            .iter()
            .filter(|(p, x, y)| *p == 0 && (x - 100.0).abs() < 0.5 && *y > 40.0)
            .collect();
        assert!(
            !right_col_page0.is_empty(),
            "right column's first inner div must appear on page 0 at x=100 (pre-fix: only on page 1); inner={inner:?}"
        );
    }

    #[test]
    fn flex_row_recursive_cells_cosplit_across_page_boundary() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 50px"></div>
              <div style="display: flex; width: 200px;">
                <div style="width: 100px; flex: 0 0 100px">
                  <div style="height: 40px; width: 100px"></div>
                  <div style="height: 40px; width: 100px"></div>
                </div>
                <div style="width: 100px; flex: 0 0 100px">
                  <div style="height: 40px; width: 100px"></div>
                  <div style="height: 40px; width: 100px"></div>
                </div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 400.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);

        let mut inner: Vec<(u32, f32, f32)> = geom
            .values()
            .flat_map(|g| g.fragments.iter())
            .filter(|f| {
                (f.height.to_f32() - 40.0).abs() < 0.5 && (f.width.to_f32() - 100.0).abs() < 0.5
            })
            .map(|f| (f.page_index, f.x.to_f32(), f.y.to_f32()))
            .collect();
        inner.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let right_col_page0: Vec<_> = inner
            .iter()
            .filter(|(p, x, y)| *p == 0 && (x - 100.0).abs() < 0.5 && *y > 40.0)
            .collect();
        assert!(
            !right_col_page0.is_empty(),
            "right column's first inner div must appear on page 0 at x=100 (pre-fix: only on page 1); inner={inner:?}"
        );
    }

    // ── break-after: page in nested block subtree (lines 2263-2279) ─────────────

    /// `break-after: page` on a child inside a non-body container must close
    /// the parent's current-page fragment and advance page_index, so the
    /// sibling that follows lands on page 1.
    #[test]
    fn fragment_block_subtree_break_after_page_pushes_sibling_to_next_page() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div>
                <div style="height: 100px; break-after: page"></div>
                <div style="height: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-after: page must push the following sibling onto page 1; geom={geom:?}"
        );
    }

    // ── zero-height body-direct children with break (lines 639-668) ─────────────

    /// A zero-height body-direct element with `break-before: page` must fire
    /// a page break when cursor_y > 0 (lines 643-644).
    #[test]
    fn zero_height_body_direct_break_before_page_fires_page_break() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 100px"></div>
              <div style="height: 0px; break-before: page"></div>
              <div style="height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-before: page on a zero-height body-direct child must advance to page 1; \
             geom={geom:?}"
        );
    }

    /// A zero-height body-direct element with `break-after: page` must fire
    /// a page break (lines 667-668).
    #[test]
    fn zero_height_body_direct_break_after_page_fires_page_break() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 100px"></div>
              <div style="height: 0px; break-after: page"></div>
              <div style="height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-after: page on a zero-height body-direct child must advance to page 1; \
             geom={geom:?}"
        );
    }

    // ── zero-height nested children with break (lines 1900-1953) ────────────────

    /// A zero-height element inside a block container with `break-before: page`
    /// must fire a page break and flush a parent fragment (lines 1900-1918).
    #[test]
    fn zero_height_nested_break_before_page_fires_page_break() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div>
                <div style="height: 100px"></div>
                <div style="height: 0px; break-before: page"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-before: page on a zero-height nested child must advance to page 1; \
             geom={geom:?}"
        );
    }

    /// A zero-height element inside a block container with `break-after: page`
    /// must fire a page break and flush a parent fragment (lines 1936-1953).
    #[test]
    fn zero_height_nested_break_after_page_fires_page_break() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div>
                <div style="height: 100px"></div>
                <div style="height: 0px; break-after: page"></div>
                <div style="height: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-after: page on a zero-height nested child must advance to page 1; \
             geom={geom:?}"
        );
    }

    /// `left`, `right`, `recto`, and `verso` breaks skip one page when the
    /// next page is of the other side (index 0 is a right page): before and
    /// after, at the body level and nested, zero-height or not, and on a
    /// page a `break-after: page` reached with nothing on it yet. A skipped
    /// page holds no fragment, not even one of the nested breaks' parents.
    #[test]
    fn side_breaks_skip_a_page_to_the_wanted_side() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div id="a" style="height: 100px"></div>
              <div id="b" style="height: 100px; break-before: right"></div>
              <div id="c" style="height: 100px; break-before: verso"></div>
              <div id="d" style="height: 100px; break-after: left"></div>
              <div id="e" style="height: 100px; break-after: page"></div>
              <div id="f" style="height: 100px; break-before: verso"></div>
              <div id="p1">
                <div id="g" style="height: 100px"></div>
                <div id="h" style="height: 100px; break-before: left"></div>
              </div>
              <div>
                <div id="i" style="height: 100px; break-after: page"></div>
                <div id="j" style="height: 100px; break-before: left"></div>
              </div>
              <div id="k" style="height: 0px; break-before: left"></div>
              <div>
                <div id="l" style="height: 100px"></div>
                <div id="m" style="height: 0px; break-before: recto"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        let pages = |id: &str| -> Vec<u32> {
            let node = find_by_id(&*doc, id).expect(id);
            geom[&node].fragments.iter().map(|f| f.page_index).collect()
        };
        let firsts: Vec<u32> = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        ]
        .into_iter()
        .map(|id| pages(id)[0])
        .collect();
        assert_eq!(
            firsts,
            [0, 2, 3, 3, 5, 7, 7, 9, 9, 11, 13, 13, 14],
            "geom={geom:?}"
        );
        assert_eq!(pages("p1"), [7, 9], "geom={geom:?}");
    }

    #[test]
    fn page_on_side_moves_one_page_to_the_wanted_side() {
        use crate::draw_primitives::PageSide::{Left, Right};
        let ltr = parse("<html><body></body></html>", 600.0);
        assert_eq!(super::page_on_side(&ltr, 1, Some(Right)), 2);
        assert_eq!(super::page_on_side(&ltr, 2, Some(Right)), 2);
        assert_eq!(super::page_on_side(&ltr, 2, Some(Left)), 3);
        assert_eq!(super::page_on_side(&ltr, 1, Some(Left)), 1);
        assert_eq!(super::page_on_side(&ltr, 1, None), 1);
        // `direction: rtl` on the root; Blitz maps no `dir` attribute to it.
        let rtl = parse(r#"<html style="direction:rtl"><body></body></html>"#, 600.0);
        assert_eq!(super::page_on_side(&rtl, 1, Some(Left)), 2);
        assert_eq!(super::page_on_side(&rtl, 2, Some(Left)), 2);
        assert_eq!(super::page_on_side(&rtl, 2, Some(Right)), 3);
    }

    // ── record_subtree_walk: position:fixed skip (line 3386) ────────────────────

    /// A `position: fixed` child inside an abs subtree must be skipped by
    /// `record_subtree_fragments_at_offset`'s walk (line 3386 `continue`).
    /// The fixed pass handles those separately; the abs walk must not emit them.
    #[test]
    fn record_subtree_walk_skips_position_fixed_child() {
        let html = r#"
            <html><body style="margin: 0">
              <div id="abs_root" style="position: absolute; top: 0; left: 0; width: 200px; height: 200px">
                <div id="fixed_child" style="position: fixed; top: 10px; left: 10px; width: 50px; height: 50px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let abs_id = find_by_id(doc.deref_mut(), "abs_root").expect("abs_root");
        let fixed_id = find_by_id(doc.deref_mut(), "fixed_child").expect("fixed_child");
        let mut geom = PaginationGeometryTable::new();
        super::record_subtree_fragments_at_offset(
            &mut geom,
            doc.deref_mut(),
            abs_id,
            (0.0, 0.0),
            (0.0, 0.0),
            800.0,
            800.0,
            1,
            false,
            &mut 0,
            None,
        );
        assert!(
            !geom.contains_key(&fixed_id),
            "position:fixed child must not be emitted by the abs subtree walk; \
             geom keys={:?}",
            geom.keys().collect::<Vec<_>>()
        );
    }

    // ── record_subtree_walk: position:absolute recursion (lines 3387-3441) ──────

    /// A `position: absolute` child nested inside an abs subtree must be
    /// recursively walked and appear in geometry (lines 3387-3441).
    #[test]
    fn record_subtree_walk_recurses_into_nested_position_absolute_child() {
        let html = r#"
            <html><body style="margin: 0">
              <div id="abs_root" style="position: absolute; top: 0; left: 0; width: 200px; height: 200px">
                <div id="nested_abs" style="position: absolute; top: 10px; left: 10px; width: 80px; height: 80px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let abs_id = find_by_id(doc.deref_mut(), "abs_root").expect("abs_root");
        let nested_id = find_by_id(doc.deref_mut(), "nested_abs").expect("nested_abs");
        let mut geom = PaginationGeometryTable::new();
        super::record_subtree_fragments_at_offset(
            &mut geom,
            doc.deref_mut(),
            abs_id,
            (0.0, 0.0),
            (0.0, 0.0),
            800.0,
            800.0,
            1,
            false,
            &mut 0,
            None,
        );
        assert!(
            geom.contains_key(&nested_id),
            "position:absolute child must be recursively walked and appear in geometry; \
             geom keys={:?}",
            geom.keys().collect::<Vec<_>>()
        );
    }

    /// The abs walk replaces a node's fragments; a line partition left from an
    /// earlier producer would index fragments that no longer exist, and with
    /// an equal fragment count it would pass the consumer's length check.
    #[test]
    fn record_subtree_walk_drops_the_line_partition_of_the_fragments_it_replaces() {
        let html = r#"
            <html><body style="margin: 0">
              <div id="abs_root" style="position: absolute; top: 0; left: 0; width: 200px; height: 2000px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let abs_id = find_by_id(doc.deref_mut(), "abs_root").expect("abs_root");
        let mut geom = PaginationGeometryTable::new();
        geom.entry(abs_id).or_default().line_boundaries = vec![0, 2, 4, 6];
        super::record_subtree_fragments_at_offset(
            &mut geom,
            doc.deref_mut(),
            abs_id,
            (0.0, 0.0),
            (0.0, 0.0),
            800.0,
            800.0,
            3,
            false,
            &mut 0,
            None,
        );
        let entry = &geom[&abs_id];
        assert_eq!(entry.fragments.len(), 3, "one fragment per page spanned");
        assert!(
            entry.line_boundaries.is_empty(),
            "stale partition survived: {:?}",
            entry.line_boundaries
        );
    }

    // ── forced_inside: a break between grandchildren ────────────────────────────

    /// `forced_inside` looks through every level: when a body-direct outer
    /// div contains a middle div (no direct break) whose first child has
    /// `break-after: page` before a sibling, the outer div is entered and
    /// the break fires.
    #[test]
    fn a_forced_break_between_grandchildren_makes_the_walk_enter_their_ancestor() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 300px">
                <div style="height: 150px">
                  <div style="height: 50px; break-after: page"></div>
                  <div style="height: 50px"></div>
                </div>
                <div style="height: 100px"></div>
              </div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        // The inner div has break-after: page; `forced_inside` must look
        // through the middle div and find it, causing the outer subtree to
        // be split and emitting a page-1 fragment.
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "forced_inside must find a break between grandchildren; geom={geom:?}"
        );
    }

    // ── break-after: page on a normal-height body-direct block (lines 1118-1124)

    /// A body-direct block with positive height and `break-after: page` must
    /// advance page_index and reset cursor_y even when the block does NOT
    /// overflow the page and does NOT need recursion. This exercises lines
    /// 1118-1124 in `fragment_pagination_root` (the normal-emit path), which
    /// is distinct from the zero-height path (lines 628-658) and the
    /// slice/recursion paths that both also carry a `break-after` arm.
    #[test]
    fn break_after_page_on_normal_height_body_direct_block_pushes_sibling() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="height: 200px; break-after: page"></div>
              <div style="height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        // The second div must land on page 1 because the first carries
        // break-after: page; both divs fit in the 800px strip so overflow
        // logic is not what triggers the break.
        assert!(
            geom.values()
                .flat_map(|g| g.fragments.iter())
                .any(|f| f.page_index == 1),
            "break-after: page on a normal-height body-direct block (normal path, \
             lines 1118-1124) must push the sibling to page 1; geom={geom:?}"
        );
    }

    // ── a float's page name in body-direct normal-height path ───────────────────

    /// A `float: left` body-direct child's used page name is not compared
    /// with its siblings' ([`opportunities`] skips floats). This exercises
    /// the normal-height float path.
    ///
    /// The float carries `page: float-page`. The following in-flow sibling
    /// has the default (unnamed) page, so comparing the float's name would
    /// force a break and push the sibling to page 1 — causing
    /// `implied_page_count` to become 2. Skipping it keeps both elements on
    /// page 0.
    #[test]
    fn body_direct_float_child_processed_without_prev_used_page_update() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="float: left; height: 200px; width: 100px; page: float-page"></div>
              <div style="clear: left; height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        // The is_float guard prevents the float's named page from being stored
        // in prev_used_page, so the following in-flow sibling must stay on page 0.
        assert_eq!(
            super::implied_page_count(&geom),
            1,
            "is_float guard must prevent float's named page from triggering a \
             forced break before the sibling; both must remain on page 0; geom={geom:?}"
        );
        assert!(
            geom.len() >= 2,
            "float and its sibling must both appear in geometry; geom={geom:?}"
        );
    }

    // ── is_float guard in body-direct slicing path (lines 1056-1058) ────────────

    /// A `float: left` body-direct child whose height exceeds the page height
    /// must be sliced across pages (the `child_h > page_height_px + 1.0` path)
    /// with `is_float = true`. This exercises the `if !is_float` guard at
    /// lines 1056-1058 inside the slicing branch of `fragment_pagination_root`.
    ///
    /// The float carries `page: float-page` so that, without the guard,
    /// `prev_used_page` would be set to `Some("float-page")` after slicing.
    /// The following `clear: left` sibling has the default page, so removing
    /// the guard would trigger `page_name_changed` and advance `page_index`
    /// to 2, landing the sibling on page 2 instead of page 1.
    #[test]
    fn body_direct_tall_float_sliced_across_pages_is_float_guard() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="float: left; height: 1200px; width: 100px; page: float-page"></div>
              <div style="clear: left; height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        // The 1200px float must be sliced into ≥ 2 fragments (slicing path exercised).
        let max_frags = geom.values().map(|g| g.fragments.len()).max().unwrap_or(0);
        assert!(
            max_frags >= 2,
            "a 1200px float on an 800px page must be sliced into ≥ 2 fragments \
             (exercises lines 1056-1058); max_frags={max_frags}, geom={geom:?}"
        );
        // The is_float guard must prevent the float's named page from being stored
        // in prev_used_page after slicing, so the clear sibling stays on page 1
        // (not forced to page 2 by a spurious page_name_changed).
        assert_eq!(
            super::implied_page_count(&geom),
            2,
            "is_float guard must keep the clear sibling on page 1, not page 2; \
             implied_page_count must be 2 (pages 0 and 1); geom={geom:?}"
        );
    }

    // ── break-inside: avoid (lines 718-720, 732-733) ─────────────────────────

    /// A body-direct element with `break-inside: avoid` must set `avoid_inside`
    /// and take the `Vec::new()` branch (lines 718-733), suppressing the inline
    /// split path so the element is treated as an atomic block.
    #[test]
    fn break_inside_avoid_suppresses_inline_split() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <p style="break-inside: avoid">first paragraph</p>
              <p style="break-inside: avoid">second paragraph</p>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        assert_eq!(
            super::implied_page_count(&geom),
            1,
            "two short avoid-inside paragraphs on an 800px page must land on page 0; \
             geom={geom:?}"
        );
    }

    /// css-break-3 §3.3: `break-inside: avoid` is a preference. A paragraph
    /// taller than a page cannot be kept whole, so it splits between its
    /// lines as an unkept one does, inside a container and directly under
    /// `body`, instead of being placed as one fragment that runs past the
    /// page foot.
    #[test]
    fn break_inside_avoid_on_a_paragraph_taller_than_a_page_splits_it() {
        let lines = (1..=12)
            .map(|n| format!("L{n}"))
            .collect::<Vec<_>>()
            .join("<br>");
        for wrap in [false, true] {
            let paragraph = format!(r#"<p id="p" style="break-inside: avoid">{lines}</p>"#);
            let body = if wrap {
                format!("<div>{paragraph}</div>")
            } else {
                paragraph
            };
            let html = format!(
                r#"<html><head><style>
                  html, body {{ margin: 0; padding: 0; }}
                  p {{ margin: 0; line-height: 30px; }}
                </style></head><body>{body}</body></html>"#
            );
            let mut doc = parse(&html, 200.0);
            let p_id = find_by_id(doc.deref_mut(), "p").expect("p#p");
            let table = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table);
            let fragments: Vec<(u32, f32)> = geom[&p_id]
                .fragments
                .iter()
                .map(|f| (f.page_index, f.height.to_f32()))
                .collect();
            assert_eq!(
                fragments,
                [(0, 180.0), (1, 180.0)],
                "six lines on each page; wrapped: {wrap}; geom={geom:?}"
            );
        }
    }

    /// The page of each fragment of the elements `ids` of `body`, laid out
    /// on 100 px pages after 65 px of content, with 20 px paragraph lines.
    fn pages_after_65px(body: &str, ids: &[&str]) -> Vec<Vec<u32>> {
        let html = format!(
            r#"<html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              p {{ margin: 0; line-height: 20px; }}
            </style></head><body><div style="height: 5px"></div><p>X</p><p>Y</p>{body}</body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let ids: Vec<usize> = ids
            .iter()
            .map(|id| find_by_id(doc.deref_mut(), id).expect(id))
            .collect();
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &table);
        ids.iter()
            .map(|id| {
                let mut pages: Vec<u32> = geom[id].fragments.iter().map(|f| f.page_index).collect();
                pages.dedup();
                pages
            })
            .collect()
    }

    /// css-break-3 §4.1 has no break point between a container's start and
    /// its first child. A box the walker places whole (kept whole, one line,
    /// a replaced element) that opens its container moves the container to
    /// the next page with it when it crosses the page foot after other
    /// content, at any depth, instead of staying past the foot. Padding above
    /// the box is no break point either: the padded container moves whole,
    /// leaving no sliver of its top edge on the first page.
    #[test]
    fn a_box_placed_whole_opening_its_container_moves_the_container_to_the_next_page() {
        let kept = r#"<p id="k" style="break-inside: avoid">K1<br>K2</p><p>N</p>"#;
        let line = r#"<p id="k" style="line-height: 40px">K2</p><p>N</p>"#;
        let image = r#"<img id="k" style="display: block; width: 10px; height: 40px"><p>N</p>"#;
        for (body, container_pages) in [
            (
                format!(r#"<p>Z</p><div id="c">{kept}</div>"#),
                [1].as_slice(),
            ),
            (
                format!(r#"<p>Z</p><section id="c"><div>{kept}</div></section>"#),
                &[1],
            ),
            (
                format!(r#"<div><p>Z</p><section id="c">{kept}</section></div>"#),
                &[1],
            ),
            (format!(r#"<p>Z</p><div id="c">{line}</div>"#), &[1]),
            (format!(r#"<p>Z</p><div id="c">{image}</div>"#), &[1]),
            (
                format!(r#"<p>Z</p><div id="c" style="padding-top: 1px">{kept}</div>"#),
                &[1],
            ),
        ] {
            assert_eq!(
                pages_after_65px(&body, &["k", "c"]),
                [&[1][..], container_pages],
                "{body}"
            );
        }
    }

    /// A box opened by a paragraph whose `orphans` cannot be met in the rest
    /// of the page moves to the next page whole, as one opened by a box
    /// placed whole does, at the body level and nested: the paragraph's
    /// leading unit is its first three lines, 62 px with the box's border,
    /// against the 35 px left. Moving the paragraph alone left the bordered
    /// box empty from its top to the page foot.
    #[test]
    fn a_box_opened_by_a_paragraph_its_orphans_move_moves_whole() {
        let para = r#"<p id="k" style="orphans: 3; widows: 1">K1<br>K2<br>K3<br>K4</p>"#;
        for body in [
            format!(r#"<p>Z</p><div id="c" style="border: 2px solid">{para}</div>"#),
            format!(
                r#"<div><p>Z</p><section id="c" style="border: 2px solid">{para}</section></div>"#
            ),
        ] {
            assert_eq!(pages_after_65px(&body, &["k", "c"]), [[1], [1]], "{body}");
        }
    }

    /// A box taller than a page in the block that is the only item of a
    /// flex column, or of a one-column grid, is sliced, and so are the item
    /// and the container around it: their fragments on each page the box
    /// crosses end at the page foot, and the text after the box follows the
    /// last slice (WPT css-page `monolithic-overflow-006-print`, `-008`).
    /// The container's first fragment took the item's whole height, so the
    /// container painted it all on the first page.
    #[test]
    fn a_wrapper_item_around_a_box_taller_than_a_page_ends_at_each_page_foot() {
        for container in [
            "display: flex; flex-flow: column",
            "display: grid; grid-template-columns: 1fr",
        ] {
            let html = format!(
                r#"<html><head><style>
                  html, body {{ margin: 0; padding: 0; }}
                </style></head><body>
                  <div id="c" style="{container}">
                    <div id="item"><div id="tall" style="contain: size; height: 250px; width: 50px"></div><span>after</span></div>
                  </div>
                </body></html>"#
            );
            let mut doc = parse(&html, 200.0);
            let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
            let geometry =
                super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
            let spans = |id: &str| -> Vec<(u32, f32, f32)> {
                let node_id = find_by_id(&doc, id).expect(id);
                geometry[&node_id]
                    .fragments
                    .iter()
                    .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                    .collect()
            };
            assert_eq!(
                spans("tall"),
                [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)],
                "{container}"
            );
            let item = spans("item");
            assert_eq!(
                item[..2],
                [(0, 0.0, 100.0), (1, 0.0, 100.0)],
                "{container}: {item:?}"
            );
            assert_eq!(item[2].0, 2, "{container}: {item:?}");
            let outer = spans("c");
            assert_eq!(
                outer[..2],
                [(0, 0.0, 100.0), (1, 0.0, 100.0)],
                "{container}: {outer:?}"
            );
        }
    }

    /// A paragraph split between pages measures its lines by their line
    /// boxes, each `line-height` tall, as its layout box holds them: six
    /// 16 px lines in 16 px Noto Sans, whose ascent and descent take more
    /// than 16 px, split three and three on 50 px pages, 48 px a page. The
    /// shaper's selection box clamps the negative half-leading to zero, so
    /// each line measured its ascent and descent, and the split paragraph
    /// ran past its box.
    #[test]
    fn a_split_paragraph_measures_its_lines_from_the_line_box() {
        let html = r#"<!doctype html><html><head><style>
              html, body { margin: 0; padding: 0; font-family: 'Noto Sans'; }
              p { margin: 0; font-size: 16px; line-height: 16px; orphans: 1; widows: 1; }
            </style></head><body><p id="p">1<br>2<br>3<br>4<br>5<br>6</p><div id="after" style="height: 2px"></div></body></html>"#;
        let fonts = [crate::test_fonts::noto_sans_regular_ttf()];
        let (mut doc, _, _) =
            blitz_adapter::parse_html_with_local_resources(html, 200.0, 10000, &fonts, false, None);
        blitz_adapter::resolve(&mut doc);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), 50.0_f32.as_px(), &styles);
        let p = find_by_id(&doc, "p").expect("p");
        let fragments: Vec<(u32, f32, f32)> = geometry[&p]
            .fragments
            .iter()
            .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
            .collect();
        assert_eq!(fragments, [(0, 0.0, 48.0), (1, 0.0, 48.0)]);
        assert_eq!(geometry[&p].line_boundaries, [0, 3, 6]);
        assert_eq!(placements(&doc, &geometry, "after"), [(1, 48.0)]);
    }

    /// `body` after a forced break on 100 px pages, the page the walk then
    /// starts empty, laid out with its break styles.
    fn after_a_forced_break(body: &str) -> (blitz_html::HtmlDocument, PaginationGeometryTable) {
        let html = format!(
            r#"<html><head><style>
              html, body {{ margin: 0; padding: 0; }}
              table {{ border-spacing: 0; }}
              td {{ padding: 0; }}
            </style></head><body><div style="height: 20px; break-after: page"></div>{body}</body></html>"#
        );
        let mut doc = parse(&html, 200.0);
        let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geometry =
            super::run_pass_with_break_styles(doc.deref_mut(), 100.0_f32.as_px(), &styles);
        (doc, geometry)
    }

    /// At the top of a page, below a padded top edge, a box whose leading
    /// unit does not fit the room left breaks where it stands (css-break-3
    /// §4.4 relaxes `avoid` when breaking elsewhere gains nothing): a 95 px
    /// block kept whole below a 10 px top padding is sliced from y 10 and
    /// continues on the next page, and so does one entered for its
    /// children, whose kept first child is sliced there too; one taller
    /// than a page also starts there. Each moved to the next page and left
    /// the padding alone on its page.
    #[test]
    fn a_box_below_a_padded_top_edge_at_the_page_top_breaks_where_it_stands() {
        let (doc, geometry) = after_a_forced_break(
            r#"<div style="padding-top: 10px"><div id="k" style="break-inside: avoid; height: 95px"></div></div>"#,
        );
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 10.0), (2, 0.0)]);

        let (doc, geometry) = after_a_forced_break(
            r#"<div style="padding-top: 10px"><div id="k"><div style="break-inside: avoid; height: 95px"></div><div id="n" style="height: 20px"></div></div></div>"#,
        );
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 10.0), (2, 0.0)]);
        assert_eq!(placements(&doc, &geometry, "n"), [(2, 5.0)]);

        let (doc, geometry) = after_a_forced_break(
            r#"<div style="padding-top: 10px"><div id="k" style="height: 120px"></div></div>"#,
        );
        assert_eq!(placements(&doc, &geometry, "k")[0], (1, 10.0));
    }

    /// A box at the top of a page that overruns the room left by less than
    /// the walk's 1 px tolerance for a box that fits (a length rounded to
    /// whole pixels, a 220pt block on a 220pt page) is placed whole, not
    /// sliced into a sliver on a page of its own.
    #[test]
    fn a_box_at_the_page_top_within_the_fit_tolerance_is_not_sliced() {
        let (doc, geometry) = after_a_forced_break(r#"<div id="k" style="height: 100.6px"></div>"#);
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 0.0)]);
    }

    /// A table row at the top of a page, below its table's top border,
    /// whose cell's first block does not fit the room left breaks where it
    /// stands (css-break-3 §4.4): the row (the row move) and a row kept
    /// whole start below the border and continue on the next page, the
    /// cell's 95 px block sliced at the foot; they moved to the next page
    /// and left the border alone. A cell's block taller than a page is
    /// sliced in the cell, where it ran past the foot.
    #[test]
    fn a_first_row_below_its_tables_top_edge_at_the_page_top_breaks_where_it_stands() {
        for row in ["", " style=\"break-inside: avoid\""] {
            let (doc, geometry) = after_a_forced_break(&format!(
                r#"<table style="border-top: 10px solid"><tr{row}><td id="c"><div id="k" style="height: 95px"></div></td></tr></table>"#
            ));
            assert_eq!(
                placements(&doc, &geometry, "c"),
                [(1, 10.0), (2, 0.0)],
                "{row}"
            );
            assert_eq!(
                placements(&doc, &geometry, "k"),
                [(1, 10.0), (2, 0.0)],
                "{row}"
            );
        }
        let (doc, geometry) = after_a_forced_break(
            r#"<table style="border-top: 10px solid"><tr><td id="c"><div id="k" style="height: 120px"></div></td></tr></table>"#,
        );
        assert_eq!(placements(&doc, &geometry, "c")[0], (1, 10.0));
        assert_eq!(placements(&doc, &geometry, "k"), [(1, 10.0), (2, 0.0)]);
    }

    /// A gap between a box's content edge and its first child's margin edge
    /// is a break opportunity (class C), so a box whose top edge fits the
    /// rest of the page is entered there; one whose top edge alone does
    /// not, a 40 px border against the 35 px left, moves to the next page
    /// whole instead of leaving its edge past the foot. The gap is the
    /// margin of an empty block before the paragraph.
    #[test]
    fn a_box_whose_top_edge_before_a_gap_does_not_fit_moves_whole() {
        let body = |border: u8| {
            format!(
                r#"<p>Z</p><div id="c" style="border-top: {border}px solid"><div style="margin-bottom: 30px"></div><p id="k">K</p></div>"#
            )
        };
        assert_eq!(pages_after_65px(&body(40), &["k", "c"]), [[1], [1]]);
        assert_eq!(
            pages_after_65px(&body(10), &["k", "c"]),
            [vec![1], vec![0, 1]]
        );
    }

    /// A flex or grid row none of whose items places content in the rest of
    /// the page moves to the next page with its items, and with the box it
    /// opens, at the body level and nested: two items kept whole, 40 px
    /// each, against the 35 px left.
    #[test]
    fn a_flex_or_grid_row_that_places_nothing_moves_with_its_container() {
        let kept = r#"<p style="break-inside: avoid">K1<br>K2</p>"#;
        let row = |display: &str| {
            format!(
                r#"<div id="row" style="display: {display}; grid-template-columns: 1fr 1fr"><div id="a">{kept}</div><div id="b">{kept}</div></div>"#
            )
        };
        for display in ["flex", "grid"] {
            let row = row(display);
            for body in [
                format!(r#"<p>Z</p>{row}"#),
                format!(r#"<p>Z</p><div id="c">{row}</div>"#),
                format!(
                    r#"<div><p>Z</p><section id="c" style="padding-top: 1px">{row}</section></div>"#
                ),
            ] {
                let ids: &[&str] = if body.contains(r#"id="c""#) {
                    &["a", "b", "row", "c"]
                } else {
                    &["a", "b", "row"]
                };
                let pages = pages_after_65px(&body, ids);
                assert!(pages.iter().all(|pages| pages == &[1]), "{body}: {pages:?}");
            }
        }
    }

    /// A flex or grid item opened by a box kept whole that crosses the page
    /// foot moves to the next page with the items beside it, as a block
    /// container does, instead of staying past the foot while they move.
    #[test]
    fn a_flex_or_grid_item_opened_by_a_kept_box_moves_with_its_row() {
        let item = r#"<div id="item"><p style="break-inside: avoid">K1<br>K2</p></div><div id="r">R</div>"#;
        for body in [
            format!(
                r#"<div style="display: flex; flex-wrap: wrap"><div style="width: 100%">Z</div>{item}</div>"#
            ),
            format!(
                r#"<div style="display: grid; grid-template-columns: 1fr 1fr"><div>Z</div><div>Z</div>{item}</div>"#
            ),
        ] {
            assert_eq!(
                pages_after_65px(&body, &["item", "r"]),
                [[1], [1]],
                "{body}"
            );
        }
    }

    /// css-break-3 §3.3: `break-inside: avoid` on a table row keeps the row
    /// whole. The row does not fit the rest of the page, so it moves to the
    /// next page with every cell, instead of splitting its tall cell between
    /// blocks and leaving the short cell behind.
    #[test]
    fn break_inside_avoid_on_a_table_row_moves_the_row_whole() {
        let html = r#"
            <html><head><style>
              html, body { margin: 0; padding: 0; }
              table { border-spacing: 0; }
              td { padding: 0; }
              tr { break-inside: avoid; }
            </style></head><body>
              <div style="height: 120px"></div>
              <table><tr>
                <td id="short">x</td>
                <td><div id="a" style="height: 30px"></div><div style="height: 30px"></div><div style="height: 30px"></div></td>
              </tr></table>
            </body></html>
        "#;
        let mut doc = parse(html, 200.0);
        let short_id = find_by_id(doc.deref_mut(), "short").expect("td#short");
        let a_id = find_by_id(doc.deref_mut(), "a").expect("div#a");
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table);
        for id in [short_id, a_id] {
            let pages: Vec<u32> = geom[&id].fragments.iter().map(|f| f.page_index).collect();
            assert_eq!(pages, [1], "the whole row moves to page 1; geom={geom:?}");
        }
    }

    /// A cell spanning rows reaches the rows its `rowspan` counts, within
    /// its row group; a cell of a row reached so spans on in turn, and a
    /// row of cells one row tall reaches none. Under
    /// `-fulgur-span-break-inside: avoid-unsplittable` the rows so chained
    /// are kept together unless one of them holds a cell of two lines
    /// (`r3`'s `f`, which frees the chain of `r3` to `r5`); by default none
    /// are.
    #[test]
    fn a_rows_span_reaches_the_rows_its_cells_span_within_its_group() {
        let spanned = |style: &str| {
            let html = format!(
                r#"<html><body><table style="{style}"><tbody>
                <tr id="r0"><td rowspan="2">a</td><td>b</td></tr><tr id="r1"><td>c</td></tr>
                <tr id="r2"><td>d</td></tr>
                <tr id="r3"><td rowspan="2">e</td><td>f<br>f</td></tr>
                <tr id="r4"><td rowspan="2">g</td></tr><tr id="r5"><td>h</td></tr>
                <tr id="r6"><td rowspan="9">i</td></tr><tr id="r7"><td>j</td></tr>
                </tbody></table></body></html>"#
            );
            let doc = parse(&html, 400.0);
            let styles = blitz_adapter::extract_column_style_table(&doc, &[]);
            let pass = super::Pass::default();
            let flow = super::Flow {
                doc: &doc,
                column_styles: Some(&styles),
                used_page_names: None,
                running_store: None,
                pass: &pass,
                forced_before: None,
                foot_clearance: 0.0,
            };
            let row = doc.get_node(find_by_id(&doc, "r0").unwrap()).unwrap();
            let (rows, _) = super::group_rows(&doc, row).unwrap();
            let ends: Vec<usize> = (0..rows.len())
                .map(|from| super::span_end(&doc, &rows, from))
                .collect();
            assert_eq!(ends, [1, 1, 2, 5, 5, 5, 7, 7]);
            super::RowGroup::new(flow, &rows).spanned
        };
        assert_eq!(
            [spanned(KEEP_UNSPLITTABLE), spanned("")],
            [
                [false, true, false, false, false, false, false, true],
                [false; 8],
            ]
        );
    }

    // ── break-after: page after recursive split (lines 876-882) ──────────────

    /// A body-direct element with `break-after: page` that triggers the
    /// recursion path (children overflow the page) must advance the page cursor
    /// *after* the subtree is processed, so the following sibling lands two
    /// pages after where the recursion started (lines 876-882).
    #[test]
    fn break_after_page_advances_after_recursive_split() {
        // 600px page; outer div (break-after: page) has two 400px children —
        // total 800px overflows the page, so `needs_recursion = true`.
        // fragment_block_subtree places inner1 on page 0, inner2 on page 1,
        // leaving page=1, cursor=400.
        // break-after fires: page=2, cursor=0.
        // Trailing 100px sibling must land on page 2.
        //
        // Discriminant vs the oversized-slice path (which would also produce
        // page=2 for #after): in the recursion path `fragment_block_subtree`
        // records inner2 on page 1; in the slice path `record_subtree_descendants`
        // records both inner children on page 0 (the first slice's page).
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="break-after: page">
                <div id="inner1" style="height: 400px"></div>
                <div id="inner2" style="height: 400px"></div>
              </div>
              <div id="after" style="height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let inner2_id = find_by_id(doc.deref_mut(), "inner2").expect("div#inner2");
        let after_id = find_by_id(doc.deref_mut(), "after").expect("div#after");
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 600.0_f32.as_px(), &table);
        // Recursion-specific check: fragment_block_subtree records inner2 on
        // page 1. The slice path would record it on page 0 instead.
        let inner2_geom = geom
            .get(&inner2_id)
            .expect("div#inner2 must be in geometry");
        assert_eq!(
            inner2_geom.fragments[0].page_index, 1,
            "recursion path must place inner2 on page 1; slice path would give page 0; \
             geom={geom:?}"
        );
        let after_geom = geom.get(&after_id).expect("div#after must be in geometry");
        assert_eq!(after_geom.fragments.len(), 1);
        assert_eq!(
            after_geom.fragments[0].page_index, 2,
            "break-after: page after recursive split must push sibling to page 2; \
             geom={geom:?}"
        );
    }

    // ── break-after: page after oversized slice (lines 1059-1065) ────────────

    /// A body-direct element taller than the page (slice path) with
    /// `break-after: page` must advance the page cursor after the last slice
    /// so the following sibling lands on a fresh page (lines 1059-1065).
    #[test]
    fn break_after_page_advances_after_oversized_slice() {
        // 800px page; childless 900px div is sliced: first slice on page 0
        // (800px), second slice on page 1 (100px), cursor=100.
        // break-after fires: page=2, cursor=0.
        // Trailing 100px sibling must land on page 2.
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div style="break-after: page; height: 900px"></div>
              <div id="after" style="height: 100px"></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let after_id = find_by_id(doc.deref_mut(), "after").expect("div#after");
        let table = blitz_adapter::extract_column_style_table(&doc, &[]);
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 800.0_f32.as_px(), &table);
        let after_geom = geom.get(&after_id).expect("div#after must be in geometry");
        assert_eq!(after_geom.fragments.len(), 1);
        assert_eq!(
            after_geom.fragments[0].page_index, 2,
            "break-after: page after slice must push sibling to page 2; geom={geom:?}"
        );
    }

    /// A box taller than a page with no children to enter is sliced per
    /// page at every depth, as at the body level, instead of being placed as
    /// one fragment past the page foot (WPT `monolithic-overflow-026-print`);
    /// the block around it has a fragment on each page it spans.
    #[test]
    fn a_nested_childless_box_taller_than_a_page_is_sliced() {
        let html = r#"
            <html><body style="margin: 0; padding: 0">
              <div id="outer"><div id="tall" style="height: 250px"></div></div>
            </body></html>
        "#;
        let mut doc = parse(html, 600.0);
        let geom = run_pass(doc.deref_mut(), 100.0);
        let fragments = |geom: &PaginationGeometryTable, id: &str| {
            geom[&find_by_id(&doc, id).unwrap()]
                .fragments
                .iter()
                .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            fragments(&geom, "tall"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)]
        );
        assert_eq!(
            fragments(&geom, "outer"),
            [(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 0.0, 50.0)]
        );
    }

    // ── zero page height guard (line 412) ─────────────────────────────────────

    /// `fragment_pagination_root` must return immediately and leave the
    /// geometry table empty when `page_height_px` is zero.
    /// `run_pass_with_break_styles` short-circuits before calling
    /// `fragment_pagination_root`, so this test exercises the guard directly.
    #[test]
    fn zero_page_height_guard_returns_empty() {
        let mut doc = parse(
            "<html><body><div style=\"height: 100px\"></div></body></html>",
            600.0,
        );
        let mut tree = PaginationLayoutTree::new(doc.deref_mut(), 0.0);
        tree.fragment_pagination_root();
        assert!(
            tree.take_geometry().is_empty(),
            "zero page height must leave geometry table empty"
        );
    }

    // ── PaginationGeometry::is_split ──────────────────────────────────────────

    #[test]
    fn is_split_empty_fragments_returns_false() {
        let geom = PaginationGeometry::default();
        assert!(!geom.is_split());
    }

    #[test]
    fn is_split_single_fragment_not_repeat_returns_false() {
        let mut geom = PaginationGeometry::default();
        geom.fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        assert!(!geom.is_split());
    }

    #[test]
    fn is_split_two_fragments_not_repeat_returns_true() {
        let mut geom = PaginationGeometry::default();
        for page in [0_u32, 1] {
            geom.fragments.push(Fragment {
                page_index: page,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 100.0_f32.as_px(),
                height: 50.0_f32.as_px(),
            });
        }
        assert!(geom.is_split());
    }

    #[test]
    fn is_split_two_fragments_is_repeat_returns_false() {
        let mut geom = PaginationGeometry {
            is_repeat: true,
            ..Default::default()
        };
        for page in [0_u32, 1] {
            geom.fragments.push(Fragment {
                page_index: page,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 100.0_f32.as_px(),
                height: 50.0_f32.as_px(),
            });
        }
        assert!(
            !geom.is_split(),
            "is_repeat=true prevents split classification"
        );
    }

    // ── collect_counter_states ───────────────────────────────────────────────

    #[test]
    fn counter_states_empty_geometry_returns_one_empty_page() {
        let geom = PaginationGeometryTable::new();
        let ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], BTreeMap::from([("page".to_string(), 1)]));
    }

    #[test]
    fn counter_states_reset_op_sets_counter_value() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let mut ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        ops.insert(
            10,
            vec![crate::gcpm::CounterOp::Reset {
                name: "chapter".into(),
                value: 3,
            }],
        );
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("chapter").copied(), Some(3));
    }

    #[test]
    fn counter_states_increment_op_adds_to_counter() {
        let mut geom = PaginationGeometryTable::new();
        for (node_id, page) in [(10_usize, 0_u32), (20, 0)] {
            geom.entry(node_id).or_default().fragments.push(Fragment {
                page_index: page,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 100.0_f32.as_px(),
                height: 50.0_f32.as_px(),
            });
        }
        let mut ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        ops.insert(
            10,
            vec![crate::gcpm::CounterOp::Reset {
                name: "chapter".into(),
                value: 0,
            }],
        );
        ops.insert(
            20,
            vec![crate::gcpm::CounterOp::Increment {
                name: "chapter".into(),
                value: 1,
            }],
        );
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("chapter").copied(), Some(1));
    }

    #[test]
    fn counter_states_set_op_forces_counter_value() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let mut ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        ops.insert(
            10,
            vec![
                crate::gcpm::CounterOp::Reset {
                    name: "idx".into(),
                    value: 99,
                },
                crate::gcpm::CounterOp::Set {
                    name: "idx".into(),
                    value: 7,
                },
            ],
        );
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].get("idx").copied(), Some(7));
    }

    #[test]
    fn counter_states_carries_across_pages() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        geom.entry(20).or_default().fragments.push(Fragment {
            page_index: 1,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let mut ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        ops.insert(
            10,
            vec![crate::gcpm::CounterOp::Reset {
                name: "chapter".into(),
                value: 1,
            }],
        );
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].get("chapter").copied(), Some(1));
        assert_eq!(states[1].get("chapter").copied(), Some(1));
    }

    #[test]
    fn counter_states_node_without_ops_is_skipped() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(99).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let ops: BTreeMap<usize, Vec<crate::gcpm::CounterOp>> = BTreeMap::new();
        let states = collect_counter_states(&geom, &ops, &[]);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], BTreeMap::from([("page".to_string(), 1)]));
    }

    /// One geometry entry per `(node_id, page_index)`.
    fn counter_geometry(nodes: &[(usize, u32)]) -> PaginationGeometryTable {
        let mut geom = PaginationGeometryTable::new();
        for &(node_id, page_index) in nodes {
            geom.entry(node_id).or_default().fragments.push(Fragment {
                page_index,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 100.0_f32.as_px(),
                height: 50.0_f32.as_px(),
            });
        }
        geom
    }

    #[test]
    fn counter_states_page_counts_the_pages() {
        let geom = counter_geometry(&[(10, 0), (20, 1), (30, 2)]);
        let states = collect_counter_states(&geom, &BTreeMap::new(), &[]);
        let pages: Vec<_> = states.iter().map(|s| s.get("page").copied()).collect();
        assert_eq!(pages, [Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn counter_states_element_page_reset_numbers_its_page_and_the_pages_after() {
        use crate::gcpm::CounterOp;
        let geom = counter_geometry(&[(10, 0), (20, 1), (30, 2)]);
        for op in [
            CounterOp::Reset {
                name: "page".into(),
                value: 5,
            },
            CounterOp::Set {
                name: "page".into(),
                value: 5,
            },
        ] {
            let ops = BTreeMap::from([(20, vec![op.clone()])]);
            let states = collect_counter_states(&geom, &ops, &[]);
            let pages: Vec<_> = states.iter().map(|s| s.get("page").copied()).collect();
            assert_eq!(pages, [Some(1), Some(5), Some(6)], "{op:?}");
        }
    }

    #[test]
    fn counter_states_page_ops_count_before_the_elements_of_the_page() {
        use crate::gcpm::CounterOp;
        let geom = counter_geometry(&[(10, 0), (20, 1), (30, 2)]);
        let page_ops = [CounterOp::Increment {
            name: "run".into(),
            value: 1,
        }];
        let ops = BTreeMap::from([(
            20,
            vec![CounterOp::Reset {
                name: "run".into(),
                value: 1,
            }],
        )]);
        let states = collect_counter_states(&geom, &ops, &page_ops);
        let runs: Vec<_> = states.iter().map(|s| s.get("run").copied()).collect();
        assert_eq!(runs, [Some(1), Some(1), Some(2)]);
    }

    #[test]
    fn counter_states_page_increment_of_page_replaces_the_implicit_one() {
        use crate::gcpm::CounterOp;
        let geom = counter_geometry(&[(10, 0), (20, 1)]);
        let page_ops = [CounterOp::Increment {
            name: "page".into(),
            value: 2,
        }];
        let states = collect_counter_states(&geom, &BTreeMap::new(), &page_ops);
        let pages: Vec<_> = states.iter().map(|s| s.get("page").copied()).collect();
        assert_eq!(pages, [Some(2), Some(4)]);
    }

    // ── fragment_inline_root: additional branch coverage ────────────────────

    /// Empty `line_metrics` must trigger the early-return guard at the top
    /// of `fragment_inline_root` and return the input coordinates unchanged
    /// with zero emitted fragments.
    #[test]
    fn fragment_inline_root_empty_line_metrics_returns_early() {
        let mut geom = PaginationGeometryTable::new();
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            42.0,
            3,
            800.0,
            &[],
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 0, "empty metrics → no fragments emitted");
        assert_eq!(new_page, 3, "page unchanged");
        assert!(
            (new_cursor - 42.0).abs() < 0.01,
            "cursor unchanged: {new_cursor}"
        );
        assert!(
            geom.is_empty(),
            "empty metrics must not create a geometry entry"
        );
    }

    /// A paragraph starting partway down a page (non-zero `initial_cursor_y`)
    /// must account for the existing offset when deciding where the page
    /// boundary falls.  This tests that `projected_bottom_in_body` is
    /// computed as `initial_cursor_y + (line_bottom_local - frag_top_local)`.
    #[test]
    fn fragment_inline_root_nonzero_initial_cursor_y_splits_correctly() {
        // Paragraph starts at y=30 on a 80px page.
        // 4 lines of 25px each → line bottoms at 25, 50, 75, 100 (para-local).
        // projected bottoms = 30 + cumulative = 55, 80, 105, 130.
        // At i=2 overflow: lines 0-1 fit (80 <= 80), remaining_size=2 → split.
        let lines = vec![(0.0_f32, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0)];
        let mut geom = PaginationGeometryTable::new();
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            5.0,
            100.0,
            /*initial_cursor_y=*/ 30.0,
            /*initial_page_index=*/ 0,
            /*page_height_px=*/ 80.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 2, "expected split into 2 fragments");
        assert_eq!(new_page, 1, "second fragment lands on page 1");
        let frags = &geom.get(&1).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].page_index, 0);
        // First fragment: lines 0-1, height = line_metrics[1].1 - line_metrics[0].0 = 50.
        assert!(
            (frags[0].height.to_f32() - 50.0).abs() < 0.01,
            "first fragment height: {}",
            frags[0].height.to_f32()
        );
        // First fragment starts at cursor y=30 (initial_cursor_y).
        assert!(
            (frags[0].y.to_f32() - 30.0).abs() < 0.01,
            "first fragment y: {}",
            frags[0].y.to_f32()
        );
        assert_eq!(frags[1].page_index, 1);
        // After the page break cursor resets to 0 for the second fragment.
        assert!(
            (frags[1].y.to_f32()).abs() < 0.01,
            "second fragment y (after reset): {}",
            frags[1].y.to_f32()
        );
        // cursor_y on page 1 = 0 + (line_metrics[3].1 - line_metrics[2].0) = 50.
        assert!(
            (new_cursor - 50.0).abs() < 0.01,
            "cursor_y on new page: {new_cursor}"
        );
    }

    /// A paragraph that spans three pages must emit three fragments.
    /// Exercises the loop body more than once (each earlier emission
    /// resets `paragraph_top_in_body` and `fragment_start_idx`).
    #[test]
    fn fragment_inline_root_three_page_span_emits_three_fragments() {
        // 6 lines of 50px each → line tops 0, 50, 100, 150, 200, 250;
        // bottoms 50, 100, 150, 200, 250, 300.
        // page_height_px=100.
        // Split 1 at i=2: first_size=2 ≥ 2, remaining_size=4 ≥ 2 → OK.
        // Split 2 at i=4: first_size=2 ≥ 2, remaining_size=2 ≥ 2 → OK.
        // Final fragment: lines 4-5.
        let lines: Vec<(f32, f32)> = (0..6)
            .map(|i| (i as f32 * 50.0, i as f32 * 50.0 + 50.0))
            .collect();
        let mut geom = PaginationGeometryTable::new();
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            7,
            0.0,
            200.0,
            0.0,
            0,
            100.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(
            emitted, 3,
            "six lines at 50px on a 100px page → 3 fragments"
        );
        assert_eq!(new_page, 2, "last fragment lands on page 2");
        assert!((new_cursor - 100.0).abs() < 0.01, "cursor: {new_cursor}");
        let frags = &geom.get(&7).unwrap().fragments;
        assert_eq!(frags.len(), 3);
        assert_eq!(frags[0].page_index, 0);
        assert_eq!(frags[1].page_index, 1);
        assert_eq!(frags[2].page_index, 2);
        for (idx, frag) in frags.iter().enumerate() {
            let h = frag.height.to_f32();
            assert!(
                (h - 100.0).abs() < 0.01,
                "fragment {idx} height should be 100px, got {h}"
            );
        }
    }

    /// codex P1 review on PR #741 (second round, commit 84654dc): unlike
    /// the oversized-block re-slice loop and the absolute-positioning
    /// path (both bounded by `crate::MAX_PAGES`), `fragment_inline_root`
    /// had no cap on how far it advances `page_index` — bounded only by
    /// the paragraph's own line count. A pathologically long
    /// `word-break: break-all` paragraph (attacker-controlled HTML) can
    /// wrap into far more lines than `MAX_PAGES`, and the resulting
    /// `page_index` feeds `implied_page_count`, which drives per-page
    /// allocation and rendering for every implied page — the exact
    /// "small input amplifies into huge output" shape the other
    /// `MAX_PAGES` guards in this file exist to prevent.
    ///
    /// 50,000 lines of 10px each on a 25px page strip split every 2
    /// lines (satisfying `ORPHANS_MIN`/`WIDOWS_MIN`), which would
    /// advance `page_index` by roughly 25,000 uncapped — far past
    /// `MAX_PAGES` (10,000).
    #[test]
    fn fragment_inline_root_caps_page_advancement_at_max_pages() {
        let lines: Vec<(f32, f32)> = (0..50_000)
            .map(|i| (i as f32 * 10.0, i as f32 * 10.0 + 10.0))
            .collect();
        let mut geom = PaginationGeometryTable::new();
        let (new_page, _new_cursor, _emitted) = split_inline_root(
            &mut geom,
            7,
            0.0,
            200.0,
            0.0,
            0,
            25.0,
            &lines,
            super::LineLimits::default(),
        );
        assert!(
            new_page <= crate::MAX_PAGES,
            "fragment_inline_root must clamp page advancement to MAX_PAGES \
             ({}); got {new_page} — uncapped this would be ~25,000",
            crate::MAX_PAGES,
        );
        let max_frag_page = geom
            .get(&7)
            .unwrap()
            .fragments
            .iter()
            .map(|f| f.page_index)
            .max()
            .unwrap();
        assert!(
            max_frag_page <= crate::MAX_PAGES,
            "no emitted fragment should exceed MAX_PAGES ({}); got {max_frag_page}",
            crate::MAX_PAGES,
        );
    }

    /// A paragraph starting on a non-zero `initial_page_index` must record
    /// its first fragment on that page and advance from it correctly.
    #[test]
    fn fragment_inline_root_nonzero_initial_page_index() {
        // 4 lines of 40px; page_height_px=100; initial_page_index=2.
        // At i=2: projected=120>100. Lines 0-1 fit, remaining_size=2 → emit on page 2.
        // Final: lines 2-3 (80px) on page 3.
        let lines: Vec<(f32, f32)> = (0..4)
            .map(|i| (i as f32 * 40.0, i as f32 * 40.0 + 40.0))
            .collect();
        let mut geom = PaginationGeometryTable::new();
        let (new_page, _new_cursor, emitted) = split_inline_root(
            &mut geom,
            99,
            0.0,
            100.0,
            0.0,
            /*initial_page_index=*/ 2,
            100.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 2, "4 lines → 2 fragments");
        let frags = &geom.get(&99).unwrap().fragments;
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].page_index, 2, "first fragment on initial page");
        assert_eq!(frags[1].page_index, 3, "second fragment on next page");
        assert_eq!(new_page, 3);
    }

    /// css-break-3 §4.4: a paragraph that starts mid-page and has room for
    /// only one line there would leave a one-line head (below orphans = 2),
    /// so the whole paragraph moves to the next page instead of being split
    /// into a gap it overflows.
    #[test]
    fn paragraph_with_one_fitting_line_mid_page_moves_whole_to_next_page() {
        // Starts at y=50 on an 80px page; 25px lines end at 75 (fits), 100.
        let lines = vec![(0.0_f32, 25.0), (25.0, 50.0), (50.0, 75.0)];
        let mut geom = PaginationGeometryTable::new();
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            50.0,
            0,
            80.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 1, "the paragraph is not split");
        assert_eq!(new_page, 1, "the paragraph moves to the next page");
        let entry = geom.get(&1).unwrap();
        assert_eq!(entry.fragments.len(), 1);
        assert_eq!(entry.fragments[0].page_index, 1);
        assert!(entry.fragments[0].y.to_f32().abs() < 0.01);
        assert!((entry.fragments[0].height.to_f32() - 75.0).abs() < 0.01);
        assert!((new_cursor - 75.0).abs() < 0.01, "cursor: {new_cursor}");
        assert_eq!(entry.line_boundaries, vec![0, 3]);
    }

    /// css-break-3 §4.4: when the last line that fits would leave a tail
    /// shorter than widows = 2, the split is pulled back so the tail keeps
    /// two lines, rather than splitting after the last line that fits.
    #[test]
    fn widow_minimum_pulls_the_split_back_to_keep_two_tail_lines() {
        // Five 50px lines on a 200px page: four fit, which would leave a
        // one-line tail. The split moves to after line 3.
        let lines: Vec<(f32, f32)> = (0..5)
            .map(|i| (i as f32 * 50.0, i as f32 * 50.0 + 50.0))
            .collect();
        let mut geom = PaginationGeometryTable::new();
        let (new_page, new_cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            0.0,
            0,
            200.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 2);
        assert_eq!(new_page, 1);
        let entry = geom.get(&1).unwrap();
        assert!((entry.fragments[0].height.to_f32() - 150.0).abs() < 0.01);
        assert!((entry.fragments[1].height.to_f32() - 100.0).abs() < 0.01);
        assert_eq!(entry.line_boundaries, vec![0, 3, 5]);
        assert!((new_cursor - 100.0).abs() < 0.01, "cursor: {new_cursor}");
    }

    /// css-break-3 §4.4: at the top of a page there is nothing to gain by
    /// moving the paragraph, so orphans and widows are relaxed and the split
    /// follows the lines that fit.
    #[test]
    fn orphan_and_widow_minimums_are_relaxed_at_the_top_of_a_page() {
        // 75px lines on a 100px page: one line fits per page.
        let lines = vec![(0.0, 75.0), (75.0, 150.0), (150.0, 225.0)];
        let mut geom = PaginationGeometryTable::new();
        let (new_page, _cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            0.0,
            0,
            100.0,
            &lines,
            super::LineLimits::default(),
        );
        assert_eq!(emitted, 3, "one line per page");
        assert_eq!(new_page, 2);
        let entry = geom.get(&1).unwrap();
        let pages: Vec<u32> = entry.fragments.iter().map(|f| f.page_index).collect();
        assert_eq!(pages, vec![0, 1, 2]);
        assert_eq!(entry.line_boundaries, vec![0, 1, 2, 3]);
    }

    /// A single line taller than an empty page still makes progress: it is
    /// emitted on a page of its own rather than looping or moving forever.
    #[test]
    fn line_taller_than_an_empty_page_is_emitted_alone() {
        let lines = vec![(0.0, 150.0), (150.0, 175.0), (175.0, 200.0)];
        let mut geom = PaginationGeometryTable::new();
        let (_page, _cursor, emitted) = split_inline_root(
            &mut geom,
            1,
            0.0,
            100.0,
            0.0,
            0,
            100.0,
            &lines,
            super::LineLimits::default(),
        );
        let entry = geom.get(&1).unwrap();
        assert_eq!(emitted, entry.fragments.len());
        assert!((entry.fragments[0].height.to_f32() - 150.0).abs() < 0.01);
        assert_eq!(entry.line_boundaries.first(), Some(&0));
        assert_eq!(entry.line_boundaries.last(), Some(&3));
        assert_eq!(entry.fragments[0].page_index, 0);
        assert_eq!(entry.fragments[1].page_index, 1);
    }

    /// Wherever the paragraph starts, every emitted fragment ends inside the
    /// page strip (a fragment of a single line taller than the page aside),
    /// and the published partition covers every line exactly once.
    #[test]
    fn no_fragment_extends_past_the_page_strip() {
        const PAGE_H: f32 = 100.0;
        for line_h in [28.0_f32, 30.0, 33.0] {
            for lines_n in 2..=12 {
                let lines: Vec<(f32, f32)> = (0..lines_n)
                    .map(|i| (i as f32 * line_h, (i + 1) as f32 * line_h))
                    .collect();
                for start in 0..PAGE_H as u32 {
                    let mut geom = PaginationGeometryTable::new();
                    split_inline_root(
                        &mut geom,
                        1,
                        0.0,
                        100.0,
                        start as f32,
                        0,
                        PAGE_H,
                        &lines,
                        super::LineLimits::default(),
                    );
                    let entry = geom.get(&1).unwrap();
                    let ctx = format!("line_h={line_h} lines={lines_n} start={start}");
                    for frag in &entry.fragments {
                        assert!(
                            frag.y.to_f32() + frag.height.to_f32() <= PAGE_H + 0.01,
                            "{ctx}: fragment {frag:?} runs past the strip"
                        );
                    }
                    assert_eq!(
                        entry.line_boundaries.len(),
                        entry.fragments.len() + 1,
                        "{ctx}"
                    );
                    assert_eq!(entry.line_boundaries.first(), Some(&0), "{ctx}");
                    assert_eq!(entry.line_boundaries.last(), Some(&lines_n), "{ctx}");
                    assert!(
                        entry.line_boundaries.windows(2).all(|w| w[0] < w[1]),
                        "{ctx}"
                    );
                }
            }
        }
    }

    /// fulgur-s67g Phase 2.2: a paragraph crossing the page foot below the
    /// top of a page places nothing there, and moves whole, only when the
    /// lines that fit, less the `widows` kept for the next page, are fewer
    /// than `orphans`: its leading unit is its first `orphans` lines, or all
    /// of them when no break keeps `widows`. Six 20px lines with 110 px
    /// left: five fit.
    #[test]
    fn an_inline_root_leads_with_the_lines_its_orphans_and_widows_keep() {
        let lines: Vec<(f32, f32)> = (0..6)
            .map(|i| (i as f32 * 20.0, (i + 1) as f32 * 20.0))
            .collect();
        let lead = |orphans, widows, room| {
            super::inline_leading(&lines, super::LineLimits { orphans, widows }, 0.0, room)
        };
        assert!(lead(1, 1, 110.0).places, "5 + 1");
        assert!(lead(2, 2, 110.0).places, "4 + 2");
        assert!(lead(3, 3, 110.0).places, "3 + 3");
        assert!(!lead(4, 4, 110.0).places, "2 + 4 leaves two orphans");
        assert_eq!(
            lead(4, 4, 110.0).extent,
            120.0,
            "no break keeps four widows"
        );
        assert!(!lead(6, 1, 110.0).places, "five lines fit, six wanted");
        assert_eq!(lead(3, 2, 110.0).extent, 60.0, "the first three lines");
        assert!(lead(2, 2, 120.0).places, "the paragraph fits whole");
        assert!(!lead(1, 1, 15.0).places, "not even the first line fits");
    }

    /// The lines of each fragment of the inline root `id`.
    fn line_partition(geom: &PaginationGeometryTable, id: usize) -> (Vec<u32>, Vec<usize>) {
        let entry = geom.get(&id).expect("the paragraph is recorded");
        (
            entry.fragments.iter().map(|f| f.page_index).collect(),
            entry.line_boundaries.clone(),
        )
    }

    /// A body-level paragraph starting mid-page splits at the page foot by
    /// its own `orphans` and `widows`: six 20px lines from y=90 on a 200px
    /// strip split 3 + 3 at `3 / 3` and move whole at `4 / 4`.
    #[test]
    fn a_paragraph_mid_page_splits_at_the_foot_by_its_orphans_and_widows() {
        let split = |limit: u32| {
            let html = format!(
                r#"<html><body style="margin:0">
                  <div style="height:90px"></div>
                  <p id="p" style="margin:0;line-height:20px;orphans:{limit};widows:{limit}">a<br>b<br>c<br>d<br>e<br>f</p>
                </body></html>"#
            );
            let mut doc = parse(&html, 200.0);
            let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
            let p_id = find_by_id(&doc, "p").expect("p#p");
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table_cs);
            line_partition(&geom, p_id)
        };
        assert_eq!(split(3), (vec![0, 1], vec![0, 3, 6]), "3 + 3");
        assert_eq!(split(4), (vec![1], vec![0, 6]), "moves whole");
    }

    /// A nested paragraph after content of its container takes `orphans`
    /// and `widows` from an ancestor (both are inherited), past an
    /// `inherit` of its own, and splits at the page foot by them.
    #[test]
    fn a_nested_paragraph_inherits_orphans_and_widows() {
        let split = |limit: u32| {
            let html = format!(
                r#"<html><body style="margin:0">
                  <div style="orphans:{limit};widows:{limit}"><div>
                    <div style="height:90px"></div>
                    <p id="p" style="margin:0;line-height:20px;orphans:inherit">a<br>b<br>c<br>d<br>e<br>f</p>
                  </div></div>
                </body></html>"#
            );
            let mut doc = parse(&html, 200.0);
            let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
            let p_id = find_by_id(&doc, "p").expect("p#p");
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 200.0_f32.as_px(), &table_cs);
            line_partition(&geom, p_id)
        };
        assert_eq!(split(1), (vec![0, 1], vec![0, 5, 6]), "5 + 1");
        assert_eq!(split(4), (vec![1], vec![0, 6]), "moves whole");
    }

    /// A box's padding is no content: a paragraph that opens a padded box
    /// at the top of a page is at the top of the page, where moving it
    /// gains nothing, so it splits after the lines that fit though its
    /// `orphans` and `widows` cannot be met (css-break-3 §4.4), rather than
    /// moving whole and leaving the box empty. Three 20px lines below 10px
    /// of padding on a 60px page: two fit.
    #[test]
    fn a_paragraph_at_the_top_of_a_page_inside_a_padded_box_relaxes_its_orphans() {
        let html = r#"<html><body style="margin:0">
              <div style="padding-top:10px">
                <p id="p" style="margin:0;line-height:20px;orphans:3;widows:2">a<br>b<br>c</p>
              </div>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let p_id = find_by_id(&doc, "p").expect("p#p");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(line_partition(&geom, p_id), (vec![0, 1], vec![0, 2, 3]));
    }

    /// The flex item before a paragraph's item sits beside it, not above:
    /// the paragraph, in a padded item at the top of a page, still splits
    /// there within relaxed `orphans` and `widows`.
    #[test]
    fn a_paragraph_beside_a_flex_item_at_the_top_of_a_page_relaxes_its_orphans() {
        let html = r#"<html><body style="margin:0">
              <div style="display:flex">
                <div style="flex:1 1 0"><div style="height:50px"></div></div>
                <div style="flex:1 1 0;padding-top:10px">
                  <p id="p" style="margin:0;line-height:20px;orphans:3;widows:2">a<br>b<br>c</p>
                </div>
              </div>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let p_id = find_by_id(&doc, "p").expect("p#p");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(line_partition(&geom, p_id), (vec![0, 1], vec![0, 2, 3]));
    }

    /// A flex item's fragment on a page its line crosses reaches the page
    /// foot whichever of its items crosses first: a paragraph that splits
    /// after an entered item that broke early grows the fragment the item
    /// closed, as an entered item grows the one a paragraph closed. The
    /// container `x` starts 20 px down a 60 px page, below content, beside a
    /// taller box, so that its line does not stretch it to the foot.
    #[test]
    fn a_split_paragraph_grows_its_containers_fragment_an_item_beside_it_closed() {
        let item = |first: bool| {
            let entered = r#"<div style="flex:1 1 0"><div style="height:10px"></div><div style="height:10px;break-before:page"></div></div>"#;
            let paragraph = r#"<p style="flex:1 1 0;margin:0;line-height:20px;orphans:1;widows:1">1<br>2<br>3<br>4<br>5</p>"#;
            if first {
                format!("{entered}{paragraph}")
            } else {
                format!("{paragraph}{entered}")
            }
        };
        for entered_first in [true, false] {
            let html = format!(
                r#"<html><body style="margin:0"><div style="height:20px"></div>
                  <div style="display:flex;align-items:flex-start"><div id="x" style="flex:1 1 0;display:flex">{}</div><div style="flex:1 1 0;height:150px"></div></div>
                </body></html>"#,
                item(entered_first)
            );
            let mut doc = parse(&html, 200.0);
            let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
            let x = find_by_id(&doc, "x").expect("div#x");
            let geom =
                super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
            let first: Vec<(u32, f32, f32)> = geom[&x]
                .fragments
                .iter()
                .filter(|f| f.page_index == 0)
                .map(|f| (f.page_index, f.y.to_f32(), f.height.to_f32()))
                .collect();
            assert_eq!(
                first,
                [(0, 20.0, 40.0)],
                "entered item first: {entered_first}; geom={geom:?}"
            );
        }
    }

    /// Blitz lays a table out as a grid of its cells: the first cell of a
    /// row sits beside the second, not above it, so the second cell's
    /// paragraph at the top of a page splits there within relaxed `orphans`
    /// and `widows`, as the first cell's would.
    #[test]
    fn a_paragraph_in_the_second_cell_of_a_table_row_at_the_top_of_a_page_relaxes_its_orphans() {
        let html = r#"<html><body style="margin:0">
              <table style="border-spacing:0"><tr>
                <td style="padding:0;line-height:20px">x</td>
                <td style="padding:0">
                  <p id="p" style="margin:0;line-height:25px;orphans:2;widows:2">a<br>b<br>c</p>
                </td>
              </tr></table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let p_id = find_by_id(&doc, "p").expect("p#p");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(line_partition(&geom, p_id), (vec![0, 1], vec![0, 2, 3]));
    }

    /// At the body level an edge is no content either: a paragraph whose
    /// `orphans` cannot be met, 10 px below the top of the first page, under
    /// the body's padding, splits on that page instead of moving to the next.
    #[test]
    fn a_body_paragraph_below_the_body_padding_at_the_top_of_a_page_relaxes_its_orphans() {
        let html = r#"<html><body style="margin:0;padding-top:10px">
              <p id="p" style="margin:0;line-height:20px;orphans:3;widows:2">a<br>b<br>c</p>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let p_id = find_by_id(&doc, "p").expect("p#p");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(line_partition(&geom, p_id), (vec![0, 1], vec![0, 2, 3]));
    }

    /// A flex row whose first item crosses the page: the next item restarts
    /// where the row did, at the top of an empty page, so its paragraph
    /// splits there within relaxed minimums; the row ends on the page the
    /// first item reached, which holds its content though the row's last
    /// item, empty, was placed on the first page: a paragraph after the row
    /// that cannot split there moves whole.
    #[test]
    fn a_flex_row_restores_its_empty_page_for_each_item_and_ends_on_its_last_page_with_content() {
        let html = r#"<html><body style="margin:0">
              <div style="display:flex">
                <div style="flex:1 1 0">
                  <p style="margin:0;line-height:20px">a<br>b<br>c<br>d<br>e</p>
                </div>
                <div style="flex:1 1 0;align-self:flex-start;padding-top:10px">
                  <p id="beside" style="margin:0;line-height:20px;orphans:3;widows:2">a<br>b<br>c</p>
                </div>
                <div style="flex:1 1 0;align-self:flex-start"></div>
              </div>
              <p id="after" style="margin:0;line-height:20px">a<br>b<br>c</p>
            </body></html>"#;
        let mut doc = parse(html, 300.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let beside = find_by_id(&doc, "beside").expect("p#beside");
        let after = find_by_id(&doc, "after").expect("p#after");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(
            (line_partition(&geom, beside), line_partition(&geom, after)),
            ((vec![0, 1], vec![0, 2, 3]), (vec![2], vec![0, 3]))
        );
    }

    /// A cell of a table's second row lies within the height of a cell of
    /// the first row that spans both, but below that cell's top: it is not
    /// beside it, and the first row's content is above it on the page. Its
    /// paragraph, crossing the page foot where its `orphans` cannot be met,
    /// moves to the next page, and splits at its top there.
    #[test]
    fn a_paragraph_in_a_row_below_a_spanning_cell_moves_on_at_the_foot() {
        let html = r#"<html><body style="margin:0">
              <table style="border-spacing:0">
                <tr>
                  <td style="padding:0;line-height:20px">x</td>
                  <td rowspan="2" style="padding:0">
                    <img style="display:block;width:10px;height:100px">
                  </td>
                </tr>
                <tr>
                  <td style="padding:0">
                    <p id="p" style="margin:0;line-height:25px;orphans:2;widows:2">a<br>b<br>c</p>
                  </td>
                </tr>
              </table>
            </body></html>"#;
        let mut doc = parse(html, 200.0);
        let table_cs = blitz_adapter::extract_column_style_table(&doc, &[]);
        let p_id = find_by_id(&doc, "p").expect("p#p");
        let geom = super::run_pass_with_break_styles(doc.deref_mut(), 60.0_f32.as_px(), &table_cs);
        assert_eq!(line_partition(&geom, p_id), (vec![1, 2], vec![0, 2, 3]));
    }

    // ── StringSetPageState / PageRunningState derived-trait coverage ─────────

    #[test]
    fn string_set_page_state_default_has_none_fields() {
        let s = StringSetPageState::default();
        assert!(s.start.is_none());
        assert!(s.first.is_none());
        assert!(s.last.is_none());
    }

    #[test]
    fn string_set_page_state_clone_and_eq() {
        let s = StringSetPageState {
            start: Some("x".into()),
            first: Some("y".into()),
            last: Some("z".into()),
        };
        let c = s.clone();
        assert_eq!(s, c);
    }

    #[test]
    fn string_set_page_state_debug_contains_field_names() {
        let s = StringSetPageState {
            start: None,
            first: Some("hello".into()),
            last: None,
        };
        let dbg = format!("{s:?}");
        assert!(dbg.contains("StringSetPageState"), "got: {dbg}");
        assert!(dbg.contains("hello"), "first value should appear: {dbg}");
    }

    #[test]
    fn page_running_state_default_has_empty_instance_ids() {
        let s = PageRunningState::default();
        assert!(s.instance_ids.is_empty());
    }

    #[test]
    fn page_running_state_clone_preserves_ids() {
        let mut s = PageRunningState::default();
        s.instance_ids.push(3);
        s.instance_ids.push(7);
        let c = s.clone();
        assert_eq!(c.instance_ids, vec![3, 7]);
    }

    #[test]
    fn page_running_state_debug_contains_struct_name() {
        let s = PageRunningState {
            instance_ids: vec![1, 2],
        };
        let dbg = format!("{s:?}");
        assert!(dbg.contains("PageRunningState"), "got: {dbg}");
        assert!(dbg.contains('1'), "instance id should appear: {dbg}");
    }

    // ── Fragment / PaginationGeometry derived-trait coverage ─────────────────

    #[test]
    fn fragment_clone_and_eq() {
        let f = Fragment {
            page_index: 5,
            x: 10.0_f32.as_px(),
            y: 20.0_f32.as_px(),
            width: 30.0_f32.as_px(),
            height: 40.0_f32.as_px(),
        };
        let c = f.clone();
        assert_eq!(f, c);
    }

    #[test]
    fn fragment_debug_contains_page_index() {
        let f = Fragment {
            page_index: 9,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 1.0_f32.as_px(),
            height: 1.0_f32.as_px(),
        };
        let dbg = format!("{f:?}");
        assert!(dbg.contains("Fragment"), "got: {dbg}");
        assert!(dbg.contains('9'), "page_index should appear: {dbg}");
    }

    #[test]
    fn pagination_geometry_clone_preserves_is_repeat_and_fragments() {
        let mut g = PaginationGeometry {
            is_repeat: true,
            ..Default::default()
        };
        g.fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 50.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let c = g.clone();
        assert!(c.is_repeat);
        assert_eq!(c.fragments.len(), 1);
    }

    #[test]
    fn pagination_geometry_debug_contains_struct_name() {
        let g = PaginationGeometry::default();
        let dbg = format!("{g:?}");
        assert!(dbg.contains("PaginationGeometry"), "got: {dbg}");
    }

    // ── collect_running_element_states: pure-function branch coverage ─────────

    /// Empty geometry → exactly one empty page state (same convention as the
    /// other collect_* functions).
    #[test]
    fn running_states_empty_geometry_returns_one_empty_page() {
        let geom = PaginationGeometryTable::new();
        let store = crate::gcpm::running::RunningElementStore::new();
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        assert!(states[0].is_empty());
    }

    /// Node appears in geometry but has no entry in the running store →
    /// its page slot stays empty (the `instance_for_node` guard fires).
    #[test]
    fn running_states_node_not_in_store_is_skipped() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(42).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let store = crate::gcpm::running::RunningElementStore::new();
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        assert!(states[0].is_empty(), "node not in store must not appear");
    }

    /// Node has geometry with empty fragments → the `first_frag` guard fires
    /// and the node is silently skipped.
    #[test]
    fn running_states_node_with_empty_fragments_is_skipped() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(5).or_default(); // entry exists but fragments vec is empty
        let mut store = crate::gcpm::running::RunningElementStore::new();
        store.register(5, "header".into(), "<h1>hi</h1>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        assert!(
            states[0].is_empty(),
            "empty fragments must not emit a state"
        );
    }

    /// A node with two fragments (page 0 and page 1) must be recorded only under
    /// the page of its *first* fragment. The implementation calls
    /// `geom.fragments.first()` and does not scan subsequent fragments, so only
    /// page 0 carries the state entry.
    #[test]
    fn running_states_only_first_fragment_page_is_used() {
        let mut geom = PaginationGeometryTable::new();
        let entry = geom.entry(7).or_default();
        entry.fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        entry.fragments.push(Fragment {
            page_index: 1,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let mut store = crate::gcpm::running::RunningElementStore::new();
        let instance_id = store.register(7, "banner".into(), "<p>B</p>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 2, "two pages expected");
        let p0 = &states[0];
        let p1 = &states[1];
        assert!(
            p0.contains_key("banner"),
            "banner must be recorded on page 0 (first fragment)"
        );
        assert_eq!(p0["banner"].instance_ids, vec![instance_id]);
        assert!(
            p1.is_empty(),
            "page 1 must not carry a duplicate entry for the second fragment"
        );
    }

    /// Two nodes sharing the same running name on the same page must accumulate
    /// both instance_ids under that name entry, ordered by geometry traversal
    /// (BTreeMap key order), not by registration order.
    ///
    /// Registering node 20 before node 10 in the store gives node 20 a lower
    /// instance_id. The assertion still expects [id_for_10, id_for_20] because
    /// the geometry loop visits node 10 (smaller key) before node 20.
    #[test]
    fn running_states_two_nodes_same_name_same_page_accumulate_ids() {
        let mut geom = PaginationGeometryTable::new();
        for node_id in [10_usize, 20] {
            geom.entry(node_id).or_default().fragments.push(Fragment {
                page_index: 0,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 50.0_f32.as_px(),
                height: 20.0_f32.as_px(),
            });
        }
        let mut store = crate::gcpm::running::RunningElementStore::new();
        // Intentionally register in reverse order (20 before 10) so that the
        // instance_id for node 20 is lower than for node 10. The assertion
        // below proves the order is driven by BTreeMap traversal, not by
        // registration order.
        let id_for_20 = store.register(20, "section".into(), "<p>B</p>".into());
        let id_for_10 = store.register(10, "section".into(), "<p>A</p>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        let entry = states[0].get("section").expect("section must be present");
        // geometry traversal: key 10 < 20, so id_for_10 is pushed first.
        assert_eq!(entry.instance_ids, vec![id_for_10, id_for_20]);
    }

    /// Two nodes with different running names each appear in their own slot.
    #[test]
    fn running_states_two_different_names_produce_separate_entries() {
        let mut geom = PaginationGeometryTable::new();
        for node_id in [1_usize, 2] {
            geom.entry(node_id).or_default().fragments.push(Fragment {
                page_index: 0,
                x: 0.0_f32.as_px(),
                y: 0.0_f32.as_px(),
                width: 50.0_f32.as_px(),
                height: 10.0_f32.as_px(),
            });
        }
        let mut store = crate::gcpm::running::RunningElementStore::new();
        store.register(1, "header".into(), "<h1>H</h1>".into());
        store.register(2, "footer".into(), "<p>F</p>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        assert!(states[0].contains_key("header"), "header entry must exist");
        assert!(states[0].contains_key("footer"), "footer entry must exist");
    }

    /// Running element on page 1 must not appear in the page 0 slot.
    #[test]
    fn running_states_node_on_page_1_appears_only_in_slot_1() {
        let mut geom = PaginationGeometryTable::new();
        // Node 10 on page 0, node 20 on page 1.
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        geom.entry(20).or_default().fragments.push(Fragment {
            page_index: 1,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        let mut store = crate::gcpm::running::RunningElementStore::new();
        store.register(20, "chapter".into(), "<h2>Ch2</h2>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 2, "two pages expected");
        assert!(
            states[0].is_empty(),
            "page 0 must not carry the page-1 element"
        );
        assert!(
            states[1].contains_key("chapter"),
            "chapter must appear on page 1"
        );
    }

    /// A node registered in the store but absent from geometry is never visited
    /// by the geometry loop — it produces no state entry.
    #[test]
    fn running_states_store_only_node_not_in_geom_produces_no_entry() {
        let mut geom = PaginationGeometryTable::new();
        geom.entry(10).or_default().fragments.push(Fragment {
            page_index: 0,
            x: 0.0_f32.as_px(),
            y: 0.0_f32.as_px(),
            width: 100.0_f32.as_px(),
            height: 50.0_f32.as_px(),
        });
        // node 77 is in the store but has no geometry entry → loop never visits it.
        let mut store = crate::gcpm::running::RunningElementStore::new();
        store.register(77, "ghost".into(), "<p>x</p>".into());
        let states = collect_running_element_states(&geom, &store);
        assert_eq!(states.len(), 1);
        assert!(
            states[0].is_empty(),
            "store-only node (not in geom) must not produce a state entry"
        );
    }
}
