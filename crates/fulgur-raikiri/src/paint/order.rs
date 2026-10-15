use raikiri_html::{NodeId, PaintEvent, PositionedGlyphRun, RunSource};
use std::collections::HashSet;

/// Whether the ordered events draw every run exactly once. `group` gives the
/// inline opacity group of a run, which picks the line event drawing it.
pub(super) fn supported(
    events: &[PaintEvent<'_>],
    runs: &[PositionedGlyphRun<'_>],
    group: impl Fn(&PositionedGlyphRun<'_>) -> Option<NodeId>,
) -> bool {
    let mut text_nodes = HashSet::new();
    let mut text_lines = HashSet::new();
    let run_lines: HashSet<_> = runs.iter().map(|run| run.line).collect();
    for event in events {
        match event {
            PaintEvent::Text(fragment) => {
                if !text_nodes.insert(fragment.node()) {
                    return false;
                }
            }
            PaintEvent::TextLine(line) => {
                if !run_lines.contains(line) || !text_lines.insert((*line, None)) {
                    return false;
                }
            }
            PaintEvent::GroupTextLine(line, owner) => {
                if !run_lines.contains(line) || !text_lines.insert((*line, Some(*owner))) {
                    return false;
                }
            }
            PaintEvent::PushOpacity(alpha) => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
                    return false;
                }
            }
            PaintEvent::PushTransform(transform) => {
                let values = [
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ];
                if !values.iter().all(|value| value.is_finite()) {
                    return false;
                }
            }
            PaintEvent::ColumnRule(_)
            | PaintEvent::Box(_)
            | PaintEvent::Replaced(_)
            | PaintEvent::MarkerImage(_)
            | PaintEvent::GeneratedImage(_)
            | PaintEvent::GeneratedBox(_)
            | PaintEvent::LineBox(_)
            | PaintEvent::PushClip(_, _)
            | PaintEvent::PopClip
            | PaintEvent::PopOpacity
            | PaintEvent::PopTransform => {}
            _ => return false,
        }
    }
    // Combining the two text streams could draw the same glyph twice.
    if !text_nodes.is_empty() && !text_lines.is_empty() {
        return false;
    }
    let line = |run: &PositionedGlyphRun<'_>| text_lines.contains(&(run.line, group(run)));
    runs.iter().all(|run| match run.source {
        RunSource::Text(node) => line(run) || text_nodes.contains(&node),
        RunSource::Generated(_, _) | RunSource::Ellipsis(_) => line(run),
        _ => false,
    })
}

#[cfg(test)]
mod tests;
