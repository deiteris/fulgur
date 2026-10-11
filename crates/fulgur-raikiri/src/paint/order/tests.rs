use super::*;
use raikiri_html::{
    GeneratedKind, LayoutConfig, LayoutStatus, PaintEvent, PaintTransform, RunSource,
};

fn document() -> raikiri_html::DocumentLayout {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.html");
    std::fs::write(&path, "<p>Hello</p>").unwrap();
    let LayoutStatus::Completed(document) = crate::layout_file(
        &path,
        &fulgur_core::Config::default(),
        LayoutConfig::default(),
    )
    .unwrap() else {
        panic!("completed layout");
    };
    document
}

#[test]
fn ordinary_text_is_supported() {
    let document = document();
    let page = document.page(0).unwrap();
    assert!(supported(&page.paint_order(), &page.text_runs(), |_| None));
}

#[test]
fn duplicate_text_events_are_not_drawn_twice() {
    let document = document();
    let page = document.page(0).unwrap();
    let mut events = page.paint_order();
    let text = events
        .iter()
        .find(|event| matches!(event, PaintEvent::Text(_)))
        .copied()
        .unwrap();
    events.push(text);
    assert!(!supported(&events, &page.text_runs(), |_| None));
}

#[test]
fn missing_text_event_uses_legacy_page() {
    let document = document();
    let page = document.page(0).unwrap();
    let events: Vec<_> = page
        .paint_order()
        .into_iter()
        .filter(|event| !matches!(event, PaintEvent::Text(_)))
        .collect();
    assert!(!supported(&events, &page.text_runs(), |_| None));
}

#[test]
fn generated_run_uses_legacy_page() {
    let document = document();
    let page = document.page(0).unwrap();
    let mut runs = page.text_runs();
    let RunSource::Text(node) = runs[0].source else {
        panic!("text run");
    };
    runs[0].source = RunSource::Generated(node, GeneratedKind::Before);
    assert!(!supported(&page.paint_order(), &runs, |_| None));
}

#[test]
fn invalid_opacity_is_rejected_before_surface_stack_changes() {
    for alpha in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(!supported(
            &[PaintEvent::PushOpacity(alpha), PaintEvent::PopOpacity],
            &[],
            |_| None
        ));
    }
    for alpha in [0.0, 1.0] {
        assert!(supported(
            &[PaintEvent::PushOpacity(alpha), PaintEvent::PopOpacity],
            &[],
            |_| None
        ));
    }
}

#[test]
fn line_events_include_generated_and_ellipsis_runs() {
    let document = document();
    let page = document.page(0).unwrap();
    let mut runs = page.text_runs();
    let RunSource::Text(node) = runs[0].source else {
        panic!("text run")
    };
    let events = page.paint_order_for_text_runs(&runs);
    assert!(supported(&events, &runs, |_| None));
    runs[0].source = RunSource::Generated(node, GeneratedKind::Before);
    assert!(supported(&events, &runs, |_| None));
    runs[0].source = RunSource::Ellipsis(node);
    assert!(supported(&events, &runs, |_| None));
}

#[test]
fn duplicate_missing_and_unknown_lines_are_rejected() {
    let document = document();
    let page = document.page(0).unwrap();
    let runs = page.text_runs();
    let events = page.paint_order_for_text_runs(&runs);
    let line = events
        .iter()
        .find(|event| matches!(event, PaintEvent::TextLine(_)))
        .copied()
        .unwrap();
    let mut duplicate = events.clone();
    duplicate.push(line);
    assert!(!supported(&duplicate, &runs, |_| None));
    let mut missing = events.clone();
    missing.retain(|event| !matches!(event, PaintEvent::TextLine(_)));
    assert!(!supported(&missing, &runs, |_| None));
    let mut unknown_line = runs[0].line;
    unknown_line.index += 100;
    let mut unknown = events;
    unknown.push(PaintEvent::TextLine(unknown_line));
    assert!(!supported(&unknown, &runs, |_| None));
}

#[test]
fn mixed_fragment_and_line_events_are_rejected() {
    let document = document();
    let page = document.page(0).unwrap();
    let runs = page.text_runs();
    let mut events = page.paint_order_for_text_runs(&runs);
    events.extend(
        page.paint_order()
            .into_iter()
            .filter(|event| matches!(event, PaintEvent::Text(_))),
    );
    assert!(!supported(&events, &runs, |_| None));
}

#[test]
fn grouped_runs_need_their_own_group_line() {
    let document = document();
    let page = document.page(0).unwrap();
    let runs = page.text_runs();
    let line = runs[0].line;
    let RunSource::Text(node) = runs[0].source else {
        panic!("text run")
    };
    let group = Some(node);
    let grouped = [PaintEvent::GroupTextLine(line, node)];
    assert!(supported(&grouped, &runs, |_| group));
    // A paragraph line does not draw grouped runs, nor the reverse.
    assert!(!supported(&[PaintEvent::TextLine(line)], &runs, |_| group));
    assert!(!supported(&grouped, &runs, |_| None));
    // Each group line is drawn once.
    let twice = [grouped[0], grouped[0]];
    assert!(!supported(&twice, &runs, |_| group));
}

#[test]
fn transforms_are_supported_when_finite() {
    let document = document();
    let page = document.page(0).unwrap();
    let mut events = page.paint_order();
    events.insert(
        0,
        PaintEvent::PushTransform(PaintTransform::new(0.0, 1.0, -1.0, 0.0, 5.0, 7.0)),
    );
    events.push(PaintEvent::PopTransform);
    assert!(supported(&events, &page.text_runs(), |_| None));
    events[0] = PaintEvent::PushTransform(PaintTransform::translate(f32::NAN, 0.0));
    assert!(!supported(&events, &page.text_runs(), |_| None));
}
