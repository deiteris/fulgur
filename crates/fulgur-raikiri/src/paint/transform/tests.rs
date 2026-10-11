use super::*;

#[test]
fn inverse_undoes_the_transform() {
    let transform = PaintTransform::new(0.0, 2.0, -3.0, 0.0, 5.0, 7.0);
    let inverse = inverse(transform).expect("invertible");
    let (x, y) = (inverse * transform).apply((11.0, 13.0));
    assert!((x - 11.0).abs() < 1e-4 && (y - 13.0).abs() < 1e-4);
    assert_eq!(
        super::inverse(PaintTransform::new(1.0, 2.0, 2.0, 4.0, 0.0, 0.0)),
        None
    );
}

#[test]
fn relative_transform_maps_run_coordinates_into_the_space() {
    let space = PaintTransform::translate(10.0, 0.0);
    assert_eq!(relative(space, Some(space)), None);
    assert_eq!(relative(PaintTransform::IDENTITY, None), None);
    let run = PaintTransform::translate(10.0, 5.0);
    assert_eq!(
        relative(space, Some(run)),
        Some(PaintTransform::translate(0.0, 5.0))
    );
    // Nothing is drawn in a singular space; the run keeps its own transform.
    let flat = PaintTransform::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(relative(flat, Some(run)), Some(run));
}

#[test]
fn bounds_enclose_the_moved_corners() {
    let quarter = PaintTransform::new(0.0, 1.0, -1.0, 0.0, 0.0, 0.0);
    assert_eq!(
        bounds(quarter, PaintRect::new(1.0, 2.0, 3.0, 4.0)),
        PaintRect::new(-6.0, 1.0, 4.0, 3.0)
    );
}
