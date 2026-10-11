use super::*;

fn rounded(radius: f32) -> RoundedRect {
    RoundedRect {
        radii: [[radius; 2]; 4],
        ..RoundedRect::rect(10.0, 20.0, 100.0, 50.0)
    }
}

#[test]
fn spread_grows_the_box_and_its_radii() {
    let grown = rounded(10.0).spread(5.0);
    assert_eq!(
        (grown.x, grown.y, grown.width, grown.height),
        (5.0, 15.0, 110.0, 60.0)
    );
    assert_eq!(grown.radii, [[15.0; 2]; 4]);
}

#[test]
fn spread_keeps_sharp_corners_sharp_and_small_radii_grow_less() {
    assert_eq!(rounded(0.0).spread(5.0).radii, [[0.0; 2]; 4]);
    // ratio 0.5: the spread counts 1 + (0.5 - 1)³ = 0.875 of itself.
    let grown = rounded(2.0).spread(4.0);
    assert_eq!(grown.radii, [[2.0 + 4.0 * 0.875; 2]; 4]);
}

#[test]
fn negative_spread_shrinks_radii_down_to_zero() {
    let shrunk = rounded(10.0).spread(-4.0);
    assert_eq!((shrunk.width, shrunk.height), (92.0, 42.0));
    assert_eq!(shrunk.radii, [[6.0; 2]; 4]);
    assert_eq!(rounded(3.0).spread(-4.0).radii, [[0.0; 2]; 4]);
    assert!(rounded(3.0).spread(-30.0).is_empty());
}

#[test]
fn broken_edges_extend_and_cut() {
    let fragment = RoundedRect::rect(0.0, 100.0, 50.0, 40.0);
    let slice = Slice {
        top: true,
        ..Slice::default()
    };
    let extended = extended(&fragment, slice, 10.0);
    assert_eq!((extended.y, extended.height), (90.0, 50.0));
    let mut area = RoundedRect::rect(-20.0, 70.0, 90.0, 100.0);
    cut(&mut area, &fragment, slice);
    assert_eq!(
        (area.x, area.y, area.width, area.height),
        (-20.0, 100.0, 90.0, 70.0)
    );
}

#[test]
fn cropping_squares_the_cut_corners() {
    let area = RoundedRect::rect(0.0, 0.0, 60.0, 100.0);
    let cut = cropped(rounded(10.0), &area);
    assert_eq!((cut.x, cut.width), (10.0, 50.0));
    // The right edge is cut, so its corners are square.
    assert_eq!(cut.radii, [[10.0; 2], [0.0; 2], [0.0; 2], [10.0; 2]]);
    assert!(cropped(rounded(10.0), &RoundedRect::rect(500.0, 0.0, 1.0, 1.0)).is_empty());
}

#[test]
fn wide_blurs_use_fewer_raster_pixels() {
    assert_eq!(raster_scale(1.0), RASTER_SCALE);
    assert_eq!(raster_scale(6.0), 2.0);
    assert_eq!(raster_scale(40.0), 1.0);
}

#[test]
fn inline_breaks_extend_and_cut_the_sides() {
    let fragment = RoundedRect::rect(10.0, 0.0, 50.0, 20.0);
    let slice = Slice {
        left: true,
        right: true,
        bottom: true,
        ..Slice::default()
    };
    let wide = extended(&fragment, slice, 5.0);
    assert_eq!((wide.x, wide.width, wide.height), (5.0, 60.0, 25.0));
    let mut area = RoundedRect::rect(-10.0, -10.0, 100.0, 50.0);
    cut(&mut area, &fragment, slice);
    assert_eq!(
        (area.x, area.y, area.width, area.height),
        (10.0, -10.0, 50.0, 30.0)
    );
}
