#![cfg(target_os = "linux")]

//! `::before` and `::after` `content: url()` images through the Raikiri
//! engine, checked in the rasterized PDF.

use std::path::Path;
use std::process::Command;

/// Render `body` with `css` and an 8x4 red `source.png` next to the input,
/// and return the first page rasterized at 96 dpi with the PDF bytes.
fn render(css: &str, body: &str) -> (image::RgbaImage, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.html");
    let pdf = dir.path().join("output.pdf");
    let prefix = dir.path().join("page");
    image::RgbaImage::from_pixel(8, 4, image::Rgba([255, 0, 0, 255]))
        .save(dir.path().join("source.png"))
        .unwrap();
    std::fs::write(
        &input,
        format!(
            "<!doctype html><style>@page{{size:100px 100px;margin:0}}body{{margin:0;background:white;font:20px/20px sans-serif}}p{{margin:0;color:transparent}}{css}</style>{body}"
        ),
    )
    .unwrap();
    let font =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fulgur-ruby/spec/fixtures/noto_sans.ttf");
    let output = Command::new(env!("CARGO_BIN_EXE_fulgur-dev"))
        .args([
            "render",
            "--engine",
            "raikiri",
            "--no-system-fonts",
            "--font",
        ])
        .arg(font)
        .arg(input)
        .arg("-o")
        .arg(&pdf)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new("pdftocairo")
        .args(["-png", "-singlefile", "-r", "96", "-f", "1", "-l", "1"])
        .arg(&pdf)
        .arg(&prefix)
        .output()
        .expect("pdftocairo is required; install poppler-utils");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        image::open(prefix.with_extension("png"))
            .unwrap()
            .to_rgba8(),
        std::fs::read(pdf).unwrap(),
    )
}

/// The bounds of the red pixels of `image` as `(x_min, x_max, y_min, y_max)`,
/// and the area they cover.
fn red(image: &image::RgbaImage) -> ((u32, u32, u32, u32), f64) {
    let pixels: Vec<_> = image
        .enumerate_pixels()
        .filter(|(_, _, p)| p[0] > p[1])
        .map(|(x, y, _)| (x, y))
        .collect();
    assert!(!pixels.is_empty(), "no image pixels drawn");
    let bounds = (
        pixels.iter().map(|p| p.0).min().unwrap(),
        pixels.iter().map(|p| p.0).max().unwrap(),
        pixels.iter().map(|p| p.1).min().unwrap(),
        pixels.iter().map(|p| p.1).max().unwrap(),
    );
    let coverage = image.pixels().map(|p| f64::from(255 - p[1]) / 255.0).sum();
    (bounds, coverage)
}

#[test]
fn before_image_is_drawn_at_the_start_of_the_first_line() {
    let (image, bytes) = render("p::before{content:url(source.png)}", "<p>A</p>");
    let ((x_min, x_max, y_min, y_max), coverage) = red(&image);
    assert!((coverage - 32.0).abs() < 0.1, "coverage={coverage}");
    assert_eq!((x_min, x_max), (0, 7));
    assert!((4..=5).contains(&(y_max - y_min + 1)));
    // The image sits on the first line's baseline, inside the 20px line.
    assert!(y_max < 20, "y_max={y_max}");
    let pdf = lopdf::Document::load_mem(&bytes).unwrap();
    let images = pdf
        .objects
        .values()
        .filter(|object| {
            object.as_stream().is_ok_and(|stream| {
                stream
                    .dict
                    .get(b"Subtype")
                    .and_then(lopdf::Object::as_name)
                    .ok()
                    == Some(b"Image".as_slice())
            })
        })
        .count();
    assert_eq!(images, 1);
}

#[test]
fn after_image_follows_the_text_and_replacing_content_is_sized_by_the_pseudo() {
    let (before, _) = render("p::before{content:url(source.png)}", "<p>A</p>");
    let (after, _) = render("p::after{content:url(source.png)}", "<p>A</p>");
    let ((before_x, ..), _) = red(&before);
    let ((after_x, ..), coverage) = red(&after);
    assert!((coverage - 32.0).abs() < 0.1, "coverage={coverage}");
    assert!(after_x > before_x + 8, "after_x={after_x}");
    // A pseudo-element whose content is only the image is replaced by it, so
    // its own `width` and `height` size the image (CSS Content 3 §2.2).
    let (sized, _) = render(
        "p::before{content:url(source.png);width:16px;height:8px}",
        "<p>A</p>",
    );
    let ((x_min, x_max, y_min, y_max), coverage) = red(&sized);
    assert!((coverage - 128.0).abs() < 0.2, "coverage={coverage}");
    assert_eq!((x_min, x_max), (0, 15));
    assert!((8..=9).contains(&(y_max - y_min + 1)));
}

#[test]
fn missing_image_draws_nothing() {
    let (image, _) = render("p::before{content:url(missing.png)}", "<p>A</p>");
    assert!(image.pixels().all(|p| p[0] == p[1]));
}
