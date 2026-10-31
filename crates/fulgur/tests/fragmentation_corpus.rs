//! Fragmentation corpus: the pagination geometry of each fixture, compared
//! line by line with the text recorded beside it.
//!
//! Each `fragmentation_corpus/<name>.html` starts with `<!-- page: W H -->`,
//! the page size in CSS px (no page margin), and is compared with
//! `<name>.geometry.txt`. Every `fulgur-vrt/fixtures/layout/*.html` is
//! rendered with its manifest page size and margin and compared with
//! `fragmentation_corpus/vrt-layout/<name>.geometry.txt`. The text is
//! `fulgur::pagination_layout::geometry_text`: one line per node and
//! fragment, so a change shows which box moved, to which page and where.
//!
//! Every fixture is laid out with the bundled Noto Sans and no system
//! fonts, so the text does not depend on the host's fonts.
//!
//! `UPDATE_GEOMETRY=1 cargo test -p fulgur --test fragmentation_corpus`
//! rewrites the recorded texts; review each changed line before committing.

use fulgur::Engine;
use fulgur::asset::AssetBundle;
use fulgur::config::{Margin, PageSize};
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn corpus_dir() -> PathBuf {
    manifest_dir().join("tests/fragmentation_corpus")
}

/// Noto Sans as the only font, for every element.
fn assets() -> AssetBundle {
    let font = manifest_dir().join("../../examples/.fonts/NotoSans-Regular.ttf");
    let mut assets = AssetBundle::default();
    assets
        .add_font_file(&font)
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", font.display()));
    assets.add_css("* { font-family: 'Noto Sans' !important; }");
    assets
}

/// The page size of `<!-- page: W H -->` on the fixture's first line, in px.
fn page_of(html: &str, path: &Path) -> (f32, f32) {
    let first = html.lines().next().unwrap_or_default().trim();
    let size = first
        .strip_prefix("<!-- page:")
        .and_then(|rest| rest.strip_suffix("-->"))
        .unwrap_or_else(|| panic!("{} must start with <!-- page: W H -->", path.display()));
    let numbers: Vec<f32> = size
        .split_whitespace()
        .map(|n| {
            n.parse()
                .unwrap_or_else(|_| panic!("{}: bad size {n}", path.display()))
        })
        .collect();
    assert_eq!(
        numbers.len(),
        2,
        "{}: the page line needs W and H",
        path.display()
    );
    (numbers[0], numbers[1])
}

/// Compare the geometry text of `html` with `expected`, or write it there
/// when `UPDATE_GEOMETRY` is set. Returns the failure, if any.
fn check(engine: &Engine, html: &str, expected: &Path) -> Option<String> {
    let actual = engine
        .geometry_text_for_testing(html)
        .unwrap_or_else(|e| panic!("{}: layout failed: {e}", expected.display()));
    if std::env::var_os("UPDATE_GEOMETRY").is_some() {
        std::fs::create_dir_all(expected.parent().unwrap()).unwrap();
        std::fs::write(expected, &actual).unwrap();
        return None;
    }
    let recorded = match std::fs::read_to_string(expected) {
        Ok(text) => text.replace("\r\n", "\n"),
        Err(e) => {
            return Some(format!(
                "{}: {e} (run with UPDATE_GEOMETRY=1)",
                expected.display()
            ));
        }
    };
    if recorded == actual {
        return None;
    }
    let mut report = format!("{} differs:\n", expected.display());
    let (old, new): (Vec<&str>, Vec<&str>) = (recorded.lines().collect(), actual.lines().collect());
    let mut shown = 0;
    for k in 0..old.len().max(new.len()) {
        let (a, b) = (old.get(k), new.get(k));
        if a != b {
            report.push_str(&format!(
                "  line {}:\n    - {}\n    + {}\n",
                k + 1,
                a.unwrap_or(&"<none>"),
                b.unwrap_or(&"<none>")
            ));
            shown += 1;
            if shown == 5 {
                report.push_str("  ...\n");
                break;
            }
        }
    }
    Some(report)
}

fn html_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "html"))
        .collect();
    files.sort();
    files
}

fn geometry_path(dir: &Path, html: &Path) -> PathBuf {
    let stem = html.file_stem().unwrap().to_str().unwrap();
    dir.join(format!("{stem}.geometry.txt"))
}

#[test]
fn the_corpus_fixtures_keep_their_recorded_geometry() {
    let dir = corpus_dir();
    let files = html_files(&dir);
    assert!(!files.is_empty(), "no fixture in {}", dir.display());
    let failures: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let html = std::fs::read_to_string(path).unwrap();
            let (w, h) = page_of(&html, path);
            let engine = Engine::builder()
                .page_size(PageSize {
                    width: w * 0.75,
                    height: h * 0.75,
                })
                .margin(Margin::uniform(0.0))
                .system_fonts(false)
                .assets(assets())
                .build();
            check(&engine, &html, &geometry_path(&dir, path))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_vrt_layout_fixtures_keep_their_recorded_geometry() {
    let vrt = manifest_dir().join("../fulgur-vrt");
    let manifest: toml::Table =
        toml::from_str(&std::fs::read_to_string(vrt.join("manifest.toml")).unwrap()).unwrap();
    let default_size = manifest["defaults"]["page_size"].as_str().unwrap();
    let fixtures = manifest["fixture"].as_array().unwrap();
    let out = corpus_dir().join("vrt-layout");
    let files = html_files(&vrt.join("fixtures/layout"));
    assert!(!files.is_empty(), "no VRT layout fixture");
    let failures: Vec<String> = files
        .iter()
        .filter_map(|path| {
            let name = path.file_name().unwrap().to_str().unwrap();
            let entry = fixtures
                .iter()
                .find(|f| f["path"].as_str() == Some(&format!("layout/{name}")));
            let size = entry
                .and_then(|f| f.get("page_size"))
                .and_then(|v| v.as_str())
                .unwrap_or(default_size);
            let mut builder = Engine::builder()
                .page_size(PageSize::from_css_keyword(size).expect("page size"))
                .system_fonts(false)
                .assets(assets());
            if let Some(margin) = entry.and_then(|f| f.get("margin_pt")) {
                let margin = margin
                    .as_float()
                    .or_else(|| margin.as_integer().map(|m| m as f64));
                builder = builder.margin(Margin::uniform(margin.unwrap() as f32));
            }
            let html = std::fs::read_to_string(path).unwrap();
            check(&builder.build(), &html, &geometry_path(&out, path))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
