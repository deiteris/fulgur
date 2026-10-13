//! A normal render writes nothing to stdout or stderr, even for a link to a
//! missing anchor.
//!
//! The renderer builds small HTML documents of its own for page-margin boxes.
//! An HTML parse error in one of them is printed by the HTML parser straight
//! to stdout, so the test renders in a child process and reads its output.

use fulgur::asset::AssetBundle;
use fulgur::engine::Engine;

const CHILD_ENV: &str = "FULGUR_QUIET_RENDER_CHILD";

#[test]
fn rendering_margin_boxes_prints_nothing() {
    if std::env::var_os(CHILD_ENV).is_some() {
        let mut assets = AssetBundle::new();
        assets.add_css(
            "@page { @top-center { content: \"Head\" } \
             @bottom-right { content: counter(page) \" of \" counter(pages) } }",
        );
        let engine = Engine::builder().assets(assets).build();
        engine
            .render("<!DOCTYPE html><html><head><title>t</title></head><body><h1>One</h1><p>Body <a href=\"#nowhere\">text</a>.</p></body></html>")
            .expect("render");
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "rendering_margin_boxes_prints_nothing",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stdout.contains("ERROR"), "stdout: {stdout}");
    assert!(stderr.is_empty(), "stderr: {stderr}");
}
