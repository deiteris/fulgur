//! Shared entry point for phase-specific WPT integration tests.
//!
//! A phase runner:
//!   1. Reads `target/wpt/css/<subdir>/` (WPT must be fetched first)
//!   2. Loads `crates/fulgur-wpt/expectations/<subdir>.txt` if present
//!   3. Runs every reftest through `harness::run_one` (panic-safe)
//!   4. Emits `target/wpt-report/<subdir>/` artifacts:
//!      - `report.json` (wptreport.json schema)
//!      - `regressions.json` (list of {test, observed_status, message})
//!      - `summary.md` (GitHub step summary block)
//!   5. Prints a one-line verdict summary to stderr
//!
//! A non-Blitz [`Backend`] reads its expectations from
//! `expectations/<backend>/` and writes its reports to
//! `target/wpt-report/<backend>/<label>/`. When the caller supplies the
//! Blitz expectations as a baseline, the run also writes
//! `baseline-diff.json` and a "vs Blitz" section in `summary.md`.
//!
//! Never panics on regressions. Nightly workflow inspects `regressions.json`
//! separately to decide whether to open an issue.

use crate::backend::Backend;
use crate::expectations::{Expectation, ExpectationFile, Verdict, judge};
use crate::harness::run_one;
use crate::reftest::collect_reftest_files;
use crate::report::{RunInfo, WptReport};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct Regression {
    pub test: String,
    pub observed: String,
    pub declared: String,
    pub message: Option<String>,
}

/// Where a backend's observed results differ from the Blitz expectations.
///
/// Only tests that both sides ran (neither SKIP nor missing from the
/// baseline) are compared.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BaselineDiff {
    /// Tests compared on both sides.
    pub compared: u32,
    /// Tests that pass on both sides.
    pub both_pass: u32,
    /// Tests that pass on this backend but are declared FAIL for Blitz.
    pub backend_only_pass: Vec<String>,
    /// Tests declared PASS for Blitz that fail on this backend.
    pub baseline_only_pass: Vec<String>,
}

impl BaselineDiff {
    fn record(&mut self, test: &str, baseline: Option<Expectation>, observed: Expectation) {
        match (baseline, observed) {
            (None | Some(Expectation::Skip), _) | (_, Expectation::Skip) => {}
            (Some(Expectation::Pass), Expectation::Pass) => {
                self.compared += 1;
                self.both_pass += 1;
            }
            (Some(Expectation::Fail), Expectation::Pass) => {
                self.compared += 1;
                self.backend_only_pass.push(test.to_string());
            }
            (Some(Expectation::Pass), Expectation::Fail) => {
                self.compared += 1;
                self.baseline_only_pass.push(test.to_string());
            }
            (Some(Expectation::Fail), Expectation::Fail) => self.compared += 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhaseOutcome {
    pub subdir: String,
    pub total: usize,
    pub pass: u32,
    pub fail: u32,
    pub skip: u32,
    pub regressions: Vec<Regression>,
    pub promotions: Vec<String>,
    pub unknown: Vec<String>,
    /// Comparison with the Blitz expectations, when a baseline was given.
    pub baseline_diff: Option<BaselineDiff>,
    pub report_dir: PathBuf,
}

/// Run every reftest under `css/<subdir>` and write artifacts under
/// `target/wpt-report/<subdir>/` (see [`Backend::report_root`]).
///
/// For a non-Blitz `backend`, `expectations/<subdir>.txt` (the Blitz
/// expectations) is compared against as the baseline.
///
/// Returns `Ok(None)` when prerequisites are missing (no WPT checkout,
/// no pdftocairo) so callers can skip silently on dev machines. Returns
/// `Ok(Some(..))` with the outcome otherwise.
pub fn run_phase(
    workspace_root: &Path,
    subdir: &str,
    dpi: u32,
    backend: Backend,
) -> Result<Option<PhaseOutcome>> {
    let wpt_root = workspace_root.join("target/wpt");
    let dir = wpt_root.join("css").join(subdir);
    if !dir.is_dir() {
        eprintln!(
            "skip: {} missing (run scripts/wpt/fetch.sh first)",
            dir.display()
        );
        return Ok(None);
    }
    if !poppler_available() {
        eprintln!("skip: pdftocairo not available on PATH");
        return Ok(None);
    }

    let expectations_root = workspace_root.join("crates/fulgur-wpt/expectations");
    let file_name = format!("{subdir}.txt");
    let expect_path = backend
        .expectations_dir(&expectations_root)
        .join(&file_name);
    let declared = if expect_path.exists() {
        ExpectationFile::load(&expect_path)
            .with_context(|| format!("load {}", expect_path.display()))?
    } else {
        eprintln!(
            "note: {} missing, treating every test as unknown",
            expect_path.display()
        );
        ExpectationFile::default()
    };

    let baseline = match backend.subdir() {
        None => None,
        Some(_) => load_baseline(&expectations_root.join(&file_name))?,
    };

    let tests = collect_reftest_files(&dir)
        .with_context(|| format!("collect reftest files in {}", dir.display()))?;

    execute_and_report(
        workspace_root,
        subdir,
        tests,
        declared,
        dpi,
        backend,
        baseline,
    )
    .map(Some)
}

/// Run exactly the reftests enumerated in `expectations_path` — keys of
/// the expectations file are interpreted as WPT-root-relative paths
/// (e.g. `css/css-page/foo.html`). Cross-subdir: paths may live under
/// any `css/<subdir>` as long as they exist in the fetched WPT snapshot.
///
/// `list_name` is used as the report subdirectory name
/// (`target/wpt-report/<list_name>/`, see [`Backend::report_root`]) and as
/// the stderr verdict tag (`wpt-<list_name>:`). Keep it filesystem-safe
/// (alphanumerics, `-`, `_`).
///
/// `baseline_path`, when given and present, is another backend's
/// expectations file (normally Blitz's) to compare the observed results
/// against.
///
/// Returns `Ok(None)` when prerequisites are missing (no WPT checkout,
/// no expectations file, or no pdftocairo).
pub fn run_list(
    workspace_root: &Path,
    list_name: &str,
    expectations_path: &Path,
    dpi: u32,
    backend: Backend,
    baseline_path: Option<&Path>,
) -> Result<Option<PhaseOutcome>> {
    // list_name ends up as a directory component under target/wpt-report/,
    // so reject anything that could escape (path separators, parent refs)
    // or otherwise produce surprising paths. A contract violation is a
    // programming error — fail loud rather than silently skip.
    if !is_safe_list_name(list_name) {
        anyhow::bail!(
            "invalid list_name `{list_name}`: must be [A-Za-z0-9_-]+ \
             (used as a directory component under target/wpt-report/)"
        );
    }

    let wpt_root = workspace_root.join("target/wpt");
    if !wpt_root.is_dir() {
        eprintln!(
            "skip: {} missing (run scripts/wpt/fetch.sh first)",
            wpt_root.display()
        );
        return Ok(None);
    }
    // Require a regular file so passing a directory doesn't sneak past
    // the skip precheck only to error later at load time.
    if !expectations_path.is_file() {
        eprintln!(
            "skip: {} missing (list has no expectations file)",
            expectations_path.display()
        );
        return Ok(None);
    }
    if !poppler_available() {
        eprintln!("skip: pdftocairo not available on PATH");
        return Ok(None);
    }

    let declared = ExpectationFile::load(expectations_path)
        .with_context(|| format!("load {}", expectations_path.display()))?;
    let baseline = match baseline_path {
        Some(p) => load_baseline(p)?,
        None => None,
    };

    // declared.paths() yields BTreeMap-sorted keys; wpt_root.join preserves
    // relative order, so tests is naturally sorted without an extra pass.
    let mut tests: Vec<PathBuf> = Vec::with_capacity(declared.len());
    let mut missing: Vec<String> = Vec::new();
    for rel in declared.paths() {
        let rel_path = Path::new(rel);
        // Reject entries that could escape wpt_root: absolute paths and
        // any component that walks upward (`..`). These are declared as
        // missing so subset.txt maintainers see a diagnostic instead of
        // a silent traversal.
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            missing.push(rel.to_string());
            continue;
        }
        let abs = wpt_root.join(rel_path);
        if abs.is_file() {
            tests.push(abs);
        } else {
            missing.push(rel.to_string());
        }
    }

    if !missing.is_empty() {
        eprintln!(
            "warning: {} test(s) declared in {} are missing from the WPT snapshot:",
            missing.len(),
            expectations_path.display()
        );
        for m in &missing {
            eprintln!("  - {m}");
        }
    }

    Ok(Some(execute_and_report(
        workspace_root,
        list_name,
        tests,
        declared,
        dpi,
        backend,
        baseline,
    )?))
}

/// Load a baseline expectations file, or `None` when it does not exist.
fn load_baseline(path: &Path) -> Result<Option<ExpectationFile>> {
    if !path.is_file() {
        eprintln!(
            "note: baseline {} missing, skipping the comparison",
            path.display()
        );
        return Ok(None);
    }
    ExpectationFile::load(path)
        .with_context(|| format!("load baseline {}", path.display()))
        .map(Some)
}

/// Iterate `tests`, judge each observed result against `declared`, and
/// write `report.json` / `regressions.json` / `summary.md` under
/// `target/wpt-report/<label>/` (see [`Backend::report_root`]). Shared by
/// `run_phase` (label = subdir) and `run_list` (label = list name). With a
/// `baseline`, also writes `baseline-diff.json`.
fn execute_and_report(
    workspace_root: &Path,
    label: &str,
    tests: Vec<PathBuf>,
    declared: ExpectationFile,
    dpi: u32,
    backend: Backend,
    baseline: Option<ExpectationFile>,
) -> Result<PhaseOutcome> {
    let wpt_root = workspace_root.join("target/wpt");
    let fonts_bundle = crate::fonts::load_fonts_dir(&wpt_root.join("fonts")).unwrap_or_else(|e| {
        log::warn!("fonts loader failed: {e}; proceeding without bundled fonts");
        fulgur::asset::AssetBundle::new()
    });
    let fonts_arg = if fonts_bundle.fonts.is_empty() {
        None
    } else {
        Some(&fonts_bundle)
    };
    let total = tests.len();

    let report_dir = backend.report_root(workspace_root).join(label);
    std::fs::create_dir_all(&report_dir)?;
    let run_root = backend.run_root(workspace_root);

    let mut report = WptReport::new(RunInfo {
        product: backend.product().into(),
        revision: env_revision(),
    });
    let mut baseline_diff = baseline.as_ref().map(|_| BaselineDiff::default());
    let mut regressions: Vec<Regression> = Vec::new();
    let mut promotions: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut verdicts: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut skip = 0u32;
    let start = Instant::now();

    for test in &tests {
        let rel = test
            .strip_prefix(&wpt_root)
            .unwrap_or(test)
            .to_string_lossy()
            .replace('\\', "/");
        let stem = test.file_stem().unwrap().to_string_lossy();
        let work = run_root.join(&*stem).join("work");
        let diff = run_root.join(&*stem).join("diff");

        let t0 = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_one(test, &work, &diff, dpi, fonts_arg, backend)
        }));
        let duration = t0.elapsed();
        let (observed, message) = match outcome {
            Ok(Ok(o)) => (o.observed, o.reason),
            Ok(Err(e)) => (Expectation::Fail, Some(format!("harness error: {e}"))),
            Err(p) => {
                let msg = panic_message(&p);
                (Expectation::Fail, Some(format!("harness panic: {msg}")))
            }
        };
        match observed {
            Expectation::Pass => pass += 1,
            Expectation::Fail => fail += 1,
            Expectation::Skip => skip += 1,
        }
        let is_harness_error = message
            .as_deref()
            .is_some_and(|m| m.starts_with("harness "));
        if is_harness_error {
            report.push_error(rel.clone(), message.clone().unwrap_or_default(), duration);
        } else {
            report.push(rel.clone(), observed, message.clone(), duration);
        }

        if let (Some(diff), Some(base)) = (baseline_diff.as_mut(), baseline.as_ref()) {
            diff.record(&rel, base.get(&rel), observed);
        }

        let declared_exp = declared.get(&rel);
        let verdict = judge(declared_exp, observed);
        let key = match verdict {
            Verdict::Ok => "ok",
            Verdict::Regression => "regression",
            Verdict::Promotion => "promotion",
            Verdict::Skipped => "skipped",
            Verdict::UnknownTest => "unknown",
        };
        *verdicts.entry(key).or_insert(0) += 1;

        match verdict {
            Verdict::Regression => {
                regressions.push(Regression {
                    test: rel.clone(),
                    observed: fmt_expectation(observed),
                    declared: declared_exp
                        .map(fmt_expectation)
                        .unwrap_or_else(|| "UNKNOWN".into()),
                    message,
                });
            }
            Verdict::Promotion => promotions.push(rel.clone()),
            Verdict::UnknownTest => unknown.push(rel.clone()),
            _ => {}
        }
    }
    let elapsed = start.elapsed();

    report.write(&report_dir.join("report.json"))?;
    std::fs::write(
        report_dir.join("regressions.json"),
        serde_json::to_string_pretty(&regressions)?,
    )?;
    let diff_path = report_dir.join("baseline-diff.json");
    match &baseline_diff {
        Some(diff) => std::fs::write(&diff_path, serde_json::to_string_pretty(diff)?)?,
        // Drop a comparison left by an earlier run so the artifacts match
        // this run's summary.
        None if diff_path.exists() => std::fs::remove_file(&diff_path)?,
        None => {}
    }
    let title = match backend.subdir() {
        None => label.to_string(),
        Some(sub) => format!("{label} ({sub})"),
    };
    write_summary(
        &report_dir,
        &title,
        total,
        pass,
        fail,
        skip,
        &regressions,
        &promotions,
        &unknown,
        baseline_diff.as_ref(),
        elapsed,
    )?;

    let tag = match backend.subdir() {
        None => format!("wpt-{label}"),
        Some(sub) => format!("wpt-{sub}-{label}"),
    };
    eprintln!(
        "{tag}: total={total} pass={pass} fail={fail} skip={skip} regressions={} promotions={} unknown={} ({:.1}s)",
        regressions.len(),
        promotions.len(),
        unknown.len(),
        elapsed.as_secs_f64(),
    );

    Ok(PhaseOutcome {
        subdir: label.to_string(),
        total,
        pass,
        fail,
        skip,
        regressions,
        promotions,
        unknown,
        baseline_diff,
        report_dir,
    })
}

/// list_name becomes a directory component under `target/wpt-report/` and
/// part of the stderr verdict tag. Restrict it to a safe whitelist so no
/// caller can produce path traversal or shell-surprising output.
fn is_safe_list_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn poppler_available() -> bool {
    std::process::Command::new("pdftocairo")
        .arg("-v")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn fmt_expectation(e: Expectation) -> String {
    match e {
        Expectation::Pass => "PASS".into(),
        Expectation::Fail => "FAIL".into(),
        Expectation::Skip => "SKIP".into(),
    }
}

fn env_revision() -> String {
    std::env::var("GITHUB_SHA")
        .or_else(|_| std::env::var("FULGUR_REVISION"))
        .unwrap_or_else(|_| "unknown".into())
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    dir: &Path,
    label: &str,
    total: usize,
    pass: u32,
    fail: u32,
    skip: u32,
    regressions: &[Regression],
    promotions: &[String],
    unknown: &[String],
    baseline_diff: Option<&BaselineDiff>,
    elapsed: Duration,
) -> Result<()> {
    let mut f = std::fs::File::create(dir.join("summary.md"))?;
    writeln!(f, "### WPT {label}")?;
    writeln!(f)?;
    writeln!(
        f,
        "- total: **{total}** ({elapsed_s:.1}s)",
        elapsed_s = elapsed.as_secs_f64(),
    )?;
    let pass_pct = if total == 0 {
        0.0
    } else {
        pass as f64 * 100.0 / total as f64
    };
    writeln!(f, "- PASS: **{pass}** ({pass_pct:.1}%)")?;
    writeln!(f, "- FAIL: {fail}")?;
    writeln!(f, "- SKIP: {skip}")?;
    writeln!(f, "- regressions: {}", regressions.len())?;
    writeln!(f, "- promotion candidates: {}", promotions.len())?;
    writeln!(f, "- unknown (no expectation entry): {}", unknown.len())?;

    if !regressions.is_empty() {
        writeln!(f, "\n#### Regressions\n")?;
        for r in regressions {
            let msg = r.message.as_deref().unwrap_or("");
            writeln!(
                f,
                "- `{}` declared={} observed={} — {msg}",
                r.test, r.declared, r.observed
            )?;
        }
    }
    if !promotions.is_empty() {
        writeln!(
            f,
            "\n#### Promotion candidates ({} tests now pass)\n",
            promotions.len()
        )?;
        for p in promotions.iter().take(30) {
            writeln!(f, "- `{p}`")?;
        }
        if promotions.len() > 30 {
            writeln!(f, "- ... (+{})", promotions.len() - 30)?;
        }
    }
    if let Some(diff) = baseline_diff {
        writeln!(f, "\n#### vs Blitz (declared expectations)\n")?;
        writeln!(f, "- compared: {}", diff.compared)?;
        writeln!(f, "- PASS on both: {}", diff.both_pass)?;
        let (here, blitz) = (&diff.backend_only_pass, &diff.baseline_only_pass);
        writeln!(f, "- PASS here, FAIL on Blitz: {}", here.len())?;
        writeln!(f, "- FAIL here, PASS on Blitz: {}", blitz.len())?;
        write_test_list(&mut f, "PASS here, FAIL on Blitz", here)?;
        write_test_list(&mut f, "FAIL here, PASS on Blitz", blitz)?;
    }
    Ok(())
}

/// Write a collapsed list of at most 30 test paths.
fn write_test_list(f: &mut impl Write, heading: &str, tests: &[String]) -> Result<()> {
    if tests.is_empty() {
        return Ok(());
    }
    writeln!(f, "\n<details><summary>{heading}</summary>\n")?;
    for t in tests.iter().take(30) {
        writeln!(f, "- `{t}`")?;
    }
    if tests.len() > 30 {
        writeln!(f, "- ... (+{})", tests.len() - 30)?;
    }
    writeln!(f, "\n</details>")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn run_list_returns_none_when_wpt_root_missing() {
        let ws = tempdir().unwrap();
        let expectations = ws.path().join("nonexistent.txt");
        let out = run_list(
            ws.path(),
            "missing-wpt",
            &expectations,
            96,
            Backend::Blitz,
            None,
        )
        .unwrap();
        assert!(out.is_none(), "expected Ok(None) when target/wpt is absent");
    }

    #[test]
    fn execute_and_report_with_empty_tests_emits_empty_reports() {
        let ws = tempdir().unwrap();
        let outcome = execute_and_report(
            ws.path(),
            "empty-label",
            Vec::new(),
            ExpectationFile::default(),
            96,
            Backend::Blitz,
            None,
        )
        .unwrap();

        assert_eq!(outcome.total, 0);
        assert_eq!(outcome.pass, 0);
        assert_eq!(outcome.fail, 0);
        assert_eq!(outcome.skip, 0);
        assert!(outcome.regressions.is_empty());
        assert!(outcome.promotions.is_empty());
        assert!(outcome.unknown.is_empty());
        assert_eq!(outcome.subdir, "empty-label");

        let report_dir = ws.path().join("target/wpt-report/empty-label");
        assert!(report_dir.join("report.json").is_file());
        assert!(report_dir.join("regressions.json").is_file());
        assert!(report_dir.join("summary.md").is_file());

        let summary = std::fs::read_to_string(report_dir.join("summary.md")).unwrap();
        assert!(summary.contains("### WPT empty-label"));
        assert!(summary.contains("- total: **0**"));
    }

    #[test]
    fn run_list_rejects_unsafe_list_name() {
        let ws = tempdir().unwrap();
        let expectations = ws.path().join("nonexistent.txt");
        let err = run_list(
            ws.path(),
            "../escape",
            &expectations,
            96,
            Backend::Blitz,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("invalid list_name"),
            "expected bail on traversal-unsafe list_name, got: {err}"
        );
    }

    #[test]
    fn run_list_returns_none_when_expectations_is_a_directory() {
        let ws = tempdir().unwrap();
        // Satisfy the wpt_root guard so control reaches the expectations check.
        std::fs::create_dir_all(ws.path().join("target/wpt")).unwrap();
        let fake_expectations = ws.path().join("actually-a-dir");
        std::fs::create_dir_all(&fake_expectations).unwrap();
        let out = run_list(
            ws.path(),
            "dirtest",
            &fake_expectations,
            96,
            Backend::Blitz,
            None,
        )
        .unwrap();
        assert!(
            out.is_none(),
            "expected Ok(None) when expectations_path is a directory (is_file() == false)"
        );
    }

    #[test]
    fn baseline_diff_compares_only_tests_run_on_both_sides() {
        let mut diff = BaselineDiff::default();
        diff.record("a", Some(Expectation::Pass), Expectation::Pass);
        diff.record("b", Some(Expectation::Fail), Expectation::Pass);
        diff.record("c", Some(Expectation::Pass), Expectation::Fail);
        diff.record("d", Some(Expectation::Fail), Expectation::Fail);
        diff.record("e", Some(Expectation::Skip), Expectation::Pass);
        diff.record("f", Some(Expectation::Pass), Expectation::Skip);
        diff.record("g", None, Expectation::Pass);
        assert_eq!(diff.compared, 4);
        assert_eq!(diff.both_pass, 1);
        assert_eq!(diff.backend_only_pass, vec!["b".to_string()]);
        assert_eq!(diff.baseline_only_pass, vec!["c".to_string()]);
    }

    #[test]
    fn summary_lists_baseline_differences() {
        let dir = tempdir().unwrap();
        let diff = BaselineDiff {
            compared: 40,
            both_pass: 3,
            backend_only_pass: vec!["css/x/only-here.html".into()],
            baseline_only_pass: (0..31).map(|i| format!("css/x/t{i}.html")).collect(),
        };
        write_summary(
            dir.path(),
            "lbl (raikiri)",
            40,
            4,
            36,
            0,
            &[],
            &[],
            &[],
            Some(&diff),
            Duration::ZERO,
        )
        .unwrap();
        let summary = std::fs::read_to_string(dir.path().join("summary.md")).unwrap();
        assert!(summary.contains("### WPT lbl (raikiri)"), "{summary}");
        assert!(summary.contains("#### vs Blitz (declared expectations)"));
        assert!(summary.contains("- PASS on both: 3"));
        assert!(summary.contains("- PASS here, FAIL on Blitz: 1"));
        assert!(summary.contains("- FAIL here, PASS on Blitz: 31"));
        assert!(summary.contains("- `css/x/only-here.html`"));
        assert!(summary.contains("- `css/x/t29.html`"));
        assert!(!summary.contains("- `css/x/t30.html`"));
        assert!(summary.contains("- ... (+1)"));
    }

    #[test]
    fn load_baseline_is_none_when_missing() {
        let dir = tempdir().unwrap();
        assert!(
            load_baseline(&dir.path().join("none.txt"))
                .unwrap()
                .is_none()
        );
        let p = dir.path().join("base.txt");
        std::fs::write(&p, "PASS  css/a.html\n").unwrap();
        let base = load_baseline(&p).unwrap().unwrap();
        assert_eq!(base.get("css/a.html"), Some(Expectation::Pass));
    }

    #[test]
    fn raikiri_report_goes_to_backend_subdirectory() {
        let ws = tempdir().unwrap();
        let outcome = execute_and_report(
            ws.path(),
            "lbl",
            Vec::new(),
            ExpectationFile::default(),
            96,
            Backend::Raikiri,
            Some(ExpectationFile::default()),
        )
        .unwrap();
        let report_dir = ws.path().join("target/wpt-report/raikiri/lbl");
        assert_eq!(outcome.report_dir, report_dir);
        assert_eq!(outcome.baseline_diff.unwrap().compared, 0);
        assert!(report_dir.join("baseline-diff.json").is_file());

        // A rerun without a baseline removes the stale comparison.
        let outcome = execute_and_report(
            ws.path(),
            "lbl",
            Vec::new(),
            ExpectationFile::default(),
            96,
            Backend::Raikiri,
            None,
        )
        .unwrap();
        assert!(outcome.baseline_diff.is_none());
        assert!(!report_dir.join("baseline-diff.json").exists());
        let report = std::fs::read_to_string(report_dir.join("report.json")).unwrap();
        assert!(report.contains("\"fulgur-raikiri\""), "{report}");
    }

    #[test]
    fn execute_and_report_compares_rendered_tests_with_baseline() {
        if !poppler_available() {
            eprintln!("skip: no pdftocairo");
            return;
        }
        let ws = tempdir().unwrap();
        let dir = ws.path().join("target/wpt/css/t");
        std::fs::create_dir_all(&dir).unwrap();
        let body = r#"<body style="margin:0"><div style="width:20px;height:20px;background:green"></div></body>"#;
        std::fs::write(
            dir.join("a.html"),
            format!(r#"<!DOCTYPE html><link rel="match" href="a-ref.html">{body}"#),
        )
        .unwrap();
        std::fs::write(dir.join("a-ref.html"), format!("<!DOCTYPE html>{body}")).unwrap();
        let declared = ExpectationFile::parse("PASS  css/t/a.html\n").unwrap();
        let baseline = ExpectationFile::parse("FAIL  css/t/a.html\n").unwrap();

        let outcome = execute_and_report(
            ws.path(),
            "t",
            vec![dir.join("a.html")],
            declared,
            96,
            Backend::Blitz,
            Some(baseline),
        )
        .unwrap();
        assert_eq!(outcome.pass, 1, "{outcome:?}");
        let diff = outcome.baseline_diff.unwrap();
        assert_eq!(diff.backend_only_pass, vec!["css/t/a.html".to_string()]);
        assert!(ws.path().join("target/wpt-run/a/work").is_dir());
    }

    #[test]
    fn run_list_and_phase_load_the_blitz_baseline() {
        let ws = tempdir().unwrap();
        std::fs::create_dir_all(ws.path().join("target/wpt/css/empty")).unwrap();
        let exp_root = ws.path().join("crates/fulgur-wpt/expectations");
        std::fs::create_dir_all(exp_root.join("raikiri")).unwrap();
        std::fs::write(exp_root.join("empty.txt"), "").unwrap();
        let list = exp_root.join("raikiri/list.txt");
        std::fs::write(&list, "").unwrap();
        let base = exp_root.join("list.txt");
        std::fs::write(&base, "").unwrap();

        // Each returns None when pdftocairo is not installed.
        let ws = ws.path();
        let with_base = run_list(ws, "list", &list, 96, Backend::Raikiri, Some(&base));
        assert!(with_base.unwrap().is_none_or(|o| o.baseline_diff.is_some()));
        let no_base = run_list(ws, "list", &list, 96, Backend::Raikiri, None);
        assert!(no_base.unwrap().is_none_or(|o| o.baseline_diff.is_none()));
        let phase = run_phase(ws, "empty", 96, Backend::Raikiri);
        assert!(phase.unwrap().is_none_or(|o| o.baseline_diff.is_some()));
        let phase = run_phase(ws, "empty", 96, Backend::Blitz);
        assert!(phase.unwrap().is_none_or(|o| o.baseline_diff.is_none()));
    }

    #[test]
    fn is_safe_list_name_accepts_valid_names() {
        assert!(is_safe_list_name("bugs"));
        assert!(is_safe_list_name("multicol-1"));
        assert!(is_safe_list_name("multicol_1"));
        assert!(is_safe_list_name("abc123"));
    }

    #[test]
    fn is_safe_list_name_rejects_unsafe_names() {
        assert!(!is_safe_list_name(""));
        assert!(!is_safe_list_name("../escape"));
        assert!(!is_safe_list_name("foo/bar"));
        assert!(!is_safe_list_name("foo.bar"));
        assert!(!is_safe_list_name("hello world"));
    }
}
