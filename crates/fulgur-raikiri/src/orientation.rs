//! Page rotation for display: the `page-orientation` descriptor of `@page`
//! (CSS Paged Media 3 §3.5).
//!
//! The page is laid out and painted upright; the descriptor only asks the
//! viewer to turn it by a quarter, which PDF expresses with the `/Rotate`
//! entry of the page dictionary (ISO 32000-2 §7.7.3.3). Krilla does not
//! write that entry, so it is added to the finished PDF.

use flpdf::ObjectHandle;
use fulgur_core::{Error, Result};
use raikiri_html::{Page, PageRotation};

/// The pages to turn, as zero-based page index and clockwise degrees.
#[derive(Debug, Default)]
pub(crate) struct Rotations {
    /// Pages seen by [`Self::add`] so far.
    pages: usize,
    turned: Vec<(usize, i64)>,
}

impl Rotations {
    /// Record the rotation of `page`, the next page of the PDF, if it is
    /// not upright.
    pub(crate) fn add(&mut self, page: &Page<'_>) {
        if let Some(degrees) = degrees(page.page_style().page_orientation()) {
            self.turned.push((self.pages, degrees));
        }
        self.pages += 1;
    }

    /// Set the `/Rotate` of every recorded page of `pdf`. A PDF with no
    /// rotated page is returned unchanged.
    pub(crate) fn apply(self, pdf: Vec<u8>) -> Result<Vec<u8>> {
        if self.turned.is_empty() {
            return Ok(pdf);
        }
        crate::merge::rewrite(pdf, "rotating pages", |document| {
            let error =
                |error: flpdf::Error| Error::PdfGeneration(format!("rotating pages: {error}"));
            let pages = flpdf::pages::page_refs(document).map_err(error)?;
            for (index, degrees) in self.turned {
                if let Some(&page) = pages.get(index) {
                    document
                        .get_object_handle(page)
                        .replace_key(b"/Rotate", ObjectHandle::integer(degrees))
                        .map_err(error)?;
                }
            }
            Ok(())
        })
    }
}

/// The clockwise `/Rotate` of `rotation`, or `None` when the page stays
/// upright. `rotate-left` turns the page counter-clockwise.
fn degrees(rotation: Option<PageRotation>) -> Option<i64> {
    match rotation? {
        PageRotation::RotateLeft => Some(270),
        PageRotation::RotateRight => Some(90),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_left_and_right_turn_the_page_by_a_quarter() {
        assert_eq!(degrees(Some(PageRotation::RotateLeft)), Some(270));
        assert_eq!(degrees(Some(PageRotation::RotateRight)), Some(90));
        assert_eq!(degrees(Some(PageRotation::Upright)), None);
        assert_eq!(degrees(None), None);
    }

    #[test]
    fn no_rotation_leaves_the_pdf_untouched() {
        let pdf = b"not a pdf".to_vec();
        assert_eq!(Rotations::default().apply(pdf.clone()).unwrap(), pdf);
    }

    #[test]
    fn an_unreadable_pdf_is_an_error() {
        let rotations = Rotations {
            pages: 1,
            turned: vec![(0, 90)],
        };
        assert!(rotations.apply(b"not a pdf".to_vec()).is_err());
    }
}
