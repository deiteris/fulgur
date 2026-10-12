//! Read and patch the outline counts of a finished PDF in place.
//!
//! Krilla writes every object uncompressed and indexes them with one classic
//! cross-reference table, so the outline dictionaries can be found through
//! the table and read without parsing the rest of the file. A changed entry
//! is written over the bytes of the old one, including the whitespace around
//! it, which Krilla's indented dictionaries leave enough of for any count.
//! Only when a new entry still does not fit do the bytes after it move, once,
//! with the table offsets shifted to match.

use std::collections::HashMap;
use std::ops::Range;

/// Nesting limit for arrays and dictionaries inside an outline dictionary.
const MAX_DEPTH: usize = 32;

/// A value of a dictionary entry, as far as the outline counts need it.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Value {
    Integer(i64),
    Reference(u32),
    Other,
}

/// A dictionary entry and the bytes it can be rewritten in: from the end of
/// the previous token to the start of the next, so with the whitespace on
/// both sides. Spans of neighbouring entries overlap; rewrite at most one
/// entry of a dictionary.
pub(super) struct Entry<'a> {
    key: &'a [u8],
    pub(super) value: Value,
    pub(super) span: Range<usize>,
}

pub(super) type Dictionary<'a> = Vec<Entry<'a>>;

pub(super) fn entry<'d, 'a>(dictionary: &'d Dictionary<'a>, key: &[u8]) -> Option<&'d Entry<'a>> {
    dictionary.iter().find(|entry| entry.key == key)
}

pub(super) fn get<'d>(dictionary: &'d Dictionary<'_>, key: &[u8]) -> Option<&'d Value> {
    entry(dictionary, key).map(|entry| &entry.value)
}

/// The uncompressed objects of a PDF with one classic cross-reference table.
pub(super) struct RawPdf<'a> {
    bytes: &'a [u8],
    offsets: HashMap<u32, usize>,
    /// Offset of the `xref` keyword.
    table: usize,
    /// Offset of the `startxref` value.
    startxref: usize,
    pub(super) root: u32,
}

impl<'a> RawPdf<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Option<Self> {
        let (startxref, table) = startxref(bytes)?;
        if !bytes.get(table..)?.starts_with(b"xref") {
            return None;
        }
        let mut lexer = Lexer::new(bytes, table + b"xref".len());
        let mut offsets = HashMap::new();
        loop {
            match lexer.token()? {
                Token::Keyword(b"trailer") => break,
                Token::Number(first) => {
                    let first: u32 = parse(first)?;
                    let Token::Number(count) = lexer.token()? else {
                        return None;
                    };
                    let count: u32 = parse(count)?;
                    lexer.skip();
                    for number in first..first.checked_add(count)? {
                        let entry = bytes.get(lexer.at..lexer.at + 20)?;
                        if entry[17] == b'n' {
                            offsets.insert(number, parse(&entry[..10])?);
                        }
                        lexer.at += 20;
                    }
                }
                _ => return None,
            }
        }
        let trailer = lexer.dictionary(0)?;
        // An incremental update would leave objects in an earlier table.
        if get(&trailer, b"Prev").is_some() {
            return None;
        }
        let Some(Value::Reference(root)) = get(&trailer, b"Root") else {
            return None;
        };
        Some(Self {
            bytes,
            offsets,
            table,
            startxref,
            root: *root,
        })
    }

    /// The dictionary of object `number`.
    pub(super) fn dictionary(&self, number: u32) -> Option<Dictionary<'a>> {
        let offset = *self.offsets.get(&number)?;
        if offset >= self.table {
            return None;
        }
        let mut lexer = Lexer::new(self.bytes, offset);
        let Token::Number(found) = lexer.token()? else {
            return None;
        };
        if parse::<u32>(found)? != number
            || !matches!(lexer.token()?, Token::Number(_))
            || lexer.token()? != Token::Keyword(b"obj")
        {
            return None;
        }
        lexer.dictionary(0)
    }
}

/// Replace each [`Entry::span`] in `edits` with its new entry and fix the
/// cross-reference table of `layout`, which was read from the same bytes.
pub(super) fn patch(
    mut pdf: Vec<u8>,
    layout: Layout,
    mut edits: Vec<(Range<usize>, Vec<u8>)>,
) -> Option<Vec<u8>> {
    edits.sort_unstable_by_key(|(span, _)| span.start);
    let mut grown = Vec::new();
    let mut previous = 0;
    for (span, text) in edits {
        if span.start < previous || span.end > layout.table {
            return None;
        }
        previous = span.end;
        if text.len() <= span.len() {
            // Pad with spaces, which separate tokens like the original
            // whitespace did.
            pdf[span.clone()].fill(b' ');
            pdf[span.start..span.start + text.len()].copy_from_slice(&text);
        } else {
            grown.push((span, text));
        }
    }
    // `growth[i]` is how far the bytes after `grown[i]` move.
    let growth: Vec<usize> = grown
        .iter()
        .scan(0, |sum, (span, text)| {
            *sum += text.len() - span.len();
            Some(*sum)
        })
        .collect();
    let Some(&total) = growth.last() else {
        return Some(pdf);
    };

    // Move every byte after an edit right by the growth of the edits before
    // it, last segment first so that nothing is overwritten before it moves.
    let length = pdf.len();
    pdf.resize(length + total, 0);
    let mut end = length;
    for (index, (span, text)) in grown.iter().enumerate().rev() {
        let shift = growth[index];
        pdf.copy_within(span.end..end, span.end + shift);
        let start = span.start + shift - (text.len() - span.len());
        pdf[start..start + text.len()].copy_from_slice(text);
        end = span.start;
    }

    let moved = |offset: usize| {
        let index = grown.partition_point(|(span, _)| span.start < offset);
        offset + index.checked_sub(1).map_or(0, |index| growth[index])
    };
    let mut lexer = Lexer::new(&pdf, layout.table + total + b"xref".len());
    let mut patches = Vec::new();
    loop {
        match lexer.token()? {
            Token::Keyword(b"trailer") => break,
            Token::Number(_) => {
                let Token::Number(count) = lexer.token()? else {
                    return None;
                };
                let count: usize = parse(count)?;
                lexer.skip();
                for _ in 0..count {
                    let entry = lexer.bytes.get(lexer.at..lexer.at + 20)?;
                    if entry[17] == b'n' {
                        patches.push((lexer.at, moved(parse(&entry[..10])?)));
                    }
                    lexer.at += 20;
                }
            }
            _ => return None,
        }
    }
    for (at, offset) in patches {
        let text = format!("{offset:010}");
        if text.len() != 10 {
            return None;
        }
        pdf[at..at + 10].copy_from_slice(text.as_bytes());
    }
    let at = layout.startxref + total;
    let digits = pdf[at..].iter().take_while(|b| b.is_ascii_digit()).count();
    pdf.splice(
        at..at + digits,
        (layout.table + total).to_string().into_bytes(),
    );
    Some(pdf)
}

/// Where the cross-reference table and the `startxref` value of a PDF are.
pub(super) struct Layout {
    table: usize,
    startxref: usize,
}

impl RawPdf<'_> {
    pub(super) fn layout(&self) -> Layout {
        Layout {
            table: self.table,
            startxref: self.startxref,
        }
    }
}

/// The offset of the `startxref` value and the value itself.
fn startxref(bytes: &[u8]) -> Option<(usize, usize)> {
    const KEYWORD: &[u8] = b"startxref";
    let tail = bytes.len().saturating_sub(1024);
    let keyword = tail
        + bytes[tail..]
            .windows(KEYWORD.len())
            .rposition(|window| window == KEYWORD)?;
    let mut lexer = Lexer::new(bytes, keyword + KEYWORD.len());
    lexer.skip();
    let at = lexer.at;
    let Token::Number(value) = lexer.token()? else {
        return None;
    };
    Some((at, parse(value)?))
}

fn parse<T: std::str::FromStr>(digits: &[u8]) -> Option<T> {
    std::str::from_utf8(digits).ok()?.parse().ok()
}

#[derive(Debug, PartialEq)]
enum Token<'a> {
    DictionaryStart,
    DictionaryEnd,
    ArrayStart,
    ArrayEnd,
    Name(&'a [u8]),
    Number(&'a [u8]),
    String,
    Keyword(&'a [u8]),
}

/// A tokenizer for the PDF object syntax (ISO 32000-2 §7.2, §7.3).
struct Lexer<'a> {
    bytes: &'a [u8],
    at: usize,
}

fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b'\0' | b'\t' | b'\n' | b'\x0c' | b'\r' | b' ')
}

fn is_delimiter(byte: u8) -> bool {
    matches!(
        byte,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_regular(byte: u8) -> bool {
    !is_whitespace(byte) && !is_delimiter(byte)
}

impl<'a> Lexer<'a> {
    fn new(bytes: &'a [u8], at: usize) -> Self {
        Self { bytes, at }
    }

    /// Skip whitespace and comments.
    fn skip(&mut self) {
        while let Some(&byte) = self.bytes.get(self.at) {
            if is_whitespace(byte) {
                self.at += 1;
            } else if byte == b'%' {
                while self
                    .bytes
                    .get(self.at)
                    .is_some_and(|byte| !matches!(byte, b'\n' | b'\r'))
                {
                    self.at += 1;
                }
            } else {
                break;
            }
        }
    }

    fn regular(&mut self) -> &'a [u8] {
        let start = self.at;
        while self.bytes.get(self.at).copied().is_some_and(is_regular) {
            self.at += 1;
        }
        &self.bytes[start..self.at]
    }

    fn token(&mut self) -> Option<Token<'a>> {
        self.skip();
        let byte = *self.bytes.get(self.at)?;
        let next = self.bytes.get(self.at + 1).copied();
        Some(match byte {
            b'<' if next == Some(b'<') => {
                self.at += 2;
                Token::DictionaryStart
            }
            b'>' if next == Some(b'>') => {
                self.at += 2;
                Token::DictionaryEnd
            }
            b'[' => {
                self.at += 1;
                Token::ArrayStart
            }
            b']' => {
                self.at += 1;
                Token::ArrayEnd
            }
            b'/' => {
                self.at += 1;
                Token::Name(self.regular())
            }
            b'<' => {
                let end = self.bytes[self.at..].iter().position(|&b| b == b'>')?;
                self.at += end + 1;
                Token::String
            }
            b'(' => {
                self.literal_string()?;
                Token::String
            }
            b'0'..=b'9' | b'+' | b'-' | b'.' => Token::Number(self.regular()),
            _ if is_regular(byte) => Token::Keyword(self.regular()),
            _ => return None,
        })
    }

    /// Skip a literal string, with its balanced parentheses and escapes.
    fn literal_string(&mut self) -> Option<()> {
        let mut depth = 0usize;
        loop {
            let byte = *self.bytes.get(self.at)?;
            self.at += 1;
            match byte {
                b'\\' => self.at += 1,
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(());
                    }
                }
                _ => {}
            }
        }
    }

    fn dictionary(&mut self, depth: usize) -> Option<Dictionary<'a>> {
        if self.token()? != Token::DictionaryStart {
            return None;
        }
        self.dictionary_entries(depth)
    }

    fn dictionary_entries(&mut self, depth: usize) -> Option<Dictionary<'a>> {
        let mut entries = Vec::new();
        loop {
            let start = self.at;
            match self.token()? {
                Token::DictionaryEnd => return Some(entries),
                Token::Name(key) => {
                    let token = self.token()?;
                    let value = self.value(token, depth)?;
                    // Whitespace between two entries belongs to both spans.
                    let mut next = Lexer::new(self.bytes, self.at);
                    next.skip();
                    entries.push(Entry {
                        key,
                        value,
                        span: start..next.at,
                    });
                }
                _ => return None,
            }
        }
    }

    fn value(&mut self, token: Token<'a>, depth: usize) -> Option<Value> {
        if depth > MAX_DEPTH {
            return None;
        }
        Some(match token {
            Token::Number(number) => {
                // `N G R` is a reference; otherwise rewind past the lookahead.
                let mut lookahead = Lexer::new(self.bytes, self.at);
                if let (Some(Token::Number(_)), Some(Token::Keyword(b"R"))) =
                    (lookahead.token(), lookahead.token())
                {
                    self.at = lookahead.at;
                    Value::Reference(parse(number)?)
                } else {
                    parse(number).map_or(Value::Other, Value::Integer)
                }
            }
            Token::DictionaryStart => {
                self.dictionary_entries(depth + 1)?;
                Value::Other
            }
            Token::ArrayStart => {
                loop {
                    match self.token()? {
                        Token::ArrayEnd => break,
                        token => {
                            self.value(token, depth + 1)?;
                        }
                    }
                }
                Value::Other
            }
            Token::Name(_) | Token::String | Token::Keyword(_) => Value::Other,
            Token::DictionaryEnd | Token::ArrayEnd => return None,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn dictionary_skips_strings_that_look_like_keys() {
        let bytes = b"<< /Title (a /Count 9 \\) (b) /Next 1 0 R) /Count -2  /A [1 [2] <</B 3>>] /First 4 0 R\n>>";
        let dictionary = Lexer::new(bytes, 0).dictionary(0).unwrap();
        assert_eq!(get(&dictionary, b"Next"), None);
        assert_eq!(get(&dictionary, b"Count"), Some(&Value::Integer(-2)));
        let first = entry(&dictionary, b"First").unwrap();
        assert_eq!(first.value, Value::Reference(4));
        assert_eq!(&bytes[first.span.clone()], b" /First 4 0 R\n");
        let count = entry(&dictionary, b"Count").unwrap();
        assert_eq!(&bytes[count.span.clone()], b" /Count -2  ");
    }

    /// A PDF with compact dictionaries, so a longer count cannot reuse
    /// whitespace and the objects after it have to move.
    fn compact_pdf() -> Vec<u8> {
        pdf_with(
            &[
                "<</Type/Catalog/Outlines 2 0 R>>",
                "<</First 3 0 R/Last 3 0 R/Count 1>>",
                "<</Title(x)/Parent 2 0 R/First 4 0 R/Last 4 0 R/Count -1>>",
                "<</Title(y)/Parent 3 0 R>>",
            ],
            "/Root 1 0 R",
        )
    }

    /// A PDF of `objects`, numbered from 1, with `trailer` entries.
    pub(in crate::bookmarks) fn pdf_with(objects: &[&str], trailer: &str) -> Vec<u8> {
        let mut pdf = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend(format!("{} 0 obj\n{object}\nendobj\n", index + 1).bytes());
        }
        let table = pdf.len();
        pdf.extend(format!("xref\n0 {}\n0000000000 65535 f\r\n", objects.len() + 1).bytes());
        for offset in offsets {
            pdf.extend(format!("{offset:010} 00000 n\r\n").bytes());
        }
        pdf.extend(
            format!(
                "trailer\n<</Size {}{trailer}>>\nstartxref\n{table}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        pdf
    }

    #[test]
    fn patch_moves_objects_after_a_longer_entry() {
        let pdf = compact_pdf();
        let raw = RawPdf::new(&pdf).unwrap();
        let root = entry(&raw.dictionary(2).unwrap(), b"Count")
            .unwrap()
            .span
            .clone();
        let item = entry(&raw.dictionary(3).unwrap(), b"Count")
            .unwrap()
            .span
            .clone();
        let layout = raw.layout();
        let edits = vec![
            (item, b"/Count 1234567".to_vec()),
            (root, b"/Count 1234568".to_vec()),
        ];
        let pdf = patch(pdf, layout, edits).unwrap();

        let raw = RawPdf::new(&pdf).unwrap();
        for number in 1..=4 {
            let offset = raw.offsets[&number];
            assert!(pdf[offset..].starts_with(format!("{number} 0 obj").as_bytes()));
        }
        assert_eq!(
            get(&raw.dictionary(2).unwrap(), b"Count"),
            Some(&Value::Integer(1234568))
        );
        assert_eq!(
            get(&raw.dictionary(3).unwrap(), b"Count"),
            Some(&Value::Integer(1234567))
        );
        assert_eq!(
            get(&raw.dictionary(3).unwrap(), b"First"),
            Some(&Value::Reference(4))
        );
        assert_eq!(
            get(&raw.dictionary(4).unwrap(), b"Parent"),
            Some(&Value::Reference(3))
        );
    }

    #[test]
    fn patch_pads_a_shorter_entry_in_place() {
        let pdf = compact_pdf();
        let length = pdf.len();
        let raw = RawPdf::new(&pdf).unwrap();
        let item = entry(&raw.dictionary(3).unwrap(), b"Count")
            .unwrap()
            .span
            .clone();
        let layout = raw.layout();
        let pdf = patch(pdf, layout, vec![(item, b"/Count 1".to_vec())]).unwrap();
        assert_eq!(pdf.len(), length);
        let raw = RawPdf::new(&pdf).unwrap();
        assert_eq!(
            get(&raw.dictionary(3).unwrap(), b"Count"),
            Some(&Value::Integer(1))
        );
    }

    #[test]
    fn incremental_updates_are_rejected() {
        let pdf = compact_pdf();
        let text = String::from_utf8(pdf)
            .unwrap()
            .replace("/Size 5", "/Size 5/Prev 9");
        assert!(RawPdf::new(text.as_bytes()).is_none());
    }

    #[test]
    fn lexer_reads_the_object_syntax_around_values() {
        let bytes = b"<< % comment /Count 1\n/A <48 49> /B (x(y)\\)) /C [1.5 true null] /D << /E /F >> /Count 2 >>";
        let dictionary = Lexer::new(bytes, 0).dictionary(0).unwrap();
        assert_eq!(get(&dictionary, b"A"), Some(&Value::Other));
        assert_eq!(get(&dictionary, b"B"), Some(&Value::Other));
        assert_eq!(get(&dictionary, b"C"), Some(&Value::Other));
        assert_eq!(get(&dictionary, b"D"), Some(&Value::Other));
        assert_eq!(get(&dictionary, b"Count"), Some(&Value::Integer(2)));
        assert_eq!(dictionary.len(), 5);
    }

    #[test]
    fn lexer_rejects_malformed_dictionaries() {
        let deep = format!("<< /A {}1{} >>", "[".repeat(40), "]".repeat(40));
        for bytes in [
            &b"[1]"[..],
            b"<< /A 1",
            b"<< 1 2 >>",
            b"<< /A ] >>",
            b"<< /A { >>",
            b"<< /A (open >>",
            b"<< /A <48",
            deep.as_bytes(),
        ] {
            assert!(
                Lexer::new(bytes, 0).dictionary(0).is_none(),
                "{}",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn malformed_tables_and_trailers_are_rejected() {
        let pdf = String::from_utf8(compact_pdf()).unwrap();
        let table = pdf.find("xref").unwrap();
        let broken = [
            "%PDF-1.7\n".to_string(),
            pdf.replacen("startxref\n", "startxref\nx", 1),
            pdf.replacen(
                &format!("startxref\n{table}"),
                &format!("startxref\n{}", table + 1),
                1,
            ),
            pdf.replacen("xref\n0 5", "xref\nx 5", 1),
            pdf.replacen("xref\n0 5", "xref\n0 x", 1),
            pdf.replacen("xref\n0 5", "xref\n0 50", 1),
            pdf.replacen("/Root 1 0 R", "", 1),
            pdf.replacen("/Root 1 0 R", "/Root 1", 1),
            pdf.replacen("trailer\n<<", "trailer\n[", 1),
        ];
        for pdf in broken {
            assert!(RawPdf::new(pdf.as_bytes()).is_none(), "{pdf}");
        }
    }

    #[test]
    fn objects_must_be_where_the_table_says() {
        let pdf = compact_pdf();
        let raw = RawPdf::new(&pdf).unwrap();
        assert!(raw.dictionary(9).is_none());
        let mut moved = RawPdf::new(&pdf).unwrap();
        let second = moved.offsets[&2];
        moved.offsets.insert(1, second);
        assert!(moved.dictionary(1).is_none());
        moved.offsets.insert(1, moved.table);
        assert!(moved.dictionary(1).is_none());
        let text = String::from_utf8(pdf.clone())
            .unwrap()
            .replacen("1 0 obj", "1 0 xbj", 1);
        assert!(
            RawPdf::new(text.as_bytes())
                .unwrap()
                .dictionary(1)
                .is_none()
        );
    }

    #[test]
    fn patch_rejects_overlapping_edits_and_edits_past_the_objects() {
        let pdf = compact_pdf();
        let raw = RawPdf::new(&pdf).unwrap();
        let span = entry(&raw.dictionary(3).unwrap(), b"Count")
            .unwrap()
            .span
            .clone();
        let table = raw.table;
        let overlapping = vec![
            (span.clone(), b"/Count 1".to_vec()),
            (span.start + 1..span.end, b"/Count 2".to_vec()),
        ];
        assert!(patch(pdf.clone(), raw.layout(), overlapping).is_none());
        let past = vec![(table..table + 4, b"/Count 1".to_vec())];
        assert!(patch(pdf.clone(), raw.layout(), past).is_none());
        assert_eq!(patch(pdf.clone(), raw.layout(), Vec::new()), Some(pdf));
    }
}
