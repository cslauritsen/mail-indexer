//! Hybrid search: semantic (sqlite-vec) + full text (FTS5), merged with Reciprocal Rank Fusion.

use crate::embed::{self, EmbedConfig, Kind};
use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};

const KNN_K: usize = 100;
const RRF_K: f64 = 60.0;
/// Vector hits farther than this are dropped (override with MAX_DIST).
const MAX_VEC_DIST: f64 = f64::INFINITY;

/// Optional filters. Dates are YYYY-MM-DD; `to` is exclusive.
#[derive(Default, Clone)]
pub struct Filters {
    pub from: Option<String>,
    pub to: Option<String>,
    pub sender: Option<String>,
}

pub struct Hit {
    pub id: String,
    pub subject: String,
    pub from_name: String,
    pub from_addr: String,
    /// "YYYY-MM-DD HH:MM:SS" in UTC
    pub date: String,
    pub folder: String,
    pub snippet: String,
}

/// Quoted search words (3+ chars, deduplicated, at most 8).
pub fn fts_terms(q: &str) -> Vec<String> {
    let mut toks: Vec<String> = Vec::new();
    for t in q.split(|c: char| !c.is_alphanumeric()) {
        let t = t.to_lowercase();
        if t.chars().count() >= 3 && toks.len() < 8 && !toks.contains(&t) {
            toks.push(t);
        }
    }
    toks.into_iter().map(|t| format!("\"{t}\"")).collect()
}

/// FTS5 query matching any `k` of the terms: (a AND b) OR (a AND c) OR ...
pub fn fts_any_k(terms: &[String], k: usize) -> String {
    fn rec(terms: &[String], k: usize, cur: &mut Vec<String>, out: &mut Vec<String>) {
        if cur.len() == k {
            out.push(format!("({})", cur.join(" AND ")));
            return;
        }
        for i in 0..terms.len() {
            if terms.len() - i < k - cur.len() {
                break;
            }
            cur.push(terms[i].clone());
            rec(&terms[i + 1..], k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(terms, k, &mut Vec::new(), &mut out);
    out.join(" OR ")
}

/// Escapes LIKE wildcards so that user text is matched literally (use with ESCAPE '\').
pub fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '\\' || c == '%' || c == '_' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn date_to_ts(conn: &Connection, s: &str) -> Result<i64> {
    conn.query_row(
        "SELECT CAST(strftime('%s', ?1) AS INTEGER)",
        params![s],
        |r| r.get::<_, Option<i64>>(0),
    )?
    .ok_or_else(|| anyhow!("invalid date '{s}', use YYYY-MM-DD"))
}

pub fn search(
    conn: &Connection,
    cfg: &EmbedConfig,
    query: &str,
    filters: &Filters,
    limit: usize,
) -> Result<Vec<Hit>> {
    let date_from = filters
        .from
        .as_deref()
        .map(|s| date_to_ts(conn, s))
        .transpose()?;
    let date_to = filters
        .to
        .as_deref()
        .map(|s| date_to_ts(conn, s))
        .transpose()?;
    let sender_pat = filters
        .sender
        .as_deref()
        .map(|s| format!("%{}%", escape_like(s)));

    // 1. Semantic ranking.
    let qvec = embed::embed(cfg, &[query.to_string()], Kind::Query)?.remove(0);
    let mut vec_ids: Vec<i64> = Vec::new();
    let max_dist: f64 = std::env::var("MAX_DIST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_VEC_DIST);
    let debug = std::env::var("SEARCH_DEBUG").is_ok();
    {
        let sql = format!(
            "SELECT rowid, distance FROM chunk_vectors WHERE embedding MATCH ?1 AND k = {KNN_K} ORDER BY distance"
        );
        let mut st = conn.prepare(&sql)?;
        let rows = st.query_map(params![embed::to_blob(&qvec)], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })?;
        for (n, r) in rows.enumerate() {
            let (id, dist) = r?;
            if debug && n < 30 {
                eprintln!("vec #{n} chunk {id} dist {dist:.4}");
            }
            if dist <= max_dist {
                vec_ids.push(id);
            }
        }
    }

    // 2. Full-text ranking.
    let mut fts_ids: Vec<i64> = Vec::new();
    // Prefer messages with every word; step down to "any 2 words" if none,
    // but never to a single word for a multi-word query.
    let terms = fts_terms(query);
    let min_k = terms.len().min(2);
    for k in (min_k..=terms.len()).rev() {
        if k == 0 {
            break;
        }
        let q = fts_any_k(&terms, k);
        let mut st = conn.prepare(&format!(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 \
             ORDER BY bm25(chunks_fts) LIMIT {KNN_K}"
        ))?;
        let rows = st.query_map(params![q], |r| r.get::<_, i64>(0))?;
        for r in rows {
            fts_ids.push(r?);
        }
        if !fts_ids.is_empty() {
            break;
        }
    }

    // 3. Reciprocal Rank Fusion.
    let mut score: HashMap<i64, f64> = HashMap::new();
    for ids in [&vec_ids, &fts_ids] {
        for (rank, id) in ids.iter().enumerate() {
            *score.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut ranked: Vec<(i64, f64)> = score.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Apply filters, keep the best chunk per message.
    let mut st = conn.prepare(
        "SELECT m.id, m.subject, m.from_name, m.from_addr, datetime(m.date_utc, 'unixepoch'), \
                m.folder, m.snippet \
         FROM chunks c JOIN messages m ON m.id = c.msg_id \
         WHERE c.id = ?1 \
           AND (?2 IS NULL OR m.date_utc >= ?2) \
           AND (?3 IS NULL OR m.date_utc <  ?3) \
           AND (?4 IS NULL OR m.from_addr LIKE ?4 ESCAPE '\\' OR m.from_name LIKE ?4 ESCAPE '\\')",
    )?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut hits: Vec<Hit> = Vec::new();
    for (chunk_id, _) in ranked {
        let row = st
            .query_row(params![chunk_id, date_from, date_to, sender_pat], |r| {
                Ok(Hit {
                    id: r.get::<_, String>(0)?,
                    subject: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    from_name: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    from_addr: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    date: r.get::<_, String>(4)?,
                    folder: r.get::<_, String>(5)?,
                    snippet: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                })
            })
            .optional()?;
        let Some(hit) = row else { continue };
        if !seen.insert(hit.id.clone()) {
            continue;
        }
        hits.push(hit);
        if hits.len() >= limit {
            break;
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_terms_quotes_words_and_drops_short_ones() {
        assert_eq!(
            fts_terms("la facture de septembre"),
            vec!["\"facture\"", "\"septembre\""]
        );
        assert!(fts_terms("a b").is_empty());
    }

    #[test]
    fn fts_terms_strips_operators_and_dedupes() {
        assert_eq!(fts_terms("foo\" OR bar* Foo"), vec!["\"foo\"", "\"bar\""]);
    }

    #[test]
    fn fts_any_k_combines_terms() {
        let t = fts_terms("alpha beta gamma");
        assert_eq!(fts_any_k(&t, 3), "(\"alpha\" AND \"beta\" AND \"gamma\")");
        assert_eq!(
            fts_any_k(&t, 2),
            "(\"alpha\" AND \"beta\") OR (\"alpha\" AND \"gamma\") OR (\"beta\" AND \"gamma\")"
        );
    }

    #[test]
    fn escape_like_escapes_wildcards() {
        assert_eq!(escape_like("50%_off\\"), "50\\%\\_off\\\\");
    }
}
