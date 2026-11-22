//! Hybrid search: semantic (sqlite-vec) + full text (FTS5), merged with Reciprocal Rank Fusion.

use crate::embed::{self, EmbedConfig, Kind};
use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};

const KNN_K: usize = 100;
const RRF_K: f64 = 60.0;

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

/// Turns free text into an FTS5 query: quoted words joined by OR.
pub fn fts_query(q: &str) -> Option<String> {
    let toks: Vec<String> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 3)
        .map(|t| format!("\"{t}\""))
        .collect();
    if toks.is_empty() {
        None
    } else {
        Some(toks.join(" OR "))
    }
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
    {
        let sql = format!(
            "SELECT rowid FROM chunk_vectors WHERE embedding MATCH ?1 AND k = {KNN_K} ORDER BY distance"
        );
        let mut st = conn.prepare(&sql)?;
        let rows = st.query_map(params![embed::to_blob(&qvec)], |r| r.get::<_, i64>(0))?;
        for r in rows {
            vec_ids.push(r?);
        }
    }

    // 2. Full-text ranking.
    let mut fts_ids: Vec<i64> = Vec::new();
    if let Some(q) = fts_query(query) {
        let mut st = conn.prepare(&format!(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 \
             ORDER BY bm25(chunks_fts) LIMIT {KNN_K}"
        ))?;
        let rows = st.query_map(params![q], |r| r.get::<_, i64>(0))?;
        for r in rows {
            fts_ids.push(r?);
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
    fn fts_query_quotes_words_and_drops_short_ones() {
        assert_eq!(
            fts_query("la facture de septembre").as_deref(),
            Some("\"facture\" OR \"septembre\"")
        );
        assert_eq!(fts_query("a b"), None);
    }

    #[test]
    fn fts_query_strips_operators() {
        assert_eq!(
            fts_query("foo\" OR bar*").as_deref(),
            Some("\"foo\" OR \"bar\"")
        );
    }

    #[test]
    fn escape_like_escapes_wildcards() {
        assert_eq!(escape_like("50%_off\\"), "50\\%\\_off\\\\");
    }
}
