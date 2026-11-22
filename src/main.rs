//! mail-indexer: index a Dovecot Maildir into SQLite (FTS5 + sqlite-vec), search it, and serve
//! a small web application on top of it.
//!
//! Commands:
//!   mail-indexer index [--force]            scan the Maildir, add/move/remove messages
//!   mail-indexer search "<text>" [filters]  hybrid (semantic + full-text) search
//!   mail-indexer show <id-prefix>           print one message
//!   mail-indexer serve                      web application with Google sign-in
//!   mail-indexer reset --yes                forget everything (needed when changing embeddings)
//!
//! Settings (flag or environment variable):
//!   --maildir / MAILDIR        root of the Maildir (required for index, show and serve)
//!   --db      / DB             SQLite file (default: mail-index.db)
//!   --skip    / SKIP_FOLDERS   comma-separated folders to ignore, e.g. .Junk,.Trash
//!   EMBED_PROVIDER             voyage (default when VOYAGE_API_KEY is set) or ollama
//!   VOYAGE_API_KEY             API key for Voyage AI (environment only)
//!   EMBED_URL, EMBED_MODEL     defaults: Voyage API and voyage-4, or local Ollama and bge-m3
//!   EMBED_DIM                  vector size (default 1024); must match the model
//!   EMBED_BATCH                texts per embedding request (default 64)
//! Search filters: --from YYYY-MM-DD  --to YYYY-MM-DD  --sender TEXT  --limit N
//! Web settings are documented in the README (BASE_URL, GOOGLE_CLIENT_ID, ...).

mod auth;
mod embed;
mod nl;
mod search;
mod telemetry;
mod web;

use anyhow::{anyhow, bail, Context, Result};
use embed::{EmbedConfig, Kind};
use mail_parser::MessageParser;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::{env, fs, time::Duration};

const SCHEMA: &str = include_str!("../schema.sql");
const MAX_CHUNKS: usize = 8; // per message, to bound embedding cost
const CHUNK_CHARS: usize = 1200;
/// `index` refuses to forget more than this share of the index in one run (unless --force):
/// a renamed Maildir or lost permissions look exactly like mass deletion.
const MAX_REMOVE_PERCENT: usize = 20;
/// Below this many removals the guard stays quiet, so small indexes are not blocked.
const MIN_REMOVE_GUARD: usize = 10;

struct Config {
    maildir: PathBuf,
    db: PathBuf,
    embed: EmbedConfig,
    skip: Vec<String>,
}

struct Found {
    folder: String,
    name: String,
    path: PathBuf,
}

struct Parsed {
    id: String,
    message_id: Option<String>,
    in_reply_to: Option<String>,
    refs: Option<String>,
    date: i64,
    from_addr: String,
    from_name: String,
    to: String,
    subject: String,
    snippet: String,
    has_attach: bool,
    body: String,
}

// ---------------------------------------------------------------- database

static VEC_INIT: Once = Once::new();

/// Opens the database with sqlite-vec loaded. Does not touch the schema.
fn open_conn(db: &Path) -> Result<Connection> {
    // Register sqlite-vec for every connection opened after this point.
    VEC_INIT.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
    let conn = Connection::open(db).context("cannot open database")?;
    conn.busy_timeout(Duration::from_secs(10))?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(conn)
}

/// Opens the database and makes sure the schema exists.
fn open_db(cfg: &Config) -> Result<Connection> {
    let conn = open_conn(&cfg.db)?;
    conn.execute_batch(&SCHEMA.replace("{DIM}", &cfg.embed.dim.to_string()))?;
    Ok(conn)
}

/// Refuses to mix vectors from different embedding models or sizes.
fn check_fingerprint(conn: &Connection, cfg: &EmbedConfig) -> Result<()> {
    let fp = cfg.fingerprint();
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embed_fingerprint'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    match stored {
        Some(s) if s == fp => Ok(()),
        Some(s) => bail!(
            "the index was built with embeddings '{s}' but the current settings are '{fp}'; \
             run 'mail-indexer reset --yes' and then 'index' to rebuild it"
        ),
        None => {
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?;
            // Indexes created before this check always used the local bge-m3 model.
            if n > 0 && !(cfg.provider == "ollama" && cfg.model == "bge-m3") {
                bail!(
                    "the existing index predates the embedding check and was built with ollama/bge-m3; \
                     run 'mail-indexer reset --yes' and then 'index' to rebuild it with '{fp}'"
                );
            }
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_fingerprint', ?1)",
                params![fp],
            )?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------- Maildir scan

/// Lists every message file. The root itself is "INBOX"; Maildir++ subfolders
/// are directories starting with '.' that contain cur/.
fn scan(root: &Path, skip: &[String]) -> Result<Vec<Found>> {
    let mut dirs: Vec<(String, PathBuf)> = Vec::new();
    if root.join("cur").is_dir() {
        dirs.push(("INBOX".to_string(), root.to_path_buf()));
    }
    for entry in fs::read_dir(root).context("cannot read maildir root")? {
        let entry = entry?;
        let fname = entry.file_name().to_string_lossy().to_string();
        if fname.starts_with('.') && entry.path().join("cur").is_dir() && !skip.contains(&fname) {
            dirs.push((fname, entry.path()));
        }
    }
    let mut out = Vec::new();
    for (folder, dir) in dirs {
        for sub in ["cur", "new"] {
            let rd = match fs::read_dir(dir.join(sub)) {
                Ok(rd) => rd,
                Err(e) => {
                    tracing::warn!(folder = %folder, sub, "cannot read directory: {e}");
                    continue;
                }
            };
            for f in rd.flatten() {
                let fname = f.file_name().to_string_lossy().to_string();
                if fname.starts_with('.') {
                    continue;
                }
                let name = fname.split(':').next().unwrap_or(&fname).to_string();
                out.push(Found {
                    folder: folder.clone(),
                    name,
                    path: f.path(),
                });
            }
        }
    }
    Ok(out)
}

/// Finds the current file for a stored (folder, unique name) pair.
fn locate(root: &Path, folder: &str, name: &str) -> Option<PathBuf> {
    let dir = if folder == "INBOX" {
        root.to_path_buf()
    } else {
        root.join(folder)
    };
    for sub in ["cur", "new"] {
        if let Ok(rd) = fs::read_dir(dir.join(sub)) {
            for f in rd.flatten() {
                let fname = f.file_name().to_string_lossy().to_string();
                if fname.split(':').next() == Some(name) {
                    return Some(f.path());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------- parsing

fn sha_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn clean_ids(s: &str) -> String {
    s.replace(['<', '>'], "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_html(h: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in h.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ => {
                if !in_tag {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn parse_file(path: &Path) -> Result<Parsed> {
    let bytes = fs::read(path)?;
    let msg = MessageParser::default()
        .parse(&bytes)
        .ok_or_else(|| anyhow!("unparseable message"))?;

    let message_id = msg
        .message_id()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());
    let in_reply_to = msg
        .header_raw("In-Reply-To")
        .map(clean_ids)
        .filter(|s| !s.is_empty());
    let refs = msg
        .header_raw("References")
        .map(clean_ids)
        .filter(|s| !s.is_empty());
    let to = msg
        .header_raw("To")
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();

    let (from_name, from_addr) = msg
        .from()
        .and_then(|a| a.first())
        .map(|a| {
            (
                a.name.as_deref().unwrap_or("").to_string(),
                a.address.as_deref().unwrap_or("").to_string(),
            )
        })
        .unwrap_or_default();

    let mtime = fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let date = msg.date().map(|d| d.to_timestamp()).unwrap_or(mtime);

    let subject = msg.subject().unwrap_or("").to_string();
    let body = match msg.body_text(0) {
        Some(t) => t.to_string(),
        None => msg.body_html(0).map(|h| strip_html(&h)).unwrap_or_default(),
    };
    let snippet: String = clean_body(&body)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(200)
        .collect();

    let id = match &message_id {
        Some(m) => sha_hex(m.as_bytes()),
        None => sha_hex(&bytes),
    };

    Ok(Parsed {
        id,
        message_id,
        in_reply_to,
        refs,
        date,
        from_addr,
        from_name,
        to,
        subject,
        snippet,
        has_attach: msg.attachment_count() > 0,
        body,
    })
}

// ---------------------------------------------------------------- chunking

/// Drops quoted replies and the signature to keep the index relevant.
fn clean_body(body: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for l in body.lines() {
        let t = l.trim_end();
        if t == "--" {
            break; // signature delimiter ("-- ")
        }
        if t.trim_start().starts_with('>') {
            continue; // quoted text
        }
        lines.push(t);
    }
    lines.join("\n")
}

fn split_long(s: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    chars
        .chunks(max)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

/// Groups paragraphs into chunks of about CHUNK_CHARS characters.
fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let paragraphs = text
        .split("\n\n")
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|p| !p.is_empty());
    for para in paragraphs {
        for piece in split_long(&para, CHUNK_CHARS) {
            if !cur.is_empty() && cur.chars().count() + piece.chars().count() + 1 > CHUNK_CHARS {
                chunks.push(std::mem::take(&mut cur));
                if chunks.len() >= MAX_CHUNKS {
                    return chunks;
                }
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&piece);
        }
    }
    if !cur.is_empty() && chunks.len() < MAX_CHUNKS {
        chunks.push(cur);
    }
    chunks
}

// ---------------------------------------------------------------- index

/// A new message that has been parsed and chunked but not yet embedded or stored.
struct Prepared {
    parsed: Parsed,
    folder: String,
    name: String,
    chunks: Vec<String>,
    /// Texts to embed, one per chunk (headers + chunk).
    inputs: Vec<String>,
}

enum Prep {
    /// Already indexed; only its location was updated.
    Moved,
    /// Same message already waiting in the current batch.
    Duplicate,
    New(Box<Prepared>),
}

fn prepare(conn: &Connection, f: &Found, pending_ids: &HashSet<String>) -> Result<Prep> {
    let p = parse_file(&f.path)?;

    // Same message already indexed (moved to another folder, or a copy): update location.
    let exists = conn
        .query_row(
            "SELECT 1 FROM messages WHERE id = ?1",
            params![p.id],
            |_| Ok(true),
        )
        .optional()?
        .unwrap_or(false);
    if exists {
        conn.execute(
            "UPDATE messages SET folder = ?1, maildir_name = ?2 WHERE id = ?3",
            params![f.folder, f.name, p.id],
        )?;
        return Ok(Prep::Moved);
    }
    if pending_ids.contains(&p.id) {
        return Ok(Prep::Duplicate);
    }

    let mut chunks = make_chunks(&clean_body(&p.body));
    if chunks.is_empty() {
        let fallback = if p.subject.is_empty() {
            "(empty message)".to_string()
        } else {
            p.subject.clone()
        };
        chunks.push(fallback);
    }
    // The embedded text carries the headers, so "from the treasurer" can match.
    let inputs: Vec<String> = chunks
        .iter()
        .map(|c| {
            format!(
                "Subject: {}\nFrom: {} <{}>\n\n{}",
                p.subject, p.from_name, p.from_addr, c
            )
        })
        .collect();
    Ok(Prep::New(Box::new(Prepared {
        parsed: p,
        folder: f.folder.clone(),
        name: f.name.clone(),
        chunks,
        inputs,
    })))
}

fn store(conn: &mut Connection, prep: &Prepared, vecs: &[Vec<f32>]) -> Result<()> {
    let p = &prep.parsed;
    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO messages (id, message_id, in_reply_to, refs, folder, maildir_name, date_utc, \
         from_addr, from_name, to_addrs, subject, snippet, has_attach) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            p.id,
            p.message_id,
            p.in_reply_to,
            p.refs,
            prep.folder,
            prep.name,
            p.date,
            p.from_addr,
            p.from_name,
            p.to,
            p.subject,
            p.snippet,
            p.has_attach as i64
        ],
    )?;
    for (i, (c, v)) in prep.chunks.iter().zip(vecs.iter()).enumerate() {
        tx.execute(
            "INSERT INTO chunks (msg_id, chunk_no, text) VALUES (?1, ?2, ?3)",
            params![p.id, i as i64, c],
        )?;
        let cid = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO chunks_fts (rowid, text, subject, sender) VALUES (?1, ?2, ?3, ?4)",
            params![
                cid,
                c,
                p.subject,
                format!("{} {}", p.from_name, p.from_addr)
            ],
        )?;
        tx.execute(
            "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?1, ?2)",
            params![cid, embed::to_blob(v)],
        )?;
    }
    tx.commit()?;
    Ok(())
}

/// Embeds all waiting messages together (few requests, friendly to API rate limits),
/// then stores them. Returns how many messages were stored.
fn flush(conn: &mut Connection, cfg: &Config, batch: &mut Vec<Prepared>) -> Result<usize> {
    if batch.is_empty() {
        return Ok(0);
    }
    let all_inputs: Vec<String> = batch
        .iter()
        .flat_map(|p| p.inputs.iter().cloned())
        .collect();
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(all_inputs.len());
    for part in all_inputs.chunks(cfg.embed.batch.max(1)) {
        vecs.extend(embed::embed(&cfg.embed, part, Kind::Document)?);
    }
    let mut it = vecs.into_iter();
    let mut stored = 0;
    for prep in batch.drain(..) {
        let mine: Vec<Vec<f32>> = it.by_ref().take(prep.inputs.len()).collect();
        store(conn, &prep, &mine)?;
        stored += 1;
    }
    Ok(stored)
}

/// True when removing `gone` of `total` indexed messages looks like a mistake, not housekeeping.
fn removal_blocked(gone: usize, total: usize) -> bool {
    gone >= MIN_REMOVE_GUARD && gone * 100 > total * MAX_REMOVE_PERCENT
}

#[tracing::instrument(skip_all)]
fn cmd_index(cfg: &Config, force: bool) -> Result<()> {
    let mut conn = open_db(cfg)?;
    check_fingerprint(&conn, &cfg.embed)?;
    let found = scan(&cfg.maildir, &cfg.skip)?;
    let on_disk: HashSet<(String, String)> = found
        .iter()
        .map(|f| (f.folder.clone(), f.name.clone()))
        .collect();

    let mut known: HashSet<(String, String)> = HashSet::new();
    {
        let mut st = conn.prepare("SELECT folder, maildir_name FROM messages")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        for r in rows {
            known.insert(r?);
        }
    }

    let (mut added, mut moved, mut dups, mut failed) = (0usize, 0usize, 0usize, 0usize);
    let mut batch: Vec<Prepared> = Vec::new();
    let mut pending_ids: HashSet<String> = HashSet::new();
    let mut pending_inputs = 0usize;
    for f in found
        .iter()
        .filter(|f| !known.contains(&(f.folder.clone(), f.name.clone())))
    {
        match prepare(&conn, f, &pending_ids) {
            Ok(Prep::Moved) => moved += 1,
            Ok(Prep::Duplicate) => dups += 1,
            Ok(Prep::New(p)) => {
                pending_ids.insert(p.parsed.id.clone());
                pending_inputs += p.inputs.len();
                batch.push(*p);
                if pending_inputs >= cfg.embed.batch {
                    added += flush(&mut conn, cfg, &mut batch)
                        .context("embedding or storing failed; run 'index' again to resume")?;
                    pending_ids.clear();
                    pending_inputs = 0;
                    tracing::info!(added, "indexed messages so far");
                }
            }
            Err(e) => {
                failed += 1;
                tracing::warn!(path = %f.path.display(), "skipped message: {e:#}");
            }
        }
    }
    added += flush(&mut conn, cfg, &mut batch)
        .context("embedding or storing failed; run 'index' again to resume")?;

    // Remove messages whose file is gone (after moves were applied above).
    let gone: Vec<String> = {
        let mut st = conn.prepare("SELECT id, folder, maildir_name FROM messages")?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut v = Vec::new();
        for r in rows {
            let (id, folder, name) = r?;
            if !on_disk.contains(&(folder, name)) {
                v.push(id);
            }
        }
        v
    };
    let total: usize = conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get::<_, i64>(0))?
        as usize;
    if !force && removal_blocked(gone.len(), total) {
        tracing::error!(
            gone = gone.len(),
            total,
            "refusing to remove {} of {} indexed messages (more than {MAX_REMOVE_PERCENT}%)",
            gone.len(),
            total
        );
        bail!(
            "{} of {total} indexed messages are no longer on disk (more than {MAX_REMOVE_PERCENT}%). \
             Is MAILDIR correct and readable? Nothing was removed; run 'index --force' if this is intended",
            gone.len()
        );
    }
    for id in &gone {
        let tx = conn.transaction()?;
        // Delete chunks explicitly so the trigger cleans FTS and vectors.
        tx.execute("DELETE FROM chunks WHERE msg_id = ?1", params![id])?;
        tx.execute("DELETE FROM messages WHERE id = ?1", params![id])?;
        tx.commit()?;
    }

    println!(
        "added {added}, moved {moved}, duplicates {dups}, removed {}, failed {failed}",
        gone.len()
    );
    Ok(())
}

fn cmd_reset(cfg: &Config) -> Result<()> {
    let mut conn = open_db(cfg)?;
    let tx = conn.transaction()?;
    // Chunks first, so the trigger empties the FTS and vector tables too.
    tx.execute("DELETE FROM chunks", [])?;
    tx.execute("DELETE FROM messages", [])?;
    tx.execute("DELETE FROM meta WHERE key = 'embed_fingerprint'", [])?;
    tx.commit()?;
    println!("index emptied; run 'index' to rebuild it");
    Ok(())
}

// ---------------------------------------------------------------- search and show

#[tracing::instrument(skip_all)]
fn cmd_search(cfg: &Config, query: &str, filters: &search::Filters, limit: usize) -> Result<()> {
    let conn = open_db(cfg)?;
    check_fingerprint(&conn, &cfg.embed)?;
    let hits = search::search(&conn, &cfg.embed, query, filters, limit)?;
    if hits.is_empty() {
        println!("no results");
    }
    for (i, h) in hits.iter().enumerate() {
        println!(
            "{}. [{}] {}  {} <{}>  ({})",
            i + 1,
            &h.id[..h.id.len().min(12)],
            h.date,
            h.from_name,
            h.from_addr,
            h.folder
        );
        println!("   {}", h.subject);
        println!("   {}", h.snippet);
    }
    Ok(())
}

/// A message found by id, with its file parsed.
struct MessageView {
    path: PathBuf,
    folder: String,
    message_id: Option<String>,
    date: String,
    parsed: Parsed,
}

/// Finds a message by the start of its id (hex only) and reads it from the Maildir.
fn lookup_message(conn: &Connection, maildir: &Path, id_prefix: &str) -> Result<MessageView> {
    if id_prefix.len() < 4 || !id_prefix.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("invalid message id");
    }
    let row = conn
        .query_row(
            "SELECT folder, maildir_name, message_id, datetime(date_utc, 'unixepoch') \
             FROM messages WHERE id LIKE ?1 || '%'",
            params![id_prefix],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, String>(3)?,
                ))
            },
        )
        .optional()?;
    let (folder, name, message_id, date) = row.ok_or_else(|| anyhow!("no message with that id"))?;
    let path = locate(maildir, &folder, &name)
        .ok_or_else(|| anyhow!("file not found on disk; run 'index' to refresh"))?;
    let parsed = parse_file(&path)?;
    Ok(MessageView {
        path,
        folder,
        message_id,
        date,
        parsed,
    })
}

fn cmd_show(cfg: &Config, id_prefix: &str) -> Result<()> {
    let conn = open_db(cfg)?;
    let v = lookup_message(&conn, &cfg.maildir, id_prefix)?;
    let p = &v.parsed;
    println!("File:    {}", v.path.display());
    println!("Folder:  {}", v.folder);
    println!("Date:    {} UTC", v.date);
    println!("From:    {} <{}>", p.from_name, p.from_addr);
    println!("To:      {}", p.to);
    println!("Subject: {}", p.subject);
    if let Some(m) = &v.message_id {
        println!("Link:    mid:{m}");
    }
    println!("\n{}", p.body);
    Ok(())
}

// ---------------------------------------------------------------- main

fn usage() {
    eprintln!(
        "usage:\n  mail-indexer index [--force] [--maildir DIR] [--db FILE] [--skip .Junk,.Trash]\n  \
         mail-indexer search \"text\" [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--sender TEXT] [--limit N]\n  \
         mail-indexer show <id-prefix> [--maildir DIR]\n  \
         mail-indexer serve [--maildir DIR] [--db FILE]\n  \
         mail-indexer reset --yes [--db FILE]"
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(cmd) = args.first().cloned() else {
        usage();
        return Ok(());
    };

    let _ = dotenvy::dotenv().ok();
    // Flushes pending traces when main returns.
    let _telemetry = telemetry::init();
    let mut opts: HashMap<String, String> = HashMap::new();
    let mut pos: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if let Some(k) = args[i].strip_prefix("--") {
            if k == "yes" || k == "force" {
                opts.insert(k.to_string(), "1".to_string());
                i += 1;
                continue;
            }
            let v = args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow!("--{k} needs a value"))?;
            opts.insert(k.to_string(), v);
            i += 2;
        } else {
            pos.push(args[i].clone());
            i += 1;
        }
    }

    let get = |flag: &str, var: &str, default: &str| -> String {
        opts.get(flag)
            .cloned()
            .or_else(|| env::var(var).ok())
            .unwrap_or_else(|| default.to_string())
    };

    // Secrets come from the environment only, never from flags.
    let api_key = env::var("VOYAGE_API_KEY").unwrap_or_default();
    let default_provider = if api_key.is_empty() {
        "ollama"
    } else {
        "voyage"
    };
    let provider = get("embed-provider", "EMBED_PROVIDER", default_provider);
    let (default_url, default_model) = match provider.as_str() {
        "voyage" => ("https://api.voyageai.com/v1/embeddings", "voyage-4"),
        "ollama" => ("http://localhost:11434/api/embed", "bge-m3"),
        other => bail!("unknown EMBED_PROVIDER '{other}' (use voyage or ollama)"),
    };
    let embed_cfg = EmbedConfig {
        url: get("embed-url", "EMBED_URL", default_url),
        model: get("embed-model", "EMBED_MODEL", default_model),
        dim: get("embed-dim", "EMBED_DIM", "1024")
            .parse()
            .context("EMBED_DIM must be a number")?,
        batch: get("embed-batch", "EMBED_BATCH", "64")
            .parse()
            .context("EMBED_BATCH must be a number")?,
        provider,
        api_key,
    };
    let cfg = Config {
        maildir: PathBuf::from(get("maildir", "MAILDIR", "")),
        db: PathBuf::from(get("db", "DB", "mail-index.db")),
        embed: embed_cfg,
        skip: get("skip", "SKIP_FOLDERS", "")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    };
    let need_maildir = || -> Result<()> {
        if cfg.maildir.as_os_str().is_empty() {
            bail!("set --maildir or MAILDIR");
        }
        Ok(())
    };

    match cmd.as_str() {
        "index" => {
            need_maildir()?;
            cmd_index(&cfg, opts.contains_key("force"))
        }
        "search" => {
            if pos.is_empty() {
                bail!("usage: search \"text\"");
            }
            let limit = opts
                .get("limit")
                .map(|s| s.parse::<usize>())
                .transpose()
                .context("--limit must be a number")?
                .unwrap_or(10);
            let filters = search::Filters {
                from: opts.get("from").cloned(),
                to: opts.get("to").cloned(),
                sender: opts.get("sender").cloned(),
            };
            cmd_search(&cfg, &pos.join(" "), &filters, limit)
        }
        "show" => {
            need_maildir()?;
            let id = pos
                .first()
                .ok_or_else(|| anyhow!("usage: show <id-prefix>"))?;
            cmd_show(&cfg, id)
        }
        "serve" => {
            need_maildir()?;
            web::serve(cfg)
        }
        "reset" => {
            if !opts.contains_key("yes") {
                bail!("reset deletes the whole index; add --yes to confirm");
            }
            cmd_reset(&cfg)
        }
        _ => {
            usage();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_guard_blocks_mass_deletion_only() {
        assert!(!removal_blocked(0, 1000));
        assert!(!removal_blocked(200, 1000)); // exactly 20%
        assert!(removal_blocked(201, 1000));
        assert!(removal_blocked(1000, 1000)); // everything gone
        assert!(!removal_blocked(5, 10)); // small deletions never trip it
        assert!(removal_blocked(10, 20));
    }

    #[test]
    fn clean_body_drops_quotes_and_signature() {
        let body = "Hello\n> quoted line\nBye\n-- \nChad";
        assert_eq!(clean_body(body), "Hello\nBye");
    }

    #[test]
    fn make_chunks_groups_paragraphs_and_caps_the_count() {
        assert_eq!(make_chunks("one\n\ntwo"), vec!["one two".to_string()]);
        let long = "word ".repeat(10_000);
        assert!(make_chunks(&long).len() <= MAX_CHUNKS);
        assert!(make_chunks("").is_empty());
    }

    #[test]
    fn clean_ids_strips_angle_brackets() {
        assert_eq!(clean_ids("<a@x>  <b@y>"), "a@x b@y");
    }
}
