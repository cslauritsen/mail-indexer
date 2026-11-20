//! mail-indexer: index a Dovecot Maildir into SQLite (FTS5 + sqlite-vec) and search it.
//!
//! Commands:
//!   mail-indexer index                      scan the Maildir, add/move/remove messages
//!   mail-indexer search "<text>" [filters]  hybrid (semantic + full-text) search
//!   mail-indexer show <id-prefix>           print one message
//!
//! Settings (flag or environment variable):
//!   --maildir / MAILDIR        root of the Maildir (required for index and show)
//!   --db      / DB             SQLite file (default: mail-index.db)
//!   --skip    / SKIP_FOLDERS   comma-separated folders to ignore, e.g. .Junk,.Trash
//!   EMBED_URL   Ollama-style endpoint (default: http://localhost:11434/api/embed)
//!   EMBED_MODEL embedding model name (default: bge-m3)
//!   EMBED_DIM   embedding size (default: 1024); must match the model
//! Search filters: --from YYYY-MM-DD  --to YYYY-MM-DD  --sender TEXT  --limit N

use anyhow::{anyhow, bail, Context, Result};
use mail_parser::MessageParser;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::{env, fs, time::Duration};

const SCHEMA: &str = include_str!("../schema.sql");
const MAX_CHUNKS: usize = 8; // per message, to bound embedding cost
const CHUNK_CHARS: usize = 1200;
const KNN_K: usize = 100;
const RRF_K: f64 = 60.0;

struct Config {
    maildir: PathBuf,
    db: PathBuf,
    embed_url: String,
    embed_model: String,
    dim: usize,
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

fn open_db(cfg: &Config) -> Result<Connection> {
    // Register sqlite-vec for every connection opened after this point.
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    let conn = Connection::open(&cfg.db).context("cannot open database")?;
    conn.execute_batch(&SCHEMA.replace("{DIM}", &cfg.dim.to_string()))?;
    Ok(conn)
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
            let Ok(rd) = fs::read_dir(dir.join(sub)) else {
                continue;
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

// ---------------------------------------------------------------- embeddings

fn embed(cfg: &Config, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
    let resp: serde_json::Value = ureq::post(&cfg.embed_url)
        .timeout(Duration::from_secs(300))
        .send_json(serde_json::json!({ "model": cfg.embed_model, "input": inputs }))
        .context("embedding request failed (is the embedding server running?)")?
        .into_json()?;
    let arr = resp["embeddings"]
        .as_array()
        .ok_or_else(|| anyhow!("no 'embeddings' field in the response"))?;
    if arr.len() != inputs.len() {
        bail!("got {} embeddings for {} inputs", arr.len(), inputs.len());
    }
    let mut out = Vec::new();
    for v in arr {
        let vec: Vec<f32> = v
            .as_array()
            .ok_or_else(|| anyhow!("bad embedding format"))?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if vec.len() != cfg.dim {
            bail!(
                "embedding has {} dimensions, expected {} (check EMBED_DIM)",
                vec.len(),
                cfg.dim
            );
        }
        out.push(vec);
    }
    Ok(out)
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ---------------------------------------------------------------- index

/// Returns Ok(true) if a new message was added, Ok(false) if it was only moved.
fn index_one(conn: &mut Connection, cfg: &Config, f: &Found) -> Result<bool> {
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
        return Ok(false);
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
    let vecs = embed(cfg, &inputs)?;

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
            f.folder,
            f.name,
            p.date,
            p.from_addr,
            p.from_name,
            p.to,
            p.subject,
            p.snippet,
            p.has_attach as i64
        ],
    )?;
    for (i, (c, v)) in chunks.iter().zip(vecs.iter()).enumerate() {
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
            params![cid, to_blob(v)],
        )?;
    }
    tx.commit()?;
    Ok(true)
}

fn cmd_index(cfg: &Config) -> Result<()> {
    let mut conn = open_db(cfg)?;
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

    let (mut added, mut moved, mut failed) = (0usize, 0usize, 0usize);
    for f in found
        .iter()
        .filter(|f| !known.contains(&(f.folder.clone(), f.name.clone())))
    {
        match index_one(&mut conn, cfg, f) {
            Ok(true) => added += 1,
            Ok(false) => moved += 1,
            Err(e) => {
                failed += 1;
                eprintln!("skipped {}: {e:#}", f.path.display());
            }
        }
    }

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
    for id in &gone {
        let tx = conn.transaction()?;
        // Delete chunks explicitly so the trigger cleans FTS and vectors.
        tx.execute("DELETE FROM chunks WHERE msg_id = ?1", params![id])?;
        tx.execute("DELETE FROM messages WHERE id = ?1", params![id])?;
        tx.commit()?;
    }

    println!(
        "added {added}, moved {moved}, removed {}, failed {failed}",
        gone.len()
    );
    Ok(())
}

// ---------------------------------------------------------------- search

/// Turns free text into an FTS5 query: quoted words joined by OR.
fn fts_query(q: &str) -> Option<String> {
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

fn date_to_ts(conn: &Connection, s: &str) -> Result<i64> {
    conn.query_row(
        "SELECT CAST(strftime('%s', ?1) AS INTEGER)",
        params![s],
        |r| r.get::<_, Option<i64>>(0),
    )?
    .ok_or_else(|| anyhow!("invalid date '{s}', use YYYY-MM-DD"))
}

fn cmd_search(
    cfg: &Config,
    query: &str,
    from: Option<&str>,
    to: Option<&str>,
    sender: Option<&str>,
    limit: usize,
) -> Result<()> {
    let conn = open_db(cfg)?;
    let date_from = from.map(|s| date_to_ts(&conn, s)).transpose()?;
    let date_to = to.map(|s| date_to_ts(&conn, s)).transpose()?;
    let sender_pat = sender.map(|s| format!("%{s}%"));

    // 1. Semantic ranking.
    let qvec = embed(cfg, &[query.to_string()])?.remove(0);
    let mut vec_ids: Vec<i64> = Vec::new();
    {
        let sql = format!(
            "SELECT rowid FROM chunk_vectors WHERE embedding MATCH ?1 AND k = {KNN_K} ORDER BY distance"
        );
        let mut st = conn.prepare(&sql)?;
        let rows = st.query_map(params![to_blob(&qvec)], |r| r.get::<_, i64>(0))?;
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
           AND (?4 IS NULL OR m.from_addr LIKE ?4 OR m.from_name LIKE ?4)",
    )?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut shown = 0usize;
    for (chunk_id, _) in ranked {
        let row = st
            .query_row(params![chunk_id, date_from, date_to, sender_pat], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, Option<String>>(6)?,
                ))
            })
            .optional()?;
        let Some((id, subject, name, addr, date, folder, snippet)) = row else {
            continue;
        };
        if !seen.insert(id.clone()) {
            continue;
        }
        shown += 1;
        println!(
            "{shown}. [{}] {date}  {} <{}>  ({folder})",
            &id[..id.len().min(12)],
            name.unwrap_or_default(),
            addr.unwrap_or_default()
        );
        println!("   {}", subject.unwrap_or_default());
        println!("   {}", snippet.unwrap_or_default());
        if shown >= limit {
            break;
        }
    }
    if shown == 0 {
        println!("no results");
    }
    Ok(())
}

// ---------------------------------------------------------------- show

fn cmd_show(cfg: &Config, id_prefix: &str) -> Result<()> {
    let conn = open_db(cfg)?;
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
    let path = locate(&cfg.maildir, &folder, &name)
        .ok_or_else(|| anyhow!("file not found on disk; run 'index' to refresh"))?;
    let p = parse_file(&path)?;
    println!("File:    {}", path.display());
    println!("Folder:  {folder}");
    println!("Date:    {date} UTC");
    println!("From:    {} <{}>", p.from_name, p.from_addr);
    println!("To:      {}", p.to);
    println!("Subject: {}", p.subject);
    if let Some(m) = message_id {
        println!("Link:    mid:{m}");
    }
    println!("\n{}", p.body);
    Ok(())
}

// ---------------------------------------------------------------- main

fn usage() {
    eprintln!(
        "usage:\n  mail-indexer index [--maildir DIR] [--db FILE] [--skip .Junk,.Trash]\n  \
         mail-indexer search \"text\" [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--sender TEXT] [--limit N]\n  \
         mail-indexer show <id-prefix> [--maildir DIR]"
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(cmd) = args.first().cloned() else {
        usage();
        return Ok(());
    };

    let mut opts: HashMap<String, String> = HashMap::new();
    let mut pos: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if let Some(k) = args[i].strip_prefix("--") {
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
    let cfg = Config {
        maildir: PathBuf::from(get("maildir", "MAILDIR", "")),
        db: PathBuf::from(get("db", "DB", "mail-index.db")),
        embed_url: get("embed-url", "EMBED_URL", "http://localhost:11434/api/embed"),
        embed_model: get("embed-model", "EMBED_MODEL", "bge-m3"),
        dim: get("embed-dim", "EMBED_DIM", "1024")
            .parse()
            .context("EMBED_DIM must be a number")?,
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
            cmd_index(&cfg)
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
            cmd_search(
                &cfg,
                &pos.join(" "),
                opts.get("from").map(|s| s.as_str()),
                opts.get("to").map(|s| s.as_str()),
                opts.get("sender").map(|s| s.as_str()),
                limit,
            )
        }
        "show" => {
            need_maildir()?;
            let id = pos
                .first()
                .ok_or_else(|| anyhow!("usage: show <id-prefix>"))?;
            cmd_show(&cfg, id)
        }
        _ => {
            usage();
            Ok(())
        }
    }
}
