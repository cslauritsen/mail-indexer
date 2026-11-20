# mail-indexer

Indexes a Dovecot Maildir into one SQLite file (FTS5 for exact words, sqlite-vec for meaning)
and searches it with hybrid ranking (Reciprocal Rank Fusion).

## Status

Written without network access, so it has **not been compiled**. The syntax parses (rustfmt),
but type errors are possible, most likely in the `mail-parser` calls (the API changes between
versions). Run `cargo build` and fix what the compiler reports.

## Requirements

- Rust toolchain and a C compiler (SQLite and sqlite-vec are built from source)
- An embedding server with an Ollama-style `/api/embed` endpoint:
  `ollama pull bge-m3` (1024 dimensions, handles French and English)

## Usage

```
export MAILDIR=/var/mail/chad/Maildir
export DB=/var/lib/mail-index/mail.db

mail-indexer index                      # first run embeds everything; later runs are incremental
mail-indexer search "facture de septembre du trésorier" --from 2026-09-01 --to 2026-10-01
mail-indexer show a3f9c1d2e4b5          # id prefix as printed by search
```

Settings: `EMBED_URL` (default `http://localhost:11434/api/embed`), `EMBED_MODEL` (`bge-m3`),
`EMBED_DIM` (`1024`), `SKIP_FOLDERS` (e.g. `.Junk,.Trash`).

## How it works

- Message id = sha256(Message-ID), falling back to sha256(raw file). Location is stored as
  (folder, filename without the `:2,FLAGS` suffix), so flag changes do not matter.
- `index` adds new files, updates the folder when a message moved, and removes messages
  whose file is gone.
- Quoted replies and signatures are dropped before chunking (about 1200 characters, at most
  8 chunks per message). Subject and sender are added to the text that gets embedded.
- Search runs a vector query and an FTS5 query (100 candidates each), fuses the ranks,
  then applies the date and sender filters.

## Known limits

- The same Message-ID in two folders (copies) counts as one message; its stored location
  flips between the copies on each run. Harmless, but noisy.
- `show` scans the folder to find the file; fine for a personal mailbox.
- Attachments are not indexed; only the text and HTML bodies.
- Not done yet: `thread` command (follow In-Reply-To/References), the MCP server, and the
  `search@` Sieve handler. They would call the same functions.

## Build for Intel Ubuntu 22.04 LTS with Docker

Build an Intel/AMD64 Ubuntu 22.04-compatible Linux binary and export it to `dist/`:

```sh
docker buildx build --platform linux/amd64 --output type=local,dest=./dist .
```

The executable will be `dist/filter-handler`. The host that runs it must provide the configured
`sievec`, `sendmail`, and `git` commands.