# mail-indexer

Indexes a Dovecot Maildir into one SQLite file (FTS5 for exact words, sqlite-vec for meaning),
searches it with hybrid ranking (Reciprocal Rank Fusion), and serves a small web application
with Google sign-in. Natural-language questions such as "the church treasurer's email about the
September invoice" are understood by the Claude API.

## Requirements

- Rust toolchain and a C compiler (SQLite and sqlite-vec are built from source)
- A [Voyage AI](https://www.voyageai.com) API key for embeddings (hosted, nothing to run locally).
  A local Ollama server with `bge-m3` still works as an alternative.
- For the web application: a Google OAuth client, and optionally an Anthropic API key

## Command line

```
export MAILDIR=/var/mail/chad/Maildir
export DB=/var/lib/mail-index/mail.db
export VOYAGE_API_KEY=...

mail-indexer index                      # first run embeds everything; later runs are incremental
mail-indexer search "facture de septembre du trésorier" --from 2026-09-01 --to 2026-10-01
mail-indexer show a3f9c1d2e4b5          # id prefix as printed by search
mail-indexer reset --yes                # empty the index (needed when the embedding model changes)
```

| Variable | Default | Meaning |
|---|---|---|
| `MAILDIR` | | Maildir root (required for `index`, `show`, `serve`) |
| `DB` | `mail-index.db` | SQLite file |
| `SKIP_FOLDERS` | | Folders to ignore, e.g. `.Junk,.Trash` |
| `VOYAGE_API_KEY` | | Selects the `voyage` provider when set |
| `EMBED_PROVIDER` | `voyage` if a key is set, else `ollama` | `voyage` or `ollama` |
| `EMBED_MODEL` | `voyage-4` (`bge-m3` for ollama) | Embedding model |
| `EMBED_DIM` | `1024` | Vector size; must be supported by the model |
| `EMBED_URL` | Voyage API (`http://localhost:11434/api/embed` for ollama) | Endpoint |
| `EMBED_BATCH` | `64` | Texts per embedding request |

Vectors from different models cannot be compared. The index remembers which provider, model and
size built it and refuses to run with other settings; run `reset --yes` and then `index` to
rebuild. An index created by an earlier version (ollama, `bge-m3`) is recognised automatically.

### Voyage notes

- Without a payment method Voyage limits accounts to 3 requests and 10,000 tokens per minute,
  which makes the first indexing of a real mailbox impractical. Adding a payment method raises
  the limits to tier 1, and the first 200 million tokens are still free.
- Requests are batched and retried on HTTP 429/5xx, honouring `Retry-After`. If a run stops, run
  `index` again: it resumes where it stopped.
- The text of your mail is sent to Voyage while indexing. Read their terms and data-use settings
  first. Only the search text is sent when you search.

## Web application

```
export BASE_URL=https://mail.example.com
export GOOGLE_CLIENT_ID=...
export GOOGLE_CLIENT_SECRET=...
export ALLOWED_EMAILS=you@example.com,partner@example.com
export ANTHROPIC_API_KEY=...            # optional: enables natural-language interpretation
mail-indexer serve                      # listens on 127.0.0.1:8080
```

| Variable | Default | Meaning |
|---|---|---|
| `BASE_URL` | | Public URL, no trailing slash. Used for the Google redirect URI. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | | Google OAuth client |
| `ALLOWED_EMAILS` | | Comma-separated Google addresses that may sign in |
| `LISTEN` | `127.0.0.1:8080` | Address to bind |
| `ANTHROPIC_API_KEY` | | Enables the "understand the sentence with Claude" option |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Model used to interpret questions |

Run `index` regularly (cron or a systemd timer); the web application only reads the index.
Run `index` and `serve` as the same user: SQLite's write-ahead log needs write access to the
database directory even for readers.

### Logging and tracing

Logs go to stderr (stdout is command output). Traces are exported over OTLP when configured.
Everything is set from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `RUST_LOG` | `info` | Log filter, e.g. `debug` or `mail_indexer=debug,warn` |
| `LOG_FORMAT` | `text` | `json` for one JSON object per line |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | OTLP/HTTP collector, e.g. `http://localhost:4318`. Setting it (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) turns trace export on |
| `OTEL_SERVICE_NAME` | `mail-indexer` | Service name on the traces |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | parent-based, always on | Sampling, e.g. `parentbased_traceidratio` and `0.1` |
| `OTEL_EXPORTER_OTLP_HEADERS` | | Auth headers for a hosted collector |
| `OTEL_TRACES_EXPORTER` | | `none` disables export even if an endpoint is set |

Only OTLP over HTTP/protobuf is supported (not gRPC). Each web request is a span
(`http.request`, with method, path and status; the query string is left out) containing the
`search`, `embed` and `interpret` spans. Search text and mail content are never recorded,
only lengths and counts. Spans are flushed when a command exits.

### Google setup

1. In the Google Cloud console, create an OAuth client of type **Web application**.
2. Add the authorized redirect URI `https://<your host>/auth/callback` (it must equal
   `BASE_URL` + `/auth/callback`).
3. On the consent screen, use scopes `openid` and `email`. While the app is in "Testing",
   list your addresses as test users.

### Cloudflare Tunnel

Point a public hostname at the local listener, for example in `config.yml`:

```yaml
ingress:
  - hostname: mail.example.com
    service: http://127.0.0.1:8080
  - service: http_status:404
```

`deploy/mail-indexer.service` is a systemd unit that reads its settings and secrets from
`/etc/mail-indexer.env` (see `.env.example`; keep that file readable by root only).

### Security notes

- Only the Google accounts in `ALLOWED_EMAILS` get in, and only with a verified address.
  Sign-in uses the authorization code flow with PKCE, `state` and `nonce`.
- The ID token comes straight from Google's token endpoint over TLS, so its signature is not
  checked (OpenID Connect Core 3.1.3.7, step 6). Issuer, audience, expiry, nonce and
  `email_verified` are checked.
- Sessions live in memory (cookie: `HttpOnly`, `SameSite=Lax`, `Secure` under https, 12 hours).
  Restarting the server signs everyone out.
- Pages ship a strict Content-Security-Policy and no JavaScript. Mail is shown as escaped text.
- Only the question text is sent to Anthropic, never mail content. The Claude step is optional.

## How it works

- Message id = sha256(Message-ID), falling back to sha256(raw file). Location is stored as
  (folder, filename without the `:2,FLAGS` suffix), so flag changes do not matter.
- `index` adds new files, updates the folder when a message moved, and removes messages
  whose file is gone.
- Quoted replies and signatures are dropped before chunking (about 1200 characters, at most
  8 chunks per message). Subject and sender are added to the text that gets embedded.
- Search runs a vector query and an FTS5 query (100 candidates each), fuses the ranks,
  then applies the date and sender filters. Filters typed in the form win over those that
  Claude infers from the question.

## Known limits

- The same Message-ID in two folders (copies) counts as one message; its stored location
  flips between the copies on each run. Harmless, but noisy.
- `show` and the web viewer scan the folder to find the file; fine for a personal mailbox.
- Attachments are not indexed; only the text and HTML bodies.
- Sessions are not shared between processes and are lost on restart.
- Not done yet: a `thread` view (follow In-Reply-To/References), the MCP server, and the
  `search@` Sieve handler. They would call the same functions.

## Build for Intel Ubuntu 22.04 LTS with Docker

Build an Intel/AMD64 Ubuntu 22.04-compatible Linux binary and export it to `dist/`:

```sh
docker buildx build --platform linux/amd64 --output type=local,dest=./dist .
```

The executable will be `dist/filter-handler`. The host that runs it must provide the configured
`sievec`, `sendmail`, and `git` commands.
