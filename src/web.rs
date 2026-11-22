//! Web application: Google sign-in, a search form and a message viewer.
//!
//! Blocking server (tiny_http) with a small pool of worker threads. No JavaScript is served;
//! messages are shown as escaped plain text.

use crate::auth::{Auth, AuthConfig, SESSION_TTL_SECS};
use crate::nl;
use crate::search::{self, Filters, Hit};
use crate::Config;
use anyhow::{anyhow, bail, Context, Result};
use std::env;
use std::sync::Arc;

const WORKERS: usize = 8;
const COOKIE_NAME: &str = "mi_session";
const MAX_QUERY_CHARS: usize = 500;
const RESULT_LIMIT: usize = 20;
const MAX_BODY_CHARS: usize = 200_000;

const CSS: &str = "\
:root{color-scheme:light dark;--fg:#1c1c1e;--bg:#fafafa;--muted:#6b6b70;--line:#d8d8dc;--accent:#2457c5}\
@media (prefers-color-scheme:dark){:root{--fg:#ececf0;--bg:#17171a;--muted:#9a9aa2;--line:#33333a;--accent:#7aa2ff}}\
*{box-sizing:border-box}\
body{margin:0;font:16px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg)}\
header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}\
.brand{font-weight:600;color:var(--fg);text-decoration:none}\
.who{display:flex;align-items:center;gap:8px;margin:0;font-size:14px;color:var(--muted)}\
main{max-width:820px;margin:0 auto;padding:16px}\
form.search{display:grid;gap:10px;margin-bottom:20px}\
input[type=text],input[type=search],input[type=date]{width:100%;padding:10px 12px;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:8px}\
button,.btn{padding:9px 14px;font:inherit;color:#fff;background:var(--accent);border:0;border-radius:8px;cursor:pointer;text-decoration:none;display:inline-block}\
.who button{padding:5px 10px;font-size:14px}\
details{border:1px solid var(--line);border-radius:8px;padding:8px 12px}\
details .grid{display:grid;gap:8px;margin-top:8px}\
label{font-size:14px;color:var(--muted)}\
.note{font-size:14px;color:var(--muted);margin:0 0 12px}\
ol.hits{list-style:none;margin:0;padding:0}\
ol.hits li{padding:12px 0;border-bottom:1px solid var(--line)}\
ol.hits a{font-weight:600;color:var(--accent);text-decoration:none}\
.meta{font-size:14px;color:var(--muted)}\
.snip{margin-top:4px;overflow-wrap:anywhere}\
dl.hdr{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:0 0 16px;font-size:14px}\
dl.hdr dt{color:var(--muted)}dl.hdr dd{margin:0;overflow-wrap:anywhere}\
pre.body{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;border-top:1px solid var(--line);padding-top:16px}\
.err{color:#c0392b}";

struct WebConfig {
    listen: String,
    base_url: String,
    secure_cookie: bool,
    anthropic_key: Option<String>,
    anthropic_model: String,
}

struct AppState {
    cfg: Config,
    wc: WebConfig,
    auth: Auth,
}

fn env_required(name: &str) -> Result<String> {
    match env::var(name) {
        Ok(v) if !v.trim().is_empty() => Ok(v.trim().to_string()),
        _ => bail!("{name} must be set"),
    }
}

/// Reads the web settings from the environment (secrets are never passed as flags).
fn load_web_config() -> Result<(WebConfig, AuthConfig)> {
    let base_url = env_required("BASE_URL")?.trim_end_matches('/').to_string();
    if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
        bail!("BASE_URL must start with https:// (or http:// for local tests)");
    }
    let allowed: Vec<String> = env_required("ALLOWED_EMAILS")?
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if allowed.is_empty() {
        bail!("ALLOWED_EMAILS must list at least one address");
    }
    let wc = WebConfig {
        listen: env::var("LISTEN").unwrap_or_else(|_| "127.0.0.1:8080".to_string()),
        secure_cookie: base_url.starts_with("https://"),
        anthropic_key: env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty()),
        anthropic_model: env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-haiku-5-5".to_string()),
        base_url: base_url.clone(),
    };
    let ac = AuthConfig {
        client_id: env_required("GOOGLE_CLIENT_ID")?,
        client_secret: env_required("GOOGLE_CLIENT_SECRET")?,
        redirect_uri: format!("{base_url}/auth/callback"),
        allowed,
    };
    Ok((wc, ac))
}

pub(crate) fn serve(cfg: Config) -> Result<()> {
    let (wc, ac) = load_web_config()?;
    {
        // Create the schema once and refuse to run against an index built with other embeddings.
        let conn = crate::open_db(&cfg)?;
        crate::check_fingerprint(&conn, &cfg.embed)?;
    }
    if wc.anthropic_key.is_none() {
        tracing::warn!("ANTHROPIC_API_KEY is not set: natural-language interpretation is disabled");
    }
    let listen = wc.listen.clone();
    let base_url = wc.base_url.clone();
    let state = Arc::new(AppState {
        cfg,
        wc,
        auth: Auth::new(ac),
    });
    let server = tiny_http::Server::http(listen.as_str())
        .map_err(|e| anyhow!("cannot listen on {listen}: {e}"))?;
    let server = Arc::new(server);
    tracing::info!(%listen, %base_url, "listening");

    let mut handles = Vec::new();
    for _ in 0..WORKERS {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            for request in server.incoming_requests() {
                // A panic in one request must not kill the worker.
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handle(&state, request)
                }))
                .is_err()
                {
                    tracing::error!("panic while handling a request");
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

// ---------------------------------------------------------------- HTTP plumbing

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Reply {
    fn html(status: u16, body: String) -> Reply {
        Reply {
            status,
            headers: vec![(
                "Content-Type".to_string(),
                "text/html; charset=utf-8".to_string(),
            )],
            body,
        }
    }

    fn text(status: u16, body: &str) -> Reply {
        Reply {
            status,
            headers: vec![(
                "Content-Type".to_string(),
                "text/plain; charset=utf-8".to_string(),
            )],
            body: body.to_string(),
        }
    }

    fn redirect(to: &str) -> Reply {
        Reply {
            status: 303,
            headers: vec![("Location".to_string(), to.to_string())],
            body: String::new(),
        }
    }

    fn with_header(mut self, name: &str, value: String) -> Reply {
        self.headers.push((name.to_string(), value));
        self
    }
}

const SECURITY_HEADERS: [(&str, &str); 5] = [
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    (
        "Content-Security-Policy",
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; \
         frame-ancestors 'none'; base-uri 'none'",
    ),
    ("X-Frame-Options", "DENY"),
];

fn send(request: tiny_http::Request, reply: Reply) {
    let mut resp = tiny_http::Response::from_string(reply.body).with_status_code(reply.status);
    for (k, v) in &reply.headers {
        if let Ok(h) = tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()) {
            resp.add_header(h);
        }
    }
    for (k, v) in SECURITY_HEADERS.iter() {
        if let Ok(h) = tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()) {
            resp.add_header(h);
        }
    }
    let _ = request.respond(resp);
}

fn header_value(request: &tiny_http::Request, name: &str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.to_string())
}

fn split_url(url: &str) -> (&str, &str) {
    match url.split_once('?') {
        Some((p, q)) => (p, q),
        None => (url, ""),
    }
}

fn cookie_get(header: &str, name: &str) -> Option<String> {
    header.split(';').find_map(|part| {
        let (k, v) = part.trim().split_once('=')?;
        if k == name {
            Some(v.to_string())
        } else {
            None
        }
    })
}

fn param<'a>(params: &'a [(String, String)], key: &str) -> &'a str {
    params
        .iter()
        .find(|(k, _)| k.as_str() == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or("")
}

fn nonempty(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn handle(st: &AppState, request: tiny_http::Request) {
    let is_get = *request.method() == tiny_http::Method::Get;
    let is_post = *request.method() == tiny_http::Method::Post;
    let raw_url = request.url().to_string();
    let cookie = header_value(&request, "Cookie");

    let (path, query) = split_url(&raw_url);
    let params: Vec<(String, String)> = form_urlencoded::parse(query.as_bytes())
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    // The query string is left out on purpose: it holds the user's search text.
    let span = tracing::info_span!(
        "http.request",
        http.method = %request.method(),
        http.path = path,
        http.status_code = tracing::field::Empty,
    );
    let _entered = span.enter();
    let sid = cookie.as_deref().and_then(|c| cookie_get(c, COOKIE_NAME));
    let user = sid.as_deref().and_then(|s| st.auth.session_email(s));

    let reply = match (is_get, is_post, path) {
        (true, _, "/healthz") => Reply::text(200, "ok"),
        (true, _, "/login") => page_login(st),
        (true, _, "/auth/callback") => page_callback(st, &params),
        (_, true, "/logout") => page_logout(st, sid.as_deref()),
        (true, _, "/") => page_home(st, user.as_deref(), &params),
        (true, _, p) if p.starts_with("/m/") => page_message(st, user.as_deref(), &p[3..]),
        _ => error_page(404, user.as_deref(), "Page introuvable."),
    };
    span.record("http.status_code", reply.status);
    if path == "/healthz" {
        tracing::debug!(status = reply.status, "request");
    } else {
        tracing::info!(status = reply.status, "request");
    }
    send(request, reply);
}

// ---------------------------------------------------------------- HTML

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            _ => o.push(c),
        }
    }
    o
}

fn layout(title: &str, user: Option<&str>, content: &str) -> String {
    let who = match user {
        Some(u) => format!(
            "<form class=\"who\" method=\"post\" action=\"/logout\"><span>{}</span>\
             <button type=\"submit\">Déconnexion</button></form>",
            esc(u)
        ),
        None => String::new(),
    };
    format!(
        "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>{}</title><style>{}</style></head><body>\
         <header><a class=\"brand\" href=\"/\">Recherche de courriels</a>{}</header>\
         <main>{}</main></body></html>",
        esc(title),
        CSS,
        who,
        content
    )
}

fn error_page(status: u16, user: Option<&str>, msg: &str) -> Reply {
    let body = layout(
        "Erreur",
        user,
        &format!(
            "<p class=\"err\">{}</p><p><a href=\"/\">Retour</a></p>",
            esc(msg)
        ),
    );
    Reply::html(status, body)
}

fn search_form(
    q: &str,
    params: &[(String, String)],
    nl_available: bool,
    nl_checked: bool,
) -> String {
    let nl = if nl_available {
        format!(
            "<label><input type=\"checkbox\" name=\"nl\" value=\"1\"{}> \
             Comprendre la phrase avec Claude (dates, expéditeur)</label>",
            if nl_checked { " checked" } else { "" }
        )
    } else {
        String::new()
    };
    let open = if !param(params, "from").is_empty()
        || !param(params, "to").is_empty()
        || !param(params, "sender").is_empty()
    {
        " open"
    } else {
        ""
    };
    format!(
        "<form class=\"search\" method=\"get\" action=\"/\">\
         <input type=\"search\" name=\"q\" value=\"{q}\" maxlength=\"{max}\" \
          placeholder=\"Ex. : courriel du trésorier sur la facture de septembre\" autofocus>\
         {nl}\
         <details{open}><summary>Filtres</summary><div class=\"grid\">\
         <label>Du <input type=\"date\" name=\"from\" value=\"{from}\"></label>\
         <label>Avant le <input type=\"date\" name=\"to\" value=\"{to}\"></label>\
         <label>Expéditeur <input type=\"text\" name=\"sender\" value=\"{sender}\"></label>\
         </div></details>\
         <div><button type=\"submit\">Chercher</button></div></form>",
        q = esc(q),
        max = MAX_QUERY_CHARS,
        nl = nl,
        open = open,
        from = esc(param(params, "from")),
        to = esc(param(params, "to")),
        sender = esc(param(params, "sender")),
    )
}

fn render_results(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "<p>Aucun résultat.</p>".to_string();
    }
    let mut out = String::from("<ol class=\"hits\">");
    for h in hits {
        let subject = if h.subject.is_empty() {
            "(sans sujet)"
        } else {
            h.subject.as_str()
        };
        let sender = if h.from_name.is_empty() {
            esc(&h.from_addr)
        } else {
            format!("{} &lt;{}&gt;", esc(&h.from_name), esc(&h.from_addr))
        };
        let date: String = h.date.chars().take(16).collect();
        out.push_str(&format!(
            "<li><a href=\"/m/{id}\">{subject}</a>\
             <div class=\"meta\">{date} UTC · {sender} · {folder}</div>\
             <div class=\"snip\">{snippet}</div></li>",
            id = esc(&h.id),
            subject = esc(subject),
            date = esc(&date),
            sender = sender,
            folder = esc(&h.folder),
            snippet = esc(&h.snippet),
        ));
    }
    out.push_str("</ol>");
    out
}

// ---------------------------------------------------------------- pages

fn page_login(st: &AppState) -> Reply {
    match st.auth.login_url() {
        Ok(url) => Reply::redirect(&url),
        Err(e) => {
            tracing::error!("login failed: {e:#}");
            error_page(503, None, "Connexion momentanément impossible.")
        }
    }
}

fn page_callback(st: &AppState, params: &[(String, String)]) -> Reply {
    if !param(params, "error").is_empty() {
        return error_page(403, None, "Connexion annulée ou refusée par Google.");
    }
    let code = param(params, "code");
    let state = param(params, "state");
    if code.is_empty() || state.is_empty() {
        return error_page(400, None, "Requête de connexion invalide.");
    }
    match st.auth.finish_login(code, state) {
        Ok((sid, email)) => {
            tracing::info!(%email, "signed in");
            let secure = if st.wc.secure_cookie { "; Secure" } else { "" };
            Reply::redirect("/").with_header(
                "Set-Cookie",
                format!(
                    "{COOKIE_NAME}={sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_TTL_SECS}{secure}"
                ),
            )
        }
        Err(e) => {
            // Details stay in the log; the browser only learns that access was refused.
            tracing::warn!("sign-in refused: {e:#}");
            error_page(403, None, "Accès refusé.")
        }
    }
}

fn page_logout(st: &AppState, sid: Option<&str>) -> Reply {
    if let Some(sid) = sid {
        st.auth.logout(sid);
    }
    let secure = if st.wc.secure_cookie { "; Secure" } else { "" };
    Reply::redirect("/").with_header(
        "Set-Cookie",
        format!("{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0{secure}"),
    )
}

fn page_home(st: &AppState, user: Option<&str>, params: &[(String, String)]) -> Reply {
    let Some(user) = user else {
        let content = "<p>Cette application est privée.</p>\
                       <p><a class=\"btn\" href=\"/login\">Se connecter avec Google</a></p>";
        return Reply::html(200, layout("Connexion", None, content));
    };

    let q: String = param(params, "q")
        .trim()
        .chars()
        .take(MAX_QUERY_CHARS)
        .collect();
    let nl_available = st.wc.anthropic_key.is_some();
    // The box is ticked on the first visit and keeps the user's choice afterwards.
    let nl_checked = nl_available && (q.is_empty() || param(params, "nl") == "1");

    let mut content = search_form(&q, params, nl_available, nl_checked);
    if !q.is_empty() {
        match run_search(st, &q, params, nl_checked) {
            Ok((hits, note)) => {
                if let Some(n) = note {
                    content.push_str(&format!("<p class=\"note\">{}</p>", esc(&n)));
                }
                content.push_str(&render_results(&hits));
            }
            Err(e) => {
                tracing::error!("search failed: {e:#}");
                content.push_str(
                    "<p class=\"err\">La recherche a échoué. Voir le journal du serveur.</p>",
                );
            }
        }
    }
    Reply::html(200, layout("Recherche de courriels", Some(user), &content))
}

/// Runs the search, optionally interpreting the text with Claude first.
/// Filters typed in the form take priority over those inferred by Claude.
fn run_search(
    st: &AppState,
    q: &str,
    params: &[(String, String)],
    use_nl: bool,
) -> Result<(Vec<Hit>, Option<String>)> {
    let conn = crate::open_conn(&st.cfg.db)?;
    let mut filters = Filters {
        from: nonempty(param(params, "from")),
        to: nonempty(param(params, "to")),
        sender: nonempty(param(params, "sender")),
    };
    let mut text = q.to_string();
    let mut note = None;

    if use_nl {
        if let Some(key) = st.wc.anthropic_key.as_deref() {
            let today: String = conn
                .query_row("SELECT date('now', 'localtime')", [], |r| r.get(0))
                .context("cannot read the date")?;
            match nl::interpret(key, &st.wc.anthropic_model, q, &today) {
                Ok(i) => {
                    if !i.query.trim().is_empty() {
                        text = i.query.clone();
                    }
                    if filters.from.is_none() {
                        filters.from = i.from.clone();
                    }
                    if filters.to.is_none() {
                        filters.to = i.to.clone();
                    }
                    if filters.sender.is_none() {
                        filters.sender = i.sender.clone();
                    }
                    note = Some(describe(&text, &filters));
                }
                Err(e) => {
                    tracing::warn!("Claude interpretation failed: {e:#}");
                    note = Some("Claude est indisponible : recherche directe.".to_string());
                }
            }
        }
    }

    let hits = search::search(&conn, &st.cfg.embed, &text, &filters, RESULT_LIMIT)?;
    Ok((hits, note))
}

fn describe(text: &str, f: &Filters) -> String {
    let mut s = format!("Compris : « {text} »");
    if let Some(from) = &f.from {
        s.push_str(&format!(", à partir du {from}"));
    }
    if let Some(to) = &f.to {
        s.push_str(&format!(", avant le {to}"));
    }
    if let Some(sender) = &f.sender {
        s.push_str(&format!(", expéditeur « {sender} »"));
    }
    s
}

fn page_message(st: &AppState, user: Option<&str>, id: &str) -> Reply {
    let Some(user) = user else {
        return Reply::redirect("/");
    };
    let conn = match crate::open_conn(&st.cfg.db) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("cannot open the database: {e:#}");
            return error_page(500, Some(user), "Base de données inaccessible.");
        }
    };
    let view = match crate::lookup_message(&conn, &st.cfg.maildir, id) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("message lookup failed: {e:#}");
            return error_page(404, Some(user), "Courriel introuvable.");
        }
    };
    let p = &view.parsed;
    let sender = if p.from_name.is_empty() {
        esc(&p.from_addr)
    } else {
        format!("{} &lt;{}&gt;", esc(&p.from_name), esc(&p.from_addr))
    };
    let subject = if p.subject.is_empty() {
        "(sans sujet)"
    } else {
        p.subject.as_str()
    };
    let body: String = p.body.chars().take(MAX_BODY_CHARS).collect();
    let content = format!(
        "<dl class=\"hdr\"><dt>Sujet</dt><dd>{subject}</dd><dt>De</dt><dd>{sender}</dd>\
         <dt>À</dt><dd>{to}</dd><dt>Date</dt><dd>{date} UTC</dd>\
         <dt>Dossier</dt><dd>{folder}</dd><dt>Message-ID</dt><dd>{mid}</dd></dl>\
         <pre class=\"body\">{body}</pre>",
        subject = esc(subject),
        sender = sender,
        to = esc(&p.to),
        date = esc(&view.date),
        folder = esc(&view.folder),
        mid = esc(view.message_id.as_deref().unwrap_or("")),
        body = esc(&body),
    );
    Reply::html(200, layout(subject, Some(user), &content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esc_escapes_html() {
        assert_eq!(
            esc("<a href=\"x\">&'</a>"),
            "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;"
        );
    }

    #[test]
    fn cookie_get_finds_named_cookie() {
        assert_eq!(
            cookie_get("a=1; mi_session=abc; b=2", "mi_session").as_deref(),
            Some("abc")
        );
        assert_eq!(cookie_get("a=1", "mi_session"), None);
    }

    #[test]
    fn split_url_separates_query() {
        assert_eq!(split_url("/?q=x"), ("/", "q=x"));
        assert_eq!(split_url("/m/abc"), ("/m/abc", ""));
    }

    #[test]
    fn describe_lists_filters() {
        let f = Filters {
            from: Some("2026-09-01".into()),
            to: Some("2026-10-01".into()),
            sender: Some("trésorier".into()),
        };
        let d = describe("facture", &f);
        assert!(d.contains("facture") && d.contains("2026-09-01") && d.contains("trésorier"));
    }
}
