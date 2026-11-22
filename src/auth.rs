//! Google sign-in (OpenID Connect, authorization code flow with PKCE) and server-side sessions.
//!
//! The ID token is received directly from Google's token endpoint over TLS, so its signature
//! is not checked: OpenID Connect Core 3.1.3.7 (step 6) allows TLS server validation to
//! replace the signature check in that case. The issuer, audience, expiry, nonce and
//! email_verified claims are all checked, and the email must be on the allow list.

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const SESSION_TTL_SECS: u64 = 12 * 3600;
const PENDING_TTL_SECS: u64 = 600;
const MAX_PENDING: usize = 1000;

pub struct AuthConfig {
    pub client_id: String,
    pub client_secret: String,
    /// Must match an authorized redirect URI of the Google OAuth client exactly.
    pub redirect_uri: String,
    /// Lower-case email addresses that may sign in.
    pub allowed: Vec<String>,
}

struct Pending {
    nonce: String,
    verifier: String,
    created: u64,
}

struct Session {
    email: String,
    expires: u64,
}

pub struct Auth {
    cfg: AuthConfig,
    pending: Mutex<HashMap<String, Pending>>,
    sessions: Mutex<HashMap<String, Session>>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Locks a mutex, recovering the data if another thread panicked while holding it.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `n` random bytes from the OS, encoded as URL-safe base64.
fn random_token(n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    std::fs::File::open("/dev/urandom")
        .context("cannot open /dev/urandom")?
        .read_exact(&mut buf)
        .context("cannot read /dev/urandom")?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

impl Auth {
    pub fn new(cfg: AuthConfig) -> Auth {
        Auth {
            cfg,
            pending: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Starts a sign-in: remembers state, nonce and PKCE verifier, returns Google's URL.
    pub fn login_url(&self) -> Result<String> {
        let state = random_token(32)?;
        let nonce = random_token(32)?;
        let verifier = random_token(48)?;
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

        {
            let mut pending = lock(&self.pending);
            let t = now();
            pending.retain(|_, p| t.saturating_sub(p.created) < PENDING_TTL_SECS);
            if pending.len() >= MAX_PENDING {
                bail!("too many sign-in attempts in progress");
            }
            pending.insert(
                state.clone(),
                Pending {
                    nonce: nonce.clone(),
                    verifier,
                    created: t,
                },
            );
        }

        let mut ser = form_urlencoded::Serializer::new(String::new());
        ser.append_pair("client_id", &self.cfg.client_id);
        ser.append_pair("redirect_uri", &self.cfg.redirect_uri);
        ser.append_pair("response_type", "code");
        ser.append_pair("scope", "openid email");
        ser.append_pair("state", &state);
        ser.append_pair("nonce", &nonce);
        ser.append_pair("code_challenge", &challenge);
        ser.append_pair("code_challenge_method", "S256");
        ser.append_pair("prompt", "select_account");
        let query = ser.finish();
        Ok(format!("{AUTH_URL}?{query}"))
    }

    /// Completes a sign-in. Returns (session id, email) on success.
    pub fn finish_login(&self, code: &str, state: &str) -> Result<(String, String)> {
        let pending = lock(&self.pending)
            .remove(state)
            .ok_or_else(|| anyhow!("unknown or already used state"))?;
        if now().saturating_sub(pending.created) >= PENDING_TTL_SECS {
            bail!("the sign-in attempt expired");
        }

        let resp = ureq::post(TOKEN_URL)
            .timeout(Duration::from_secs(20))
            .send_form(&[
                ("code", code),
                ("client_id", self.cfg.client_id.as_str()),
                ("client_secret", self.cfg.client_secret.as_str()),
                ("redirect_uri", self.cfg.redirect_uri.as_str()),
                ("grant_type", "authorization_code"),
                ("code_verifier", pending.verifier.as_str()),
            ]);
        let tokens: Value = match resp {
            Ok(r) => r
                .into_json::<Value>()
                .context("invalid JSON from the token endpoint")?,
            Err(ureq::Error::Status(code, r)) => {
                let msg: String = r
                    .into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(200)
                    .collect();
                bail!("token endpoint returned {code}: {msg}");
            }
            Err(e) => return Err(anyhow::Error::new(e).context("token request failed")),
        };
        let id_token = tokens["id_token"]
            .as_str()
            .ok_or_else(|| anyhow!("no id_token in the token response"))?;

        let claims = decode_claims(id_token)?;
        let email = check_claims(
            &claims,
            &self.cfg.client_id,
            &pending.nonce,
            &self.cfg.allowed,
            now(),
        )?;

        let sid = random_token(32)?;
        let mut sessions = lock(&self.sessions);
        let t = now();
        sessions.retain(|_, s| s.expires > t);
        sessions.insert(
            sid.clone(),
            Session {
                email: email.clone(),
                expires: t + SESSION_TTL_SECS,
            },
        );
        Ok((sid, email))
    }

    /// The signed-in email for a session id, if the session exists and has not expired.
    pub fn session_email(&self, sid: &str) -> Option<String> {
        let sessions = lock(&self.sessions);
        sessions
            .get(sid)
            .filter(|s| s.expires > now())
            .map(|s| s.email.clone())
    }

    pub fn logout(&self, sid: &str) {
        lock(&self.sessions).remove(sid);
    }
}

/// Decodes the payload of a JWT (no signature check, see the module comment).
fn decode_claims(id_token: &str) -> Result<Value> {
    let payload = id_token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("malformed id_token"))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .context("bad base64 in the id_token")?;
    serde_json::from_slice(&bytes).context("bad JSON in the id_token")
}

/// Validates the claims and returns the lower-case email when the user may sign in.
fn check_claims(
    claims: &Value,
    client_id: &str,
    nonce: &str,
    allowed: &[String],
    now_secs: u64,
) -> Result<String> {
    let iss = claims["iss"].as_str().unwrap_or("");
    if iss != "https://accounts.google.com" && iss != "accounts.google.com" {
        bail!("unexpected issuer '{iss}'");
    }
    let aud_ok = match &claims["aud"] {
        Value::String(s) => s == client_id,
        Value::Array(a) => a.iter().any(|x| x.as_str() == Some(client_id)),
        _ => false,
    };
    if !aud_ok {
        bail!("the id_token was issued for another client");
    }
    match claims["exp"].as_u64() {
        Some(exp) if exp > now_secs => {}
        _ => bail!("the id_token has expired"),
    }
    if claims["nonce"].as_str() != Some(nonce) {
        bail!("nonce mismatch");
    }
    if claims["email_verified"].as_bool() != Some(true) {
        bail!("the Google email address is not verified");
    }
    let email = claims["email"]
        .as_str()
        .ok_or_else(|| anyhow!("no email in the id_token"))?
        .to_lowercase();
    if !allowed.iter().any(|a| *a == email) {
        bail!("{email} is not on the allow list");
    }
    Ok(email)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims() -> Value {
        serde_json::json!({
            "iss": "https://accounts.google.com",
            "aud": "client-1",
            "exp": 2000,
            "nonce": "n1",
            "email_verified": true,
            "email": "Chad@Example.com",
        })
    }

    fn allowed() -> Vec<String> {
        vec!["chad@example.com".to_string()]
    }

    #[test]
    fn accepts_valid_claims_and_lowercases_email() {
        let e = check_claims(&claims(), "client-1", "n1", &allowed(), 1000).unwrap();
        assert_eq!(e, "chad@example.com");
    }

    #[test]
    fn rejects_wrong_nonce_audience_expiry_issuer_and_unlisted_email() {
        assert!(check_claims(&claims(), "client-1", "other", &allowed(), 1000).is_err());
        assert!(check_claims(&claims(), "client-2", "n1", &allowed(), 1000).is_err());
        assert!(check_claims(&claims(), "client-1", "n1", &allowed(), 3000).is_err());
        assert!(check_claims(&claims(), "client-1", "n1", &[], 1000).is_err());
        let mut c = claims();
        c["iss"] = serde_json::json!("https://evil.example");
        assert!(check_claims(&c, "client-1", "n1", &allowed(), 1000).is_err());
        let mut c = claims();
        c["email_verified"] = serde_json::json!(false);
        assert!(check_claims(&c, "client-1", "n1", &allowed(), 1000).is_err());
    }

    #[test]
    fn decodes_jwt_payload() {
        let payload = URL_SAFE_NO_PAD.encode(br#"{"email":"a@b.c"}"#);
        let token = format!("xxx.{payload}.yyy");
        assert_eq!(decode_claims(&token).unwrap()["email"], "a@b.c");
        assert!(decode_claims("nodots").is_err());
    }
}
