//! Embedding providers: Voyage AI (hosted API) and Ollama (local server).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::time::Duration;

/// Voyage tunes vectors differently for stored documents and for search queries.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Document,
    Query,
}

#[derive(Clone)]
pub struct EmbedConfig {
    /// "voyage" or "ollama"
    pub provider: String,
    pub url: String,
    pub model: String,
    pub dim: usize,
    /// Only used by the voyage provider. Read from the environment, never from flags.
    pub api_key: String,
    /// Maximum number of texts per request.
    pub batch: usize,
}

impl EmbedConfig {
    /// Identifies the vector space. Vectors with different fingerprints must never be mixed.
    pub fn fingerprint(&self) -> String {
        format!("{}:{}:{}", self.provider, self.model, self.dim)
    }
}

/// Embeds the given texts. The result has one vector per input, in the same order.
#[tracing::instrument(skip_all, fields(provider = %cfg.provider, model = %cfg.model, inputs = inputs.len()))]
pub fn embed(cfg: &EmbedConfig, inputs: &[String], kind: Kind) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let vectors = match cfg.provider.as_str() {
        "voyage" => {
            if cfg.api_key.is_empty() {
                bail!("VOYAGE_API_KEY is not set");
            }
            let input_type = match kind {
                Kind::Document => "document",
                Kind::Query => "query",
            };
            let body = serde_json::json!({
                "input": inputs,
                "model": cfg.model,
                "input_type": input_type,
                "output_dimension": cfg.dim,
            });
            let resp = post_json(&cfg.url, Some(cfg.api_key.as_str()), &body)?;
            parse_voyage(&resp, inputs.len())?
        }
        "ollama" => {
            let body = serde_json::json!({ "model": cfg.model, "input": inputs });
            let resp = post_json(&cfg.url, None, &body)?;
            parse_ollama(&resp, inputs.len())?
        }
        other => bail!("unknown embedding provider '{other}' (use voyage or ollama)"),
    };
    for v in &vectors {
        if v.len() != cfg.dim {
            bail!(
                "embedding has {} dimensions, expected {} (check EMBED_DIM and the model)",
                v.len(),
                cfg.dim
            );
        }
    }
    Ok(vectors)
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// POSTs JSON and retries on rate limits (429), server errors and network failures.
fn post_json(url: &str, bearer: Option<&str>, body: &Value) -> Result<Value> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let mut req = ureq::post(url).timeout(Duration::from_secs(120));
        if let Some(key) = bearer {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        match req.send_json(body.clone()) {
            Ok(resp) => {
                return resp
                    .into_json::<Value>()
                    .context("invalid JSON in the embedding response");
            }
            Err(ureq::Error::Status(code, resp)) => {
                let retry_after = resp
                    .header("retry-after")
                    .and_then(|v| v.trim().parse::<u64>().ok());
                let text = resp.into_string().unwrap_or_default();
                if (code == 429 || code >= 500) && attempt < 8 {
                    let wait = retry_after.unwrap_or_else(|| (2u64 << attempt.min(5)).min(60));
                    tracing::warn!(code, attempt, wait_s = wait, "embedding API error, retrying");
                    std::thread::sleep(Duration::from_secs(wait));
                    continue;
                }
                let short: String = text.chars().take(300).collect();
                bail!("embedding API error {code}: {short}");
            }
            Err(e) => {
                if attempt < 4 {
                    std::thread::sleep(Duration::from_secs(2u64 << attempt));
                    continue;
                }
                return Err(anyhow::Error::new(e).context("embedding request failed"));
            }
        }
    }
}

fn to_f32(v: &Value) -> Result<Vec<f32>> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow!("bad embedding format"))?;
    Ok(arr
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect())
}

/// Voyage: {"data": [{"index": 0, "embedding": [...]}, ...]}
fn parse_voyage(resp: &Value, n: usize) -> Result<Vec<Vec<f32>>> {
    let data = resp["data"]
        .as_array()
        .ok_or_else(|| anyhow!("no 'data' field in the Voyage response"))?;
    let mut out: Vec<Option<Vec<f32>>> = vec![None; n];
    for item in data {
        let idx = item["index"]
            .as_u64()
            .ok_or_else(|| anyhow!("missing 'index' in the Voyage response"))?
            as usize;
        if idx >= n {
            bail!("unexpected embedding index {idx}");
        }
        out[idx] = Some(to_f32(&item["embedding"])?);
    }
    out.into_iter()
        .collect::<Option<Vec<Vec<f32>>>>()
        .ok_or_else(|| anyhow!("the Voyage response is missing some embeddings"))
}

/// Ollama: {"embeddings": [[...], ...]}
fn parse_ollama(resp: &Value, n: usize) -> Result<Vec<Vec<f32>>> {
    let arr = resp["embeddings"]
        .as_array()
        .ok_or_else(|| anyhow!("no 'embeddings' field in the response"))?;
    if arr.len() != n {
        bail!("got {} embeddings for {} inputs", arr.len(), n);
    }
    arr.iter().map(to_f32).collect()
}
