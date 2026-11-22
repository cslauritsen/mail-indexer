//! Logging and tracing, configured from the environment.
//!
//! Logs go to stderr (stdout is reserved for command output):
//!   RUST_LOG      filter, e.g. `info` (default), `debug`, `mail_indexer=debug,warn`
//!   LOG_FORMAT    `text` (default) or `json`
//!
//! Traces are exported over OTLP/HTTP (protobuf) when an endpoint is configured, using the
//! standard OpenTelemetry variables:
//!   OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT   enables export
//!   OTEL_EXPORTER_OTLP_HEADERS, OTEL_EXPORTER_OTLP_TIMEOUT             as per the OTel spec
//!   OTEL_SERVICE_NAME (default `mail-indexer`), OTEL_RESOURCE_ATTRIBUTES
//!   OTEL_TRACES_SAMPLER, OTEL_TRACES_SAMPLER_ARG                       sampling
//!   OTEL_TRACES_EXPORTER=none                                          disables export
//!
//! Search text and mail content are never recorded in spans, only lengths and counts.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::{trace::SdkTracerProvider, Resource};
use std::env;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

/// Flushes pending spans when dropped; keep it alive for the whole program.
pub struct Guard {
    provider: Option<SdkTracerProvider>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(p) = self.provider.take() {
            if let Err(e) = p.shutdown() {
                eprintln!("could not flush traces: {e}");
            }
        }
    }
}

fn otlp_enabled() -> bool {
    let set = |k: &str| env::var(k).map(|v| !v.trim().is_empty()).unwrap_or(false);
    let disabled = env::var("OTEL_TRACES_EXPORTER")
        .map(|v| v.trim().eq_ignore_ascii_case("none"))
        .unwrap_or(false);
    !disabled && (set("OTEL_EXPORTER_OTLP_ENDPOINT") || set("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
}

fn build_provider() -> Result<SdkTracerProvider, Box<dyn std::error::Error>> {
    let exporter = SpanExporter::builder()
        .with_http()
        .build()
        .map_err(|e| format!("cannot create the OTLP exporter: {e}"))?;
    let service = env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "mail-indexer".to_string());
    let resource = Resource::builder().with_service_name(service).build();
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build())
}

/// Installs the global subscriber. Call once, after the environment (.env) is loaded.
pub fn init() -> Guard {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let json = env::var("LOG_FORMAT")
        .map(|v| v.trim().eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    let fmt_layer = if json {
        fmt::layer().json().with_writer(std::io::stderr).boxed()
    } else {
        fmt::layer().with_writer(std::io::stderr).boxed()
    };

    let mut provider = None;
    let mut otel_error = None;
    if otlp_enabled() {
        match build_provider() {
            Ok(p) => provider = Some(p),
            Err(e) => otel_error = Some(e.to_string()),
        }
    }
    let otel_layer = provider
        .as_ref()
        .map(|p| tracing_opentelemetry::layer().with_tracer(p.tracer("mail-indexer")));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    match (&provider, otel_error) {
        (Some(_), _) => tracing::info!("trace export to OTLP is enabled"),
        (None, Some(e)) => tracing::warn!("trace export disabled: {e}"),
        _ => {}
    }
    Guard { provider }
}
