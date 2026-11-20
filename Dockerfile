# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 rust:1-bullseye AS rust-toolchain

FROM --platform=linux/amd64 ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}"

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

ADD keyfactorswroot.pem /etc/ssl/certs/keyfactorswroot.pem

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY schema.sql ./

RUN cargo build --release --locked \
    && install -D target/release/mail-indexer /out/mail-indexer

FROM scratch AS artifact
COPY --from=build /out/mail-indexer /mail-indexer
