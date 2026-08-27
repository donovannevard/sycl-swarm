# sycl-swarm

Local LLM inference on an Intel Arc iGPU, and the services built on it. Everything
runs on one Nobara Linux desktop — no cloud APIs, no external databases.

This repo is the **configuration and build record for the inference stack**. It is
deliberately hardware-specific: the whole point was getting `llama.cpp` running well
on Intel's SYCL backend on integrated graphics. On different hardware you would take
this repo as a starting point and change the build and driver layer, not run it as-is.

## Status

A personal build record, published so it survives this machine. It is not a
maintained project and there is nothing to sign up to — no issue triage, no
roadmap, no promises that a later llama.cpp still builds against it.

It is MIT licensed, so if any of it is useful to you, take it. The parts most
likely to be worth reading are in [`docs/agent-forge.md`](docs/agent-forge.md):
the SYCL benchmark numbers on an Arc iGPU, and the caveats that cost real time —
oneMKL being a hard requirement, `$ORIGIN`-relative RPATH breaking a copied
binary, and a homeless system user making model loads hang with no error at all.

## What runs

| Component | What | Where |
|---|---|---|
| `llama-swap` | Model router in front of `llama.cpp`'s SYCL backend | systemd service, `svc-agent-forge`, `0.0.0.0:8090` (API-key protected) |
| `llama-server` (×N, on demand) | The actual SYCL inference backends | spawned per request, `127.0.0.1:10001+` |
| Open WebUI | LAN-facing chat UI with its own login | Docker, `--network host`, `0.0.0.0:3000` |
| `media-digest` | Skeptical-persona news digest into Open WebUI | systemd timer, **disabled** by default — kept as a reference job |
| Continue.dev | VS Code coding assistant on the local models | extension config |

Six models are registered — Qwen2.5 3B/7B/Coder-7B, DeepSeek-Coder-V2-Lite,
DeepSeek-R1-Distill-14B and nomic-embed-text — auto-unloading after 30 minutes idle.
The GGUF files themselves are not in git (~30GB); `configs/llama-swap.yaml` names
exactly which ones and where they came from.

## The endpoint

Everything here exists to serve one thing: an OpenAI-compatible endpoint at
`http://<host>:8090/v1`, with an API key from `/etc/agent-forge/llama-swap.env`.
Anything speaking that protocol can use it — any chat UI, editor plugin, or
application pointed at the URL and key.

## Install

```bash
sudo ./install.sh          # service account, dirs, API key, config, unit
sudo ./uninstall.sh        # tear it down again
```

`install.sh` sets up everything it can and is safe to re-run — an existing API
key, config and models are never overwritten. It deliberately does **not** build
`llama.cpp` against SYCL or download ~22GB of weights: the build is
hardware-specific and the weights come from HuggingFace. It checks for both and
prints exactly what is missing and how to get it.

`uninstall.sh` shows what it will remove and asks first. By default it keeps two
things it would be rude to delete unasked — the model weights, and the Open WebUI
container whose volume holds your chat history and login. `--purge` and
`--remove-webui` opt in. Note that `--prefix` does not make it safe to try out:
most of what it removes lives at fixed paths, so it refuses outright if the
installed prefix disagrees with the one given.

## Layout

- [`docs/agent-forge.md`](docs/agent-forge.md) — the full build record: environment
  setup, SYCL benchmarks across three model classes, the serving layer and its
  caveats, the job queue, and the VS Code integration. Written to explain *why*
  each decision was made, not just what's running.
- [`configs/`](configs/) — `llama-swap.yaml` (the model
  registry), `llama-server-sycl` (the wrapper that sets `HOME` and
  `LD_LIBRARY_PATH` for the SYCL kernel cache — without it model loads hang
  silently), and the Continue.dev config.
- [`systemd/`](systemd/) — the unit files as installed.
- [`src/media-digest/`](src/media-digest/) — the job queue
  and the digest job.
- [`brief.md`](brief.md) — the original project scope.

## Machine

ASUS NUC 15 Pro+, Nobara Linux 43 (Fedora-based), Intel Core Ultra 7 255H,
Intel Arc Pro 130T/140T integrated graphics, 62GB RAM.

Key finding from benchmarking: token generation is **memory-bandwidth-bound**, not
compute-bound — both 7B models land around ~53GB/s effective bandwidth, and prompt
processing is comfortably fast at every size. On integrated graphics sharing system
RAM, that ceiling is what decides model choice. Full numbers in the build record.

## No backups here

Nothing in this repo needs backing up beyond git itself — it's configuration and
docs, and the models re-download from HuggingFace on request.
