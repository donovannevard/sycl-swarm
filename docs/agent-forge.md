# agent-forge

SYCL-accelerated local LLM inference on an Intel Arc iGPU, plus a lightweight
orchestration layer so local models can run scheduled or event-driven tasks without
burning hosted-API tokens. Original scope in [`brief.md`](../brief.md).

Running as systemd services on the machine described below. This document covers the
environment, the inference backend and its benchmarks, the serving layer, the job
queue, and the VS Code integration — what's running, why it's built the way it is, and
the caveats worth knowing before doing this again on different hardware.

| Component | What |
|---|---|
| Environment | oneAPI DPC++/C++ compiler, Level Zero runtime, SYCL device visibility |
| Inference backend | `llama.cpp` built with `-DGGML_SYCL=ON`, benchmarked across three model classes |
| Serving layer | `llama-swap` (model router) + Open WebUI (chat) + per-user agent profiles |
| Job queue | SQLite-backed queue, one scheduled job (media digest), disabled by default |
| Editor integration | Continue.dev in VS Code, wired to the same inference endpoint |

---

## The machine

- Host: Nobara Linux 43 (Fedora 43-based), kernel `6.19.8-200.nobara.fc43.x86_64`
- ASUS NUC 15 Pro+, Intel Core Ultra 7 255H, 16 cores, 62GB RAM, 1.9TB free on `/home`
- GPU: Intel Arc Pro 130T/140T (Arrow Lake-P integrated graphics), PCI `00:02.0`
- `i915` claims the GPU, not `xe` — despite the `xe` kernel module also being loaded, and
  despite Intel's newer Arc/Xe2 compute stack generally trending toward `xe`. Level Zero,
  OpenCL and `sycl-ls` all enumerate the device correctly under `i915`, so no driver
  switch is needed.

Everything here runs on that single shared iGPU. That constraint drives most of the
design decisions further down — there is no way for game rendering and LLM inference to
both get smooth performance from the same physical GPU at once.

---

## 1. Environment

### Installed

1. **Compute runtime** (Fedora repos, no extra repo needed):
   `intel-compute-runtime`, `intel-level-zero`, `intel-level-zero-devel`,
   `oneapi-level-zero`, `oneapi-level-zero-devel` — gives Level Zero + Intel OpenCL
   Graphics platform visibility, confirmed via `clinfo -l` and the `zello_world` Level
   Zero smoke test.

2. **Intel oneAPI DPC++/C++ Compiler** (Intel's official YUM repo at
   `/etc/yum.repos.d/oneAPI.repo`): `intel-oneapi-compiler-dpcpp-cpp` 2026.1.0-235 —
   the full DPC++/C++ stack (`icpx`, `sycl-ls`, debugger, OpenMP, TBB), installed to
   `/opt/intel/oneapi`, ~3GB on disk. Fedora is not an officially validated distro for
   this repo; the RPMs install and run cleanly regardless.

### Deliverable

```
source /opt/intel/oneapi/setvars.sh
sycl-ls
```
```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) Graphics 12.74.4 [1.14.37435+12]
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 7 255H OpenCL 3.0 (Build 0) [2026.21.6.0.17_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) Graphics OpenCL 3.0 NEO  [26.09.37435.12]
```

Full output in [`sycl-ls-output.txt`](sycl-ls-output.txt).

**Note:** `setvars.sh` doesn't reliably put `sycl-ls`/`icpx` on `PATH` in a
non-interactive shell — the binaries live at `/opt/intel/oneapi/compiler/2026.1/bin/`.
Worth setting `PATH` explicitly when scripting a build rather than assuming
`source setvars.sh` alone is enough.

---

## 2. Inference backend benchmarks

### Build

`llama.cpp` is built out-of-tree. The source is not vendored and neither are the
resulting binaries: `llama-server` plus ~29 SYCL shared libraries come to ~294MB,
they are specific to this GPU and this oneAPI version, and they rebuild from a
public repo in minutes.

```
cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

**Note:** llama.cpp's SYCL backend hard-requires Intel oneMKL — `find_package(MKL)`
fails the CMake configure otherwise. Install `intel-oneapi-mkl-devel` (~2GB) from
Intel's oneAPI YUM repo first; without it `-DGGML_SYCL=ON` won't configure at all.

**Note:** pin a specific `llama.cpp` commit (`git checkout <sha>`) once a build is
verified working, rather than tracking the branch tip — a later upstream change can
stop building against a given oneAPI version with no warning.

`llama-ls-sycl-device` confirms detection:

```
| 0| [level_zero:gpu:0]| Intel Arc Graphics | 12.74 | 128 compute units | 62218M | driver 1.14.37435+12
```

The 62GB reported "global mem" reflects shared system RAM — this is integrated graphics,
not a discrete VRAM pool.

### Results

`llama-bench`, SYCL backend, `ngl=-1` (full GPU offload). All models from the `bartowski`
GGUF quant releases on HuggingFace, downloaded via `hf download`. Same model family
(Qwen2.5) across all three sizes deliberately, for a clean apples-to-apples comparison.

| model | role | size | params | quant | pp512 (t/s) | tg128 (t/s) | effective decode BW |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | fast/small | 1.79 GiB | 3.09B | Q4_K_M | 532.91 ± 1.78 | 24.48 ± 0.12 | ~44 GB/s |
| Qwen2.5-7B-Instruct | general | 4.36 GiB | 7.62B | Q4_K_M | 330.74 ± 3.04 | 11.38 ± 0.13 | ~53 GB/s |
| Qwen2.5-Coder-7B-Instruct | code | 4.36 GiB | 7.62B | Q4_K_M | 322.14 ± 3.10 | 11.56 ± 0.04 | ~53 GB/s |

### Interpretation

- **Token generation is memory-bandwidth-bound, not compute-bound.** Effective bandwidth =
  `tg128_t/s × model_size_bytes`. Both 7B runs land around ~53GB/s effective, the 3B run
  ~44GB/s — consistent with a single memory-bandwidth ceiling being the real constraint,
  not per-model compute cost. This machine has 64GB dual-channel DDR5 (theoretical ceiling
  roughly 80-100GB/s), so this is ~50-65% real-world bandwidth efficiency, normal for
  an integrated GPU sharing system RAM (vs. 70-85% typical on a mature discrete-GPU stack).
- **Prompt processing is compute-bound** and comfortably fast at every size tested
  (322-533 t/s) — the Arc iGPU's compute isn't the bottleneck; decode throughput is capped
  by RAM bandwidth.
- Coder-7B and general-7B perform near-identically, as expected (same architecture and
  size, different fine-tune).

### Addendum: does MoE decode track *active* params?

One Mixture-of-Experts model, to check whether decode speed tracks active params rather
than total params given the bandwidth-bound finding above.

| model | type | size | active/total params | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | MoE | 9.65 GiB | 2.4B / 15.7B | 159.26 ± 2.88 | 18.90 ± 0.15 |

**Partially confirmed.** Decode (18.9 t/s) is far faster than a dense model of similar
total size would be — extrapolating the dense trend above (24.5 t/s @ 3B, 11.4 t/s @ 7-8B)
predicts roughly 4-6 t/s for a dense ~16B model, so MoE delivers **~3-4x** the decode
throughput its total size suggests. It doesn't reach pure 3B-class speed (24.5 t/s) as a
naive "only active params get read" model would predict, likely because shared/attention
layers are read in full every token regardless of expert routing, and gathering weights
from selected experts is a less memory-friendly access pattern than one dense model's
sequential weight stream. Prompt processing (159 t/s) is also below the dense 7B's 331 t/s,
consistent with MoE routing overhead being proportionally more visible when per-token
compute is already cheap.

**Note:** on bandwidth-bound hardware, a well-chosen MoE model is a genuinely good way
to get more capability per tok/s than a same-speed dense model offers — worth
preferring MoE variants for tasks needing more reasoning depth than a 3B model can give
but that can't afford dense-7B's ~11 tok/s.

### Task routing guidance

- **3B class (~24 tok/s decode):** short, frequent jobs — classification, single-field
  extraction, short RSS-digest summaries.
- **7-8B class (~11-12 tok/s decode):** less frequent, longer-form jobs (multi-paragraph
  summarization, drafting) where a few extra seconds of latency doesn't matter.
- Per-request decode speed is a **single-stream** number. A job queue gets meaningfully
  better *aggregate* throughput by batching queued jobs concurrently through
  `llama-server`'s continuous batching, since compute headroom sits idle during
  memory-bound decode.

---

## 3. Serving layer

### What's running

| Component | What | Where | Exposure |
|---|---|---|---|
| `llama-swap` | Model router/swapper (v237) | systemd service, `User=svc-agent-forge` | `0.0.0.0:8090`, API-key protected |
| `llama-server` (×N, on demand) | Actual SYCL inference backends | spawned by `llama-swap` per request | `127.0.0.1:<10001+>` only |
| Open WebUI | Chat UI, own login | Docker container, `--network host`, `--restart unless-stopped` | `0.0.0.0:3000` — LAN-facing |

Data flow: `browser/editor on the LAN → llama-swap (API key) → llama-server (SYCL/GPU)`,
and separately `browser → Open WebUI (its own login) → localhost → llama-swap`.

**Note:** `llama-swap` listens on all interfaces rather than loopback-only, because an
editor running on a different physical machine on the LAN has no way to reach a
`127.0.0.1`-bound service — there's no host-networking trick across two separate
computers, unlike the Open WebUI/Docker case below. The API key is the only guard;
network segmentation was never a strong second layer on a single-machine home LAN.

### Isolation

Long-running services on this machine run as their own system users rather than the
desktop account:

- System user `svc-agent-forge` (no login shell, no home directory — *almost*; see the
  note on `$HOME` below).
- Model files, config and cache under `/var/lib/agent-forge/`, owned by `svc-agent-forge`,
  mode 750.
- API key in `/etc/agent-forge/llama-swap.env`, mode 640, root:svc-agent-forge —
  unreadable by the desktop user (confirmed: `grep` as that user gets `Permission denied`).
- `llama-swap.service` runs with `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`,
  `ProtectHome=true`, `ReadWritePaths=/var/lib/agent-forge` only.
- GPU access needs no special group grant — `/dev/dri/renderD128` is world-accessible
  (`crw-rw-rw-`).

### Binaries

- `/usr/local/bin/llama-server` — copied from the SYCL build's `build/bin/llama-server`.
- `/usr/local/lib/agent-forge/*.so*` — the ~29 shared libraries (`libggml-sycl.so`,
  `libllama-server-impl.so`, etc.) `llama-server` needs at runtime. **Note:** the build
  output relies on `$ORIGIN`-relative RPATH, so copying just the executable without its
  sibling `.so` files fails with `cannot open shared object file`. The libs live in a
  dedicated dir with `LD_LIBRARY_PATH` set in the wrapper below.
- `/usr/local/bin/llama-swap` — prebuilt release binary (v237) from
  `github.com/mostlygeek/llama-swap`.
- `/usr/local/bin/llama-server-sycl` — the wrapper script below.

### Note: `$HOME` for a system user with no home directory

`useradd --system --no-create-home` sets `$HOME=/home/svc-agent-forge`, but that directory
doesn't exist. Intel's compute runtime needs a writable cache dir under `$HOME/.cache` for
SYCL kernel JIT compilation — without it, `llama-server` doesn't error, it **hangs
indefinitely** during model load, since it's silently failing to create its cache dir.
The wrapper script sets both explicitly:

```bash
#!/bin/bash
export HOME=/var/lib/agent-forge
export XDG_CACHE_HOME=/var/lib/agent-forge/cache
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/usr/local/lib/agent-forge:${LD_LIBRARY_PATH}"
exec /usr/local/bin/llama-server "$@"
```

First model load after this fix takes **~77 seconds** (cold SYCL kernel cache build,
one-time per user). Warm loads / cached requests are sub-second — 0.44s measured for a
follow-up request with 24 cached prompt tokens.

### `llama-swap.yaml`

- 4 models registered: `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-coder-7b`,
  `deepseek-coder-v2-lite` (the same GGUFs benchmarked above, copied to
  `/var/lib/agent-forge/models/`), plus `nomic-embed-text` for embedding calls.
- `apiKeys: ["${env.LLAMA_SWAP_API_KEY}"]` — read from the systemd `EnvironmentFile`.
  Confirmed: `/v1/models` returns 401 without the key, 200 with it.
- `globalTTL: 1800` — models auto-unload after 30 minutes idle. Verified empirically
  rather than trusted from docs: temporarily set to 15s, loaded a model, confirmed via
  `journalctl` that it logged `Unloading model, TTL of 15s reached` and the process
  actually exited.
- `--ctx-size 8192` per model — a reasonable default, not benchmarked; revisit if a task
  needs more.
- No per-backend `--api-key`: backends are only ever reached via `llama-swap` on localhost,
  so a single API key at the `llama-swap` layer is the whole auth boundary for the
  inference API.

`llama-swap` also serves `POST /api/models/unload/{model}` for manual unload rather than
waiting out the idle TTL — useful for releasing the GPU immediately when something else
needs it.

### Open WebUI

`ghcr.io/open-webui/open-webui:main`, run with **`--network host`** (not the default
bridge) and `-e PORT=3000`, connected to `llama-swap` at `http://127.0.0.1:8090/v1` with
the same API key.

First-run admin account creation is a personal login credential and is left for manual
setup rather than scripted. **Agent profiles** use Workspace → Models in the admin UI to
create named presets (base model + custom system prompt) — e.g. a coding-focused preset
wrapping `qwen2.5-coder-7b`, a different persona wrapping `qwen2.5-7b`. This is Open
WebUI's built-in feature; no custom code is needed.

**Note: use host networking, not the default Docker bridge.** The default bridge network
reaches the host via the `docker0` bridge IP, which loopback-only binding explicitly
excludes — a service bound to `127.0.0.1:8090` on the host is unreachable from a
bridge-networked container regardless of `host.docker.internal` mapping. Opening
`llama-swap` to `0.0.0.0` with a firewall rule is one fix; simpler is host networking,
which keeps `llama-swap` genuinely loopback-only with zero firewall changes — inside a
host-networked container, `127.0.0.1` really is the host's own loopback.

**Note: Open WebUI persists connection settings into its own database on first run.**
`openai.api_base_urls` and similar are seeded from environment variables only when no
config row exists yet in `/app/backend/data/webui.db` (the named Docker volume). If a
volume is reused across a config change, the container ignores the new environment
variable entirely. Fix by updating the `config` table's `openai.api_base_urls` row
directly, or by starting from a fresh volume. Worth checking the DB-persisted config
first any time an env var change doesn't seem to take effect after a container recreate.

### Verified

- `curl http://127.0.0.1:8090/health` → `OK` (200).
- `curl http://127.0.0.1:8090/v1/models` → 401 without API key, 200 with it, lists all
  registered models.
- Real chat completion through the full stack (llama-swap → llama-server-sycl → SYCL/GPU).
- Open WebUI healthy at both `127.0.0.1:3000` and the LAN IP, confirmed end-to-end from
  another device on the LAN.
- Idle-unload confirmed live.
- `systemctl is-enabled llama-swap` → enabled (starts at boot, no login session needed).
  Open WebUI survives reboots via Docker's `--restart unless-stopped`.

### Non-goals

No TLS — plain HTTP, consistent with the other home-LAN-only services on this box.
No custom profile-switching code — Open WebUI's native feature covers it entirely.

---

## 4. Job queue and the media digest

`media-digest.timer` is installed but disabled by default
(`systemctl disable --now media-digest.timer`) — kept as a working example of the
job-queue pattern rather than run continuously. Fully reversible
(`systemctl enable --now media-digest.timer`); the job queue and the job's code stay in
place either way.

### What's built

- **Generic SQLite job queue** (`/var/lib/agent-forge/jobs.db`, schema and helpers in
  `src/media-digest/jobqueue.py`) — one row per job run: status, timing, full input/output
  as JSON. Reusable by other jobs, not one-off.
- **`media_digest` job** (`src/media-digest/media_digest.py`, installed to
  `/var/lib/agent-forge/jobs/`): fetches recent headlines, runs each through a skeptical
  media-literacy persona via the local inference API, posts the result into Open WebUI as a
  dated Note.
- **`media-digest.timer`**: `OnCalendar=*-*-* 07:00:00`, `Persistent=true` (catches up
  shortly after boot if the machine was off at 7am). Runs as `svc-agent-forge` via a
  hardened service, same isolation pattern as `llama-swap`.

### Sources

Live-checked before committing to them — don't assume RSS feeds are still around. BBC,
Guardian and NYT work. **Note:** Reuters and AP have discontinued public RSS (404/HTML
page, not a feed) — worth checking rather than assuming a feed is still live. Al Jazeera
and Fox News are included deliberately for framing diversity across the political
spectrum (UK public/left, US left, non-Western, US right) — a persona that only questions
one side's narratives isn't media literacy, it's just a different bias.

### The persona prompt

> You are a media literacy analyst. You do not trust mainstream framing by default -- you
> treat every headline as a claim to interrogate, not a fact to accept. For the article
> given, respond with three parts: **What happened** (neutral summary, stripped of loaded
> language), **The intended takeaway** (what conclusion/reaction the framing seems designed
> to produce), **Worth asking** (2-3 concrete questions about why this framing was chosen
> and what's absent). IMPORTANT: Use ONLY the names, titles, and facts given in the article
> text below. Do not substitute any other name or role, even if a different name feels more
> familiar... Ground everything in specifics -- no vague suspicion, no reflexive
> contrarianism for its own sake.

Model: `qwen2.5-7b` — a general instruct model, deliberately not a coder variant; this is a
prose and reasoning task.

### Note: model size does not fix fabricated details

Given only a short RSS summary — a single sentence, not the full article — both
`qwen2.5-7b` and a larger reasoning model (`DeepSeek-R1-Distill-Qwen-14B`) produced
confidently wrong names that weren't in the source text at all: they came from the
model's training data, not the given text. The larger model's own reasoning trace
correctly identified the right facts, but its final written answer contradicted that
reasoning and invented a name anyway — bigger is not more grounded. The 14B model is
also far too slow for a whole-feed job (~2m47s per article at ~5.25 t/s).

What actually fixes it: an explicit prompt instruction forbidding name/fact substitution
from outside knowledge, plus fetching **full article text** via `trafilatura` instead of
relying on a thin RSS summary, with a fallback to the summary only when extraction fails
(some sites, especially paywalled ones, block scraping — the tightened prompt keeps a
summary-only fallback grounded too).

### Note: a broad `except Exception` can hide a systemic bug as easily as a transient one

`trafilatura.fetch_url(url, timeout=15)` throws `TypeError` on every call under the
installed library version, which doesn't accept a `timeout` kwarg — silently swallowed by
a broad exception handler, so full-text fetching was failing 100% of the time and falling
back to the same thin summaries that cause the fabrication problem above. A job can look
like it "worked" (it still produces output for most articles) while a core piece of its
pipeline is failing every single time. Logging the exception, and testing the failure
path rather than only the happy path, is what surfaces this.

### Category sections and filtering

Articles are grouped into newspaper-style sections rather than by source, with sports and
celebrity content filtered out entirely. Implemented as a **cheap classification pass
gating the expensive analysis**, not the other way round:

1. Every article is classified into `Politics / War & Conflict / Economy / World / Sports /
   Celebrity` by the **fast 3B model** off just title+summary (cheap, ~10 output tokens).
2. `Sports`/`Celebrity` are dropped **before** the expensive step — no `trafilatura` fetch,
   no 7B analysis call. This makes the whole run faster overall despite the extra step.
3. The digest renders grouped by category (fixed order: Politics, War & Conflict, Economy,
   World), each article still showing its source inline.
4. Classification failures default to `"World"` (fail open — better to show an
   uncategorized article than silently drop one on a transient error), and both included
   and excluded articles are logged in the job's output JSON for auditability.

**Known limitation:** the 3B classifier isn't 100% reliable off title+summary alone —
ambiguous headlines (e.g. "Argentina get ready: Fans react to England win") can pass
through undetected. A cheap, more reliable backstop would be checking the RSS URL path
too (many outlets put sport under `/sport/`) rather than relying on the LLM classification
alone — not implemented.

### Verified

- The pipeline confirmed manually on articles that previously fabricated details —
  correct across two separate full 20-article runs (0 model-call errors).
- Runs **through the real systemd service** (`systemctl start media-digest.service`), not
  just manually as the service user — `status=0/SUCCESS`.
- `systemctl list-timers media-digest.timer` shows it scheduled and enabled.
- Note appears correctly in Open WebUI, readable from any device on the LAN.
- Full input/output/latency logged per article for every run.

### Credentials

`/etc/agent-forge/openwebui.env` (root:svc-agent-forge, mode 640) — an Open WebUI
personal API key, generated through Settings → Account rather than set by any script
(same principle as not scripting a login password); only the global
`auth.enable_api_keys` flag needed flipping first.

---

## 5. Editor integration — Continue.dev in VS Code

Offloads routine coding tasks to the local model and saves hosted-API usage for harder
problems. Local inference is for volume/cost tradeoffs, not for replacing
judgment-critical calls — see [`brief.md`](../brief.md).

### What's set up

- **Continue.dev** as a VS Code extension, installed via
  `flatpak run com.visualstudio.code --install-extension Continue.continue` (VS Code is
  Flatpak-packaged here — the plain `code` CLI isn't on `PATH`).
- Config at `~/.continue/config.yaml`, copy in
  [`configs/continue-config.yaml`](../configs/continue-config.yaml).
  Two models: `qwen2.5-coder-7b` for `chat`/`autocomplete`/`edit`/`apply`, and `qwen2.5-7b`
  for `chat` only as a general fallback.
- Backend connectivity verified against the exact config Continue uses — same model name,
  same endpoint, same key — before touching the VS Code side.

### Setting this up on another device

1. Install the **Continue** extension from the VS Code marketplace (publisher Continue.dev).
2. Copy `configs/continue-config.yaml` to `~/.continue/config.yaml` as-is — it
   already points at the LAN IP, not localhost.
3. Reload the VS Code window if the model list doesn't appear immediately.
4. Confirm the device is on the same LAN and that `http://<host-ip>:8090/health` is
   reachable from it.

**Note:** the LAN IP is DHCP-assigned — it can change on a reboot or lease renewal,
breaking every device's Continue config until updated. A DHCP reservation on the router
avoids this.

### Verified

- `curl http://<host-ip>:8090/health` reachable from the LAN interface, not just loopback.
- Real chat completion against `qwen2.5-coder-7b` via the exact endpoint/model/key
  Continue's config specifies.
- Extension installed and config in place.
