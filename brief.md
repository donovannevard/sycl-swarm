# Project Brief: Local LLM Agent Orchestration Stack

**Codename:** `local-agent-forge` (built as `agent-forge`, part of the `sycl-swarm` repo)
**Owner:** Donovan Nevard
**Target hardware:** ASUS NUC 15 Pro+, Nobara Linux, Intel Arc GPU, 4TB local storage
**Primary goal:** Get real GPU-accelerated local LLM inference working via Intel's SYCL backend, then build a lightweight orchestration layer so local models can run scheduled or event-driven agent tasks (starting with low-stakes, high-frequency jobs) without burning API tokens on every run.

---

## 1. Scope and phasing

Same principle as the finance stack: each phase is a working checkpoint. Don't build orchestration on top of an inference backend that hasn't been benchmarked and proven stable first.

### Phase 0 — Environment verification
- Confirm current Intel GPU driver and kernel versions; Arc support in `llama.cpp`'s SYCL backend has moved fast, so verify against current documentation rather than assuming an older guide is accurate.
- Install Intel oneAPI toolkit (Level Zero runtime + SYCL) following Intel's official Linux instructions for the specific Arc card.
- Verify device visibility with `sycl-ls` before writing any application code — if the GPU doesn't show up here, nothing downstream will work correctly, so this is a hard gate.

**Deliverable:** confirmed `sycl-ls` output showing the Arc GPU as a Level Zero device.

### Phase 1 — Inference backend benchmark
- Build `llama.cpp` with `-DGGML_SYCL=ON` (and `-DGGML_SYCL_F16=ON` if supported by the card) rather than defaulting to Vulkan — Vulkan support on Arc is currently the weaker path.
- Alternatively/additionally trial Intel's IPEX-LLM, which wraps the same SYCL backend with an easier setup path and broader framework integration (Ollama, HuggingFace, LangChain).
- Benchmark 2-3 candidate models at different sizes (e.g. a 7-8B general model, a smaller 3B fast model, and one code-oriented model) using `llama-bench`, recording prompt-processing and token-generation throughput for each.
- Decide, based on actual numbers rather than assumption, which model size is viable for which task class (e.g. small model for frequent simple classification tasks, larger model for less frequent synthesis tasks).

**Deliverable:** a short benchmark table (model, quantization, tokens/sec) that future task-routing decisions can reference.

### Phase 2 — Serving layer
- Stand up a persistent inference server rather than spinning up `llama.cpp` fresh per task (Ollama with IPEX-LLM backend, or `llama.cpp`'s own server mode) so multiple agent jobs can share a loaded model without repeated load latency.
- Confirm the server only listens on localhost/internal network by default — no need for external exposure for a single-machine setup.
- Add basic resource guardrails: max concurrent requests, timeout handling, and a way to unload a model to free VRAM if something else needs the GPU.

**Deliverable:** a running local inference server with a simple health-check endpoint.

### Phase 3 — Task orchestration layer
- Build a minimal job queue (something like a simple SQLite-backed queue or Redis if you want a bit more infrastructure practice) that can accept scheduled and event-triggered tasks.
- Design each task as a self-contained job definition: prompt template, target model, expected output schema, and a fallback/retry policy.
- Start with **read-only, low-stakes tasks first** — e.g. summarizing RSS feeds, classifying incoming data, drafting (not sending) messages — before trusting local inference with anything that writes to another system.
- Log every job's input, output, and latency for later review; this is useful both for debugging and for deciding which tasks are actually worth running locally vs. worth the token cost of a hosted model.

**Deliverable:** a working queue that can run at least one real scheduled task end-to-end with full logging.

### Phase 4 — Integration with existing projects
- Identify 1-2 genuinely good local-model candidates from your existing project list — the earlier "media/paper-trail digest" idea is a strong fit here specifically because it's high-frequency and doesn't need frontier-model reasoning quality, which is exactly where a local model earns its keep over API costs.
- Keep a clear boundary: tasks needing high-quality reasoning, nuanced judgment, or anything touching real money (the finance stack's decisions) should stay on hosted models — local inference is for volume/cost tradeoffs, not for replacing judgment-critical calls.

**Deliverable:** at least one existing or new project genuinely running on the local stack in production use, not just as a demo.

---

## 2. Design principles to hold onto

- **Benchmark before you build on top.** Don't architect an ambitious orchestration layer around inference performance you haven't actually measured on your hardware.
- **Right-size the task to the model.** Local models are for high-frequency, lower-stakes work. This is a cost/throughput tool, not a quality upgrade — don't route judgment-heavy tasks here just because it's free to run.
- **Keep the serving layer boring.** A stable, simple, always-on inference server beats a clever one. Complexity belongs in the orchestration layer, not the backend.
- **Isolate from anything sensitive.** This stack shouldn't have credentials or access to the finance stack's systems — keep local-agent tasks and financial systems on separate service users and separate secrets, even if they run on the same physical machine.

## 3. Suggested tech stack

- **Inference backend:** `llama.cpp` (SYCL build) or IPEX-LLM + Ollama
- **Queue/orchestration:** Python with a simple SQLite queue to start (upgrade to Redis/Celery only if volume genuinely demands it)
- **Scheduling:** systemd timers (fits a single-machine Linux setup better than adding Airflow/cron complexity)
- **Logging/observability:** plain structured logs to start; can feed into the same Prometheus/Grafana setup as the finance stack if you want one unified monitoring view across projects

## 4. Explicit non-goals for this phase

- No multi-GPU or distributed inference — single Arc card, single machine.
- No fine-tuning or training work yet — this phase is inference-only.
- No exposure of the inference server beyond the local machine/network.
- No routing of financially consequential or judgment-critical decisions to local models (see Section 2).

---

Complete and verify each phase before starting the next. Phase 0's `sycl-ls` check is
a hard gate — do not proceed to building inference code until GPU visibility is
confirmed. Phase 1's benchmark numbers should inform (not be skipped in favor of
assumptions about) model selection in later phases.