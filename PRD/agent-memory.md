# Agent Memory PRD

## Problem Statement

As someone talking to the voice agent, it forgets everything. Within a single call it only keeps the last ~20 turns, so on a long call it loses what I told it earlier. And the moment the call ends, the whole conversation is discarded — when I come back, it has no idea who I am or anything we discussed before. It can't act like it remembers me, because today it literally can't: the transcript is held in memory for one session, windowed, and thrown away on disconnect. There is no persistence and no notion of a returning person.

## Solution

Give the agent memory in two layers, both of which surface to the LLM as additional context before it answers:

1. **In-session memory** — when a call runs past the existing turn window, the agent keeps a rolling summary of the earlier part of the conversation so it doesn't lose the thread on long calls.
2. **Cross-session memory** — at the end of each call, the agent distills durable facts and a short summary of what was discussed, and stores them. On future calls it retrieves the most relevant of those memories and brings them into the conversation, so it recognizes the person and recalls prior context.

Memory is **distilled**, not raw transcript replay: an LLM extracts a compact set of durable facts and a summary, which keeps what gets injected small and on-point. Retrieval is **semantic (RAG)** — memories are embedded and the most relevant ones to the current moment are pulled in. For now the agent **always assumes it is talking to the same person**, so there is a single global memory — no login, phone number, or per-user keying required.

Both layers attach at the same point a live turn already assembles its context, so memory shows up to the model as just another system message — consistent with how intent-inference context already works. Everything works in **mock mode** without provider keys (deterministic embeddings + heuristic distillation) so the feature is fully usable and testable offline; **live mode** uses real embeddings and LLM distillation.

## User Stories

1. As a caller, I want the agent to remember what I said earlier in a long call, so that I don't have to repeat myself when the conversation runs long.
2. As a caller, I want the agent to recall facts I shared on a previous call, so that it feels like it actually knows me.
3. As a caller, I want the agent to remember my name once I've told it, so that it can address me naturally next time.
4. As a caller, I want the agent to remember my stated preferences (likes, dislikes, choices), so that its responses stay consistent with what I've told it.
5. As a caller, I want the agent to remember ongoing topics or plans we discussed, so that we can pick up where we left off.
6. As a caller, I want the agent to bring up remembered context only when it's relevant, so that it doesn't dump everything it knows into every reply.
7. As a caller, I want the agent to weave remembered details in naturally, so that it sounds like memory, not a database read-back.
8. As a caller, I want the agent not to invent memories it was never told, so that I can trust what it claims to remember.
9. As a caller, I want what I share to be distilled rather than stored verbatim, so that the agent recalls the gist instead of replaying whole transcripts.
10. As a caller, I want recent and more relevant memories to take precedence, so that stale details don't crowd out what matters now.
11. As a caller, I want the agent to stay in character while using memory, so that recalled facts are delivered in the persona's voice.
12. As an operator, I want memory to be on by default in mock mode and explicitly opt-in for live, so that turning it on is a deliberate decision in production — mirroring how proactive behavior is gated.
13. As an operator, I want memory to work in mock mode without any API keys, so that I can develop and demo it offline.
14. As an operator, I want a clear warning (not a crash) if memory is enabled in live mode but the embedding provider key is missing, so that I understand why retrieval isn't happening.
15. As an operator, I want memory stored locally with no cloud service or vendor lock-in, so that I can run it without external infrastructure or a deployment step.
16. As an operator, I want to inspect what the agent currently remembers, so that I can see and debug its memory.
17. As an operator, I want to reset/clear the agent's memory, so that I can start clean during testing or between demos.
18. As an operator, I want end-of-session distillation to not slow down the live conversation, so that adding memory doesn't hurt turn latency.
19. As an operator, I want the in-session summary to kick in only once a call exceeds the existing turn window, so that short calls pay no extra cost.
20. As an operator, I want near-duplicate facts to be de-duplicated before storage, so that memory doesn't bloat with repeats of the same thing.
21. As an operator, I want to swap the embedding provider later without rewriting memory, so that I'm not locked into one vendor.
22. As an operator, I want to tune how many memories are retrieved per turn, so that I can balance recall against prompt size.
23. As an operator, I want memory injection reported as a pipeline stage (like intent inference), so that I can see when and how much memory was used.
24. As a developer, I want memory to attach at the same context-assembly seam intent inference already uses, so that we don't add a second, divergent injection path.
25. As a developer, I want the vector store and embedder behind small interfaces, so that they're independently testable and replaceable.
26. As a developer, I want distillation and retrieval covered by tests that run offline, so that CI doesn't depend on network or keys.
27. As a caller, I want memory to degrade gracefully if a provider fails mid-call, so that a memory problem never breaks the conversation.
28. As an operator, I want memory persisted atomically to disk, so that a crash mid-write can't corrupt the store.
29. As an operator, I want memory to survive a server restart, so that the agent still remembers after a redeploy.
30. As a developer, I want a single global memory (one person) for now, with identity-keying explicitly deferred, so that scope stays contained and the data model can grow later.

## Implementation Decisions

### Memory layers

- **In-session running summary.** The session already windows the transcript to the last N turns before sending it to the LLM. When the conversation grows past that window, the turns that fall out are folded into a rolling summary held on the session for the duration of the call. The summary is regenerated incrementally (cheap) and injected so long calls keep their early context. This reuses the existing windowing point and the existing LLM provider.
- **Cross-session long-term memory.** At the end of a call, the full transcript is **distilled** into (a) a small set of durable facts and (b) a concise session summary. These are embedded and stored. On subsequent calls, the current user utterance is used to **retrieve** the top-K most relevant stored memories, which are formatted into a memory block and injected.

### Injection seam (reuse, don't fork)

- Memory attaches at the **same context-assembly seam intent inference uses** (the pure function that prepends a system message to the windowed transcript before the agent streams a response). The function is extended so the assembled context can include a **memory block** system message (retrieved long-term memories + the in-session running summary) in addition to the existing intent-inference message. Keeping it a pure transform preserves the existing high, easily-tested seam and avoids a second injection path inside the session.
- When memory is injected, the session emits a **pipeline stage** event (mirroring the existing `llm_context` intent-inference stage) reporting that memory was used and how many memories were included.

### Formation trigger

- **End-of-session distillation** runs when the session closes, over the accumulated transcript. It is **non-blocking with respect to the live turn loop** — distillation happens at/after teardown so per-turn latency is unaffected.
- The **in-session summary** updates during the call as turns exceed the window; it does not require session end.

### Representation & storage

- **Distilled, not verbatim.** Stored memory records carry the distilled text, a kind (durable *fact* vs session *summary*), an embedding vector, a creation timestamp, and the originating session id. No raw transcripts are persisted by default.
- **Local vector store, no lock-in.** Memories live in a local, file-backed store under a data directory, written **atomically** (same safe-write pattern the character files use). Retrieval is brute-force **cosine similarity in numpy** — more than sufficient for a single person's distilled memory, with zero external service and no format lock-in. `sqlite-vec` is the documented upgrade path if scale ever demands it.
- **De-duplication.** Before a fact is stored, it is compared by cosine similarity against existing memories; near-duplicates above a configurable threshold are skipped so memory doesn't accumulate repeats.
- **Retrieval policy.** Top-K by similarity to the current utterance, with recency used to break ties / bias toward fresher memories. K is configurable.

### Embeddings — swappable interface

- Embeddings go through a small **Embedder interface** so the provider is replaceable without touching memory logic. Two implementations:
  - **OpenAI embedder** (live) — uses OpenAI's embeddings model (text-embedding-3-small class); requires a new `OPENAI_API_KEY`. (Chosen because the existing LLM provider, Groq, has no embeddings endpoint.)
  - **Deterministic mock embedder** (mock/keyless/tests) — maps text to a fixed-dimension normalized vector deterministically, so retrieval is exercisable and tests are reproducible offline.
- Selection follows the existing mock/live convention. In **live** mode with `OPENAI_API_KEY` absent, memory degrades gracefully (falls back to the mock embedder and/or disables retrieval) and surfaces a **config warning** rather than crashing — reusing the existing missing-key warning pattern.

### Distillation provider

- Live mode distills with the existing **Groq** LLM (extract durable facts + summary via a dedicated instruction). Mock mode uses a **lightweight heuristic** extractor so memory is demonstrable without keys. A provider failure during distillation is caught and logged; it never breaks the conversation.

### Single global person

- All memory is keyed to **one global person** — a single store, no identity capture. Identity-keying (e.g., Twilio caller number, per-browser id) is explicitly out of scope and noted as the future extension point; the record schema leaves room to add an owner key later.

### Configuration & management surface

- New settings (all `VOICE_AGENT_*`, following the existing config module): memory enable mode (`auto`/`true`/`false`, defaulting like proactive — on in mock, opt-in for live), embedding provider + model, `OPENAI_API_KEY`, retrieval top-K, dedupe threshold, memory storage directory, and the in-session summary window threshold. The public config-status report gains a memory section (enabled, embedder readiness) without leaking secrets.
- A minimal **management surface**: an endpoint to **inspect** current memories and one to **clear/reset** them, for observability and testing. (The existing Personality Studio can surface these later; full memory-editing UI is out of scope here.)

## Testing Decisions

**What a good test looks like here:** assert externally observable behavior — given a seeded memory store and an utterance, the assembled agent context contains the expected memory block; given a transcript, distillation produces de-duplicated facts; given two stored memories, retrieval returns the more relevant one. Do **not** assert internal prompt wording or private helpers. Everything runs **offline** using the deterministic mock embedder and a heuristic/mocked distiller, so no network or keys are needed in CI.

- **Vector store (pure unit tests):** add + query returns nearest by cosine; the dedupe threshold suppresses near-duplicates; persistence round-trips (write → reload from disk yields the same memories). Mirrors the file round-trip style of the character tests and the pure-function style of the Groq-agent tests.
- **Embedder:** the mock embedder is deterministic (same text → same vector; different text → different) and unit-length; the OpenAI embedder is tested with its HTTP call monkeypatched (no network), mirroring how config/provider behavior is tested today.
- **Memory manager:** `retrieve` returns the relevant memory given a seeded store; `remember`/distill de-duplicates; distillation over a sample transcript yields facts + summary using a mocked LLM. 
- **Injection seam (highest, existing):** extend the conversation-context pure-function tests so that, given a memory block, the assembled transcript includes the memory system message alongside intent inference, and excludes it when memory is empty/disabled. This is the same seam intent inference is already tested at.
- **Session integration:** instantiate the session (as the existing character-select test does), drive a turn, and assert close() triggers distillation (with the manager mocked) — without hitting providers.
- **Config:** new memory settings parse from env and the public status reports memory readiness, following the existing config-test patterns (`Settings(_env_file=None)` + monkeypatch).
- **Management endpoints:** inspect returns stored memories and reset clears them, via the test client with the memory directory redirected to a temp dir (same monkeypatch pattern the character endpoint tests use).
- **Prior art to follow:** the character endpoint/file-round-trip tests, the config/provider-status tests, the Groq-agent pure-function tests, and the conversation-context intent-inference tests.

## Out of Scope

- **Per-user / multi-person identity.** A single global memory only; no login, no per-browser id, and no Twilio caller-number capture (the agent assumes one person).
- **Cloud or server-based vector databases** (e.g. hosted Pinecone/Chroma). Local file-backed store only; `sqlite-vec` is a noted upgrade path, not part of this work.
- **Embedding-vendor lock-in.** Only the OpenAI embedder is implemented now; the interface exists so others can be added later, but adding them is out of scope.
- **Verbatim transcript archival** as the primary memory. Distilled facts/summaries are the memory; raw-transcript storage is not a goal.
- **Real-time per-turn fact extraction.** Distillation is end-of-session plus the in-session rolling summary, to protect turn latency.
- **Forgetting / decay / TTL policies** and conflict resolution between contradictory memories — noted as future work.
- **A full memory-management dashboard.** Only inspect + reset are in scope; rich editing UI is deferred (could later live in the Personality Studio).

## Further Notes

- The central design choice is that **memory is just another system message** at the context-assembly seam intent inference already uses — this keeps a single, well-tested injection path and makes memory composable with the existing pipeline.
- Keeping **distillation at end-of-session** (and the in-session summary incremental) means memory adds no measurable per-turn latency in the live loop.
- **numpy-cosine over a local file** is deliberately humble: at single-person scale it's instant and dependency-free, and because the records are plain data there's no lock-in if we later move to `sqlite-vec` or add identity keys.
- The **record schema intentionally leaves room** for an owner/identity key, so moving from "one person" to "many callers" later is an additive change, not a rewrite.

## Verification

1. **Tests:** `pytest` — vector-store, embedder, manager, conversation-context injection, session-close distillation, config, and inspect/reset endpoint tests all pass, all offline.
2. **Mock mode end-to-end** (no keys): with memory enabled and `VOICE_AGENT_MODE=mock`, use the already-running dev server (do not start a second one), then:
   - In one session, state a durable fact (e.g. "my name is Sam and I love jazz"); end the session.
   - Start a new session and reference it ("what kind of music do I like?") — confirm the agent recalls it and that a memory pipeline stage was emitted.
   - Hit the inspect endpoint and confirm the distilled fact is stored; hit reset and confirm it clears.
   - Run a long session past the turn window and confirm earlier context is still reflected (in-session summary).
3. **Live mode spot check** (if keys available): with `OPENAI_API_KEY` + Groq, confirm real embedding retrieval brings back relevant memories; then remove `OPENAI_API_KEY` in live mode and confirm a graceful config warning with no crash.
