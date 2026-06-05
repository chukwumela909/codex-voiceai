# Personality Studio Dashboard PRD

## Problem Statement

As an operator tuning the agent's personality, I can only edit a **character** through a small form buried in the diagnostics drawer of the main call UI. That form doesn't expose every field of the character model — notably it can't edit **example exchanges** at all, and it edits list fields (tone, speaking style rules, forbidden phrases) as lossy free-text blobs. There is no way to create a new character or start from a copy of an existing one, and the only way to judge whether my wording actually changes how the agent talks is to place a full call. Iterating on a persona is slow, lossy, and hard to evaluate.

## Solution

Add a dedicated **Personality Studio** page where I can see every character at a glance, open any one in a complete editor that exposes all schema fields, create a brand-new character or duplicate an existing one as a starting point, save changes, and immediately hear a full **voice preview** — the character's system prompt run through the LLM and spoken by TTS — for a test message I type, without starting a real conversation.

The Studio reuses the backend that already exists. Characters are defined by the `Character` model (`id`, `name`, `role`, `tone`, `grammar`, `forbidden_phrases`, `identity_response_style`, `speaking_style_rules`, `example_exchanges`), compiled into the Groq system prompt by `build_system_prompt()` and persisted atomically by `save_character()`. The REST endpoints `GET /characters`, `PUT /characters/{id}`, and `POST /characters` already accept and validate the full model. The Studio is therefore mostly a new frontend over these existing seams, plus one new stateless preview endpoint that composes the same provider adapters the live session uses, so what you hear in the Studio matches production.

Scope decided with the requester: a **new dedicated page**; voice stays **global** (not per-character); **full voice preview** (LLM + TTS); capabilities are **create + duplicate + full-field editor**. Delete and set-default are out of scope.

## User Stories

1. As an operator, I want a dedicated Personality Studio page, so that managing characters isn't squeezed into a diagnostics drawer.
2. As an operator, I want the Studio reachable from the main call UI, so that I can jump to it without remembering a URL.
3. As an operator, I want to see a list/grid of all characters with name and role, so that I can find the one I want.
4. As an operator, I want the current default character marked in the list, so that I know which persona starts a session.
5. As an operator, I want to select a character and open it in an editor, so that I can review its full definition.
6. As an operator, I want to edit the display **name**, so that the agent introduces itself correctly.
7. As an operator, I want to edit the **role** description, so that I can reshape the persona's backstory.
8. As an operator, I want to edit the **grammar / speaking style** descriptor, so that I can set the dialect and register.
9. As an operator, I want to edit the **tone** list as discrete entries, so that I can add/remove descriptors without mangling a textarea.
10. As an operator, I want to edit the **speaking style rules** as a list, so that each behavioral rule is its own editable item.
11. As an operator, I want to edit the **forbidden phrases** list, so that I can stop the agent from saying off-brand things.
12. As an operator, I want to edit the **identity response style**, so that I control how the agent deflects "are you an AI?" questions.
13. As an operator, I want to add, edit, reorder, and remove **example exchanges** (user/assistant pairs), so that I can steer style by demonstration — a field the current form can't edit at all.
14. As an operator, I want required fields (name, role) validated before save, so that I can't persist a broken character.
15. As an operator, I want the character **id** auto-derived from the name on create (lowercase slug) and shown read-only thereafter, so that ids stay stable and filesystem-safe.
16. As an operator, I want to create a new character from a blank template, so that I can build a persona from scratch.
17. As an operator, I want to duplicate an existing character under a new name/id, so that I can start from something close and tweak it.
18. As an operator, I want a clear save action with success/error feedback, so that I know my change was persisted.
19. As an operator, I want to be warned about unsaved changes before navigating away or switching characters, so that I don't lose edits.
20. As an operator, I want to type a test message and hear the selected character speak its reply, so that I can judge the persona end-to-end.
21. As an operator, I want the voice preview to use the *unsaved* edits in the editor, so that I can iterate before committing to disk.
22. As an operator, I want to see the preview's text response alongside the audio, so that I can read and hear the result together.
23. As an operator, I want the preview to reuse the same LLM + TTS pipeline as a live call, so that what I hear matches production behavior.
24. As an operator, in **mock mode** I want the preview to still work (canned text + synthesized tone), so that I can build the dashboard without live API keys.
25. As an operator, in **live mode** I want the preview to surface a clear message if a required provider key is missing, so that I understand why audio didn't play.
26. As an operator, I want the preview to indicate latency (LLM and TTS), so that I have a rough sense of responsiveness.
27. As an operator, I want changes I save in the Studio to take effect on the next session/turn, so that the live agent reflects my edits.
28. As an operator, I want the Studio to share the app's existing visual language, so that it feels part of the product.
29. As an operator, I want the page to work without a build step, so that deployment stays as simple as the rest of the static frontend.
30. As a developer, I want the Studio to reuse the existing `/characters` REST endpoints, so that we don't fork persistence logic.
31. As a developer, I want a single new preview endpoint that reuses the existing provider adapters, so that preview and production stay behaviorally consistent.
32. As a developer, I want the new endpoint covered by tests at the HTTP seam, so that regressions are caught.

## Implementation Decisions

### Surface & placement

- A new **dedicated page**, the *Personality Studio*, served as a static HTML file by FastAPI alongside the existing pages (same pattern as `GET /` and `GET /pipecat` returning a `FileResponse`), reachable at a new route (e.g. `/studio`). A link/button from the main call UI navigates to it. No build step; vanilla HTML/CSS/JS reusing the existing stylesheet and design tokens.
- Layout: a left list/grid of characters, a right editor panel, and a preview panel. The default character is visually flagged, read from the `default` field already returned by `GET /characters`.

### Persistence — reuse existing seams, no schema change

- Listing reuses `GET /characters`. Saving reuses `PUT /characters/{id}` (existing character) and `POST /characters` (new/duplicate). These already accept and validate the full `Character` model and write atomically via `save_character()`, so **no model or storage changes are required**.
- **Duplicate** is a client-side compose: load the source character, present it in the create flow with a new name → new slug id, then `POST /characters`. No new backend endpoint.
- The editor edits the full model client-side, including `example_exchanges` (list of `{user, assistant}`) and the list fields as structured rows rather than free-text blobs, then serializes to the model shape the endpoints already expect.
- Voice remains **global** (Cartesia env config); the Studio does not edit voice. The `Character` schema is unchanged.

### Voice preview — one new seam

- Add a **single new endpoint**, `POST /characters/preview`, that accepts a candidate character payload (the in-editor, possibly unsaved character) plus a test `message`, and returns the agent's text reply and spoken audio. It does **not** persist anything.
- Internally it reuses the exact production building blocks so preview matches live behavior: `build_system_prompt(character)` → `GroqStreamingAgent` for the reply → `CartesiaStreamingTTS` for audio. It honors `VOICE_AGENT_MODE`: in **mock** mode it returns the canned response and synthesized tone (mirroring how `MockConversationSession` falls back), so the feature is fully usable without keys; in **live** mode it uses real providers and returns a structured warning (reusing the same missing-keys reporting surfaced by `config.public_status()` / the existing `config.warning` path) when a required provider isn't configured.
- Response contract (JSON): `{ character_id, text, audio (base64 PCM), sample_rate, encoding, latency: { llm_ms, tts_ms }, warnings: [...] }` — mirroring the field names already used by the `audio.chunk` WebSocket event so the frontend can reuse its existing PCM playback path.
- Rationale for a dedicated HTTP endpoint over driving `/ws/browser`: it is the **highest stable seam** for a one-shot, stateless preview, matches the existing REST style of the other `/characters` routes, and is trivially testable with `TestClient`. Driving the full WebSocket session handshake for a single synthetic turn would be a lower, more brittle seam.

### Frontend behavior

- A small client module (new static JS, e.g. `frontend/studio.js`, loaded by the Studio page) owns: fetch and render the character list; the structured editor (with add/remove rows for list fields and example exchanges); create/duplicate flows; save via PUT/POST; unsaved-change guarding; and the preview call plus audio playback. PCM playback reuses the scheduling approach already in `frontend/app.js`.

## Testing Decisions

**What a good test looks like here:** assert externally observable behavior at the HTTP seam — request in, JSON/character-file out — not internal prompt wording or private helpers. Use the existing provider-mocking approach so tests run in `mock` mode without network or keys.

- **Reuse the existing high seam.** Extend `tests/test_characters.py` (already uses FastAPI `TestClient` against `app.main:app`, with `CHARACTERS_DIR` redirected to a temp dir via monkeypatch) to cover:
  - Round-trip save/load of a character that includes `example_exchanges` and multi-item `speaking_style_rules` through `PUT /characters/{id}` and `GET /characters` (guards the full-field-editor promise — the fields the old form dropped).
  - **Duplicate** flow at the API level: `POST /characters` with a new id derived from an existing character persists a new file and leaves the source unchanged.
  - Validation: missing required fields / bad slug id are rejected (existing pattern).
- **New preview endpoint** `POST /characters/preview`, tested at the HTTP seam in `mock` mode:
  - Returns non-empty `text`, base64 `audio`, a `sample_rate`/`encoding`, and the echoed character id for a typed message — without writing any file.
  - Accepts an unsaved/candidate character body (not just an existing id) and previews it.
  - Surfaces a structured `warnings` entry (not a 500) when a required provider key is absent in a live-style config — reuse the config-status pattern from `tests/test_config.py` (`Settings(_env_file=None)` + monkeypatch).
- **Pure-function unit test:** `build_system_prompt()` includes the new-to-the-UI fields (`example_exchanges`, all `speaking_style_rules`) in its output, so the editor's added fields actually reach the model. This mirrors existing unit tests in `tests/test_groq_agent.py`.
- **Prior art:** `tests/test_characters.py` (endpoint + file round-trip), `tests/test_config.py` (env/config + provider-status), `tests/test_groq_agent.py` (pure-function parsing). Follow these directly.
- **Frontend:** there is no JS test harness in the repo today; the Studio's UI is verified manually. Adding a JS test framework is out of scope.

## Out of Scope

- **Per-character voice / TTS settings** — voice stays a global env setting; the `Character` schema is not extended.
- **Deleting characters** and a **set-default-from-dashboard** control (no `DELETE` endpoint; default still via `VOICE_AGENT_DEFAULT_CHARACTER`).
- **Authentication / multi-user / RBAC** — the Studio assumes the same trust model as the rest of the app (single-operator/local).
- **Character versioning, history, or audit metadata** (timestamps, authorship) on character files.
- **Tuning non-personality runtime config** (ambience, proactive policy, input gain, latency knobs) — this dashboard is personality-only.
- **A JS unit-test framework** for the frontend.
- **Reworking the existing diagnostics-drawer form** beyond linking out to the new Studio (the old form may remain or be slimmed later; not required here).
- **Cross-session agent memory** — being explored separately; not part of this PRD.

## Further Notes

- The backend lift is small: persistence and validation already exist via the `/characters` REST seams and the unchanged `Character` model; the only genuinely new server code is the stateless `POST /characters/preview` endpoint, which composes existing adapters (`build_system_prompt`, `GroqStreamingAgent`, `CartesiaStreamingTTS`).
- Keeping preview on the same adapters as the live session is the key correctness decision — it guarantees "what you hear in the Studio is what the call sounds like."
- The biggest *frontend* value is the structured editor for list fields and `example_exchanges`, which the current textarea form handles poorly or not at all.

## Verification

1. **Tests:** `pytest` — extended `tests/test_characters.py` (full-field round-trip + duplicate), new preview-endpoint tests, and the `build_system_prompt` unit test all pass.
2. **Mock mode end-to-end** (no keys): with `VOICE_AGENT_MODE=mock`, use the already-running dev server (another agent owns `uvicorn` — do not start a second one), open `/studio`, then:
   - Confirm all characters list and the default is flagged.
   - Open a character; confirm every field (incl. `example_exchanges`) is editable.
   - Create a new character and duplicate an existing one; confirm new files appear under `app/characters/`.
   - Type a test message → hear synthesized audio and see the text reply + latency.
   - Save edits, start a real call selecting that character, confirm the change is reflected on the next turn.
3. **Live mode spot check** (if keys available): repeat the preview and confirm real TTS audio; remove a required key and confirm a clear warning (not a crash).
