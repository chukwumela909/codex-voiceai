/* Personality Studio — browse, edit, create, duplicate, and voice-preview characters.
 * Reuses the existing REST seams: GET/PUT/POST /characters, plus POST /characters/preview.
 */
(() => {
  "use strict";

  const LIST_FIELDS = {
    tone: { el: "listTone", kind: "string" },
    speaking_style_rules: { el: "listRules", kind: "string" },
    forbidden_phrases: { el: "listForbidden", kind: "string" },
    values: { el: "listValues", kind: "string" },
    likes: { el: "listLikes", kind: "string" },
    dislikes: { el: "listDislikes", kind: "string" },
    boundaries: { el: "listBoundaries", kind: "string" },
    caller_goals: { el: "listGoals", kind: "string" },
    example_exchanges: {
      el: "listExamples",
      kind: "pair",
      pair: [
        { key: "user", ph: "User says…" },
        { key: "assistant", ph: "Character replies…" },
      ],
    },
    profile_facts: {
      el: "listFacts",
      kind: "pair",
      pair: [
        { key: "label", ph: "Label (e.g. Age)" },
        { key: "value", ph: "Value (e.g. 47)" },
      ],
    },
    stories: {
      el: "listStories",
      kind: "pair",
      pair: [
        { key: "title", ph: "Title (e.g. The band)" },
        { key: "content", ph: "What happened" },
      ],
    },
  };

  const $ = (id) => document.getElementById(id);

  const state = {
    charactersById: {},
    defaultId: null,
    currentId: null, // id of the loaded existing character; null when drafting a new one
    dirty: false,
  };

  // ---- helpers ----------------------------------------------------------------

  function slugify(name) {
    let slug = (name || "")
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^[-_]+|[-_]+$/g, "")
      .slice(0, 48);
    if (!slug) slug = "character";
    if (!/^[a-z0-9]/.test(slug)) slug = "c" + slug.slice(0, 47);
    return slug;
  }

  function markDirty() {
    state.dirty = true;
    $("saveStatus").textContent = "";
  }

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      /* no body */
    }
    if (!res.ok) {
      const detail = data && data.detail ? JSON.stringify(data.detail) : res.statusText;
      throw new Error(detail);
    }
    return data;
  }

  // ---- list-row editors -------------------------------------------------------

  function makeRowControls(container, row) {
    const controls = document.createElement("div");
    controls.className = "list-rows__ctl";
    const up = document.createElement("button");
    up.type = "button";
    up.className = "icon-btn";
    up.textContent = "↑";
    up.title = "Move up";
    up.addEventListener("click", () => {
      const prev = row.previousElementSibling;
      if (prev) container.insertBefore(row, prev);
      markDirty();
    });
    const down = document.createElement("button");
    down.type = "button";
    down.className = "icon-btn";
    down.textContent = "↓";
    down.title = "Move down";
    down.addEventListener("click", () => {
      const next = row.nextElementSibling;
      if (next) container.insertBefore(next, row);
      markDirty();
    });
    const del = document.createElement("button");
    del.type = "button";
    del.className = "icon-btn icon-btn--danger";
    del.textContent = "✕";
    del.title = "Remove";
    del.addEventListener("click", () => {
      row.remove();
      markDirty();
    });
    controls.append(up, down, del);
    return controls;
  }

  function addStringRow(field, value = "") {
    const container = $(LIST_FIELDS[field].el);
    const row = document.createElement("div");
    row.className = "list-rows__row";
    const input = document.createElement("input");
    input.type = "text";
    input.value = value;
    input.dataset.role = "value";
    input.addEventListener("input", markDirty);
    row.append(input, makeRowControls(container, row));
    container.appendChild(row);
    return row;
  }

  function addPairRow(field, value = {}) {
    const cfg = LIST_FIELDS[field];
    const container = $(cfg.el);
    const row = document.createElement("div");
    row.className = "list-rows__row list-rows__row--pair";
    const pair = document.createElement("div");
    pair.className = "list-rows__pair";
    cfg.pair.forEach(({ key, ph }) => {
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = ph;
      input.value = value[key] || "";
      input.dataset.role = key;
      input.addEventListener("input", markDirty);
      pair.appendChild(input);
    });
    row.append(pair, makeRowControls(container, row));
    container.appendChild(row);
    return row;
  }

  function addRow(field, value) {
    if (LIST_FIELDS[field].kind === "pair") return addPairRow(field, value);
    return addStringRow(field, value);
  }

  function clearList(field) {
    $(LIST_FIELDS[field].el).innerHTML = "";
  }

  function readList(field) {
    const container = $(LIST_FIELDS[field].el);
    const rows = [...container.querySelectorAll(".list-rows__row")];
    if (LIST_FIELDS[field].kind === "pair") {
      const keys = LIST_FIELDS[field].pair.map((p) => p.key);
      return rows
        .map((r) => {
          const obj = {};
          keys.forEach((k) => {
            obj[k] = r.querySelector(`[data-role="${k}"]`).value.trim();
          });
          return obj;
        })
        .filter((obj) => keys.some((k) => obj[k]));
    }
    return rows.map((r) => r.querySelector('[data-role="value"]').value.trim()).filter(Boolean);
  }

  // ---- form <-> character -----------------------------------------------------

  function writeForm(char) {
    $("fName").value = char.name || "";
    $("fRole").value = char.role || "";
    $("fGrammar").value = char.grammar || "";
    $("fIdentity").value = char.identity_response_style || "";
    $("fBackstory").value = char.backstory || "";
    $("fCallerRelationship").value = char.caller_relationship || "";
    $("fConversationSetting").value = char.conversation_setting || "";
    for (const field of Object.keys(LIST_FIELDS)) {
      clearList(field);
      const values = char[field] || [];
      values.forEach((v) => addRow(field, v));
    }
  }

  function readForm() {
    const char = {
      name: $("fName").value.trim(),
      role: $("fRole").value.trim(),
      grammar: $("fGrammar").value.trim(),
      identity_response_style: $("fIdentity").value.trim(),
      backstory: $("fBackstory").value.trim(),
      caller_relationship: $("fCallerRelationship").value.trim(),
      conversation_setting: $("fConversationSetting").value.trim(),
      tone: readList("tone"),
      speaking_style_rules: readList("speaking_style_rules"),
      forbidden_phrases: readList("forbidden_phrases"),
      values: readList("values"),
      likes: readList("likes"),
      dislikes: readList("dislikes"),
      boundaries: readList("boundaries"),
      caller_goals: readList("caller_goals"),
      profile_facts: readList("profile_facts"),
      stories: readList("stories"),
      example_exchanges: readList("example_exchanges"),
    };
    if (state.currentId) char.id = state.currentId;
    return char;
  }

  function setEditorEnabled(enabled) {
    $("saveBtn").disabled = !enabled;
    $("duplicateBtn").disabled = !enabled;
    $("previewBtn").disabled = !enabled;
  }

  function showEditor(char, { isNew } = { isNew: false }) {
    state.currentId = isNew ? null : char.id;
    state.dirty = isNew;
    writeForm(char);
    $("editorTitle").textContent = isNew ? "New character" : char.name || char.id;
    $("editorId").textContent = isNew
      ? `id will be: ${slugify(char.name) || "character"}`
      : `id: ${char.id}`;
    $("saveStatus").textContent = "";
    setEditorEnabled(true);
    renderList();
  }

  // ---- character list ---------------------------------------------------------

  function renderList() {
    const ul = $("characterList");
    ul.innerHTML = "";
    const chars = Object.values(state.charactersById).sort((a, b) =>
      (a.name || a.id).localeCompare(b.name || b.id)
    );
    for (const char of chars) {
      const li = document.createElement("li");
      li.className = "char-list__item";
      if (char.id === state.currentId) li.classList.add("is-active");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "char-list__btn";
      const name = document.createElement("span");
      name.className = "char-list__name";
      name.textContent = char.name || char.id;
      const role = document.createElement("span");
      role.className = "char-list__role";
      role.textContent = char.role || "";
      btn.append(name, role);
      if (char.id === state.defaultId) {
        const badge = document.createElement("span");
        badge.className = "char-list__default";
        badge.textContent = "default";
        name.appendChild(badge);
      }
      btn.addEventListener("click", () => onSelect(char.id));
      li.appendChild(btn);
      if (char.id !== state.defaultId) {
        const makeDefault = document.createElement("button");
        makeDefault.type = "button";
        makeDefault.className = "char-list__make-default";
        makeDefault.textContent = "Make default";
        makeDefault.title = `Use ${char.name || char.id} on the phone (applies to new calls)`;
        makeDefault.addEventListener("click", (event) => {
          event.stopPropagation();
          setDefault(char.id);
        });
        li.appendChild(makeDefault);
      }
      ul.appendChild(li);
    }
  }

  async function setDefault(id) {
    const status = $("saveStatus");
    try {
      const data = await api("PUT", "/characters/default", { id });
      state.defaultId = data.default;
      renderList();
      if (status) {
        const name = state.charactersById[data.default]?.name || data.default;
        status.textContent = `Default set to ${name} — new calls use it (no restart needed).`;
        status.dataset.tone = "ok";
      }
    } catch (err) {
      if (status) {
        status.textContent = `Could not set default: ${err.message}`;
        status.dataset.tone = "error";
      }
    }
  }

  function confirmDiscard() {
    if (!state.dirty) return true;
    return window.confirm("You have unsaved changes. Discard them?");
  }

  function onSelect(id) {
    if (id === state.currentId && !state.dirty) return;
    if (!confirmDiscard()) return;
    const char = state.charactersById[id];
    if (char) showEditor(char, { isNew: false });
  }

  async function loadCharacters() {
    const data = await api("GET", "/characters");
    state.defaultId = data.default;
    state.charactersById = {};
    for (const c of data.characters) state.charactersById[c.id] = c;
    renderList();
  }

  // ---- actions ----------------------------------------------------------------

  function onNew() {
    if (!confirmDiscard()) return;
    showEditor({ name: "", role: "" }, { isNew: true });
  }

  function onDuplicate() {
    const draft = readForm();
    draft.name = (draft.name || "Character") + " copy";
    delete draft.id;
    showEditor(draft, { isNew: true });
    $("saveStatus").textContent = "Duplicated — edit and save to create a new character.";
  }

  async function onSave(event) {
    event.preventDefault();
    const form = $("editorForm");
    if (!form.reportValidity()) return;
    const draft = readForm();
    const status = $("saveStatus");
    try {
      let saved;
      if (state.currentId) {
        saved = await api("PUT", `/characters/${encodeURIComponent(state.currentId)}`, draft);
      } else {
        draft.id = slugify(draft.name);
        saved = await api("POST", "/characters", draft);
      }
      state.charactersById[saved.id] = saved;
      state.currentId = saved.id;
      state.dirty = false;
      $("editorTitle").textContent = saved.name || saved.id;
      $("editorId").textContent = `id: ${saved.id}`;
      status.textContent = "Saved ✓";
      status.dataset.tone = "ok";
      renderList();
    } catch (err) {
      status.textContent = `Save failed: ${err.message}`;
      status.dataset.tone = "error";
    }
  }

  // ---- voice preview ----------------------------------------------------------

  let audioCtx = null;

  function playPcm(base64, sampleRate) {
    if (!base64) return;
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const frames = Math.floor(bytes.byteLength / 2);
    const buffer = audioCtx.createBuffer(1, frames, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < frames; i++) {
      channel[i] = view.getInt16(i * 2, true) / 32768;
    }
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.start();
  }

  async function onPreview() {
    const message = $("previewMessage").value.trim();
    if (!message) {
      $("previewMessage").focus();
      return;
    }
    const btn = $("previewBtn");
    const result = $("previewResult");
    const meta = $("previewMeta");
    const warnings = $("previewWarnings");
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
      const data = await api("POST", "/characters/preview", {
        message,
        character: readForm(),
      });
      result.hidden = false;
      $("previewText").textContent = data.text || "";
      meta.textContent = `LLM ${data.providers.llm} · ${data.latency.llm_ms} ms · TTS ${data.providers.tts} · ${data.latency.tts_ms} ms`;
      warnings.innerHTML = "";
      (data.warnings || []).forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w.code ? `${w.code}: ${w.message}` : `${w.provider}: ${w.message}`;
        warnings.appendChild(li);
      });
      playPcm(data.audio, data.sample_rate);
    } catch (err) {
      result.hidden = false;
      $("previewText").textContent = "";
      meta.textContent = "";
      warnings.innerHTML = "";
      const li = document.createElement("li");
      li.textContent = `Preview failed: ${err.message}`;
      warnings.appendChild(li);
    } finally {
      btn.disabled = false;
      btn.textContent = "▶ Preview voice";
    }
  }

  // ---- wiring -----------------------------------------------------------------

  function init() {
    $("newBtn").addEventListener("click", onNew);
    $("duplicateBtn").addEventListener("click", onDuplicate);
    $("editorForm").addEventListener("submit", onSave);
    $("previewBtn").addEventListener("click", onPreview);

    document.querySelectorAll("[data-add]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const field = btn.dataset.add;
        const isPair = LIST_FIELDS[field] && LIST_FIELDS[field].kind === "pair";
        addRow(field, isPair ? {} : "");
        markDirty();
      });
    });

    // keep the derived id hint fresh while drafting a new character
    $("fName").addEventListener("input", () => {
      markDirty();
      if (!state.currentId) {
        $("editorId").textContent = `id will be: ${slugify($("fName").value) || "character"}`;
      }
    });

    window.addEventListener("beforeunload", (e) => {
      if (state.dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    });

    loadCharacters().catch((err) => {
      $("editorTitle").textContent = "Failed to load characters";
      $("editorId").textContent = err.message;
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
