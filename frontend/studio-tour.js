// studio-tour.js — guided walkthrough for the /studio Personality Studio.
//
// Same engine as the /pipecat tour (tour.js); this file only holds the
// studio-specific theme override, step config, and boot wiring. It reads only
// DOM state studio.js maintains (button `disabled` attributes) so a tour
// failure can never break the studio itself.

import { createTour, mountReplayButton, storage } from "./tour.js";

const STORAGE_KEY = "codex.voiceai.tourDone.studio";

// The studio is a light-themed page (styles.css tokens); override the tour's
// dark defaults so the tooltip stays readable.
const THEME_CSS = `
:root {
  --ctour-panel: #ffffff;
  --ctour-ink: #0c0a09;
  --ctour-muted: #777169;
  --ctour-hairline: #e7e5e4;
  --ctour-accent: #a7e5d3;
  --ctour-on-accent: #0c0a09;
  --ctour-warn: #b45309;
}`;

function injectTheme() {
  if (document.getElementById("ctour-theme-studio")) return;
  const style = document.createElement("style");
  style.id = "ctour-theme-studio";
  style.textContent = THEME_CSS;
  document.head.appendChild(style);
}

// studio.js enables Save/Duplicate/Preview via setEditorEnabled(true) when a
// character is selected or "+ New" is clicked — the removed `disabled`
// attribute on #saveBtn is the "editor is active" signal.
const editorActive = () => {
  const btn = document.getElementById("saveBtn");
  return !!btn && !btn.disabled;
};

const steps = [
  {
    id: "welcome",
    title: "Welcome to the Personality Studio",
    body:
      "This is where the people you talk to are made — their backstory, voice " +
      "style, and how they answer the phone. Quick look around?",
    nextLabel: "Start tour",
    advance: { type: "next" },
  },
  {
    id: "characters",
    target: ".studio__list",
    title: "Pick a character",
    body:
      "Your characters live here. Click one to open it in the editor — or hit " +
      "“+ New” to start from scratch.",
    interactive: true,
    skipIf: editorActive,
    advance: { type: "attr", target: "#saveBtn", attr: "disabled", equals: null },
    fallbackNextAfterMs: 15000,
    fallbackNextText: "You can also press Next and pick a character later.",
  },
  {
    id: "essentials",
    target: "#fRole",
    title: "Name and role",
    body:
      "The essentials: a name, and a one-line role that pitches who this person " +
      "is. Everything else builds on these.",
    advance: { type: "next" },
  },
  {
    id: "mode",
    target: "#fConversationMode",
    title: "Conversation mode",
    body:
      "Social characters behave like a participant with their own point of view; " +
      "assistants stay task- and service-oriented.",
    advance: { type: "next" },
  },
  {
    id: "greeting",
    target: "#fGreeting",
    title: "Call greeting",
    body: "The exact line they answer the call with — it plays the moment you connect.",
    advance: { type: "next" },
  },
  {
    id: "backstory",
    target: "#fBackstory",
    title: "Life & backstory",
    body:
      "The fixed facts of this person's life. The more you fill in here — and in " +
      "the facts, values, and stories below — the less the voice improvises or " +
      "contradicts itself.",
    advance: { type: "next" },
  },
  {
    id: "preview",
    target: ".studio__preview",
    title: "Hear it before you save",
    body:
      "Voice preview runs your current edits through the language model and " +
      "text-to-speech without saving anything.",
    advance: { type: "next" },
  },
  {
    id: "save",
    target: "#saveBtn",
    title: "Save your character",
    body:
      "Save publishes the character to the server — it then shows up in the " +
      "character picker on the call page.",
    placement: "bottom",
    advance: { type: "next" },
  },
  {
    id: "done",
    title: "That's the Studio",
    body:
      "Head “Back to call” to talk to your creation. Replay this tour any time " +
      "with the “?” button in the corner.",
    nextLabel: "Finish",
    advance: { type: "next" },
  },
];

const tour = createTour(steps, { storageKey: STORAGE_KEY });

function boot() {
  injectTheme();
  mountReplayButton({ onClick: () => tour.start() });
  // setTimeout rather than requestAnimationFrame: rAF is suspended in hidden
  // or embedded tabs, which would silently kill the auto-start.
  setTimeout(() => {
    if (storage.get(STORAGE_KEY, "unavailable") === null) tour.start();
  }, 0);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
