// pipecat-tour.js — guided walkthrough for the /pipecat UI.
//
// Declarative step config for the generic engine in tour.js. Reads only DOM
// state the app maintains (element `disabled`, pill `data-on` attributes) —
// never variables from the page's inline module, so a tour failure can never
// break the voice app.

import { createTour, mountReplayButton, storage } from "./tour.js";

const STORAGE_KEY = "codex.voiceai.tourDone.pipecat";

const connected = () => document.getElementById("connPill")?.dataset.on === "true";

// The tour explains the whole UI first and ends on the one required action:
// clicking Connect. Once the connection holds, the wrap-up step closes out.
const steps = [
  {
    id: "welcome",
    title: "Welcome to the voice agent",
    body:
      "This quick tour walks you through starting your first conversation. " +
      "You'll need a microphone — the browser will ask for access at the end.",
    nextLabel: "Start tour",
    advance: { type: "next" },
  },
  {
    id: "character",
    target: "#characterSelect",
    title: "Pick a character",
    body:
      "Choose who you want to talk to — each character has its own personality. " +
      "The default works fine too.",
    skipIf: connected,
    waitFor: { check: (el) => !el.disabled, timeoutMs: 4000, onTimeout: "skip" },
    advance: { type: "next" },
  },
  {
    id: "voice",
    target: "#voiceBar",
    title: "Choose a voice",
    body:
      "Pick an ElevenLabs voice from the list, or paste a voice id. " +
      "“Save voice” makes it stick for future sessions.",
    advance: { type: "next" },
  },
  {
    id: "model",
    target: "#modelBar",
    title: "Choose a model",
    body:
      "This is the language model that powers the conversation — pick a preset " +
      "or paste any OpenRouter model id.",
    advance: { type: "next" },
  },
  {
    id: "pills",
    target: ".pills",
    title: "Status at a glance",
    body:
      "These pills show the connection, your microphone, and whether the agent " +
      "currently hears you speaking.",
    advance: { type: "next" },
  },
  {
    id: "transcript",
    target: ".transcript",
    title: "Live transcript",
    body:
      "Everything you and the agent say will land here — the italic line at the " +
      "bottom is live partial speech; finished lines appear above it.",
    advance: { type: "next" },
  },
  {
    id: "disconnect",
    target: "#disconnect",
    title: "Ending a session",
    body:
      "When you're done talking, click Disconnect. Curious about the plumbing? " +
      "The Event log below the transcript shows the raw session events.",
    advance: { type: "next" },
  },
  {
    id: "connect",
    target: "#connect",
    title: "Now — click Connect",
    body:
      "This starts a live session. Your browser may ask for microphone access — " +
      "click Allow. The tour wraps up once you're connected.",
    interactive: true,
    skipIf: connected,
    // The pill can flash "connected" before mic capture fails and flips it to
    // "error" — require the connected state to hold before moving on.
    advance: { type: "attr", target: "#connPill", attr: "data-on", equals: "true", stableForMs: 800 },
    failOn: {
      target: "#connPill",
      attr: "data-on",
      equals: "error",
      message: "Connection failed — check the server and mic permission, then click Connect again.",
      action: "stay",
    },
  },
  {
    id: "done",
    title: "You're connected — say hello!",
    body:
      "Just start talking; the agent is listening. Replay this tour any time " +
      "with the “?” button in the corner.",
    nextLabel: "Finish",
    advance: { type: "next" },
    failOn: {
      target: "#connPill",
      attr: "data-on",
      equals: ["false", "error"],
      message: "Looks like the session dropped — let's reconnect.",
      action: { goto: "connect" },
    },
  },
];

const tour = createTour(steps, { storageKey: STORAGE_KEY });

function boot() {
  mountReplayButton({ onClick: () => tour.start() });
  // setTimeout rather than requestAnimationFrame: rAF is suspended in hidden
  // or embedded tabs, which would silently kill the auto-start.
  setTimeout(() => {
    // Auto-start only on a genuine first visit. If localStorage is unavailable
    // the sentinel is returned and we never auto-start (better than nagging on
    // every visit); the "?" button still allows manual runs.
    if (storage.get(STORAGE_KEY, "unavailable") === null) tour.start();
  }, 0);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
