// tour.js — generic coach-mark / guided-walkthrough engine.
//
// Renders a full-viewport dimming overlay with a spotlight "hole" over one
// target element at a time, plus a tooltip describing the step. Steps advance
// either passively (Next button) or by observing real user actions (DOM
// events, attribute changes, predicates). Page-agnostic: step definitions
// live with the page that uses the engine (see pipecat-tour.js).
//
// The engine must never break the host page: every failure path degrades to
// console.warn + skip.

const SVG_NS = "http://www.w3.org/2000/svg";
const STYLE_ID = "ctour-style";
const PAD = 8; // spotlight padding around the target rect, px
const RADIUS = 10; // spotlight corner radius, px
const ANIM_MS = 180; // hole move animation between steps

const CSS = `
.ctour-overlay { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 60; pointer-events: none; }
.ctour-overlay .ctour-dim { pointer-events: auto; fill: rgba(12, 10, 9, 0.75); }
.ctour-overlay .ctour-ring { fill: none; stroke: var(--accent, #a7e5d3); stroke-width: 1.5; pointer-events: none; }
.ctour-overlay .ctour-blocker { fill: transparent; pointer-events: auto; }
.ctour-tooltip {
  position: fixed; z-index: 61; box-sizing: border-box;
  max-width: min(320px, calc(100vw - 32px));
  background: var(--panel, #1c1917); color: var(--ink, #f5f5f4);
  border: 1px solid var(--hairline, #292524); border-radius: 10px;
  padding: 0.9rem 1rem; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
}
.ctour-tooltip:focus { outline: none; }
.ctour-tooltip[data-place="center"] { left: 50%; top: 50%; transform: translate(-50%, -50%); }
.ctour-tooltip--pulse { box-shadow: 0 0 0 2px var(--accent, #a7e5d3), 0 12px 32px rgba(0, 0, 0, 0.45); }
.ctour-count { font-size: 0.72rem; color: var(--muted, #a8a29e); letter-spacing: 0.05em; margin-bottom: 0.35rem; }
.ctour-title { margin: 0 0 0.4rem; font-size: 1rem; font-weight: 600; }
.ctour-body { font-size: 0.86rem; color: var(--muted, #a8a29e); line-height: 1.45; }
.ctour-note { display: none; font-size: 0.8rem; margin-top: 0.5rem; color: var(--speak, #f4c5a8); line-height: 1.4; }
.ctour-note--show { display: block; }
.ctour-footer { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.75rem; }
.ctour-skip { background: none; border: none; color: var(--muted, #a8a29e); cursor: pointer; font-size: 0.8rem; padding: 0.25rem 0; font-family: inherit; }
.ctour-skip:hover { color: var(--ink, #f5f5f4); }
.ctour-next {
  margin-left: auto; background: var(--accent, #a7e5d3); color: #0c0a09;
  border: none; border-radius: 7px; padding: 0.4rem 0.9rem; cursor: pointer;
  font-size: 0.85rem; font-weight: 600; font-family: inherit;
}
.ctour-skip:focus-visible, .ctour-next:focus-visible { outline: 2px solid var(--accent, #a7e5d3); outline-offset: 2px; }
.ctour-arrow {
  position: absolute; width: 10px; height: 10px;
  background: var(--panel, #1c1917);
  border-left: 1px solid var(--hairline, #292524); border-top: 1px solid var(--hairline, #292524);
}
.ctour-tooltip[data-place="bottom"] .ctour-arrow { top: -6px; transform: rotate(45deg); }
.ctour-tooltip[data-place="top"] .ctour-arrow { bottom: -6px; transform: rotate(225deg); }
.ctour-tooltip[data-place="center"] .ctour-arrow { display: none; }
.ctour-replay {
  position: fixed; right: 16px; bottom: 16px; z-index: 55;
  width: 32px; height: 32px; border-radius: 50%;
  background: var(--panel, #1c1917); color: var(--muted, #a8a29e);
  border: 1px solid var(--hairline, #292524); cursor: pointer; font-size: 0.95rem; font-family: inherit;
}
.ctour-replay:hover { color: var(--ink, #f5f5f4); border-color: var(--accent, #a7e5d3); }
.ctour-replay:focus-visible { outline: 2px solid var(--accent, #a7e5d3); outline-offset: 2px; }
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}

// localStorage may be unavailable (private mode, blocked). get() returns
// `fallback` only when storage itself errors, so callers can tell "unset"
// (null) apart from "unavailable".
export const storage = {
  get(key, fallback = null) {
    try {
      return localStorage.getItem(key);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  },
};

export function mountReplayButton({ onClick, title = "Replay the guided tour" } = {}) {
  ensureStyle();
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ctour-replay";
  btn.textContent = "?";
  btn.title = title;
  btn.setAttribute("aria-label", title);
  btn.addEventListener("click", () => {
    try {
      onClick?.();
    } catch (err) {
      console.warn("[tour] replay handler failed", err);
    }
  });
  document.body.appendChild(btn);
  return { el: btn, remove: () => btn.remove() };
}

const safeCall = (fn, ...args) => {
  if (typeof fn !== "function") return undefined;
  try {
    return fn(...args);
  } catch (err) {
    console.warn("[tour] hook failed", err);
    return undefined;
  }
};

const valueMatches = (value, equals) =>
  Array.isArray(equals) ? equals.includes(value) : value === equals;

const inflate = (r, pad) => ({ x: r.left - pad, y: r.top - pad, w: r.width + pad * 2, h: r.height + pad * 2 });

const lerpRect = (a, b, t) => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
  w: a.w + (b.w - a.w) * t,
  h: a.h + (b.h - a.h) * t,
});

const reducedMotion = () =>
  typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

function roundedRectPath(r) {
  const rad = Math.max(0, Math.min(RADIUS, r.w / 2, r.h / 2));
  const w = r.w - 2 * rad;
  const h = r.h - 2 * rad;
  return (
    `M${r.x + rad},${r.y} h${w} a${rad},${rad} 0 0 1 ${rad},${rad} v${h} ` +
    `a${rad},${rad} 0 0 1 -${rad},${rad} h${-w} a${rad},${rad} 0 0 1 -${rad},-${rad} ` +
    `v${-h} a${rad},${rad} 0 0 1 ${rad},-${rad} z`
  );
}

// Outer viewport rect + optional inner hole; with fill-rule="evenodd" the
// hole is unpainted, so native SVG hit-testing lets clicks fall through to
// the real element beneath while the painted dim region swallows the rest.
function dimPath(hole) {
  let d = `M0,0 H${window.innerWidth} V${window.innerHeight} H0 Z`;
  if (hole) d += " " + roundedRectPath(hole);
  return d;
}

export function createTour(steps, options = {}) {
  let active = false;
  let index = -1;
  let token = 0; // bumps on every step transition; async work checks it
  let stepDisposables = [];
  let globalDisposables = [];
  let dom = null;
  let holeRect = null;
  let currentStep = null;
  let currentTarget = null;
  let prevFocus = null;
  let animFrame = null;
  let animWatchdog = null;

  const cancelAnim = () => {
    if (animFrame) cancelAnimationFrame(animFrame);
    animFrame = null;
    if (animWatchdog) clearTimeout(animWatchdog);
    animWatchdog = null;
  };

  function buildDom() {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "ctour-overlay");
    svg.setAttribute("aria-hidden", "true");
    const dim = document.createElementNS(SVG_NS, "path");
    dim.setAttribute("class", "ctour-dim");
    dim.setAttribute("fill-rule", "evenodd");
    const ring = document.createElementNS(SVG_NS, "path");
    ring.setAttribute("class", "ctour-ring");
    const blocker = document.createElementNS(SVG_NS, "rect");
    blocker.setAttribute("class", "ctour-blocker");
    blocker.style.display = "none";
    svg.append(dim, ring, blocker);
    dim.addEventListener("click", pulse);
    blocker.addEventListener("click", pulse);

    const tooltip = document.createElement("div");
    tooltip.className = "ctour-tooltip";
    tooltip.setAttribute("role", "dialog");
    tooltip.setAttribute("aria-labelledby", "ctourTitle");
    tooltip.setAttribute("aria-describedby", "ctourBody");
    tooltip.tabIndex = -1;
    tooltip.innerHTML = `
      <div class="ctour-count"></div>
      <h2 class="ctour-title" id="ctourTitle"></h2>
      <div class="ctour-body" id="ctourBody" aria-live="polite"></div>
      <div class="ctour-note" aria-live="polite"></div>
      <div class="ctour-footer">
        <button type="button" class="ctour-skip">Skip tour</button>
        <button type="button" class="ctour-next">Next</button>
      </div>
      <div class="ctour-arrow"></div>`;
    dom = {
      svg,
      dim,
      ring,
      blocker,
      tooltip,
      count: tooltip.querySelector(".ctour-count"),
      title: tooltip.querySelector(".ctour-title"),
      body: tooltip.querySelector(".ctour-body"),
      note: tooltip.querySelector(".ctour-note"),
      skip: tooltip.querySelector(".ctour-skip"),
      next: tooltip.querySelector(".ctour-next"),
      arrow: tooltip.querySelector(".ctour-arrow"),
    };
    dom.skip.addEventListener("click", () => stop("skipped"));
    dom.next.addEventListener("click", () => next());
    document.body.append(svg, tooltip);
  }

  function destroyDom() {
    if (!dom) return;
    dom.svg.remove();
    dom.tooltip.remove();
    dom = null;
  }

  let pulseTimer = null;
  function pulse() {
    if (!dom) return;
    dom.tooltip.classList.add("ctour-tooltip--pulse");
    clearTimeout(pulseTimer);
    pulseTimer = setTimeout(() => dom?.tooltip.classList.remove("ctour-tooltip--pulse"), 250);
  }

  function setNote(text, warn) {
    if (!dom) return;
    dom.note.textContent = text || "";
    dom.note.classList.toggle("ctour-note--show", !!text);
    dom.note.classList.toggle("ctour-note--warn", !!warn);
  }

  function renderHole() {
    if (!dom) return;
    dom.dim.setAttribute("d", dimPath(holeRect));
    if (holeRect) {
      dom.ring.setAttribute("d", roundedRectPath(holeRect));
      dom.ring.style.display = "";
    } else {
      dom.ring.style.display = "none";
    }
  }

  function setHole(target, animate) {
    cancelAnim();
    if (!animate || !target || !holeRect || reducedMotion()) {
      holeRect = target;
      renderHole();
      return;
    }
    const from = holeRect;
    const t0 = performance.now();
    const finish = () => {
      cancelAnim();
      holeRect = target;
      renderHole();
    };
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / ANIM_MS);
      if (k >= 1) return finish();
      holeRect = lerpRect(from, target, k * (2 - k)); // easeOutQuad
      renderHole();
      animFrame = requestAnimationFrame(tick);
    };
    animFrame = requestAnimationFrame(tick);
    // rAF is suspended in hidden/embedded tabs; make sure the hole still
    // lands on the target even if no frame ever ticks.
    animWatchdog = setTimeout(finish, ANIM_MS + 100);
  }

  function syncBlocker(rect, step) {
    if (!dom) return;
    if (rect && !step.interactive) {
      dom.blocker.setAttribute("x", rect.x);
      dom.blocker.setAttribute("y", rect.y);
      dom.blocker.setAttribute("width", rect.w);
      dom.blocker.setAttribute("height", rect.h);
      dom.blocker.style.display = "";
    } else {
      dom.blocker.style.display = "none";
    }
  }

  function placeTooltip(rect, step) {
    if (!dom) return;
    const t = dom.tooltip;
    if (!rect) {
      t.dataset.place = "center";
      t.style.left = "";
      t.style.top = "";
      return;
    }
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 12;
    const edge = 16;
    const tw = t.offsetWidth;
    const th = t.offsetHeight;
    // Single-column page: side placements fall back to vertical flow.
    let place = step.placement === "top" ? "top" : "bottom";
    if (place === "bottom" && rect.y + rect.h + margin + th > vh - 8) place = "top";
    if (place === "top" && rect.y - margin - th < 8) place = "bottom";
    let left = Math.round(rect.x + rect.w / 2 - tw / 2);
    left = Math.max(edge, Math.min(left, vw - edge - tw));
    const top = place === "bottom" ? Math.round(rect.y + rect.h + margin) : Math.round(rect.y - margin - th);
    t.dataset.place = place;
    t.style.left = left + "px";
    t.style.top = top + "px";
    const ax = Math.max(14, Math.min(rect.x + rect.w / 2 - left, tw - 14));
    dom.arrow.style.left = ax - 5 + "px";
  }

  function ensureVisible(el) {
    const r = el.getBoundingClientRect();
    if (r.top < 0 || r.bottom > window.innerHeight) {
      try {
        el.scrollIntoView({ block: "center" });
      } catch {
        /* ignore */
      }
    }
  }

  function reposition() {
    if (!active || !currentStep) return;
    let rect = null;
    if (currentStep.target) {
      const el = document.querySelector(currentStep.target);
      if (!el) return; // target vanished; keep last known layout
      currentTarget = el;
      rect = inflate(el.getBoundingClientRect(), PAD);
    }
    cancelAnim();
    holeRect = rect;
    renderHole();
    syncBlocker(rect, currentStep);
    placeTooltip(rect, currentStep);
  }

  const ctxFor = (el) => ({ el, index, next, goto, stop });

  function focusables() {
    const items = [];
    if (currentStep?.interactive && currentTarget?.isConnected && !currentTarget.disabled) {
      items.push(currentTarget);
    }
    if (dom && dom.next.style.display !== "none") items.push(dom.next);
    if (dom) items.push(dom.skip);
    return items;
  }

  function handleKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      stop("skipped");
      return;
    }
    if (e.key === "Tab") {
      const items = focusables();
      if (!items.length) return;
      e.preventDefault();
      const i = items.indexOf(document.activeElement);
      const j = e.shiftKey
        ? i <= 0
          ? items.length - 1
          : i - 1
        : i === -1 || i === items.length - 1
          ? 0
          : i + 1;
      try {
        items[j].focus({ preventScroll: true });
      } catch {
        /* ignore */
      }
    }
  }

  function syncNextButton(step, visible) {
    if (!dom) return;
    dom.next.style.display = visible ? "" : "none";
    const isLast = steps.indexOf(step) === steps.length - 1;
    dom.next.textContent = step.nextLabel || (isLast ? "Finish" : "Next");
  }

  // Schedules fn on a fresh macrotask, cancelled if the step changes first.
  function deferred(myToken, fn) {
    const timer = setTimeout(() => {
      if (token === myToken) fn();
    }, 0);
    stepDisposables.push(() => clearTimeout(timer));
  }

  function armAdvance(step, myToken) {
    const a = step.advance || { type: "next" };
    if (a.type === "next") {
      syncNextButton(step, true);
      return;
    }
    syncNextButton(step, false);
    if (step.fallbackNextAfterMs) {
      const timer = setTimeout(() => {
        setNote(step.fallbackNextText || "You can continue with Next.", false);
        syncNextButton(step, true);
      }, step.fallbackNextAfterMs);
      stepDisposables.push(() => clearTimeout(timer));
    }
    if (a.type === "event") {
      const sel = a.target || step.target;
      const handler = (e) => {
        if (e.target instanceof Element && e.target.closest(sel)) deferred(myToken, next);
      };
      document.addEventListener(a.on, handler, true);
      stepDisposables.push(() => document.removeEventListener(a.on, handler, true));
    } else if (a.type === "attr") {
      const el = document.querySelector(a.target);
      if (!el) {
        console.warn(`[tour] advance target ${a.target} missing; showing Next instead`);
        syncNextButton(step, true);
        return;
      }
      const matches = () => valueMatches(el.getAttribute(a.attr), a.equals);
      // stableForMs: the value must hold for N ms before advancing — guards
      // against transient states (e.g. "connected" flashing before an error).
      let stableTimer = null;
      const settle = () => {
        if (!a.stableForMs) {
          next();
          return;
        }
        clearTimeout(stableTimer);
        stableTimer = setTimeout(() => {
          if (token === myToken && matches()) next();
        }, a.stableForMs);
      };
      stepDisposables.push(() => clearTimeout(stableTimer));
      const mo = new MutationObserver(() => {
        if (matches()) settle();
        else clearTimeout(stableTimer);
      });
      mo.observe(el, { attributes: true, attributeFilter: [a.attr] });
      stepDisposables.push(() => mo.disconnect());
      if (matches()) {
        if (a.stableForMs) settle();
        else deferred(myToken, next);
      }
    } else if (a.type === "predicate") {
      const iv = setInterval(() => {
        if (safeCall(a.check)) next();
      }, a.pollMs || 300);
      stepDisposables.push(() => clearInterval(iv));
    }
  }

  function armFail(step, myToken) {
    const f = step.failOn;
    if (!f) return;
    const el = document.querySelector(f.target);
    if (!el) return;
    const matches = () => valueMatches(el.getAttribute(f.attr), f.equals);
    const trigger = () => {
      if (f.action && typeof f.action === "object" && f.action.goto) {
        goto(f.action.goto, { notice: f.message });
      } else {
        setNote(f.message, true); // "stay": advance observers remain armed
      }
    };
    if (matches()) {
      deferred(myToken, trigger);
      return;
    }
    const mo = new MutationObserver(() => {
      if (matches()) trigger();
    });
    mo.observe(el, { attributes: true, attributeFilter: [f.attr] });
    stepDisposables.push(() => mo.disconnect());
  }

  function waitForCheck(spec, el, myToken) {
    return new Promise((resolve) => {
      const started = performance.now();
      let timer = null;
      const tick = () => {
        if (token !== myToken) return resolve(false);
        if (safeCall(spec.check, el)) return resolve(true);
        if (performance.now() - started >= (spec.timeoutMs ?? 4000)) return resolve(false);
        timer = setTimeout(tick, 150);
      };
      stepDisposables.push(() => clearTimeout(timer));
      tick();
    });
  }

  function disposeStep() {
    const fns = stepDisposables;
    stepDisposables = [];
    for (const fn of fns.reverse()) {
      try {
        fn();
      } catch {
        /* ignore */
      }
    }
    if (currentStep) safeCall(currentStep.onExit, ctxFor(currentTarget));
    currentStep = null;
    currentTarget = null;
    setNote("", false);
  }

  function showStep(step, el, opts, myToken) {
    index = steps.indexOf(step);
    currentStep = step;
    currentTarget = el;
    dom.count.textContent = `${index + 1} of ${steps.length}`;
    dom.title.textContent = step.title || "";
    dom.body.textContent = step.body || "";
    setNote(opts.notice || "", !!opts.notice);

    let rect = null;
    if (el) {
      ensureVisible(el);
      rect = inflate(el.getBoundingClientRect(), PAD);
    }
    setHole(rect, true);
    syncBlocker(rect, step);
    armAdvance(step, myToken);
    armFail(step, myToken);
    placeTooltip(rect, step); // after armAdvance: Next visibility affects height
    safeCall(step.onEnter, ctxFor(el));

    const focusTarget =
      step.interactive && el && !el.disabled
        ? el
        : dom.next.style.display !== "none"
          ? dom.next
          : dom.tooltip;
    try {
      focusTarget.focus({ preventScroll: true });
    } catch {
      /* ignore */
    }
  }

  async function enterStep(i, opts = {}) {
    if (!active) return;
    const myToken = ++token;
    disposeStep();
    if (i >= steps.length) {
      stop("finished");
      return;
    }
    const step = steps[i];
    try {
      if (step.skipIf && safeCall(step.skipIf)) {
        enterStep(i + 1, opts);
        return;
      }
      const el = step.target ? document.querySelector(step.target) : null;
      if (step.target && !el) {
        console.warn(`[tour] missing target ${step.target}; skipping step "${step.id}"`);
        enterStep(i + 1, opts);
        return;
      }
      if (step.waitFor && el) {
        const ok = await waitForCheck(step.waitFor, el, myToken);
        if (token !== myToken || !active) return;
        if (!ok) {
          enterStep(i + 1, opts); // onTimeout: "skip" is the only mode for now
          return;
        }
      }
      showStep(step, el, opts, myToken);
    } catch (err) {
      console.warn(`[tour] step "${step.id}" failed; skipping`, err);
      enterStep(i + 1);
    }
  }

  function next() {
    if (!active) return;
    enterStep(index + 1);
  }

  function goto(id, opts = {}) {
    if (!active) return;
    const i = steps.findIndex((s) => s.id === id);
    if (i === -1) {
      console.warn(`[tour] goto: unknown step "${id}"`);
      return;
    }
    enterStep(i, opts);
  }

  function start(fromIndex = 0) {
    if (active) return;
    active = true;
    index = fromIndex - 1;
    prevFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    ensureStyle();
    buildDom();

    const onKey = handleKey;
    document.addEventListener("keydown", onKey, true);
    globalDisposables.push(() => document.removeEventListener("keydown", onKey, true));
    const onResize = () => reposition();
    window.addEventListener("resize", onResize);
    globalDisposables.push(() => window.removeEventListener("resize", onResize));
    const onScroll = () => reposition();
    document.addEventListener("scroll", onScroll, { capture: true, passive: true });
    globalDisposables.push(() => document.removeEventListener("scroll", onScroll, { capture: true }));
    if (typeof ResizeObserver === "function") {
      const ro = new ResizeObserver(() => reposition());
      ro.observe(document.body);
      globalDisposables.push(() => ro.disconnect());
    }
    enterStep(fromIndex);
  }

  function stop(reason = "skipped") {
    if (!active) return;
    active = false;
    token++;
    disposeStep();
    const fns = globalDisposables;
    globalDisposables = [];
    for (const fn of fns.reverse()) {
      try {
        fn();
      } catch {
        /* ignore */
      }
    }
    cancelAnim();
    destroyDom();
    index = -1;
    holeRect = null;
    if (options.storageKey) storage.set(options.storageKey, reason);
    if (prevFocus && prevFocus.isConnected) {
      try {
        prevFocus.focus({ preventScroll: true });
      } catch {
        /* ignore */
      }
    }
    prevFocus = null;
    safeCall(options.onStop, reason);
  }

  return { start, stop, next, goto, isActive: () => active };
}
