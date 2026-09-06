/* Loads the real frontend/src/app.js under a minimal DOM stub.
 *
 * There is no JS toolchain here and adding one would cost more than it is
 * worth, so app.js is evaluated in a vm context with just enough of a browser
 * around it to reach `window.OmniCare`. Nothing is reimplemented: the functions
 * under test are the ones that ship.
 *
 * Shared by every harness in this directory, so the stub cannot drift into two
 * versions that disagree about what a browser does.
 */
import { readFileSync } from "node:fs";
import vm from "node:vm";

const escape = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function stubEl() {
  return {
    addEventListener() {},
    appendChild() {},
    removeChild() {},
    remove() {},
    focus() {},
    classList: { add() {}, remove() {} },
    children: [],
    dataset: {},
    style: {},
    hidden: false,
    value: "",
    set textContent(v) {
      this.innerHTML = escape(v);
    },
    get textContent() {
      return "";
    },
    innerHTML: "",
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    closest: () => null,
    parentElement: null,
  };
}

/** Evaluate app.js and return its `window.OmniCare` export. */
export function loadApp({ randomUUID = true } = {}) {
  // Counted rather than time-based: the real `randomUUID` returns a different
  // value every call, and a stub keyed on Date.now() returns the same one twice
  // inside a millisecond - which fails a "two resets differ" check for a reason
  // that has nothing to do with the code under test.
  let n = 0;
  const crypto = randomUUID
    ? { randomUUID: () => `11111111-2222-4333-8444-${String(++n).padStart(12, "0")}` }
    : {};

  const sandbox = {
    window: { crypto },
    crypto,
    console,
    localStorage: { getItem: () => "usr_test", setItem() {} },
    document: {
      createElement: stubEl,
      getElementById: stubEl,
      querySelector: stubEl,
      addEventListener() {},
    },
    WebSocket: function () {
      return { addEventListener() {}, close() {} };
    },
    fetch: async () => ({ ok: true, json: async () => ({}) }),
    setTimeout,
    clearTimeout,
    Math,
    JSON,
    Date,
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(readFileSync("frontend/src/app.js", "utf8"), sandbox);
  return sandbox.window.OmniCare;
}

/** Report results and exit non-zero if any check failed. */
export function report(checks, extra = "") {
  const failed = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  if (failed.length) {
    console.error("FAILED: " + failed.join("; "));
    if (extra) console.error(extra);
    process.exit(1);
  }
  console.log("ok " + Object.keys(checks).length + " checks");
}
