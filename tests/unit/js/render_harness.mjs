/* The chat renderer, exercised against the answer that was reported broken.
 *
 * Loads the real app.js through the shared DOM stub and calls its exported
 * renderText directly. Run from the repo root by
 * tests/unit/test_chat_renderer.py, which skips when node is not installed.
 *
 * Exits non-zero and names every failed check.
 */
import { loadApp, report } from "./dom_stub.mjs";

const { renderText } = loadApp();


/* Reported with a screenshot: this arrived on screen with the hashes and the
   hyphens showing, because the renderer handled bold and paragraphs and
   nothing else. */
const REPORTED = [
  "Based on the policy documents, here is how your situation is covered:",
  "",
  "### Section 1: Home Water Damage Coverage",
  "- **Coverage**: This section covers water damage caused by sudden pipe bursts.",
  "- **Limitations**:",
  "- The coverage includes up to $25,000 for water damage with a $500 deductible.",
  "- Gradual leaks or flood damage are strictly excluded.",
  "",
  "### Summary",
  "1. The pipe burst is covered up to $25,000 with a $500 deductible.",
  "2. Your TV falls under Section 2.",
].join("\n");

const out = renderText(REPORTED);
const FENCE = "```";

const checks = {
  "no literal hashes": !out.includes("###"),
  "heading becomes an element": /<h[1-6] class="md-h">Section 1: Home Water Damage Coverage<\/h[1-6]>/.test(out),
  "bullets become a list": (out.match(/<li>/g) || []).length >= 6,
  "ordered list is ordered": out.includes("<ol>"),
  "no stray bullet markers": !/>\s*-\s/.test(out),
  "bold still renders": out.includes("<strong>Coverage</strong>"),
  "figures survive": out.includes("$25,000") && out.includes("$500"),
  // Escaping happens before any markdown substitution, so nothing the model
  // writes can inject markup. This is the property that matters most here: the
  // renderer gained block elements, and a block-element renderer that trusted
  // its input would be a far worse bug than the one it fixed.
  "escapes html first": (() => {
    const evil = renderText("### <img src=x onerror=alert(1)>");
    return !evil.includes("<img") && evil.includes("&lt;img");
  })(),
  "a fenced whole answer is unfenced": (() => {
    const fenced = FENCE + "\nYour deductible is $500.\n" + FENCE;
    return !renderText(fenced).includes(FENCE);
  })(),
  // Found in review. Classifying a block by whether its rendered HTML starts
  // with "<" welded paragraphs together, because "**Coverage**: ..." - the most
  // common shape a model emits, and the shape in the reported screenshot -
  // renders to inline HTML beginning with a tag.
  "a paragraph starting with bold keeps its own <p>":
    renderText("**Coverage**: sudden pipe bursts.") ===
    "<p><strong>Coverage</strong>: sudden pipe bursts.</p>",
  "two such paragraphs do not weld together": (() => {
    const out = renderText("**bold** and more\n\n**bold2** again");
    return (out.match(/<p>/g) || []).length === 2;
  })(),
};

const failed = Object.entries(checks)
  .filter(([, ok]) => !ok)
  .map(([name]) => name);

if (failed.length) {
  console.error("FAILED: " + failed.join("; "));
  console.error("rendered: " + out);
  process.exit(1);
}
console.log("ok " + Object.keys(checks).length + " checks");
