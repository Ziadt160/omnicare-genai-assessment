/* "New conversation" has to actually start one.
 *
 * Reported by a user and reproduced against the running stack: clicking it
 * emptied the screen and nothing else. The next message came back answered
 * with context the policyholder could no longer see - "What did I just tell
 * you about?" described the burst pipe from the conversation they had just
 * cleared.
 *
 * The cause was a reasonable-looking assumption. `resetConversation` set the
 * id to null, and the comment above it said the gateway would mint a fresh one
 * on the next turn. It does the opposite: `ConversationStore.ensure` resolves a
 * request with no conversation id to the user's *most recent* conversation, on
 * purpose, so that the graded request schema - which has no conversation_id
 * field at all - can still hold a multi-turn thread. Dropping the id therefore
 * resolved straight back to the conversation being abandoned.
 *
 * So the new conversation is named by the client. These checks pin that, and
 * pin the fallback for a context where `crypto.randomUUID` is unavailable.
 */
import { loadApp, report } from "./dom_stub.mjs";

const app = loadApp();

const userBefore = app.USER_ID;

// Fresh page: no id yet, so the first message continues the user's most recent
// conversation. That is deliberate and must not change - a reload should not
// lose the thread.
const atLoad = app.conversationId;

app.conversationId = "conversation-from-the-server";
const beforeReset = app.conversationId;

app.resetConversation();
const afterReset = app.conversationId;

app.resetConversation();
const afterSecondReset = app.conversationId;

const withoutWebCrypto = loadApp({ randomUUID: false });
withoutWebCrypto.resetConversation();
const fallback = withoutWebCrypto.conversationId;

report({
  "a fresh page starts with no id": atLoad === null,
  "an id from the server is kept": beforeReset === "conversation-from-the-server",
  // The bug: this used to be null, which resolves server-side to the
  // conversation just abandoned.
  "reset does not merely clear the id": afterReset !== null,
  "reset leaves a new id": afterReset !== beforeReset,
  "reset twice gives two different ids": afterSecondReset !== afterReset,
  "the id fits the contract's 64-char limit": afterReset.length <= 64,
  "there is a fallback without crypto.randomUUID":
    typeof fallback === "string" && fallback.length > 8 && fallback.length <= 64,

  // A new conversation is a new person. History, claims and the rate limiter
  // are all keyed on the user, and `ensure` resolves an id-less request to the
  // user's most recent conversation - so keeping the identity leaves a route
  // back into the thread that was just cleared.
  "reset takes a new user identity": app.USER_ID !== userBefore,

  // Reported from a real session: the readback appeared twice, once as a chat
  // bubble and once verbatim in the panel below it. Tolerable while it was one
  // line; it is five now, with the payment split.
  "the confirm panel asks only the question": (() => {
    const readback = [
      "I'm about to file a Water Damage claim on policy P-O-L, one zero nine two for $40000.",
      "",
      "Claim amount: $40,000.00",
      "OmniCare pays: $25,000.00",
      "You pay: $15,000.00",
      "",
      "Shall I go ahead?",
    ].join(String.fromCharCode(10));
    return app.confirmQuestion(readback) === "Shall I go ahead?";
  })(),
  "an unrecognised readback still gets a question": (() => {
    return app.confirmQuestion("Filing this now.") === "File this claim?";
  })(),
  "the exported identity follows the reset":
    app.USER_ID !== null && app.USER_ID.startsWith("usr_"),
}, `afterReset=${afterReset} fallback=${fallback} user=${userBefore}->${app.USER_ID}`);
