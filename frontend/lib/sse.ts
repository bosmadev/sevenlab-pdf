/**
 * Minimal SSE parser over fetch streams.
 * Backend contract (T-012, rag-impl-1): newline-delimited `data: <json>\n\n`.
 * Frame shapes (per rag-impl-1):
 *   { "type": "citations", "citations": [{n, doc_id, chunk_id, page, filename, score, preview}] }
 *   { "type": "token",      "text": "..." }         // many, in order
 *   { "type": "done" }                               // once at end
 *   { "type": "error",      "message": "..." }
 * The parser also accepts `"delta"` as a tolerant alias of `"token"` so a
 * downgraded backend or a test harness still streams.
 * Ordering guarantee from T-012: `citations` arrives before the first token.
 */
import { redirectToLogin } from "./session";
import type { Citation } from "./types";

export type SseEvent =
  | { type: "token"; text: string }
  | { type: "delta"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "done" }
  | { type: "error"; message: string };

export interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (ev: SseEvent) => void;
}

export async function streamChat(
  body: unknown,
  { signal, onEvent }: StreamOptions,
): Promise<void> {
  const res = await fetch("/api/chat/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
    credentials: "same-origin",
  });

  if (res.status === 401) {
    redirectToLogin();
    onEvent({ type: "error", message: "session expired — sign in again" });
    return;
  }
  if (!res.ok || !res.body) {
    const msg = `HTTP ${res.status}`;
    onEvent({ type: "error", message: msg });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        const ev = JSON.parse(payload) as SseEvent;
        onEvent(ev);
        if (ev.type === "done" || ev.type === "error") return;
      } catch {
        // ignore malformed frame; backend owns schema
      }
    }
  }
}
