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
 *
 * `streamChat` resolves with how the stream ended so the caller can tell a
 * clean `done` from a socket that closed early (`eof`) and from an `error`
 * frame or non-2xx response. A 401 is reported with `status: 401` instead of
 * redirecting here; the caller owns navigation.
 */
import type { Citation } from "./types";

export type SseEvent =
  | { type: "token"; text: string }
  | { type: "delta"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "done" }
  | { type: "error"; message: string; status?: number };

export type StreamEnd = "done" | "error" | "eof";

export interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (ev: SseEvent) => void;
}

function parseFrame(frame: string): SseEvent | null {
  const line = frame.split("\n").find((l) => l.startsWith("data:"));
  if (!line) return null;
  const payload = line.slice(5).trim();
  if (!payload) return null;
  try {
    return JSON.parse(payload) as SseEvent;
  } catch {
    return null; // malformed frame; backend owns schema
  }
}

export async function streamChat(
  body: unknown,
  { signal, onEvent }: StreamOptions,
): Promise<StreamEnd> {
  const res = await fetch("/api/chat/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
    credentials: "same-origin",
  });

  if (res.status === 401) {
    onEvent({ type: "error", message: "session expired — sign in again", status: 401 });
    return "error";
  }
  if (!res.ok || !res.body) {
    onEvent({ type: "error", message: `HTTP ${res.status}`, status: res.status });
    return "error";
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (ev: SseEvent): StreamEnd | null => {
    onEvent(ev);
    if (ev.type === "done") return "done";
    if (ev.type === "error") return "error";
    return null;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const ev = parseFrame(frame);
      if (!ev) continue;
      const end = dispatch(ev);
      if (end) {
        void reader.cancel().catch(() => undefined);
        return end;
      }
    }
  }

  // Server closed without the trailing blank line: flush the last frame.
  buffer += decoder.decode();
  const tail = buffer.trim() ? parseFrame(buffer) : null;
  if (tail) {
    const end = dispatch(tail);
    if (end) return end;
  }
  return "eof";
}
