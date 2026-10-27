/**
 * Per-document chat history over `GET /api/documents/{id}/chat` (T-013).
 *
 * Writes happen server-side: `app/rag/pipeline.answer_stream` stores the
 * question and the finished answer (with citations) before it sends the
 * `done` frame, so the client only ever reads. A client-side POST here would
 * store every exchange twice.
 *
 * The stored citation carries the T-013 fields (`page_num`, `chunk_id`,
 * `chunk_index`) plus the live stream fields (`n`, `doc_id`, `filename`,
 * `score`, `preview`); rows written before those were stored fall back to
 * positional numbering and the document's own name.
 */
import { api } from "./api";
import type { ChatMessage, Citation, MessageRole } from "./types";

interface BackendCitation {
  n?: number | null;
  doc_id?: string | null;
  page_num?: number | null;
  chunk_id?: string | null;
  chunk_index?: number | null;
  filename?: string | null;
  score?: number | null;
  preview?: string | null;
}

interface BackendChatMessage {
  id: string;
  document_id: string;
  role: MessageRole;
  content: string;
  citations?: BackendCitation[];
  created_at: string;
}

interface BackendHistory {
  document_id: string;
  messages: BackendChatMessage[];
  total: number;
}

const historyPath = (docId: string): string =>
  `/api/documents/${encodeURIComponent(docId)}/chat`;

export async function loadHistory(
  docId: string,
  docName?: string,
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const h = await api.get<BackendHistory>(`${historyPath(docId)}?limit=200`, { signal });
  return (h.messages ?? []).map((m) => {
    const cites = m.citations ?? [];
    const citations: Citation[] | undefined =
      cites.length > 0
        ? cites.map((c, i) => ({
            n: c.n ?? i + 1,
            doc_id: c.doc_id ?? docId,
            page: c.page_num ?? 0,
            chunk_id: c.chunk_id ?? undefined,
            filename: c.filename ?? docName,
            score: c.score ?? undefined,
            preview: c.preview ?? undefined,
          }))
        : undefined;
    const ts = Date.parse(m.created_at);
    return {
      id: `h-${m.id}`,
      role: m.role,
      content: m.content,
      citations,
      createdAt: Number.isFinite(ts) ? ts : Date.now(),
    };
  });
}
