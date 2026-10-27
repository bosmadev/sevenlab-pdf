/**
 * Citation shape from T-012 RAG pipeline (`rag-impl-1` contract).
 * `n` is the 1-indexed label the model uses in inline `[n]` refs.
 * Older/local synonyms (`doc_name`, `snippet`) are kept so exported
 * chats and legacy fixtures still render.
 */
export interface Citation {
  n?: number;
  doc_id: string;
  chunk_id?: string;
  page: number;
  filename?: string;
  doc_name?: string;
  preview?: string;
  snippet?: string;
  score?: number;
}

/** Resolve the best display name/snippet from either the live or legacy fields. */
export const citationName = (c: Citation): string =>
  c.filename ?? c.doc_name ?? c.doc_id;
export const citationPreview = (c: Citation): string | undefined =>
  c.preview ?? c.snippet;

export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  createdAt: number;
  streaming?: boolean;
  error?: string;
}

export interface DocumentSummary {
  id: string;
  name: string;
  pages?: number;
  size_bytes?: number;
  uploaded_at?: string;
  status?: "processing" | "ready" | "failed";
}

export type ChatStatus = "idle" | "streaming" | "error";
