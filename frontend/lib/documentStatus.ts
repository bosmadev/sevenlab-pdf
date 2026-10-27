/**
 * Document processing status client + poller (T-014).
 *
 * Mirrors the FastAPI response from `GET /api/documents/{id}/status`:
 *
 *   { doc_id, status, message, progress_pct, chunk_count,
 *     page_count, ocr_tier_used, created_at, processed_at, is_terminal }
 *
 * The chat and upload UIs call `useDocumentStatus(docId)` and get a live
 * progress bar until status reaches a terminal state (`ready` | `error`).
 */

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";

export type ProcessingStatus =
  | "pending"
  | "parsing"
  | "chunking"
  | "embedding"
  | "ready"
  | "error";

export interface DocumentStatus {
  doc_id: string;
  status: ProcessingStatus;
  message: string | null;
  progress_pct: number;
  chunk_count: number;
  page_count: number | null;
  ocr_tier_used: string | null;
  created_at: string | null;
  processed_at: string | null;
  is_terminal: boolean;
}

export async function fetchDocumentStatus(
  docId: string,
  init?: { signal?: AbortSignal },
): Promise<DocumentStatus> {
  return api.get<DocumentStatus>(`/api/documents/${docId}/status`, init);
}

export async function retryDocument(docId: string): Promise<void> {
  await api.postJson(`/api/documents/${docId}/retry`, {});
}

export interface UseDocumentStatusOptions {
  /** Poll interval while non-terminal. Default 2s. */
  intervalMs?: number;
  /** Max consecutive network errors before giving up. Default 5. */
  maxErrors?: number;
  /** Stop polling automatically once status is `ready` or `error`. Default true. */
  stopOnTerminal?: boolean;
}

export interface UseDocumentStatusResult {
  status: DocumentStatus | null;
  error: Error | null;
  isLoading: boolean;
  isTerminal: boolean;
  refetch: () => Promise<void>;
}

/**
 * React hook that polls the backend status endpoint until the document
 * reaches a terminal state. Cleans up its own interval + abort controller.
 */
export function useDocumentStatus(
  docId: string | null,
  options: UseDocumentStatusOptions = {},
): UseDocumentStatusResult {
  const { intervalMs = 2_000, maxErrors = 5, stopOnTerminal = true } = options;

  const [status, setStatus] = useState<DocumentStatus | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(docId != null);

  const errorStreakRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cancelledRef = useRef(false);

  async function pull(): Promise<void> {
    if (docId == null) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const next = await fetchDocumentStatus(docId, { signal: ctrl.signal });
      if (cancelledRef.current) return;
      setStatus(next);
      setError(null);
      errorStreakRef.current = 0;
      setIsLoading(false);
      if (stopOnTerminal && next.is_terminal && timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } catch (err) {
      if (cancelledRef.current) return;
      // 404 is terminal — the document was deleted, do not keep polling.
      if (err instanceof ApiError && err.status === 404 && timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      errorStreakRef.current += 1;
      setError(err instanceof Error ? err : new Error(String(err)));
      setIsLoading(false);
      if (errorStreakRef.current >= maxErrors && timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }

  useEffect(() => {
    cancelledRef.current = false;
    if (docId == null) {
      setStatus(null);
      setError(null);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    void pull();
    timerRef.current = setInterval(() => {
      void pull();
    }, intervalMs);
    return () => {
      cancelledRef.current = true;
      abortRef.current?.abort();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId, intervalMs, maxErrors, stopOnTerminal]);

  const isTerminal = status?.is_terminal ?? false;

  return {
    status,
    error,
    isLoading,
    isTerminal,
    refetch: pull,
  };
}

/** Human label for a processing status. Kept with the client so the chat
 *  UI (T-011) and upload UI can render consistent progress copy. */
export function statusLabel(status: ProcessingStatus): string {
  switch (status) {
    case "pending":
      return "Queued";
    case "parsing":
      return "Reading PDF";
    case "chunking":
      return "Splitting text";
    case "embedding":
      return "Embedding chunks";
    case "ready":
      return "Ready";
    case "error":
      return "Failed";
  }
}
