/**
 * DocumentList — sidebar listing uploaded PDFs.
 *
 * Polls `/api/documents` on an interval while any document is still
 * processing so the status badge flips without a reload. The polling
 * stops once every row is `ready` or `failed`.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UnauthorizedError } from "@/lib/api";
import { deleteDocument, listDocuments } from "@/lib/documents";
import type { DocumentSummary } from "@/lib/types";

interface Props {
  activeId: string | null;
  onSelect: (doc: DocumentSummary) => void;
  /** Trigger a refetch; parent increments after a successful upload. */
  refreshToken?: number;
}

const POLL_MS = 3000;

function statusColor(status: DocumentSummary["status"]): string {
  switch (status) {
    case "ready":
      return "text-tui-success";
    case "failed":
      return "text-tui-danger";
    case "processing":
      return "text-tui-warn";
    default:
      return "text-tui-textMuted";
  }
}

function fmtSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function DocumentList({ activeId, onSelect, refreshToken = 0 }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const next = await listDocuments(signal);
      setDocs(next);
      setError(null);
    } catch (e) {
      if ((e as DOMException)?.name === "AbortError") return;
      // 401: lib/session.redirectToLogin() already navigated to /login.
      if (e instanceof UnauthorizedError) return;
      setError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    void refresh(ctrl.signal);
    return () => ctrl.abort();
  }, [refresh, refreshToken]);

  // Poll while anything is still processing.
  useEffect(() => {
    const anyProcessing = docs.some((d) => d.status === "processing");
    if (!anyProcessing) return;
    timerRef.current = setTimeout(() => void refresh(), POLL_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [docs, refresh]);

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    if (!confirm("delete this document?")) return;
    try {
      await deleteDocument(id);
      setDocs((d) => d.filter((x) => x.id !== id));
    } catch (err) {
      if (err instanceof UnauthorizedError) return;
      setError(err instanceof Error ? err.message : "delete failed");
    }
  }

  return (
    <aside
      className="flex flex-col border border-tui-border rounded-tui bg-tui-surface h-full min-h-0"
      aria-label="Documents"
    >
      <header className="px-3 py-2 border-b border-tui-border flex items-center justify-between">
        <span className="text-tui-textMuted text-xs uppercase tracking-wider">
          ── documents ──
        </span>
        <button
          type="button"
          onClick={() => void refresh()}
          className="text-xs text-tui-accent hover:underline"
          aria-label="Refresh document list"
        >
          ↻
        </button>
      </header>

      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="p-3 text-xs text-tui-textMuted">loading…</div>
        )}
        {!loading && error && (
          <div className="p-3 text-xs text-tui-danger" role="alert">
            ! {error}
          </div>
        )}
        {!loading && !error && docs.length === 0 && (
          <div className="p-3 text-xs text-tui-textMuted">
            no documents yet. upload a pdf to begin.
          </div>
        )}
        <ul className="divide-y divide-tui-border">
          {docs.map((d) => {
            const isActive = d.id === activeId;
            const disabled = d.status === "failed";
            return (
              <li
                key={d.id}
                className={`flex items-start gap-2 px-3 py-2 hover:bg-tui-elevated transition-colors ${
                  isActive ? "bg-tui-elevated" : ""
                }`}
              >
                <button
                  type="button"
                  onClick={() => !disabled && onSelect(d)}
                  className={`flex-1 text-left text-xs min-w-0 ${
                    disabled ? "opacity-60 cursor-not-allowed" : ""
                  }`}
                  disabled={disabled}
                  aria-current={isActive ? "true" : undefined}
                >
                  <div className="text-tui-text truncate font-medium">
                    {d.name}
                  </div>
                  <div className="mt-0.5 flex gap-2 text-tui-textMuted">
                    <span className={statusColor(d.status)}>
                      {d.status ?? "unknown"}
                    </span>
                    {d.pages ? <span>· {d.pages}p</span> : null}
                    {d.size_bytes ? <span>· {fmtSize(d.size_bytes)}</span> : null}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => void handleDelete(e, d.id)}
                  className="text-tui-textMuted hover:text-tui-danger text-xs shrink-0 px-1"
                  aria-label={`Delete ${d.name}`}
                >
                  ✕
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
