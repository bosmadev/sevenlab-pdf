/**
 * Home dashboard — upload PDFs, view indexing status, open chat.
 *
 * `/chat` holds the full RAG interface; this screen is the operator's
 * landing page: it shows what has been uploaded and lets a new PDF join
 * the index without jumping into the chat view.
 */
"use client";

import Link from "next/link";
import { useState } from "react";
import { TopBar } from "@/components/common/TopBar";
import { DocumentList } from "@/components/documents/DocumentList";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import type { DocumentSummary } from "@/lib/types";

export default function Home() {
  const [refreshToken, setRefreshToken] = useState(0);
  const [selected, setSelected] = useState<DocumentSummary | null>(null);

  return (
    <main className="min-h-screen bg-tui-base text-tui-text flex flex-col">
      <TopBar />

      <div className="flex-1 grid gap-4 p-4 md:grid-cols-[18rem_1fr] max-w-6xl w-full mx-auto">
        <div className="min-h-[60vh]">
          <DocumentList
            activeId={selected?.id ?? null}
            onSelect={setSelected}
            refreshToken={refreshToken}
          />
        </div>

        <section className="flex flex-col gap-4 min-w-0">
          <UploadDropzone
            maxSizeMb={50}
            onUploaded={() => setRefreshToken((t) => t + 1)}
          />

          <section className="border border-tui-border bg-tui-surface rounded-tui p-4">
            <div className="text-tui-textMuted text-xs uppercase tracking-wider mb-2">
              ── selected ──
            </div>
            {selected ? (
              <div className="space-y-2 text-sm">
                <div className="text-tui-textBright font-medium">
                  {selected.name}
                </div>
                <dl className="grid grid-cols-[8rem_1fr] text-xs gap-y-1">
                  <dt className="text-tui-textMuted">id</dt>
                  <dd className="font-mono text-tui-text truncate">
                    {selected.id}
                  </dd>
                  <dt className="text-tui-textMuted">status</dt>
                  <dd className="text-tui-text">{selected.status ?? "unknown"}</dd>
                  {selected.pages != null && (
                    <>
                      <dt className="text-tui-textMuted">pages</dt>
                      <dd className="text-tui-text">{selected.pages}</dd>
                    </>
                  )}
                  {selected.uploaded_at && (
                    <>
                      <dt className="text-tui-textMuted">uploaded</dt>
                      <dd className="text-tui-text">
                        {new Date(selected.uploaded_at).toLocaleString()}
                      </dd>
                    </>
                  )}
                </dl>
                <div className="pt-2">
                  <Link
                    href={`/chat?doc=${encodeURIComponent(selected.id)}`}
                    className="inline-block text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-accent hover:border-tui-borderActive"
                  >
                    open in chat →
                  </Link>
                </div>
              </div>
            ) : (
              <p className="text-tui-textMuted text-xs">
                pick a document on the left to inspect it, or upload a new one.
              </p>
            )}
          </section>

          <section className="border border-tui-border bg-tui-surface rounded-tui p-4">
            <div className="text-tui-textMuted text-xs uppercase tracking-wider mb-2">
              ── chat ──
            </div>
            <p className="text-tui-textMuted text-xs mb-3">
              hybrid search across every indexed PDF (vector + BM25 + rerank).
            </p>
            <Link
              href="/chat"
              className="inline-block text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-accent hover:border-tui-borderActive"
            >
              open chat →
            </Link>
          </section>
        </section>
      </div>
    </main>
  );
}
