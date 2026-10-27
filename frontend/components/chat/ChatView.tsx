"use client";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useChatStore } from "@/lib/chatStore";
import { streamChat } from "@/lib/sse";
import { download, toMarkdown, toPlainText } from "@/lib/export";
import type { ChatMessage, DocumentSummary } from "@/lib/types";
import { ChatInput } from "./ChatInput";
import { DocumentPicker } from "./DocumentPicker";
import { MessageBubble } from "./MessageBubble";

const uid = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);

async function loadDocuments(): Promise<DocumentSummary[]> {
  try {
    const r = await fetch("/api/documents/", { credentials: "same-origin" });
    if (!r.ok) return [];
    const j = (await r.json()) as { documents?: DocumentSummary[] };
    return j.documents ?? [];
  } catch {
    return [];
  }
}

export function ChatView() {
  const messages = useChatStore((s) => s.messages);
  const status = useChatStore((s) => s.status);
  const documents = useChatStore((s) => s.documents);
  const activeDocId = useChatStore((s) => s.activeDocId);
  const error = useChatStore((s) => s.error);

  const appendMessage = useChatStore((s) => s.appendMessage);
  const appendDelta = useChatStore((s) => s.appendDelta);
  const updateMessage = useChatStore((s) => s.updateMessage);
  const attachCitations = useChatStore((s) => s.attachCitations);
  const setStatus = useChatStore((s) => s.setStatus);
  const setError = useChatStore((s) => s.setError);
  const setDocuments = useChatStore((s) => s.setDocuments);
  const setActiveDoc = useChatStore((s) => s.setActiveDoc);
  const clearChat = useChatStore((s) => s.clearChat);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void loadDocuments().then(setDocuments);
  }, [setDocuments]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  const streaming = status === "streaming";

  const send = useCallback(
    async (text: string) => {
      if (streaming) return;
      setError(null);

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };
      appendMessage(userMsg);

      const asstId = uid();
      appendMessage({
        id: asstId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        streaming: true,
      });

      setStatus("streaming");
      const ac = new AbortController();
      abortRef.current = ac;

      try {
        // T-012 (rag-impl-1) body contract: {question, doc_id?} only — no history field.
        const body: { question: string; doc_id?: string } = { question: text };
        if (activeDocId) body.doc_id = activeDocId;
        await streamChat(body, {
          signal: ac.signal,
          onEvent: (ev) => {
            if (ev.type === "token" || ev.type === "delta") {
              appendDelta(asstId, ev.text);
            } else if (ev.type === "citations") {
              attachCitations(asstId, ev.citations);
            } else if (ev.type === "done") {
              updateMessage(asstId, { streaming: false });
              setStatus("idle");
            } else if (ev.type === "error") {
              updateMessage(asstId, { streaming: false, error: ev.message });
              setStatus("error");
              setError(ev.message);
            }
          },
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "stream failed";
        if ((e as { name?: string })?.name === "AbortError") {
          updateMessage(asstId, { streaming: false, content: "" , error: "cancelled" });
          setStatus("idle");
        } else {
          updateMessage(asstId, { streaming: false, error: msg });
          setStatus("error");
          setError(msg);
        }
      } finally {
        if (abortRef.current === ac) abortRef.current = null;
      }
    },
    [activeDocId, appendDelta, appendMessage, attachCitations, setError, setStatus, streaming, updateMessage],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const canExport = useMemo(() => messages.some((m) => !m.streaming), [messages]);

  const exportAs = (fmt: "md" | "txt") => {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    if (fmt === "md") {
      download(`sevenlab-chat-${ts}.md`, toMarkdown(messages), "text/markdown");
    } else {
      download(`sevenlab-chat-${ts}.txt`, toPlainText(messages), "text/plain");
    }
  };

  const copyAll = async () => {
    try {
      await navigator.clipboard.writeText(toMarkdown(messages));
    } catch {
      // ignore
    }
  };

  return (
    <div className="h-full flex gap-3 p-3">
      <DocumentPicker
        documents={documents}
        activeId={activeDocId}
        onSelect={setActiveDoc}
      />

      <section className="flex-1 flex flex-col min-w-0 gap-3">
        <header className="border border-tui-border bg-tui-surface rounded-tui px-3 py-2 flex items-center justify-between text-xs">
          <div className="text-tui-textMuted uppercase tracking-wider">
            ── chat ──
          </div>
          <div className="flex gap-2 items-center">
            <button
              type="button"
              onClick={copyAll}
              disabled={!canExport}
              className="px-2 py-1 border border-tui-border rounded-tui text-tui-text hover:border-tui-borderActive disabled:opacity-40"
              aria-label="copy chat as markdown"
            >
              copy
            </button>
            <button
              type="button"
              onClick={() => exportAs("md")}
              disabled={!canExport}
              className="px-2 py-1 border border-tui-border rounded-tui text-tui-text hover:border-tui-borderActive disabled:opacity-40"
            >
              .md
            </button>
            <button
              type="button"
              onClick={() => exportAs("txt")}
              disabled={!canExport}
              className="px-2 py-1 border border-tui-border rounded-tui text-tui-text hover:border-tui-borderActive disabled:opacity-40"
            >
              .txt
            </button>
            <button
              type="button"
              onClick={clearChat}
              disabled={streaming || messages.length === 0}
              className="px-2 py-1 border border-tui-border rounded-tui text-tui-text hover:border-tui-danger disabled:opacity-40"
            >
              clear
            </button>
          </div>
        </header>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto space-y-3 pr-1"
          aria-live="polite"
        >
          {messages.length === 0 ? (
            <div className="text-tui-textMuted text-xs text-center py-16">
              <pre className="inline-block text-left leading-tight">
{`+------------------------------+
|   sevenlab-pdf · hybrid rag  |
|   cohere v5-pro · mongo atlas|
+------------------------------+`}
              </pre>
              <p className="mt-4">upload a PDF, then ask a question.</p>
            </div>
          ) : (
            messages.map((m) => <MessageBubble key={m.id} message={m} />)
          )}
        </div>

        {error && (
          <div
            role="alert"
            className="text-xs text-tui-danger border border-tui-danger rounded-tui px-2 py-1"
          >
            ! {error}
          </div>
        )}

        <ChatInput
          disabled={streaming}
          streaming={streaming}
          onSubmit={send}
          onCancel={cancel}
        />
      </section>
    </div>
  );
}
