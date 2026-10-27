"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { UnauthorizedError } from "@/lib/api";
import { selectMessages, threadKey, useChatStore } from "@/lib/chatStore";
import { listDocuments } from "@/lib/documents";
import { download, toMarkdown, toPlainText } from "@/lib/export";
import { loadHistory } from "@/lib/history";
import { streamChat } from "@/lib/sse";
import type { ChatMessage } from "@/lib/types";
import { ChatInput } from "./ChatInput";
import { DocumentPicker } from "./DocumentPicker";
import { MessageBubble } from "./MessageBubble";

const POLL_MS = 3000;
const LOGIN_PATH = `/login?from=${encodeURIComponent("/chat")}`;

const uid = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);

export function ChatView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const docParam = searchParams.get("doc");

  const messages = useChatStore(selectMessages);
  const status = useChatStore((s) => s.status);
  const documents = useChatStore((s) => s.documents);
  const activeDocId = useChatStore((s) => s.activeDocId);
  const error = useChatStore((s) => s.error);
  const hydrated = useChatStore((s) => s.hydrated);

  const appendMessage = useChatStore((s) => s.appendMessage);
  const appendDelta = useChatStore((s) => s.appendDelta);
  const updateMessage = useChatStore((s) => s.updateMessage);
  const attachCitations = useChatStore((s) => s.attachCitations);
  const setThread = useChatStore((s) => s.setThread);
  const setStatus = useChatStore((s) => s.setStatus);
  const setError = useChatStore((s) => s.setError);
  const setDocuments = useChatStore((s) => s.setDocuments);
  const setActiveDoc = useChatStore((s) => s.setActiveDoc);
  const clearChat = useChatStore((s) => s.clearChat);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const streaming = status === "streaming";

  const activeDoc = useMemo(
    () => documents.find((d) => d.id === activeDocId) ?? null,
    [documents, activeDocId],
  );
  const activeDocName = activeDoc?.name;

  // Restore the persisted threads once on mount, then honour ?doc= from the home page.
  useEffect(() => {
    void (async () => {
      try {
        await useChatStore.persist?.rehydrate?.();
      } catch {
        // sessionStorage unavailable (private mode); start empty
      }
      useChatStore.getState().setHydrated(true);
    })();
  }, []);

  useEffect(() => {
    if (hydrated && docParam) setActiveDoc(docParam);
  }, [hydrated, docParam, setActiveDoc]);

  const refreshDocs = useCallback(
    async (signal?: AbortSignal) => {
      try {
        setDocuments(await listDocuments(signal));
      } catch (e) {
        if (e instanceof UnauthorizedError) router.replace(LOGIN_PATH);
      }
    },
    [router, setDocuments],
  );

  useEffect(() => {
    const ctrl = new AbortController();
    void refreshDocs(ctrl.signal);
    return () => ctrl.abort();
  }, [refreshDocs]);

  // Keep polling while any PDF is still being indexed so the picker flips to ready.
  useEffect(() => {
    if (!documents.some((d) => d.status === "processing")) return;
    const t = setTimeout(() => void refreshDocs(), POLL_MS);
    return () => clearTimeout(t);
  }, [documents, refreshDocs]);

  // Seed an empty per-document thread from the server-side history.
  useEffect(() => {
    if (!hydrated || !activeDocId) return;
    if ((useChatStore.getState().threads[activeDocId] ?? []).length > 0) return;
    const ctrl = new AbortController();
    void loadHistory(activeDocId, activeDocName, ctrl.signal)
      .then((hist) => {
        if (ctrl.signal.aborted || hist.length === 0) return;
        const current = useChatStore.getState().threads[activeDocId] ?? [];
        if (current.length === 0) setThread(activeDocId, hist);
      })
      .catch((e) => {
        if (e instanceof UnauthorizedError) router.replace(LOGIN_PATH);
      });
    return () => ctrl.abort();
  }, [hydrated, activeDocId, activeDocName, router, setThread]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  const send = useCallback(
    async (text: string) => {
      if (streaming) return;
      setError(null);

      const docId = activeDocId;
      const key = threadKey(docId);

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };
      appendMessage(key, userMsg);

      const asstId = uid();
      appendMessage(key, {
        id: asstId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        streaming: true,
      });

      setStatus("streaming");
      const ac = new AbortController();
      abortRef.current = ac;
      let unauthorized = false;

      const assistantNow = (): ChatMessage | undefined =>
        useChatStore.getState().threads[key]?.find((m) => m.id === asstId);

      try {
        // T-012 (rag-impl-1) body contract: {question, doc_id?} only — no history field.
        const body: { question: string; doc_id?: string } = { question: text };
        if (docId) body.doc_id = docId;

        const end = await streamChat(body, {
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
              if (ev.status === 401) unauthorized = true;
              updateMessage(asstId, { streaming: false, error: ev.message });
              setStatus("error");
              setError(ev.message);
            }
          },
        });

        if (end === "eof") {
          const partial = assistantNow()?.content.trim() ?? "";
          if (partial) {
            updateMessage(asstId, { streaming: false, cancelled: true });
            setStatus("idle");
          } else {
            const msg = "connection closed before an answer arrived";
            updateMessage(asstId, { streaming: false, error: msg });
            setStatus("error");
            setError(msg);
          }
        }
        // On `done` the backend has already stored this exchange
        // (pipeline.answer_stream persists before the done frame).
      } catch (e) {
        if ((e as { name?: string })?.name === "AbortError") {
          updateMessage(asstId, { streaming: false, cancelled: true });
          setStatus("idle");
        } else {
          const msg = e instanceof Error ? e.message : "stream failed";
          updateMessage(asstId, { streaming: false, error: msg });
          setStatus("error");
          setError(msg);
        }
      } finally {
        if (abortRef.current === ac) abortRef.current = null;
        if (unauthorized) router.replace(LOGIN_PATH);
      }
    },
    [
      activeDocId,
      appendDelta,
      appendMessage,
      attachCitations,
      router,
      setError,
      setStatus,
      streaming,
      updateMessage,
    ],
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
      // clipboard blocked; nothing to surface
    }
  };

  const readyCount = documents.filter((d) => d.status === "ready").length;
  const scopeLabel = activeDocId ? (activeDocName ?? activeDocId) : "all documents";

  return (
    <div className="h-full flex gap-3 p-3 min-h-0">
      <DocumentPicker
        documents={documents}
        activeId={activeDocId}
        onSelect={setActiveDoc}
        disabled={streaming}
      />

      <section className="flex-1 flex flex-col min-w-0 gap-3">
        <header className="border border-tui-border bg-tui-surface rounded-tui px-3 py-2 flex items-center justify-between gap-3 text-xs">
          <div className="text-tui-textMuted uppercase tracking-wider truncate min-w-0">
            ── chat ──{" "}
            <span className="normal-case tracking-normal text-tui-textBright" data-testid="chat-scope">
              {scopeLabel}
            </span>
          </div>
          <div className="flex gap-2 items-center shrink-0">
            <button type="button" onClick={copyAll} disabled={!canExport} className="tui-btn" aria-label="copy chat as markdown">
              copy
            </button>
            <button type="button" onClick={() => exportAs("md")} disabled={!canExport} className="tui-btn" aria-label="export chat as markdown file">
              .md
            </button>
            <button type="button" onClick={() => exportAs("txt")} disabled={!canExport} className="tui-btn" aria-label="export chat as text file">
              .txt
            </button>
            <button
              type="button"
              onClick={clearChat}
              disabled={streaming || messages.length === 0}
              className="tui-btn tui-btn-danger"
              title="clear this thread on this device"
            >
              clear
            </button>
          </div>
        </header>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto space-y-3 pr-1 min-h-0"
          aria-live="polite"
          data-testid="chat-log"
        >
          {!hydrated ? (
            <div className="text-tui-textMuted text-xs text-center py-16">restoring session…</div>
          ) : messages.length === 0 ? (
            <div className="text-tui-textMuted text-xs text-center py-16">
              <pre className="inline-block text-left leading-tight">
{`+------------------------------+
|   sevenlab-pdf · hybrid rag  |
|   cohere v5-pro · mongo atlas|
+------------------------------+`}
              </pre>
              <p className="mt-4">
                {activeDocId
                  ? `ask a question about ${scopeLabel}.`
                  : "upload a PDF, then ask a question."}
              </p>
            </div>
          ) : (
            messages.map((m) => <MessageBubble key={m.id} message={m} />)
          )}
        </div>

        {error && (
          <div role="alert" className="text-xs text-tui-danger border border-tui-danger rounded-tui px-2 py-1">
            ! {error}
          </div>
        )}

        <ChatInput disabled={streaming} streaming={streaming} onSubmit={send} onCancel={cancel} />

        <footer
          className="text-[0.6875rem] text-tui-textMuted flex items-center justify-between px-1"
          data-testid="chat-status"
        >
          <span>
            [{status}] {readyCount}/{documents.length} docs ready · scope: {scopeLabel}
          </span>
          <span>enter send · shift+enter newline · esc stop</span>
        </footer>
      </section>
    </div>
  );
}
