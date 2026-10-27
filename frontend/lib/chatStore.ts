/**
 * Chat state: one message thread per document (plus one for "all documents"),
 * persisted to sessionStorage so a reload keeps the conversation.
 *
 * Hydration is deferred (`skipHydration`) and triggered by ChatView in an
 * effect, so the server render and the first client render agree.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { ChatMessage, ChatStatus, Citation, DocumentSummary } from "./types";

export const ALL_DOCS_KEY = "*";
export const threadKey = (docId: string | null | undefined): string => docId ?? ALL_DOCS_KEY;

const EMPTY: ChatMessage[] = [];

type Threads = Record<string, ChatMessage[]>;

function patchById(
  threads: Threads,
  id: string,
  fn: (m: ChatMessage) => ChatMessage,
): Threads {
  for (const key of Object.keys(threads)) {
    const list = threads[key];
    const idx = list.findIndex((m) => m.id === id);
    if (idx === -1) continue;
    const next = list.slice();
    next[idx] = fn(list[idx]);
    return { ...threads, [key]: next };
  }
  return threads;
}

interface ChatState {
  threads: Threads;
  status: ChatStatus;
  activeDocId: string | null;
  documents: DocumentSummary[];
  error: string | null;
  hydrated: boolean;

  setActiveDoc: (id: string | null) => void;
  setDocuments: (docs: DocumentSummary[]) => void;
  setHydrated: (v: boolean) => void;

  setThread: (key: string, messages: ChatMessage[]) => void;
  appendMessage: (key: string, msg: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendDelta: (id: string, delta: string) => void;
  attachCitations: (id: string, citations: Citation[]) => void;

  setStatus: (s: ChatStatus) => void;
  setError: (e: string | null) => void;
  clearChat: () => void;
}

type Persisted = Pick<ChatState, "threads" | "activeDocId">;

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      threads: {},
      status: "idle",
      activeDocId: null,
      documents: [],
      error: null,
      hydrated: false,

      setActiveDoc: (id) => set({ activeDocId: id, error: null }),
      setDocuments: (docs) => set({ documents: docs }),
      setHydrated: (v) => set({ hydrated: v }),

      setThread: (key, messages) =>
        set((s) => ({ threads: { ...s.threads, [key]: messages } })),

      appendMessage: (key, msg) =>
        set((s) => ({
          threads: { ...s.threads, [key]: [...(s.threads[key] ?? []), msg] },
        })),

      updateMessage: (id, patch) =>
        set((s) => ({ threads: patchById(s.threads, id, (m) => ({ ...m, ...patch })) })),

      appendDelta: (id, delta) =>
        set((s) => ({
          threads: patchById(s.threads, id, (m) => ({ ...m, content: m.content + delta })),
        })),

      attachCitations: (id, citations) =>
        set((s) => ({ threads: patchById(s.threads, id, (m) => ({ ...m, citations })) })),

      setStatus: (status) => set({ status }),
      setError: (error) => set({ error }),

      clearChat: () =>
        set((s) => {
          const threads = { ...s.threads };
          delete threads[threadKey(s.activeDocId)];
          return { threads, error: null, status: "idle" };
        }),
    }),
    {
      name: "sevenlab-chat",
      storage: createJSONStorage(() => sessionStorage),
      skipHydration: true,
      partialize: (s): Persisted => ({ threads: s.threads, activeDocId: s.activeDocId }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<Persisted>;
        const threads: Threads = {};
        for (const [key, list] of Object.entries(p.threads ?? {})) {
          threads[key] = (list ?? [])
            .filter((m) => m && typeof m.id === "string")
            // A reload mid-stream leaves a half answer; mark it so it is not mistaken for complete.
            .map((m) => (m.streaming ? { ...m, streaming: false, cancelled: true } : m));
        }
        return { ...current, threads, activeDocId: p.activeDocId ?? null };
      },
    },
  ),
);

export const selectMessages = (s: ChatState): ChatMessage[] =>
  s.threads[threadKey(s.activeDocId)] ?? EMPTY;
