import { create } from "zustand";
import type { ChatMessage, ChatStatus, Citation, DocumentSummary } from "./types";

interface ChatState {
  messages: ChatMessage[];
  status: ChatStatus;
  activeDocId: string | null;
  documents: DocumentSummary[];
  error: string | null;

  setActiveDoc: (id: string | null) => void;
  setDocuments: (docs: DocumentSummary[]) => void;

  appendMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendDelta: (id: string, delta: string) => void;
  attachCitations: (id: string, citations: Citation[]) => void;

  setStatus: (s: ChatStatus) => void;
  setError: (e: string | null) => void;
  clearChat: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  status: "idle",
  activeDocId: null,
  documents: [],
  error: null,

  setActiveDoc: (id) => set({ activeDocId: id }),
  setDocuments: (docs) => set({ documents: docs }),

  appendMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  updateMessage: (id, patch) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    })),

  appendDelta: (id, delta) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + delta } : m,
      ),
    })),

  attachCitations: (id, citations) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, citations } : m)),
    })),

  setStatus: (status) => set({ status }),
  setError: (error) => set({ error }),
  clearChat: () => set({ messages: [], error: null, status: "idle" }),
}));
