"use client";
import type { DocumentSummary } from "@/lib/types";

interface Props {
  documents: DocumentSummary[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
}

export function DocumentPicker({ documents, activeId, onSelect }: Props) {
  return (
    <aside className="border border-tui-border bg-tui-surface rounded-tui p-2 w-64 shrink-0 h-full overflow-y-auto">
      <header className="flex items-center justify-between text-xs text-tui-textMuted uppercase tracking-wider mb-2 px-1">
        <span>── docs ──</span>
        <span>{documents.length}</span>
      </header>

      <button
        type="button"
        onClick={() => onSelect(null)}
        className={`w-full text-left text-xs px-2 py-1 rounded-tui mb-1 ${
          activeId === null
            ? "bg-tui-elevated text-tui-textBright border border-tui-borderActive"
            : "text-tui-text hover:bg-tui-elevated border border-transparent"
        }`}
      >
        * all documents
      </button>

      {documents.length === 0 ? (
        <p className="text-tui-textMuted text-xs px-2 py-4 text-center">
          no PDFs yet. upload one to begin.
        </p>
      ) : (
        <ul className="space-y-1">
          {documents.map((d) => {
            const active = d.id === activeId;
            const statusColor =
              d.status === "ready"
                ? "text-tui-success"
                : d.status === "failed"
                  ? "text-tui-danger"
                  : "text-tui-warn";
            return (
              <li key={d.id}>
                <button
                  type="button"
                  onClick={() => onSelect(d.id)}
                  className={`w-full text-left text-xs px-2 py-1 rounded-tui truncate ${
                    active
                      ? "bg-tui-elevated text-tui-textBright border border-tui-borderActive"
                      : "text-tui-text hover:bg-tui-elevated border border-transparent"
                  }`}
                  title={d.name}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate">{d.name}</span>
                    <span className={`${statusColor} shrink-0`}>●</span>
                  </div>
                  {typeof d.pages === "number" && (
                    <div className="text-tui-textMuted">{d.pages}p</div>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
