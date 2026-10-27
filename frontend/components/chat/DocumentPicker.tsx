"use client";
import type { DocumentSummary } from "@/lib/types";

interface Props {
  documents: DocumentSummary[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
  /** Lock selection while an answer is streaming so deltas land in the right thread. */
  disabled?: boolean;
}

const statusColor = (s: DocumentSummary["status"]): string =>
  s === "ready"
    ? "text-tui-success"
    : s === "failed"
      ? "text-tui-danger"
      : s === "processing"
        ? "text-tui-warn"
        : "text-tui-textMuted";

// Only an indexed PDF has chunks to retrieve from; anything else is shown but not selectable.
const selectable = (d: DocumentSummary): boolean => d.status === "ready" || d.status === undefined;

export function DocumentPicker({ documents, activeId, onSelect, disabled }: Props) {
  const rowClass = (active: boolean, enabled: boolean): string =>
    `w-full text-left text-xs px-2 py-1 rounded-tui border transition-colors ${
      active
        ? "bg-tui-elevated text-tui-textBright border-tui-borderActive"
        : enabled
          ? "text-tui-text border-transparent hover:bg-tui-elevated hover:border-tui-borderHover"
          : "text-tui-textMuted border-transparent opacity-60 cursor-not-allowed"
    }`;

  return (
    <aside
      className="border border-tui-border bg-tui-surface rounded-tui p-2 w-64 shrink-0 h-full overflow-y-auto"
      aria-label="document scope"
    >
      <header className="flex items-center justify-between text-xs text-tui-textMuted uppercase tracking-wider mb-2 px-1">
        <span>── docs ──</span>
        <span>{documents.length}</span>
      </header>

      <button
        type="button"
        onClick={() => onSelect(null)}
        disabled={disabled}
        className={`${rowClass(activeId === null, !disabled)} mb-1`}
        aria-pressed={activeId === null}
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
            const enabled = !disabled && selectable(d);
            return (
              <li key={d.id}>
                <button
                  type="button"
                  onClick={() => enabled && onSelect(d.id)}
                  disabled={!enabled}
                  className={`${rowClass(active, enabled)} truncate`}
                  title={d.status && d.status !== "ready" ? `${d.name} (${d.status})` : d.name}
                  aria-pressed={active}
                  data-status={d.status ?? "unknown"}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate">{d.name}</span>
                    <span className={`${statusColor(d.status)} shrink-0`}>●</span>
                  </div>
                  <div className="text-tui-textMuted flex gap-2">
                    <span className={statusColor(d.status)}>{d.status ?? "unknown"}</span>
                    {typeof d.pages === "number" && <span>· {d.pages}p</span>}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
