"use client";
import { citationName, citationPreview, type Citation } from "@/lib/types";

interface Props {
  citations: Citation[];
  msgId: string;
}

export function CitationList({ citations, msgId }: Props) {
  if (!citations || citations.length === 0) return null;
  return (
    <div className="mt-3 border-t border-tui-border pt-2">
      <div className="text-tui-textMuted text-xs uppercase tracking-wider mb-1.5">
        ── sources ──
      </div>
      <ol className="space-y-1 text-xs">
        {citations.map((c, i) => {
          const n = typeof c.n === "number" ? c.n : i + 1;
          const name = citationName(c);
          const preview = citationPreview(c);
          return (
            <li
              key={`${c.doc_id}-${c.page}-${n}-${i}`}
              id={`cite-${msgId}-${n}`}
              className="flex gap-2 rounded-tui transition-shadow"
            >
              <span className="text-tui-accent shrink-0">[{n}]</span>
              <span className="text-tui-text">
                <span className="text-tui-textBright">{name}</span>
                <span className="text-tui-textMuted"> · p.{c.page}</span>
                {typeof c.score === "number" && (
                  <span className="text-tui-textMuted"> · {c.score.toFixed(3)}</span>
                )}
                {preview && (
                  <div className="text-tui-textMuted italic mt-0.5 line-clamp-2">
                    {preview}
                  </div>
                )}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
