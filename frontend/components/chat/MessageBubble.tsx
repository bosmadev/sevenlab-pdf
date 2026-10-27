"use client";
import type { JSX } from "react";
import type { ChatMessage, Citation } from "@/lib/types";
import { CitationList } from "./CitationList";

interface Props {
  message: ChatMessage;
}

/**
 * Replace bracketed citation refs like [1], [2, 3] with clickable badges that
 * scroll the matching entry in the per-message sources list into view.
 *
 * The number inside the brackets matches `Citation.n` (set by rag-impl-1), with
 * a fallback to 1-based array position when `n` is absent (older fixtures).
 */
function citationLookup(citations: Citation[]): Set<number> {
  const s = new Set<number>();
  citations.forEach((c, i) => {
    s.add(typeof c.n === "number" ? c.n : i + 1);
  });
  return s;
}

function renderWithCitations(
  text: string,
  msgId: string,
  valid: Set<number>,
): JSX.Element[] {
  const parts: JSX.Element[] = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    }
    const refs = m[1].split(/\s*,\s*/).map((n) => Number.parseInt(n, 10));
    for (let i = 0; i < refs.length; i++) {
      const n = refs[i];
      const ok = Number.isFinite(n) && valid.has(n);
      if (i > 0) parts.push(<span key={key++}>, </span>);
      if (i === 0) parts.push(<span key={key++}>[</span>);
      if (ok) {
        parts.push(
          <a
            key={key++}
            href={`#cite-${msgId}-${n}`}
            className="text-tui-accent hover:text-tui-textBright underline-offset-2"
            onClick={(e) => {
              e.preventDefault();
              const el = document.getElementById(`cite-${msgId}-${n}`);
              if (el) {
                el.scrollIntoView({ behavior: "smooth", block: "nearest" });
                el.classList.add("ring-1", "ring-tui-borderActive");
                setTimeout(() => el.classList.remove("ring-1", "ring-tui-borderActive"), 1200);
              }
            }}
          >
            {n}
          </a>,
        );
      } else {
        parts.push(<span key={key++}>{n}</span>);
      }
      if (i === refs.length - 1) parts.push(<span key={key++}>]</span>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(<span key={key++}>{text.slice(last)}</span>);
  return parts;
}

const prefix = (role: ChatMessage["role"]): string => {
  if (role === "user") return "you@sevenlab:~$";
  if (role === "assistant") return "sevenlab-pdf:~#";
  return "system:~#";
};

const roleColor = (role: ChatMessage["role"]): string => {
  if (role === "user") return "text-tui-info";
  if (role === "assistant") return "text-tui-accent";
  return "text-tui-warn";
};

export function MessageBubble({ message }: Props) {
  const isErr = Boolean(message.error);
  return (
    <article
      className={`border border-tui-border bg-tui-surface rounded-tui p-3 ${
        isErr ? "border-tui-danger" : ""
      }`}
      aria-label={`${message.role} message`}
    >
      <header className={`text-xs mb-1 ${roleColor(message.role)}`}>
        <span className="font-semibold">{prefix(message.role)}</span>
        <span className="text-tui-textMuted ml-2">
          {new Date(message.createdAt).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })}
        </span>
      </header>

      <div
        className={`whitespace-pre-wrap break-words text-sm text-tui-text ${
          message.streaming ? "tui-caret" : ""
        }`}
      >
        {isErr ? (
          <span className="text-tui-danger">! {message.error}</span>
        ) : message.role === "assistant" && message.citations && message.citations.length > 0 ? (
          renderWithCitations(
            message.content,
            message.id,
            citationLookup(message.citations),
          )
        ) : (
          message.content || (message.streaming ? " " : "")
        )}
      </div>

      {message.citations && message.citations.length > 0 && (
        <CitationList citations={message.citations} msgId={message.id} />
      )}
    </article>
  );
}
