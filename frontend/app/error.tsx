/**
 * Route-segment error boundary (Next.js App Router convention).
 *
 * Catches render / effect errors thrown below the root layout so a broken
 * view shows a recoverable panel instead of a blank page. `reset()` re-renders
 * the failed segment; the home link hard-navigates when the failure sticks.
 * Errors thrown by app/layout.tsx itself are caught by global-error.tsx.
 */
"use client";

import Link from "next/link";
import { useEffect } from "react";

interface Props {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function RouteError({ error, reset }: Props) {
  useEffect(() => {
    console.error("[sevenlab-pdf] route error", error);
  }, [error]);

  return (
    <main className="min-h-screen bg-tui-base text-tui-text flex items-center justify-center p-4">
      <section
        role="alert"
        className="w-full max-w-lg border border-tui-danger bg-tui-surface rounded-tui p-4 text-sm"
      >
        <div className="text-tui-textMuted text-xs uppercase tracking-wider mb-2">
          ── error ──
        </div>
        <p className="text-tui-danger">! this page failed to render.</p>
        <p className="text-tui-textMuted text-xs mt-2 break-words">
          {error.message || "unknown error"}
        </p>
        {error.digest && (
          <p className="text-tui-textMuted text-xs mt-1 font-mono">
            ref: {error.digest}
          </p>
        )}
        <div className="flex gap-2 mt-4">
          <button
            type="button"
            onClick={reset}
            className="text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-accent hover:border-tui-borderActive"
          >
            retry
          </button>
          <Link
            href="/"
            className="text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-text hover:border-tui-borderActive"
          >
            home →
          </Link>
        </div>
      </section>
    </main>
  );
}
