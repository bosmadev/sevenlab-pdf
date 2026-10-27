/**
 * Root-layout error boundary (Next.js App Router convention).
 *
 * Replaces app/layout.tsx when the layout itself throws, so it must render
 * <html> and <body> and cannot rely on globals.css or Tailwind being loaded.
 * Colours mirror the :root tokens in app/globals.css.
 */
"use client";

import type { CSSProperties } from "react";

interface Props {
  error: Error & { digest?: string };
  reset: () => void;
}

const button: CSSProperties = {
  fontSize: 11,
  padding: "6px 12px",
  border: "1px solid #2a3040",
  borderRadius: 6,
  background: "transparent",
  color: "#94a3b8",
  cursor: "pointer",
  textDecoration: "none",
  fontFamily: "inherit",
};

export default function GlobalError({ error, reset }: Props) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 16,
          background: "#0b0e11",
          color: "#c9d1d9",
          fontFamily:
            '"JetBrains Mono", "Fira Code", "Cascadia Code", "SF Mono", ui-monospace, monospace',
          fontSize: 14,
          lineHeight: 1.6,
        }}
      >
        <section
          role="alert"
          style={{
            width: "100%",
            maxWidth: 520,
            border: "1px solid #f87171",
            borderRadius: 6,
            padding: 16,
            background: "#12161c",
          }}
        >
          <div
            style={{
              color: "#929faf",
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 8,
            }}
          >
            ── fatal ──
          </div>
          <p style={{ color: "#f87171", margin: 0 }}>! the app shell failed to render.</p>
          <p style={{ color: "#929faf", fontSize: 12, marginTop: 8, wordBreak: "break-word" }}>
            {error.message || "unknown error"}
          </p>
          {error.digest && (
            <p style={{ color: "#929faf", fontSize: 12, marginTop: 4 }}>ref: {error.digest}</p>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button type="button" onClick={reset} style={button}>
              retry
            </button>
            <a href="/" style={button}>
              reload →
            </a>
          </div>
        </section>
      </body>
    </html>
  );
}
