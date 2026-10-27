/**
 * UploadDropzone — drag-and-drop PDF uploader.
 *
 * Mirrors backend validation (extension + MIME + size) client-side via
 * `validatePdfFile` so the user gets a fast rejection; the backend
 * `services/pdf_validator.py` is the authoritative check. Streams progress
 * via XHR (fetch cannot expose request-body progress yet).
 *
 * Phases: idle → uploading (bytes in flight) → uploading at 100% (server is
 * validating + sanitizing) → done | error. A 401 routes to /login.
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { UnauthorizedError } from "@/lib/api";
import {
  DEFAULT_MAX_PDF_MB,
  uploadPdf,
  validatePdfFile,
  type UploadResponse,
} from "@/lib/documents";

interface Props {
  /** Maximum file size in MB — must match backend MAX_PDF_SIZE_MB. */
  maxSizeMb?: number;
  /** Called with the backend response on each successful upload. */
  onUploaded?: (doc: UploadResponse) => void;
}

type UploadState =
  | { phase: "idle" }
  | { phase: "uploading"; name: string; ratio: number }
  | { phase: "error"; message: string; name?: string }
  | { phase: "done"; name: string; pages?: number };

export function UploadDropzone({
  maxSizeMb = DEFAULT_MAX_PDF_MB,
  onUploaded,
}: Props) {
  const [state, setState] = useState<UploadState>({ phase: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(
    async (file: File) => {
      const err = validatePdfFile(file, maxSizeMb);
      if (err) {
        setState({ phase: "error", message: err, name: file.name });
        return;
      }

      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      setState({ phase: "uploading", name: file.name, ratio: 0 });
      try {
        const doc = await uploadPdf({
          file,
          signal: ctrl.signal,
          onProgress: (ratio) =>
            setState({ phase: "uploading", name: file.name, ratio }),
        });
        setState({ phase: "done", name: file.name, pages: doc.page_count });
        onUploaded?.(doc);
      } catch (e) {
        if ((e as DOMException)?.name === "AbortError") {
          setState({ phase: "idle" });
          return;
        }
        if (e instanceof UnauthorizedError) {
          // lib/session.redirectToLogin() already navigated to /login.
          setState({ phase: "idle" });
          return;
        }
        setState({
          phase: "error",
          name: file.name,
          message: e instanceof Error ? e.message : "upload failed",
        });
      }
    },
    [maxSizeMb, onUploaded],
  );

  const onFiles = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files || files.length === 0) return;
      // One at a time keeps the UX legible; multi-upload is Phase 2.
      setNotice(
        files.length > 1
          ? `${files.length - 1} extra file(s) ignored, one upload at a time`
          : null,
      );
      await start(files[0]);
    },
    [start],
  );

  const cancel = () => abortRef.current?.abort();

  const busy = state.phase === "uploading";
  const sent = state.phase === "uploading" && state.ratio >= 1;
  const pct = state.phase === "uploading" ? Math.round(state.ratio * 100) : 0;

  return (
    <section
      className={`border rounded-tui p-4 transition-colors ${
        dragging
          ? "border-tui-borderActive bg-tui-elevated"
          : "border-tui-border bg-tui-surface"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (busy) return;
        void onFiles(e.dataTransfer.files);
      }}
      aria-busy={busy}
      aria-label="Upload PDF"
      data-testid="upload-dropzone"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-tui-textBright text-sm font-semibold">
            upload pdf
          </div>
          <div className="text-tui-textMuted text-xs mt-1">
            drag &amp; drop a .pdf file (max {maxSizeMb} MB), or
          </div>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-accent hover:border-tui-borderActive disabled:opacity-50"
        >
          choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          data-testid="upload-input"
          onChange={(e) => {
            const files = e.target.files;
            // Reset so picking the same file again re-fires onChange.
            e.target.value = "";
            void onFiles(files);
          }}
          disabled={busy}
        />
      </div>

      {notice && (
        <div className="mt-2 text-xs text-tui-warn" role="note">
          · {notice}
        </div>
      )}

      {state.phase === "uploading" && (
        <div className="mt-3" data-testid="upload-progress">
          <div className="text-tui-textMuted text-xs mb-1 flex justify-between">
            <span className="truncate pr-2">{state.name}</span>
            <span
              className={`shrink-0 text-tui-accent ${sent ? "animate-pulse" : ""}`}
            >
              {sent ? "validating on server…" : `${pct}%`}
            </span>
          </div>
          <div className="h-1.5 bg-tui-base rounded overflow-hidden">
            <div
              className={`h-full bg-tui-accent transition-all ${
                sent ? "animate-pulse" : ""
              }`}
              style={{ width: `${Math.max(2, pct)}%` }}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={pct}
              aria-label={sent ? "validating on server" : "uploading"}
            />
          </div>
          <button
            type="button"
            onClick={cancel}
            className="mt-2 text-xs text-tui-danger hover:underline"
          >
            cancel
          </button>
        </div>
      )}

      {state.phase === "error" && (
        <div
          role="alert"
          data-testid="upload-error"
          className="mt-3 text-xs text-tui-danger border border-tui-danger/40 rounded p-2"
        >
          ! {state.name ? `${state.name}: ` : ""}
          {state.message}
        </div>
      )}

      {state.phase === "done" && (
        <div
          role="status"
          data-testid="upload-done"
          className="mt-3 text-xs text-tui-success"
        >
          ✓ queued: {state.name}
          {state.pages != null ? ` (${state.pages} page${state.pages === 1 ? "" : "s"})` : ""}
        </div>
      )}
    </section>
  );
}
