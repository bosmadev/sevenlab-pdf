/**
 * UploadDropzone — drag-and-drop PDF uploader.
 *
 * Mirrors backend validation (MIME + extension + size) client-side so the
 * user gets a fast rejection; the backend `services/pdf_validator.py` is
 * the authoritative check. Streams progress via XHR (fetch cannot expose
 * request-body progress yet).
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { uploadPdf, type UploadResponse } from "@/lib/documents";

interface Props {
  /** Maximum file size in MB — must match backend MAX_PDF_SIZE_MB. */
  maxSizeMb?: number;
  /** Called with the backend response on each successful upload. */
  onUploaded?: (doc: UploadResponse) => void;
}

type UploadState =
  | { phase: "idle" }
  | { phase: "uploading"; name: string; ratio: number }
  | { phase: "error"; message: string }
  | { phase: "done"; name: string };

function clientValidate(file: File, maxBytes: number): string | null {
  if (!/\.pdf$/i.test(file.name)) return "only .pdf files are accepted";
  // The input accept="application/pdf" sometimes yields an empty type on
  // drag-dropped files, so we also check extension above.
  if (file.type && file.type !== "application/pdf") {
    return `unexpected mime: ${file.type}`;
  }
  if (file.size === 0) return "empty file";
  if (file.size > maxBytes) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    return `too large (${mb} MB)`;
  }
  return null;
}

export function UploadDropzone({ maxSizeMb = 50, onUploaded }: Props) {
  const [state, setState] = useState<UploadState>({ phase: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const maxBytes = maxSizeMb * 1024 * 1024;

  const start = useCallback(
    async (file: File) => {
      const err = clientValidate(file, maxBytes);
      if (err) {
        setState({ phase: "error", message: err });
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
        setState({ phase: "done", name: file.name });
        onUploaded?.(doc);
      } catch (e) {
        if ((e as DOMException)?.name === "AbortError") {
          setState({ phase: "idle" });
          return;
        }
        setState({
          phase: "error",
          message: e instanceof Error ? e.message : "upload failed",
        });
      }
    },
    [maxBytes, onUploaded],
  );

  const onFiles = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files || files.length === 0) return;
      // One at a time keeps the UX legible; multi-upload is Phase 2.
      await start(files[0]);
    },
    [start],
  );

  const cancel = () => abortRef.current?.abort();

  const busy = state.phase === "uploading";

  return (
    <section
      className={`border rounded-tui p-4 transition-colors ${
        dragging
          ? "border-tui-borderActive bg-tui-elevated"
          : "border-tui-border bg-tui-surface"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
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
          onChange={(e) => void onFiles(e.target.files)}
          disabled={busy}
        />
      </div>

      {state.phase === "uploading" && (
        <div className="mt-3">
          <div className="text-tui-textMuted text-xs mb-1 flex justify-between">
            <span className="truncate pr-2">{state.name}</span>
            <span className="shrink-0 text-tui-accent">
              {Math.round(state.ratio * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-tui-base rounded overflow-hidden">
            <div
              className="h-full bg-tui-accent transition-all"
              style={{ width: `${Math.max(2, state.ratio * 100)}%` }}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(state.ratio * 100)}
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
          className="mt-3 text-xs text-tui-danger border border-tui-danger/40 rounded p-2"
        >
          ! {state.message}
        </div>
      )}

      {state.phase === "done" && (
        <div
          role="status"
          className="mt-3 text-xs text-tui-success"
        >
          ✓ queued: {state.name}
        </div>
      )}
    </section>
  );
}
