/**
 * Documents API — thin typed wrapper over `/api/documents`.
 *
 * Keeps route surface and shapes in one place so the sidebar, uploader,
 * and chat composer agree. The backend contract (T-004 + T-013) returns
 * `original_filename` / `page_count` / `created_at` and a status drawn
 * from `pending | parsing | chunking | embedding | ready | error`; this
 * module normalises both to the three-state UI shape in `lib/types.ts`.
 */
import { api } from "./api";
import type { DocumentSummary } from "./types";

/** Shape returned by `POST /api/documents/upload`. The sanitized filename
 * is distinct from the original filename the user picked; both are stored
 * server-side but only the sanitized one is echoed here. */
export interface UploadResponse {
  id: string;
  status?: string;
  sanitized_filename?: string;
  sha256?: string;
  size_bytes?: number;
  page_count?: number;
  mime_type?: string;
  stripped_keys?: string[];
}

// Backend row as returned by GET /api/documents/{id?} — fields the UI
// cares about. Any field we do not read is intentionally omitted.
interface BackendDoc {
  id: string;
  original_filename?: string;
  sanitized_filename?: string;
  size_bytes?: number;
  page_count?: number;
  status?: string;
  created_at?: string;
  processed_at?: string | null;
}

function mapStatus(s: string | undefined): DocumentSummary["status"] {
  if (!s) return undefined;
  if (s === "ready") return "ready";
  if (s === "error" || s === "failed") return "failed";
  // pending / parsing / chunking / embedding all collapse to "processing"
  // so the status badge turns green only when the pipeline is actually done.
  return "processing";
}

function normaliseDoc(d: BackendDoc): DocumentSummary {
  return {
    id: d.id,
    name: d.original_filename ?? d.sanitized_filename ?? d.id,
    pages: d.page_count,
    size_bytes: d.size_bytes,
    uploaded_at: d.created_at,
    status: mapStatus(d.status),
  };
}

export async function listDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
  const raw = await api.get<BackendDoc[] | { documents?: BackendDoc[] }>(
    "/api/documents/",
    { signal },
  );
  const rows = Array.isArray(raw) ? raw : (raw?.documents ?? []);
  return rows.map(normaliseDoc);
}

export async function getDocument(
  id: string,
  signal?: AbortSignal,
): Promise<DocumentSummary> {
  const row = await api.get<BackendDoc>(
    `/api/documents/${encodeURIComponent(id)}`,
    { signal },
  );
  return normaliseDoc(row);
}

export async function deleteDocument(id: string): Promise<void> {
  await api.del(`/api/documents/${encodeURIComponent(id)}`);
}

export interface UploadOptions {
  file: File;
  signal?: AbortSignal;
  onProgress?: (ratio: number) => void;
}

/**
 * Upload a single PDF. Uses XHR instead of fetch so we can surface upload
 * progress (fetch does not expose it for request bodies yet in Chrome 141).
 */
export function uploadPdf({ file, signal, onProgress }: UploadOptions): Promise<UploadResponse> {
  return new Promise<UploadResponse>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents/upload", true);
    xhr.withCredentials = true;
    xhr.responseType = "json";

    if (onProgress) {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) onProgress(ev.loaded / ev.total);
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve((xhr.response ?? {}) as UploadResponse);
      } else if (xhr.status === 401) {
        reject(new Error("unauthorized"));
      } else {
        const msg =
          (xhr.response && (xhr.response.detail || xhr.response.error)) ||
          `upload failed (${xhr.status})`;
        reject(new Error(String(msg)));
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.onabort = () => reject(new DOMException("aborted", "AbortError"));

    if (signal) {
      if (signal.aborted) {
        xhr.abort();
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }

    xhr.send(form);
  });
}

export async function logout(): Promise<void> {
  await api.del("/api/auth");
}
