/**
 * Documents API — thin typed wrapper over `/api/documents`.
 *
 * Keeps route surface and shapes in one place so the sidebar, uploader,
 * and chat composer agree. The backend contract (T-004 + T-013) returns
 * `original_filename` / `page_count` / `created_at` and a status drawn
 * from `pending | parsing | chunking | embedding | ready | error`; this
 * module normalises both to the three-state UI shape in `lib/types.ts`.
 */
import { ApiError, UnauthorizedError, api, apiUrl, errorMessageFromBody } from "./api";
import { redirectToLogin } from "./session";
import type { DocumentSummary } from "./types";

/** Must match backend `MAX_PDF_SIZE_MB` (services/pdf_validator.py). */
export const DEFAULT_MAX_PDF_MB = 50;

/**
 * Client-side mirror of `services/pdf_validator.py`: extension, declared
 * MIME, non-empty, size cap. Returns a human reason or `null` when the file
 * is worth sending. The backend stays authoritative (magic bytes, sanitizer).
 */
export function validatePdfFile(
  file: File,
  maxSizeMb: number = DEFAULT_MAX_PDF_MB,
): string | null {
  if (!/\.pdf$/i.test(file.name)) return "only .pdf files are accepted";
  // Drag-dropped files sometimes carry an empty `type`, so the extension
  // check above is the one that always runs; MIME is checked when present.
  if (file.type && file.type !== "application/pdf") {
    return `unexpected file type: ${file.type}`;
  }
  if (file.size === 0) return "empty file";
  const maxBytes = maxSizeMb * 1024 * 1024;
  if (file.size > maxBytes) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    return `too large (${mb} MB, limit ${maxSizeMb} MB)`;
  }
  return null;
}

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
  /** Ratio 0..1 of the request body sent. Reaches 1 before the server answers. */
  onProgress?: (ratio: number) => void;
  /** Give up if the server has not answered within this window. */
  timeoutMs?: number;
}

/** Sanitizing a 50 MB PDF inside the Docker sandbox can take minutes. */
export const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Upload a single PDF. Uses XHR instead of fetch so we can surface upload
 * progress (fetch does not expose it for request bodies yet in Chrome 141).
 *
 * Rejects with `UnauthorizedError` on 401 (after `redirectToLogin`), with
 * `ApiError` carrying a readable message for every other failure, and with
 * an `AbortError` DOMException when cancelled.
 */
export function uploadPdf({
  file,
  signal,
  onProgress,
  timeoutMs = UPLOAD_TIMEOUT_MS,
}: UploadOptions): Promise<UploadResponse> {
  return new Promise<UploadResponse>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/documents/upload"), true);
    xhr.withCredentials = true;
    xhr.responseType = "json";
    xhr.timeout = timeoutMs;

    if (onProgress) {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) onProgress(Math.min(1, ev.loaded / ev.total));
      };
      // Body fully sent: the server is now validating + sanitizing.
      xhr.upload.onload = () => onProgress(1);
    }

    xhr.onload = () => {
      const body: unknown = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) {
        // A 2xx without a document id means a proxy or redirect swallowed
        // the real answer (e.g. an HTML login page); never report success.
        if (body && typeof body === "object" && typeof (body as UploadResponse).id === "string") {
          resolve(body as UploadResponse);
        } else {
          reject(new ApiError(xhr.status, body, "unexpected response from server"));
        }
      } else if (xhr.status === 401) {
        redirectToLogin();
        reject(new UnauthorizedError());
      } else {
        reject(new ApiError(xhr.status, body, errorMessageFromBody(xhr.status, body)));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError(0, null, "network error, backend unreachable"));
    xhr.ontimeout = () =>
      reject(new ApiError(0, null, `upload timed out after ${Math.round(timeoutMs / 60000)} min`));
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
