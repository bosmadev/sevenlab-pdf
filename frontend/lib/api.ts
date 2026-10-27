/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * - Always sends the HTTP-only `app_session` cookie (`credentials: "same-origin"`),
 *   so a signed-in user is authenticated transparently.
 * - On a 401 the browser is sent to /login (`redirectToLogin`, current path
 *   kept in `?from=`) and the caller sees `UnauthorizedError`.
 * - Base URL defaults to a same-origin reverse proxy (Caddy -> FastAPI on
 *   127.0.0.1). Override with `NEXT_PUBLIC_API_BASE` during local dev when
 *   hitting FastAPI directly.
 */
import { redirectToLogin } from "./session";

export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
    this.name = "UnauthorizedError";
  }
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(status: number, body: unknown, message?: string) {
    super(message ?? `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

const STATUS_FALLBACK: Record<number, string> = {
  400: "bad request",
  401: "unauthorized",
  403: "forbidden",
  404: "not found",
  409: "conflict",
  413: "file too large",
  415: "unsupported file type",
  422: "invalid request",
  429: "too many requests, slow down",
  500: "server error",
  502: "backend unavailable",
  503: "backend unavailable",
  504: "backend timed out",
};

/**
 * Collapse every error body the backend can emit into one human string.
 *
 * Shapes seen in the wild (all must stay supported):
 *   {detail: "Not authenticated"}                        FastAPI HTTPException
 *   {detail: {code, reason}}                             routes/documents.py
 *   {detail: [{loc, msg, type}]}                         FastAPI 422 validation
 *   {error: "Not authenticated"}                         security/middleware.py
 *   {error: {code, message, request_id}}                 errors.py envelope
 *   "<html>…" / "" / null                                Caddy / uvicorn / aborted
 */
export function errorMessageFromBody(status: number, body: unknown): string {
  const fallback = STATUS_FALLBACK[status] ?? `HTTP ${status}`;
  if (body == null) return fallback;

  if (typeof body === "string") {
    const text = body.trim();
    if (!text || text.startsWith("<")) return fallback;
    return text.length > 200 ? `${text.slice(0, 200)}…` : text;
  }

  if (typeof body !== "object") return fallback;
  const obj = body as Record<string, unknown>;

  const pick = (v: unknown): string | null => {
    if (typeof v === "string" && v.trim()) return v.trim();
    if (Array.isArray(v)) {
      const msgs = v
        .map((item) =>
          item && typeof item === "object" && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : null,
        )
        .filter((m): m is string => !!m);
      return msgs.length ? msgs.join("; ") : null;
    }
    if (v && typeof v === "object") {
      const o = v as Record<string, unknown>;
      return pick(o.reason) ?? pick(o.message) ?? pick(o.detail) ?? null;
    }
    return null;
  };

  return pick(obj.detail) ?? pick(obj.error) ?? pick(obj.message) ?? fallback;
}

export { loginPathFor } from "./session";

function url(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const normalised = path.startsWith("/") ? path : `/${path}`;
  return `${BASE}${normalised}`;
}

export { url as apiUrl };

async function parseBody(res: Response): Promise<unknown> {
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    try {
      return await res.json();
    } catch {
      return null;
    }
  }
  try {
    return await res.text();
  } catch {
    return null;
  }
}

export interface RequestInitWithTimeout extends RequestInit {
  timeoutMs?: number;
}

async function request<T>(path: string, init: RequestInitWithTimeout = {}): Promise<T> {
  const { timeoutMs, ...rest } = init;
  const controller = new AbortController();
  const timer = timeoutMs
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;

  // Compose signals: caller abort still wins.
  const callerSignal = rest.signal;
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  let res: Response;
  try {
    res = await fetch(url(path), {
      ...rest,
      credentials: "same-origin",
      signal: controller.signal,
    });
  } finally {
    if (timer) clearTimeout(timer);
  }

  if (res.status === 401) {
    redirectToLogin();
    throw new UnauthorizedError();
  }

  if (!res.ok) {
    const body = await parseBody(res);
    throw new ApiError(res.status, body, errorMessageFromBody(res.status, body));
  }

  if (res.status === 204) return undefined as T;
  return (await parseBody(res)) as T;
}

export const api = {
  get: <T>(path: string, init?: RequestInitWithTimeout) =>
    request<T>(path, { ...init, method: "GET" }),

  del: <T>(path: string, init?: RequestInitWithTimeout) =>
    request<T>(path, { ...init, method: "DELETE" }),

  postJson: <T>(path: string, body: unknown, init?: RequestInitWithTimeout) =>
    request<T>(path, {
      ...init,
      method: "POST",
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      body: JSON.stringify(body),
    }),

  postForm: <T>(path: string, form: FormData, init?: RequestInitWithTimeout) =>
    request<T>(path, {
      ...init,
      method: "POST",
      body: form,
    }),
};
