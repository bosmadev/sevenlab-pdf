/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * - Always sends the HTTP-only `app_session` cookie (`credentials: "same-origin"`),
 *   so a signed-in user is authenticated transparently.
 * - On a 401 the caller sees `UnauthorizedError`, which the top-level app uses
 *   to route back to /login without losing the current path.
 * - Base URL defaults to a same-origin reverse proxy (Caddy -> FastAPI on
 *   127.0.0.1). Override with `NEXT_PUBLIC_API_BASE` during local dev when
 *   hitting FastAPI directly.
 */

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

function url(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const normalised = path.startsWith("/") ? path : `/${path}`;
  return `${BASE}${normalised}`;
}

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

  if (res.status === 401) throw new UnauthorizedError();

  if (!res.ok) {
    const body = await parseBody(res);
    throw new ApiError(res.status, body);
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
