/**
 * Login Page — Terminal Auth Gate
 *
 * Password gate matching ~/source/scraper/main/app/login/page.tsx TUI
 * aesthetic. POSTs to the FastAPI /api/auth route (same-origin via the
 * Next.js rewrite in next.config.mjs) and redirects on success.
 */
"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, Suspense, useState } from "react";
import { errorMessageFromBody } from "@/lib/api";
import { APP_DESCRIPTION, APP_NAME, APP_VERSION } from "@/lib/app-config";
import styles from "./login.module.css";

function LockIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function AlertIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

function LoginForm() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ password }),
      });

      if (res.ok) {
        const redirectTo = searchParams.get("from") || "/";
        router.push(redirectTo);
        router.refresh();
        return;
      }

      if (res.status === 429) {
        const retryAfter = res.headers.get("retry-after") || "60";
        setError(`Too many attempts. Try again in ${retryAfter}s.`);
      } else {
        // The backend answers with the errors.py envelope; a 422 carries a
        // list in `detail`, which must never reach the DOM as an object.
        let body: unknown = null;
        try {
          body = await res.json();
        } catch {
          /* non-JSON response (proxy / aborted) — status fallback below */
        }
        setError(errorMessageFromBody(res.status, body));
      }
    } catch {
      setError("Connection failed. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.header}>
          <div className={styles.headerLine}>AUTH</div>
          <div className={styles.lockBadge}>
            <LockIcon />
          </div>
          <h1 className={styles.title}>{APP_NAME}</h1>
          <p className={styles.subtitle}>{APP_DESCRIPTION}</p>
        </div>

        <form onSubmit={handleSubmit}>
          <label className={styles.prompt} htmlFor="password-input">
            <span className={styles.promptSymbol}>$</span>
            Password
          </label>
          <div className={styles.inputWrapper}>
            <input
              id="password-input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              disabled={loading}
              className={`${styles.input} ${error ? styles.inputError : ""}`}
              aria-invalid={!!error}
              autoComplete="current-password"
              autoFocus
            />
          </div>

          {error && (
            <div className={styles.error} role="alert">
              <AlertIcon className={styles.errorIcon} />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !password}
            className={`${styles.button} ${loading ? styles.buttonLoading : ""}`}
          >
            {loading ? (
              <>
                <span className={styles.spinner} />
                Authenticating...
              </>
            ) : (
              "Authenticate"
            )}
          </button>
        </form>

        <div className={styles.footer}>
          <span>v{APP_VERSION}</span>
          <span className={styles.footerDot} />
          <span>session: 7d</span>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
