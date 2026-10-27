/**
 * Session helpers shared by every backend-calling module.
 *
 * `redirectToLogin()` is the single place a 401 turns into navigation. The
 * Next.js edge middleware only checks that the `app_session` cookie EXISTS;
 * the FastAPI backend is what rejects an expired or forged one. Without this
 * hook a stale cookie leaves the user on a page whose every request fails
 * with "unauthorized" instead of landing them on /login.
 */

export const LOGIN_PATH = "/login";

/** Path of the login page that returns the user to `current` after sign-in. */
export function loginPathFor(current: string): string {
  return `${LOGIN_PATH}?from=${encodeURIComponent(current || "/")}`;
}

/** Navigate to /login, preserving the current path in `?from=`. No-op on the server and on /login itself. */
export function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  const { pathname, search } = window.location;
  if (pathname === LOGIN_PATH) return;
  window.location.assign(loginPathFor(pathname + search));
}
