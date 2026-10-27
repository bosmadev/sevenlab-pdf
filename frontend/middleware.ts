/**
 * Edge middleware — redirect un-authenticated users to /login.
 *
 * Only the presence of the `app_session` cookie is checked here; the HMAC
 * is verified by the FastAPI backend on every API call. That split keeps
 * `SESSION_SECRET` out of the Next.js runtime.
 *
 * Open paths: /login, /api/* (FastAPI enforces auth there and answers a
 * JSON 401 that `lib/session.redirectToLogin` turns into navigation; an
 * HTML redirect would make XHR/fetch callers see a 200 login page instead),
 * static assets, and favicon. In production Caddy routes /api/* straight
 * to FastAPI, so this only matters for the local dev proxy.
 */
import { NextResponse, type NextRequest } from "next/server";

const COOKIE_NAME = "app_session";
const OPEN_PATHS = ["/login", "/api"];

function isOpen(pathname: string): boolean {
  if (OPEN_PATHS.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    return true;
  }
  // Next.js internals + static assets.
  if (pathname.startsWith("/_next") || pathname.startsWith("/favicon")) {
    return true;
  }
  return false;
}

export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  if (isOpen(pathname)) return NextResponse.next();

  const token = request.cookies.get(COOKIE_NAME)?.value;
  if (token) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?from=${encodeURIComponent(pathname + search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  matcher: [
    // Everything except Next.js internals and common static files.
    "/((?!_next/static|_next/image|favicon.ico|robots.txt|.*\\..*).*)",
  ],
};
