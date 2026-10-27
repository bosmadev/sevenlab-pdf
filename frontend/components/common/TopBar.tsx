/**
 * TopBar — brand, section nav and session controls on authenticated pages.
 */
"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { APP_NAME, APP_VERSION } from "@/lib/app-config";
import { logout } from "@/lib/documents";

const NAV = [
  { href: "/", label: "docs" },
  { href: "/chat", label: "chat" },
] as const;

export function TopBar() {
  const router = useRouter();
  const pathname = usePathname();
  const [busy, setBusy] = useState(false);

  async function handleLogout() {
    setBusy(true);
    try {
      await logout();
    } catch {
      // Logout should never block the UI. Clear cookie happens server-side;
      // even on failure, routing to /login is the right next step.
    }
    router.push("/login");
    router.refresh();
  }

  return (
    <header className="border-b border-tui-border bg-tui-surface px-4 py-2 flex items-center justify-between">
      <div className="flex items-baseline gap-4">
        <div className="flex items-baseline gap-2">
          <span className="text-tui-accent text-sm font-semibold">{APP_NAME}</span>
          <span className="text-tui-textMuted text-xs">v{APP_VERSION}</span>
        </div>
        <nav className="flex items-center gap-3 text-xs" aria-label="sections">
          {NAV.map((n) => {
            const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                aria-current={active ? "page" : undefined}
                className={
                  active
                    ? "text-tui-textBright border-b border-tui-borderActive"
                    : "text-tui-textMuted hover:text-tui-text"
                }
              >
                [{n.label}]
              </Link>
            );
          })}
        </nav>
      </div>
      <button
        type="button"
        onClick={() => void handleLogout()}
        disabled={busy}
        className="text-xs text-tui-textMuted hover:text-tui-danger transition-colors disabled:opacity-50"
      >
        {busy ? "signing out…" : "logout"}
      </button>
    </header>
  );
}
