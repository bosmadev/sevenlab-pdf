/**
 * 404 page (Next.js App Router convention). Rendered for unknown routes and
 * whenever a page calls `notFound()`.
 */
import Link from "next/link";

export const metadata = {
  title: "sevenlab-pdf · not found",
  robots: "noindex, nofollow",
};

export default function NotFound() {
  return (
    <main className="min-h-screen bg-tui-base text-tui-text flex items-center justify-center p-4">
      <section className="w-full max-w-md border border-tui-border bg-tui-surface rounded-tui p-4 text-sm">
        <div className="text-tui-textMuted text-xs uppercase tracking-wider mb-2">
          ── 404 ──
        </div>
        <p>! no such page.</p>
        <p className="text-tui-textMuted text-xs mt-2">
          the address may be mistyped, or the page was removed.
        </p>
        <div className="mt-4">
          <Link
            href="/"
            className="inline-block text-xs px-3 py-1.5 border border-tui-border rounded-tui text-tui-accent hover:border-tui-borderActive"
          >
            home →
          </Link>
        </div>
      </section>
    </main>
  );
}
