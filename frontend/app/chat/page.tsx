import { Suspense } from "react";
import { ChatView } from "@/components/chat/ChatView";
import { TopBar } from "@/components/common/TopBar";

export const metadata = {
  title: "sevenlab-pdf · chat",
  description: "Chat with your indexed PDFs",
  robots: "noindex, nofollow",
};

export default function ChatPage() {
  return (
    <main className="h-screen bg-tui-base text-tui-text flex flex-col overflow-hidden">
      <TopBar />
      <div className="flex-1 min-h-0">
        <Suspense
          fallback={<div className="p-3 text-xs text-tui-textMuted">loading chat…</div>}
        >
          <ChatView />
        </Suspense>
      </div>
    </main>
  );
}
