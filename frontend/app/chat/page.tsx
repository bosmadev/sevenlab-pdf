import { ChatView } from "@/components/chat/ChatView";

export const metadata = {
  title: "sevenlab-pdf · chat",
  description: "Chat with your indexed PDFs",
  robots: "noindex, nofollow",
};

export default function ChatPage() {
  return (
    <main className="h-screen w-screen bg-tui-base text-tui-text">
      <ChatView />
    </main>
  );
}
