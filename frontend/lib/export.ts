import { citationName, citationPreview, type ChatMessage, type Citation } from "./types";

const label = (c: Citation, i: number): number =>
  typeof c.n === "number" ? c.n : i + 1;

function fmtCitations(cites: Citation[] | undefined): string {
  if (!cites || cites.length === 0) return "";
  const lines = cites.map((c, i) => {
    const prev = citationPreview(c);
    return `  [${label(c, i)}] ${citationName(c)} p.${c.page}${
      prev ? ` — ${prev.replace(/\s+/g, " ").slice(0, 160)}` : ""
    }`;
  });
  return "\nSources:\n" + lines.join("\n");
}

export function toMarkdown(messages: ChatMessage[]): string {
  const header = `# sevenlab-pdf chat\n\n_Exported ${new Date().toISOString()}_\n\n---\n\n`;
  const body = messages
    .map((m) => {
      const who = m.role === "user" ? "**You**" : m.role === "assistant" ? "**Assistant**" : "**System**";
      const cites =
        m.citations && m.citations.length > 0
          ? "\n\n**Sources**\n" +
            m.citations
              .map((c, i) => {
                const prev = citationPreview(c);
                return `${label(c, i)}. ${citationName(c)} — p.${c.page}${
                  prev ? ` — _${prev.replace(/\s+/g, " ").slice(0, 200)}_` : ""
                }`;
              })
              .join("\n")
          : "";
      return `### ${who}\n\n${m.content}${cites}`;
    })
    .join("\n\n---\n\n");
  return header + body + "\n";
}

export function toPlainText(messages: ChatMessage[]): string {
  const header = `sevenlab-pdf chat — exported ${new Date().toISOString()}\n${"=".repeat(60)}\n\n`;
  const body = messages
    .map((m) => {
      const who = m.role === "user" ? "YOU" : m.role === "assistant" ? "ASSISTANT" : "SYSTEM";
      return `[${who}]\n${m.content}${fmtCitations(m.citations)}`;
    })
    .join("\n\n" + "-".repeat(60) + "\n\n");
  return header + body + "\n";
}

export function download(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
