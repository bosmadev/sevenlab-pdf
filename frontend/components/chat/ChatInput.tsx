"use client";
import { useRef, useState, type FormEvent, type KeyboardEvent } from "react";

interface Props {
  disabled?: boolean;
  streaming?: boolean;
  onSubmit: (text: string) => void;
  onCancel?: () => void;
}

export function ChatInput({ disabled, streaming, onSubmit, onCancel }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  function submit(e?: FormEvent) {
    if (e) e.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;
    onSubmit(text);
    setValue("");
    requestAnimationFrame(() => {
      const el = ref.current;
      if (!el) return;
      el.style.height = "auto";
      el.focus();
    });
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape" && streaming && onCancel) {
      e.preventDefault();
      onCancel();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form
      onSubmit={submit}
      className="border border-tui-border bg-tui-surface rounded-tui p-2 flex gap-2 items-end focus-within:border-tui-borderActive transition-colors"
    >
      <span className="text-tui-accent select-none pt-1 text-sm">&gt;</span>
      <textarea
        ref={ref}
        rows={1}
        value={value}
        placeholder={streaming ? "streaming… esc to stop, or type the next question" : "ask a question about your PDFs"}
        onChange={(e) => {
          setValue(e.target.value);
          const el = e.currentTarget;
          el.style.height = "auto";
          el.style.height = Math.min(el.scrollHeight, 200) + "px";
        }}
        onKeyDown={onKey}
        disabled={disabled && !streaming}
        className="flex-1 bg-transparent outline-none resize-none text-sm text-tui-text placeholder:text-tui-textMuted min-h-[1.5rem] max-h-[200px]"
        aria-label="chat input"
        autoFocus
      />
      {streaming && onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          className="tui-btn tui-btn-danger shrink-0"
          aria-label="cancel stream"
        >
          STOP
        </button>
      ) : (
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="tui-btn tui-btn-primary shrink-0"
          aria-label="send message"
        >
          SEND ↵
        </button>
      )}
    </form>
  );
}
