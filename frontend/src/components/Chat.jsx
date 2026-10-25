import { useEffect, useRef, useState } from "react";
import { SendIcon, SparkIcon } from "./icons.jsx";

function CitationCard({ citation, index }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-semibold text-indigo-600">
          [{index + 1}] {citation.filename}
        </span>
        <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-medium text-slate-500 ring-1 ring-slate-200">
          page {citation.page}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-slate-600">“{citation.excerpt}”</p>
    </div>
  );
}

function Bubble({ message }) {
  const isUser = message.role === "user";
  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
          {message.text}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-3">
        <div
          className={`rounded-2xl rounded-bl-sm border px-4 py-3 text-sm leading-relaxed ${
            message.isError
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-slate-200 bg-white text-slate-800"
          }`}
        >
          {message.text}
          {message.usedLlm === false && !message.isError && (
            <p className="mt-2 text-[11px] italic text-slate-400">
              Retrieval-only response (no LLM key configured).
            </p>
          )}
        </div>
        {message.citations && message.citations.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Sources
            </p>
            {message.citations.map((c, i) => (
              <CitationCard key={i} citation={c} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Chat({ messages, onSend, loading, onReset, selectedCount }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    onSend(q);
    setInput("");
  };

  return (
    <section className="flex h-full flex-col bg-slate-50">
      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 && !loading && (
            <div className="mt-10 flex flex-col items-center text-center">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-100">
                <SparkIcon className="h-6 w-6 text-indigo-600" />
              </div>
              <h3 className="text-base font-semibold text-slate-700">
                Ask a question about your documents
              </h3>
              <p className="mt-1 max-w-sm text-sm text-slate-400">
                Upload a PDF on the left, then ask anything. Answers come with
                citations to the exact page the evidence was found on.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <Bubble key={i} message={m} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3">
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.3s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:-0.15s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300" />
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white px-4 py-3 sm:px-8">
        <form onSubmit={submit} className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) submit(e);
            }}
            rows={1}
            placeholder={
              selectedCount > 0
                ? `Ask about ${selectedCount} selected document(s)…`
                : "Ask about all your documents…"
            }
            className="max-h-32 flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2.5 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
            title="Send"
          >
            <SendIcon className="h-5 w-5" />
          </button>
        </form>
        <div className="mx-auto mt-2 flex max-w-3xl justify-between text-[11px] text-slate-400">
          <span>Enter to send · Shift+Enter for a new line</span>
          {messages.length > 0 && (
            <button onClick={onReset} className="hover:text-slate-600">
              New conversation
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
