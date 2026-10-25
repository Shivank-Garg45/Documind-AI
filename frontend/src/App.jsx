import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Chat from "./components/Chat.jsx";
import { SparkIcon } from "./components/icons.jsx";
import {
  askQuestion,
  deleteDocument,
  getHealth,
  listDocuments,
  uploadDocument,
} from "./services/api.js";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [messages, setMessages] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const { documents } = await listDocuments();
      setDocuments(documents);
      // Drop selections that no longer exist.
      setSelectedIds((ids) =>
        ids.filter((id) => documents.some((d) => d.document_id === id))
      );
    } catch {
      /* backend offline — surfaced via health banner */
    }
  }, []);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    refresh();
  }, [refresh]);

  const handleUpload = async (file) => {
    setUploadError("");
    setUploading(true);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setUploadError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await deleteDocument(id);
      await refresh();
    } catch {
      /* ignore; list will re-sync on next refresh */
    }
  };

  const toggleSelect = (id) =>
    setSelectedIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    );

  const handleSend = async (query) => {
    setMessages((m) => [...m, { role: "user", text: query }]);
    setLoading(true);
    try {
      const res = await askQuestion(query, selectedIds);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          citations: res.citations,
          usedLlm: res.used_llm,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: err.message || "Something went wrong.",
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setSidebarOpen(true)}
            className="-ml-1 rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 md:hidden"
            title="Documents"
            aria-label="Open documents panel"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600">
            <SparkIcon className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold leading-tight text-slate-800">
              DocuMind AI
            </h1>
            <p className="text-xs text-slate-400">
              Ask your PDFs — answers with page citations
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {health ? (
            <>
              <span className="hidden rounded-full bg-slate-100 px-2.5 py-1 text-slate-500 sm:inline">
                embeddings: {health.embedding_backend}
              </span>
              <span
                className={`rounded-full px-2.5 py-1 font-medium ${
                  health.llm_enabled
                    ? "bg-emerald-50 text-emerald-600"
                    : "bg-amber-50 text-amber-600"
                }`}
                title={
                  health.llm_enabled
                    ? "LLM answer generation is enabled"
                    : "No API key: retrieval-only answers"
                }
              >
                {health.llm_enabled
                  ? `LLM ready${health.llm_provider ? ` (${health.llm_provider})` : ""}`
                  : "retrieval-only"}
              </span>
            </>
          ) : (
            <span className="rounded-full bg-red-50 px-2.5 py-1 font-medium text-red-600">
              backend offline
            </span>
          )}
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[300px_1fr]">
        {/* Desktop sidebar */}
        <div className="hidden md:block">
          <Sidebar
            documents={documents}
            selectedIds={selectedIds}
            onToggleSelect={toggleSelect}
            onUpload={handleUpload}
            onDelete={handleDelete}
            uploading={uploading}
            uploadError={uploadError}
          />
        </div>

        {/* Mobile sidebar drawer */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-20 md:hidden">
            <div
              className="absolute inset-0 bg-slate-900/40"
              onClick={() => setSidebarOpen(false)}
            />
            <div className="absolute left-0 top-0 h-full w-80 max-w-[85%] shadow-xl">
              <Sidebar
                documents={documents}
                selectedIds={selectedIds}
                onToggleSelect={toggleSelect}
                onUpload={handleUpload}
                onDelete={handleDelete}
                uploading={uploading}
                uploadError={uploadError}
              />
            </div>
          </div>
        )}

        <Chat
          messages={messages}
          onSend={handleSend}
          loading={loading}
          onReset={() => setMessages([])}
          selectedCount={selectedIds.length}
        />
      </div>
    </div>
  );
}
