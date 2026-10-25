import { useRef, useState } from "react";
import { DocIcon, TrashIcon, UploadIcon } from "./icons.jsx";

function StatusDot({ status }) {
  const color =
    status === "ready"
      ? "bg-emerald-500"
      : status === "error"
      ? "bg-red-500"
      : "bg-amber-500";
  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} />;
}

export default function Sidebar({
  documents,
  selectedIds,
  onToggleSelect,
  onUpload,
  onDelete,
  uploading,
  uploadError,
}) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files) => {
    if (files && files[0]) onUpload(files[0]);
  };

  return (
    <aside className="flex h-full w-full flex-col border-r border-slate-200 bg-white">
      <div className="p-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-6 text-center transition ${
            dragOver
              ? "border-indigo-500 bg-indigo-50"
              : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50"
          }`}
        >
          <UploadIcon className="mb-2 h-6 w-6 text-indigo-500" />
          <p className="text-sm font-medium text-slate-700">
            {uploading ? "Processing…" : "Drop a PDF or click to upload"}
          </p>
          <p className="mt-1 text-xs text-slate-400">PDF only · up to 10&nbsp;MB</p>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
        {uploadError && (
          <p className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600">
            {uploadError}
          </p>
        )}
      </div>

      <div className="flex items-center justify-between px-4 pb-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Documents
        </h2>
        <span className="text-xs text-slate-400">{documents.length}</span>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto px-2 pb-4">
        {documents.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-slate-400">
            No documents yet. Upload a PDF to start asking questions.
          </p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => {
              const selected = selectedIds.includes(doc.document_id);
              return (
                <li
                  key={doc.document_id}
                  className={`group flex items-center gap-2 rounded-lg px-2 py-2 text-sm transition ${
                    selected ? "bg-indigo-50" : "hover:bg-slate-50"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => onToggleSelect(doc.document_id)}
                    className="h-4 w-4 rounded border-slate-300 text-indigo-600"
                    title="Include in questions"
                  />
                  <DocIcon className="h-5 w-5 shrink-0 text-slate-400" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium text-slate-700" title={doc.filename}>
                      {doc.filename}
                    </p>
                    <p className="flex items-center gap-1.5 text-xs text-slate-400">
                      <StatusDot status={doc.status} />
                      {doc.pages} pages · {doc.chunks} chunks
                    </p>
                  </div>
                  <button
                    onClick={() => onDelete(doc.document_id)}
                    className="rounded p-1 text-slate-300 opacity-0 transition hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
                    title="Delete document"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="border-t border-slate-100 p-3">
        <p className="text-[11px] leading-relaxed text-slate-400">
          Tip: select one or more documents to scope your questions. With none
          selected, all documents are searched.
        </p>
      </div>
    </aside>
  );
}
