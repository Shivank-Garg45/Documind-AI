// Thin client for the DocuMind AI backend.
// Uses relative /api URLs; Vite proxies these to the FastAPI server in dev.

const BASE = import.meta.env.VITE_API_BASE || "";

async function handle(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getHealth() {
  return handle(await fetch(`${BASE}/api/health`));
}

export async function listDocuments() {
  return handle(await fetch(`${BASE}/api/documents`));
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return handle(
    await fetch(`${BASE}/api/documents`, { method: "POST", body: form })
  );
}

export async function deleteDocument(documentId) {
  return handle(
    await fetch(`${BASE}/api/documents/${documentId}`, { method: "DELETE" })
  );
}

export async function askQuestion(query, documentIds) {
  return handle(
    await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        document_ids: documentIds && documentIds.length ? documentIds : null,
      }),
    })
  );
}
