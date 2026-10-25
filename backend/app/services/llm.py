"""LLM answer generation over retrieved evidence.

This is the *generation* half of RAG, kept separate from retrieval. It takes a
question plus already-retrieved passages and asks an LLM to answer using ONLY
that evidence.

Two providers are supported behind one small interface:

* ``OpenAIClient``  — OpenAI chat models (needs ``OPENAI_API_KEY``, paid).
* ``GeminiClient``  — Google Gemini models (needs ``GOOGLE_API_KEY``; has a free
  tier, so it is a good default for a no-cost demo).

Both share graceful degradation: with no key they return a transparent
retrieval-only answer instead of faking a generated one, and provider/network
failures raise :class:`LLMError` so the API can report a clear error.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.logging_config import get_logger
from app.schemas import Citation, Passage

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are DocuMind AI, a careful assistant that answers questions strictly "
    "from the provided document excerpts.\n"
    "Rules:\n"
    "1. Use ONLY the information in the EVIDENCE section. Do not use outside "
    "knowledge.\n"
    "2. If the evidence does not contain enough information to answer, reply "
    "exactly: 'The documents do not contain enough information to answer this "
    "question.'\n"
    "3. Cite the specific sources you used by their [#] markers.\n"
    "4. Never invent facts, page numbers, quotations, or citations.\n"
    "5. Be concise and factual."
)

INSUFFICIENT_MSG = (
    "The documents do not contain enough information to answer this question."
)


class LLMError(RuntimeError):
    """Raised for any failure talking to the LLM provider."""


@dataclass
class LLMResult:
    answer: str
    used_llm: bool


def build_evidence_block(passages: list[Passage]) -> str:
    """Render retrieved passages into a numbered evidence block."""
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {p.filename}, page {p.page})\n{p.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, passages: list[Passage]) -> str:
    evidence = build_evidence_block(passages)
    return (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "Answer the question using only the evidence above, and cite the "
        "[#] markers you relied on."
    )


def passages_to_citations(
    passages: list[Passage], *, max_excerpt: int = 320
) -> list[Citation]:
    citations = []
    for p in passages:
        excerpt = p.text.strip()
        if len(excerpt) > max_excerpt:
            excerpt = excerpt[:max_excerpt].rstrip() + "…"
        citations.append(
            Citation(
                document_id=p.document_id,
                filename=p.filename,
                page=p.page,
                excerpt=excerpt,
            )
        )
    return citations


class BaseLLMClient(ABC):
    """Shared RAG-answer flow; providers implement :meth:`_generate`."""

    def __init__(self, api_key: str | None, model: str, *, timeout: float = 30.0):
        self._api_key = (api_key or "").strip()
        self._model = model
        self._timeout = timeout

    # Each provider defines what counts as a usable key and how to call out.
    _PLACEHOLDERS: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key) and self._api_key not in self._PLACEHOLDERS

    @abstractmethod
    def _generate(self, question: str, passages: list[Passage]) -> str:
        """Call the provider and return the answer text. Raise on failure."""

    def answer(self, question: str, passages: list[Passage]) -> LLMResult:
        if not passages:
            return LLMResult(answer=INSUFFICIENT_MSG, used_llm=False)

        if not self.enabled:
            logger.info("LLM disabled (no API key); returning retrieval-only answer.")
            return LLMResult(
                answer=self._retrieval_only_answer(passages), used_llm=False
            )

        try:
            text = (self._generate(question, passages) or "").strip()
            if not text:
                raise LLMError("The LLM returned an empty response.")
            return LLMResult(answer=text, used_llm=True)
        except LLMError:
            raise
        except Exception as exc:  # network, auth, rate limit, provider errors
            # Never leak the key or raw provider internals to the client.
            logger.error("LLM request failed: %s", type(exc).__name__)
            raise LLMError(
                "The AI service is currently unavailable. Please try again later."
            ) from exc

    @staticmethod
    def _retrieval_only_answer(passages: list[Passage]) -> str:
        top = passages[0]
        return (
            "LLM answer generation is disabled (no API key configured), so here "
            "is the most relevant passage found instead:\n\n"
            f"\"{top.text[:500].strip()}\"\n\n"
            f"(Source: {top.filename}, page {top.page}.) "
            "Configure an LLM API key to get a synthesised, cited answer."
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI chat-completions provider."""

    _PLACEHOLDERS = ("sk-your-key-here",)

    def _generate(self, question: str, passages: list[Passage]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, passages)},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


class GeminiClient(BaseLLMClient):
    """Google Gemini provider (free tier available)."""

    _PLACEHOLDERS = ("your-google-api-key-here",)

    def _generate(self, question: str, passages: list[Passage]) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout * 1000)),
        )
        response = client.models.generate_content(
            model=self._model,
            contents=build_user_prompt(question, passages),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        return response.text or ""


# Backwards-compatible alias: historically the OpenAI client was the only one.
LLMClient = OpenAIClient
