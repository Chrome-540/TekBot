"""
TekMark RAG Pipeline
────────────────────
PDF → Hierarchical Chunking → Gemini Embeddings → Pinecone → Gemini 2.5 Flash
Uses: google-genai (new SDK), pinecone, langchain-text-splitters, pdfplumber
"""

from __future__ import annotations
import re, time
from typing import Any

import pdfplumber
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


# ── Section detection ──────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    (r"(?i)^home\s*page",                 "Home Page"),
    (r"(?i)^know\s*us",                   "Know Us"),
    (r"(?i)^ai\s*testing",                "AI Testing"),
    (r"(?i)eft\s*testing",                "EFT Testing"),
    (r"(?i)self.?heal",                   "Self-Healing Automation"),
    (r"(?i)regression\s*planning",        "AI Regression Planning"),
    (r"(?i)cloud\s*load",                 "Cloud Load Testing"),
    (r"(?i)testing\s*services",           "Testing Services"),
    (r"(?i)testing\s*roadmap",            "Testing Roadmap"),
    (r"(?i)testing\s*managed|taas",       "Testing Managed Services / TaaS"),
    (r"(?i)client\s*(success|engagement)", "Client Engagement"),
    (r"(?i)^it\s*services",               "IT Services"),
    (r"(?i)sign.?off",                    "Testing Sign-off Criteria"),
    (r"(?i)qa\s*maturity",                "QA Maturity"),
]

def detect_section(text: str) -> str:
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, text[:120]):
            return label
    return "General"


# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        current_section = "Home Page"
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            detected = detect_section(text)
            if detected != "General":
                current_section = detected
            pages.append({"page_num": i, "text": text, "section": current_section})
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )
    chunks, chunk_id = [], 0
    for page in pages:
        for t in splitter.split_text(page["text"]):
            t = t.strip()
            if len(t) < 40:
                continue
            chunks.append({
                "id":       f"chunk_{chunk_id:04d}",
                "text":     t,
                "page_num": page["page_num"],
                "section":  page["section"],
            })
            chunk_id += 1
    return chunks


# ── Main RAG class ─────────────────────────────────────────────────────────────
class TekMarkRAG:
    EMBED_MODEL = "gemini-embedding-001"
    CHAT_MODEL  = "gemini-2.5-flash"
    TOP_K       = 5
    VECTOR_DIM  = 3072
    BATCH_SIZE  = 50

    SYSTEM_PROMPT = (
        "You are TekMark's expert AI assistant with deep knowledge of TekMark's "
        "QA services, AI testing capabilities, industry expertise and client engagement models.\n\n"
        "Guidelines:\n"
        "- Answer ONLY from the provided context chunks.\n"
        "- Be precise, professional and helpful.\n"
        "- If the context does not cover the question, say so clearly.\n"
        "- Use bullet points where appropriate for clarity.\n"
        "- Keep answers concise but complete.\n"
        "- Do NOT mention page numbers, section names, or source references in your answer."
    )

    def __init__(self, gemini_api_key: str, pinecone_api_key: str, pinecone_index: str = "tekmark-rag"):
        self.client = genai.Client(api_key=gemini_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = pinecone_index
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def _embed_batch(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task), 
        )
        return [e.values for e in response.embeddings]


#Functional testing = 0000
#AI Functional testing = 0001
#Tekmark = 0123

#query = what is fuctional testing = 0010

#result(data) = [0001,0000]! = [0123]

#AI judge (LLM)= [{result}, query] =  not relevant  =20% or 30%




#llm(query, result) = output (fucnt is  a....) i dont know

#faitfullnss = #AI judge (LLM) = [output, result] how faitfull the output to the result = 100%






    def _embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query], task="RETRIEVAL_QUERY")[0]

    def index_document(self, pdf_path: str) -> int:
        pages  = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        for start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[start: start + self.BATCH_SIZE]
            embeddings = self._embed_batch([c["text"] for c in batch])
            self.index.upsert(vectors=[
                {
                    "id":     c["id"],
                    "values": emb,
                    "metadata": {"text": c["text"], "page": c["page_num"], "section": c["section"]},
                }
                for c, emb in zip(batch, embeddings)
            ])
        return len(chunks)

    def query(self, question: str, chat_history: list[dict] | None = None) -> dict[str, Any]:
        q_vec = self._embed_query(question)
        results = self.index.query(vector=q_vec, top_k=self.TOP_K, include_metadata=True)

        sources, context_parts = [], []
        for match in results.matches:
            meta = match.metadata
            sources.append({
                "text":    meta.get("text", ""),
                "page":    meta.get("page", "?"),
                "section": meta.get("section", "General"),
                "score":   round(match.score, 3),
            })
            context_parts.append(meta.get("text", ""))

        context = "\n\n---\n\n".join(context_parts)

        # Build chat history block
        history_block = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 user-assistant pairs
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content']}\n"

        prompt = f"{self.SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
        if history_block:
            prompt += f"\n\nCONVERSATION HISTORY:\n{history_block}"
        prompt += f"\n\nQUESTION: {question}\n\nANSWER:"

        response = self.client.models.generate_content(model=self.CHAT_MODEL, contents=prompt)
        return {"answer": response.text.strip(), "sources": sources}
