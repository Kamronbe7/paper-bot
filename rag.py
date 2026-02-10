# rag.py
from __future__ import annotations

import os, re, uuid
from typing import List, Dict, Any, Tuple

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# -------------------- TEXT UTILS --------------------
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def redact_pii(text: str) -> str:
    text = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", "[PHONE]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"@\w{3,}", "@[USERNAME]", text)
    return text


# -------------------- PDF EXTRACT --------------------
def extract_pdf_text(pdf_path: str) -> Tuple[str, List[str]]:
    reader = PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = redact_pii(clean_text(txt))
        if txt:
            pages_text.append(txt)
    full_text = "\n\n".join(pages_text)
    return full_text, pages_text


# -------------------- CHUNKING --------------------
def smart_chunk(text: str, size: int = 1600, overlap: int = 250) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        part = text[i:i + size]
        part = part.strip()
        if len(part) > 300:
            chunks.append(part)
        i += size - overlap
    return chunks


# -------------------- RAG CORE --------------------
class PaperRAG:
    def __init__(self, persist_dir: str = "chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._openai = None
        if self.openai_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=self.openai_key)
            except Exception:
                self._openai = None

    def _collection(self, paper_id: str):
        return self.client.get_or_create_collection(
            name=f"paper_{paper_id}",
            metadata={"hnsw:space": "cosine"}
        )

    # -------------------- INGEST --------------------
    def ingest_pdf(self, pdf_path: str, filename: str) -> str:
        paper_id = uuid.uuid4().hex[:12]
        full_text, _ = extract_pdf_text(pdf_path)
        chunks = smart_chunk(full_text)

        col = self._collection(paper_id)

        ids = [uuid.uuid4().hex for _ in chunks]
        metas = [
            {
                "filename": filename,
                "chunk_no": i + 1,          # source sifatida ishlatamiz
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]

        embeddings = self.embedder.encode(chunks, normalize_embeddings=True).tolist()
        col.add(documents=chunks, metadatas=metas, embeddings=embeddings, ids=ids)

        # full_text cache (summary uchun)
        with open(f"paper_{paper_id}.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

        return paper_id

    def _load_full_text(self, paper_id: str) -> str:
        path = f"paper_{paper_id}.txt"
        return open(path, encoding="utf-8").read() if os.path.exists(path) else ""

    # -------------------- SUMMARY --------------------
    def summarize(self, paper_id: str, lang: str = "uz") -> str:
        text = self._load_full_text(paper_id)
        if not text:
            return "Matn topilmadi."

        # oddiy extractive (OpenAI bo'lmasa)
        if not self._openai:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
            top = sorted(sentences, key=len, reverse=True)[:8]
            if lang == "ru":
                return "Кратко (TL;DR):\n- " + "\n- ".join(top)
            if lang == "en":
                return "TL;DR:\n- " + "\n- ".join(top)
            return "TL;DR:\n- " + "\n- ".join(top)

        # OpenAI bo'lsa — tilda formatli summary
        sys_map = {
            "uz": "Siz ilmiy maqola summarizatorisiz. Javobni o‘zbek tilida bering.",
            "ru": "Вы — научный суммаризатор. Отвечайте на русском языке.",
            "en": "You are a scientific paper summarizer. Answer in English."
        }
        sys_msg = sys_map.get(lang, sys_map["uz"])

        prompt_map = {
            "uz": (
                "Quyidagi matnni ilmiy uslubda qisqacha xulosa qiling.\n"
                "Format:\n"
                "1) TL;DR (5-7 gap)\n"
                "2) Asosiy hissalar (5-8 punkt)\n"
                "3) Metod (3-6 gap)\n"
                "4) Natijalar (3-6 gap)\n"
                "5) Cheklovlar/Future work (2-5 punkt)\n"
                "6) Keywords (5-10 ta)\n\n"
                "MATN:\n"
            ),
            "ru": (
                "Сделай краткое научное резюме текста.\n"
                "Формат:\n"
                "1) TL;DR (5-7 предложений)\n"
                "2) Основные вклады (5-8 пунктов)\n"
                "3) Метод (3-6 предложений)\n"
                "4) Результаты (3-6 предложений)\n"
                "5) Ограничения/Future work (2-5 пунктов)\n"
                "6) Keywords (5-10)\n\n"
                "ТЕКСТ:\n"
            ),
            "en": (
                "Write a concise scientific summary of the text.\n"
                "Format:\n"
                "1) TL;DR (5-7 sentences)\n"
                "2) Main contributions (5-8 bullets)\n"
                "3) Method (3-6 sentences)\n"
                "4) Results (3-6 sentences)\n"
                "5) Limitations/Future work (2-5 bullets)\n"
                "6) Keywords (5-10)\n\n"
                "TEXT:\n"
            ),
        }

        text_cut = text[:18000]
        try:
            r = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt_map.get(lang, prompt_map["uz"]) + text_cut}
                ],
                temperature=0.2,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            # fallback oddiy
            return self.summarize(paper_id, lang="en" if lang == "en" else "uz") if not self._openai else "Xatolik."

    # -------------------- HYBRID SEARCH --------------------
    def _keyword_search(self, question: str, docs: List[str], top_n: int = 4) -> List[int]:
        """
        docs ichida keyword overlap bo'yicha eng mos indekslarni qaytaradi
        """
        q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3]
        if not q_words:
            return []
        scores = []
        for i, d in enumerate(docs):
            dl = d.lower()
            score = sum(1 for w in q_words if w in dl)
            scores.append((score, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        best = [i for s, i in scores if s > 0][:top_n]
        return best

    def _rerank(self, question: str, docs: List[str], metas: List[Dict[str, Any]], top_n: int = 4):
        q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3]
        scored = []
        for d, m in zip(docs, metas):
            dl = d.lower()
            overlap = sum(1 for w in q_words if w in dl)
            scored.append((overlap, d, m))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:top_n]

    # -------------------- ASK (with sources + language) --------------------
    def ask(self, paper_id: str, question: str, lang: str = "uz") -> Dict[str, Any]:
        col = self._collection(paper_id)

        q_emb = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        res = col.query(
            query_embeddings=[q_emb],
            n_results=8,
            include=["documents", "metadatas", "distances"]
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        if not docs:
            msg = {
                "uz": "Maqolada bu haqida ma’lumot topilmadi.",
                "ru": "В тексте не найдено точной информации по этому вопросу.",
                "en": "No clear information was found in the document."
            }.get(lang, "Maqolada bu haqida ma’lumot topilmadi.")
            return {"answer": msg, "sources": []}

        # 1) keyword fallback (faqat indekslar)
        idxs = self._keyword_search(question, docs, top_n=4)
        if idxs:
            docs = [docs[i] for i in idxs]
            metas = [metas[i] for i in idxs]

        # 2) rerank
        ranked = self._rerank(question, docs, metas, top_n=4)

        # context + sources
        context_parts = []
        sources = []
        for i, (_, d, m) in enumerate(ranked, start=1):
            context_parts.append(f"[{i}] {d}")
            sources.append({
                "ref": i,
                "filename": m.get("filename"),
                "chunk_no": m.get("chunk_no"),
                "total_chunks": m.get("total_chunks"),
                "snippet": d[:240].replace("\n", " ") + ("..." if len(d) > 240 else "")
            })

        context = "\n\n".join(context_parts)

        if not self._openai:
            # OpenAI bo'lmasa ham sources bilan qaytaradi
            return {
                "answer": ranked[0][1][:1200],
                "sources": sources
            }

        sys_map = {
            "uz": "Siz evidence-based assistentsiz. Faqat kontekstdan foydalaning va o‘zbekcha javob bering.",
            "ru": "Вы evidence-based ассистент. Используйте только контекст и отвечайте по-русски.",
            "en": "You are an evidence-based assistant. Use only the provided context and answer in English."
        }
        sys_msg = sys_map.get(lang, sys_map["uz"])

        prompt = (
            "Qoidalar:\n"
            "- Faqat KONTEKSTga tayangan holda javob ber.\n"
            "- Agar KONTEKSTda aniq bo‘lmasa, shuni ayt.\n"
            "- Javob oxirida ishlatilgan reference raqamlarini yoz (masalan: Sources: [1][3]).\n\n"
            f"KONTEKST:\n{context}\n\n"
            f"SAVOL:\n{question}"
        )

        r = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        answer = r.choices[0].message.content.strip()

        return {"answer": answer, "sources": sources}
