"""대출 정책 문서(loan_policy.md) 기반 RAG.

섹션(## 헤더) 단위로 청크를 나누고 Gemini 임베딩으로 벡터화한 뒤,
질의(신청자 상황 요약)와 코사인 유사도가 높은 조항을 top-k로 검색한다.
임베딩은 정책 문서가 바뀌지 않는 한 재사용하도록 로컬에 캐싱한다.
"""

import json
import re
from pathlib import Path

import numpy as np

from gemini_client import EMBEDDING_MODEL, get_client, with_rate_limit_retry

AGENT_DIR = Path(__file__).resolve().parent
POLICY_PATH = AGENT_DIR / "policy_docs" / "loan_policy.md"
CACHE_PATH = AGENT_DIR / "policy_docs" / ".embedding_cache.json"


def _split_into_chunks(text: str) -> list[dict]:
    """'## ' 로 시작하는 섹션 단위로 문서를 분할한다."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("#") and not section.startswith("##"):
            if not section.startswith("##"):
                continue
        title_match = re.match(r"^##\s*(.+)$", section, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else section[:30]
        chunks.append({"title": title, "text": section})
    return chunks


@with_rate_limit_retry
def _embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return np.array([e.values for e in resp.embeddings], dtype=np.float32)


class PolicyRAG:
    def __init__(self):
        raw = POLICY_PATH.read_text(encoding="utf-8")
        self.chunks = _split_into_chunks(raw)
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        chunk_texts = [c["text"] for c in self.chunks]
        if CACHE_PATH.exists():
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cache.get("chunk_texts") == chunk_texts:
                return np.array(cache["embeddings"], dtype=np.float32)

        embeddings = _embed_texts(chunk_texts)
        CACHE_PATH.write_text(
            json.dumps({"chunk_texts": chunk_texts, "embeddings": embeddings.tolist()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return embeddings

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = _embed_texts([query])[0]
        sims = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {"title": self.chunks[i]["title"], "text": self.chunks[i]["text"], "score": float(sims[i])}
            for i in top_idx
        ]


if __name__ == "__main__":
    rag = PolicyRAG()
    print(f"청크 {len(rag.chunks)}개 로드/임베딩 완료")
    results = rag.retrieve("조건부승인 등급인데 최근 매출이 회복되고 있는 신청자를 승인으로 올릴 수 있나?", top_k=2)
    for r in results:
        print(f"\n[{r['score']:.3f}] {r['title']}")
        print(r["text"][:200])
