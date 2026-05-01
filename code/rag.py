from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# constants 

CHUNK_SIZE   = 400   # words per chunk
CHUNK_OVERLAP = 80   # word overlap between adjacent chunks
TOP_K_DEFAULT = 5

# Map CSV "company" values to top-level data/ subdirectory names
COMPANY_DIR_MAP: dict[str, str] = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}

# data structures 

class Chunk:
    """A single text chunk with provenance metadata."""
    __slots__ = ("text", "source_file", "company", "chunk_idx")

    def __init__(self, text: str, source_file: str, company: str, chunk_idx: int):
        self.text        = text
        self.source_file = source_file
        self.company     = company  # normalised lowercase
        self.chunk_idx   = chunk_idx

    def __repr__(self) -> str:
        return f"Chunk(company={self.company!r}, src={self.source_file!r}, idx={self.chunk_idx})"


#  corpus loader 

def _iter_markdown_files(data_root: Path):
    """Yield (path, company_key) for every .md file under data_root."""
    for company_key, sub in COMPANY_DIR_MAP.items():
        company_dir = data_root / sub
        if not company_dir.exists():
            continue
        for md_path in company_dir.rglob("*.md"):
            yield md_path, company_key


def _clean_markdown(text: str) -> str:
    """Strip Markdown syntax to leave plain prose for indexing."""
    # Remove YAML front matter
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove Markdown headings/bold/italic markers
    text = re.sub(r"[#*_~>|]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, max(1, len(words) - overlap), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_corpus(data_root: str | Path) -> list[Chunk]:
    """Load, clean, and chunk every Markdown file under data_root."""
    data_root = Path(data_root)
    all_chunks: list[Chunk] = []

    for md_path, company_key in _iter_markdown_files(data_root):
        raw  = md_path.read_text(encoding="utf-8", errors="ignore")
        text = _clean_markdown(raw)
        if not text:
            continue
        for idx, chunk_text in enumerate(_chunk_words(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            all_chunks.append(
                Chunk(
                    text        = chunk_text,
                    source_file = str(md_path.relative_to(data_root)),
                    company     = company_key,
                    chunk_idx   = idx,
                )
            )

    return all_chunks


# retriever 

class Retriever:
    """TF-IDF retriever over the loaded corpus."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        texts = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        k: int = TOP_K_DEFAULT,
    ) -> list[tuple[Chunk, float]]:
        """
        Return top-k (chunk, score) pairs.

        If company is given, first try within-company retrieval.
        If fewer than k results exist for that company, fall back to global.
        """
        q_vec = self._vectorizer.transform([query])
        scores: np.ndarray = cosine_similarity(q_vec, self._matrix).flatten()

        if company:
            company_norm = company.strip().lower()
            mapped = COMPANY_DIR_MAP.get(company_norm, company_norm)
            company_mask = np.array(
                [1.0 if c.company == mapped else 0.0 for c in self._chunks]
            )
            filtered_scores = scores * company_mask

            top_idxs = np.argsort(filtered_scores)[::-1][:k]
            results  = [(self._chunks[i], float(filtered_scores[i])) for i in top_idxs
                        if filtered_scores[i] > 0]

            if len(results) >= k:
                return results

            # Fallback: blend company-filtered + global
            global_top = np.argsort(scores)[::-1][: k * 2]
            seen = {i for i in top_idxs if filtered_scores[i] > 0}
            for i in global_top:
                if i not in seen:
                    results.append((self._chunks[i], float(scores[i])))
                    seen.add(i)
                if len(results) >= k:
                    break
            return results[:k]

        top_idxs = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in top_idxs]

    def format_context(self, results: list[tuple[Chunk, float]]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        for chunk, score in results:
            parts.append(
                f"[Source: {chunk.source_file} | company: {chunk.company} | score: {score:.3f}]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)