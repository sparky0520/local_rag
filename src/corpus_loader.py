"""Corpus loader and document chunker."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    content: str
    text_for_embedding: str
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    doc_id: str
    doc_title: str
    file_path: str
    chunks: List[Chunk] = Field(default_factory=list)

class CorpusLoader:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)

    def load_corpus(self) -> List[Chunk]:
        """Loads all markdown files in corpus_dir and returns structured chunks."""
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        markdown_files = sorted(list(self.corpus_dir.glob("*.md")) + list(self.corpus_dir.glob("*.txt")))
        all_chunks: List[Chunk] = []

        for file_path in markdown_files:
            chunks = self._process_file(file_path)
            all_chunks.extend(chunks)

        return all_chunks

    def _process_file(self, file_path: Path) -> List[Chunk]:
        doc_id = file_path.stem
        raw_text = file_path.read_text(encoding="utf-8")

        # Extract top-level document title (# Title)
        title_match = re.search(r"^#\s+(.+)$", raw_text, flags=re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else doc_id.replace("_", " ").title()

        # Split on section headings (## Heading)
        raw_sections = re.split(r"\n(?=##\s+)", raw_text)
        chunks: List[Chunk] = []
        chunk_idx = 0

        for section_text in raw_sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            # Skip header-only blocks that don't have substantial content
            if section_text.startswith("# ") and "\n" not in section_text:
                continue

            # Extract section title
            sec_match = re.search(r"^##\s+(.+)$", section_text, flags=re.MULTILINE)
            sec_title = sec_match.group(1).strip() if sec_match else "Overview"

            # Clean content
            cleaned_content = section_text

            # Create rich text representation for semantic embedding
            text_for_embedding = f"Document: {doc_title}\nSection: {sec_title}\n\n{cleaned_content}"

            chunk_id = f"{doc_id}#chunk-{chunk_idx}"
            word_count = len(cleaned_content.split())
            char_count = len(cleaned_content)

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                section_title=sec_title,
                content=cleaned_content,
                text_for_embedding=text_for_embedding,
                word_count=word_count,
                char_count=char_count,
                metadata={
                    "filename": file_path.name,
                    "section_index": chunk_idx
                }
            )
            chunks.append(chunk)
            chunk_idx += 1

        return chunks
