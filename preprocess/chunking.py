"""
Multi-Granular Chunking for Query-Aware RAG System

Implements chunking strategies from docs/Approach/chunking2.md:
- OLTP: Fine-grained semantic chunks (sentence groups, 384-512 tokens)
- OLAP: Coarse-grained hierarchical chunks (sections with parent-child linking)

Best practices applied:
- Semantic sentence segmentation (not fixed-size)
- Heading-aware splitting
- Parent-child hierarchical store
- Metadata preservation (heading_path, section, source)
- Deduplication
- Small-to-big retrieval support
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Base chunk with metadata for RAG retrieval."""
    id: str
    text: str
    chunk_type: str  # "oltp" or "olap"
    
    # Metadata for retrieval and citation
    source_file: str = ""
    heading_path: List[str] = field(default_factory=list)
    section: str = ""
    chunk_index: int = 0
    
    # Hierarchical linking
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    level: str = "leaf"  # "parent", "child", or "leaf"
    
    # For deduplication
    text_hash: str = ""
    
    def __post_init__(self):
        if not self.text_hash:
            self.text_hash = hashlib.md5(self.text.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "source_file": self.source_file,
            "heading_path": self.heading_path,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "level": self.level,
            "text_hash": self.text_hash,
        }


class SentenceTokenizer:
    """
    Simple sentence tokenizer that preserves sentence boundaries.
    Falls back to regex if nltk/spacy not available.
    """
    
    def __init__(self):
        self._nltk_available = False
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            self._nltk_available = True
        except (ImportError, LookupError):
            pass
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self._nltk_available:
            import nltk
            return nltk.sent_tokenize(text)
        else:
            # Regex fallback - handles common sentence endings
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(pattern, text)
            return [s.strip() for s in sentences if s.strip()]


class OLTPChunker:
    """
    Fine-grained chunker for OLTP (factoid) queries.
    
    Strategy:
    - Sentence-level semantic splitting
    - Target size: 384-512 tokens (~300-400 words)
    - Group sentences until target size reached
    - Preserve sentence boundaries
    - Aggressive deduplication
    """
    
    def __init__(
        self,
        target_tokens: int = 450,
        min_tokens: int = 100,
        max_tokens: int = 600,
        overlap_sentences: int = 1,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.tokenizer = SentenceTokenizer()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~0.75 tokens per word)."""
        words = len(text.split())
        return int(words * 1.3)  # Conservative estimate
    
    def chunk(
        self,
        text: str,
        source_file: str = "",
        base_heading_path: List[str] = None,
    ) -> List[Chunk]:
        """
        Chunk text into fine-grained OLTP chunks.
        
        Args:
            text: Input text to chunk
            source_file: Source filename for metadata
            base_heading_path: Heading context for this text
            
        Returns:
            List of Chunk objects
        """
        if base_heading_path is None:
            base_heading_path = []
        
        sentences = self.tokenizer.tokenize(text)
        if not sentences:
            return []
        
        chunks = []
        current_sentences = []
        current_tokens = 0
        seen_hashes = set()
        chunk_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence exceeds max
            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                text_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
                
                # Deduplication check
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    chunks.append(Chunk(
                        id=f"oltp_{source_file}_{chunk_idx}",
                        text=chunk_text,
                        chunk_type="oltp",
                        source_file=source_file,
                        heading_path=base_heading_path.copy(),
                        section=base_heading_path[-1] if base_heading_path else "",
                        chunk_index=chunk_idx,
                        level="leaf",
                        text_hash=text_hash,
                    ))
                    chunk_idx += 1
                
                # Keep overlap sentences for context continuity
                if self.overlap_sentences > 0:
                    current_sentences = current_sentences[-self.overlap_sentences:]
                    current_tokens = sum(self._estimate_tokens(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_tokens = 0
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_sentences and current_tokens >= self.min_tokens:
            chunk_text = " ".join(current_sentences)
            text_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
            
            if text_hash not in seen_hashes:
                chunks.append(Chunk(
                    id=f"oltp_{source_file}_{chunk_idx}",
                    text=chunk_text,
                    chunk_type="oltp",
                    source_file=source_file,
                    heading_path=base_heading_path.copy(),
                    section=base_heading_path[-1] if base_heading_path else "",
                    chunk_index=chunk_idx,
                    level="leaf",
                    text_hash=text_hash,
                ))
        
        return chunks


class OLAPChunker:
    """
    Coarse-grained hierarchical chunker for OLAP (analytical) queries.
    
    Strategy:
    - Section/paragraph-level chunks (parents)
    - Finer sub-chunks within sections (children)
    - Preserve heading paths and document structure
    - Parent-child linking for small-to-big retrieval
    """
    
    def __init__(
        self,
        parent_target_tokens: int = 1500,
        child_target_tokens: int = 500,
        min_parent_tokens: int = 300,
    ):
        self.parent_target_tokens = parent_target_tokens
        self.child_target_tokens = child_target_tokens
        self.min_parent_tokens = min_parent_tokens
        self.oltp_chunker = OLTPChunker(
            target_tokens=child_target_tokens,
            min_tokens=100,
            max_tokens=child_target_tokens + 100,
            overlap_sentences=0,
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        words = len(text.split())
        return int(words * 1.3)
    
    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text by markdown-style headings.
        
        Returns:
            List of (heading, content) tuples
        """
        # Match markdown headings (# Header, ## Header, etc.)
        heading_pattern = r'^(#{1,6})\s+(.+?)$'
        
        sections = []
        current_heading = ""
        current_content = []
        
        for line in text.split('\n'):
            match = re.match(heading_pattern, line.strip())
            if match:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_heading, content))
                
                current_heading = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_heading, content))
        
        # If no headings found, treat entire text as one section
        if not sections and text.strip():
            sections.append(("", text.strip()))
        
        return sections
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines (paragraphs)."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk(
        self,
        text: str,
        source_file: str = "",
        base_heading_path: List[str] = None,
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Chunk text into hierarchical OLAP chunks.
        
        Args:
            text: Input text to chunk
            source_file: Source filename for metadata
            base_heading_path: Existing heading context
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        if base_heading_path is None:
            base_heading_path = []
        
        parent_chunks = []
        child_chunks = []
        parent_idx = 0
        child_idx = 0
        
        # First, split by headings
        sections = self._split_by_headings(text)
        
        for heading, content in sections:
            heading_path = base_heading_path + ([heading] if heading else [])
            
            # Check if section is large enough
            section_tokens = self._estimate_tokens(content)
            
            if section_tokens < self.min_parent_tokens:
                # Too small for parent, just create as leaf
                parent_chunks.append(Chunk(
                    id=f"olap_parent_{source_file}_{parent_idx}",
                    text=content,
                    chunk_type="olap",
                    source_file=source_file,
                    heading_path=heading_path,
                    section=heading or "Untitled",
                    chunk_index=parent_idx,
                    level="leaf",
                    child_ids=[],
                ))
                parent_idx += 1
                continue
            
            # Create parent chunk (entire section)
            parent_id = f"olap_parent_{source_file}_{parent_idx}"
            
            # Create child chunks from paragraphs
            paragraphs = self._split_by_paragraphs(content)
            section_child_ids = []
            
            # Group paragraphs into child chunks
            current_para_group = []
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                
                if current_tokens + para_tokens > self.child_target_tokens and current_para_group:
                    # Create child chunk
                    child_text = "\n\n".join(current_para_group)
                    child_id = f"olap_child_{source_file}_{child_idx}"
                    
                    child_chunks.append(Chunk(
                        id=child_id,
                        text=child_text,
                        chunk_type="olap",
                        source_file=source_file,
                        heading_path=heading_path,
                        section=heading or "Untitled",
                        chunk_index=child_idx,
                        parent_id=parent_id,
                        level="child",
                    ))
                    section_child_ids.append(child_id)
                    child_idx += 1
                    
                    current_para_group = []
                    current_tokens = 0
                
                current_para_group.append(para)
                current_tokens += para_tokens
            
            # Last child chunk
            if current_para_group:
                child_text = "\n\n".join(current_para_group)
                child_id = f"olap_child_{source_file}_{child_idx}"
                
                child_chunks.append(Chunk(
                    id=child_id,
                    text=child_text,
                    chunk_type="olap",
                    source_file=source_file,
                    heading_path=heading_path,
                    section=heading or "Untitled",
                    chunk_index=child_idx,
                    parent_id=parent_id,
                    level="child",
                ))
                section_child_ids.append(child_id)
                child_idx += 1
            
            # Create parent chunk with child references
            parent_chunks.append(Chunk(
                id=parent_id,
                text=content,
                chunk_type="olap",
                source_file=source_file,
                heading_path=heading_path,
                section=heading or "Untitled",
                chunk_index=parent_idx,
                level="parent",
                child_ids=section_child_ids,
            ))
            parent_idx += 1
        
        return parent_chunks, child_chunks


class MultiGranularChunker:
    """
    Combined chunker that produces both OLTP and OLAP chunks.
    
    This is the main entry point for the chunking pipeline.
    """
    
    def __init__(
        self,
        oltp_target_tokens: int = 450,
        olap_parent_tokens: int = 1500,
        olap_child_tokens: int = 500,
    ):
        self.oltp_chunker = OLTPChunker(target_tokens=oltp_target_tokens)
        self.olap_chunker = OLAPChunker(
            parent_target_tokens=olap_parent_tokens,
            child_target_tokens=olap_child_tokens,
        )
    
    def chunk_document(
        self,
        text: str,
        source_file: str = "",
        heading_path: List[str] = None,
    ) -> Dict[str, List[Chunk]]:
        """
        Chunk a document for both OLTP and OLAP pipelines.
        
        Args:
            text: Document text
            source_file: Source filename
            heading_path: Optional heading context
            
        Returns:
            Dict with keys: "oltp", "olap_parents", "olap_children"
        """
        if heading_path is None:
            heading_path = []
        
        # Generate OLTP chunks
        oltp_chunks = self.oltp_chunker.chunk(text, source_file, heading_path)
        
        # Generate OLAP chunks
        olap_parents, olap_children = self.olap_chunker.chunk(text, source_file, heading_path)
        
        return {
            "oltp": oltp_chunks,
            "olap_parents": olap_parents,
            "olap_children": olap_children,
        }
    
    def chunk_passages(
        self,
        passages: List[Dict],
        id_field: str = "id",
        text_field: str = "text",
    ) -> Dict[str, List[Chunk]]:
        """
        Chunk a list of passages (e.g., from MS MARCO or HotpotQA).
        
        Args:
            passages: List of passage dicts
            id_field: Key for passage ID
            text_field: Key for passage text
            
        Returns:
            Dict with all chunk types
        """
        all_oltp = []
        all_olap_parents = []
        all_olap_children = []
        
        for passage in passages:
            source = str(passage.get(id_field, "unknown"))
            text = passage.get(text_field, "")
            
            if not text:
                continue
            
            result = self.chunk_document(text, source_file=source)
            all_oltp.extend(result["oltp"])
            all_olap_parents.extend(result["olap_parents"])
            all_olap_children.extend(result["olap_children"])
        
        return {
            "oltp": all_oltp,
            "olap_parents": all_olap_parents,
            "olap_children": all_olap_children,
        }


# Convenience function
def create_chunker(
    oltp_tokens: int = 450,
    olap_parent_tokens: int = 1500,
    olap_child_tokens: int = 500,
) -> MultiGranularChunker:
    """Create a configured multi-granular chunker."""
    return MultiGranularChunker(
        oltp_target_tokens=oltp_tokens,
        olap_parent_tokens=olap_parent_tokens,
        olap_child_tokens=olap_child_tokens,
    )


# Testing
if __name__ == "__main__":
    # Sample document with headings
    sample_text = """
# Introduction

This is the introduction section. It provides an overview of the document and sets the context for the reader. The introduction is typically short but informative, giving readers a sense of what to expect.

# Methods

## Data Collection

We collected data from multiple sources including surveys, interviews, and public datasets. The data collection process took approximately three months and involved over 500 participants from diverse backgrounds.

## Analysis Approach

Our analysis used both quantitative and qualitative methods. Statistical analysis was performed using Python with pandas and scikit-learn. Qualitative coding was done using thematic analysis.

# Results

The results show significant improvements across all metrics. We observed a 25% increase in accuracy compared to baseline methods. The following subsections detail specific findings.

## Primary Findings

Our primary finding is that the new approach outperforms existing methods by a substantial margin. This was consistent across different test conditions and datasets.

## Secondary Findings

Additional analysis revealed interesting patterns in user behavior. These secondary findings suggest opportunities for future research.

# Conclusion

In conclusion, this work demonstrates the effectiveness of our proposed approach. Future work should explore additional applications and edge cases.
"""

    print("=" * 60)
    print("Multi-Granular Chunking Test")
    print("=" * 60)
    
    chunker = create_chunker()
    result = chunker.chunk_document(sample_text, source_file="sample_doc")
    
    print(f"\n📄 OLTP Chunks: {len(result['oltp'])}")
    for chunk in result['oltp'][:2]:
        print(f"  [{chunk.id}] {chunk.text[:80]}...")
        print(f"    Section: {chunk.section}, Tokens: ~{len(chunk.text.split()) * 1.3:.0f}")
    
    print(f"\n📚 OLAP Parent Chunks: {len(result['olap_parents'])}")
    for chunk in result['olap_parents'][:2]:
        print(f"  [{chunk.id}] Section: {chunk.section}")
        print(f"    Children: {len(chunk.child_ids)}, Level: {chunk.level}")
        print(f"    Text: {chunk.text[:60]}...")
    
    print(f"\n📑 OLAP Child Chunks: {len(result['olap_children'])}")
    for chunk in result['olap_children'][:2]:
        print(f"  [{chunk.id}] Parent: {chunk.parent_id}")
        print(f"    Section: {chunk.section}")
        print(f"    Text: {chunk.text[:60]}...")
    
    print("\n✅ Chunking test complete!")
