"""
Plagiarism Detection Module for Speech Analysis

This module provides comprehensive plagiarism detection capabilities including:
- Text normalization and chunking
- Web search for similar content (via Tavily)
- Zero-shot classification (via HuggingFace)
- Semantic similarity detection (embeddings)
- Lexical similarity detection (n-gram, Jaccard)
- Report generation and risk scoring
"""

import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import asyncio

# For embeddings
try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

# For HuggingFace classifier
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


@dataclass
class SpeechChunk:
    """Represents a chunk of the suspect speech document"""
    chunk_id: str
    speech_id: str
    text: str
    paragraph_index: int
    sentence_indices: List[int]
    normalized_text: str
    word_count: int
    char_count: int


@dataclass
class SourceMatch:
    """Represents a potential plagiarism match from a source"""
    source_url: str
    source_title: str
    source_date: Optional[str]
    source_speaker: Optional[str]
    source_chunk: str
    similarity_semantic: float  # 0-1 cosine similarity
    similarity_lexical: float   # 0-1 Jaccard/n-gram score
    overlap_ratio: float        # What % of suspect chunk overlaps
    match_type: str            # "direct_quote", "close_paraphrase", "common_phrase", "unique"
    classification_label: Optional[str]  # From HF classifier
    evidence_snippets: List[str]


@dataclass
class ChunkAnalysis:
    """Analysis results for a single speech chunk"""
    chunk: SpeechChunk
    top_matches: List[SourceMatch]
    risk_score: float  # 0-1 plagiarism risk
    risk_level: str    # "high", "medium", "low", "none"
    is_boilerplate: bool
    explanation: str


class TextNormalizer:
    """Handles text normalization and chunking"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence boundaries
        text = re.sub(r'[^\w\s\.\,\;\:\?\!]', '', text)
        # Lowercase for comparison
        text = text.lower().strip()
        return text
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """Remove common header/footer patterns"""
        # Remove page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove common footer patterns
        text = re.sub(r'(page \d+ of \d+|confidential|draft)', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    @staticmethod
    def segment_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    @staticmethod
    def segment_into_sentences(paragraph: str) -> List[str]:
        """Split paragraph into sentences"""
        # Simple sentence splitting (could use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    @staticmethod
    def create_chunks(
        text: str,
        speech_id: str = None,
        chunk_by: str = "paragraph",  # "paragraph" or "sentence"
        verbose: bool = False
    ) -> List[SpeechChunk]:
        """
        Create speech chunks from text
        
        Args:
            text: The speech text
            speech_id: Unique identifier for this speech
            chunk_by: How to chunk ("paragraph" or "sentence")
            verbose: Print progress information
        """
        if not speech_id:
            speech_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Preprocess
        text = TextNormalizer.remove_headers_footers(text)
        paragraphs = TextNormalizer.segment_into_paragraphs(text)
        
        if verbose:
            print(f"  Text preprocessing: {len(paragraphs)} paragraphs found")
        
        chunks = []
        chunk_counter = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            if chunk_by == "paragraph":
                # One chunk per paragraph
                normalized = TextNormalizer.normalize_text(paragraph)
                chunk_id = f"{speech_id}_chunk_{chunk_counter:03d}"
                
                chunk = SpeechChunk(
                    chunk_id=chunk_id,
                    speech_id=speech_id,
                    text=paragraph,
                    paragraph_index=para_idx,
                    sentence_indices=[],
                    normalized_text=normalized,
                    word_count=len(paragraph.split()),
                    char_count=len(paragraph)
                )
                chunks.append(chunk)
                chunk_counter += 1
                
            else:  # chunk_by == "sentence"
                sentences = TextNormalizer.segment_into_sentences(paragraph)
                for sent_idx, sentence in enumerate(sentences):
                    normalized = TextNormalizer.normalize_text(sentence)
                    chunk_id = f"{speech_id}_chunk_{chunk_counter:03d}"
                    
                    chunk = SpeechChunk(
                        chunk_id=chunk_id,
                        speech_id=speech_id,
                        text=sentence,
                        paragraph_index=para_idx,
                        sentence_indices=[sent_idx],
                        normalized_text=normalized,
                        word_count=len(sentence.split()),
                        char_count=len(sentence)
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
        
        return chunks


class SearchQueryGenerator:
    """Generates search queries from speech chunks"""
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 5) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction (could use RAKE, YAKE, or TF-IDF)
        
        # Common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Filter stopwords
        words = [w for w in words if w not in stopwords]
        
        # Count frequency
        word_counts = Counter(words)
        
        # Return top-k
        return [word for word, _ in word_counts.most_common(top_k)]
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract named entities (simple pattern-based)"""
        entities = []
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.extend(numbers)
        
        # Dates
        dates = re.findall(r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text)
        entities.extend(dates)
        
        return list(set(entities))
    
    @staticmethod
    def generate_search_queries(chunk: SpeechChunk, num_queries: int = 2) -> List[str]:
        """
        Generate search queries optimized for finding similar content
        
        Args:
            chunk: The speech chunk to generate queries for
            num_queries: Number of query variations to create
        """
        queries = []
        
        # Query 1: Keywords + entities
        keywords = SearchQueryGenerator.extract_keywords(chunk.text, top_k=5)
        entities = SearchQueryGenerator.extract_entities(chunk.text)
        
        if keywords:
            query1 = " ".join(keywords[:3])
            if entities:
                query1 += " " + " ".join(entities[:2])
            queries.append(query1.strip())
        
        # Query 2: First significant sentence (30-100 chars)
        sentences = TextNormalizer.segment_into_sentences(chunk.text)
        for sent in sentences:
            if 30 <= len(sent) <= 100:
                queries.append(sent)
                break
        
        # Query 3: Extract key phrase (noun phrases)
        # Simple: take sequences of 3-5 words that don't start with common words
        words = chunk.text.split()
        if len(words) >= 5:
            for i in range(len(words) - 4):
                phrase = " ".join(words[i:i+5])
                if not phrase.lower().startswith(('the ', 'a ', 'an ', 'and ', 'but ')):
                    queries.append(phrase)
                    break
        
        # Limit to num_queries
        return queries[:num_queries]


class SimilarityDetector:
    """Handles semantic and lexical similarity detection"""
    
    @staticmethod
    def compute_ngrams(text: str, n: int = 3) -> set:
        """Compute n-grams from text"""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
        """Compute n-gram overlap ratio"""
        ngrams1 = SimilarityDetector.compute_ngrams(text1, n)
        ngrams2 = SimilarityDetector.compute_ngrams(text2, n)
        
        if not ngrams1:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        return len(intersection) / len(ngrams1)
    
    @staticmethod
    def containment_score(suspect: str, source: str, n: int = 3) -> float:
        """
        Compute containment: what fraction of suspect's n-grams appear in source
        This is asymmetric and good for plagiarism detection
        """
        suspect_ngrams = SimilarityDetector.compute_ngrams(suspect, n)
        source_ngrams = SimilarityDetector.compute_ngrams(source, n)
        
        if not suspect_ngrams:
            return 0.0
        
        contained = suspect_ngrams.intersection(source_ngrams)
        return len(contained) / len(suspect_ngrams)
    
    @staticmethod
    def lexical_similarity(text1: str, text2: str) -> Dict[str, float]:
        """
        Compute multiple lexical similarity metrics
        Returns dict with Jaccard, 3-gram, 5-gram scores
        """
        return {
            "jaccard": SimilarityDetector.jaccard_similarity(text1, text2),
            "trigram_overlap": SimilarityDetector.ngram_overlap(text1, text2, n=3),
            "fivegram_overlap": SimilarityDetector.ngram_overlap(text1, text2, n=5),
            "containment_3gram": SimilarityDetector.containment_score(text1, text2, n=3)
        }
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class PlagiarismClassifier:
    """Classifies match type based on similarity scores"""
    
    @staticmethod
    def classify_match(
        semantic_sim: float,
        lexical_sim: float,
        containment: float
    ) -> str:
        """
        Classify the type of match
        
        Returns: "direct_quote", "close_paraphrase", "common_phrase", "unique"
        """
        # Direct quote: high lexical similarity
        if lexical_sim > 0.7 and containment > 0.6:
            return "direct_quote"
        
        # Close paraphrase: high semantic, moderate lexical
        if semantic_sim > 0.8 and lexical_sim > 0.4:
            return "close_paraphrase"
        
        # Common phrase: moderate similarities
        if semantic_sim > 0.6 or lexical_sim > 0.5:
            return "common_phrase"
        
        # Unique content
        return "unique"
    
    @staticmethod
    def compute_risk_score(
        match_type: str,
        semantic_sim: float,
        lexical_sim: float,
        chunk_length: int,
        is_boilerplate: bool = False
    ) -> float:
        """
        Compute plagiarism risk score (0-1)
        
        Factors:
        - Match type (direct quote = highest risk)
        - Similarity scores
        - Chunk length (longer matches = higher risk)
        - Boilerplate discount
        """
        # Base score from match type
        base_scores = {
            "direct_quote": 0.9,
            "close_paraphrase": 0.7,
            "common_phrase": 0.3,
            "unique": 0.0
        }
        base = base_scores.get(match_type, 0.0)
        
        # Weighted similarity
        sim_score = (semantic_sim * 0.6 + lexical_sim * 0.4)
        
        # Length weight (longer chunks = more significant)
        length_weight = min(1.0, chunk_length / 100)
        
        # Combine
        risk = base * 0.5 + sim_score * 0.3 + length_weight * 0.2
        
        # Boilerplate discount
        if is_boilerplate:
            risk *= 0.3
        
        return min(1.0, max(0.0, risk))
    
    @staticmethod
    def detect_boilerplate(text: str) -> bool:
        """
        Detect if text is likely boilerplate/standard phrasing
        
        Common patterns:
        - "Thank you for"
        - "I am pleased to"
        - "Ladies and gentlemen"
        - Standard greetings/closings
        """
        boilerplate_patterns = [
            r'thank you (?:for|very much)',
            r'(?:good|great) (?:morning|afternoon|evening)',
            r'ladies and gentlemen',
            r'i am (?:pleased|honored|delighted) to',
            r'it is (?:a pleasure|an honor)',
            r'on behalf of',
            r'in conclusion',
            r'to sum(?:marize)? up',
        ]
        
        text_lower = text.lower()
        
        for pattern in boilerplate_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Also check for very short, generic statements
        if len(text.split()) < 10:
            return True
        
        return False


class PlagiarismChecker:
    """Main plagiarism detection orchestrator"""
    
    def __init__(
        self,
        azure_client: Optional[Any] = None,
        tavily_client: Optional[Any] = None,
        use_hf_classifier: bool = False
    ):
        """
        Initialize plagiarism checker
        
        Args:
            azure_client: Azure OpenAI client for embeddings
            tavily_client: Tavily client for web search
            use_hf_classifier: Whether to use HuggingFace classifier
        """
        self.azure_client = azure_client
        self.tavily_client = tavily_client
        self.use_hf_classifier = use_hf_classifier
        
        # Initialize HF classifier if requested
        self.hf_classifier = None
        if use_hf_classifier and pipeline:
            try:
                # Use zero-shot classification for topic labeling
                self.hf_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except Exception as e:
                print(f"Warning: Could not load HF classifier: {e}")
    
    async def analyze_speech(
        self,
        speech_text: str,
        speech_metadata: Optional[Dict[str, Any]] = None,
        chunk_by: str = "paragraph"
    ) -> Dict[str, Any]:
        """
        Main entry point: Analyze a speech for plagiarism
        
        Args:
            speech_text: The full speech text
            speech_metadata: Optional metadata (speaker, institution, date)
            chunk_by: How to chunk text ("paragraph" or "sentence")
        
        Returns:
            Comprehensive plagiarism analysis report
        """
        print("\n" + "="*70)
        print("PLAGIARISM DETECTION ANALYSIS")
        print("="*70)
        
        # Step 0: Ingest and chunk the speech
        print("\n[Step 0] Ingesting and chunking speech...")
        speech_id = speech_metadata.get('id') if speech_metadata else None
        chunks = TextNormalizer.create_chunks(speech_text, speech_id, chunk_by, verbose=True)
        print(f"  Created {len(chunks)} chunks")
        total_words = sum(c.word_count for c in chunks)
        print(f"  Total words: {total_words}")
        
        # Step 1: Search for similar content via Tavily
        print("\n[Step 1] Searching web for similar content...")
        search_results = await self._search_similar_content(chunks)
        print(f"  Found {len(search_results)} potential source documents")
        
        # Step 2: Chunk and preprocess sources
        print("\n[Step 2] Processing source documents...")
        source_chunks = self._process_sources(search_results)
        print(f"  Created {len(source_chunks)} source chunks")
        
        # Step 3: Classify chunks (optional HF classifier)
        print("\n[Step 3] Classifying chunks...")
        if self.hf_classifier:
            await self._classify_chunks(chunks, source_chunks)
        
        # Step 4: Detect plagiarism matches
        print("\n[Step 4] Detecting plagiarism matches...")
        chunk_analyses = await self._detect_matches(chunks, source_chunks)
        print(f"  Analyzed {len(chunk_analyses)} chunks")
        
        # Step 5: Generate explanations via Azure OpenAI
        print("\n[Step 5] Generating explanations...")
        if self.azure_client:
            chunk_analyses = await self._generate_explanations(chunk_analyses)
        
        # Step 6: Aggregate results
        print("\n[Step 6] Aggregating results...")
        overall_report = self._aggregate_results(
            chunks=chunks,
            chunk_analyses=chunk_analyses,
            speech_metadata=speech_metadata
        )
        
        print("\n✓ Plagiarism analysis complete")
        print(f"\nSUMMARY:")
        print(f"  • Total chunks analyzed: {overall_report['statistics']['total_chunks']}")
        print(f"  • Sources found: {len(overall_report['top_sources'])}")
        print(f"  • High risk chunks: {overall_report['statistics']['high_risk_chunks']}")
        print(f"  • Medium risk chunks: {overall_report['statistics']['medium_risk_chunks']}")
        print(f"  • Overall risk: {overall_report['overall_risk_level'].upper()} ({overall_report['overall_risk_score']:.3f})")
        return overall_report
    
    async def _search_similar_content(
        self,
        chunks: List[SpeechChunk]
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content using Tavily (optimized strategy)
        
        OPTIMIZATION: Instead of 2 queries per chunk (20 total), we now:
        1. Sample 3-5 representative chunks
        2. Generate 2-3 strategic queries covering main themes
        3. Use higher max_results per query (10 instead of 5)
        4. Deduplicate immediately during collection
        
        Expected: ~85% reduction in API calls (20 → 3 searches)
        
        Returns list of source documents
        """
        all_results = []
        seen_urls = set()  # Immediate deduplication
        
        if not self.tavily_client:
            print("  No Tavily client - skipping web search")
            return []
        
        # Sample fewer but representative chunks
        sample_size = min(3, len(chunks))
        sampled_chunks = chunks[:sample_size]
        print(f"  Sampling {sample_size} representative chunks...")
        
        # Aggregate content from sampled chunks
        aggregated_text = " ".join([chunk.text for chunk in sampled_chunks])
        
        # Generate strategic queries (2-3 queries total, not per chunk)
        print(f"  Generating strategic search queries...")
        strategic_queries = []
        
        # Query 1: Main topic with key entities
        keywords = SearchQueryGenerator.extract_keywords(aggregated_text, top_k=5)
        entities = SearchQueryGenerator.extract_entities(aggregated_text)
        
        if keywords:
            main_query = " ".join(keywords[:3])
            if entities:
                main_query += f" {entities[0]}"
            strategic_queries.append(main_query)
            print(f"    Query 1 (main topic): '{main_query[:60]}{'...' if len(main_query) > 60 else ''}'")
        
        # Query 2: Most distinctive phrase (exact match)
        # Find the longest substantive sentence from first chunk
        if sampled_chunks:
            sentences = sampled_chunks[0].text.split('.')
            substantive = [s.strip() for s in sentences if len(s.split()) > 8 and len(s.split()) < 20]
            if substantive:
                exact_phrase = f'"{substantive[0]}"'
                strategic_queries.append(exact_phrase)
                print(f"    Query 2 (exact phrase): '{exact_phrase[:60]}{'...' if len(exact_phrase) > 60 else ''}'")
        
        # Query 3: Alternative formulation (if we have enough keywords)
        if len(keywords) >= 4:
            alt_query = " ".join(keywords[2:5])
            if len(entities) > 1:
                alt_query += f" {entities[1]}"
            strategic_queries.append(alt_query)
            print(f"    Query 3 (alternative): '{alt_query[:60]}{'...' if len(alt_query) > 60 else ''}'")
        
        # Execute searches with higher max_results
        total_queries = len(strategic_queries)
        print(f"  Executing {total_queries} strategic searches...")
        
        for i, query in enumerate(strategic_queries, 1):
            try:
                print(f"    Search {i}/{total_queries}: ", end="", flush=True)
                results = await self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=10,  # Increased from 5 to compensate for fewer queries
                    include_raw_content=True
                )
                found_count = len(results.get('results', []))
                print(f"found {found_count} results", end="")
                
                # Extract relevant sources with immediate deduplication
                new_sources = 0
                for result in results.get('results', []):
                    url = result.get('url')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            'url': url,
                            'title': result.get('title'),
                            'content': result.get('raw_content', result.get('content')),
                            'score': result.get('score', 0),
                            'published_date': result.get('published_date')
                        })
                        new_sources += 1
                
                print(f" ({new_sources} unique)")
                
            except Exception as e:
                print(f"⚠️ Failed: {str(e)[:60]}")
        
        print(f"  ✓ Collected {len(all_results)} unique sources from {total_queries} searches")
        print(f"    (Previous strategy would have used ~{sample_size * 2} searches)")
        
        return all_results
    
    def _process_sources(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk and normalize source documents
        
        Returns list of source chunk dicts
        """
        source_chunks = []
        print(f"  Processing {len(search_results)} source documents...")
        
        for idx, source in enumerate(search_results, 1):
            content = source.get('content', '')
            if not content or len(content) < 50:
                print(f"    Source {idx}: Skipped (insufficient content)")
                continue
            
            # Create chunks from source
            chunks = TextNormalizer.create_chunks(
                content,
                speech_id=hashlib.md5(source['url'].encode()).hexdigest()[:8],
                chunk_by="paragraph"
            )
            print(f"    Source {idx}: '{source['title'][:50]}...' → {len(chunks)} chunks")
            
            # Add source metadata to each chunk
            for chunk in chunks:
                source_chunks.append({
                    'chunk': chunk,
                    'source_url': source['url'],
                    'source_title': source['title'],
                    'source_date': source.get('published_date'),
                    'source_score': source.get('score', 0)
                })
        
        return source_chunks
    
    async def _classify_chunks(
        self,
        speech_chunks: List[SpeechChunk],
        source_chunks: List[Dict[str, Any]]
    ):
        """
        Classify chunks using HuggingFace zero-shot classifier
        
        Labels: policy_signal, horizon, topic, etc.
        """
        if not self.hf_classifier:
            print("  HuggingFace classifier not available - skipping classification")
            return
        
        print(f"  Classifying {len(speech_chunks)} speech chunks...")
        
        candidate_labels = [
            "monetary policy",
            "economic outlook",
            "financial stability",
            "regulation",
            "market commentary",
            "procedural",
            "greeting",
            "conclusion"
        ]
        
        # Classify speech chunks
        for idx, chunk in enumerate(speech_chunks, 1):
            try:
                result = self.hf_classifier(
                    chunk.text,
                    candidate_labels,
                    multi_label=False
                )
                chunk.classification = result['labels'][0]
                chunk.classification_score = result['scores'][0]
                if idx <= 3 or idx % 5 == 0:  # Print first 3 and every 5th
                    print(f"    Chunk {idx}: {chunk.classification} ({chunk.classification_score:.2f})")
            except Exception as e:
                chunk.classification = "unknown"
                chunk.classification_score = 0.0
        
        # Classify source chunks
        print(f"  Classifying {len(source_chunks)} source chunks...")
        for source_chunk in source_chunks:
            try:
                result = self.hf_classifier(
                    source_chunk['chunk'].text,
                    candidate_labels,
                    multi_label=False
                )
                source_chunk['classification'] = result['labels'][0]
                source_chunk['classification_score'] = result['scores'][0]
            except Exception as e:
                source_chunk['classification'] = "unknown"
                source_chunk['classification_score'] = 0.0
    
    async def _detect_matches(
        self,
        speech_chunks: List[SpeechChunk],
        source_chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[ChunkAnalysis]:
        """
        Detect plagiarism matches for each speech chunk
        
        Two-stage:
        1. Candidate retrieval (semantic similarity via embeddings)
        2. Verification (lexical overlap)
        """
        chunk_analyses = []
        
        # Get embeddings if Azure client available
        embeddings_map = {}
        if self.azure_client:
            await self._compute_embeddings(speech_chunks, source_chunks, embeddings_map)
        else:
            print(f"  No Azure client - using lexical similarity only")
        
        print(f"\n  Analyzing {len(speech_chunks)} speech chunks against {len(source_chunks)} source chunks...")
        for chunk_idx, speech_chunk in enumerate(speech_chunks, 1):
            print(f"\n  [{chunk_idx}/{len(speech_chunks)}] Analyzing chunk: {speech_chunk.chunk_id}")
            print(f"      Words: {speech_chunk.word_count}, Text: '{speech_chunk.text[:50]}...'")
            # Stage 1: Candidate retrieval
            candidates = []
            print(f"      Stage 1: Candidate retrieval...", end="", flush=True)
            
            if embeddings_map:
                # Use semantic similarity
                speech_emb = embeddings_map.get(speech_chunk.chunk_id)
                if speech_emb is not None:
                    for source_chunk in source_chunks:
                        source_emb = embeddings_map.get(source_chunk['chunk'].chunk_id)
                        if source_emb is not None:
                            semantic_sim = SimilarityDetector.cosine_similarity(
                                speech_emb, source_emb
                            )
                            if semantic_sim > 0.5:  # Threshold
                                candidates.append((source_chunk, semantic_sim))
            else:
                # Fallback: use lexical similarity for candidate retrieval
                for source_chunk in source_chunks:
                    jaccard = SimilarityDetector.jaccard_similarity(
                        speech_chunk.normalized_text,
                        source_chunk['chunk'].normalized_text
                    )
                    if jaccard > 0.3:
                        candidates.append((source_chunk, jaccard))
            
            # Sort by similarity and take top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:top_k]
            print(f" found {len(candidates)} candidates")
            
            # Stage 2: Verification with lexical analysis
            matches = []
            if candidates:
                print(f"      Stage 2: Verifying {len(candidates)} candidates...")
            for cand_idx, (source_chunk, semantic_sim) in enumerate(candidates, 1):
                # Compute lexical metrics
                lex_metrics = SimilarityDetector.lexical_similarity(
                    speech_chunk.normalized_text,
                    source_chunk['chunk'].normalized_text
                )
                
                # Overall lexical score (weighted average)
                lexical_sim = (
                    lex_metrics['jaccard'] * 0.3 +
                    lex_metrics['trigram_overlap'] * 0.4 +
                    lex_metrics['fivegram_overlap'] * 0.3
                )
                containment = lex_metrics['containment_3gram']
                
                # Classify match type
                match_type = PlagiarismClassifier.classify_match(
                    semantic_sim, lexical_sim, containment
                )
                
                print(f"        Candidate {cand_idx}: Semantic={semantic_sim:.3f}, Lexical={lexical_sim:.3f}, Type={match_type}")
                
                # Create match object
                match = SourceMatch(
                    source_url=source_chunk['source_url'],
                    source_title=source_chunk['source_title'],
                    source_date=source_chunk.get('source_date'),
                    source_speaker=None,  # Could extract if available
                    source_chunk=source_chunk['chunk'].text,
                    similarity_semantic=semantic_sim,
                    similarity_lexical=lexical_sim,
                    overlap_ratio=containment,
                    match_type=match_type,
                    classification_label=source_chunk.get('classification'),
                    evidence_snippets=[]  # Could extract matching n-grams
                )
                matches.append(match)
            
            # Compute chunk risk
            is_boilerplate = PlagiarismClassifier.detect_boilerplate(speech_chunk.text)
            
            if matches:
                best_match = matches[0]
                risk_score = PlagiarismClassifier.compute_risk_score(
                    match_type=best_match.match_type,
                    semantic_sim=best_match.similarity_semantic,
                    lexical_sim=best_match.similarity_lexical,
                    chunk_length=speech_chunk.word_count,
                    is_boilerplate=is_boilerplate
                )
            else:
                risk_score = 0.0
            
            # Determine risk level
            if risk_score > 0.7:
                risk_level = "high"
            elif risk_score > 0.4:
                risk_level = "medium"
            elif risk_score > 0.15:
                risk_level = "low"
            else:
                risk_level = "none"
            
            print(f"      Risk: {risk_level.upper()} (score={risk_score:.3f}){' [BOILERPLATE]' if is_boilerplate else ''}")
            
            # Create analysis
            analysis = ChunkAnalysis(
                chunk=speech_chunk,
                top_matches=matches,
                risk_score=risk_score,
                risk_level=risk_level,
                is_boilerplate=is_boilerplate,
                explanation=""  # Will be filled by LLM
            )
            chunk_analyses.append(analysis)
        
        return chunk_analyses
    
    async def _compute_embeddings(
        self,
        speech_chunks: List[SpeechChunk],
        source_chunks: List[Dict[str, Any]],
        embeddings_map: Dict[str, np.ndarray]
    ):
        """Compute embeddings for all chunks using Azure OpenAI"""
        if not self.azure_client:
            return
        
        print(f"  Computing embeddings for {len(speech_chunks)} speech chunks + {len(source_chunks)} source chunks...")
        
        # Collect all texts
        texts_to_embed = []
        ids = []
        
        for chunk in speech_chunks:
            texts_to_embed.append(chunk.text)
            ids.append(chunk.chunk_id)
        
        for source_chunk in source_chunks:
            texts_to_embed.append(source_chunk['chunk'].text)
            ids.append(source_chunk['chunk'].chunk_id)
        
        # Batch embed (Azure supports up to 16 at a time for text-embedding-3-small)
        batch_size = 16
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        print(f"  Processing {total_batches} embedding batches...")
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                print(f"    Batch {batch_num}/{total_batches}: Embedding {len(batch_texts)} chunks...", end="", flush=True)
                response = await self.azure_client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small"
                )
                
                for j, embedding_obj in enumerate(response.data):
                    embeddings_map[batch_ids[j]] = np.array(embedding_obj.embedding)
                print(" ✓")
            
            except Exception as e:
                print(f" ✗")
                print(f"      ⚠️ Batch failed: {str(e)[:60]}")
    
    async def _generate_explanations(
        self,
        chunk_analyses: List[ChunkAnalysis]
    ) -> List[ChunkAnalysis]:
        """Generate human-readable explanations using Azure OpenAI"""
        # Only explain chunks with medium/high risk
        flagged_chunks = [a for a in chunk_analyses if a.risk_level in ["medium", "high"]]
        if flagged_chunks:
            print(f"  Generating explanations for {len(flagged_chunks)} flagged chunks...")
        
        for idx, analysis in enumerate(chunk_analyses, 1):
            if analysis.risk_level in ["medium", "high"] and analysis.top_matches:
                best_match = analysis.top_matches[0]
                
                prompt = f"""Analyze this potential plagiarism match:

SUSPECT TEXT:
{analysis.chunk.text}

MATCHED SOURCE:
Title: {best_match.source_title}
URL: {best_match.source_url}
Text: {best_match.source_chunk}

SIMILARITY SCORES:
- Semantic: {best_match.similarity_semantic:.2f}
- Lexical: {best_match.similarity_lexical:.2f}
- Match type: {best_match.match_type}

Provide a brief 1-2 sentence explanation of the similarity and whether it appears to be:
1. Direct copying
2. Paraphrasing  
3. Common phrasing (acceptable overlap)
4. Coincidental similarity
"""
                
                try:
                    print(f"    Chunk {idx}: Generating explanation...", end="", flush=True)
                    response = await self.azure_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert in plagiarism detection and academic integrity."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )
                    
                    analysis.explanation = response.choices[0].message.content.strip()
                    print(" ✓")
                
                except Exception as e:
                    analysis.explanation = f"Could not generate explanation: {e}"
                    print(f" ✗")
        
        return chunk_analyses
    
    def _aggregate_results(
        self,
        chunks: List[SpeechChunk],
        chunk_analyses: List[ChunkAnalysis],
        speech_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate chunk-level results into overall report
        
        Returns comprehensive plagiarism report
        """
        # Overall statistics
        print(f"  Calculating overall risk scores...")
        total_chunks = len(chunks)
        high_risk_chunks = [a for a in chunk_analyses if a.risk_level == "high"]
        medium_risk_chunks = [a for a in chunk_analyses if a.risk_level == "medium"]
        low_risk_chunks = [a for a in chunk_analyses if a.risk_level == "low"]
        
        print(f"    High risk: {len(high_risk_chunks)}")
        print(f"    Medium risk: {len(medium_risk_chunks)}")
        print(f"    Low risk: {len(low_risk_chunks)}")
        
        # Overall risk score (weighted by chunk length)
        total_words = sum(c.word_count for c in chunks)
        if total_words > 0:
            overall_risk = sum(
                a.risk_score * a.chunk.word_count for a in chunk_analyses
            ) / total_words
        else:
            overall_risk = 0.0
        
        # Determine overall risk level
        if overall_risk > 0.6 or len(high_risk_chunks) > total_chunks * 0.2:
            overall_level = "high"
        elif overall_risk > 0.3 or len(medium_risk_chunks) > total_chunks * 0.3:
            overall_level = "medium"
        elif overall_risk > 0.1:
            overall_level = "low"
        else:
            overall_level = "minimal"
        
        print(f"  Overall risk level: {overall_level.upper()} (score={overall_risk:.3f})")
        
        # Collect unique sources
        print(f"  Collecting unique sources...")
        all_sources = {}
        for analysis in chunk_analyses:
            for match in analysis.top_matches:
                url = match.source_url
                if url not in all_sources:
                    all_sources[url] = {
                        'url': url,
                        'title': match.source_title,
                        'date': match.source_date,
                        'match_count': 0,
                        'max_similarity': 0.0
                    }
                all_sources[url]['match_count'] += 1
                all_sources[url]['max_similarity'] = max(
                    all_sources[url]['max_similarity'],
                    match.similarity_semantic
                )
        
        # Sort sources by match count
        top_sources = sorted(
            all_sources.values(),
            key=lambda x: (x['match_count'], x['max_similarity']),
            reverse=True
        )[:10]
        
        if top_sources:
            print(f"  Top {len(top_sources)} matching sources identified")
            for idx, src in enumerate(top_sources[:3], 1):
                print(f"    {idx}. {src['title'][:50]}... ({src['match_count']} matches)")
        
        # Build report
        print(f"  Building final report...")
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "speech_metadata": speech_metadata or {},
            "overall_risk_score": round(overall_risk, 3),
            "overall_risk_level": overall_level,
            "statistics": {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "high_risk_chunks": len(high_risk_chunks),
                "medium_risk_chunks": len(medium_risk_chunks),
                "low_risk_chunks": len(low_risk_chunks),
                "clean_chunks": total_chunks - len(high_risk_chunks) - len(medium_risk_chunks) - len(low_risk_chunks)
            },
            "top_sources": top_sources,
            "flagged_chunks": [
                {
                    "chunk_id": a.chunk.chunk_id,
                    "text": a.chunk.text,
                    "risk_score": round(a.risk_score, 3),
                    "risk_level": a.risk_level,
                    "is_boilerplate": a.is_boilerplate,
                    "top_match": {
                        "source_url": a.top_matches[0].source_url,
                        "source_title": a.top_matches[0].source_title,
                        "similarity_semantic": round(a.top_matches[0].similarity_semantic, 3),
                        "similarity_lexical": round(a.top_matches[0].similarity_lexical, 3),
                        "match_type": a.top_matches[0].match_type
                    } if a.top_matches else None,
                    "explanation": a.explanation
                }
                for a in chunk_analyses
                if a.risk_level in ["high", "medium"]
            ],
            "all_chunk_analyses": [
                {
                    "chunk_id": a.chunk.chunk_id,
                    "paragraph_index": a.chunk.paragraph_index,
                    "risk_score": round(a.risk_score, 3),
                    "risk_level": a.risk_level,
                    "match_count": len(a.top_matches)
                }
                for a in chunk_analyses
            ]
        }
        
        return report


# Utility function for easy integration
async def check_plagiarism(
    speech_text: str,
    azure_client: Optional[Any] = None,
    tavily_client: Optional[Any] = None,
    speech_metadata: Optional[Dict[str, Any]] = None,
    use_hf_classifier: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run plagiarism check
    
    Args:
        speech_text: The speech to analyze
        azure_client: Azure OpenAI client
        tavily_client: Tavily search client
        speech_metadata: Optional metadata
        use_hf_classifier: Whether to use HuggingFace classifier
    
    Returns:
        Plagiarism analysis report
    """
    checker = PlagiarismChecker(
        azure_client=azure_client,
        tavily_client=tavily_client,
        use_hf_classifier=use_hf_classifier
    )
    
    return await checker.analyze_speech(
        speech_text=speech_text,
        speech_metadata=speech_metadata
    )
