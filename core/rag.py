import os
import re
import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class RAGManager:
    """
    Lightweight in-process vector DB (ChromaDB) for portfolio ingestion.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
        # Initialize an ephemeral standard ChromaDB client
        self.chroma_client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Create a new memory collection
        self.collection = self.chroma_client.create_collection(
            name="portfolio",
            embedding_function=self.embedding_fn
        )

    def ingest_portfolio(self):
        """
        Reads, chunks, and embeds portfolio.md.
        Focuses on AI, Python/OOP, IoT/Embedded, and Hardware logic.
        """
        if not os.path.exists(self.data_path):
            logger.warning(f"Portfolio file not found at {self.data_path}. Skipping ingestion.")
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple chunking by paragraph/section
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
            
            if not paragraphs:
                logger.warning("Portfolio is empty or poorly formatted. Skipped ingestion.")
                return

            ids = [f"chunk_{i}" for i in range(len(paragraphs))]
            
            self.collection.add(
                documents=paragraphs,
                ids=ids
            )
            logger.info(f"Successfully ingested {len(paragraphs)} chunks from {self.data_path} into ChromaDB.")
            
        except Exception as e:
            logger.error(f"Failed to ingest portfolio: {e}")

    # Chunks with L2 distance above this threshold are considered irrelevant
    # ChromaDB uses squared L2 by default: 0 = identical, 2 = max distance
    # Lower threshold = stricter matching
    SIMILARITY_THRESHOLD = 1.0

    # Query expansion dictionary for technical terms
    QUERY_EXPANSIONS = {
        "rag": "RAG retrieval augmented generation vector database embeddings",
        "llm": "LLM large language model GPT transformer",
        "api": "API REST endpoint HTTP web service",
        "async": "async await asyncio concurrency parallelism",
        "db": "database SQL NoSQL storage persistence",
        "ml": "machine learning ML AI model training inference",
        "dl": "deep learning neural network CNN RNN transformer",
        "nlp": "natural language processing NLP text tokenization",
        "cv": "computer vision image processing CNN OpenCV",
        "iot": "IoT Internet of Things embedded sensors MQTT",
        "embedded": "embedded systems firmware microcontroller ESP32",
        "python": "Python programming language scripting code",
    }

    # Common stop words to filter
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "among", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "can", "shall", "tell", "me", "about", "your", "how", "what",
        "why", "when", "where", "which", "who", "whom", "whose", "this", "that"
    }

    def _expand_query(self, query: str) -> str:
        """Expand query with related technical terms for better retrieval."""
        query_lower = query.lower()
        expanded = query

        for term, expansion in self.QUERY_EXPANSIONS.items():
            if term in query_lower:
                expanded += " " + expansion

        return expanded

    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text, filtering out stop words."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return {w for w in words if w not in self.STOP_WORDS and len(w) > 2}

    def _calculate_keyword_score(self, query: str, document: str) -> float:
        """Calculate keyword overlap score between query and document."""
        query_keywords = self._extract_keywords(query)
        doc_keywords = self._extract_keywords(document)

        if not query_keywords:
            return 0.0

        overlap = len(query_keywords.intersection(doc_keywords))
        return overlap / len(query_keywords)

    def _rerank_chunks(self, query: str, chunks: list, distances: list) -> list:
        """
        Rerank chunks using a combination of semantic distance and keyword overlap.
        Returns list of (chunk, combined_score) tuples sorted by score (descending).
        """
        scored_chunks = []

        for chunk, distance in zip(chunks, distances):
            # Semantic score: invert distance (lower distance = higher score)
            # ChromaDB returns squared L2 distance, normalize to 0-1 range
            semantic_score = 1.0 / (1.0 + distance)

            # Keyword score: Jaccard-like overlap
            keyword_score = self._calculate_keyword_score(query, chunk)

            # Combined score: weighted combination
            # Weight semantic slightly higher as it's more reliable
            combined_score = (0.6 * semantic_score) + (0.4 * keyword_score)

            scored_chunks.append((chunk, combined_score, distance))

        # Sort by combined score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks

    def retrieve_context(self, query: str, n_results: int = 3, conversation_history: list = None) -> str:
        """
        Retrieves relevant concepts based on the ongoing conversation.
        Uses query expansion, hybrid retrieval, and reranking for better results.
        """
        try:
            if self.collection.count() == 0:
                return "No portfolio context available."

            # Step 1: Expand query with related terms
            expanded_query = self._expand_query(query)
            logger.info(f"RAG: Expanded query from '{query[:50]}...' to include technical terms")

            # Step 2: Retrieve more candidates than needed for reranking
            retrieve_n = min(n_results * 2 + 1, self.collection.count())

            results = self.collection.query(
                query_texts=[expanded_query],
                n_results=retrieve_n,
                include=["documents", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return "No relevant portfolio context found."

            docs = results["documents"][0]
            distances = results["distances"][0]

            # Step 3: Rerank chunks using combined semantic + keyword scoring
            reranked = self._rerank_chunks(query, docs, distances)

            # Step 4: Filter and select top chunks
            good_chunks = []
            for chunk, combined_score, distance in reranked:
                logger.info(f"RAG: score={combined_score:.3f}, distance={distance:.3f} | chunk='{chunk[:60]}...'")
                # Use combined score threshold (0.3 is reasonable for combined score)
                if combined_score >= 0.3 and distance <= self.SIMILARITY_THRESHOLD:
                    good_chunks.append(chunk)
                if len(good_chunks) >= n_results:
                    break

            if not good_chunks:
                logger.info(f"RAG: No chunks passed the confidence thresholds. Returning empty context.")
                return "(No specific portfolio context for this question. Answer from general best practices and industry knowledge.)"

            logger.info(f"RAG: Selected top {len(good_chunks)} chunks after reranking.")
            return "\n---\n".join(good_chunks)

        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {e}")
            return "Error retrieving context."
