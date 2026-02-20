import os
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
    # Empirically, > 1.2 = poor match, keep only good matches
    SIMILARITY_THRESHOLD = 1.2

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieves relevant concepts based on the ongoing conversation.
        Filters out chunks whose similarity score is below the confidence threshold
        to prevent irrelevant noise from polluting the LLM context.
        """
        try:
            if self.collection.count() == 0:
                return "No portfolio context available."

            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return "No relevant portfolio context found."

            docs = results["documents"][0]
            distances = results["distances"][0]

            # Filter: keep only chunks that are genuinely relevant
            good_chunks = []
            for doc, dist in zip(docs, distances):
                logger.info(f"RAG: distance={dist:.3f} | chunk='{doc[:60]}...'")
                if dist <= self.SIMILARITY_THRESHOLD:
                    good_chunks.append(doc)

            if not good_chunks:
                logger.info(f"RAG: No chunks passed the confidence threshold ({self.SIMILARITY_THRESHOLD}). Returning empty context.")
                return "(No specific portfolio context for this question. Answer from general best practices and industry knowledge.)"

            logger.info(f"RAG: {len(good_chunks)}/{len(docs)} chunks passed the confidence filter.")
            return "\n---\n".join(good_chunks)

        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {e}")
            return "Error retrieving context."
