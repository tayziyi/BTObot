"""BTObot — Retrieval Module
==========================
Implements the full pipeline:

    Query
      ↓  OllamaEmbeddings (nomic-embed-text)
    Child Retrieval  (ChromaDB similarity search)
      ↓
    Parent Expansion  (LangChain ParentDocumentRetriever)
      ↓
    Cosine-similarity Reranker  (Ollama embeddings + numpy, no downloads)
      ↓
    Top-3 Parent Chunks  → returned to app.py
"""

from __future__ import annotations

from typing import List

import numpy as np
from langchain_chroma import Chroma
from stores import LocalFileStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    OLLAMA_BASE_URL, EMBED_MODEL,
    CHROMA_PATH, COLLECTION_NAME, PARENT_STORE_PATH,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    TOP_K_CHILDREN, TOP_N_FINAL,
)


class BTORetriever:
    """
    Wraps the parent-child retrieval + reranking pipeline.

    Instantiate once at application startup; the object is reused across
    every user query to avoid re-loading models on each request.
    """

    def __init__(self) -> None:
        # --- Embeddings -------------------------------------------------
        self._embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # --- ChromaDB (child chunk vector store) -----------------------
        self._vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PATH,
        )

        # --- LocalFileStore (parent chunk document store) --------------
        self._store = LocalFileStore(PARENT_STORE_PATH)

        # --- Splitter definitions (required by ParentDocumentRetriever) -
        # These are only used during ingestion; at retrieval time the
        # retriever uses them for interface compatibility only.
        _parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
        )
        _child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )

        # --- ParentDocumentRetriever -----------------------------------
        # search_kwargs controls how many children are fetched from Chroma;
        # unique parent IDs are then resolved via the docstore.
        self._parent_retriever = ParentDocumentRetriever(
            vectorstore=self._vectorstore,
            docstore=self._store,
            parent_splitter=_parent_splitter,
            child_splitter=_child_splitter,
            search_kwargs={"k": TOP_K_CHILDREN},
        )

        # --- Reranker --------------------------------------------------
        # Uses cosine similarity between the query embedding and each parent
        # doc embedding — no external model downloads required.
        # (numpy is already present via chromadb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """
        Full pipeline: embed once → child search → parent expansion → rerank.

        The query is embedded exactly once and the vector is reused for both
        the ChromaDB similarity search and the cosine-similarity reranker,
        avoiding the redundant second embed_query call that the previous
        implementation made inside _rerank().

        Returns at most TOP_N_FINAL parent documents, ordered by relevance.
        Returns an empty list if the knowledge base is empty or inaccessible.
        """
        # Embed the query ONCE — vector is reused for search and reranking
        query_emb: List[float] = self._embeddings.embed_query(query)

        # Search ChromaDB with the pre-computed vector
        child_docs = self._vectorstore.similarity_search_by_vector(
            query_emb, k=TOP_K_CHILDREN
        )
        if not child_docs:
            return []

        # Expand child → parent: collect unique parent IDs (insertion order = relevance order)
        id_key = self._parent_retriever.id_key   # "doc_id"
        seen: dict[str, None] = {}
        for doc in child_docs:
            pid = doc.metadata.get(id_key)
            if pid:
                seen[pid] = None

        parent_docs_raw = self._store.mget(list(seen.keys()))
        parent_docs = [d for d in parent_docs_raw if d is not None]
        if not parent_docs:
            return []

        # Rerank with the cached query vector (no second embed_query call)
        return self._rerank(np.array(query_emb), parent_docs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rerank(self, query_emb: np.ndarray, docs: List[Document]) -> List[Document]:
        """Re-score parent docs by cosine similarity to the pre-computed query vector.

        Accepts the already-computed query embedding so no second embed_query
        API call is needed.
        """
        if len(docs) <= TOP_N_FINAL:
            return docs  # nothing to rerank

        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        # Truncate doc text to 512 chars to keep embed calls fast
        doc_texts = [doc.page_content[:512] for doc in docs]
        doc_embs = np.array(self._embeddings.embed_documents(doc_texts))
        doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9

        scores = doc_embs @ query_emb          # cosine similarity for each doc
        top_indices = np.argsort(scores)[::-1][:TOP_N_FINAL]
        return [docs[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def child_count(self) -> int:
        """Number of child chunks currently stored in ChromaDB."""
        return self._vectorstore._collection.count()
