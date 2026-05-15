"""
BTObot — Persistent disk store for parent documents.

LangChain 1.x removed LocalFileStore from langchain_community.
This module provides a drop-in replacement backed by Python's shelve
(a persistent dict-like file on disk).

The store satisfies the BaseStore[str, Document] interface required by
ParentDocumentRetriever.
"""

from __future__ import annotations

import shelve
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class LocalFileStore(BaseStore[str, Document]):
    """
    Simple persistent store that serialises LangChain Documents to a
    shelve database file at `root_path`.

    The shelve file is opened/closed on every operation so that multiple
    processes can share the store safely (important during ingest).
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = str(root_path)
        # Ensure the parent directory exists
        Path(self.root_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        results: List[Optional[Document]] = []
        with shelve.open(self.root_path, flag="c") as db:
            for key in keys:
                results.append(db.get(key, None))
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        with shelve.open(self.root_path, flag="c") as db:
            for key, value in key_value_pairs:
                db[key] = value

    def mdelete(self, keys: Sequence[str]) -> None:
        with shelve.open(self.root_path, flag="c") as db:
            for key in keys:
                try:
                    del db[key]
                except KeyError:
                    pass

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with shelve.open(self.root_path, flag="r") as db:
            for key in db.keys():
                if prefix is None or key.startswith(prefix):
                    yield key
