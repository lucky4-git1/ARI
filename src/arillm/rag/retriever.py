"""Fast, local keyword and BM25 retrieval engine for Ari-LLM.

Provides instant (< 3ms) contextual retrieval over local templates, components,
and documents without requiring external APIs, vector DBs, or internet access.
"""

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple


class LocalKnowledgeRetriever:
    """Local BM25 / TF-IDF retriever indexing components, templates, and documents."""

    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.documents: Dict[str, str] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into normalized words with fuzzy/prefix handling."""
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        # Add basic normalized forms to handle typos like 'secton' -> 'section', 'carosel' -> 'carousel'
        normalized = []
        synonyms = {
            "secton": "section",
            "carosel": "carousel",
            "carusel": "carousel",
            "slider": "carousel",
            "sliders": "carousel",
            "slide": "carousel",
            "slides": "carousel",
            "auto": "automatic",
            "nav": "navbar",
            "wd": "with",
            "landingpage": "landing",
        }
        for w in words:
            normalized.append(w)
            if w == "landingpage":
                normalized.append("landing")
                normalized.append("page")
            elif w in synonyms:
                normalized.append(synonyms[w])
                if "slide" in w or "slider" in w:
                    normalized.append("slide")
                    normalized.append("carousel")
        return normalized

    def load_and_index(self):
        """Scans and indexes all text/html/py/sql documents in the knowledge directory."""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            return

        total_len = 0
        for fname in os.listdir(self.knowledge_dir):
            fpath = os.path.join(self.knowledge_dir, fname)
            if os.path.isfile(fpath):
                try:
                    if fname.endswith(".json"):
                        import json
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            facts_data = json.load(f)
                        if isinstance(facts_data, list):
                            for idx, entry in enumerate(facts_data):
                                key_name = f"{fname}#{entry.get('topic', idx)}"
                                content = entry.get("answer", "")
                                kw_str = " ".join(entry.get("keywords", [])) + " " + entry.get("topic", "")
                                self.documents[key_name] = content
                                tokens = self._tokenize(kw_str + " " + content)
                                self.doc_tokens[key_name] = tokens
                                total_len += len(tokens)
                                for term in set(tokens):
                                    self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                            continue

                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    if content:
                        self.documents[fname] = content
                        tokens = self._tokenize(fname + " " + content)
                        self.doc_tokens[fname] = tokens
                        total_len += len(tokens)
                        
                        unique_terms = set(tokens)
                        for term in unique_terms:
                            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                except Exception as e:
                    print(f"Notice reading {fpath}: {e}")

        num_docs = len(self.documents)
        self.avg_doc_len = (total_len / num_docs) if num_docs > 0 else 1.0
        print(f"[RAG] Indexed {num_docs} local knowledge documents from {self.knowledge_dir}.")

    def retrieve(
        self, query: str, top_k: int = 1, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Retrieves top-k most relevant documents for a given query.
        
        Args:
            query: User's query string.
            top_k: Number of matching documents to return.
            threshold: Minimum relevance score to trigger context injection.
            
        Returns:
            List of (doc_name, content, score)
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}
        num_docs = len(self.documents)
        k1 = 1.5
        b = 0.75

        for doc_name, tokens in self.doc_tokens.items():
            doc_len = len(tokens)
            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            doc_score = 0.0
            for qt in query_tokens:
                if qt in term_counts:
                    tf = term_counts[qt]
                    df = self.doc_freqs.get(qt, 1)
                    # Standard BM25 IDF
                    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    # BM25 TF weight
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                    doc_score += idf * (numerator / denominator)

            if doc_score >= threshold:
                scores[doc_name] = doc_score

        # Sort descending by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(name, self.documents[name], score) for name, score in sorted_docs]
        return results

    def format_context_prompt(self, query: str, top_k: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """Returns (injected_prompt, retrieved_doc_name) or (None, None) if no match."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return None, None

        doc_name, content, score = results[0]
        # Wrap context clearly
        context_block = (
            f"Here is a reference template ({doc_name}):\n"
            f"```\n{content}\n```\n"
        )
        return context_block, doc_name
