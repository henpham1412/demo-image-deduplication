# similarity_search/minhash_search.py

import numpy as np
import random
import pickle
from typing import List, Tuple

class MinHashIndex:
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.signatures = {}
        self.ids = []

    def _minhash_signature(self, vector: np.ndarray) -> np.ndarray:
        """
        Simple hash-based minhash signature from vector.
        Here we assume binary-like feature (0/1). For real features you can binarize or threshold.
        """
        # Convert vector to set of indices
        s = set(np.where(vector > 0)[0])
        sig = []
        for seed in range(self.num_perm):
            random.seed(seed)
            perm = list(range(max(s) + 1)) if s else []
            random.shuffle(perm)
            val = min([perm[i] for i in s]) if s else 0
            sig.append(val)
        return np.array(sig)

    def add_batch(self, features: np.ndarray, ids: List[str]):
        for vec, id_ in zip(features, ids):
            sig = self._minhash_signature(vec)
            self.signatures[id_] = sig
            self.ids.append(id_)

    def query(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        qsig = self._minhash_signature(query_vec)
        results = []
        for id_, sig in self.signatures.items():
            jaccard_est = np.mean(qsig == sig)
            results.append((id_, 1.0 - jaccard_est))  # treat (1 - sim) as "distance"
        results.sort(key=lambda x: x[1])
        return results[:k]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump((self.signatures, self.ids), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.signatures, self.ids = pickle.load(f)


# Wrappers expected by pipeline
def build_index(features: np.ndarray, ids: List[str], out_path: str, num_perm: int = 128):
    index = MinHashIndex(num_perm=num_perm)
    index.add_batch(features, ids)
    index.save(out_path)
    return index

def search_index(index_path: str, features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    index = MinHashIndex()
    index.load(index_path)
    N = len(features)
    I = np.full((N, k), -1, dtype=int)
    D = np.full((N, k), np.inf, dtype=float)
    for i, vec in enumerate(features):
        neighbors = index.query(vec, k=k)
        for j, (nid, dist) in enumerate(neighbors):
            I[i, j] = index.ids.index(nid)
            D[i, j] = dist
    return D, I
