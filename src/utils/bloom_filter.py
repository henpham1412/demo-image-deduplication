# utils/bloom_filter.py

import mmh3
import math
import bitarray

class BloomFilter:
    """
    Simple Bloom Filter implementation for duplicate detection.
    """

    def __init__(self, n_items: int, false_positive_prob: float = 0.01):
        """
        Initialize Bloom Filter with optimal size and number of hash functions.
        Args:
            n_items (int): Expected number of items.
            false_positive_prob (float): Desired false positive rate.
        """
        self.size = self._get_size(n_items, false_positive_prob)
        self.hash_count = self._get_hash_count(self.size, n_items)
        self.bit_array = bitarray.bitarray(self.size)
        self.bit_array.setall(0)

    def _get_size(self, n: int, p: float) -> int:
        """Calculate bloom filter size (m) given items n and fp prob p."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _get_hash_count(self, m: int, n: int) -> int:
        """Calculate number of hash functions (k)."""
        k = (m / n) * math.log(2)
        return int(k)

    def add(self, item: str):
        """Add an item (string)."""
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = 1

    def check(self, item: str) -> bool:
        """Check membership (may return false positive)."""
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if not self.bit_array[digest]:
                return False
        return True
