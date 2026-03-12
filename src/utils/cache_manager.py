"""
Gemini API Response Cache Manager
DeepSceneLoc — Semester 2, Weeks 11–13

Author: Anuj Kondawar (Preprocessing & Pipeline Lead)

Caching strategy for Gemini API calls to:
  - Reduce API cost for repeated / similar images
  - Improve end-to-end latency for cached hits
  - Provide offline fallback capability

Design decisions:
  - Cache key: perceptual hash (pHash) of the image  →  similar images share a key
  - Storage: SQLite (zero-config, file-based, across-session persistent)
  - Eviction: LRU with configurable max-size and TTL
  - Thread-safe: RLock guards all cache operations
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Perceptual hasher
# ─────────────────────────────────────────────────────────────

def _phash(image, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash (pHash) of ``image``.

    Two visually similar images produce the same hash when their
    average DCT-domain difference is below the quantisation threshold.

    Falls back to a plain MD5 of the raw pixel bytes if ``PIL`` is not
    available (should not happen — PIL is a project dependency).

    Args:
        image: PIL Image object.
        hash_size: Grid side length.  8 → 64-bit hash.

    Returns:
        Hex-string hash (16 chars for hash_size=8).
    """
    try:
        from PIL import Image
        import struct

        # Reduce to grayscale, resize to (hash_size × 2) × hash_size
        img = image.convert("L").resize(
            (hash_size * 2, hash_size), resample=Image.Resampling.LANCZOS
        )
        pixels = list(img.getdata())
        avg    = sum(pixels) / len(pixels)
        bits   = "".join("1" if p >= avg else "0" for p in pixels)
        # Pack bits into bytes and hex-encode
        n     = int(bits, 2)
        nbytes = (len(bits) + 7) // 8
        return n.to_bytes(nbytes, "big").hex()

    except Exception:
        # Fallback: MD5 of raw bytes
        raw = image.tobytes() if hasattr(image, "tobytes") else b""
        return hashlib.md5(raw).hexdigest()


def _content_hash(text: str) -> str:
    """SHA-256 of an arbitrary string (used for prompt-keyed caches)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────
# Cache Manager
# ─────────────────────────────────────────────────────────────

class GeminiCacheManager:
    """
    Persistent LRU cache for Gemini API responses.

    Each cache entry stores:
      - ``image_hash``  : pHash of the input image
      - ``category``    : Scene classification category hint
      - ``response``    : JSON-serialised Gemini response dict
      - ``created_at``  : Unix timestamp of first insertion
      - ``last_used``   : Unix timestamp of latest hit
      - ``hit_count``   : Number of cache hits

    Usage::

        cache = GeminiCacheManager(db_path="results/gemini_cache.db")

        hit = cache.get(image, category="Coastal")
        if hit:
            return hit        # Skip API call entirely

        result = gemini_analyzer.analyze_location(image, category)
        cache.put(image, category, result)
        return result
    """

    def __init__(
        self,
        db_path: str = "results/gemini_cache.db",
        max_size: int = 1000,
        ttl_seconds: Optional[int] = 7 * 24 * 3600,   # 7 days default
    ):
        """
        Args:
            db_path: Path for the SQLite database file.
            max_size: Maximum number of entries before LRU eviction.
            ttl_seconds: Time-to-live per entry (None = no expiry).
        """
        self.db_path   = Path(db_path)
        self.max_size  = max_size
        self.ttl       = ttl_seconds
        self._lock     = threading.RLock()
        self._stats    = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def get(self, image, category: str = "") -> Optional[Dict]:
        """
        Retrieve a cached Gemini response.

        Args:
            image: PIL Image (used to compute pHash).
            category: Scene category hint (included in cache key).

        Returns:
            Cached response dict, or ``None`` on a cache miss / expiry.
        """
        key = self._make_key(image, category)
        now = time.time()

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, response, created_at FROM cache WHERE key = ?", (key,)
                ).fetchone()

                if row is None:
                    self._stats["misses"] += 1
                    return None

                entry_id, response_json, created_at = row

                # TTL expiry check
                if self.ttl and (now - created_at) > self.ttl:
                    conn.execute("DELETE FROM cache WHERE id = ?", (entry_id,))
                    conn.commit()
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                    return None

                # Update last-used timestamp and hit count (LRU tracking)
                conn.execute(
                    "UPDATE cache SET last_used = ?, hit_count = hit_count + 1 WHERE id = ?",
                    (now, entry_id),
                )
                conn.commit()
                self._stats["hits"] += 1
                return json.loads(response_json)

            finally:
                conn.close()

    def put(self, image, category: str, response: Dict) -> None:
        """
        Store or update a Gemini response in the cache.

        Args:
            image: PIL Image.
            category: Scene category hint.
            response: Gemini response dict (must be JSON serialisable).
        """
        key  = self._make_key(image, category)
        now  = time.time()
        resp = json.dumps(response)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO cache (key, response, created_at, last_used, hit_count)
                    VALUES (?, ?, ?, ?, 0)
                    ON CONFLICT(key) DO UPDATE
                    SET response = excluded.response,
                        created_at = excluded.created_at,
                        last_used  = excluded.last_used
                    """,
                    (key, resp, now, now),
                )
                conn.commit()
                self._evict_if_needed(conn)
            finally:
                conn.close()

    def invalidate(self, image, category: str = "") -> bool:
        """Remove a specific entry from the cache. Returns True if deleted."""
        key = self._make_key(image, category)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def clear(self) -> int:
        """Flush the entire cache. Returns number of entries deleted."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM cache")
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            finally:
                conn.close()

    def get_stats(self) -> Dict:
        """Return hit/miss/eviction counters plus cache size."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size":      self.size(),
            "max_size":  self.max_size,
            "hit_rate":  round(self._stats["hits"] / total, 4) if total else 0.0,
        }

    def print_stats(self):
        """Print a formatted statistics summary."""
        s = self.get_stats()
        print(f"\n{'='*50}")
        print("Gemini Cache Statistics")
        print(f"{'='*50}")
        print(f"  Entries  : {s['size']} / {s['max_size']}")
        print(f"  Hits     : {s['hits']}")
        print(f"  Misses   : {s['misses']}")
        print(f"  Hit rate : {s['hit_rate']:.1%}")
        print(f"  Evictions: {s['evictions']}")
        print(f"  Expired  : {s['expirations']}")
        print(f"{'='*50}")

    def export_to_json(self, path: str = "results/gemini_cache_export.json") -> int:
        """Dump all cache entries to a JSON file for inspection / backup."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT key, response, created_at, last_used, hit_count FROM cache"
                ).fetchall()
            finally:
                conn.close()

        entries = [
            {
                "key":        r[0],
                "response":   json.loads(r[1]),
                "created_at": r[2],
                "last_used":  r[3],
                "hit_count":  r[4],
            }
            for r in rows
        ]
        with open(out, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Exported {len(entries)} cache entries to {out}")
        return len(entries)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _make_key(self, image, category: str) -> str:
        """Combine image pHash + category into a unique cache key."""
        img_hash = _phash(image)
        cat_hash = _content_hash(category.lower().strip())
        return f"{img_hash}-{cat_hash}"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self):
        """Create the cache table if it does not exist."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    key        TEXT    UNIQUE NOT NULL,
                    response   TEXT    NOT NULL,
                    created_at REAL    NOT NULL,
                    last_used  REAL    NOT NULL,
                    hit_count  INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON cache (last_used)")
            conn.commit()
        finally:
            conn.close()

    def _evict_if_needed(self, conn: sqlite3.Connection):
        """Remove oldest (LRU) entries until size <= max_size."""
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > self.max_size:
            excess = count - self.max_size
            conn.execute(
                """
                DELETE FROM cache WHERE id IN (
                    SELECT id FROM cache ORDER BY last_used ASC LIMIT ?
                )
                """,
                (excess,),
            )
            conn.commit()
            self._stats["evictions"] += excess


# ─────────────────────────────────────────────────────────────
# Cached Gemini wrapper
# ─────────────────────────────────────────────────────────────

class CachedGeminiAnalyzer:
    """
    A transparent caching wrapper around ``GeminiLocationAnalyzer``.

    Drop-in replacement — same ``analyze_location()`` signature as the
    original, but responses are served from cache on repeated calls.

    Usage::

        from src.utils.gemini_integration import GeminiLocationAnalyzer
        from src.utils.cache_manager import CachedGeminiAnalyzer

        base    = GeminiLocationAnalyzer(api_key)
        cached  = CachedGeminiAnalyzer(base)
        result  = cached.analyze_location(image, "Coastal", 0.92)
    """

    def __init__(
        self,
        analyzer,
        cache: Optional[GeminiCacheManager] = None,
        db_path: str = "results/gemini_cache.db",
    ):
        self.analyzer = analyzer
        self.cache    = cache or GeminiCacheManager(db_path=db_path)

    def analyze_location(
        self,
        image,
        predicted_category: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Dict:
        """
        Analyse an image, serving from cache when available.

        Args:
            image: PIL Image.
            predicted_category: Scene category hint from Stage 1.
            confidence: Stage 1 confidence score.

        Returns:
            Gemini location analysis dict.
        """
        category = predicted_category or ""

        # Try cache first
        cached = self.cache.get(image, category)
        if cached is not None:
            cached["_from_cache"] = True
            return cached

        # Miss — call real API
        result = self.analyzer.analyze_location(image, predicted_category, confidence)
        result["_from_cache"] = False

        # Store only on success
        if "error" not in result:
            self.cache.put(image, category, result)

        return result

    def get_location_summary(self, analysis_result: Dict) -> str:
        """Delegate to underlying analyzer."""
        return self.analyzer.get_location_summary(analysis_result)

    @property
    def stats(self) -> Dict:
        return self.cache.get_stats()


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────

def _smoke_test():
    """Self-contained test that does not require Gemini credentials."""
    import tempfile, os

    print("Smoke-testing cache_manager.py...")

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test_cache.db")
        cache = GeminiCacheManager(db_path=db, max_size=5, ttl_seconds=3600)

        # Create a minimal fake image
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (100, 100), color=(100, 150, 200))

        # Initial miss
        assert cache.get(img, "Coastal") is None
        assert cache.size() == 0

        # Put + hit
        fake_response = {"exact_location": "Test Beach", "country": "Testland"}
        cache.put(img, "Coastal", fake_response)
        assert cache.size() == 1

        hit = cache.get(img, "Coastal")
        assert hit is not None
        assert hit["exact_location"] == "Test Beach"

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

        # LRU eviction: fill past max_size
        for i in range(10):
            img_i = PILImage.new("RGB", (100, 100), color=(i * 10, i * 5, 200))
            cache.put(img_i, f"cat{i}", {"location": f"place{i}"})
        assert cache.size() <= cache.max_size

        print("  All checks passed.")


if __name__ == "__main__":
    _smoke_test()
