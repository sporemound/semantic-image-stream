"""Bounded SQLite query/image/history store. Transcripts are never persisted."""
from pathlib import Path
import json
import re
import sqlite3
import threading
import time

from .models import ImageCandidate


class Cache:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings.cache_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "cache.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS queries (key TEXT PRIMARY KEY, ts REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS images (url TEXT PRIMARY KEY, ts REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, ts REAL, payload TEXT);
        """)
        self.db.commit()

    def get_query(self, key: str) -> list[ImageCandidate] | None:
        with self.lock:
            row = self.db.execute("SELECT ts,payload FROM queries WHERE key=?", (key,)).fetchone()
        if row and row[0] > time.time() - self.settings.cache_ttl_hours * 3600:
            return [ImageCandidate.from_dict(x) for x in json.loads(row[1])]
        return None

    def put_query(self, key: str, images: list[ImageCandidate]) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO queries VALUES (?,?,?)",
                            (key, time.time(), json.dumps([i.to_dict() for i in images])))
            self.db.execute("DELETE FROM queries WHERE key NOT IN (SELECT key FROM queries ORDER BY ts DESC LIMIT 500)")

    def get_image(self, url: str) -> ImageCandidate | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM images WHERE url=?", (url,)).fetchone()
        if row:
            image = ImageCandidate.from_dict(json.loads(row[0]))
            path = Path(image.local_path).resolve()
            if path.is_relative_to(self.root / "images") and path.is_file():
                return image
        return None

    def put_image(self, image: ImageCandidate) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?)",
                            (image.image_url, time.time(), json.dumps(image.to_dict())))
            self.db.execute("DELETE FROM images WHERE url NOT IN (SELECT url FROM images ORDER BY ts DESC LIMIT 2000)")

    def record(self, image: ImageCandidate) -> None:
        if self.settings.history_enabled:
            with self.lock, self.db:
                self.db.execute("INSERT INTO history(ts,payload) VALUES (?,?)", (time.time(), json.dumps(image.to_dict())))
                self.db.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                                (self.settings.history_limit,))

    def history(self) -> list[ImageCandidate]:
        with self.lock:
            rows = self.db.execute("SELECT payload FROM history ORDER BY id DESC LIMIT ?",
                                   (self.settings.history_limit,)).fetchall()
        return [ImageCandidate.from_dict(json.loads(row[0])) for row in rows]

    def clear_history(self) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM history")
            # VACUUM removes deleted rows from the main database as well.
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")

    def clear(self) -> None:
        self.clear_history()
        with self.lock, self.db:
            self.db.execute("DELETE FROM queries")
            self.db.execute("DELETE FROM images")
        # Only owned regular image files in this exact directory are removed.
        image_dir = self.root / "images"
        if image_dir.is_dir() and not image_dir.is_symlink():
            for path in image_dir.iterdir():
                if (path.is_file() and not path.is_symlink()
                        and re.fullmatch(r"[0-9a-f]{64}\.(jpg|png|webp|jpeg)", path.name)):
                    path.unlink(missing_ok=True)
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")

    def prune(self, protected: set[str] | None = None) -> None:
        protected = protected or set()
        image_dir = self.root / "images"
        if not image_dir.is_dir() or image_dir.is_symlink():
            return
        files = [p for p in image_dir.iterdir() if p.is_file() and not p.is_symlink()
                 and re.fullmatch(r"[0-9a-f]{64}\.(jpg|png|webp|jpeg)", p.name)]
        files.sort(key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        limit = self.settings.cache_max_mb * 1024 * 1024
        for path in files:
            if size <= limit:
                break
            if str(path) not in protected:
                size -= path.stat().st_size
                path.unlink(missing_ok=True)
        with self.lock, self.db:
            self.db.execute("DELETE FROM queries WHERE ts < ?", (time.time() - self.settings.cache_ttl_hours * 3600,))

    def close(self) -> None:
        with self.lock:
            self.db.close()
