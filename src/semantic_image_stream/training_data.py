"""Explicit, bounded public-data acquisition; never called by monitoring.

No user transcript is accepted by this module or sent to a dataset service.
Dataset scripts and remote Python are never executed. Downloaded rows remain
untrusted data; only fixed providers, schemas, and raster formats are accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlsplit

import httpx
from PIL import Image
import imagehash

IMAGE_DATASET = "Mitsua/safe-commons-pd-3m"
STORY_DATASET = "WorkInTheDark/FairytaleQA"
STORY_REVISION = "655b20b370f8ec6ef0cae5ac08f1c27ec6e72aaf"
COMMONS_REVISION = "d3126767adf3b8ad9779fe2914eeea33fca06eb9"
MUSEUM_DATASET = "Mitsua/art-museums-pd-440k"
MUSEUM_REVISION = "fba945da78b36262eb9272067197cc28d06cffbf"
ALLOWED_HOSTS = {"huggingface.co", "datasets-server.huggingface.co", "upload.wikimedia.org",
                 "raw.githubusercontent.com", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
                 "cas-bridge.xethub.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co"}
MAX_TOTAL_BYTES = 160 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checked_url(url):
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username
            or parts.password or parts.port not in {None, 443}):
        raise ValueError("Unapproved public dataset host")
    return url


class PublicDataClient:
    def __init__(self):
        self.client = httpx.Client(timeout=30, trust_env=False, follow_redirects=False,
                                   headers={"User-Agent": "SemanticImageStream/0.2 local-training (github.com/sporemound/semantic-image-stream)"})
        self.total = 0

    def close(self):
        self.client.close()

    def get(self, url, limit, params=None):
        for _ in range(4):
            checked_url(url)
            with self.client.stream("GET", url, params=params) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers["location"]))
                    params = None
                    continue
                response.raise_for_status()
                chunks, received = [], 0
                for chunk in response.iter_bytes():
                    received += len(chunk)
                    self.total += len(chunk)
                    if received > limit or self.total > MAX_TOTAL_BYTES:
                        raise ValueError("Public training download exceeded its byte limit")
                    chunks.append(chunk)
                return b"".join(chunks)
        raise ValueError("Too many dataset download redirects")

    def revision(self, dataset):
        data = json.loads(self.get("https://huggingface.co/api/datasets/" + dataset, 1_000_000))
        revision = data["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Invalid dataset revision")
        return revision


def image_record(row):
    if row.get("License") not in {"Public Domain", "CC0"}:
        return None
    identifier = row.get("ImageID", "")
    if not re.fullmatch(r"[a-f0-9]{64}", identifier):
        return None
    captions = [item["text"] for item in row.get("Caption", [])
                if item.get("lang") == "en" and item.get("type") == "commons"
                and isinstance(item.get("text"), str) and 25 <= len(item["text"]) <= 500]
    if not captions:
        return None
    caption = captions[0]
    if len(re.findall(r"\b[a-zA-Z]{3,}\b", caption)) < 5:
        return None
    url = row.get("ImageURL", "")
    source = row.get("URL", "")
    if urlsplit(source).hostname != "commons.wikimedia.org" or not source.startswith("https://"):
        return None
    try:
        checked_url(url)
    except ValueError:
        return None
    parts = urlsplit(url)
    if parts.hostname != "upload.wikimedia.org" or not parts.path.startswith("/wikipedia/commons/"):
        return None
    if not re.search(r"\.(?:jpe?g|png|webp)$", parts.path, re.IGNORECASE):
        return None
    tail = parts.path.removeprefix("/wikipedia/commons/")
    if len(tail.split("/")) != 3:
        return None
    filename = tail.rsplit("/", 1)[-1]
    thumb = "https://upload.wikimedia.org/wikipedia/commons/thumb/" + tail + "/330px-" + filename
    return {"id": identifier, "caption": caption, "caption_type": "commons", "caption_language": "en",
            "title": str(row.get("Title", ""))[:500], "author": str(row.get("Author", ""))[:5000],
            "image_license": row["License"], "source_url": source, "image_url": url,
            "thumbnail_url": thumb, "dataset": IMAGE_DATASET, "dataset_split": "train",
            "caption_license": "CC BY-SA 4.0"}


def story_examples(csv_bytes, max_examples=128):
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    examples, counts = [], {}
    for index, row in enumerate(reader):
        source, answer = row.get("story_section", ""), row.get("answer1", "")
        attribute = row.get("attribute", "").strip().casefold()
        explicit = row.get("ex_or_im", row.get("ex-or-im", "")).strip().casefold()
        if explicit not in {"explicit", "ex"} or attribute not in {"character", "setting", "action", "outcome resolution"}:
            continue
        if not (30 <= len(source) <= 1200 and 1 <= len(answer) <= 300 and answer.casefold() in source.casefold()
                and 1 <= len(row.get("question", "")) <= 240):
            continue
        story = row.get("story_name", "")
        if counts.get(story, 0) >= 2:
            continue
        examples.append({"id": f"fairytaleqa-train-{index}", "source": source,
                         "question": row["question"], "answer": answer, "attribute": attribute,
                         "provenance": {"dataset": STORY_DATASET, "revision": STORY_REVISION,
                                        "split": "train", "license": "apache-2.0", "story_name": story,
                                        "row_index": index, "answer_type": "explicit",
                                        "license_url": "https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE"}})
        counts[story] = counts.get(story, 0) + 1
        if len(examples) >= max_examples:
            break
    if not examples:
        raise ValueError("No supported explicit story examples found")
    return {"version": 1, "examples": examples}


def museum_record(row):
    data = row.get("json", {})
    identifier = data.get("ImageID", "")
    caption = data.get("captionEn", "")
    asset = row.get("jpg", {}).get("src", "")
    if (not re.fullmatch(r"[a-f0-9-]{36}", identifier) or data.get("License") != "CC0"
            or not isinstance(caption, str) or not 25 <= len(caption) <= 700
            or f"@{MUSEUM_REVISION}/" not in row.get("__url__", "")
            or urlsplit(data.get("URL", "")).scheme != "https"):
        return None
    if (urlsplit(asset).hostname != "datasets-server.huggingface.co"
            or f"/{MUSEUM_DATASET}/--/{MUSEUM_REVISION}/--/" not in urlsplit(asset).path):
        return None
    return {"id": identifier, "caption": caption, "caption_type": "museum human metadata",
            "caption_language": "en", "title": str(data.get("Title", ""))[:500],
            "author": str(data.get("Author", ""))[:5000], "image_license": "CC0",
            "source_url": data["URL"], "image_url": data.get("ImageURL", ""),
            "thumbnail_url": asset, "dataset": MUSEUM_DATASET, "dataset_split": "train",
            "caption_license": "CC BY 4.0"}


def acquire(destination, target=320, source="museum"):
    if not 100 <= target <= 512:
        raise ValueError("Choose between 100 and 512 images")
    destination.mkdir(parents=True, exist_ok=True)
    pixels = destination / "images"
    pixels.mkdir(exist_ok=True)
    public = PublicDataClient()
    try:
        print("Downloading pinned human story annotations", flush=True)
        story_bytes = public.get(f"https://huggingface.co/datasets/{STORY_DATASET}/resolve/{STORY_REVISION}/train.csv", 16_000_000)
        refs = story_examples(story_bytes)
        (destination / "story_examples.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "fairytaleqa-train.csv").write_bytes(story_bytes)
        dataset, revision = (MUSEUM_DATASET, MUSEUM_REVISION) if source == "museum" else (IMAGE_DATASET, COMMONS_REVISION)
        if public.revision(dataset) != revision:
            raise ValueError("Image dataset changed: review revision before acquiring new rows")
        rows, hashes, failures = [], [], {}
        offsets = random.Random(20261013).sample(range(0, 16_000 if source == "museum" else 3_040_583, 100), 100)
        for offset in offsets:
            raw = public.get("https://datasets-server.huggingface.co/rows", 3_000_000,
                             {"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": 100})
            data = json.loads(raw)
            for item in data["rows"]:
                record = museum_record(item["row"]) if source == "museum" else image_record(item["row"])
                if record is None:
                    continue
                path = pixels / (record["id"] + ".jpg")
                try:
                    content = public.get(record["thumbnail_url"], 2_000_000)
                    with Image.open(io.BytesIO(content)) as image:
                        if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > 10_000_000:
                            raise ValueError("Invalid raster geometry/format")
                        if min(image.size) < 100 or max(image.size) / min(image.size) > 4:
                            raise ValueError("Unsuitable thumbnail")
                        converted = image.convert("RGB")
                        phash = str(imagehash.phash(converted))
                        if any((int(phash, 16) ^ int(old, 16)).bit_count() <= 6 for old in hashes):
                            continue
                        converted.save(path, "JPEG", quality=90)
                        converted.close()
                    record.update(row_index=item["row_idx"], dataset_revision=revision,
                                  row_sha256=sha256(json.dumps(item["row"], sort_keys=True).encode()),
                                  thumbnail_sha256=sha256(content), file_sha256=sha256(path.read_bytes()),
                                  phash=phash, filename="images/" + path.name)
                    rows.append(record)
                    hashes.append(phash)
                    if len(rows) % 20 == 0:
                        print(f"Prepared {len(rows)}/{target} licensed image-caption examples", flush=True)
                except (httpx.HTTPError, ValueError, OSError) as exc:
                    name = str(exc.response.status_code) if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
                    failures[name] = failures.get(name, 0) + 1
                    if name == "429":
                        print("Source requested a pause; stopping image downloads", flush=True)
                        break
                time.sleep(.25)
                if len(rows) >= target:
                    break
            if len(rows) >= target or "429" in failures:
                break
        if public.revision(dataset) != revision:
            raise ValueError("Dataset changed during row sampling; pack is not ready for training")
        # Image/pHash deduplication happens before splitting. No source image can
        # appear in another split, including through a different caption.
        random.Random(4242).shuffle(rows)
        n = len(rows)
        for index, record in enumerate(rows):
            record["local_split"] = "train" if index < int(n * .75) else "calibration" if index < int(n * .875) else "test"
        manifest = {"version": 1, "seed": 20261013, "image_dataset": dataset,
                    "image_revision": revision, "image_rows_pinned": False,
                    "sampling": "HF rows API: repository revision checked before and after; retained rows hashed",
                    "story_dataset": STORY_DATASET, "story_revision": STORY_REVISION,
                    "story_train_sha256": sha256(story_bytes), "story_examples": len(refs["examples"]),
                    "download_bytes": public.total, "failures": failures, "images": rows}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Pack ready: {n} image pairs; {len(refs['examples'])} human story examples", flush=True)
        if n < 100:
            raise ValueError("Not enough independent images for train/calibration/test; no model will be fitted")
    finally:
        public.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare a bounded public Hugging Face training pack locally")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=320)
    parser.add_argument("--source", choices=("museum", "commons"), default="museum")
    args = parser.parse_args()
    acquire(args.output.resolve(), args.images, args.source)


if __name__ == "__main__":
    main()
