"""Fit the small relevance head on public examples with an offline CLIP encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .configuration import Settings
from .image_ranking import embed_pairs, encoder_fingerprint
from .learned_relevance import save_adapter, train_adapter


def read_pack(directory):
    directory = Path(directory).resolve()
    with (directory / "manifest.json").open("rb") as source:
        raw = source.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Public pack manifest exceeded its size limit")
    manifest = json.loads(raw)
    rows = manifest["images"]
    if (manifest.get("version") != 1 or not isinstance(rows, list) or not 100 <= len(rows) <= 512
            or manifest.get("image_dataset") not in {"Mitsua/safe-commons-pd-3m", "Mitsua/art-museums-pd-440k"}):
        raise ValueError("Unsupported public image pack")
    paths, captions, groups, splits = [], [], [], []
    ownership, phashes = {}, []
    for row in rows:
        filename = row["filename"]
        if not isinstance(filename, str) or not re.fullmatch(r"images/[a-f0-9-]{36,64}\.jpg", filename):
            raise ValueError("Unsafe public image filename")
        path = (directory / filename).resolve()
        if directory not in path.parents:
            raise ValueError("Public image escaped its pack folder")
        content = path.read_bytes()
        if len(content) > 2_000_000 or hashlib.sha256(content).hexdigest() != row["file_sha256"]:
            raise ValueError("Public training image checksum changed")
        caption = row["caption"]
        split = row["local_split"]
        group = row["source_url"]
        if (not isinstance(caption, str) or not 1 <= len(caption) <= 700
                or row.get("dataset_split") != "train" or split not in {"train", "calibration", "test"}
                or not isinstance(group, str) or not group.startswith("https://")):
            raise ValueError("Invalid public image training provenance")
        expected_license = "CC BY 4.0" if manifest["image_dataset"].endswith("440k") else "CC BY-SA 4.0"
        if (row.get("dataset") != manifest["image_dataset"]
                or row.get("dataset_revision") != manifest.get("image_revision")
                or row.get("image_license") not in {"CC0", "Public Domain"}
                or row.get("caption_license") != expected_license):
            raise ValueError("Public training source/license does not match its manifest")
        if group in ownership and ownership[group] != split:
            raise ValueError("Source image leaked across training splits")
        ownership[group] = split
        phash = row["phash"]
        if not re.fullmatch(r"[a-f0-9]{16}", phash):
            raise ValueError("Invalid deduplication fingerprint")
        if any((int(phash, 16) ^ int(previous, 16)).bit_count() <= 6 for previous in phashes):
            raise ValueError("Near-duplicate image appeared in the public training pack")
        phashes.append(phash)
        paths.append(path)
        captions.append(caption)
        groups.append(group)
        splits.append(split)
    return manifest, hashlib.sha256(raw).hexdigest(), paths, captions, groups, splits


def train_local_pack(directory, settings):
    import torch
    torch.set_num_threads(6)
    manifest, digest, paths, captions, groups, splits = read_pack(directory)
    directory = Path(directory)
    print(f"Encoding {len(paths)} public image-caption pairs with cached CLIP; no downloads", flush=True)
    images, texts = embed_pairs(settings, paths, captions)
    np.savez_compressed(directory / "public-embeddings.npz", images=images, texts=texts)
    metadata = {"encoder_model": settings.clip_model, "encoder_fingerprint": encoder_fingerprint(settings), "manifest_sha256": digest,
                "dataset": manifest["image_dataset"], "dataset_revision": manifest["image_revision"],
                "source_split": "train", "local_split": "source-image/pHash deduplicated 75/12.5/12.5",
                "license": "CC BY 4.0" if manifest["image_dataset"].endswith("440k") else "CC BY-SA 4.0",
                "frozen_encoder": True, "private_transcripts_used": False}
    print("Training small relevance model; calibrating and evaluating separate source groups", flush=True)
    artifact, report = train_adapter(images, texts, groups, splits, metadata)
    report["scope"] = "Image-caption pairing benchmark, not audiobook scene relevance"
    report["limitations"] = ["Small museum subset may not transfer to contemporary photographs or audiobook scenes.",
                             "Wrong-source captions are weak negative labels; similar captions may describe multiple images.",
                             "Qwen and CLIP weights are unchanged. Only the small relevance head was trained."]
    artifact["report"] = report
    save_adapter(directory / "relevance-model.json", artifact)
    (directory / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train a small local relevance head using an already prepared public pack")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True, help="Existing app cache containing complete CLIP weights")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    settings = Settings(cache_dir=args.cache_dir, clip_device=args.device).validate()
    train_local_pack(args.pack.resolve(), settings)


if __name__ == "__main__":
    main()
