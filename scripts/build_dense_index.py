import argparse
import hashlib
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Embed prepared chunks and build an exact FAISS index.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", help="Default: sentence-transformers' automatic choice")
    args = parser.parse_args()

    paths = {name: args.output_dir / name
             for name in ("index.faiss", "chunk_ids.json", "dense_index_manifest.json")}
    if any(path.exists() for path in paths.values()):
        parser.error("Index outputs already exist; choose a new output directory")

    manifest = json.loads((args.processed_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus_path = args.processed_dir / "corpus.jsonl"
    corpus_hash = sha256(corpus_path)
    if corpus_hash != manifest["output_sha256"]["corpus.jsonl"]:
        parser.error("corpus.jsonl differs from the preprocessing manifest")
    chunks = list(read_jsonl(corpus_path))
    texts = [chunk["text"] for chunk in chunks]

    model = SentenceTransformer(args.model, device=args.device)
    # Refuse to build if any chunk would be truncated by the encoder.
    token_counts = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    if max(token_counts) > model.max_seq_length:
        parser.error(f"{sum(n > model.max_seq_length for n in token_counts)} chunks exceed "
                     f"the model limit of {model.max_seq_length} tokens")

    start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    encode_seconds = time.perf_counter() - start

    # Normalized vectors make inner product equal to cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths["index.faiss"]))
    paths["chunk_ids.json"].write_text(json.dumps([c["chunk_id"] for c in chunks]) + "\n", encoding="utf-8")

    result = {
        "schema_version": 1, "dataset": manifest["dataset"], "split": manifest["split"],
        "corpus_sha256": corpus_hash, "model": args.model, "device": str(model.device),
        "max_seq_length": model.max_seq_length, "max_chunk_tokens": max(token_counts),
        "embedding_dim": int(embeddings.shape[1]), "similarity": "cosine (IndexFlatIP, normalized)",
        "counts": {"chunks": len(chunks)}, "encode_seconds": round(encode_seconds, 1),
        "output_sha256": {name: sha256(paths[name]) for name in ("index.faiss", "chunk_ids.json")},
    }
    paths["dense_index_manifest.json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
