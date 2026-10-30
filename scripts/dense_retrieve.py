import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-K chunks for each query from a dense index.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Run file (.jsonl)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", help="Default: sentence-transformers' automatic choice")
    args = parser.parse_args()

    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Run outputs already exist; choose a new output path")

    index_manifest_path = args.index_dir / "dense_index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if sha256(args.processed_dir / "corpus.jsonl") != index_manifest["corpus_sha256"]:
        parser.error("The index was built from a different corpus.jsonl")
    for name, digest in index_manifest["output_sha256"].items():
        if sha256(args.index_dir / name) != digest:
            parser.error(f"{name} differs from the index manifest")
    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    chunk_ids = json.loads((args.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    queries = list(read_jsonl(args.processed_dir / "queries.jsonl"))

    model = SentenceTransformer(index_manifest["model"], device=args.device)
    # Warm up so model loading and kernel compilation are not charged to the first query.
    model.encode(["warm-up query"], normalize_embeddings=True)

    latencies = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        for query in queries:
            # Latency covers query encoding and index search, one query at a time.
            start = time.perf_counter()
            vector = model.encode([query["text"]], normalize_embeddings=True, convert_to_numpy=True)
            scores, rows = index.search(vector, args.top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            for rank, (score, row) in enumerate(zip(scores[0], rows[0]), start=1):
                file.write(json.dumps({"query_id": query["query_id"], "chunk_id": chunk_ids[row],
                                       "rank": rank, "score": round(float(score), 6),
                                       "latency_ms": round(latency_ms, 3)}) + "\n")

    latencies.sort()
    result = {
        "schema_version": 1, "retriever": "dense", "model": index_manifest["model"],
        "device": str(model.device), "split": index_manifest["split"], "top_k": args.top_k,
        "counts": {"queries": len(queries)},
        "latency_ms": {"mean": round(statistics.mean(latencies), 3),
                       "p50": round(latencies[len(latencies) // 2], 3),
                       "p95": round(latencies[int(len(latencies) * 0.95)], 3)},
        "input_sha256": {"queries.jsonl": sha256(args.processed_dir / "queries.jsonl"),
                         "dense_index_manifest.json": sha256(index_manifest_path)},
        "output_sha256": {args.output.name: sha256(args.output)},
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
