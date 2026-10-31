import argparse
import hashlib
import heapq
import json
import math
import re
import statistics
import time
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def top_ranked_documents(tokens, index, chunk_ids, top_k, zero_score_order):
    """Return document indices ordered by BM25 score, then chunk ID for deterministic ties."""
    document_lengths = index["document_lengths"]
    postings = index["postings"]
    document_count = len(document_lengths)
    average_document_length = index["average_document_length"]
    k1, b = index["k1"], index["b"]
    scores = {}

    # Query term frequency is intentionally ignored: repeated query words should not
    # give a different ranking for the same set of query concepts.
    for term in set(tokens):
        term_postings = postings.get(term, [])
        if not term_postings:
            continue
        document_frequency = len(term_postings)
        idf = math.log(1 + (document_count - document_frequency + 0.5) /
                       (document_frequency + 0.5))
        for document_id, term_frequency in term_postings:
            length_normalizer = 1 - b + b * document_lengths[document_id] / average_document_length
            term_score = idf * (term_frequency * (k1 + 1)) / (
                term_frequency + k1 * length_normalizer
            )
            scores[document_id] = scores.get(document_id, 0.0) + term_score

    best = heapq.nsmallest(
        top_k, scores.items(), key=lambda item: (-item[1], chunk_ids[item[0]])
    )
    ranked = [(document_id, score) for document_id, score in best]
    if len(ranked) < top_k:
        scored_ids = set(scores)
        ranked.extend((document_id, 0.0) for document_id in zero_score_order
                      if document_id not in scored_ids and len(ranked) < top_k)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-K chunks from a BM25 index.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Run file (.jsonl)")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("top-k must be positive")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Run outputs already exist; choose a new output path")

    index_manifest_path = args.index_dir / "bm25_index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if index_manifest.get("retriever") != "bm25":
        parser.error("Index manifest is not a BM25 index")
    if sha256(args.processed_dir / "corpus.jsonl") != index_manifest["corpus_sha256"]:
        parser.error("The index was built from a different corpus.jsonl")
    for name, digest in index_manifest["output_sha256"].items():
        if sha256(args.index_dir / name) != digest:
            parser.error(f"{name} differs from the index manifest")

    index = json.loads((args.index_dir / "bm25_index.json").read_text(encoding="utf-8"))
    chunk_ids = json.loads((args.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    if len(index["document_lengths"]) != len(chunk_ids):
        parser.error("BM25 index and chunk_ids.json have different document counts")
    if args.top_k > len(chunk_ids):
        parser.error(f"top-k ({args.top_k}) exceeds the {len(chunk_ids)} indexed chunks")
    queries = list(read_jsonl(args.processed_dir / "queries.jsonl"))
    if not queries:
        parser.error("queries.jsonl contains no queries")
    query_ids = []
    for query in queries:
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            parser.error("queries.jsonl contains a missing or invalid query_id")
        query_ids.append(query_id)
    if len(set(query_ids)) != len(query_ids):
        parser.error("queries.jsonl contains duplicate query_id values")

    zero_score_order = sorted(range(len(chunk_ids)), key=lambda document_id: chunk_ids[document_id])
    latencies = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        for query in queries:
            if not isinstance(query.get("text"), str):
                parser.error(f"Query {query['query_id']} has non-string text")
            start = time.perf_counter()
            ranked = top_ranked_documents(
                tokenize(query["text"]), index, chunk_ids, args.top_k, zero_score_order
            )
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            for rank, (document_id, score) in enumerate(ranked, start=1):
                file.write(json.dumps({
                    "query_id": query["query_id"],
                    "chunk_id": chunk_ids[document_id],
                    "rank": rank,
                    "score": round(score, 6),
                    "latency_ms": round(latency_ms, 3),
                }) + "\n")

    latencies.sort()
    result = {
        "schema_version": 1,
        "retriever": "bm25",
        "tokenizer": index_manifest["tokenizer"],
        "k1": index_manifest["k1"],
        "b": index_manifest["b"],
        "split": index_manifest["split"],
        "top_k": args.top_k,
        "counts": {"queries": len(queries)},
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(latencies[math.ceil(len(latencies) * 0.95) - 1], 3),
        },
        "input_sha256": {
            "queries.jsonl": sha256(args.processed_dir / "queries.jsonl"),
            "bm25_index_manifest.json": sha256(index_manifest_path),
        },
        "output_sha256": {args.output.name: sha256(args.output)},
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
