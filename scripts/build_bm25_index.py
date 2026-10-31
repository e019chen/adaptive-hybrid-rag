import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
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
    """Lowercase Unicode word tokenizer shared by index construction and retrieval."""
    return TOKEN_PATTERN.findall(text.lower())


def main():
    parser = argparse.ArgumentParser(description="Build a reproducible BM25 index over corpus.jsonl.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--tokenizer", choices=["regex-lower"], default="regex-lower")
    args = parser.parse_args()

    if args.k1 < 0:
        parser.error("k1 must be nonnegative")
    if not 0 <= args.b <= 1:
        parser.error("b must be between 0 and 1")

    paths = {name: args.output_dir / name for name in (
        "bm25_index.json", "chunk_ids.json", "bm25_index_manifest.json"
    )}
    if any(path.exists() for path in paths.values()):
        parser.error("Index outputs already exist; choose a new output directory")

    manifest_path = args.processed_dir / "manifest.json"
    corpus_path = args.processed_dir / "corpus.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_hash = sha256(corpus_path)
    if corpus_hash != manifest["output_sha256"]["corpus.jsonl"]:
        parser.error("corpus.jsonl differs from the preprocessing manifest")

    chunk_ids, document_lengths = [], []
    postings = defaultdict(list)
    seen_chunk_ids = set()
    for document_id, chunk in enumerate(read_jsonl(corpus_path)):
        try:
            chunk_id, text = chunk["chunk_id"], chunk["text"]
        except KeyError as error:
            parser.error(f"corpus.jsonl record {document_id + 1} is missing {error.args[0]!r}")
        if not isinstance(chunk_id, str) or not chunk_id:
            parser.error(f"corpus.jsonl record {document_id + 1} has an invalid chunk_id")
        if not isinstance(text, str):
            parser.error(f"corpus.jsonl record {document_id + 1} has non-string text")
        if chunk_id in seen_chunk_ids:
            parser.error(f"Duplicate chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        frequencies = Counter(tokenize(text))
        chunk_ids.append(chunk_id)
        document_lengths.append(sum(frequencies.values()))
        for term, frequency in frequencies.items():
            postings[term].append([document_id, frequency])

    if not chunk_ids:
        parser.error("corpus.jsonl contains no chunks")
    if not any(document_lengths):
        parser.error("corpus.jsonl contains no tokens under the selected tokenizer")
    average_document_length = sum(document_lengths) / len(document_lengths)
    index = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "k1": args.k1,
        "b": args.b,
        "document_lengths": document_lengths,
        "average_document_length": average_document_length,
        "postings": dict(postings),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths["bm25_index.json"].write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    paths["chunk_ids.json"].write_text(
        json.dumps(chunk_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "retriever": "bm25",
        "corpus_sha256": corpus_hash,
        "tokenizer": args.tokenizer,
        "k1": args.k1,
        "b": args.b,
        "average_document_length": average_document_length,
        "counts": {"chunks": len(chunk_ids), "terms": len(postings)},
        "output_sha256": {
            name: sha256(paths[name]) for name in ("bm25_index.json", "chunk_ids.json")
        },
    }
    paths["bm25_index_manifest.json"].write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
