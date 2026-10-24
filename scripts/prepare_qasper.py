import argparse
import hashlib
import json
import re
from pathlib import Path


def word_windows(text, size, overlap):
    """Yield overlapping word windows within a single source paragraph."""
    words = list(re.finditer(r"\S+", text))
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end, words[start].start(), words[end - 1].end()
        if end == len(words):
            break
        start = end - overlap


def main():
    parser = argparse.ArgumentParser(description="Prepare QASPER paragraphs and chunks.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=30)
    args = parser.parse_args()

    if not 0 <= args.overlap_words < args.chunk_words:
        parser.error("Require 0 <= overlap-words < chunk-words")
    expected_name = f"qasper-{args.split}-v0.3.json"
    if args.input.name != expected_name:
        parser.error(f"Expected {expected_name} for split {args.split}")

    raw = args.input.read_bytes()
    papers = json.loads(raw)
    if not isinstance(papers, dict) or not papers:
        parser.error("Expected a nonempty original QASPER JSON object")

    paths = {name: args.output_dir / f"{name}.jsonl"
             for name in ["paragraphs", "corpus"]}
    manifest_path = args.output_dir / "manifest.json"
    if any(path.exists() for path in [*paths.values(), manifest_path]):
        parser.error("Output files already exist; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paragraph_count = chunk_count = 0
    policy = f"w{args.chunk_words}-o{args.overlap_words}"
    with paths["paragraphs"].open("x", encoding="utf-8") as paragraphs_file, \
            paths["corpus"].open("x", encoding="utf-8") as corpus_file:
        for paper_id, paper in sorted(papers.items()):
            sources = [("abstract", "Abstract", None, 0, paper["abstract"])]
            for si, section in enumerate(paper["full_text"]):
                for pi, text in enumerate(section["paragraphs"]):
                    sources.append((f"s{si}:p{pi}", section["section_name"] or "", si, pi, text))

            for suffix, section_name, si, pi, text in sources:
                # Skip blank sources; preserve the text of retained paragraphs.
                if not text.strip():
                    continue
                paragraph_id = f"qasper:{paper_id}:{suffix}"
                metadata = {
                    "dataset": "qasper", "split": args.split,
                    "doc_id": f"qasper:{paper_id}",
                    "paragraph_id": paragraph_id, "title": paper["title"],
                    "section_name": section_name, "section_index": si,
                    "paragraph_index": pi,
                    "source_type": "abstract" if si is None else "paragraph",
                }
                paragraphs_file.write(json.dumps({**metadata, "text": text}, ensure_ascii=False) + "\n")
                paragraph_count += 1
                for ws, we, cs, ce in word_windows(text, args.chunk_words, args.overlap_words):
                    chunk = {
                        **metadata, "chunk_id": f"{paragraph_id}:{policy}:{ws}-{we}",
                        "text": text[cs:ce], "word_start": ws, "word_end": we,
                        "char_start": cs, "char_end": ce,
                    }
                    corpus_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1

    manifest = {
        "schema_version": 1, "dataset": "qasper", "version": "0.3",
        "split": args.split, "source_file": args.input.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "chunking": {"size_words": args.chunk_words, "overlap_words": args.overlap_words},
        "counts": {"papers": len(papers), "paragraphs": paragraph_count, "chunks": chunk_count},
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths.values()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
