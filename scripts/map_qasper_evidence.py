import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def match_evidence(text, doc_id, exact, normalized, headings):
    key = " ".join(text.split())
    if not key:
        return "empty", []
    if key.startswith("FLOAT SELECTED"):
        return "figure_or_table", []
    matches = exact.get((doc_id, text), [])
    status = "exact"
    if not matches:
        matches = normalized.get((doc_id, key), [])
        status = "whitespace_match"
    if not matches:
        return ("section_heading" if (doc_id, key) in headings else "unmatched"), []
    return ("ambiguous" if len(matches) > 1 else status), matches


def main():
    parser = argparse.ArgumentParser(description="Export QASPER questions and map annotated evidence.")
    parser.add_argument("input", type=Path, help="Original QASPER JSON")
    parser.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.processed_dir
    paths = {name: root / f"{name}.jsonl" for name in ("queries", "gold", "evidence_audit")}
    manifest_path = root / "evidence_manifest.json"
    if any(path.exists() for path in [*paths.values(), manifest_path]):
        parser.error("Evidence outputs already exist; move them aside before rerunning")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw = args.input.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != manifest["source_sha256"]:
        parser.error("Raw input does not match the source used to prepare this corpus")
    input_hashes = {}
    for name in ("paragraphs.jsonl", "corpus.jsonl"):
        input_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if input_hashes[name] != manifest["output_sha256"][name]:
            parser.error(f"{name} differs from the preprocessing manifest")
    input_hashes["manifest.json"] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()

    exact, normalized, chunk_ids = defaultdict(list), defaultdict(list), defaultdict(list)
    paragraphs = {}
    for paragraph in read_jsonl(root / "paragraphs.jsonl"):
        pid = paragraph["paragraph_id"]
        if pid in paragraphs or paragraph["split"] != manifest["split"]:
            parser.error("Duplicate paragraph ID or incorrect paragraph split")
        paragraphs[pid] = paragraph
        exact[paragraph["doc_id"], paragraph["text"]].append(pid)
        normalized[paragraph["doc_id"], " ".join(paragraph["text"].split())].append(pid)
    seen_chunks = set()
    for chunk in read_jsonl(root / "corpus.jsonl"):
        pid = chunk["paragraph_id"]
        if pid not in paragraphs or chunk["doc_id"] != paragraphs[pid]["doc_id"]:
            parser.error("Chunk has a missing or incorrect parent paragraph")
        if chunk["chunk_id"] in seen_chunks:
            parser.error("Duplicate chunk ID")
        seen_chunks.add(chunk["chunk_id"])
        chunk_ids[pid].append(chunk["chunk_id"])
    if set(chunk_ids) != set(paragraphs):
        parser.error("Some paragraphs have no chunks")

    papers = json.loads(raw)
    headings = {(f"qasper:{paper_id}", " ".join((s["section_name"] or "").split()))
                for paper_id, paper in papers.items() for s in paper["full_text"]}
    queries, gold, audit = [], [], []
    statuses, answerability_counts = Counter(), Counter()
    seen_queries = set()
    for paper_id, paper in sorted(papers.items()):
        doc_id = f"qasper:{paper_id}"
        for qa in sorted(paper["qas"], key=lambda q: q["question_id"]):
            qid = f"qasper:{qa['question_id']}"
            if qid in seen_queries:
                parser.error(f"Duplicate question ID: {qid}")
            seen_queries.add(qid)
            queries.append({"query_id": qid, "source_id": qa["question_id"],
                            "doc_id": doc_id, "split": manifest["split"], "text": qa["question"]})
            annotations, question_paragraphs = [], set()
            flags = {a["answer"]["unanswerable"] for a in qa["answers"]}
            answerability = ({False: "answerable", True: "unanswerable"}[next(iter(flags))]
                             if len(flags) == 1 else "disagreement" if flags else "no_annotations")
            answerability_counts[answerability] += 1
            complete = answerability == "answerable"
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                mappings, matched_paragraphs = [], set()
                if not answer["evidence"]:
                    complete = False
                for text in answer["evidence"]:
                    status, candidates = match_evidence(text, doc_id, exact, normalized, headings)
                    statuses[status] += 1
                    mappings.append({"text": text, "status": status, "paragraph_ids": candidates})
                    if status not in {"exact", "whitespace_match"}:
                        complete = False
                        audit.append({"query_id": qid, "annotation_id": annotation["annotation_id"],
                                      "text": text, "status": status, "paragraph_ids": candidates})
                    elif not answer["unanswerable"]:
                        matched_paragraphs.update(candidates)
                # Full original answers preserve yes/no=False, extractive spans,
                # free-form text, and highlighted evidence without reinterpretation.
                annotations.append({"annotation_id": annotation["annotation_id"], "answer": answer,
                                    "evidence_mapping": mappings,
                                    "paragraph_ids": sorted(matched_paragraphs),
                                    "chunk_ids": sorted(c for p in matched_paragraphs for c in chunk_ids[p])})
                question_paragraphs.update(matched_paragraphs)
            gold.append({"query_id": qid, "answerability": answerability,
                         "all_evidence_mapped": complete, "annotations": annotations,
                         "paragraph_ids": sorted(question_paragraphs),
                         "chunk_ids": sorted(c for p in question_paragraphs for c in chunk_ids[p])})

    rows = {"queries": queries, "gold": gold, "evidence_audit": audit}
    for name, path in paths.items():
        write_jsonl(path, rows[name])
    counts = {"questions": len(queries), "annotations": sum(len(g["annotations"]) for g in gold),
              "questions_with_mapped_evidence": sum(bool(g["paragraph_ids"]) for g in gold),
              "questions_with_all_evidence_mapped": sum(g["all_evidence_mapped"] for g in gold),
              "answerability": dict(sorted(answerability_counts.items())),
              "evidence_statuses": dict(sorted(statuses.items()))}
    result = {"schema_version": 1, "dataset": "qasper", "split": manifest["split"],
              "source_file": args.input.name, "source_sha256": source_hash,
              "input_sha256": input_hashes, "counts": counts,
              "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in paths.values()}}
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
