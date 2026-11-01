import json
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_retrieval.py"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def annotation(evidence, status):
    return {"annotation_id": "a", "answer": {"evidence": evidence, "unanswerable": False},
            "evidence_mapping": [{"text": evidence[0] if evidence else "", "status": status,
                                  "paragraph_ids": []}],
            "paragraph_ids": [], "chunk_ids": []}


def build_fixture(root):
    """Two chunks of paragraph p1 are one evidence item; q3's gold paragraph is only
    reachable through a zero-score BM25 padding row."""
    processed = root / "processed"
    processed.mkdir()
    corpus = processed / "corpus.jsonl"
    write_jsonl(corpus, [
        {"chunk_id": "c1a", "paragraph_id": "p1", "text": "evidence first half"},
        {"chunk_id": "c1b", "paragraph_id": "p1", "text": "evidence second half"},
        {"chunk_id": "c2", "paragraph_id": "p2", "text": "other paragraph"},
        {"chunk_id": "c3", "paragraph_id": "p3", "text": "unrelated paragraph"},
    ])
    queries = processed / "queries.jsonl"
    write_jsonl(queries, [{"query_id": q, "doc_id": "d1", "text": q} for q in ("q1", "q2", "q3")])
    gold = processed / "gold.jsonl"
    write_jsonl(gold, [
        {"query_id": "q1", "answerability": "answerable", "all_evidence_mapped": True,
         "annotations": [annotation(["evidence first half"], "exact")],
         "paragraph_ids": ["p1"], "chunk_ids": ["c1a", "c1b"]},
        {"query_id": "q2", "answerability": "unanswerable", "all_evidence_mapped": False,
         "annotations": [annotation([], "empty")], "paragraph_ids": [], "chunk_ids": []},
        {"query_id": "q3", "answerability": "answerable", "all_evidence_mapped": True,
         "annotations": [annotation(["other paragraph"], "exact")],
         "paragraph_ids": ["p2"], "chunk_ids": ["c2"]},
    ])
    (processed / "manifest.json").write_text(json.dumps({
        "dataset": "qasper", "split": "test",
        "output_sha256": {"corpus.jsonl": digest(corpus)},
    }), encoding="utf-8")
    (processed / "evidence_manifest.json").write_text(json.dumps({
        "input_sha256": {"corpus.jsonl": digest(corpus)},
        "output_sha256": {"queries.jsonl": digest(queries), "gold.jsonl": digest(gold)},
    }), encoding="utf-8")

    run = root / "bm25.jsonl"
    write_jsonl(run, [
        {"query_id": "q1", "chunk_id": "c1a", "rank": 1, "score": 2.0, "latency_ms": 1.0},
        {"query_id": "q1", "chunk_id": "c1b", "rank": 2, "score": 1.0, "latency_ms": 1.0},
        {"query_id": "q3", "chunk_id": "c3", "rank": 1, "score": 1.5, "latency_ms": 3.0},
        {"query_id": "q3", "chunk_id": "c2", "rank": 2, "score": 0.0, "latency_ms": 3.0},
    ])
    run.with_suffix(".manifest.json").write_text(json.dumps({
        "schema_version": 1, "retriever": "bm25", "split": "test", "top_k": 2,
        "counts": {"queries": 3},
        "input_sha256": {"queries.jsonl": digest(queries)},
        "output_sha256": {"bm25.jsonl": digest(run)},
    }), encoding="utf-8")
    return processed, run


def evaluate(processed, run, output, *extra):
    completed = subprocess.run(
        [sys.executable, str(EVAL), "--processed-dir", str(processed), "--run", f"bm25={run}",
         "--output", str(output), "--recall-at", "1", "2", "--mrr-at", "2", *extra],
        check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf-8")), completed.stdout


class EvalRetrievalTest(unittest.TestCase):
    def test_paragraph_credit_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, run = build_fixture(root)
            summary, stdout = evaluate(processed, run, root / "summary.json")

            eligibility = summary["eligibility"]
            self.assertEqual(eligibility["eligible_queries"], 2)
            self.assertEqual(eligibility["excluded_by_reason"], {"answerability:unanswerable": 1})
            self.assertEqual(eligibility["eligible_queries"] + eligibility["excluded_queries"],
                             eligibility["total_queries"])

            metrics = summary["runs"]["bm25"]["metrics"]
            # q1: both retrieved chunks belong to gold paragraph p1, so recall is 1/1, never 2/1.
            # q3: its gold paragraph p2 appears only in a zero-score padding row, which earns nothing.
            self.assertEqual(metrics["recall_at"]["1"], 0.5)
            self.assertEqual(metrics["recall_at"]["2"], 0.5)
            self.assertEqual(metrics["mrr_at_2"], 0.5)
            self.assertEqual(summary["runs"]["bm25"]["padding_dropped_rows"], 1)
            self.assertEqual(summary["runs"]["bm25"]["diagnostics"]["mean_distinct_paragraphs"], 1.0)
            self.assertIn("eligible 2 / 3", stdout)
            self.assertTrue((root / "summary.manifest.json").exists())

    def test_keeping_padding_rows_would_inflate_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, run = build_fixture(root)
            summary, _ = evaluate(processed, run, root / "kept.json", "--padding-policy", "keep-all")
            self.assertEqual(summary["runs"]["bm25"]["metrics"]["recall_at"]["2"], 1.0)
            self.assertEqual(summary["runs"]["bm25"]["padding_dropped_rows"], 0)

    def test_refuses_run_from_another_query_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, run = build_fixture(root)
            manifest_path = run.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_sha256"]["queries.jsonl"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError) as error:
                evaluate(processed, run, root / "rejected.json")
            self.assertIn("produced from a different queries.jsonl", error.exception.stderr)


if __name__ == "__main__":
    unittest.main()
