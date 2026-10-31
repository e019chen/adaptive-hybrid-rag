import json
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_bm25_index.py"
RETRIEVE = ROOT / "scripts" / "bm25_retrieve.py"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class Bm25PipelineTest(unittest.TestCase):
    def test_build_and_retrieve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            corpus = processed / "corpus.jsonl"
            write_jsonl(corpus, [
                {"chunk_id": "c1", "text": "retrieval evidence corpus"},
                {"chunk_id": "c2", "text": "retrieval latency measurement"},
                {"chunk_id": "c3", "text": "dense retrieval semantic similarity"},
            ])
            corpus_hash = sha256(corpus.read_bytes()).hexdigest()
            (processed / "manifest.json").write_text(json.dumps({
                "dataset": "qasper", "split": "test",
                "output_sha256": {"corpus.jsonl": corpus_hash},
            }), encoding="utf-8")
            write_jsonl(processed / "queries.jsonl", [
                {"query_id": "q1", "text": "retrieval latency"},
                {"query_id": "q2", "text": "unseen term"},
            ])

            index_dir = root / "index"
            run = root / "run.jsonl"
            subprocess.run([sys.executable, str(BUILD), "--processed-dir", str(processed),
                            "--output-dir", str(index_dir)], check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, str(RETRIEVE), "--index-dir", str(index_dir),
                            "--processed-dir", str(processed), "--output", str(run), "--top-k", "2"],
                           check=True, capture_output=True, text=True)

            rows = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([(row["chunk_id"], row["rank"]) for row in rows[:2]],
                             [("c2", 1), ("c1", 2)])
            self.assertEqual([(row["chunk_id"], row["rank"], row["score"]) for row in rows[2:]],
                             [("c1", 1, 0.0), ("c2", 2, 0.0)])
            self.assertTrue(run.with_suffix(".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
