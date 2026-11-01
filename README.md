# Adaptive Hybrid RAG

Adaptive Hybrid RAG explores the trade-off between retrieval quality and latency
in retrieval-augmented generation. The approach combines BM25 and dense retrieval
using Reciprocal Rank Fusion (RRF), then uses confidence in the fused ranking to
decide when to apply cross-encoder reranking.

## Getting started

Use **Python 3.11**. Run these commands from the repository root to create an
environment and install dependencies. If `.venv` already exists, reuse it and
skip the first command.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preprocessing scripts use only the Python standard library. The requirements
file records the additional dependencies used for data analysis.

## Download QASPER

Download the original QASPER v0.3 train/dev archive and extract its JSON files:

```bash
mkdir -p data/raw/qasper

curl --fail --location \
  --output data/raw/qasper/qasper-train-dev-v0.3.tgz \
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

tar -xzf data/raw/qasper/qasper-train-dev-v0.3.tgz \
  -C data/raw/qasper \
  qasper-train-v0.3.json qasper-dev-v0.3.json
```

The download is a gzip-compressed tar archive (`.tgz`); the extracted datasets
are JSON files, each keyed by paper ID. The resulting paths are:

```text
data/raw/qasper/qasper-train-v0.3.json
data/raw/qasper/qasper-dev-v0.3.json
```

## Prepare a dataset

First create source paragraphs and retrieval chunks, then map the questions'
annotated evidence to those sources:

```bash
python scripts/prepare_qasper.py \
  data/raw/qasper/qasper-train-v0.3.json \
  --split train --output-dir data/processed/qasper/train

python scripts/map_qasper_evidence.py \
  data/raw/qasper/qasper-train-v0.3.json \
  --processed-dir data/processed/qasper/train
```

To prepare dev, use its input file and output directory:

```bash
python scripts/prepare_qasper.py \
  data/raw/qasper/qasper-dev-v0.3.json \
  --split dev --output-dir data/processed/qasper/dev

python scripts/map_qasper_evidence.py \
  data/raw/qasper/qasper-dev-v0.3.json \
  --processed-dir data/processed/qasper/dev
```

Chunks default to **180 words with 30 words of overlap** and stay within one
source paragraph. Change these values with `--chunk-words` and `--overlap-words`.
The scripts refuse to overwrite existing outputs. For another configuration,
choose a new output directory and use it for both commands.

## Generated files

Each processed split contains:

| File | Contents |
|---|---|
| `corpus.jsonl` | Searchable chunks, source IDs, metadata, and word/character offsets |
| `paragraphs.jsonl` | Complete source paragraphs and abstracts |
| `queries.jsonl` | All questions and their IDs |
| `gold.jsonl` | Original human answers and evidence-to-source mappings |
| `evidence_audit.jsonl` | Evidence that needs attention before paragraph-based evaluation |
| `manifest.json` | Input checksum, chunk settings, counts, and corpus checksums |
| `evidence_manifest.json` | Mapping input/output checksums and coverage counts |

JSONL files contain one JSON object per line. Index `corpus.jsonl`, read questions
from `queries.jsonl`, and join reference annotations from `gold.jsonl` using
`query_id`. Keep reference answers and evidence outside retrieval indexes.

Raw datasets and generated outputs are excluded from Git. Small manifest copies
under `data/manifests/` record reproducible dataset builds.

See [the preprocessing reference](docs/preprocessing.md) for ID relationships,
evidence-matching rules, limitations, experiment conventions, and manifest usage.

## Sparse retrieval (BM25)

Build a BM25 index over the same corpus used for dense retrieval, then retrieve
the top 100 chunks for every question:

```bash
python scripts/build_bm25_index.py \
  --processed-dir data/processed/qasper/train \
  --output-dir data/indexes/qasper-train-bm25

python scripts/bm25_retrieve.py \
  --index-dir data/indexes/qasper-train-bm25 \
  --processed-dir data/processed/qasper/train \
  --output runs/qasper-train/bm25.jsonl
```

The implementation lowercases Unicode word tokens and uses BM25 with `k1=1.5`
and `b=0.75` by default. The index and retrieval commands verify checksums so
an index cannot be accidentally used with a different corpus. The run uses the
same JSONL schema as dense retrieval: `query_id`, `chunk_id`, `rank`, `score`,
and `latency_ms`.

## Dense retrieval

Embed a prepared split with `BAAI/bge-m3` and build an exact FAISS index, then
retrieve the top 100 chunks for every question:

```bash
python scripts/build_dense_index.py \
  --processed-dir data/processed/qasper/train \
  --output-dir data/indexes/qasper-train-bge-m3

python scripts/dense_retrieve.py \
  --index-dir data/indexes/qasper-train-bge-m3 \
  --processed-dir data/processed/qasper/train \
  --output runs/qasper-train/dense-bge-m3.jsonl
```

The model accepts up to 8,192 tokens; the longest QASPER chunk is about 1,100,
so no chunk is truncated. The index builder refuses to run if any chunk would
exceed the model limit. Embeddings are normalized, so scores are cosine
similarities. The first run downloads the model (about 2.3 GB).

Each line of the run file is one retrieved chunk:

```json
{"query_id": "qasper:...", "chunk_id": "qasper:...", "rank": 1, "score": 0.71, "latency_ms": 18.4}
```

`latency_ms` is per query (query encoding plus index search, after a warm-up
query) and repeats on each of that query's rows. A `.manifest.json` next to the
run records the model, device, latency summary, and checksums. Indexes and runs
are excluded from Git.

## Sparse retrieval

Build a BM25 index over a prepared split and retrieve the top 100 chunks for
every question. Both scripts use only the Python standard library:

```bash
python scripts/build_bm25_index.py \
  --processed-dir data/processed/qasper/train \
  --output-dir data/indexes/qasper-train-bm25

python scripts/bm25_retrieve.py \
  --index-dir data/indexes/qasper-train-bm25 \
  --processed-dir data/processed/qasper/train \
  --output runs/qasper-train/bm25.jsonl
```

Run files use the same schema as dense retrieval, so both are scored by the
same evaluator. Change `--k1`, `--b`, or `--tokenizer` on the index builder;
retrieval reads those settings from the index manifest rather than the command
line. When fewer than `--top-k` chunks share a term with the query, the run is
padded with `score` 0.0 rows ordered by `chunk_id`; those rows are not
retrievals and the evaluator drops them.

## Evaluate runs

Score one or more run files against the annotated evidence paragraphs:

```bash
python scripts/eval_retrieval.py \
  --processed-dir data/processed/qasper/train \
  --run bm25=runs/qasper-train/bm25.jsonl \
  --run dense-bge-m3=runs/qasper-train/dense-bge-m3.jsonl \
  --index-dir bm25=data/indexes/qasper-train-bm25 \
  --index-dir dense-bge-m3=data/indexes/qasper-train-bge-m3 \
  --reference-manifest data/manifests/qasper-train.json \
  --output runs/qasper-train/eval/retrieval-baselines.json \
  --per-query-dir runs/qasper-train/eval/per-query
```

Scoring several runs in one invocation is what makes them comparable: they are
forced to share one eligible question set, one chunk-to-paragraph fold, and one
latency estimator. Each `--index-dir` is optional and verifies that the run came
from an index built on this exact corpus; without it the evaluator only checks
that every retrieved `chunk_id` exists in the corpus.

**Credit is attributed per paragraph, not per chunk.** Retrieved chunks are
folded onto their `paragraph_id`, keeping each paragraph's best rank, before
anything is scored. Overlapping chunks from one gold paragraph are a single
evidence item, so `k` counts distinct paragraphs. `recall_at_chunk_depth` in the
summary applies the same cutoffs at chunk depth instead; the two agree at the
largest `k`.

The default `--eligibility all-evidence-mapped` scores only questions whose
every evidence entry mapped unambiguously to a paragraph (1,836 of 2,593 on
train). The summary records the policy, the eligible count, and one mutually
exclusive exclusion reason per remaining question. `--eligibility
any-evidence-mapped` widens this to every question with at least one mapped
paragraph (2,098 on train) for a sensitivity check; its recall denominators come
from an incomplete gold set, so it does not belong in a headline table.

Latency percentiles are recomputed here with a single nearest-rank estimator
over one `latency_ms` per eligible query. The percentiles each retrieval script
writes into its own run manifest use different estimators and must not be
compared with each other.

Outputs are a summary JSON, a `.manifest.json` beside it recording every input
and output checksum, and one JSONL per run under `--per-query-dir` for error
analysis. The script refuses to overwrite existing outputs.
