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
