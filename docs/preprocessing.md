# QASPER preprocessing reference

This document describes the implemented data-preparation component and its
interface to downstream retrieval and evaluation. For setup and runnable
commands, see the [project README](../README.md).

## Implementation status and scope

Implemented: QASPER v0.3 source extraction, overlapping chunk construction,
question export, gold-evidence mapping, manifests, and an EDA notebook.

| Component | Responsibility |
|---|---|
| `scripts/prepare_qasper.py` | Extract source paragraphs, construct chunks, preserve provenance, and record checksums |
| `scripts/map_qasper_evidence.py` | Export all questions and original answers; map evidence text to paragraph and chunk IDs |
| `notebooks/qasper_eda.ipynb` | Inspect dataset structure, distributions, and mapping coverage without modifying files |

Not yet implemented: sparse/dense retrieval, RRF, cross-encoder reranking,
adaptive thresholds, evaluation metrics, answer generation, and SciFact
preparation. These downstream components consume the prepared QASPER files.

## Running exploratory analysis

Install the pinned analysis dependencies with
`python -m pip install -r requirements.txt` in the activated `.venv`. In VS Code,
open `notebooks/qasper_eda.ipynb`, select `.venv/bin/python` as the kernel, and
choose **Run All**. The raw train/dev files are required; sections concerning
processed data skip missing generated files. The notebook locates the repository
from the root or `notebooks/`, reads data without modifying it, and displays
tables and plots. Clear its cell outputs before committing.

## Source extraction and chunking

The raw input is the original QASPER v0.3 JSON object keyed by paper ID. Each
paper supplies a title, abstract, full-text sections, questions with answer
annotations, and figure/table captions.

The current corpus includes abstracts and body paragraphs. Empty strings and
whitespace-only entries are skipped. Retained paragraph text is unchanged:
case, equations, punctuation, citation markers, and whitespace are preserved.
Titles and section names remain metadata. Figure/table images and captions are
not included in chunk text.

Each paragraph is split into windows of up to 180 whitespace-delimited words,
with 30 overlapping words by default. A short paragraph produces one short
chunk; windows do not cross paragraph boundaries. The final window may be
shorter. Model subword tokens can differ substantially from these word counts;
the retrieval implementation should check its tokenizer's lengths/truncation.

Chunk text is the exact source slice `paragraph_text[char_start:char_end]`.
Leading/trailing whitespace outside the window's first and last words is not
included; internal whitespace is preserved. Both character and word positions
use exclusive ends. Character positions are Python string indices, not bytes.

## IDs and joins

| Field | Meaning |
|---|---|
| `doc_id` | `qasper:` followed by the original paper ID |
| `paragraph_id` | Document ID plus `abstract`, or section/paragraph positions such as `s0:p2` |
| `chunk_id` | Paragraph ID plus chunk size, overlap, and word range |
| `query_id` | `qasper:` followed by the original question ID |
| Query `source_id` | Original question ID without the prefix |
| `annotation_id` | Original identifier of one human answer annotation |

These IDs are deterministic for the same source positions and settings; they
are not content hashes. Use manifest checksums to identify the actual dataset
build. Source indices start at zero; skipping a blank paragraph does not
renumber the remaining paragraphs.

Index the chunk `text` from `corpus.jsonl`, retaining its IDs and metadata.
Run questions from `queries.jsonl`. Join each query to its record in `gold.jsonl`
using `query_id`. Returned chunks link back to `paragraphs.jsonl` through
`paragraph_id` and to their source paper through `doc_id`.

## Evidence mapping

Each raw human answer has an `evidence` list of strings. The mapper looks for
each string within that question's source paper:

1. Match the exact paragraph text.
2. If there is no exact match, compare after collapsing whitespace.
3. Record any ambiguity or unsupported evidence in the audit.

Whitespace normalization only assists matching; it never changes the stored
source or answer text. Matches are paper-local, so identical text in a different
paper cannot become a positive for the question.

Each gold record retains every annotation's original `answer` dictionary,
including `unanswerable`, `yes_no` (where false is a valid No answer), extractive
spans, free-form answers, evidence, and highlighted evidence. Annotator worker
IDs and question-writer background metadata are not exported; they remain in
the raw JSON.

An annotation's `paragraph_ids` contains its unambiguous matches only when the
annotation is answerable. Its `chunk_ids` contains all chunks derived from those
paragraphs. Top-level IDs are the union across annotations. Detailed
`evidence_mapping` entries preserve the text, match status, and candidate IDs.

| Audit status | Interpretation |
|---|---|
| `ambiguous` | Multiple paragraphs match; candidate IDs are retained but excluded from automatic gold-positive IDs |
| `section_heading` | The evidence matches a section name rather than an indexed paragraph |
| `figure_or_table` | Evidence begins with `FLOAT SELECTED`; corresponding images/captions are not indexed |
| `unmatched` | No paragraph or recognized heading matches |
| `empty` | An evidence entry contains no text |

Audit records come from human evidence annotations. Ordinary paragraphs that
were never selected as evidence do not enter the audit. An annotation with an
empty evidence list has no entries to audit, but prevents complete mapping.
No question is removed from the exported queries or gold records.

`answerability` is `answerable` when all annotators mark the question answerable,
`unanswerable` when all mark it unanswerable, `disagreement` when these flags
differ, or `no_annotations` if there are no answers. This does not measure
disagreement over answer wording or correctness.

`all_evidence_mapped` is true only when all annotations are answerable, each has
nonempty evidence, and every entry has an unambiguous exact or whitespace match.
It is a conservative coverage flag, not a retrieval score or a course requirement.
At the current default build, it is true for 1,836 train and 646 dev questions.

## Retrieval and evaluation conventions

The current plan uses all 2,593 train questions for development and milestone
experiments, reserving all 1,005 dev questions for held-out project evaluation
after settings are fixed. These are experiment choices, not restrictions
imposed by the source filenames. The notebook provides descriptive dataset
statistics; it does not evaluate retrieval systems or tune their settings.

For the current setup, search all papers in the chosen split. A query's parent
`doc_id` is provenance, not an automatic candidate filter. All compared systems
must use identical corpus/chunking, questions, and evidence-handling rules.
Read the complete `queries.jsonl`; no separate question-ID list is necessary.

The evaluation implementation must state its annotation policy, eligible
question count, and exclusions. Overlapping chunks from one gold paragraph
are not multiple independently required pieces of evidence. The linked chunks
contain parts of an evidence paragraph; they are not independently verified
answer spans. Duplicate matching paragraphs may be treated as equivalent
locations by a future evaluation policy, but the current mapper conservatively
excludes them from automatic positives.

## Reproducibility

`manifest.json` records the raw input checksum, split, numeric chunk settings,
counts, and output checksums. Before mapping, the second script verifies that
the raw JSON and prepared paragraph/corpus files match that manifest.
`evidence_manifest.json` records its own input/output checksums and coverage.
Neither manifest updates continuously; each is generated by its script.

Both scripts refuse to overwrite their outputs. For another configuration,
prepare a fresh directory, map evidence against it, and rebuild all compared
retrieval indexes. Keep matching manifests with reported experiment results.

Only small manifest copies are tracked; raw datasets and generated JSONL files
are ignored. To record a prepared build:

```bash
mkdir -p data/manifests
cp data/processed/qasper/train/manifest.json data/manifests/qasper-train.json
cp data/processed/qasper/train/evidence_manifest.json data/manifests/qasper-train-evidence.json
cp data/processed/qasper/dev/manifest.json data/manifests/qasper-dev.json
cp data/processed/qasper/dev/evidence_manifest.json data/manifests/qasper-dev-evidence.json
```

These copies do not refresh automatically. Notebook code is versioned with cell
outputs cleared; executing it locally displays statistics and figures without
rewriting the datasets or preprocessing manifests.
