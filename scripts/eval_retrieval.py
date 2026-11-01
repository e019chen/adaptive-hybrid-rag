import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


NAME_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
UNMAPPED_STATUS_PRIORITY = ("unmatched", "figure_or_table", "section_heading", "ambiguous", "empty")
ROW_FIELDS = (("query_id", str), ("chunk_id", str), ("rank", int), ("score", float), ("latency_ms", float))


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(sorted_values, quantile):
    """Nearest-rank percentile. Both run manifests use different estimators, so latency
    columns are only comparable when this one recomputes them from the run rows."""
    return sorted_values[max(0, math.ceil(quantile * len(sorted_values)) - 1)]


def parse_named_path(parser, value, flag):
    name, separator, path = value.partition("=")
    if not separator:
        name, path = Path(value).stem, value
    if not name or set(name) - NAME_CHARACTERS:
        parser.error(f"{flag} name {name!r} must be nonempty and use only letters, digits, '.', '_' or '-'")
    return name, Path(path)


def exclusion_reason(record):
    """One mutually exclusive reason per excluded question, mirroring the three conditions
    that map_qasper_evidence.py uses to clear all_evidence_mapped."""
    if record["answerability"] != "answerable":
        return f"answerability:{record['answerability']}"
    statuses = set()
    for annotation in record["annotations"]:
        if not annotation["answer"]["evidence"]:
            return "annotation_without_evidence"
        statuses.update(mapping["status"] for mapping in annotation["evidence_mapping"])
    for status in UNMAPPED_STATUS_PRIORITY:
        if status in statuses:
            return f"unmapped_evidence:{status}"
    return "unattributed"


def grouped_rows(parser, name, path, query_ids, chunk_to_paragraph):
    """Yield (query_id, rows, latency_ms) per query, validating structure as it streams.
    Both producers write a query's rows contiguously, so one query is held at a time."""
    current_id, rows, seen_queries = None, [], set()
    for number, row in enumerate(read_jsonl(path), start=1):
        for field, expected in ROW_FIELDS:
            value = row.get(field)
            if not isinstance(value, expected) and not (expected is float and isinstance(value, int)):
                parser.error(f"Run {name} row {number} has a missing or invalid {field}")
        if row["query_id"] not in query_ids:
            parser.error(f"Run {name} contains unknown query_id {row['query_id']}")
        if row["chunk_id"] not in chunk_to_paragraph:
            parser.error(f"Run {name} row {number} has chunk_id not present in corpus.jsonl")
        if row["query_id"] != current_id:
            if row["query_id"] in seen_queries:
                parser.error(f"Run {name} interleaves rows for query {row['query_id']}")
            if current_id is not None:
                yield current_id, rows, rows[0]["latency_ms"]
            seen_queries.add(row["query_id"])
            current_id, rows = row["query_id"], []
        if row["rank"] != len(rows) + 1:
            parser.error(f"Run {name} query {current_id} has non-contiguous ranks")
        if any(row["chunk_id"] == earlier["chunk_id"] for earlier in rows):
            parser.error(f"Run {name} query {current_id} returned chunk {row['chunk_id']} twice")
        if rows and row["latency_ms"] != rows[0]["latency_ms"]:
            parser.error(f"Run {name} query {current_id} has inconsistent latency_ms")
        rows.append(row)
    if current_id is not None:
        yield current_id, rows, rows[0]["latency_ms"]


def fold_to_paragraphs(rows, chunk_to_paragraph, drop_nonpositive):
    """Overlapping chunks from one paragraph are a single evidence item
    (docs/preprocessing.md), so credit is attributed at paragraph granularity."""
    sequence, paragraphs, seen = [], [], set()
    for row in rows:
        if drop_nonpositive and row["score"] <= 0.0:
            continue
        paragraph_id = chunk_to_paragraph[row["chunk_id"]]
        sequence.append(paragraph_id)
        if paragraph_id not in seen:
            seen.add(paragraph_id)
            paragraphs.append(paragraph_id)
    return sequence, paragraphs


def main():
    parser = argparse.ArgumentParser(description="Score retrieval runs against QASPER gold evidence paragraphs.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                        help="Run file (.jsonl); repeat to score several runs into one table")
    parser.add_argument("--index-dir", action="append", default=[], metavar="NAME=PATH",
                        help="Index directory for a run, to verify the full provenance chain")
    parser.add_argument("--output", type=Path, required=True, help="Summary file (.json)")
    parser.add_argument("--per-query-dir", type=Path, default=None)
    parser.add_argument("--reference-manifest", type=Path, default=None)
    parser.add_argument("--recall-at", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    parser.add_argument("--mrr-at", type=int, default=10)
    parser.add_argument("--eligibility", choices=["all-evidence-mapped", "any-evidence-mapped"],
                        default="all-evidence-mapped")
    parser.add_argument("--padding-policy", choices=["auto", "drop-nonpositive", "keep-all"], default="auto")
    args = parser.parse_args()

    if args.mrr_at < 1 or any(k < 1 for k in args.recall_at):
        parser.error("recall-at and mrr-at values must be positive")
    if len(set(args.recall_at)) != len(args.recall_at):
        parser.error("recall-at values must be unique")
    recall_at = sorted(args.recall_at)

    runs = {}
    for value in args.run:
        name, path = parse_named_path(parser, value, "--run")
        if name in runs:
            parser.error(f"--run name {name!r} is used twice")
        runs[name] = path
    index_dirs = {}
    for value in args.index_dir:
        name, path = parse_named_path(parser, value, "--index-dir")
        if name not in runs:
            parser.error(f"--index-dir {name!r} does not name a run")
        index_dirs[name] = path

    summary_manifest_path = args.output.with_suffix(".manifest.json")
    per_query_paths = {name: args.per_query_dir / f"{name}.jsonl" for name in runs} if args.per_query_dir else {}
    if (args.output.exists() or summary_manifest_path.exists()
            or any(path.exists() for path in per_query_paths.values())):
        parser.error("Evaluation outputs already exist; choose a new output path")

    manifest_path = args.processed_dir / "manifest.json"
    evidence_manifest_path = args.processed_dir / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    corpus_path = args.processed_dir / "corpus.jsonl"
    corpus_hash = sha256(corpus_path)
    if corpus_hash != manifest["output_sha256"]["corpus.jsonl"]:
        parser.error("corpus.jsonl differs from the preprocessing manifest")
    if evidence_manifest["input_sha256"]["corpus.jsonl"] != corpus_hash:
        parser.error("The evidence mapping was built from a different corpus.jsonl")
    queries_path = args.processed_dir / "queries.jsonl"
    gold_path = args.processed_dir / "gold.jsonl"
    queries_hash = sha256(queries_path)
    for path, digest in ((queries_path, queries_hash), (gold_path, sha256(gold_path))):
        if evidence_manifest["output_sha256"][path.name] != digest:
            parser.error(f"{path.name} differs from the evidence manifest")
    if args.reference_manifest is not None:
        reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        if reference != manifest:
            parser.error(f"Local manifest.json does not match the recorded build in {args.reference_manifest}")

    chunk_to_paragraph = {chunk["chunk_id"]: chunk["paragraph_id"] for chunk in read_jsonl(corpus_path)}
    doc_ids = {query["query_id"]: query["doc_id"] for query in read_jsonl(queries_path)}
    query_ids = set(doc_ids)

    eligible, reasons, gold_sizes = {}, Counter(), []
    partially_mapped_excluded = 0
    for record in read_jsonl(gold_path):
        if record["query_id"] not in query_ids:
            parser.error("gold.jsonl and queries.jsonl cover different questions")
        paragraph_ids = set(record["paragraph_ids"])
        if args.eligibility == "all-evidence-mapped":
            is_eligible = record["all_evidence_mapped"]
        else:
            is_eligible = bool(paragraph_ids)
        if is_eligible and not paragraph_ids:
            parser.error(f"Query {record['query_id']} is eligible with no gold paragraph_ids")
        if is_eligible:
            eligible[record["query_id"]] = paragraph_ids
            gold_sizes.append(len(paragraph_ids))
        else:
            reasons[exclusion_reason(record)] += 1
            partially_mapped_excluded += bool(paragraph_ids)
    if len(doc_ids) != sum(1 for _ in read_jsonl(gold_path)):
        parser.error("gold.jsonl and queries.jsonl cover different questions")
    if not eligible:
        parser.error(f"No questions satisfy the {args.eligibility} eligibility policy")
    if len(eligible) + sum(reasons.values()) != len(query_ids):
        parser.error("Eligible and excluded questions do not sum to the question count")

    max_k = max(recall_at + [args.mrr_at])
    results, run_manifests = {}, {}
    for name, run_path in runs.items():
        run_manifest_path = run_path.with_suffix(".manifest.json")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest["output_sha256"].get(run_path.name) != sha256(run_path):
            parser.error(f"{run_path.name} differs from its run manifest")
        if run_manifest["input_sha256"]["queries.jsonl"] != queries_hash:
            parser.error(f"Run {name} was produced from a different queries.jsonl")
        if run_manifest["split"] != manifest["split"]:
            parser.error(f"Run {name} is a {run_manifest['split']} run; this evaluation uses {manifest['split']}")
        if run_manifest["top_k"] < max_k:
            parser.error(f"Run {name} has top_k={run_manifest['top_k']}; cannot report depth {max_k}")
        provenance = "chunk-id-membership"
        if name in index_dirs:
            index_manifest_name = f"{run_manifest['retriever']}_index_manifest.json"
            index_manifest_path = index_dirs[name] / index_manifest_name
            index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
            if index_manifest["corpus_sha256"] != corpus_hash:
                parser.error(f"Run {name} was retrieved from an index built on a different corpus.jsonl")
            if run_manifest["input_sha256"].get(index_manifest_name) != sha256(index_manifest_path):
                parser.error(f"Run {name} was not retrieved from the index in {index_dirs[name]}")
            provenance = "index-manifest-chain"
        run_manifests[name] = (run_manifest, run_manifest_path, provenance)

        drop_nonpositive = (args.padding_policy == "drop-nonpositive"
                            or (args.padding_policy == "auto" and run_manifest["retriever"] == "bm25"))
        hits = {k: 0.0 for k in recall_at}
        chunk_hits = {k: 0.0 for k in recall_at}
        found = {k: 0 for k in recall_at}
        reciprocal_ranks, latencies, dropped = [], [], 0
        distinct_counts, scored_counts, per_query_rows = [], [], []
        for query_id, rows, latency_ms in grouped_rows(parser, name, run_path, query_ids, chunk_to_paragraph):
            if query_id not in eligible:
                continue
            gold_paragraphs = eligible[query_id]
            sequence, paragraphs = fold_to_paragraphs(rows, chunk_to_paragraph, drop_nonpositive)
            dropped += len(rows) - len(sequence)
            first_hit = next((rank for rank, paragraph_id in enumerate(paragraphs, start=1)
                              if paragraph_id in gold_paragraphs), None)
            recalls = {}
            for k in recall_at:
                matched = len(gold_paragraphs.intersection(paragraphs[:k]))
                recalls[k] = matched / len(gold_paragraphs)
                hits[k] += recalls[k]
                chunk_hits[k] += len(gold_paragraphs.intersection(sequence[:k])) / len(gold_paragraphs)
                found[k] += matched > 0
            reciprocal_ranks.append(1.0 / first_hit if first_hit and first_hit <= args.mrr_at else 0.0)
            latencies.append(latency_ms)
            distinct_counts.append(len(paragraphs))
            scored_counts.append(len(sequence))
            per_query_rows.append({
                "query_id": query_id,
                "doc_id": doc_ids[query_id],
                "n_gold_paragraphs": len(gold_paragraphs),
                "n_rows": len(rows),
                "n_rows_scored": len(sequence),
                "n_distinct_paragraphs": len(paragraphs),
                "first_gold_paragraph_rank": first_hit,
                "recall_at": {str(k): round(recalls[k], 6) for k in recall_at},
                "reciprocal_rank": round(reciprocal_ranks[-1], 6),
                "latency_ms": latency_ms,
            })

        missing = set(eligible) - {row["query_id"] for row in per_query_rows}
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            parser.error(f"Run {name} is missing {len(missing)} eligible queries (e.g. {examples})")
        latencies.sort()
        count = len(per_query_rows)
        results[name] = {
            "retriever": run_manifest["retriever"],
            "top_k": run_manifest["top_k"],
            "provenance": provenance,
            "padding_dropped_rows": dropped,
            "metrics": {
                "recall_at": {str(k): round(hits[k] / count, 6) for k in recall_at},
                "recall_at_chunk_depth": {str(k): round(chunk_hits[k] / count, 6) for k in recall_at},
                "hit_rate_at": {str(k): round(found[k] / count, 6) for k in recall_at},
                f"mrr_at_{args.mrr_at}": round(statistics.fmean(reciprocal_ranks), 6),
            },
            "diagnostics": {
                "mean_distinct_paragraphs": round(statistics.fmean(distinct_counts), 3),
                "mean_scored_rows": round(statistics.fmean(scored_counts), 3),
            },
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3),
                "p50": round(percentile(latencies, 0.5), 3),
                "p95": round(percentile(latencies, 0.95), 3),
            },
        }
        for key in ("model", "device", "tokenizer", "k1", "b"):
            if key in run_manifest:
                results[name][key] = run_manifest[key]
        if name in per_query_paths:
            per_query_paths[name].parent.mkdir(parents=True, exist_ok=True)
            with per_query_paths[name].open("x", encoding="utf-8") as file:
                for row in per_query_rows:
                    file.write(json.dumps(row) + "\n")

    summary = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "credit_granularity": "paragraph",
        "percentile_estimator": "nearest-rank",
        "padding_policy": args.padding_policy,
        "recall_at": recall_at,
        "mrr_at": args.mrr_at,
        "eligibility": {
            "policy": args.eligibility,
            "eligible_queries": len(eligible),
            "total_queries": len(query_ids),
            "mean_gold_paragraphs": round(statistics.fmean(gold_sizes), 3),
            "excluded_queries": sum(reasons.values()),
            "excluded_by_reason": dict(sorted(reasons.items())),
            "partially_mapped_excluded": partially_mapped_excluded,
        },
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    input_hashes = {
        "corpus.jsonl": corpus_hash, "queries.jsonl": queries_hash, "gold.jsonl": sha256(gold_path),
        "manifest.json": sha256(manifest_path), "evidence_manifest.json": sha256(evidence_manifest_path),
    }
    for name, run_path in runs.items():
        run_manifest, run_manifest_path, _ = run_manifests[name]
        input_hashes[f"{name}/{run_path.name}"] = sha256(run_path)
        input_hashes[f"{name}/{run_manifest_path.name}"] = sha256(run_manifest_path)
    output_hashes = {args.output.name: sha256(args.output)}
    for name, path in per_query_paths.items():
        output_hashes[f"{name}/{path.name}"] = sha256(path)
    evaluation_manifest = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "eligibility": summary["eligibility"]["policy"],
        "eligible_queries": len(eligible),
        "credit_granularity": "paragraph",
        "percentile_estimator": "nearest-rank",
        "padding_policy": args.padding_policy,
        "recall_at": recall_at,
        "mrr_at": args.mrr_at,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }
    summary_manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8")

    eligibility = summary["eligibility"]
    share = 100 * eligibility["eligible_queries"] / eligibility["total_queries"]
    print(f"Eligibility policy: {args.eligibility}   eligible "
          f"{eligibility['eligible_queries']} / {eligibility['total_queries']} ({share:.1f}%)")
    print(f"Excluded {eligibility['excluded_queries']}: "
          + " | ".join(f"{reason} {total}" for reason, total in eligibility["excluded_by_reason"].items())
          + f" | of which partially mapped {eligibility['partially_mapped_excluded']}")
    print(f"Gold paragraphs per eligible query: mean {eligibility['mean_gold_paragraphs']}")
    print(f"Credit granularity: paragraph   latency percentiles recomputed by this script (nearest-rank)")
    print()
    names = list(results)
    width = max([14] + [len(name) for name in names])
    print("metric".ljust(16) + "".join(name.rjust(width + 2) for name in names))
    rows = ([(f"Recall@{k}", lambda run, k=k: run["metrics"]["recall_at"][str(k)]) for k in recall_at]
            + [(f"MRR@{args.mrr_at}", lambda run: run["metrics"][f"mrr_at_{args.mrr_at}"])]
            + [(f"HitRate@{k}", lambda run, k=k: run["metrics"]["hit_rate_at"][str(k)]) for k in recall_at])
    for label, getter in rows:
        print(label.ljust(16) + "".join(f"{getter(results[name]):.4f}".rjust(width + 2) for name in names))
    for label, key in (("latency p50 ms", "p50"), ("latency p95 ms", "p95")):
        print(label.ljust(16)
              + "".join(f"{results[name]['latency_ms'][key]:.2f}".rjust(width + 2) for name in names))
    print("dropped rows".ljust(16)
          + "".join(str(results[name]["padding_dropped_rows"]).rjust(width + 2) for name in names))


if __name__ == "__main__":
    main()
