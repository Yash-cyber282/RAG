#!/usr/bin/env python3
"""
scripts/ingest.py — CLI tool for bulk PDF ingestion

Usage:
    python scripts/ingest.py --dir ./data/pdfs
    python scripts/ingest.py --file report.pdf
    python scripts/ingest.py --dir ./data/pdfs --reset
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--dir", type=Path, help="Directory of PDFs to ingest")
    parser.add_argument("--file", type=Path, help="Single PDF to ingest")
    parser.add_argument("--reset", action="store_true", help="Clear collection before ingesting")
    parser.add_argument("--output", type=Path, help="Save results JSON to file")
    args = parser.parse_args()

    if not args.dir and not args.file:
        parser.print_help()
        sys.exit(1)

    pipeline = IngestionPipeline()

    if args.reset:
        logger.warning("Resetting ChromaDB collection…")
        vs = VectorStore()
        vs._client.delete_collection(vs._collection.name)
        logger.success("Collection reset.")

    results = []
    if args.file:
        results = [pipeline.ingest_pdf(args.file)]
    elif args.dir:
        results = pipeline.ingest_directory(args.dir)

    # Summary
    total_files = len(results)
    total_chunks = sum(r.get("chunks_stored", 0) for r in results)
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*50}")
    print(f"Ingestion summary")
    print(f"  Files processed : {total_files}")
    print(f"  Total chunks    : {total_chunks:,}")
    print(f"  Errors          : {len(errors)}")
    print(f"{'='*50}\n")

    if errors:
        for e in errors:
            print(f"  ✗ {e['filename']}: {e['error']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
