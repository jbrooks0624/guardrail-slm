"""Run dataset pipeline stages. Typer wrapping comes in a later phase."""

from __future__ import annotations

import argparse
import json
import sys

from guardrail_slm.data.label import label_normalized, summarize_labels
from guardrail_slm.data.leak_audit import audit_processed
from guardrail_slm.data.schema import read_labeled_records
from guardrail_slm.data.sources import load_all_sources, summarize, write_normalized
from guardrail_slm.data.splits import split_labeled
from guardrail_slm.data.synth import synthesize_labeled
from guardrail_slm.settings import get_settings

STAGES = ("sources", "labels", "synth", "splits", "leak-audit")


def build(
    *,
    stage: str = "sources",
    allow_hub: bool = False,
    allow_llm: bool = False,
) -> None:
    if stage not in {*STAGES, "all"}:
        raise ValueError(f"unknown stage {stage!r}; known: {STAGES}")
    settings = get_settings()
    run_sources = stage in {"sources", "all"}
    run_labels = stage in {"labels", "all"}
    run_synth = stage in {"synth", "all"}
    run_splits = stage in {"splits", "all"}
    run_leak = stage in {"leak-audit", "all"}
    if run_sources:
        records = load_all_sources(settings, allow_hub=allow_hub)
        path = write_normalized(records)
        print(f"wrote {len(records)} rows to {path}", file=sys.stderr)
        print(json.dumps(summarize(records), indent=2, sort_keys=True))
    if run_labels:
        path = label_normalized(settings, allow_llm=allow_llm)
        labeled = list(read_labeled_records(path))
        print(f"wrote {len(labeled)} rows to {path}", file=sys.stderr)
        print(json.dumps(summarize_labels(labeled), indent=2, sort_keys=True))
    if run_synth:
        path, summary = synthesize_labeled(settings)
        print(f"wrote synth merge to {path}", file=sys.stderr)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if run_splits:
        path, summary = split_labeled(settings)
        print(f"wrote splits and manifest to {path}", file=sys.stderr)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if run_leak:
        summary = audit_processed(settings)
        print("wrote leak audit plot and counts", file=sys.stderr)
        print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build guardrail-slm dataset stages.")
    parser.add_argument("--stage", default="sources", choices=[*STAGES, "all"])
    parser.add_argument(
        "--allow-hub",
        action="store_true",
        help="Download missing sources from the Hugging Face Hub into data/raw/.",
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Call OpenAI for leftover deepset/jackhhao attacks that rules missed.",
    )
    args = parser.parse_args(argv)
    build(stage=args.stage, allow_hub=args.allow_hub, allow_llm=args.allow_llm)


if __name__ == "__main__":
    main()
