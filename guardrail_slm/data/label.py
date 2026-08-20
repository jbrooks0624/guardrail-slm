"""Rule-based plus LLM category labeling with the taxonomy precedence rule."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, Field

from guardrail_slm.data.schema import (
    LabeledRecord,
    LabelMethod,
    LabelToken,
    SourceRecord,
    read_source_records,
    write_jsonl,
)
from guardrail_slm.settings import Settings, get_settings
from guardrail_slm.taxonomy import LABEL_TOKENS, Taxonomy, load_taxonomy

LlmFn = Callable[[SourceRecord], frozenset[LabelToken]]

ROLEPLAY_PATTERNS = (
    re.compile(r"\byou are (?:now |an? |the )", re.I),
    re.compile(r"\bact as\b", re.I),
    re.compile(r"\bpretend (?:you|to be)\b", re.I),
    re.compile(r"\brole[- ]?play\b", re.I),
    re.compile(r"\bDAN\b"),
    re.compile(r"\bin your role as\b", re.I),
    re.compile(r"\bas an? (?:ai|assistant|unrestricted)\b", re.I),
    re.compile(r"\bimagine (?:that )?you(?:'re| are)\b", re.I),
    re.compile(r"\bdeveloper mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
)
OVERRIDE_PATTERNS = (
    re.compile(r"ignore (?:all )?(?:previous|prior|above) (?:instructions|rules|prompts)", re.I),
    re.compile(r"disregard (?:your )?(?:instructions|guidelines|rules)", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"override (?:your|the) (?:instructions|rules|safety)", re.I),
)
ENCODING_PATTERNS = (
    re.compile(r"\bbase64\b", re.I),
    re.compile(r"\brot13\b", re.I),
    re.compile(r"\bhex(?:adecimal)?\b", re.I),
    re.compile(r"\batbash\b", re.I),
    re.compile(r"decode (?:this|the following)", re.I),
    re.compile(r"\bleet(?:speak)?\b", re.I),
)
CONTEXT_PATTERNS = (
    re.compile(r"\bin this (?:document|email|webpage|article|context)\b", re.I),
    re.compile(r"the following (?:document|retrieved|context)\b", re.I),
    re.compile(r"\bindirect (?:prompt )?injection\b", re.I),
    re.compile(r"ignore .{0,40}\b(?:document|context|retrieved)\b", re.I),
)


class LabelingConfig(BaseModel):
    llm_model: str = Field(min_length=1)
    reasoning_effort: str = "none"
    llm_sources: tuple[str, ...] = ()
    audit_size: int = Field(ge=1)
    residual_token: LabelToken = "C"
    tactic_keywords: dict[LabelToken, tuple[str, ...]] = Field(default_factory=dict)


def load_labeling_config(path: Path | None = None) -> LabelingConfig:
    labeling_path = path if path is not None else get_settings().labeling_path
    with labeling_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return LabelingConfig.model_validate(payload)


def apply_precedence(
    categories: Iterable[LabelToken], taxonomy: Taxonomy
) -> tuple[LabelToken, tuple[LabelToken, ...]]:
    found = {token for token in categories if token in taxonomy.labels}
    if not found:
        raise ValueError("categories must include at least one taxonomy token")
    ordered = tuple(token for token in taxonomy.precedence if token in found)
    return ordered[0], ordered


def map_tactics(tactics: Sequence[str], config: LabelingConfig) -> frozenset[LabelToken]:
    hits: set[LabelToken] = set()
    blob = " ".join(tactics).lower()
    if not blob:
        return frozenset()
    for token, keywords in config.tactic_keywords.items():
        if any(keyword.lower() in blob for keyword in keywords):
            hits.add(token)
    return frozenset(hits)


def apply_rules(text: str) -> frozenset[LabelToken]:
    hits: set[LabelToken] = set()
    if any(pattern.search(text) for pattern in CONTEXT_PATTERNS):
        hits.add("E")
    if any(pattern.search(text) for pattern in ENCODING_PATTERNS):
        hits.add("D")
    if any(pattern.search(text) for pattern in ROLEPLAY_PATTERNS):
        hits.add("B")
    if any(pattern.search(text) for pattern in OVERRIDE_PATTERNS):
        hits.add("C")
    return frozenset(hits)


def _as_labeled(
    record: SourceRecord,
    categories: Iterable[LabelToken],
    methods: dict[LabelToken, LabelMethod],
    taxonomy: Taxonomy,
) -> LabeledRecord:
    label, categories_all = apply_precedence(categories, taxonomy)
    payload = record.model_dump()
    payload.update(
        {
            "label": label,
            "categories_all": categories_all,
            "label_method": methods[label],
        }
    )
    return LabeledRecord.model_validate(payload)


def label_record(
    record: SourceRecord,
    taxonomy: Taxonomy,
    config: LabelingConfig,
    *,
    llm_fn: LlmFn | None = None,
) -> LabeledRecord:
    """Assign A-E without changing the source binary attack/benign label."""
    if record.binary_label == "benign":
        return _as_labeled(record, ["A"], {"A": "source_benign"}, taxonomy)

    if record.source_id == "xtram1":
        return _as_labeled(record, ["C"], {"C": "source_binary"}, taxonomy)

    categories: set[LabelToken] = set()
    methods: dict[LabelToken, LabelMethod] = {}

    for token in map_tactics(record.tactics, config):
        categories.add(token)
        methods[token] = "tactic_map"

    rule_text = record.text if not record.scaffold else f"{record.text}\n{record.scaffold}"
    for token in apply_rules(rule_text):
        categories.add(token)
        methods.setdefault(token, "rule")

    if record.wildjailbreak_data_type == "vanilla_harmful" and not categories:
        categories.add("C")
        methods["C"] = "source_type"

    if not categories and llm_fn is not None and record.source_id in config.llm_sources:
        for token in llm_fn(record):
            categories.add(token)
            methods[token] = "llm"

    if not categories:
        residual = config.residual_token
        categories.add(residual)
        methods[residual] = "residual"

    return _as_labeled(record, categories, methods, taxonomy)


class LlmCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[str, list[LabelToken]] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._rows[row["text_hash"]] = list(row["categories"])

    def get(self, text_hash: str) -> list[LabelToken] | None:
        return self._rows.get(text_hash)

    def put(self, text_hash: str, categories: Sequence[LabelToken]) -> None:
        self._rows[text_hash] = list(categories)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"text_hash": text_hash, "categories": list(categories)}))
            handle.write("\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_output_text(payload: dict[str, object]) -> str:
    """Pull the model text out of a Responses API body."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


def parse_category_tokens(content: str) -> tuple[LabelToken, ...]:
    parsed = json.loads(content)
    raw = parsed.get("categories") or [parsed.get("label")]
    tokens = tuple(
        token for token in raw if isinstance(token, str) and token in LABEL_TOKENS
    )
    return tokens or ("C",)


def llm_user_input(text: str) -> str:
    """Responses json_object mode requires the word 'json' in the input, not instructions."""
    return "Return JSON with key categories listing A-E tokens.\n\n" + text[:8000]


def openai_label_fn(
    *,
    api_key: str,
    model: str,
    taxonomy: Taxonomy,
    cache: LlmCache,
    reasoning_effort: str = "none",
) -> LlmFn:
    names = ", ".join(
        f"{token}={taxonomy.labels[token].name}: {taxonomy.labels[token].description}"
        for token in LABEL_TOKENS
    )
    instructions = (
        "Classify prompt-injection / jailbreak intent. Reply with JSON only: "
        '{"categories": ["B"]}. Use the tokens A-E. Multi-label is allowed. '
        f"Definitions: {names}. Precedence if several apply is "
        f"{' > '.join(taxonomy.precedence)}."
    )

    def _call(record: SourceRecord) -> frozenset[LabelToken]:
        digest = text_hash(f"responses|{model}|{reasoning_effort}\n{record.text}")
        cached = cache.get(digest)
        if cached is not None:
            return frozenset(cached)
        body = json.dumps(
            {
                "model": model,
                "store": False,
                "reasoning": {"effort": reasoning_effort},
                "text": {"format": {"type": "json_object"}},
                "instructions": instructions,
                "input": llm_user_input(record.text),
            }
        ).encode("utf-8")
        request = Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API {exc.code}: {detail}") from exc
        tokens = parse_category_tokens(response_output_text(payload))
        cache.put(digest, tokens)
        return frozenset(tokens)

    return _call


def label_records(
    records: Iterable[SourceRecord],
    *,
    taxonomy: Taxonomy | None = None,
    config: LabelingConfig | None = None,
    llm_fn: LlmFn | None = None,
) -> list[LabeledRecord]:
    taxonomy = taxonomy or load_taxonomy()
    config = config or load_labeling_config()
    return [label_record(record, taxonomy, config, llm_fn=llm_fn) for record in records]


def summarize_labels(records: Iterable[LabeledRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        by_source = summary.setdefault(record.source_id, Counter())
        by_source[record.label] += 1
        by_source[record.label_method] += 1
        by_source["total"] += 1
    return {source: dict(counts) for source, counts in summary.items()}


def write_audit_sample(
    records: Sequence[LabeledRecord],
    path: Path,
    *,
    size: int,
    seed: int = 0,
) -> int:
    """Stratified sample for the 200-example hand audit. human_label is left blank."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[LabeledRecord]] = {}
    for record in records:
        buckets.setdefault((record.source_id, record.label), []).append(record)
    keys = sorted(buckets)
    if not keys:
        return 0
    per_bucket = max(1, size // len(keys))
    chosen: list[LabeledRecord] = []
    for key in keys:
        pool = buckets[key]
        rng.shuffle(pool)
        chosen.extend(pool[:per_bucket])
    rng.shuffle(chosen)
    chosen = chosen[:size]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "source_id",
                "label",
                "categories_all",
                "label_method",
                "text",
                "human_label",
            ],
        )
        writer.writeheader()
        for record in chosen:
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "source_id": record.source_id,
                    "label": record.label,
                    "categories_all": "".join(record.categories_all),
                    "label_method": record.label_method,
                    "text": record.text[:2000],
                    "human_label": "",
                }
            )
    return len(chosen)


def label_normalized(
    settings: Settings | None = None,
    *,
    allow_llm: bool = False,
) -> Path:
    settings = settings or get_settings()
    taxonomy = load_taxonomy()
    config = load_labeling_config(settings.labeling_path)
    source_path = settings.interim_dir / "normalized.jsonl"
    records = list(read_source_records(source_path))
    llm_fn = None
    if allow_llm:
        cache = LlmCache(settings.interim_dir / "llm_label_cache.jsonl")
        llm_fn = openai_label_fn(
            api_key=settings.require_openai_api_key(),
            model=config.llm_model,
            taxonomy=taxonomy,
            cache=cache,
            reasoning_effort=config.reasoning_effort,
        )
    labeled = label_records(records, taxonomy=taxonomy, config=config, llm_fn=llm_fn)
    output = settings.interim_dir / "labeled.jsonl"
    write_jsonl(labeled, output)
    audit_path = settings.interim_dir / "label_audit_sample.csv"
    write_audit_sample(labeled, audit_path, size=config.audit_size)
    return output
