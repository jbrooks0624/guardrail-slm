"""Load and normalize Hugging Face sources into a common schema."""

from __future__ import annotations

import ast
import csv
import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from guardrail_slm.data.config import SOURCE_IDS, SourcesConfig, SourceSpec, load_sources_config
from guardrail_slm.data.schema import (
    BinaryLabel,
    SourceRecord,
    WildjailbreakType,
    write_jsonl,
)
from guardrail_slm.settings import Settings, get_settings

RowsBySplit = Mapping[str, Sequence[Mapping[str, Any]]]


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def extract_scaffold(adversarial: str, vanilla: str) -> str:
    """Residual of an adversarial prompt after removing the vanilla payload."""
    adversarial = adversarial.strip()
    vanilla = vanilla.strip()
    if not adversarial or not vanilla:
        return ""
    if vanilla in adversarial:
        residual = adversarial.replace(vanilla, " ", 1)
        return " ".join(residual.split())
    matcher = SequenceMatcher(a=vanilla, b=adversarial, autojunk=False)
    parts: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            parts.append(adversarial[j1:j2])
    return " ".join("".join(parts).split())


def parse_tactics(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped in {"[]", "()"}:
            return ()
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return (stripped,)
        if isinstance(parsed, (list, tuple)):
            return tuple(str(item).strip() for item in parsed if str(item).strip())
        return (str(parsed).strip(),) if str(parsed).strip() else ()
    return (str(value).strip(),) if str(value).strip() else ()


def _cell(row: Mapping[str, Any], field: str | None, default: str = "") -> str:
    if not field:
        return default
    value = row.get(field, default)
    if value is None:
        return default
    return str(value).strip()


def _binary_from_map(
    raw: object, mapping: Mapping[str, BinaryLabel], *, source_id: str
) -> BinaryLabel:
    key = str(raw).strip()
    if key not in mapping:
        raise ValueError(f"{source_id}: unknown label {raw!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def _record(
    *,
    spec: SourceSpec,
    text: str,
    binary_label: BinaryLabel,
    original_split: str,
    wildjailbreak_data_type: WildjailbreakType | None = None,
    tactics: tuple[str, ...] = (),
    vanilla_text: str | None = None,
    scaffold: str = "",
) -> SourceRecord:
    vanilla_id = stable_hash(vanilla_text) if vanilla_text else None
    record_id = stable_hash(
        spec.source_id,
        original_split,
        text,
        vanilla_text or "",
        wildjailbreak_data_type or "",
    )
    return SourceRecord(
        record_id=record_id,
        text=text,
        source_id=spec.source_id,
        hub_name=spec.hub_name,
        license=spec.license,
        provenance=spec.provenance,
        binary_label=binary_label,
        original_split=original_split,
        allowed_in_train=spec.allowed_in_train,
        wildjailbreak_data_type=wildjailbreak_data_type,
        tactics=tactics,
        vanilla_text=vanilla_text,
        vanilla_id=vanilla_id,
        scaffold=scaffold,
        synthetic=False,
    )


def normalize_row(spec: SourceSpec, row: Mapping[str, Any], *, split: str) -> SourceRecord | None:
    """Map one Hub row onto SourceRecord. Returns None when the prompt text is empty."""
    if spec.source_id == "dolly":
        instruction = _cell(row, spec.instruction_field)
        context = _cell(row, spec.context_field)
        text = f"{instruction}\n\n{context}".strip() if context else instruction
        if not text:
            return None
        return _record(spec=spec, text=text, binary_label="benign", original_split=split)

    if spec.source_id == "wildjailbreak":
        data_type = _cell(row, spec.data_type_field)
        if data_type not in spec.data_type_labels:
            raise ValueError(f"wildjailbreak: unknown data_type {data_type!r}")
        vanilla = _cell(row, spec.vanilla_field)
        adversarial = _cell(row, spec.adversarial_field)
        is_adversarial = data_type.startswith("adversarial_")
        text = adversarial if is_adversarial else vanilla
        if not text:
            return None
        return _record(
            spec=spec,
            text=text,
            binary_label=spec.data_type_labels[data_type],
            original_split=split,
            wildjailbreak_data_type=cast(WildjailbreakType, data_type),
            tactics=parse_tactics(row.get(spec.tactics_field) if spec.tactics_field else None),
            vanilla_text=vanilla or None,
            scaffold=extract_scaffold(adversarial, vanilla) if is_adversarial else "",
        )

    text = _cell(row, spec.text_field)
    if not text:
        return None
    if spec.always_benign:
        binary: BinaryLabel = "benign"
    else:
        raw_label = row.get(spec.label_field) if spec.label_field else None
        binary = _binary_from_map(raw_label, spec.label_map, source_id=spec.source_id)
    return _record(spec=spec, text=text, binary_label=binary, original_split=split)


def normalize_source(
    spec: SourceSpec,
    rows_by_split: RowsBySplit,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            record = normalize_row(spec, row, split=split)
            if record is None or record.record_id in seen:
                continue
            seen.add(record.record_id)
            records.append(record)
    return records


def check_hub_split_counts(spec: SourceSpec, rows_by_split: RowsBySplit) -> None:
    """Compare Hub split sizes to the dataset card, before duplicate collapse."""
    for split, expected in spec.expected_rows.items():
        actual = len(rows_by_split.get(split, ()))
        if actual != expected:
            raise ValueError(
                f"{spec.source_id} Hub split {split!r}: expected {expected} rows, got {actual}"
            )


def cache_dir(raw_dir: Path, source_id: str) -> Path:
    return raw_dir / source_id


def cache_exists(path: Path) -> bool:
    return (path / "dataset_dict.json").is_file() or (path / "state.json").is_file()


def _hub_load_args(spec: SourceSpec) -> tuple[str, ...]:
    if spec.hub_config:
        return (spec.hub_name, spec.hub_config)
    return (spec.hub_name,)


def normalize_hub_load_kwargs(spec: SourceSpec) -> dict[str, Any]:
    """Builder kwargs from YAML. Coerce a literal '\\t' into a real tab."""
    kwargs: dict[str, Any] = dict(spec.hub_load_kwargs)
    delimiter = kwargs.get("delimiter")
    if delimiter in {"\\t"}:
        kwargs["delimiter"] = "\t"
    return kwargs


def _hub_call_kwargs(spec: SourceSpec, *, token: str | None) -> dict[str, Any]:
    kwargs = normalize_hub_load_kwargs(spec)
    if spec.string_columns:
        from datasets import Features, Value

        kwargs["features"] = Features({name: Value("string") for name in spec.string_columns})
    if spec.gated and not token:
        raise ValueError(
            f"{spec.source_id} is gated; set HF_TOKEN before downloading {spec.hub_name}"
        )
    if token:
        kwargs["token"] = token
    return kwargs


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Parse a quoted tab-separated file, including fields that contain newlines."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {str(key): (value or "") for key, value in row.items() if key is not None}
            )
        return rows


def load_tsv_dataset(spec: SourceSpec, *, token: str | None, destination: Path) -> Any:
    """Download a Hub TSV and parse it without datasets type inference."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import hf_hub_download

    if not spec.hub_tsv_file:
        raise ValueError(f"{spec.source_id} has no hub_tsv_file")
    local_path = hf_hub_download(
        repo_id=spec.hub_name,
        filename=spec.hub_tsv_file,
        repo_type="dataset",
        token=token,
    )
    split = spec.hub_config or "train"
    loaded = DatasetDict({split: Dataset.from_list(read_tsv_rows(Path(local_path)))})
    destination.parent.mkdir(parents=True, exist_ok=True)
    loaded.save_to_disk(str(destination))
    return loaded


def load_hub_dataset(spec: SourceSpec, *, token: str | None, destination: Path) -> Any:
    """Download a Hub dataset into data/raw. Tests must not call this."""
    if spec.hub_tsv_file:
        return load_tsv_dataset(spec, token=token, destination=destination)

    from datasets import DatasetDict, load_dataset

    loaded = load_dataset(*_hub_load_args(spec), **_hub_call_kwargs(spec, token=token))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"train": loaded})
    loaded.save_to_disk(str(destination))
    return loaded


def load_cached_dataset(path: Path) -> Any:
    from datasets import load_from_disk

    return load_from_disk(str(path))


def dataset_to_rows(dataset: Any) -> dict[str, list[dict[str, Any]]]:
    if hasattr(dataset, "items"):
        return {str(split): [dict(row) for row in subset] for split, subset in dataset.items()}
    return {"train": [dict(row) for row in dataset]}


def load_source(
    spec: SourceSpec,
    *,
    settings: Settings | None = None,
    rows_by_split: RowsBySplit | None = None,
    allow_hub: bool = False,
) -> list[SourceRecord]:
    """Normalize one source. Hub access is opt-in via allow_hub."""
    if rows_by_split is not None:
        return normalize_source(spec, rows_by_split)

    settings = settings or get_settings()
    destination = cache_dir(settings.raw_dir, spec.source_id)
    if cache_exists(destination):
        print(f"[{spec.source_id}] cache {destination}", file=sys.stderr, flush=True)
        dataset = load_cached_dataset(destination)
        rows = dataset_to_rows(dataset)
        check_hub_split_counts(spec, rows)
        return normalize_source(spec, rows)

    if not allow_hub:
        raise FileNotFoundError(
            f"No local cache at {destination}. Re-run with allow_hub=True after downloading "
            f"{spec.hub_name}, or pass rows_by_split for tests."
        )

    print(f"[{spec.source_id}] download {spec.hub_name}", file=sys.stderr, flush=True)
    token = settings.hf_token.strip() or None
    if spec.gated:
        token = settings.require_hf_token()
    dataset = load_hub_dataset(spec, token=token, destination=destination)
    rows = dataset_to_rows(dataset)
    check_hub_split_counts(spec, rows)
    return normalize_source(spec, rows)


def load_all_sources(
    settings: Settings | None = None,
    *,
    catalog: SourcesConfig | None = None,
    rows_by_source: Mapping[str, RowsBySplit] | None = None,
    allow_hub: bool = False,
) -> list[SourceRecord]:
    settings = settings or get_settings()
    catalog = catalog or load_sources_config(settings.sources_path)
    records: list[SourceRecord] = []
    for source_id in SOURCE_IDS:
        spec = catalog.sources[source_id]
        source_rows = None if rows_by_source is None else rows_by_source[source_id]
        source_records = load_source(
            spec,
            settings=settings,
            rows_by_split=source_rows,
            allow_hub=allow_hub,
        )
        if rows_by_source is None:
            print(
                f"[{source_id}] {len(source_records)} unique rows",
                file=sys.stderr,
                flush=True,
            )
        records.extend(source_records)
    return records


def summarize(records: Iterable[SourceRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        by_split = summary.setdefault(record.source_id, {})
        by_split[record.original_split] = by_split.get(record.original_split, 0) + 1
        by_split["total"] = by_split.get("total", 0) + 1
        by_split[record.binary_label] = by_split.get(record.binary_label, 0) + 1
    return summary


def write_normalized(records: Sequence[SourceRecord], path: Path | None = None) -> Path:
    settings = get_settings()
    output = path if path is not None else settings.interim_dir / "normalized.jsonl"
    write_jsonl(records, output)
    return output
