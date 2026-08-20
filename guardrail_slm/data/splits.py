"""Pair-aware and tactic-aware group splits at the declared prevalences."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from guardrail_slm.data.config import HUMAN_ATTACK_SOURCES, SplitConfig, load_sources_config
from guardrail_slm.data.dedupe import cluster_records
from guardrail_slm.data.schema import (
    AssignedRecord,
    LabeledRecord,
    SliceId,
    SplitName,
    read_labeled_records,
    write_jsonl,
)
from guardrail_slm.eval.contract import EvalContract, load_eval
from guardrail_slm.settings import Settings, get_settings

TacticSet = tuple[str, ...]
Group = tuple[str, list[LabeledRecord]]

SLICE_FILES: dict[tuple[SplitName, SliceId | None], str] = {
    ("train", None): "train.jsonl",
    ("val", None): "val.jsonl",
    ("test", "natural"): "test_natural.jsonl",
    ("test", "gpt_generated"): "test_gpt_generated.jsonl",
    ("test", "ood"): "test_ood.jsonl",
}


def tactic_key(record: LabeledRecord) -> TacticSet:
    return tuple(sorted(item.strip() for item in record.tactics if item.strip()))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_slice_for(record: LabeledRecord) -> SliceId | None:
    if not record.eligible_for_test():
        return None
    if record.source_id == "xtram1":
        return "ood" if record.original_split == "test" else None
    if record.source_id in HUMAN_ATTACK_SOURCES or record.source_id == "dolly":
        return "natural"
    if record.source_id == "wildjailbreak":
        if record.wildjailbreak_data_type == "adversarial_harmful":
            return "gpt_generated"
        if record.wildjailbreak_data_type == "adversarial_benign":
            return "natural"
        return None
    return None


def choose_held_out_tactic_sets(
    records: Sequence[LabeledRecord],
    *,
    fraction: float,
    rng: random.Random,
) -> tuple[TacticSet, ...]:
    keys = sorted(
        {
            tactic_key(record)
            for record in records
            if record.source_id == "wildjailbreak" and tactic_key(record)
        }
    )
    if not keys or fraction <= 0:
        return ()
    rng.shuffle(keys)
    count = min(len(keys), max(1, int(round(len(keys) * fraction))))
    return tuple(sorted(keys[:count]))


def _as_assigned(
    record: LabeledRecord,
    *,
    split: SplitName,
    slice_id: SliceId | None,
    group_id: str,
) -> AssignedRecord:
    payload = record.model_dump()
    payload.update({"split": split, "slice_id": slice_id, "group_id": group_id})
    return AssignedRecord.model_validate(payload)


def _has_human_original_test(members: Sequence[LabeledRecord]) -> bool:
    return any(
        record.source_id in HUMAN_ATTACK_SOURCES and record.original_split == "test"
        for record in members
    )


def _has_held_out_tactic(members: Sequence[LabeledRecord], held: set[TacticSet]) -> bool:
    return any(tactic_key(record) in held for record in members)


def subsample_prevalence(
    pairs: Sequence[tuple[LabeledRecord, str]],
    *,
    attack_prevalence: float,
    rng: random.Random,
    target_size: int | None,
    tolerance: float,
) -> list[tuple[LabeledRecord, str]]:
    attacks = [pair for pair in pairs if pair[0].binary_label == "attack"]
    benigns = [pair for pair in pairs if pair[0].binary_label == "benign"]
    rng.shuffle(attacks)
    rng.shuffle(benigns)
    if not attacks or not benigns:
        return []
    max_total = min(
        int(len(attacks) / attack_prevalence),
        int(len(benigns) / (1 - attack_prevalence)),
    )
    if target_size is not None:
        max_total = min(max_total, target_size)
    for total in range(max_total, 1, -1):
        n_attack = int(round(total * attack_prevalence))
        n_benign = total - n_attack
        if n_attack < 1 or n_benign < 1:
            continue
        if n_attack > len(attacks) or n_benign > len(benigns):
            continue
        if abs(n_attack / total - attack_prevalence) <= tolerance:
            return attacks[:n_attack] + benigns[:n_benign]
    return []


def subsample_natural_test(
    pairs: Sequence[tuple[LabeledRecord, str]],
    *,
    attack_prevalence: float,
    rng: random.Random,
) -> list[tuple[LabeledRecord, str]]:
    """Keep every natural attack; downsample benign to the declared prevalence."""
    attacks = [pair for pair in pairs if pair[0].binary_label == "attack"]
    benigns = [pair for pair in pairs if pair[0].binary_label == "benign"]
    rng.shuffle(benigns)
    if not attacks:
        return []
    n_benign = int(round(len(attacks) * (1 - attack_prevalence) / attack_prevalence))
    n_benign = max(0, min(n_benign, len(benigns)))
    return attacks + benigns[:n_benign]


def attack_fraction(records: Sequence[AssignedRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.binary_label == "attack" for record in records) / len(records)


def assert_split_invariants(
    records: Sequence[AssignedRecord],
    held_out_tactic_sets: Sequence[TacticSet],
) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    train_tactics: set[TacticSet] = set()
    for record in records:
        if record.split == "test" and not record.eligible_for_test():
            raise ValueError(f"synthetic or test-ineligible row in test: {record.record_id}")
        if record.source_id == "xtram1" and record.split in {"train", "val"}:
            raise ValueError("xtram1 must not appear in train or val")
        if record.vanilla_id:
            by_split[record.split].add(record.vanilla_id)
        groups[record.group_id].add(record.split)
        if record.split in {"train", "val"} and tactic_key(record):
            train_tactics.add(tactic_key(record))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_split[left] & by_split[right]
        if overlap:
            raise ValueError(f"vanilla_id leaked across {left}/{right}: {len(overlap)}")
    split_groups = {group_id: splits for group_id, splits in groups.items() if len(splits) > 1}
    if split_groups:
        raise ValueError(f"group assigned to multiple splits: {len(split_groups)}")
    held = set(held_out_tactic_sets)
    leaked = sorted(train_tactics & held)
    if leaked:
        raise ValueError(f"held-out tactic-set appeared in train/val: {leaked[:3]}")


def summarize_splits(records: Sequence[AssignedRecord]) -> dict[str, dict[str, int | float]]:
    buckets: dict[str, list[AssignedRecord]] = defaultdict(list)
    for record in records:
        key = record.split if record.split != "test" else f"test_{record.slice_id}"
        buckets[key].append(record)
    summary: dict[str, dict[str, int | float]] = {}
    for key, rows in sorted(buckets.items()):
        attacks = sum(row.binary_label == "attack" for row in rows)
        summary[key] = {
            "total": len(rows),
            "attack": attacks,
            "benign": len(rows) - attacks,
            "attack_prevalence": round(attacks / len(rows), 4) if rows else 0.0,
            "synthetic": sum(row.synthetic for row in rows),
        }
    return summary


def _split_train_val(
    groups: list[Group],
    *,
    fraction: float,
    rng: random.Random,
) -> tuple[list[Group], list[Group]]:
    ordered = sorted(groups, key=lambda item: item[0])
    rng.shuffle(ordered)
    n_val = int(round(len(ordered) * fraction))
    if len(ordered) >= 2:
        n_val = min(max(n_val, 1), len(ordered) - 1)
    return ordered[n_val:], ordered[:n_val]


def assign_splits(
    records: Sequence[LabeledRecord],
    *,
    split_config: SplitConfig | None = None,
    eval_contract: EvalContract | None = None,
    held_out_tactic_sets: Sequence[TacticSet] | None = None,
) -> tuple[list[AssignedRecord], tuple[TacticSet, ...]]:
    split_config = split_config or load_sources_config().splits
    eval_contract = eval_contract or load_eval()
    rng = random.Random(split_config.seed)
    group_map = cluster_records(
        records,
        jaccard=split_config.minhash_jaccard,
        num_perm=split_config.minhash_num_perm,
        shingle_n=split_config.shingle_n,
    )
    grouped: dict[str, list[LabeledRecord]] = defaultdict(list)
    for record in records:
        grouped[group_map[record.record_id]].append(record)

    if held_out_tactic_sets is not None:
        held = tuple(held_out_tactic_sets)
    else:
        held = choose_held_out_tactic_sets(
            records, fraction=split_config.tactic_holdout_fraction, rng=rng
        )
    held_set = set(held)

    ood_pairs: list[tuple[LabeledRecord, str]] = []
    test_pairs: list[tuple[LabeledRecord, str]] = []
    train_candidates: list[Group] = []

    for group_id, members in grouped.items():
        xtram = [row for row in members if row.source_id == "xtram1"]
        rest = [row for row in members if row.source_id != "xtram1"]
        xtram_test = [row for row in xtram if row.original_split == "test"]
        for row in xtram_test:
            ood_pairs.append((row, group_id))
        if not rest:
            continue
        force_test = (
            bool(xtram_test)
            or _has_human_original_test(rest)
            or _has_held_out_tactic(rest, held_set)
        )
        if force_test:
            test_pairs.extend((row, group_id) for row in rest)
        else:
            train_candidates.append((group_id, rest))

    wj_held = {
        group_id
        for group_id, members in grouped.items()
        if any(row.source_id == "wildjailbreak" for row in members)
        and (
            _has_held_out_tactic(members, held_set)
            or _has_human_original_test(members)
            or any(row.source_id == "xtram1" and row.original_split == "test" for row in members)
        )
    }
    wj_train = [
        group
        for group in train_candidates
        if any(row.source_id == "wildjailbreak" for row in group[1])
    ]
    n_wj = len(wj_train) + len(wj_held)
    need = 0
    if n_wj and split_config.pair_holdout_fraction > 0:
        need = min(n_wj, max(1, int(round(n_wj * split_config.pair_holdout_fraction))))
    extra = [group for group in wj_train if group[0] not in wj_held]
    rng.shuffle(extra)
    hold_from_train: set[str] = set()
    for group_id, members in extra:
        if len(wj_held) + len(hold_from_train) >= need:
            break
        hold_from_train.add(group_id)
        test_pairs.extend((row, group_id) for row in members)
    train_candidates = [group for group in train_candidates if group[0] not in hold_from_train]

    natural_attack_n = sum(
        1
        for record, _group_id in test_pairs
        if test_slice_for(record) == "natural" and record.binary_label == "attack"
    )
    already_benign = sum(
        1
        for record, _group_id in test_pairs
        if test_slice_for(record) == "natural" and record.binary_label == "benign"
    )
    benign_ratio = (1 - eval_contract.test.attack_prevalence) / eval_contract.test.attack_prevalence
    n_benign_target = int(round(natural_attack_n * benign_ratio))
    dolly_need = max(0, n_benign_target - already_benign)
    dolly_only = [
        group
        for group in train_candidates
        if group[1] and all(row.source_id == "dolly" for row in group[1])
    ]
    rng.shuffle(dolly_only)
    taken_dolly: set[str] = set()
    for group_id, members in dolly_only:
        if dolly_need <= 0:
            break
        test_pairs.extend((row, group_id) for row in members)
        taken_dolly.add(group_id)
        dolly_need -= len(members)
    train_candidates = [group for group in train_candidates if group[0] not in taken_dolly]

    train_groups, val_groups = _split_train_val(
        train_candidates, fraction=split_config.val_fraction, rng=rng
    )
    train_pairs = [(row, group_id) for group_id, members in train_groups for row in members]
    val_pairs = [(row, group_id) for group_id, members in val_groups for row in members]

    train_kept = subsample_prevalence(
        train_pairs,
        attack_prevalence=eval_contract.train.attack_prevalence,
        rng=rng,
        target_size=split_config.train_target_size,
        tolerance=split_config.prevalence_tolerance,
    )
    val_target = max(2, int(round(split_config.train_target_size * split_config.val_fraction)))
    val_kept = subsample_prevalence(
        val_pairs,
        attack_prevalence=eval_contract.train.attack_prevalence,
        rng=rng,
        target_size=val_target,
        tolerance=split_config.prevalence_tolerance,
    )
    if not train_kept:
        raise ValueError("could not subsample train to the declared attack prevalence")

    natural_pairs = [pair for pair in test_pairs if test_slice_for(pair[0]) == "natural"]
    gpt_pairs = [pair for pair in test_pairs if test_slice_for(pair[0]) == "gpt_generated"]
    natural_kept = subsample_natural_test(
        natural_pairs,
        attack_prevalence=eval_contract.test.attack_prevalence,
        rng=rng,
    )

    assigned: list[AssignedRecord] = []
    for record, group_id in train_kept:
        assigned.append(_as_assigned(record, split="train", slice_id=None, group_id=group_id))
    for record, group_id in val_kept:
        assigned.append(_as_assigned(record, split="val", slice_id=None, group_id=group_id))
    for record, group_id in natural_kept:
        assigned.append(_as_assigned(record, split="test", slice_id="natural", group_id=group_id))
    for record, group_id in gpt_pairs:
        assigned.append(
            _as_assigned(record, split="test", slice_id="gpt_generated", group_id=group_id)
        )
    for record, group_id in ood_pairs:
        assigned.append(_as_assigned(record, split="test", slice_id="ood", group_id=group_id))

    assigned.sort(key=lambda row: (row.split, row.slice_id or "", row.record_id))
    assert_split_invariants(assigned, held)
    return assigned, held


def manifest_payload(
    records: Sequence[AssignedRecord],
    *,
    seed: int,
    held_out_tactic_sets: Sequence[TacticSet],
) -> dict[str, object]:
    rows = [
        {
            "record_id": record.record_id,
            "text_sha256": text_sha256(record.text),
            "split": record.split,
            "slice": record.slice_id,
            "source_id": record.source_id,
            "label": record.label,
            "license": record.license,
            "synthetic": record.synthetic,
        }
        for record in records
    ]
    return {
        "seed": seed,
        "held_out_tactic_sets": [list(item) for item in held_out_tactic_sets],
        "counts": summarize_splits(records),
        "rows": rows,
    }


def write_processed(
    records: Sequence[AssignedRecord],
    *,
    processed_dir: Path,
    seed: int,
    held_out_tactic_sets: Sequence[TacticSet],
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[tuple[SplitName, SliceId | None], list[AssignedRecord]] = defaultdict(list)
    for record in records:
        buckets[(record.split, record.slice_id)].append(record)
    for key, filename in SLICE_FILES.items():
        write_jsonl(buckets.get(key, []), processed_dir / filename)
    manifest_path = processed_dir / "split_manifest.json"
    payload = manifest_payload(records, seed=seed, held_out_tactic_sets=held_out_tactic_sets)
    text_keys = {"text", "vanilla_text", "scaffold"}
    for row in payload["rows"]:  # type: ignore[assignment]
        if text_keys & set(row):
            raise ValueError("manifest must be hashes, not text")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def split_labeled(
    settings: Settings | None = None,
) -> tuple[Path, dict[str, dict[str, int | float]]]:
    settings = settings or get_settings()
    catalog = load_sources_config(settings.sources_path)
    eval_contract = load_eval(settings.eval_path)
    synth_path = settings.interim_dir / "with_synth.jsonl"
    source_path = synth_path if synth_path.is_file() else settings.interim_dir / "labeled.jsonl"
    records = list(read_labeled_records(source_path))
    if len(records) > 1000:
        print(
            f"[splits] loaded {len(records)} rows from {source_path}",
            file=sys.stderr,
            flush=True,
        )
    assigned, held = assign_splits(
        records, split_config=catalog.splits, eval_contract=eval_contract
    )
    manifest_path = write_processed(
        assigned,
        processed_dir=settings.processed_dir,
        seed=catalog.splits.seed,
        held_out_tactic_sets=held,
    )
    return manifest_path, summarize_splits(assigned)
