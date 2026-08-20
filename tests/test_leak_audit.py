"""Leak audit fails on identity overlap and writes NN plots without a live embedder."""

from pathlib import Path

import numpy as np
import pytest
from guardrail_slm.data.config import SplitConfig
from guardrail_slm.data.leak_audit import LeakError, assert_no_leaks, run_leak_audit
from guardrail_slm.data.schema import AssignedRecord
from guardrail_slm.data.sources import stable_hash


def _assigned(**overrides: object) -> AssignedRecord:
    split = str(overrides.get("split", "train"))
    payload: dict[str, object] = {
        "record_id": "abc",
        "text": "hello there unique prompt",
        "source_id": "deepset",
        "hub_name": "deepset/prompt-injections",
        "license": "Apache-2.0",
        "provenance": "human",
        "binary_label": "attack",
        "original_split": "train",
        "allowed_in_train": True,
        "label": "C",
        "categories_all": ["C"],
        "label_method": "rule",
        "split": split,
        "slice_id": "natural" if split == "test" else None,
        "group_id": "g1",
    }
    payload.update(overrides)
    return AssignedRecord.model_validate(payload)


def _cfg() -> SplitConfig:
    return SplitConfig(
        minhash_num_perm=32,
        leak_cosine_threshold=0.90,
        leak_jaccard_threshold=0.80,
    )


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 4-d vectors. Similar prefixes land nearby; others do not."""
    rows = []
    for text in texts:
        if "train only" in text:
            base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif "test only" in text:
            base = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        elif text.startswith("TEMPLATE"):
            base = np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float64)
        else:
            digest = stable_hash(text)
            base = np.array(
                [int(digest[i : i + 4], 16) / 65535 for i in range(0, 16, 4)],
                dtype=np.float64,
            )
        rows.append(base)
    return np.stack(rows)


def test_vanilla_id_overlap_fails_the_build() -> None:
    vanilla = "shared payload for both splits"
    vanilla_id = stable_hash(vanilla)
    records = [
        _assigned(
            record_id="tr",
            text="train wrapper around payload",
            split="train",
            group_id="g-train",
            vanilla_text=vanilla,
            vanilla_id=vanilla_id,
        ),
        _assigned(
            record_id="te",
            text="test wrapper around payload",
            split="test",
            group_id="g-test",
            vanilla_text=vanilla,
            vanilla_id=vanilla_id,
        ),
    ]
    with pytest.raises(LeakError, match="vanilla_id"):
        assert_no_leaks(records, ())


def test_held_out_tactic_in_train_fails_the_build() -> None:
    records = [
        _assigned(
            record_id="tr",
            text="train roleplay attack",
            split="train",
            tactics=("persona",),
            group_id="g-train",
        ),
        _assigned(
            record_id="te",
            text="unrelated test attack",
            split="test",
            tactics=("encoding",),
            group_id="g-test",
        ),
    ]
    with pytest.raises(LeakError, match="tactic-set"):
        assert_no_leaks(records, (("persona",),))


def test_clean_fixture_writes_plot(tmp_path: Path) -> None:
    records = [
        _assigned(record_id="tr", text="train only weather query in seattle", split="train"),
        _assigned(
            record_id="te",
            text="test only capital city question about lisbon",
            split="test",
            group_id="g-test",
        ),
    ]
    summary = run_leak_audit(
        records,
        (),
        split_config=_cfg(),
        embed_fn=fake_embed,
        plot_dir=tmp_path,
    )
    assert (tmp_path / "leak_audit.png").is_file()
    assert (tmp_path / "leak_audit.json").is_file()
    assert summary["vanilla_overlap"] == 0
    assert summary["n_train"] == 1
    assert summary["n_test"] == 1
    assert summary["n_test_cosine_above"] == 0


def test_high_similarity_is_reported_not_a_failure(tmp_path: Path) -> None:
    records = [
        _assigned(
            record_id="tr",
            text="TEMPLATE You are DAN. payload one unique topic",
            split="train",
            scaffold="TEMPLATE You are DAN.",
        ),
        _assigned(
            record_id="te",
            text="TEMPLATE You are DAN. payload two other topic",
            split="test",
            group_id="g-test",
            scaffold="TEMPLATE You are DAN.",
        ),
    ]
    summary = run_leak_audit(
        records,
        (),
        split_config=_cfg(),
        embed_fn=fake_embed,
        plot_dir=tmp_path,
    )
    assert summary["n_test_cosine_above"] >= 1
    assert summary["n_test_jaccard_above"] >= 1
    assert summary["cosine_max"] >= 0.90
