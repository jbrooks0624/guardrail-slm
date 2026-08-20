"""Scaffold-stripped near-duplicate clustering."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence

from datasketch import MinHash, MinHashLSH

from guardrail_slm.data.schema import SourceRecord
from guardrail_slm.data.sources import stable_hash


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def components(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            buckets[self.find(item)].append(item)
        return dict(buckets)


def clustering_text(record: SourceRecord) -> str:
    """Payload-stripped text: scaffold when present, otherwise the full prompt."""
    if record.scaffold.strip():
        return " ".join(record.scaffold.lower().split())
    return " ".join(record.text.lower().split())


def char_shingles(text: str, n: int) -> list[str]:
    if not text:
        return [" "]
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def minhash_of(text: str, *, num_perm: int, shingle_n: int) -> MinHash:
    digest = MinHash(num_perm=num_perm)
    for shingle in char_shingles(text, shingle_n):
        digest.update(shingle.encode("utf-8"))
    return digest


def cluster_records(
    records: Sequence[SourceRecord],
    *,
    jaccard: float = 0.80,
    num_perm: int = 128,
    shingle_n: int = 5,
) -> dict[str, str]:
    """Return record_id -> group_id. Same vanilla_id or near-dup scaffold stay together."""
    if not records:
        return {}
    if len(records) > 1000:
        print(f"[dedupe] clustering {len(records)} rows", file=sys.stderr, flush=True)
    ids = [record.record_id for record in records]
    union = UnionFind(ids)

    by_vanilla: dict[str, list[str]] = defaultdict(list)
    by_exact: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.vanilla_id:
            by_vanilla[record.vanilla_id].append(record.record_id)
        by_exact[clustering_text(record)].append(record.record_id)
    for members in by_vanilla.values():
        for other in members[1:]:
            union.union(members[0], other)
    for members in by_exact.values():
        for other in members[1:]:
            union.union(members[0], other)

    representatives: list[tuple[str, str]] = []
    seen_roots: set[str] = set()
    for record in records:
        root = union.find(record.record_id)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        representatives.append((root, clustering_text(record)))

    if len(representatives) > 1:
        if len(records) > 1000:
            print(
                f"[dedupe] {len(representatives)} unique clustering texts",
                file=sys.stderr,
                flush=True,
            )
        lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
        hashes: dict[str, MinHash] = {}
        for root, text in representatives:
            digest = minhash_of(text, num_perm=num_perm, shingle_n=shingle_n)
            hashes[root] = digest
            lsh.insert(root, digest)
        for root, digest in hashes.items():
            for neighbor in lsh.query(digest):
                if neighbor != root:
                    union.union(root, neighbor)

    group_ids: dict[str, str] = {}
    for root, members in union.components().items():
        group_id = stable_hash("group", *sorted(members))
        for record_id in members:
            group_ids[record_id] = group_id
    return group_ids
