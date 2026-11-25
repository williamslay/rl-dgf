"""Deterministic candidate seed selection for fixed-slot RL actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Result of deterministic K-slot candidate mapping."""

    ranked_seeds: list[dict[str, Any]]
    slot_seed_ids: list[int]
    slot_seeds: list[dict[str, Any] | None]
    slot_wrapped: list[bool]
    valid_seed_count: int


def _seed_sort_key(seed: dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic priority ranking for valid queue entries.

    Order:
      1. Seeds with known AFLRun CFG/CG target distance come first.
      2. Smaller min_target_dist (closer to target) comes first.
      3. Favored seeds come first.
      4. Fewer fuzzed_times (less explored) come first.
      5. Smaller input len (more compact seed) comes first.
      6. Queue id ascending as strict deterministic tie-breaker.
    """
    has_dist = bool(seed.get("has_target_dist", 0))
    min_dist = float(seed.get("min_target_dist", -1.0))
    if not has_dist or min_dist < 0.0:
        dist_rank = (1, float("inf"))
    else:
        dist_rank = (0, min_dist)

    return (
        dist_rank[0],
        dist_rank[1],
        0 if seed.get("favored", 0) else 1,
        int(seed.get("fuzzed_times", 0)),
        int(seed.get("len", 0)),
        int(seed["id"]),
    )


def select_candidates(
    seeds: Sequence[dict[str, Any]],
    k: int = 4,
) -> CandidateSelectionResult:
    """Select up to K valid candidate seeds deterministically and map slots.

    When 0 < valid_seed_count < K:
      - Slots [0 .. valid_seed_count-1] map 1-to-1 to distinct ranked valid seeds
        and their slot_seeds entries contain those seed dicts (slot_wrapped=False).
      - Slots [valid_seed_count .. K-1] explicitly wrap around via
        `slot % valid_seed_count` to a valid existing seed ID so every discrete
        action executes a real, enabled seed (slot_wrapped=True), while
        `slot_seeds[slot]` is None so observations mark empty slots with
        slot_valid=0.0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    valid = [s for s in seeds if not int(s.get("disabled", 0))]
    if not valid:
        raise RuntimeError("No valid (non-disabled) seeds in AFLRun queue.")

    ranked = sorted(valid, key=_seed_sort_key)
    top_k = ranked[:k]

    slot_seed_ids: list[int] = []
    slot_seeds: list[dict[str, Any] | None] = []
    slot_wrapped: list[bool] = []

    n_valid = len(top_k)
    for slot in range(k):
        if slot < n_valid:
            slot_seed_ids.append(int(top_k[slot]["id"]))
            slot_seeds.append(top_k[slot])
            slot_wrapped.append(False)
        else:
            wrapped_seed = top_k[slot % n_valid]
            slot_seed_ids.append(int(wrapped_seed["id"]))
            slot_seeds.append(None)
            slot_wrapped.append(True)

    return CandidateSelectionResult(
        ranked_seeds=top_k,
        slot_seed_ids=slot_seed_ids,
        slot_seeds=slot_seeds,
        slot_wrapped=slot_wrapped,
        valid_seed_count=len(valid),
    )


class SeedCandidateSelector:
    """Stateful wrapper around deterministic candidate selection."""

    def __init__(self, k: int = 4) -> None:
        self.k = k

    def select(self, seeds: Sequence[dict[str, Any]]) -> CandidateSelectionResult:
        return select_candidates(seeds, k=self.k)
