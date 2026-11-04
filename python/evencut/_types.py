"""Result types and the error class shared by every backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

U64_MAX = (1 << 64) - 1


class EvencutError(ValueError):
    """Invalid input: empty weights, k < 1, negative or non-integer values, totals above
    2**64 - 1, out-of-range forbidden cuts, infeasible secondary limits, malformed trees."""


def _imbalance(max_load: int, total: int, parts: int) -> float:
    if total == 0 or parts == 0:
        return 1.0
    return max_load / (total / parts)


@dataclass(frozen=True)
class Partition:
    """A split of an ordered list into contiguous parts.

    Attributes:
        ends: Exclusive end index of every part; part ``p`` covers items
            ``[ends[p-1], ends[p])`` and ``ends[-1] == len(weights)``.
        loads: Weight sum of every part.
        max_load: The largest part weight.
        total: The sum of all weights.
        sum_of_squares: Exact ``sum(load ** 2)``.
        imbalance: ``max_load / (total / parts)``; 1.0 means perfectly even.
        objective: ``"min_max"``, ``"sum_squares"`` or ``"evaluate"``.
    """

    ends: tuple[int, ...]
    loads: tuple[int, ...]
    max_load: int
    total: int
    sum_of_squares: int
    imbalance: float
    objective: str

    @property
    def parts(self) -> int:
        """Number of parts."""
        return len(self.ends)

    def segments(self) -> list[tuple[int, int]]:
        """``[(begin, end), ...]`` index ranges, end exclusive."""
        begins = (0,) + self.ends[:-1]
        return list(zip(begins, self.ends))

    def assignment(self) -> list[int]:
        """Part number of every item."""
        out: list[int] = []
        for part, (begin, end) in enumerate(self.segments()):
            out.extend([part] * (end - begin))
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict; ``sum_of_squares`` becomes a string (it can exceed 2**53)."""
        d = asdict(self)
        d["ends"] = list(self.ends)
        d["loads"] = list(self.loads)
        d["sum_of_squares"] = str(self.sum_of_squares)
        d["parts"] = self.parts
        return d

    @staticmethod
    def from_ends(weights: list[int], ends: list[int], objective: str) -> "Partition":
        loads: list[int] = []
        begin = 0
        for end in ends:
            loads.append(sum(weights[begin:end]))
            begin = end
        total = sum(loads)
        max_load = max(loads)
        return Partition(
            ends=tuple(ends),
            loads=tuple(loads),
            max_load=max_load,
            total=total,
            sum_of_squares=sum(x * x for x in loads),
            imbalance=_imbalance(max_load, total, len(loads)),
            objective=objective,
        )


@dataclass(frozen=True)
class TreePartition:
    """A split of a tree into connected parts. ``part_of[v]`` is the part of node ``v``;
    the root is in part 0. ``cut_nodes`` lists the nodes whose edge to their parent is cut."""

    part_of: tuple[int, ...]
    loads: tuple[int, ...]
    cut_nodes: tuple[int, ...]
    max_load: int
    total: int
    sum_of_squares: int
    imbalance: float

    @property
    def parts(self) -> int:
        return len(self.loads)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("part_of", "loads", "cut_nodes"):
            d[key] = list(d[key])
        d["sum_of_squares"] = str(self.sum_of_squares)
        d["parts"] = self.parts
        return d

    @staticmethod
    def from_assignment(weights: list[int], part_of: list[int], cut_nodes: list[int]) -> "TreePartition":
        parts = max(part_of) + 1
        loads = [0] * parts
        for v, p in enumerate(part_of):
            loads[p] += weights[v]
        total = sum(loads)
        max_load = max(loads)
        return TreePartition(
            part_of=tuple(part_of),
            loads=tuple(loads),
            cut_nodes=tuple(sorted(cut_nodes)),
            max_load=max_load,
            total=total,
            sum_of_squares=sum(x * x for x in loads),
            imbalance=_imbalance(max_load, total, parts),
        )
