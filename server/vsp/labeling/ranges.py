"""Inclusive integer ranges, kept sorted and non-overlapping. Pure functions."""

from __future__ import annotations

Range = tuple[int, int]


def add_range(ranges: list[Range], lo: int, hi: int) -> list[Range]:
    """Union with [lo, hi]; overlapping and adjacent ranges are merged."""
    if lo > hi:
        lo, hi = hi, lo
    out: list[Range] = []
    for a, b in sorted(ranges):
        if b < lo - 1 or a > hi + 1:
            out.append((a, b))
        else:
            lo, hi = min(lo, a), max(hi, b)
    out.append((lo, hi))
    return sorted(out)


def subtract_range(ranges: list[Range], lo: int, hi: int) -> list[Range]:
    """Remove [lo, hi] from every range."""
    if lo > hi:
        lo, hi = hi, lo
    out: list[Range] = []
    for a, b in sorted(ranges):
        if b < lo or a > hi:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, lo - 1))
        if b > hi:
            out.append((hi + 1, b))
    return out
