"""Which composites and lighting copies an augment run generates under an image limit (pure, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPlan:
    composites: list[int]  # composite numbers n, ascending
    lighting: list[tuple[int, int]]  # (frame index i, copy k), in round order (k-major, then i ascending)
    requested_composites: int
    requested_lighting: int
    capped: bool  # True when the image limit reduced the plan


def plan_generation(real_train: int, per_image: int, lighting: int, max_train_images: int | None) -> GenerationPlan:
    """The units to generate. The limit counts the whole train split, so the room for generated images is
    `max_train_images - real_train`, split between composites and lighting copies in proportion to what was
    requested. Composites keep `n = 0 .. c-1`; lighting keeps whole rounds over all frames, then a partial
    round of evenly spaced frames."""
    n_comp = real_train * per_image
    n_light = real_train * lighting
    c, light = n_comp, n_light
    capped = False
    if max_train_images is not None and n_comp + n_light > 0:
        room = max(max_train_images - real_train, 0)
        if n_comp + n_light > room:
            capped = True
            c = round(room * n_comp / (n_comp + n_light))
            light = room - c
    full, rem = divmod(light, real_train) if real_train else (0, 0)
    units = [(i, k) for k in range(full) for i in range(real_train)]
    units += [(j * real_train // rem, full) for j in range(rem)]
    return GenerationPlan(list(range(c)), units, n_comp, n_light, capped)
