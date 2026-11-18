"""Explicit, untuned settings for the Li et al. (2025) adaptation."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class MOALNSParameters:
    scores: tuple = (25.0, 5.0, 1.0)
    segment_length: int = 100
    reaction: float = 0.2
    threshold_steps: tuple = (0.02, 0.02)

    @classmethod
    def from_dict(cls, values=None):
        settings = cls(**(values or {}))
        if (isinstance(settings.segment_length, bool)
                or not isinstance(settings.segment_length, int)
                or settings.segment_length <= 0):
            raise ValueError("segment_length must be a positive integer")
        if not math.isfinite(settings.reaction) or not 0 < settings.reaction <= 1:
            raise ValueError("reaction must lie in (0, 1]")
        for name, size in (("scores", 3), ("threshold_steps", 2)):
            values = getattr(settings, name)
            if len(values) != size or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError(f"Invalid {name}")
        return settings

    def as_dict(self):
        return asdict(self)
