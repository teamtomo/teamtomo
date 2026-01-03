from typing import Sequence


def get_target_fftfreq(
    source_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> tuple[float, ...]:
    target_fftfreq = [
        (_src / _target) * 0.5
        for _src, _target
        in zip(source_spacing, target_spacing)
    ]
    return tuple(target_fftfreq)
