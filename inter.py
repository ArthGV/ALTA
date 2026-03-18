import numpy as np
from typing import List, Literal

def interpolation(interpolation_style: Literal['linear', 'sigmoid'], x: float) -> float:
    if interpolation_style == 'linear':
        return x
    elif interpolation_style == 'sigmoid':
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

def select_anchors(anchoring_style: Literal['border', 'equidistant'], duration, num_anchors) -> List[int]:
    if num_anchors > duration:
        raise ValueError("num_anchors cannot be greater than duration.")
    if anchoring_style == 'border':
        return [0, duration - 1]
    elif anchoring_style == 'equidistant':
        return np.linspace(0, duration - 1, num_anchors, dtype=int).tolist()


      