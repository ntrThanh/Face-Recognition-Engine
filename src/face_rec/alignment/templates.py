"""Reference 5-point landmark templates: (left eye, right eye, nose, left mouth, right mouth)."""

import numpy as np

TEMPLATES: dict[str, tuple[np.ndarray, tuple[int, int]]] = {
    "arcface_112": (
        np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        ),
        (112, 112),
    ),
}
