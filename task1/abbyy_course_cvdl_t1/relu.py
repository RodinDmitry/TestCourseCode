import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._zero_mult = (input > 0).astype(int)
        return np.maximum(input, 0)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self._zero_mult
