import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.parameters.append(
            np.zeros((out_features, in_features))
        )
        self.parameters.append(
            np.zeros((out_features))
        )

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._last_input = input
        return np.tensordot(
            input, # [N, in]
            self.parameters[0], #[out, in]
            axes=[[-1], [1]]
        )  + self.parameters[1] #[N, out]

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.parameters_grads.append(
            np.tensordot(
                output_grad.reshape(-1, self.parameters[0].shape[0]),
                self._last_input.reshape(-1, self.parameters[0].shape[1]),
                axes=[[-2], [-2]]
            )
        )
        self.parameters_grads.append(
            np.sum(
                output_grad.reshape((-1, len(self.parameters[1]))),
                axis=0
            )
        )
        return np.tensordot(
            output_grad,
            self.parameters[0],
            axes=[[-1], [-2]]
        )
