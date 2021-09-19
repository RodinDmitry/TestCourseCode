import numpy as np
from .base import BaseLayer


class CrossEntropyLoss(BaseLayer):
    """
    Слой-функция потерь, категориальная кросс-энтропия для задачи класификации на
    N классов.
    Применяет softmax к входным данных.
    """

    def forward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Prediction и target имеют одинаковый размер вида
         [B, C, 1, ... 1] т.е. имеют 2 оси (batch size и channels) длины больше 1
          и любое количество осей единичной длины.
        В predictions находятся логиты, т.е. к ним должен применятся softmax перед вычислением
         энтропии.
        В target[B, C] находится 1, если объект B принадлежит классу C, иначе 0 (one-hot представление).
        Возвращает np.array[B] c лоссом по каждому объекту в мини-батче.

        P[B, c] = exp(pred[B, c]) / Sum[c](exp(pred[B, c])
        Loss[B] = - Sum[c]log( prob[B, C] * target[B, C]) ) = -log(prob[B, C-correct])
        """
        self._pred = pred
        self._target = target
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        assert np.allclose(np.sum(target, axis=-1), 1)

        loss = -np.sum(target * pred, axis=-1)
        loss += np.log(np.sum(np.exp(pred), axis=-1))
        if not isinstance(loss, np.ndarray):
            loss = np.array([loss])
        return loss

    def backward(self) -> np.ndarray:
        """
        Возвращает градиент лосса по pred, т.е. первому аргументу .forward
        """
        pred = np.squeeze(self._pred.copy())
        target = np.squeeze(self._target.copy())
        grad = -target + np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True)
        return grad.reshape(self._pred.shape)