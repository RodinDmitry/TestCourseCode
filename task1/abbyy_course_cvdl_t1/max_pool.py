import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert(kernel_size % 2 == 1)

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @staticmethod
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pad_width = [(0, 0) for _ in range(len(tensor.shape))]
        for a in axis:
            pad_width[a] = (one_size_pad, one_size_pad)
        return np.pad(tensor, pad_width, constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        input_p = self._pad_neg_inf(input, self.padding)

        out_size = (input_p.shape[-1] - self.kernel_size) // self.stride + 1
        shape = list(input.shape)
        shape[-1] = out_size
        shape[-2] = out_size
        self._max_linear_indices = np.zeros(shape, dtype=int) - 1
        output = np.zeros(shape)

        self._input_shape = input_p.shape

        slice_shape = (input_p.shape[0] * input_p.shape[1], self.kernel_size * self.kernel_size)
        for b in range(self._input_shape[0]):
            for c in range(self._input_shape[1]):
                for h in range(out_size):
                    for w in range(out_size):
                        hs = h * self.stride
                        ws = w * self.stride
                        input_slice = input_p[b, c, hs:hs+self.kernel_size, ws:ws+self.kernel_size]
                        self._max_linear_indices[b, c, h, w] = np.argmax(input_slice)
                        output[b, c, h, w] = input_slice.flatten()[self._max_linear_indices[b, c, h, w]]
        return output

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        input_grad = np.zeros(self._input_shape)
        out_size = (self._input_shape[-1] - self.kernel_size) // self.stride + 1
        slice_shape = (self._input_shape[0], self._input_shape[1], self.kernel_size * self.kernel_size)
        for b in range(self._input_shape[0]):
            for c in range(self._input_shape[1]):
                for h in range(out_size):
                    for w in range(out_size):
                        hs = h * self.stride
                        ws = w * self.stride
                        output_grad_val = output_grad[b, c, h, w]
                        h_idx = self._max_linear_indices[b, c, h, w] // self.kernel_size
                        w_idx = self._max_linear_indices[b, c, h, w] % self.kernel_size
                        input_grad[b, c, hs + h_idx, ws + w_idx] += output_grad_val
        return input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

