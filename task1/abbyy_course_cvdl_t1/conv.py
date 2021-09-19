import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)

        super().__init__()
        for par in [self.parameters, self.parameters_grads]:
            par.append(
                np.zeros((out_channels, in_channels, kernel_size, kernel_size))
            )
            par.append(
                np.zeros((out_channels))
            )

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pad_width = [(0, 0) for _ in range(len(tensor.shape))]
        for a in axis:
            pad_width[a] = (one_size_pad, one_size_pad)
        return np.pad(tensor, pad_width)

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        ks = kernel.shape[-1]
        shape = list(input.shape)
        C_out = kernel.shape[0]
        shape[1] = C_out
        shape[-1] -= ks - 1
        shape[-2] -= ks - 1
        output = np.zeros(shape)
        for c_out in range(C_out):
            for h in range(output.shape[-2]):
                for w in range(output.shape[-1]):
                    input_slice = input[:, :, h: h + ks, w: w + ks]
                    kernel_slice = kernel[c_out]
                    result = np.tensordot(
                        input_slice,
                        kernel_slice,
                        axes=[[-3,-2,-1], [-3, -2, -1]]
                    ).flatten()
                    output[:, c_out, h, w] = result

        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = input
        kernel = self.parameters[0]
        bias = self.parameters[1]
        input_p = self._pad_zeros(input, self.kernel_size // 2)
        return self._cross_correlate(input_p, kernel) + bias[None, :, None, None]

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        input_grad = np.zeros(self._input.shape)
        kernel = self.parameters[0]

        output_grad_p = self._pad_zeros(output_grad, self.kernel_size // 2)
        for c_in in range(self.in_channels):
            kernel_slice = kernel[:, c_in, ::-1, ::-1][None, None]
            input_grad[:, c_in:c_in+1] = self._cross_correlate(output_grad_p, kernel_slice)

        input_p = self._pad_zeros(self._input, self.kernel_size // 2)
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                inp_slice = input_p[None, :, c_in]
                out_g_slice = output_grad[None, :, c_out]
                self.parameters_grads[0][c_out, c_in] = self._cross_correlate(inp_slice, out_g_slice)

        self.parameters_grads[1] = np.sum(output_grad, axis=(0, 2, 3))
        return input_grad
