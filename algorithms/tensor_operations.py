# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape or tt1.order != tt2.order:
        raise ValueError('Формы и порядки тензоров должны совпадать для сложения')

    new_cores = []
    d = tt1.order

    for k in range(d):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]

        sh1 = backend.shape(c1)
        sh2 = backend.shape(c2)
        n_k = sh1[1]

        r_left_new = 1 if k == 0 else (sh1[0] + sh2[0])
        r_right_new = 1 if k == d - 1 else (sh1[2] + sh2[2])

        new_core = backend.zeros((r_left_new, n_k, r_right_new))

        for n_idx in range(n_k):
            for i in range(sh1[0]):
                for j in range(sh1[2]):
                    val = backend.get_element(c1, (i, n_idx, j))
                    target_i = i
                    target_j = j
                    backend.set_element(new_core, (target_i, n_idx, target_j), val)

            for i in range(sh2[0]):
                for j in range(sh2[2]):
                    val = backend.get_element(c2, (i, n_idx, j))
                    target_i = 0 if k == 0 else (sh1[0] + i)
                    target_j = 0 if k == d - 1 else (sh1[2] + j)
                    backend.set_element(new_core, (target_i, n_idx, target_j), val)

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    new_cores = list(tt.cores)
    new_cores[0] = backend.scale(new_cores[0], alpha)
    return TTTensor(new_cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape or tt1.order != tt2.order:
        raise ValueError("Формы и порядки тензоров должны совпадать")

    new_cores = []
    for k in range(tt1.order):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]

        sh1 = backend.shape(c1)
        sh2 = backend.shape(c2)
        n_k = sh1[1]
        new_shape = (sh1[0] * sh2[0], n_k, sh1[2] * sh2[2])
        new_core = backend.zeros(new_shape)

        for n_idx in range(n_k):
            for i1 in range(sh1[0]):
                for j1 in range(sh1[2]):
                    val1 = backend.get_element(c1, (i1, n_idx, j1))

                    for i2 in range(sh2[0]):
                        for j2 in range(sh2[2]):
                            val2 = backend.get_element(c2, (i2, n_idx, j2))

                            target_i = i1 * sh2[0] + i2
                            target_j = j1 * sh2[2] + j2
                            backend.set_element(new_core, (target_i, n_idx, target_j), val1 * val2)

        new_cores.append(new_core)

    return TTTensor(new_cores)

def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape or tt1.order != tt2.order:
        raise ValueError("Формы и порядки тензоров должны совпадать")

    res = backend.eye(1)

    for k in range(tt1.order):
        c1 = tt1.cores[k]
        c2 = tt2.cores[k]

        sh1 = backend.shape(c1)
        sh2 = backend.shape(c2)

        m1 = backend.reshape(c1, (sh1[0], sh1[1] * sh1[2]))

        res_m1 = backend.matmul(res, m1)
        res_m1_flat = backend.reshape(res_m1, (sh2[0] * sh1[1], sh1[2]))
        m2_flat = backend.reshape(c2, (sh2[0] * sh2[1], sh2[2]))
        m2_T = backend.transpose(m2_flat)
        res = backend.matmul(m2_T, res_m1_flat)

    return backend.get_element(res, (0, 0))

def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    a_norm_sq = tt_dot(tt1, tt1, backend)
    b_norm_sq = tt_dot(tt2, tt2, backend)
    ab_dot = tt_dot(tt1, tt2, backend)
    return math.sqrt(max(0, a_norm_sq - 2 * ab_dot + b_norm_sq))