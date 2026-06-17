# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """

    matrix_data = backend.copy(tensor)
    shape = backend.shape(tensor)
    d = len(shape)

    if d == 1:
        core = backend.reshape(matrix_data, (1, shape[0], 1))
        return TTTensor([core])

    tensor_norm = backend.norm(tensor)
    if tensor_norm > 1e-30:
        delta = (eps / math.sqrt(d-1)) * tensor_norm
    else:
        delta = 0.0

    cores = []
    r = [1] * (d + 1)
    for k in range(1, d):
        rows = r[k - 1] * shape[k - 1]
        cols = backend.size(matrix_data) // rows
        matrix_2d = backend.reshape(matrix_data, (rows, cols))
        U, S, Vt = backend.svd(matrix_2d, full_matrices=False)

        r_k = _compute_truncated_rank(S, delta, max_rank)
        r[k] = r_k

        U_trunc = _truncate_columns(U, r_k, backend)
        core_shape = (r[k - 1], shape[k - 1], r[k])
        cores.append(backend.reshape(U_trunc, core_shape))
        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)
        matrix_data = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)

    last_core = backend.reshape(matrix_data, (r[d - 1], shape[d - 1], 1))
    cores.append(last_core)

    return TTTensor(cores)




# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    total_elements = S.shape[0]

    if abs(S[0]) < 1e-12:
        return 1
    lmt = max(1e-12, 1e-8 * S[0])
    rnk = total_elements

    for j in range(total_elements):
        if S[j] <= lmt:
            rnk = j
            break

    if delta > 0:
        squared_sum = 0.0
        for j in range(rnk - 1, -1, -1):
            squared_sum += S[j] ** 2
            if squared_sum > delta ** 2:
                break
            rnk = j
    rnk = max(1, rnk)

    if max_rank is not None:
        rnk = min(rnk, max_rank)

    return rnk


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    rows = backend.shape(matrix)[0]
    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            val = backend.get_element(matrix, (i, j))
            result = backend.set_element(result, (i, j), val)
    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    cols = backend.shape(matrix)[1]
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            val = backend.get_element(matrix, (i, j))
            result = backend.set_element(result, (i, j), val)
    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    result = backend.zeros((rank,))
    for i in range(rank):
        val = backend.get_element(vector, (i,))
        result = backend.set_element(result, (i,), val)
    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    cols = backend.shape(matrix)[1]
    result = backend.zeros((rank, cols))
    for i in range(rank):
        factor = backend.get_element(diag_vec, (i,))
        for j in range(cols):
            val = backend.get_element(matrix, (i, j))
            result = backend.set_element(result, (i, j), factor * val)

    return result