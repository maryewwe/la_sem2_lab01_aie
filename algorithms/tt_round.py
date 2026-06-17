# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if tt.order <= 1:
        return tt.copy()

    tt = right_canonicalize(tt, backend)

    cores = [backend.copy(c) for c in tt.cores]
    d = len(cores)

    if d > 1:
        delta = (eps / math.sqrt(d - 1)) * backend.norm(tt.full())
    else:
        delta = 0.0

    for k in range(d - 1):
        sh = backend.shape(cores[k])
        matrix = backend.reshape(cores[k], (sh[0] * sh[1], sh[2]))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        cores[k] = backend.reshape(U_trunc, (sh[0], sh[1], rank))
        remind = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_sh = backend.shape(cores[k + 1])
        next_core = backend.reshape(cores[k + 1], (next_sh[0], next_sh[1] * next_sh[2]))

        updated = backend.matmul(remind, next_core)
        cores[k + 1] = backend.reshape(updated, (rank, next_sh[1], next_sh[2]))

    return TTTensor(cores)



# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    total_elements = S.shape[0]

    if abs(float(S[0])) < 1e-12:
        return 1

    lmt = max(1e-12, 1e-8 * float(S[0]))
    rank = total_elements

    for j in range(total_elements):
        if S[j] <= lmt:
            rank = j
            break

    if delta > 0:
        squared_sum = 0.0
        for j in range(rank - 1, -1, -1):
            squared_sum += S[j] ** 2
            if squared_sum > delta ** 2:
                break
            rank = j
    rank = max(1, rank)

    if max_rank is not None:
        rank = min(rank, max_rank)

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

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