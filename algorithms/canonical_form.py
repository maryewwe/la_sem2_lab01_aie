# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """

    cores = [backend.copy(c) for c in tt.cores]
    d = len(cores)

    if d <= 1:
        return TTTensor(cores)

    for k in range(d-1):
        sh = backend.shape(cores[k])
        rows = sh[0] * sh[1]
        cols = sh[2]
        matrix = backend.reshape(cores[k], (rows, cols))
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _numerical_rank(S)

        U_trunc = _truncate_columns(U, rank, backend)
        cores[k] = backend.reshape(U_trunc, (sh[0], sh[1], rank))

        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        remind = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        next_sh = backend.shape(next_core)
        next_matrix = backend.reshape(next_core, (next_sh[0], next_sh[1] * next_sh[2]))

        updated = backend.matmul(remind, next_matrix)
        cores[k+1] = backend.reshape(updated, (rank, next_sh[1], next_sh[2]))

    return TTTensor(cores)

def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [backend.copy(c) for c in tt.cores]
    d = len(cores)

    if d <= 1:
        return TTTensor(cores)


    for k in range(d - 1, 0, -1):
        sh = backend.shape(cores[k])
        rows = sh[0]
        cols = sh[1] * sh[2]
        matrix = backend.reshape(cores[k], (rows, cols))

        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _numerical_rank(S)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        cores[k] = backend.reshape(Vt_trunc, (rank, sh[1], sh[2]))
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        remind = _multiply_columns_by_diag(U_trunc, S_trunc, backend)
        prev_core = cores[k - 1]
        prev_sh = backend.shape(prev_core)
        prev_matrix = backend.reshape(prev_core, (prev_sh[0] * prev_sh[1], prev_sh[2]))

        updated = backend.matmul(prev_matrix, remind)
        cores[k - 1] = backend.reshape(updated, (prev_sh[0], prev_sh[1], rank))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    total = S.shape[0]
    if total == 0:
        return 1

    first_val = abs(float(S[0]))
    limit = max(abs_tol, rel_tol * first_val)

    rank = 0
    for j in range(total):
        if abs(float(S[j])) > limit:
            rank = j + 1
        else:
            break

    return max(1, rank)


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
        rank:     длина диагонального вектора
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    sh = backend.shape(matrix)
    rows, cols = sh[0], sh[1]
    result = backend.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            factor = backend.get_element(diag_vec, (j,))
            val = backend.get_element(matrix, (i, j))
            result = backend.set_element(result, (i, j), val * factor)
    return result