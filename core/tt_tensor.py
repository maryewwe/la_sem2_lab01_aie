# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations
import random
import itertools
from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        self.cores = cores
        self.order = len(cores)

        shape_list = []
        ranks = [cores[0].shape[0]]

        for k, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError(f'Ядро должно быть 3D, получено {core.ndim}D')

            if core.shape[0] != ranks[-1]:
                raise ValueError('Ошибка стыковки рангов в ядре')

            shape_list.append(core.shape[1])
            ranks.append(core.shape[2])

        if ranks[-1] != 1:
            raise ValueError('Последний ранг должен быть равен 1')

        if ranks[0] != 1:
            raise ValueError('Первый ранг должен быть равен 1')

        self.shape = tuple(shape_list)
        self.ranks = tuple(ranks)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        if seed is not None:
            random.seed(seed)

        d = len(shape)

        if len(ranks) == d-1:
            full_ranks = (1,) + tuple(ranks) + (1,)
        else:
            full_ranks = tuple(ranks)

        if len(full_ranks) != d + 1:
            raise ValueError('Некорректное количество рангов')

        cores = []
        for k in range(d):
            r_prev = full_ranks[k]
            n = shape[k]
            r_next = full_ranks[k+1]
            data = [random.uniform(-1.0, 1.0) for _ in range(r_prev * n * r_next)]

            core = DenseTensor((r_prev, n, r_next), data=data)
            cores.append(core)

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """

        res = [1.0]

        for k in range(self.order):
            core = self.cores[k]
            idx = indices[k]
            r_prev, _, r_next = core.shape
            new_res = [0.0] * r_next

            for r2 in range(r_next):
                val_sum = 0.0
                for r1 in range(r_prev):
                    val_sum += res[r1] * core[r1, idx, r2]
                new_res[r2] = val_sum
            res = new_res
        return res[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        full_data = []

        for idx in itertools.product(*[range(n) for n in self.shape]):
            full_data.append(self.get_element(idx))

        return DenseTensor(self.shape, data=full_data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = 1
        for n in self.shape:
            full_size *= n
        return full_size / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        core_shapes = [str(c.shape) for c in self.cores]
        return f"TTTensor(shape={self.shape}, ranks={self.ranks}, cores={core_shapes})"

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()