import numpy as np
from typing import List


# Блок 1: Конструктивное число
class ConstructiveNumber:
    """
    Конструктивное число - это интервал [a, b],
    где a, b - вещественные числа (на практике float).

    Идея: вместо точного вещественного числа мы храним нижнюю и верхнюю границы,
    между которыми "сидит" настоящее число. Точность (ширина интервала) ε = b - a

    Параметр α ∈ [0, 1] позволяет получить конкретное значение:
      - при α=0 -> возвращаем a (левую границу)
      - при α=1 -> возвращаем b (правую границу)
      - при α=0.5 -> возвращаем середину интервала
    """

    def __init__(self, a: float, b: float):
        """
        Создаём конструктивное число по двум границам a ≤ b, где a - нижняя граница интервала;
        b - верхняя граница интервала.
        """
        # Убеждаемся, что границы идут в правильном порядке
        if a > b:
            a, b = b, a
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def from_real(cls, x: float, eps: float) -> "ConstructiveNumber":
        """
        Создаём конструктивное число из вещественного x с точностью ε.
        Интервал симметричен относительно x.
        """
        half = abs(eps) / 2.0
        return cls(x - half, x + half)

    @property
    def eps(self) -> float:
        """Текущая точность (ширина интервала ε = b - a)."""
        return self.b - self.a

    def get_value(self, alpha: float = 0.5) -> float:
        """Получить конкретное вещественное число из интервала."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        return self.a + alpha * (self.b - self.a)  # a + alpha * (b - a)

    # 1.1 Арифметические операции
    # При арифметике интервалов правила такие:
    # [a1, b1] + [a2, b2] = [a1+a2, b1+b2]
    # [a1, b1] - [a2, b2] = [a1-b2, b1-a2]
    # [a1, b1] * [a2, b2] = [min(...), max(...)] (все 4 комбинации)
    # Деление возможно только если [a2, b2] не содержит 0:
    # Тогда: [a1, b1] / [a2, b2] = [a1, b1] * [1/b2, 1/a2]

    def __add__(self, other) -> "ConstructiveNumber":
        """Сложение двух конструктивных чисел или конструктивного с обычным."""
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a + other.a, self.b + other.b)
        else:
            # Прибавляем обычное число - просто сдвигаем интервал
            other = float(other)
            return ConstructiveNumber(self.a + other, self.b + other)

    def __radd__(self, other) -> "ConstructiveNumber":
        """Поддержка записи: число + конструктивное число."""
        return self.__add__(other)

    def __sub__(self, other) -> "ConstructiveNumber":
        """Вычитание."""
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a - other.b, self.b - other.a)
        else:
            other = float(other)
            return ConstructiveNumber(self.a - other, self.b - other)

    def __rsub__(self, other) -> "ConstructiveNumber":
        """Поддержка записи: число - конструктивное число."""
        other = float(other)
        return ConstructiveNumber(other - self.b, other - self.a)

    def __mul__(self, other) -> "ConstructiveNumber":
        """Умножение."""
        if isinstance(other, ConstructiveNumber):
            # Перемножаем все 4 комбинации концов интервалов
            products = [
                self.a * other.a,
                self.a * other.b,
                self.b * other.a,
                self.b * other.b,
            ]
            return ConstructiveNumber(min(products), max(products))
        else:
            other = float(other)
            products = [self.a * other, self.b * other]
            return ConstructiveNumber(min(products), max(products))

    def __rmul__(self, other) -> "ConstructiveNumber":
        """Поддержка записи: число * конструктивное число."""
        return self.__mul__(other)

    def __truediv__(self, other) -> "ConstructiveNumber":
        """
        Деление конструктивных чисел.
        Если делим на интервал, содержащий 0 - выбрасываем ошибку.
        Иначе используем правило:
        [a, b] / [c, d] = [a, b] * [1/d, 1/c]
        """
        if isinstance(other, ConstructiveNumber):
            if other.a <= 0 <= other.b:
                raise ZeroDivisionError("Деление на интервал, содержащий 0")

            inv_other = ConstructiveNumber(1.0 / other.b, 1.0 / other.a)
            return self * inv_other
        else:
            other = float(other)
            if other == 0:
                raise ZeroDivisionError("Деление на 0")

            vals = [self.a / other, self.b / other]
            return ConstructiveNumber(min(vals), max(vals))

    def __rtruediv__(self, other) -> "ConstructiveNumber":
        """Поддержка записи: число / конструктивное число."""
        other = float(other)
        if self.a <= 0 <= self.b:
            raise ZeroDivisionError("Деление на интервал, содержащий 0")
        inv_self = ConstructiveNumber(1.0 / self.b, 1.0 / self.a)
        return inv_self.__mul__(other)

    def __neg__(self) -> "ConstructiveNumber":
        """Унарный минус."""
        return ConstructiveNumber(-self.b, -self.a)

    # 1.2 Операции сравнения
    # Сравнение с интервальной гарантией:
    #  x < y, если весь интервал x левее интервала y
    #  x > y, если весь интервал x правее интервала y
    # Проверка приближённого совпадения интервалов по границам
    def __eq__(self, other):
        if not isinstance(other, ConstructiveNumber):
            return False
        tol = 1e-12
        return abs(self.a - other.a) < tol and abs(self.b - other.b) < tol

    def __lt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.b < other.a  # интервальная логика
        else:
            return self.b < float(other)

    def __le__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.b <= other.a or self.__eq__(other)
        return self.b <= float(other)

    def __gt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a > other.b
        else:
            return self.a > float(other)

    def __ge__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a >= other.b or self.__eq__(other)
        return self.a >= float(other)

    def __float__(self) -> float:
        """Приведение к float: возвращаем середину интервала."""
        return self.get_value(alpha=0.5)

    def __repr__(self) -> str:
        return f"ConstructiveNumber([{self.a:.6g}, {self.b:.6g}], ε={self.eps:.2e})"

    def __abs__(self) -> "ConstructiveNumber":
        """
        Модуль конструктивного числа.
        Если интервал содержит ноль, результат [0, max(|a|,|b|)].
        Иначе - [min(|a|,|b|), max(|a|,|b|)].
        """
        if self.a <= 0 <= self.b:
            return ConstructiveNumber(0.0, max(abs(self.a), abs(self.b)))
        else:
            return ConstructiveNumber(
                min(abs(self.a), abs(self.b)), max(abs(self.a), abs(self.b))
            )


# Вспомогательная функция: оборачиваем список обычных float в конструктивные числа
def to_constructive(point: List[float], eps: float = 1e-6) -> List[ConstructiveNumber]:
    """
    Конвертируем список обычных чисел в список конструктивных с заданной точностью ε,
    где point: список координат (обычные числа), eps: ширина интервала (погрешность)
    """
    return [
        ConstructiveNumber.from_real(x, eps) for x in point
    ]  # список конструктивных чисел


def from_constructive(
    point: List[ConstructiveNumber], alpha: float = 0.5
) -> List[float]:
    """
    Конвертируем список конструктивных чисел обратно в обычные float,
    где point: список конструктивных чисел, alpha: параметр для get_value (0=нижняя граница, 1=верхняя, 0.5=середина).
    """
    return [x.get_value(alpha) for x in point]  # список обычных чисел


# Блок 2: Чёрные ящики
class BlackBox:
    """
    Базовый класс для тестовых функций оптимизации.

    У каждого ящика единый интерфейс:
      - __call__(x) - вычислить значение функции
      - gradient(x) - вычислить градиент (вектор частных производных)
      - call_count - счётчик вызовов функции
      - grad_count - счётчик вызовов градиента

    Совместим с конструктивными числами:
    если передать список ConstructiveNumber, вернём ConstructiveNumber.
    """

    def __init__(self, name, n_dim: int):
        self.name = name
        self.n_dim = n_dim
        self.call_count = 0
        self.grad_count = 0
        self.hessian_calls = 0

    def reset_counters(self):
        self.call_count = 0
        self.grad_count = 0
        self.hessian_calls = 0

    def _check_dim(self, x):
        """Проверяем, что размерность входного вектора совпадает с ожидаемой."""
        if len(x) != self.n_dim:
            raise ValueError(
                f"{self.name}: ожидалась размерность {self.n_dim}, получено {len(x)}"
            )

    def __call__(self, x):
        """
        Вычисление значения функции.
        Поддерживает:
        - обычные numpy / list (возвращает float)
        - список ConstructiveNumber (возвращает ConstructiveNumber)
        Определяет тип входа и выбирает численную или интервальную реализацию функции.
        """
        self.call_count += 1

        if isinstance(x, np.ndarray):
            x = x.tolist()

        self._check_dim(x)

        if (
            isinstance(x, (list, tuple))
            and len(x) > 0
            and isinstance(x[0], ConstructiveNumber)
        ):
            return self._evaluate_constructive(x)

        return self._evaluate_numeric(np.array(x, dtype=float))

    def gradient(self, x):
        if isinstance(x, np.ndarray):
            x = x.tolist()

        self._check_dim(x)
        self.grad_count += 1
        return self._gradient(np.array(x, dtype=float))

    def hessian(self, x):
        if isinstance(x, np.ndarray):
            x = x.tolist()

        self._check_dim(x)
        self.hessian_calls += 1
        return self._hessian(np.array(x, dtype=float))

    def _evaluate_numeric(self, x):
        raise NotImplementedError("Подкласс должен реализовать _evaluate_numeric")

    def _evaluate_constructive(self, x):
        raise NotImplementedError("Подкласс должен реализовать _evaluate_constructive")

    def _gradient(self, x):
        raise NotImplementedError("Подкласс должен реализовать _gradient")

    def _hessian(self, x):
        raise NotImplementedError("Подкласс должен реализовать _hessian")


class QuadraticWellConditioned(BlackBox):
    """
    Квадратичная функция 6-мерная с хорошим числом обусловленности (≈ 1).

    f(x) = sum_i (x_i - center_i)^2

    Число обусловленности матрицы Гессе ≈ 1, потому что все коэффициенты одинаковые.
    Это "идеальная" функция - градиентный спуск сходится быстро.
    Минимум: в точке center = [1, 2, 3, 4, 5, 6]
    """

    def __init__(self):
        super().__init__(name="Квадратичная_6мерная_хорошая", n_dim=6)
        # Центр (минимум) - сместим из нуля, чтобы было интереснее
        self.center = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # Коэффициенты все примерно одинаковые -> число обусловленности матрицы Гессе ≈ 1
        # (почти идеальная квадратичная функция)
        self.coeffs = np.array([1.0, 1.0, 1.1, 0.9, 1.0, 1.0])

    def _evaluate_numeric(self, x):
        """Вычисляет значение квадратичной функции."""
        diff = x - self.center
        return float(np.sum(self.coeffs * diff * diff))

    def _evaluate_constructive(self, x):
        total = ConstructiveNumber(0.0, 0.0)
        for xi, ci, ai in zip(x, self.coeffs, self.center):
            diff = xi - ai
            total = total + ci * diff * diff
        return total

    def _gradient(self, x):
        """Градиент квадратичной функции: 2 * c_i * (x_i - center_i)"""
        return 2.0 * self.coeffs * (x - self.center)

    def _hessian(self, x):
        """Матрица Гессе - диагональная, так как переменные независимы."""
        return np.diag(2.0 * self.coeffs)


class QuadraticPoorConditioned(BlackBox):
    """
    Квадратичная функция 4-мерная с плохим числом обусловленности (≈ 100).

    f(x) = sum_i c_i * (x_i - center_i)^2

    Коэффициенты сильно разные: от 1 до 100 -> число обусловленности ≈ 100.
    На такой функции градиентный спуск "зигзагует" и сходится медленно.
    Минимум: в точке center = [1, 2, 3, 4]
    """

    def __init__(self):
        super().__init__(name="Квадратичная_4мерная_плохая", n_dim=4)
        self.center = np.array([1.0, 2.0, 3.0, 4.0])
        # Коэффициенты различаются по порядку величины (1–100) -> число обусловленности = 100/1 = 100
        self.coeffs = np.array([1.0, 10.0, 50.0, 100.0])

    def _evaluate_numeric(self, x):
        diff = x - self.center
        return float(np.sum(self.coeffs * diff * diff))

    def _evaluate_constructive(self, x):
        total = ConstructiveNumber(0.0, 0.0)
        for xi, ci, ai in zip(x, self.coeffs, self.center):
            diff = xi - ai
            total = total + ci * diff * diff
        return total

    def _gradient(self, x):
        return 2.0 * self.coeffs * (x - self.center)

    def _hessian(self, x):
        return np.diag(2.0 * self.coeffs)


class RosenbrockFunction(BlackBox):
    """
    Функция Розенброка 3-мерная.

    f(x, y, z) = (1-x)^2 + 100*(y-x^2)^2 + (1-y)^2 + 100*(z-y^2)^2

    "Банановая" (ill-conditioned) функция Розенброк - минимум в точке (1, 1, 1),
    но добраться до него сложно: функция вытянута вдоль изогнутой долины,
    зато хорошо тестирует устойчивость методов.
    """

    def __init__(self):
        super().__init__(name="Розенброк_3мерный", n_dim=3)

    def _evaluate_numeric(self, x):
        x0, x1, x2 = x[0], x[1], x[2]
        term1 = (1.0 - x0) * (1.0 - x0) + 100.0 * (x1 - x0 * x0) * (x1 - x0 * x0)
        term2 = (1.0 - x1) * (1.0 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1)
        return float(term1 + term2)

    def _evaluate_constructive(self, x):
        """
        Для простоты используем обычное численное вычисление,
        так как реализация через интервалы сложнее.
        """
        return self._evaluate_numeric(x)

    def _gradient(self, x):
        x0, x1, x2 = x[0], x[1], x[2]
        df_dx0 = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0**2)
        df_dx1 = 200.0 * (x1 - x0**2) - 2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1**2)
        df_dx2 = 200.0 * (x2 - x1**2)
        return np.array([df_dx0, df_dx1, df_dx2])

    def _hessian(self, x):
        x0, x1, x2 = x[0], x[1], x[2]
        H = np.zeros((3, 3))
        H[0, 0] = 2 - 400 * x1 + 1200 * x0**2
        H[0, 1] = H[1, 0] = -400 * x0
        H[1, 1] = 202 - 400 * x2 + 1200 * x1**2
        H[1, 2] = H[2, 1] = -400 * x1
        H[2, 2] = 200
        return H


# Блок 3: Методы оптимизации
class OptimizationResult:
    """
    Результат работы метода оптимизации.
    Хранит:
    - найденную точку минимума
    - значение функции
    - число итераций
    - историю движения (trajectory)
    - информацию о сходимости метода
    """

    def __init__(
        self,
        x_opt: np.ndarray,  # найденный минимум
        f_opt: float,  # значение функции в минимуме
        n_iterations: int,  # число итераций
        history_x: List[np.ndarray],  # траектория точек (все итерации)
        history_f: List[float],  # значения функции по итерациям
        converged: bool,  # сошёлся ли метод
        method_name: str,
    ):
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.n_iterations = n_iterations
        self.history_x = history_x
        self.history_f = history_f
        self.converged = converged
        self.method_name = method_name

    def __repr__(self):
        status = "сошёлся" if self.converged else "не сошёлся"
        return (
            f"[{self.method_name}] {status} за {self.n_iterations} итераций, "
            f"f(x*) = {self.f_opt:.6g}"
        )


class NelderMead:
    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 10000,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def optimize(
        self,
        func: BlackBox,
        x_start: np.ndarray,
        initial_step: float = 0.5,
    ) -> OptimizationResult:
        """
        Метод Нелдера-Мида (симплекс-метод, без градиентов).
        Идея:
        - строим симплекс (n+1 точек)
        - на каждой итерации отражаем / растягиваем / сжимаем его
        - двигаемся к минимуму функции
        """
        func.reset_counters()
        x_start = np.array(x_start, dtype=float)
        n = len(x_start)

        simplex = [x_start.copy()]
        for i in range(n):
            vertex = x_start.copy()
            vertex[i] += initial_step
            simplex.append(vertex)

        f_values = [float(func(v)) for v in simplex]

        best_idx = int(np.argmin(f_values))
        history_x = [simplex[best_idx].copy()]
        history_f = [f_values[best_idx]]

        converged = False
        n_iterations = 0

        for _ in range(self.max_iter):
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            x_best = simplex[0]
            x_worst = simplex[-1]
            f_worst = f_values[-1]

            simplex_diameter = max(
                np.linalg.norm(simplex[i] - x_best) for i in range(1, len(simplex))
            )
            f_spread = max(f_values) - min(f_values)

            if simplex_diameter < self.tol and f_spread < self.tol:
                converged = True
                break

            centroid = np.mean(simplex[:-1], axis=0)

            x_reflect = centroid + self.alpha * (centroid - x_worst)
            f_reflect = float(func(x_reflect))

            if f_values[0] <= f_reflect < f_values[-2]:
                simplex[-1] = x_reflect
                f_values[-1] = f_reflect

            elif f_reflect < f_values[0]:
                x_expand = centroid + self.gamma * (x_reflect - centroid)
                f_expand = float(func(x_expand))
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    f_values[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect

            else:
                x_contract = centroid + self.rho * (x_worst - centroid)
                f_contract = float(func(x_contract))

                if f_contract < f_worst:
                    simplex[-1] = x_contract
                    f_values[-1] = f_contract
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + self.sigma * (simplex[i] - x_best)
                        f_values[i] = float(func(simplex[i]))

            n_iterations += 1

            best_idx = int(np.argmin(f_values))
            history_x.append(simplex[best_idx].copy())
            history_f.append(f_values[best_idx])

        best_idx = int(np.argmin(f_values))
        return OptimizationResult(
            x_opt=simplex[best_idx],
            f_opt=f_values[best_idx],
            n_iterations=n_iterations,
            history_x=history_x,
            history_f=history_f,
            converged=converged,
            method_name="Нелдер-Мид (0-й порядок)",
        )


class GradientDescent:
    def __init__(
        self,
        lr: float = 0.01,
        lr_strategy: str = "fixed",
        tol: float = 1e-6,
        max_iter: int = 10000,
        backtrack_beta: float = 0.5,
        backtrack_c: float = 1e-4,
    ):
        if lr_strategy not in {"fixed", "backtracking"}:
            raise ValueError("lr_strategy must be 'fixed' or 'backtracking'")

        self.lr = lr
        self.lr_strategy = lr_strategy
        self.tol = tol
        self.max_iter = max_iter
        self.backtrack_beta = backtrack_beta
        self.backtrack_c = backtrack_c

    def optimize(
        self,
        func: BlackBox,
        x_start: np.ndarray,
    ) -> OptimizationResult:
        """
        Градиентный спуск.
        На каждой итерации:
        - считаем градиент
        - двигаемся в сторону антиградиента
        """
        func.reset_counters()
        x = np.array(x_start, dtype=float)

        history_x = [x.copy()]
        history_f = [float(func(x))]

        converged = False
        n_iterations = 0

        for _ in range(self.max_iter):
            grad = func.gradient(x)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < self.tol:
                # Критерий остановки: норма градиента меньше tol
                converged = True
                break

            if len(history_f) > 1 and abs(history_f[-1] - history_f[-2]) < self.tol:
                # Критерий стационарности: изменение функции меньше tol
                converged = True
                break

            if self.lr_strategy == "backtracking":
                step = self._backtracking_line_search(func, x, grad)
            else:
                step = self.lr

            x = x - step * grad
            n_iterations += 1

            history_x.append(x.copy())
            history_f.append(float(func(x)))

        return OptimizationResult(
            x_opt=x,
            f_opt=float(func(x)),
            n_iterations=n_iterations,
            history_x=history_x,
            history_f=history_f,
            converged=converged,
            method_name=f"Градиентный спуск (lr={self.lr}, стратегия={self.lr_strategy})",
        )

    def _backtracking_line_search(
        self,
        func: BlackBox,
        x: np.ndarray,
        grad: np.ndarray,
    ) -> float:
        """
        Поиск шага методом backtracking (условие Армихо).

        Уменьшаем шаг, пока не выполнится условие убывания функции:
        f(x - step * grad) <= f(x) - c * step * ||grad||^2
        """
        step = self.lr
        f_current = float(func(x))
        grad_sq = np.dot(grad, grad)

        while (
            float(func(x - step * grad)) > f_current - self.backtrack_c * step * grad_sq
        ):
            step *= self.backtrack_beta
            if step < 1e-15:
                break

        return step


class NewtonMethod:
    def __init__(self, tol: float = 1e-7, max_iter: int = 50, lr: float = 1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr

    def optimize(self, func: BlackBox, x_start: np.ndarray) -> OptimizationResult:
        """
        Метод Ньютона (второй порядок).
        Использует:
        - градиент
        - матрицу Гессе
        Шаг вычисляется как решение:
        H * delta = -grad
        """
        func.reset_counters()
        x = np.array(x_start, dtype=float)

        history_x = [x.copy()]
        history_f = [float(func(x))]
        converged = False
        n_iterations = 0

        for _ in range(self.max_iter):
            grad = func.gradient(x)

            if np.linalg.norm(grad) < self.tol:
                converged = True
                break

            hess = func.hessian(x)

            try:
                # Решаем H * delta_x = -grad (шаг Ньютона)
                delta_x = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                # Если матрица Гессе вырождена или плохо обусловлена, делаем шаг как в градиентном спуске
                delta_x = -grad

            x = x + self.lr * delta_x
            n_iterations += 1

            history_x.append(x.copy())
            history_f.append(float(func(x)))

        return OptimizationResult(
            x_opt=x,
            f_opt=float(func(x)),
            n_iterations=n_iterations,
            history_x=history_x,
            history_f=history_f,
            converged=converged,
            method_name=f"Метод Ньютона (lr={self.lr})",
        )
