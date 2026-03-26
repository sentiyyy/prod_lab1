import random
import time
import numpy as np
from typing import Any, Optional, List
from tabulate import tabulate
import matplotlib.pyplot as plt
from model import ConstructiveNumber, to_constructive

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def condition_number(func, x_point: np.ndarray) -> float:
    """
    Вычисляет число обусловленности через отношение максимального и 
    минимального ненулевого собственного значения Гессиана
    """
    hessian = func.hessian(x_point)

    eigenvalues = np.linalg.eigvalsh(hessian)

    non_zero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]

    if len(non_zero_eigs) < 2:
        return 1.0
    
    non_zero_eigs = np.sort(non_zero_eigs)

    return float(non_zero_eigs[-1] / non_zero_eigs[0])

# Запускает оптимизацию и собирает метрики для сравнения методов
def run_experiment(
    func: Any,
    optimizer: Any,
    x_start: np.ndarray,
    true_minimum: Optional[np.ndarray] = None,
) -> dict:
    start_time = time.time()
    result = optimizer.optimize(func, x_start)
    elapsed = time.time() - start_time

    metrics = {
        "method": result.method_name,
        "func_name": func.name,
        "x_opt": result.x_opt,
        "f_opt": result.f_opt,
        "n_iterations": result.n_iterations,
        "n_func_calls": func.call_count,
        "n_grad_calls": getattr(func, "grad_count", None),
        "n_hessian_calls": getattr(func, "hessian_calls", None), # число вызовов Гессиана (если поддерживается функцией)
        "time_sec": elapsed,
        "converged": result.converged,
        "result": result, # полный результат для дальнейшего анализа и визуализации
    }

    if true_minimum is not None:
        metrics["dist_to_true_min"] = np.linalg.norm(result.x_opt - true_minimum)

    return metrics

def track_eps_during_optimization(func, result, eps_init: float = 1e-6) -> List[float]:
    """
    Отслеживаем не сумму eps координат (она константа),
    а eps значения функции на конструктивной окрестности текущей точки.
    """
    eps_history = []

    for x_point in result.history_x:
        x_cn = to_constructive(x_point.tolist(), eps=eps_init)
        f_cn = func(x_cn)

        if isinstance(f_cn, ConstructiveNumber):
            eps_history.append(f_cn.eps)
        else:
            eps_history.append(np.nan)

    return eps_history

def print_results_table_custom(metrics_list):
    headers = [
        "Метод",
        "Функция",
        "f(x*)",
        "Итер.",
        "Вызовы f",
        "Вызовы grad",
        "Время, с",
        "Сошёлся"
    ]

    rows = []
    for m in metrics_list:
        rows.append([
            m["method"],
            m["func_name"],
            f"{m['f_opt']:.4g}",
            m["n_iterations"],
            m["n_func_calls"],
            m.get("n_grad_calls", 0),
            f"{m['time_sec']:.4f}",
            "да" if m["converged"] else "нет",
        ])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def plot_convergence(results_list, title='Скорость сходимости'):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in results_list:
        label = f"{m['method']} | {m['func_name']}"
        ax.semilogy(m['result'].history_f, label=label)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('f(x) (лог. шкала)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Визуализация траектории оптимизации на уровне линий уровня функции (2D срез)
def plot_trajectory_2d(func, result, axis_indices=(0, 1), n_levels=30, title='Траектория оптимизации'):
    i_ax, j_ax = axis_indices
    traj = np.array(result.history_x)

    margin = 0.3
    x_min, x_max = traj[:, i_ax].min() - margin, traj[:, i_ax].max() + margin
    y_min, y_max = traj[:, j_ax].min() - margin, traj[:, j_ax].max() + margin

    xg = np.linspace(x_min, x_max, 120)
    yg = np.linspace(y_min, y_max, 120)
    X, Y = np.meshgrid(xg, yg)

    x_fixed = result.x_opt.copy()
    Z = np.zeros_like(X)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            pt = x_fixed.copy()
            pt[i_ax] = X[r, c]
            pt[j_ax] = Y[r, c]
            Z[r, c] = float(func(pt))

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, Z, levels=n_levels, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=n_levels, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='f(x)')

    ax.plot(traj[:, i_ax], traj[:, j_ax], 'o-', color='red',
            markersize=3, linewidth=1.5, label='Траектория', alpha=0.8)
    ax.scatter(traj[0, i_ax], traj[0, j_ax], color='lime', s=100, zorder=5, label='Старт')
    ax.scatter(traj[-1, i_ax], traj[-1, j_ax], color='red', s=120, marker='*', zorder=5, label='Финиш')

    ax.set_xlabel(f'x[{i_ax}]')
    ax.set_ylabel(f'x[{j_ax}]')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Отображает изменение ε-значений функции на каждой итерации оптимизации
def plot_eps_dynamics(
    eps_histories,
    labels,
    title='Динамика ε при оптимизации',
    max_iter_to_show=None
):
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist, lbl in zip(eps_histories, labels):
        hist = np.array(hist, dtype=float)
        finite_mask = np.isfinite(hist)
        if not finite_mask.any():
            continue

        last_finite = np.where(finite_mask)[0][-1] + 1
        hist = hist[:last_finite]

        if max_iter_to_show is not None:
            hist = hist[:max_iter_to_show]

        if len(hist) == 0:
            continue

        ax.plot(hist, label=lbl)

    ax.set_xlabel('Итерация')
    ax.set_ylabel('ε значения функции')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
