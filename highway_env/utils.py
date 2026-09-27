from __future__ import annotations

import copy
import importlib
import itertools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, List, Mapping, Sequence, Tuple, Union

import numpy as np


# 常用类型
Vector = Union[np.ndarray, Sequence[float]]
ColumnVector = Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """将取值范围为 x 的数值 v，线性映射到目标范围 y。"""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def get_class_path(cls: Callable) -> str:
    return cls.__module__ + "." + cls.__qualname__


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    检查点是否位于矩形内部。

    :param point: 点 (x, y)
    :param rect_min: 矩形的最小坐标 x_min, y_min
    :param rect_max: 矩形的最大坐标 x_max, y_max
    """
    return (
        rect_min[0] <= point[0] <= rect_max[0]
        and rect_min[1] <= point[1] <= rect_max[1]
    )


def point_in_rotated_rectangle(
    point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float
) -> bool:
    """
    检查点是否位于旋转后的矩形内部。

    :param point: 点
    :param center: 矩形中心
    :param length: 矩形长度
    :param width: 矩形宽度
    :param angle: 矩形角度，单位为弧度
    :return: 点是否位于矩形内部
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def point_in_ellipse(
    point: Vector, center: Vector, angle: float, length: float, width: float
) -> bool:
    """
    检查点是否位于椭圆内部。

    :param point: 点
    :param center: 椭圆中心
    :param angle: 椭圆主轴角度
    :param length: 椭圆长轴
    :param width: 椭圆短轴
    :return: 点是否位于椭圆内部
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(
    rect1: tuple[Vector, float, float, float], rect2: tuple[Vector, float, float, float]
) -> bool:
    """
    判断两个旋转矩形是否相交。

    :param rect1: 第一个矩形 (center, length, width, angle)
    :param rect2: 第二个矩形 (center, length, width, angle)
    :return: 两个矩形是否相交
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def rect_corners(
    center: np.ndarray,
    length: float,
    width: float,
    angle: float,
    include_midpoints: bool = False,
    include_center: bool = False,
) -> list[np.ndarray]:
    """
    返回矩形各个角点的位置。

    :param center: 矩形中心
    :param length: 矩形长度
    :param width: 矩形宽度
    :param angle: 矩形角度
    :param include_midpoints: 是否包含各边的中点
    :param include_center: 是否包含矩形中心
    :return: 位置列表
    """
    center = np.array(center)
    half_l = np.array([length / 2, 0])
    half_w = np.array([0, width / 2])
    corners = [-half_l - half_w, -half_l + half_w, +half_l + half_w, +half_l - half_w]
    if include_center:
        corners += [[0, 0]]
    if include_midpoints:
        corners += [-half_l, half_l, -half_w, half_w]

    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ np.array(corners).T).T + np.tile(center, (len(corners), 1))


def has_corner_inside(
    rect1: tuple[Vector, float, float, float], rect2: tuple[Vector, float, float, float]
) -> bool:
    """
    检查 rect1 是否有角点位于 rect2 内部。

    :param rect1: 第一个矩形 (center, length, width, angle)
    :param rect2: 第二个矩形 (center, length, width, angle)
    """
    return any(
        [
            point_in_rotated_rectangle(p1, *rect2)
            for p1 in rect_corners(*rect1, include_midpoints=True, include_center=True)
        ]
    )


def project_polygon(polygon: Vector, axis: Vector) -> tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """
    计算区间 [minA, maxA] 与 [minB, maxB] 之间的距离。
    如果两个区间重叠，距离为负。
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(
    a: Vector, b: Vector, displacement_a: Vector, displacement_b: Vector
) -> tuple[bool, bool, np.ndarray | None]:
    """
    检查两个多边形是否相交。

    参考 https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: 多边形 A，用 [x, y] 点列表表示
    :param b: 多边形 B，用 [x, y] 点列表表示
    :param displacement_a: 多边形 A 的运动位移
    :param displacement_b: 多边形 B 的运动位移
    :return: 当前是否相交、之后是否相交、平移向量
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:], strict=False):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # 中心位置之差
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def confidence_ellipsoid(
    data: dict[str, np.ndarray],
    lambda_: float = 1e-5,
    delta: float = 0.1,
    sigma: float = 0.1,
    param_bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    计算参数 theta 的置信椭球，其中 y = theta^T phi。

    :param data: 字典 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: L2 正则化参数
    :param delta: 置信水平
    :param sigma: 噪声协方差
    :param param_bound: 参数范数的上界
    :return: 估计参数 theta、Gram 矩阵 G_N_lambda、半径 beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1 / sigma * np.transpose(phi) @ phi + lambda_ * np.identity(
        phi.shape[-1]
    )
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = (
        np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_**d) / delta))
        + np.sqrt(lambda_ * d) * param_bound
    )
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(
    data: dict, parameter_box: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    计算参数 theta 的置信多面体，其中 y = theta^T phi。

    :param data: 字典 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: 包含参数 theta 的盒状区间 [theta_min, theta_max]
    :return: 估计参数 theta、多面体顶点、Gram 矩阵 G_N_lambda、半径 beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(
        data, param_bound=param_bound
    )

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # 将参数和置信区域裁剪到先验参数区间之内。
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(
            d_theta[k],
            parameter_box[0] - theta_n_lambda,
            parameter_box[1] - theta_n_lambda,
        )
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    gramian: np.ndarray,
    beta: float,
    sigma: float = 0.1,
) -> bool:
    """
    根据 theta 的置信椭球，检查新观测 (phi, y) 是否有效。

    :param y: 观测值
    :param phi: 特征
    :param theta: 估计参数
    :param gramian: Gram 矩阵
    :param beta: 椭球半径
    :param sigma: 噪声协方差
    :return: 观测是否有效
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    检查数据集 {phi_n, y_n} 是否一致。

    最后一个观测应位于前 N-1 个观测得到的置信椭球之内。

    :param data: 字典 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: 包含参数 theta 的盒状区间 [theta_min, theta_max]
    :return: 数据集是否一致
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(
            train_set, parameter_box=parameter_box
        )
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True


def near_split(x, num_bins=None, size_bins=None):
    """
    将一个数尽量均匀地分配到多个分组中。

    可以指定分组数量或每组大小，所有分组之和始终等于原总数。
    :param x: 要分配的数
    :param num_bins: 分组数量
    :param size_bins: 分组大小
    :return: 各组大小的列表
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def distance_to_circle(center, radius, direction):
    scaling = radius * np.ones((2, 1))
    a = np.linalg.norm(direction / scaling) ** 2
    b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
    c = np.linalg.norm(center / scaling) ** 2 - 1
    root_inf, root_sup = solve_trinom(a, b, c)
    if root_inf and root_inf > 0:
        distance = root_inf
    elif root_sup and root_sup > 0:
        distance = 0
    else:
        distance = np.inf
    return distance


def distance_to_rect(line: tuple[np.ndarray, np.ndarray], rect: list[np.ndarray]):
    """
    计算线段与矩形的交点距离。

    参考 https://math.stackexchange.com/a/2788041。
    :param line: 线段 [R, Q]
    :param rect: 矩形 [A, B, C, D]
    :return: R 到线段 RQ 与矩形 ABCD 交点之间的距离
    """
    r, q = line
    a, b, c, d = rect
    u = b - a
    v = d - a
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    rqu = (q - r) @ u
    rqv = (q - r) @ v
    with np.errstate(divide="ignore", invalid="ignore"):
        interval_1 = [(a - r) @ u / rqu, (b - r) @ u / rqu]
        interval_2 = [(a - r) @ v / rqv, (d - r) @ v / rqv]
    interval_1 = interval_1 if rqu >= 0 else list(reversed(interval_1))
    interval_2 = interval_2 if rqv >= 0 else list(reversed(interval_2))
    if (
        interval_distance(*interval_1, *interval_2) <= 0
        and interval_distance(0, 1, *interval_1) <= 0
        and interval_distance(0, 1, *interval_2) <= 0
    ):
        return max(interval_1[0], interval_2[0]) * np.linalg.norm(q - r)
    else:
        return np.inf


def solve_trinom(a, b, c):
    delta = b**2 - 4 * a * c
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
    else:
        return None, None


_config_path: ContextVar[str] = ContextVar("_config_path", default="config")


@contextmanager
def track_config_path(key: str):
    """跟踪配置路径的上下文管理器，用于提供清楚的错误信息。"""
    token = _config_path.set(f"{_config_path.get()}.{key}")
    try:
        yield
    finally:
        _config_path.reset(token)


def update_config_check(config: dict[str, Any], delta: Mapping[str, Any]) -> None:
    """
    检查 ``delta`` 中的嵌套映射是否重新定义了 ``config`` 中的所有键。

    :param config: 待更新的配置字典
    :param delta: 要应用到 ``config`` 上的新值
    """
    for key, val in config.items():
        if key not in delta or not isinstance(val, Mapping):
            continue
        with track_config_path(key):
            path = _config_path.get()
            new_val = delta[key]
            assert isinstance(
                new_val, Mapping
            ), f"{path} must be a mapping, got {type(new_val).__name__}"

            # 处理多智能体环境，其键没有定义在外层字典中
            if key in ("action", "observation"):
                nested = new_val.get(key + "_config")
                if isinstance(nested, Mapping):
                    new_val = new_val | nested

            missing_keys = val.keys() - new_val.keys()
            assert not missing_keys, f"{path} invalid: {missing_keys=}"
            update_config_check(val, new_val)


def update_config(config: dict[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """
    验证嵌套映射后，用 ``delta`` 原地更新 ``config``。

    :param config: 待更新的配置字典
    :param delta: 要应用到 ``config`` 上的新值
    :return: 更新后的 ``config`` 字典
    """
    update_config_check(config, delta)
    config.update(delta)
    return config
