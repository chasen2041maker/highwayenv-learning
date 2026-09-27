from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
from numpy.linalg import LinAlgError

from highway_env.road.lane import AbstractLane
from highway_env.utils import ColumnVector, Interval, Matrix, Vector


def intervals_product(a: Interval, b: Interval) -> np.ndarray:
    """
    计算两个区间的乘积。

    :param a: 区间 [a_min, a_max]
    :param b: 区间 [b_min, b_max]
    :return: 乘积 ab 所在的区间
    """
    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [
            np.dot(p(a[0]), p(b[0]))
            - np.dot(p(a[1]), n(b[0]))
            - np.dot(n(a[0]), p(b[1]))
            + np.dot(n(a[1]), n(b[1])),
            np.dot(p(a[1]), p(b[1]))
            - np.dot(p(a[0]), n(b[1]))
            - np.dot(n(a[1]), p(b[0]))
            + np.dot(n(a[0]), n(b[0])),
        ]
    )


def intervals_scaling(a: Interval, b: Interval) -> np.ndarray:
    """
    对区间进行缩放。

    :param a: 矩阵 a
    :param b: 区间 [b_min, b_max]
    :return: 乘积 ab 所在的区间
    """
    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [
            np.dot(p(a), b[0]) - np.dot(n(a), b[1]),
            np.dot(p(a), b[1]) - np.dot(n(a), b[0]),
        ]
    )


def intervals_diff(a: Interval, b: Interval) -> np.ndarray:
    """
    计算两个区间的差。

    :param a: 区间 [a_min, a_max]
    :param b: 区间 [b_min, b_max]
    :return: 差 a - b 所在的区间
    """
    return np.array([a[0] - b[1], a[1] - b[0]])


def interval_negative_part(a: Interval) -> np.ndarray:
    """
    计算区间的负值部分。

    :param a: 区间 [a_min, a_max]
    :return: min(a, 0) 所在的区间
    """
    return np.minimum(a, 0)


def integrator_interval(x: Interval, k: Interval) -> np.ndarray:
    """
    计算积分系统 dx = -k*x 的导数区间。

    :param x: 状态区间
    :param k: 增益区间，必须为正
    :return: dx 所在的区间
    """

    if x[0] >= 0:
        interval_gain = np.flip(-k, 0)
    elif x[1] <= 0:
        interval_gain = -k
    else:
        interval_gain = -np.array([k[0], k[0]])
    return (
        interval_gain * x
    )  # 注意：这里不翻转 x，与使用 intervals_product(k,interval_minus(x)) 不同。


def vector_interval_section(v_i: Interval, direction: Vector) -> np.ndarray:
    corners = [
        [v_i[0, 0], v_i[0, 1]],
        [v_i[0, 0], v_i[1, 1]],
        [v_i[1, 0], v_i[0, 1]],
        [v_i[1, 0], v_i[1, 1]],
    ]
    corners_dist = [np.dot(corner, direction) for corner in corners]
    return np.array([min(corners_dist), max(corners_dist)])


def interval_absolute_to_local(
    position_i: Interval, lane: AbstractLane
) -> tuple[np.ndarray, np.ndarray]:
    """
    将绝对坐标 x、y 的区间转换为局部纵向、横向坐标的区间。

    :param position_i: 位置区间 [x_min, x_max]
    :param lane: 提供局部坐标系的车道
    :return: 对应的局部坐标区间
    """
    position_corners = np.array(
        [
            [position_i[0, 0], position_i[0, 1]],
            [position_i[0, 0], position_i[1, 1]],
            [position_i[1, 0], position_i[0, 1]],
            [position_i[1, 0], position_i[1, 1]],
        ]
    )
    corners_local = np.array([lane.local_coordinates(c) for c in position_corners])
    longitudinal_i = np.array([min(corners_local[:, 0]), max(corners_local[:, 0])])
    lateral_i = np.array([min(corners_local[:, 1]), max(corners_local[:, 1])])
    return longitudinal_i, lateral_i


def interval_local_to_absolute(
    longitudinal_i: Interval, lateral_i: Interval, lane: AbstractLane
) -> Interval:
    """
    将局部纵向、横向坐标的区间转换为绝对坐标 x、y 的区间。

    :param longitudinal_i: 纵向区间 [L_min, L_max]
    :param lateral_i: 横向区间 [l_min, l_max]
    :param lane: 提供局部坐标系的车道
    :return: 对应的绝对坐标区间
    """
    corners_local = [
        [longitudinal_i[0], lateral_i[0]],
        [longitudinal_i[0], lateral_i[1]],
        [longitudinal_i[1], lateral_i[0]],
        [longitudinal_i[1], lateral_i[1]],
    ]
    corners_absolute = np.array([lane.position(*c) for c in corners_local])
    position_i = np.array(
        [np.amin(corners_absolute, axis=0), np.amax(corners_absolute, axis=0)]
    )
    return position_i


def polytope(
    parametrized_f: Callable[[np.ndarray], np.ndarray], params_intervals: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    根据参数化矩阵函数和参数的盒状区间构造矩阵多面体。

    :param parametrized_f: 参数化矩阵函数
    :param params_intervals: 各轴分别为 [min, max] 和参数
    :return: 表示矩阵区间的多面体 a0, d_a
    """
    params_means = params_intervals.mean(axis=0)
    a0 = parametrized_f(params_means)
    vertices_id = itertools.product([0, 1], repeat=params_intervals.shape[1])
    d_a = []
    for vertex_id in vertices_id:
        params_vertex = params_intervals[vertex_id, np.arange(len(vertex_id))]
        d_a.append(parametrized_f(params_vertex) - parametrized_f(params_means))
    d_a = list({str(d_a_i): d_a_i for d_a_i in d_a}.values())
    return a0, d_a


def is_metzler(matrix: np.ndarray, eps: float = 1e-9) -> bool:
    return (matrix - np.diag(np.diag(matrix)) >= -eps).all()


class LPV:
    def __init__(
        self,
        x0: Vector,
        a0: Matrix,
        da: Sequence[Vector],
        b: Matrix = None,
        d: ColumnVector = None,
        omega_i: Matrix = None,
        u: ColumnVector = None,
        k: Matrix = None,
        center: Vector = None,
        x_i: Matrix = None,
    ) -> None:
        """
        线性变参数系统（LPV）：

            dx = (a0 + sum(da))(x - center) + bd + c

        :param x0: 初始状态
        :param a0: 标称动力学矩阵
        :param da: 动力学偏差矩阵列表
        :param b: 控制矩阵
        :param d: 扰动矩阵
        :param omega_i: 扰动上下界
        :param u: 已知的恒定控制输入
        :param k: 线性反馈：a0 x + bu -> (a0+bk)x + b(u-kx)，其中 a0+bk 稳定
        :param center: 渐近状态
        :param x_i: 初始状态区间
        """
        self.x0 = np.array(x0, dtype=float)
        self.a0 = np.array(a0, dtype=float)
        self.da = [np.array(da_i) for da_i in da]
        self.b = np.array(b) if b is not None else np.zeros((*self.x0.shape, 1))
        self.d = np.array(d) if d is not None else np.zeros((*self.x0.shape, 1))
        self.omega_i = np.array(omega_i) if omega_i is not None else np.zeros((2, 1))
        self.u = np.array(u) if u is not None else np.zeros((1,))
        self.k = (
            np.array(k)
            if k is not None
            else np.zeros((self.b.shape[1], self.b.shape[0]))
        )
        self.center = (
            np.array(center) if center is not None else np.zeros(self.x0.shape)
        )

        # 闭环动力学
        self.a0 += self.b @ self.k

        self.coordinates = None

        self.x_t = self.x0
        self.x_i = np.array(x_i) if x_i is not None else np.array([self.x0, self.x0])
        self.x_i_t = None

        self.update_coordinates_frame(self.a0)

    def update_coordinates_frame(self, a0: np.ndarray) -> None:
        """
        确保动力学矩阵 A0 是 Metzler 矩阵。

        否则，构造坐标变换，并将其应用于模型和状态区间。
        :param a0: 动力学矩阵 A0
        """
        self.coordinates = None
        # 旋转
        if not is_metzler(a0):
            eig_v, transformation = np.linalg.eig(a0)
            if np.isreal(eig_v).all():
                try:
                    self.coordinates = (transformation, np.linalg.inv(transformation))
                except LinAlgError:
                    pass
            if not self.coordinates:
                print("Non Metzler A0 with eigenvalues: ", eig_v)
        else:
            self.coordinates = (np.eye(a0.shape[0]), np.eye(a0.shape[0]))

        # 对状态和模型进行正向坐标变换
        self.a0 = self.change_coordinates(self.a0, matrix=True)
        self.da = self.change_coordinates(self.da, matrix=True)
        self.b = self.change_coordinates(self.b, offset=False)
        self.x_i_t = np.array(self.change_coordinates([x for x in self.x_i]))

    def set_control(self, control: np.ndarray, state: np.ndarray = None) -> None:
        if state is not None:
            control = (
                control - self.k @ state
            )  # 控制中的 Kx 部分已经包含在 A0 中。
        self.u = control

    def change_coordinates(
        self,
        value: np.ndarray | list[np.ndarray],
        matrix: bool = False,
        back: bool = False,
        interval: bool = False,
        offset: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """
        进行坐标变换：旋转和平移到中心。

        :param value: 待变换的对象
        :param matrix: 对象是矩阵还是向量
        :param back: 若为 True，则变换回原坐标系
        :param interval: 变换区间时需使用会损失精度的区间运算，以保持包含关系
        :param offset: 是否应用中心平移
        :return: 变换后的对象
        """
        if self.coordinates is None:
            return value
        transformation, transformation_inv = self.coordinates
        if interval:
            if back:
                value = intervals_scaling(
                    transformation, value[:, :, np.newaxis]
                ).squeeze() + offset * np.array([self.center, self.center])
                return value
            else:
                value = value - offset * np.array([self.center, self.center])
                value = intervals_scaling(
                    transformation_inv, value[:, :, np.newaxis]
                ).squeeze()
                return value
        elif matrix:  # 矩阵
            if back:
                return transformation @ value @ transformation_inv
            else:
                return transformation_inv @ value @ transformation
        elif isinstance(value, list):  # 列表
            return [self.change_coordinates(v, back) for v in value]
        else:
            if back:
                value = transformation @ value
                if offset:
                    value += self.center
                return value
            else:
                if offset:
                    value -= self.center
                return transformation_inv @ value

    def step(self, dt: float) -> None:
        if is_metzler(self.a0):
            self.x_i_t = self.step_interval_predictor(self.x_i_t, dt)
        else:
            self.x_i_t = self.step_naive_predictor(self.x_i_t, dt)
        dx = self.a0 @ self.x_t + self.b @ self.u.squeeze(-1)
        self.x_t = self.x_t + dx * dt

    def step_naive_predictor(self, x_i: Interval, dt: float) -> np.ndarray:
        """
        对具有盒状不确定性的区间预测器推进一步。

        :param x_i: t 时刻的状态区间
        :param dt: 时间步长
        :return: t+dt 时刻的状态区间
        """
        a0, da, d, omega_i, b, u = (
            self.a0,
            self.da,
            self.d,
            self.omega_i,
            self.b,
            self.u,
        )
        a_i = a0 + sum(intervals_product([0, 1], [da_i, da_i]) for da_i in da)
        bu = (b @ u).squeeze(-1)
        dx_i = (
            intervals_product(a_i, x_i)
            + intervals_product([d, d], omega_i)
            + np.array([bu, bu])
        )
        return x_i + dx_i * dt

    def step_interval_predictor(self, x_i: Interval, dt: float) -> np.ndarray:
        """
        对具有多面体不确定性的区间预测器推进一步。

        :param x_i: t 时刻的状态区间
        :param dt: 时间步长
        :return: t+dt 时刻的状态区间
        """
        a0, da, d, omega_i, b, u = (
            self.a0,
            self.da,
            self.d,
            self.omega_i,
            self.b,
            self.u,
        )
        p = lambda x: np.maximum(x, 0)
        n = lambda x: np.maximum(-x, 0)
        da_p = sum(p(da_i) for da_i in da)
        da_n = sum(n(da_i) for da_i in da)
        x_m, x_M = x_i[0, :, np.newaxis], x_i[1, :, np.newaxis]
        o_m, o_M = omega_i[0, :, np.newaxis], omega_i[1, :, np.newaxis]
        dx_m = (
            a0 @ x_m - da_p @ n(x_m) - da_n @ p(x_M) + p(d) @ o_m - n(d) @ o_M + b @ u
        )
        dx_M = (
            a0 @ x_M + da_p @ p(x_M) + da_n @ n(x_m) + p(d) @ o_M - n(d) @ o_m + b @ u
        )
        dx_i = np.array([dx_m.squeeze(axis=-1), dx_M.squeeze(axis=-1)])
        return x_i + dx_i * dt
