from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


def rk4(func: Callable, state: np.ndarray, dt: float = 0.01, t: float = 0, **kwargs):
    """
    单步四阶龙格-库塔数值积分（RK4）。
    func：一阶常微分方程组。
    state：当前状态向量 [y1, y2, y3, ...]。
    dt：离散时间步长。
    t：当前时刻。
    **kwargs：传给方程组的其他参数。
    返回：k+1 时刻的状态 y。
    """

    # 在当前时间间隔的多个阶段计算导数。
    f1 = func(t, state, **kwargs)
    f2 = func(t + dt / 2, state + (f1 * (dt / 2)), **kwargs)
    f3 = func(t + dt / 2, state + (f2 * (dt / 2)), **kwargs)
    f4 = func(t + dt, state + (f3 * dt), **kwargs)
    return state + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


class BicycleVehicle(Vehicle):
    """
    包含轮胎摩擦与侧滑的动力学自行车模型。

    参见 Rajamani, R. (2011) 的《Vehicle Dynamics and Control》第 2 章 Lateral Vehicle Dynamics。
    """

    MASS: float = 1  # 单位：千克
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1 / 12 * MASS * (Vehicle.LENGTH**2 + Vehicle.WIDTH**2)  # 单位：千克·平方米
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # 单位：弧度/秒

    def __init__(
        self, road: Road, position: Vector, heading: float = 0, speed: float = 0
    ) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self) -> np.ndarray:
        return np.array(
            [
                [self.position[0]],
                [self.position[1]],
                [self.heading],
                [self.speed],
                [self.lateral_speed],
                [self.yaw_rate],
            ]
        )

    @property
    def derivative(self):
        return self.derivative_func(None, self.state)

    def derivative_func(self, time: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        参见 Rajamani, R. (2011) 的《Vehicle Dynamics and Control》第 2 章 Lateral Vehicle Dynamics。

        :return: 状态导数
        """
        del time
        heading, speed, lateral_speed, yaw_rate = state[2:, 0]
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(lateral_speed + self.LENGTH_A * yaw_rate, speed)  # (2.27)
        theta_vr = np.arctan2(lateral_speed - self.LENGTH_B * yaw_rate, speed)  # (2.28)
        f_yf = 2 * self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2 * self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(speed) < 1:  # 低速动力学：对横向速度和横摆角速度施加阻尼。
            f_yf = (
                -self.MASS * lateral_speed - self.INERTIA_Z / self.LENGTH_A * yaw_rate
            )
            f_yr = (
                -self.MASS * lateral_speed + self.INERTIA_Z / self.LENGTH_A * yaw_rate
            )
        d_lateral_speed = 1 / self.MASS * (f_yf + f_yr) - yaw_rate * speed  # (2.21)
        d_yaw_rate = (
            1 / self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)
        )  # (2.22)
        c, s = np.cos(heading), np.sin(heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([speed, lateral_speed])
        return np.array(
            [
                [speed[0]],
                [speed[1]],
                [yaw_rate],
                [self.action["acceleration"]],
                [d_lateral_speed],
                [d_yaw_rate],
            ]
        )

    @property
    def derivative_linear(self) -> np.ndarray:
        """
        线性化的横向动力学。

        模型基于以下假设：
        - 车辆保持恒定纵向速度；
        - 前轮转向输入及相应侧偏角较小。

        参见下列文献的第 3 章：
        https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf
        """
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action["steering"]]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array(
            [
                [speed[0]],
                [speed[1]],
                [self.yaw_rate],
                [self.action["acceleration"]],
                dx[0],
                dx[1],
            ]
        )

    def step(self, dt: float) -> None:
        self.clip_actions()
        new_state = rk4(self.derivative_func, self.state, dt=dt)
        self.position = new_state[0:2, 0]
        self.heading = new_state[2, 0]
        self.speed = new_state[3, 0]
        self.lateral_speed = new_state[4, 0]
        self.yaw_rate = new_state[5, 0]

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # 线性化要求进行此处理。
        self.action["steering"] = np.clip(
            self.action["steering"], -np.pi / 2, np.pi / 2
        )
        self.yaw_rate = np.clip(
            self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED
        )

    def lateral_lpv_structure(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        状态：[横向速度 v, 横摆角速度 r]。

        :return: 横向动力学 A0、phi、B，使 dx = (A0 + theta^T phi)x + B u
        """
        B = np.array(
            [
                [2 * self.FRICTION_FRONT / self.MASS],
                [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z],
            ]
        )

        speed_body_x = self.speed
        A0 = np.array([[0, -speed_body_x], [0, 0]])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B * 0

        phi = np.array(
            [
                [
                    [
                        -2 / (self.MASS * speed_body_x),
                        -2 * self.LENGTH_A / (self.MASS * speed_body_x),
                    ],
                    [
                        -2 * self.LENGTH_A / (self.INERTIA_Z * speed_body_x),
                        -2 * self.LENGTH_A**2 / (self.INERTIA_Z * speed_body_x),
                    ],
                ],
                [
                    [
                        -2 / (self.MASS * speed_body_x),
                        2 * self.LENGTH_B / (self.MASS * speed_body_x),
                    ],
                    [
                        2 * self.LENGTH_B / (self.INERTIA_Z * speed_body_x),
                        -2 * self.LENGTH_B**2 / (self.INERTIA_Z * speed_body_x),
                    ],
                ],
            ]
        )
        return A0, phi, B

    def lateral_lpv_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        状态：[横向速度 v, 横摆角速度 r]。

        :return: 横向动力学矩阵 A、B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        状态：[位置 y, 横摆角 psi, 横向速度 v, 横摆角速度 r]。

        系统在 psi = 0 附近进行线性化。

        :return: 横向动力学 A、phi、B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([[0, speed_body_x, 1, 0], [0, 0, 0, 1]])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array(
            [
                np.concatenate(
                    (
                        np.zeros((2, 4)),
                        np.concatenate((np.zeros((2, 2)), phi_i), axis=1),
                    )
                )
                for phi_i in phi_lat
            ]
        )
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        状态：[位置 y, 横摆角 psi, 横向速度 v, 横摆角速度 r]。

        系统在 psi = 0 附近进行线性化。

        :return: 横向动力学矩阵 A、B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B


def simulate(dt: float = 0.1) -> None:
    import control

    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], speed=8.3)
    xx, uu = [], []
    from highway_env.interval import LPV

    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(
        x0=vehicle.state[[1, 2, 4, 5]].squeeze(),
        a0=A,
        da=[np.zeros(A.shape)],
        b=B,
        d=[[0], [0], [0], [1]],
        omega_i=[[0], [0]],
        u=None,
        k=K,
        center=None,
        x_i=None,
    )

    for t in time:
        # 执行动作
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2 * np.pi / 20
        u_p = 0 * np.array([[-20 * omega * np.sin(omega * t) * dt]])
        u += u_p
        # 记录数据
        xx.append(
            np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[
                :, np.newaxis
            ]
        )
        uu.append(u)
        # 区间
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        # x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # 推进一步
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time: np.ndarray, xx: np.ndarray, uu: np.ndarray) -> None:
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    _, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1 / 5
    ax.quiver(
        pos_x[::20] - 0.5 / dir_scale * psi_x[::20],
        pos_y[::20] - 0.5 / dir_scale * psi_y[::20],
        psi_x[::20],
        psi_y[::20],
        angles="xy",
        scale_units="xy",
        scale=dir_scale,
        width=0.005,
        headwidth=1,
    )
    ax.quiver(
        pos_x[::20] + 0.5 / dir_scale * psi_x[::20],
        pos_y[::20] + 0.5 / dir_scale * psi_y[::20],
        dir_x[::20],
        dir_y[::20],
        angles="xy",
        scale_units="xy",
        scale=0.25,
        width=0.005,
        color="r",
    )
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()


def main() -> None:
    simulate()


if __name__ == "__main__":
    main()
