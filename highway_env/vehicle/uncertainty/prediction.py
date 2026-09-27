from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

from highway_env import utils
from highway_env.interval import (
    LPV,
    integrator_interval,
    interval_absolute_to_local,
    interval_local_to_absolute,
    interval_negative_part,
    intervals_diff,
    intervals_product,
    polytope,
    vector_interval_section,
)
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.vehicle.objects import RoadObject

Polytope = Tuple[np.ndarray, List[np.ndarray]]


class IntervalVehicle(LinearVehicle):
    """
    在参数不确定时，估计 LinearVehicle 状态所属的区间。

    模型轨迹保存在 model_vehicle 中，状态下界和上界分别保存在
    min_vehicle 和 max_vehicle 中。这些车辆对象不遵循正常的车辆动力学，
    只用于存储边界。
    """

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        theta_a_i: list[list[float]] = None,
        theta_b_i: list[list[float]] = None,
        data: dict = None,
    ) -> None:
        """
        :param theta_a_i: 加速度参数的可能取值区间
        :param theta_b_i: 转向参数的可能取值区间
        """
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )
        self.theta_a_i = (
            np.asarray(theta_a_i)
            if theta_a_i is not None
            else LinearVehicle.ACCELERATION_RANGE
        )
        self.theta_b_i = (
            np.asarray(theta_b_i)
            if theta_b_i is not None
            else LinearVehicle.STEERING_RANGE
        )
        self.data = data
        self.interval = VehicleInterval(self)
        self.trajectory = []
        self.interval_trajectory = []
        self.longitudinal_lpv, self.lateral_lpv = None, None
        self.previous_target_lane_index = self.target_lane_index

    @classmethod
    def create_from(cls, vehicle: Vehicle) -> IntervalVehicle:
        return cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=getattr(vehicle, "target_lane_index", None),
            target_speed=getattr(vehicle, "target_speed", None),
            route=getattr(vehicle, "route", None),
            timer=getattr(vehicle, "timer", None),
            theta_a_i=getattr(vehicle, "theta_a_i", None),
            theta_b_i=getattr(vehicle, "theta_b_i", None),
            data=getattr(vehicle, "data", None),
        )

    def step(self, dt: float, mode: str = "partial") -> None:
        self.store_trajectories()
        if self.crashed:
            self.interval = VehicleInterval(self)
        else:
            if mode == "partial":
                # self.observer_step(dt)
                self.partial_observer_step(dt)
            elif mode == "predictor":
                self.predictor_step(dt)
        super().step(dt)

    def observer_step(self, dt: float) -> None:
        """
        推进一步区间观测器的动力学。

        :param dt: 时间步长，单位为秒
        """
        # 输入状态区间
        position_i = self.interval.position
        v_i = self.interval.speed
        psi_i = self.interval.heading

        # 特征区间
        front_interval = self.get_front_interval()

        # 加速度特征
        phi_a_i = np.zeros((2, 3))
        phi_a_i[:, 0] = [0, 0]
        if front_interval:
            phi_a_i[:, 1] = interval_negative_part(
                intervals_diff(front_interval.speed, v_i)
            )
            # 车道距离区间
            lane_psi = self.lane.heading_at(
                self.lane.local_coordinates(self.position)[0]
            )
            lane_direction = [np.cos(lane_psi), np.sin(lane_psi)]
            diff_i = intervals_diff(front_interval.position, position_i)
            d_i = vector_interval_section(diff_i, lane_direction)

            d_safe_i = self.DISTANCE_WANTED + self.TIME_WANTED * v_i
            phi_a_i[:, 2] = interval_negative_part(intervals_diff(d_i, d_safe_i))

        # 转向特征
        phi_b_i = None
        lanes = self.get_followed_lanes()
        for lane_index in lanes:
            lane = self.road.network.get_lane(lane_index)
            longitudinal_pursuit = (
                lane.local_coordinates(self.position)[0] + self.speed * self.TAU_PURSUIT
            )
            lane_psi = lane.heading_at(longitudinal_pursuit)
            _, lateral_i = interval_absolute_to_local(position_i, lane)
            lateral_i = -np.flip(lateral_i)
            i_v_i = 1 / np.flip(v_i, 0)
            phi_b_i_lane = np.transpose(
                np.array([[0, 0], intervals_product(lateral_i, i_v_i)])
            )
            # 合并候选特征区间
            if phi_b_i is None:
                phi_b_i = phi_b_i_lane
            else:
                phi_b_i[0] = np.minimum(phi_b_i[0], phi_b_i_lane[0])
                phi_b_i[1] = np.maximum(phi_b_i[1], phi_b_i_lane[1])

        # 控制指令区间
        a_i = intervals_product(self.theta_a_i, phi_a_i)
        b_i = intervals_product(self.theta_b_i, phi_b_i)

        # 速度区间
        keep_stability = False
        if keep_stability:
            dv_i = integrator_interval(v_i - self.target_speed, self.theta_a_i[:, 0])
        else:
            dv_i = intervals_product(
                self.theta_a_i[:, 0], self.target_speed - np.flip(v_i, 0)
            )
        dv_i += a_i
        dv_i = np.clip(dv_i, -self.ACC_MAX, self.ACC_MAX)
        keep_stability = True
        if keep_stability:
            delta_psi = list(map(utils.wrap_to_pi, psi_i - lane_psi))
            d_psi_i = integrator_interval(delta_psi, self.theta_b_i[:, 0])
        else:
            d_psi_i = intervals_product(
                self.theta_b_i[:, 0], lane_psi - np.flip(psi_i, 0)
            )
        d_psi_i += b_i

        # 位置区间
        cos_i = [
            -1 if psi_i[0] <= np.pi <= psi_i[1] else min(map(np.cos, psi_i)),
            1 if psi_i[0] <= 0 <= psi_i[1] else max(map(np.cos, psi_i)),
        ]
        sin_i = [
            -1 if psi_i[0] <= -np.pi / 2 <= psi_i[1] else min(map(np.sin, psi_i)),
            1 if psi_i[0] <= np.pi / 2 <= psi_i[1] else max(map(np.sin, psi_i)),
        ]
        dx_i = intervals_product(v_i, cos_i)
        dy_i = intervals_product(v_i, sin_i)

        # 对区间动力学进行积分
        self.interval.speed += dv_i * dt
        self.interval.heading += d_psi_i * dt
        self.interval.position[:, 0] += dx_i * dt
        self.interval.position[:, 1] += dy_i * dt

        # 加入噪声
        noise = 0.3
        self.interval.position[:, 0] += noise * dt * np.array([-1, 1])
        self.interval.position[:, 1] += noise * dt * np.array([-1, 1])
        self.interval.heading += noise * dt * np.array([-1, 1])

    def predictor_step(self, dt: float) -> None:
        """
        推进一步区间预测器的动力学。

        :param dt: 时间步长，单位为秒
        """
        # 创建纵向和横向 LPV 模型
        self.predictor_init()
        self.lateral_lpv: LPV
        self.longitudinal_lpv: LPV

        # 检测变道，并在新坐标系中更新局部坐标区间
        if self.target_lane_index != self.previous_target_lane_index:
            position_i = self.interval.position
            target_lane = self.road.network.get_lane(self.target_lane_index)
            previous_target_lane = self.road.network.get_lane(
                self.previous_target_lane_index
            )
            longi_i, lat_i = interval_absolute_to_local(position_i, target_lane)
            psi_i = (
                self.interval.heading
                + target_lane.heading_at(longi_i.mean())
                - previous_target_lane.heading_at(longi_i.mean())
            )
            x_i_local_unrotated = np.transpose([lat_i, psi_i])
            new_x_i_t = self.lateral_lpv.change_coordinates(
                x_i_local_unrotated, back=False, interval=True
            )
            delta = new_x_i_t.mean(axis=0) - self.lateral_lpv.x_i_t.mean(axis=0)
            self.lateral_lpv.x_i_t += delta
            x_i_local_unrotated = self.longitudinal_lpv.change_coordinates(
                self.longitudinal_lpv.x_i_t, back=True, interval=True
            )
            x_i_local_unrotated[:, 0] = longi_i
            new_x_i_t = self.longitudinal_lpv.change_coordinates(
                x_i_local_unrotated, back=False, interval=True
            )
            self.longitudinal_lpv.x_i_t += new_x_i_t.mean(
                axis=0
            ) - self.longitudinal_lpv.x_i_t.mean(axis=0)
            self.previous_target_lane_index = self.target_lane_index

        # 推进一步
        self.longitudinal_lpv.step(dt)
        self.lateral_lpv.step(dt)

        # 反向坐标变换
        x_i_long = self.longitudinal_lpv.change_coordinates(
            self.longitudinal_lpv.x_i_t, back=True, interval=True
        )
        x_i_lat = self.lateral_lpv.change_coordinates(
            self.lateral_lpv.x_i_t, back=True, interval=True
        )

        # 从校正坐标转换回真实坐标
        target_lane = self.road.network.get_lane(self.target_lane_index)
        position_i = interval_local_to_absolute(
            x_i_long[:, 0], x_i_lat[:, 0], target_lane
        )
        self.interval.position = position_i
        self.interval.speed = x_i_long[:, 2]
        self.interval.heading = x_i_lat[:, 1]

    def predictor_init(self) -> None:
        """初始化区间预测使用的 LPV 模型。"""
        position_i = self.interval.position
        target_lane = self.road.network.get_lane(self.target_lane_index)
        longi_i, lat_i = interval_absolute_to_local(position_i, target_lane)
        v_i = self.interval.speed
        psi_i = self.interval.heading - self.lane.heading_at(longi_i.mean())

        # 纵向预测器
        if not self.longitudinal_lpv:
            front_interval = self.get_front_interval()

            # LPV 模型定义
            if front_interval:
                f_longi_i, _ = interval_absolute_to_local(
                    front_interval.position, target_lane
                )
                f_pos = f_longi_i[0]
                f_vel = front_interval.speed[0]
            else:
                f_pos, f_vel = 0, 0
            x0 = [longi_i[0], f_pos, v_i[0], f_vel]
            center = [
                -self.DISTANCE_WANTED - self.target_speed * self.TIME_WANTED,
                0,
                self.target_speed,
                self.target_speed,
            ]
            noise = 1
            b = np.eye(4)
            d = np.array([[1], [0], [0], [0]])
            omega_i = np.array([[-1], [1]]) * noise
            u = [[self.target_speed], [self.target_speed], [0], [0]]
            a0, da = self.longitudinal_matrix_polytope()
            self.longitudinal_lpv = LPV(x0, a0, da, b, d, omega_i, u, center=center)

            # 横向预测器
            if not self.lateral_lpv:
                # LPV 模型定义
                x0 = [lat_i[0], psi_i[0]]
                center = [0, 0]
                noise = 0.5
                b = np.identity(2)
                d = np.array([[1], [0]])
                omega_i = np.array([[-1], [1]]) * noise
                u = [[0], [0]]
                a0, da = self.lateral_matrix_polytope()
                self.lateral_lpv = LPV(x0, a0, da, b, d, omega_i, u, center=center)

    def longitudinal_matrix_polytope(self) -> Polytope:
        return IntervalVehicle.parameter_box_to_polytope(
            self.theta_a_i, self.longitudinal_structure
        )

    def lateral_matrix_polytope(self) -> Polytope:
        return IntervalVehicle.parameter_box_to_polytope(
            self.theta_b_i, self.lateral_structure
        )

    @staticmethod
    def parameter_box_to_polytope(
        parameter_box: np.ndarray, structure: Callable
    ) -> Polytope:
        a, phi = structure()
        a_theta = lambda params: a + np.tensordot(phi, params, axes=[0, 0])
        return polytope(a_theta, parameter_box)

    def get_front_interval(self) -> VehicleInterval | None:
        # TODO：目前假设前车跟随模型中的前车。
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if front_vehicle:
            if isinstance(front_vehicle, IntervalVehicle):
                # 使用前车观测器估计得到的区间
                front_interval = front_vehicle.interval
            else:
                # 这里没有估计前车轨迹区间，因此
                # 将它视为确定状态。根据前车的当前状态
                # 创建新的观测器，其初始状态完全确定。
                front_interval = IntervalVehicle.create_from(front_vehicle).interval
        else:
            front_interval = None
        return front_interval

    def get_followed_lanes(
        self, lane_change_model: str = "model", squeeze: bool = True
    ) -> list[LaneIndex]:
        """
        获取该车辆可能跟随的车道列表。

        :param lane_change_model:
            - model：假设车辆跟随其行为模型选择的车道。
            - all：假设任意时刻都可能作出任意变道决策。
            - right：假设任意时刻都可能决定向右变道。
        :param squeeze: 若为 True，则移除道路边界处出现的重复车道
        :return: 可能跟随的车道索引列表
        """
        self.target_lane_index: LaneIndex
        lanes = []
        if lane_change_model == "model":
            lanes = [self.target_lane_index]
        elif lane_change_model == "all":
            lanes = self.road.network.side_lanes(self.target_lane_index) + [
                self.target_lane_index
            ]
        elif lane_change_model == "right":
            lanes = [self.target_lane_index]
            _from, _to, _id = self.target_lane_index
            if _id < len(
                self.road.network.graph[_from][_to]
            ) - 1 and self.road.network.get_lane(
                (_from, _to, _id + 1)
            ).is_reachable_from(
                self.position
            ):
                lanes.append((_from, _to, _id + 1))
            elif not squeeze:
                lanes.append(self.target_lane_index)  # 右侧车道同时也是当前车道
        return lanes

    def partial_observer_step(self, dt: float, alpha: float = 0) -> None:
        """
        推进当前状态区间的边界部分。

        1. 将 x_i(t) 拆分为下边界区间 x_i_-(t) 和上边界区间 x_i_+(t)。
        2. 推进各自的观测器动力学，得到 x_i_-(t+dt) 和 x_i_+(t+dt)。
        3. 将结果合并为 x_i(t+dt)。

        :param dt: 时间步长，单位为秒
        :param alpha: 边界区间占完整区间的比例
        """
        # 1. 将 x_i(t) 拆分为下边界区间 x_i_-(t) 和上边界区间 x_i_+(t)
        o = self.interval
        v_minus = IntervalVehicle.create_from(self)
        v_minus.interval = copy.deepcopy(self.interval)
        v_minus.interval.position[1, :] = (1 - alpha) * o.position[
            0, :
        ] + alpha * o.position[1, :]
        v_minus.interval.speed[1] = (1 - alpha) * o.speed[0] + alpha * o.speed[1]
        v_minus.interval.heading[1] = (1 - alpha) * o.heading[0] + alpha * o.heading[1]
        v_plus = IntervalVehicle.create_from(self)
        v_plus.interval = copy.deepcopy(self.interval)
        v_plus.interval.position[0, :] = (
            alpha * o.position[0, :] + (1 - alpha) * o.position[1, :]
        )
        v_plus.interval.speed[0] = alpha * o.speed[0] + (1 - alpha) * o.speed[1]
        v_plus.interval.heading[0] = alpha * o.heading[0] + (1 - alpha) * o.heading[1]
        # 2. 推进各自的观测器动力学，得到 x_i_-(t+dt) 和 x_i_+(t+dt)
        v_minus.road = copy.copy(v_minus.road)
        v_minus.road.vehicles = [
            v if v is not self else v_minus for v in v_minus.road.vehicles
        ]
        v_plus.road = copy.copy(v_plus.road)
        v_plus.road.vehicles = [
            v if v is not self else v_plus for v in v_plus.road.vehicles
        ]
        v_minus.observer_step(dt)
        v_plus.observer_step(dt)
        # 3. 将结果合并为 x_i(t+dt)
        self.interval.position = np.array(
            [v_minus.interval.position[0], v_plus.interval.position[1]]
        )
        self.interval.speed = np.array(
            [v_minus.interval.speed[0], v_plus.interval.speed[1]]
        )
        self.interval.heading = np.array(
            [
                min(v_minus.interval.heading[0], v_plus.interval.heading[0]),
                max(v_minus.interval.heading[1], v_plus.interval.heading[1]),
            ]
        )

    def store_trajectories(self) -> None:
        """将当前模型状态、状态下界和上界存入轨迹列表。"""
        self.trajectory.append(LinearVehicle.create_from(self))
        self.interval_trajectory.append(copy.deepcopy(self.interval))

    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        """
        检查最坏情况下的碰撞。

        为进行鲁棒规划，假设 MDPVehicle 与 IntervalVehicle 的不确定状态集合
        发生碰撞时就算碰撞，这对应最坏情况。

        :param other: 另一辆车
        :param dt: 时间步长
        """
        if not isinstance(other, MDPVehicle):
            super().handle_collisions(other)
            return

        if not self.collidable or self.crashed or other is self:
            return

        # 快速矩形预检查
        if not utils.point_in_rectangle(
            other.position,
            self.interval.position[0] - self.LENGTH,
            self.interval.position[1] + self.LENGTH,
        ):
            return

        # 将另一辆车投影到不确定性矩形上，得到本车所有可能位置中
        # 最容易与对方发生碰撞的位置
        projection = np.minimum(
            np.maximum(other.position, self.interval.position[0]),
            self.interval.position[1],
        )
        # 精确矩形检查
        if utils.rotated_rectangles_intersect(
            (projection, self.LENGTH, self.WIDTH, self.heading),
            (other.position, 0.9 * other.LENGTH, 0.9 * other.WIDTH, other.heading),
        ):
            self.speed = other.speed = min(self.speed, other.speed)
            self.crashed = other.crashed = True


class VehicleInterval:
    def __init__(self, vehicle: Vehicle) -> None:
        self.position = np.array([vehicle.position, vehicle.position], dtype=float)
        self.speed = np.array([vehicle.speed, vehicle.speed], dtype=float)
        self.heading = np.array([vehicle.heading, vehicle.heading], dtype=float)
