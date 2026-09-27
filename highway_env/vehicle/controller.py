import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class ControlledVehicle(Vehicle):
    """
    由两个底层控制器驱动的车辆，支持巡航控制、变道等高层动作。

    - 纵向控制器负责速度；
    - 横向控制器由横向位置控制器与航向控制器串联组成。
    """

    target_speed: float
    """期望速度。"""

    """特征时间常数"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # 单位：弧度
    DELTA_SPEED = 5  # [m/s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        根据已有车辆创建新车辆。

        复制车辆的运动状态和目标状态，其他属性采用默认值。

        :param vehicle: 原车辆
        :return: 具有相同运动状态的新车辆
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        在道路网络中规划通往目的地的路线。

        :param destination: 道路网络中的目的节点
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [
                (path[i], path[i + 1], None) for i in range(len(path) - 1)
            ]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        执行高层动作，改变目标车道或目标速度。

        - 收到高层动作时，更新目标速度或车道；
        - 随后进行纵向与横向控制。

        :param action: 高层动作
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def follow_road(self) -> None:
        """到达车道末端时，自动衔接下一条车道。"""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        控制车辆转向，使其沿指定车道的中心行驶。

        1. 横向位置比例控制器生成横向速度指令。
        2. 将横向速度指令转换为参考航向。
        3. 航向比例控制器生成航向角速度指令。
        4. 将航向角速度指令转换为转向角。

        :param target_lane_index: 要跟随的车道索引
        :return: 转向角指令，单位为弧度
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # 横向位置控制
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # 将横向速度转换为航向
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # 航向控制
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # 将航向角速度转换为转向角
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        通过简单的比例控制器控制车速。

        :param target_speed: 期望速度
        :return: 加速度指令，单位为 m/s²
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """获取在下一个交叉路口可以选择的路线列表。"""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        设置车辆在下一个交叉路口要驶入的道路。

        清除当前规划路线。

        :param _to: 道路网络中，下一个路口所选道路的目标节点
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        假设速度不变，预测车辆沿规划路线行驶时的未来位置。

        :param times: 各预测时刻
        :return: 位置与航向
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads, strict=False))


class MDPVehicle(ControlledVehicle):
    """目标速度只能从指定离散档位中选择的受控车辆。"""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
    ) -> None:
        """
        初始化 MDPVehicle。

        :param road: 车辆行驶的道路
        :param position: 车辆位置
        :param heading: 航向角
        :param speed: 当前速度
        :param target_lane_index: 正在跟随的目标车道索引
        :param target_speed: 正在跟踪的目标速度
        :param target_speeds: 通过加速/减速动作可选择的离散目标速度列表
        :param route: 车辆的规划路线，用于处理交叉路口
        """
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        执行高层动作。

        - 若动作为速度调整，则从允许的离散速度档位中选择目标速度；
        - 否则将动作交给 ControlledVehicle 处理。

        :param action: 高层动作
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(
            np.clip(self.speed_index, 0, self.target_speeds.size - 1)
        )
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        将允许速度列表中的索引转换为对应速度。

        :param index: 速度档位索引，无量纲
        :return: 对应速度，单位为 m/s
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        找到与给定速度最接近的可用速度档位索引。

        假设目标速度列表等间隔排列，因此无需逐一搜索最近速度。

        :param speed: 输入速度，单位为 m/s
        :return: 最接近的可用速度档位索引，无量纲
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        找到与给定速度最接近的可用速度档位索引。

        假设目标速度列表等间隔排列，因此无需逐一搜索最近速度。

        :param speed: 输入速度，单位为 m/s
        :return: 最接近的可用速度档位索引，无量纲
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        根据给定动作序列，预测车辆未来轨迹。

        :param actions: 未来动作序列
        :param action_duration: 每个动作的持续时间
        :param trajectory_timestep: 相邻两次保存车辆状态的时间间隔
        :param dt: 仿真时间步长
        :return: 未来状态序列
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # 高层决策
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # 底层控制动作
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
