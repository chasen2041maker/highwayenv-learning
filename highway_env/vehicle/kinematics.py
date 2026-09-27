from __future__ import annotations

import copy
from collections import deque
from typing import Any

import numpy as np

from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.objects import RoadObject


class Vehicle(RoadObject):
    """
    道路上运动的车辆及其运动学模型。

    使用修正的自行车模型描述车辆动态系统，
    根据转向与加速度动作推进车辆状态。
    """

    LENGTH = 5.0
    """车辆长度，单位为米。"""
    WIDTH = 2.0
    """车辆宽度，单位为米。"""
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """随机初始速度的范围，单位为 m/s。"""
    MAX_SPEED = 40.0
    """可达到的最大速度，单位为 m/s。"""
    MIN_SPEED = -40.0
    """可达到的最小速度，单位为 m/s。"""
    HISTORY_SIZE = 30
    """车辆状态历史记录的长度，用于显示轨迹。"""

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        predition_type: str = "constant_steering",
    ):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {"steering": 0, "acceleration": 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.goal: Any = None

    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_from: str | None = None,
        lane_to: str | None = None,
        lane_id: int | None = None,
        spacing: float = 1,
    ) -> Vehicle:
        """
        在道路上随机创建车辆。

        随机选择车道和/或速度；纵向位置放在道路中最后一辆车的后方，
        车辆分布密度根据车道数量确定。

        :param road: 车辆行驶的道路
        :param speed: 初始速度，单位为 m/s；为 None 时随机选择
        :param lane_from: 生成车道的起始节点
        :param lane_to: 生成车道的结束节点
        :param lane_id: 生成车道的编号
        :param spacing: 与前车间距的倍率，默认值为 1
        :return: 位置和/或速度随机的车辆
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: Vehicle) -> Vehicle:
        """
        根据已有车辆创建新车辆。

        只复制车辆的运动状态，其他属性采用默认值。

        :param vehicle: 原车辆
        :return: 具有相同运动状态的新车辆
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, "color"):
            v.color = vehicle.color
        return v

    def act(self, action: dict | str = None) -> None:
        """
        保存一个要持续执行的动作。

        :param action: 输入动作
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        根据动作推进车辆状态。

        使用修正的自行车模型积分，更新位置、朝向和速度。
        碰撞后，clip_actions 将转向设为 0，并制动至停车。
        同时更新车辆当前所在车道。

        :param dt: 模型积分的时间步长，单位为秒
        """
        self.clip_actions()
        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed
        self.action["steering"] = float(self.action["steering"])
        self.action["acceleration"] = float(self.action["acceleration"])
        if self.speed > self.MAX_SPEED:
            self.action["acceleration"] = min(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )
        elif self.speed < self.MIN_SPEED:
            self.action["acceleration"] = max(
                self.action["acceleration"], 1.0 * (self.MIN_SPEED - self.speed)
            )

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position, self.heading
            )
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        if self.prediction_type == "zero_steering":
            action = {"acceleration": 0.0, "steering": 0.0}
        elif self.prediction_type == "constant_steering":
            action = {"acceleration": 0.0, "steering": self.action["steering"]}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO：这里应使用侧滑角 beta。

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = (
                last_lane_index
                if last_lane_index[-1] is not None
                else (*last_lane_index[:-1], 0)
            )
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(
                self.destination - self.position
            )
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    def to_dict(
        self, origin_vehicle: Vehicle = None, observe_intentions: bool = True
    ) -> dict:
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "heading": self.heading,
            "cos_h": self.direction[0],
            "sin_h": self.direction[1],
            "cos_d": self.destination_direction[0],
            "sin_d": self.destination_direction[1],
            "long_off": self.lane_offset[0],
            "lat_off": self.lane_offset[1],
            "ang_off": self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(
            self.__class__.__name__, id(self) % 1000, self.position
        )

    def __repr__(self):
        return self.__str__()

    def predict_trajectory(
        self,
        actions: list,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> list[Vehicle]:
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
            v.act(action)  # 底层控制动作
            for _ in range(int(action_duration / dt)):
                t += 1
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
