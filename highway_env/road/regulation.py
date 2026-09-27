from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class RegulatedRoad(Road):
    YIELDING_COLOR: tuple[float, float, float] = None
    REGULATION_FREQUENCY: int = 2
    YIELD_DURATION: float = 0.0

    def __init__(
        self,
        network: RoadNetwork = None,
        vehicles: list[Vehicle] = None,
        obstacles: list[Obstacle] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
        neighbour_vehicles_connected_lanes: bool = False,
    ) -> None:
        super().__init__(
            network,
            vehicles,
            obstacles,
            np_random,
            record_history,
            neighbour_vehicles_connected_lanes,
        )
        self.steps = 0

    def step(self, dt: float) -> None:
        self.steps += 1
        if self.steps % int(1 / dt / self.REGULATION_FREQUENCY) == 0:
            self.enforce_road_rules()
        return super().step(dt)

    def enforce_road_rules(self) -> None:
        """查找冲突，指定需要让行的车辆并让其停车，以解决冲突。"""

        # 解除此前让行车辆的停车状态
        for v in self.vehicles:
            if getattr(v, "is_yielding", False):
                if v.yield_timer >= self.YIELD_DURATION * self.REGULATION_FREQUENCY:
                    v.target_speed = v.lane.speed_limit
                    delattr(v, "color")
                    v.is_yielding = False
                else:
                    v.yield_timer += 1

        # 查找并解决新的冲突
        for i in range(len(self.vehicles) - 1):
            for j in range(i + 1, len(self.vehicles)):
                if self.is_conflict_possible(self.vehicles[i], self.vehicles[j]):
                    yielding_vehicle = self.respect_priorities(
                        self.vehicles[i], self.vehicles[j]
                    )
                    if (
                        yielding_vehicle is not None
                        and isinstance(yielding_vehicle, ControlledVehicle)
                        and not isinstance(yielding_vehicle, MDPVehicle)
                    ):
                        yielding_vehicle.color = self.YIELDING_COLOR
                        yielding_vehicle.target_speed = 0
                        yielding_vehicle.is_yielding = True
                        yielding_vehicle.yield_timer = 0

    @staticmethod
    def respect_priorities(v1: Vehicle, v2: Vehicle) -> Vehicle:
        """
        确定哪辆车应当让行，从而解决两车之间的冲突。

        :param v1: 第一辆车
        :param v2: 第二辆车
        :return: 需要让行的车辆
        """
        if v1.lane.priority > v2.lane.priority:
            return v2
        elif v1.lane.priority < v2.lane.priority:
            return v1
        else:  # 后方车辆应当让行
            return v1 if v1.front_distance_to(v2) > v2.front_distance_to(v1) else v2

    @staticmethod
    def is_conflict_possible(
        v1: ControlledVehicle,
        v2: ControlledVehicle,
        horizon: int = 3,
        step: float = 0.25,
    ) -> bool:
        times = np.arange(step, horizon, step)
        positions_1, headings_1 = v1.predict_trajectory_constant_speed(times)
        positions_2, headings_2 = v2.predict_trajectory_constant_speed(times)

        for position_1, heading_1, position_2, heading_2 in zip(
            positions_1, headings_1, positions_2, headings_2, strict=False
        ):
            # 快速球形预检查
            if np.linalg.norm(position_2 - position_1) > v1.LENGTH:
                continue

            # 精确矩形检查
            if utils.rotated_rectangles_intersect(
                (position_1, 1.5 * v1.LENGTH, 0.9 * v1.WIDTH, heading_1),
                (position_2, 1.5 * v2.LENGTH, 0.9 * v2.WIDTH, heading_2),
            ):
                return True
