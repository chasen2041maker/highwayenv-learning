from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


class TwoWayEnv(AbstractEnv):
    """
    风险管理任务：智能体在有对向来车的双向道路上驾驶。

    它必须在超车前进与保证安全之间作出权衡。

    在 CMDP/BMDP 框架中，使用奖励信号和约束信号描述这两个相互冲突的目标。
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {"type": "TimeToCollision", "horizon": 5},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "collision_reward": 0,
                "left_lane_constraint": 1,
                "left_lane_reward": 0.2,
                "high_speed_reward": 0.8,
            },
        )
        return config

    def _reward(self, action: int) -> float:
        """
        车辆高速行驶时获得奖励。
        :param action: 执行的动作
        :return: 此次状态和动作转移对应的奖励
        """
        return sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )

    def _rewards(self, action: int) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        return {
            "high_speed_reward": self.vehicle.speed_index
            / (self.vehicle.target_speeds.size - 1),
            "left_lane_reward": (
                len(neighbours) - 1 - self.vehicle.target_lane_index[2]
            )
            / (len(neighbours) - 1),
        }

    def _is_terminated(self) -> bool:
        """自车碰撞或时间耗尽时，本回合结束。"""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=800):
        """
        创建双向道路。

        :return: 道路
        """
        net = RoadNetwork()

        # 车道
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0],
                [length, 0],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, StraightLane.DEFAULT_WIDTH],
                [length, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )
        net.add_lane(
            "b",
            "a",
            StraightLane(
                [length, 0], [0, 0], line_types=(LineType.NONE, LineType.NONE)
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        在道路上放置若干车辆。

        :return: 自车
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(
                    road,
                    position=road.network.get_lane(("a", "b", 1)).position(
                        70.0 + 40.0 * float(i) + 10.0 * self.np_random.normal(), 0.00
                    ),
                    heading=road.network.get_lane(("a", "b", 1)).heading_at(
                        70.0 + 40.0 * float(i)
                    ),
                    speed=24.0 + 2.0 * self.np_random.normal(),
                    enable_lane_change=False,
                )
            )
        for i in range(2):
            v = vehicles_type(
                road,
                position=road.network.get_lane(("b", "a", 0)).position(
                    200.0 + 100.0 * float(i) + 10.0 * self.np_random.normal(), 0
                ),
                heading=road.network.get_lane(("b", "a", 0)).heading_at(
                    200.0 + 100.0 * float(i)
                ),
                speed=20.0 + 5.0 * self.np_random.normal(),
                enable_lane_change=False,
            )
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)
