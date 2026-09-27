from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    高速公路驾驶环境。

    车辆在多车道直线高速公路上行驶；高速行驶、靠右行驶和避免碰撞会获得奖励。
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # 与其他车辆碰撞时的奖励。
                "right_lane_reward": 0.1,  # 在最右侧车道行驶时的奖励；其他车道的奖励
                # 按车道位置线性递减至零。
                "high_speed_reward": 0.4,  # 达到奖励速度上限时的奖励；速度较低时，
                # 根据 config["reward_speed_range"] 线性递减至零。
                "lane_change_reward": 0,  # 每次变道动作对应的奖励。
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            },
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """创建由相邻直线车道组成的道路。"""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )

    def _create_vehicles(self) -> None:
        """随机创建指定类型的车辆，并将其加入道路。"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        奖励鼓励车辆高速行驶、靠右行驶并避免碰撞。
        :param action: 上一次执行的动作
        :return: 对应的奖励
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # 使用前向速度分量，参见 https://github.com/Farama-Foundation/HighwayEnv/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """自车发生碰撞时，本回合结束。"""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """达到时间上限时截断本回合。"""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    运行更快的 highway-v0 变体：
    - 降低仿真频率；
    - 减少场景中的车辆和车道，并缩短回合时长；
    - 只检查受控车辆与其他车辆之间的碰撞。
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        utils.update_config(
            cfg,
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            },
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # 关闭非受控车辆之间的碰撞检查。
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
