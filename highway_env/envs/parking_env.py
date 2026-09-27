from __future__ import annotations

from abc import abstractmethod
from itertools import repeat

import numpy as np
from gymnasium import Env

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    基于目标的环境接口。

    Stable Baselines3 的事后经验回放（HER）等智能体需要此接口。
    它最初属于 https://github.com/openai/gym，后来移至
    https://github.com/Farama-Foundation/gym-robotics。
    原实现说明：当时 gym-robotics 没有正式 PyPI 包，而 PyPI 不允许直接依赖 Git 仓库，
    因此这里复现接口，而不将 gym-robotics 加入依赖。

    这种环境的运行方式与普通 OpenAI Gym 环境相同，但要求观察空间具有指定结构：
    至少包含 `observation`、`desired_goal` 和 `achieved_goal` 三项。
    `desired_goal` 表示智能体应尝试达到的目标；`achieved_goal` 表示当前已达到的目标；
    `observation` 包含通常意义上的环境观察。
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """计算单步奖励。将奖励函数独立出来，使其由期望目标与实际达到的目标共同决定。
        若还需加入与目标无关的奖励，可在 info 中提供必要数据并据此计算。

        Args:
            achieved_goal (object): 执行过程中实际达到的目标
            desired_goal (object): 希望智能体尝试达到的目标
            info (dict): 包含附加信息的字典
        Returns:
            float: 实际目标相对于期望目标所对应的奖励。原接口示例应满足：
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    连续控制环境。

    实现到达目标的任务：智能体观察自身位置和速度，通过控制加速度和转向到达指定目标。

    感谢 Munir Jojo-Verge 提出思路并完成最初实现。
    """

    # 泊车环境使用 GrayscaleObservation 时，
    # 仍需利用 PARKING_OBS 计算奖励和附加信息。
    # 此问题由 Mcfly 修复：https://github.com/McflyWZX
    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 100,
                "screen_width": 600,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],
                "scaling": 7,
                "controlled_vehicles": 1,
                "vehicles_count": 0,
                "add_walls": True,
            },
        )
        return config

    def define_spaces(self) -> None:
        """
        根据配置设置观察和动作的类型及空间。
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 14) -> None:
        """
        创建由相邻直线车道组成的道路。

        :param spots: 停车位数量
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [x, y_offset], [x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [x, -y_offset], [x, -y_offset - length], width=width, line_types=lt
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            neighbour_vehicles_connected_lanes=self.config[
                "neighbour_vehicles_connected_lanes"
            ],
        )

    def _create_vehicles(self) -> None:
        """随机创建指定类型的车辆，并将其加入道路。"""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # 受控车辆
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0.0], 2.0 * np.pi * self.np_random.uniform(), 0.0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # 目标
        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # 其他车辆
        for _ in repeat(None, self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, longitudinal=4.0, speed=0.0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # 围墙
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        根据与目标的接近程度给予奖励。

        使用加权 p 范数。

        :param achieved_goal: 实际达到的目标
        :param desired_goal: 期望目标
        :param dict info: 附加信息
        :param p: 奖励使用的 Lp^p 范数指数；p<1 时可使 [0,1] 范围内奖励呈现较高峰度
        :return: 对应的奖励
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        """自车碰撞、到达目标或时间耗尽时，本回合结束。"""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """时间耗尽时截断本回合。"""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})
