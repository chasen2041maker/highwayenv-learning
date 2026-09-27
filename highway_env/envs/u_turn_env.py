from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, ConnectedLaneNeighboursMixin
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle


class UTurnEnv(AbstractEnv):
    """
    掉头风险分析任务：智能体超越阻碍通行的车辆。
    需要在高速超车与保证安全之间作出权衡。
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {"type": "TimeToCollision", "horizon": 16},
                "action": {"type": "DiscreteMetaAction", "target_speeds": [8, 16, 24]},
                "screen_width": 789,
                "screen_height": 289,
                "duration": 10,
                "collision_reward": -1.0,  # 车辆碰撞时的惩罚。
                "left_lane_reward": 0.1,  # 保持在最左侧车道时的奖励。
                "high_speed_reward": 0.4,  # 保持巡航速度时的奖励。
                "reward_speed_range": [8, 24],
                "normalize_reward": True,
                "offroad_terminal": False,
            },
        )
        return config

    def _reward(self, action: int) -> float:
        """
        奖励车辆高速行驶并避免碰撞。
        :param action: 执行的动作
        :return: 此次状态和动作转移对应的奖励
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
                    self.config["high_speed_reward"] + self.config["left_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "left_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=128):
        """
        创建带逆时针掉头弯道的双车道道路。
        :return: 道路
        """
        net = RoadNetwork()

        # 定义掉头后的上方直线车道。
        # 这些车道的 x 坐标从 length 延伸到 0。
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length, StraightLane.DEFAULT_WIDTH],
                [0, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length, 0],
                [0, 0],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        # 定义逆时针的圆形掉头车道。
        center = [length, StraightLane.DEFAULT_WIDTH + 20]  # [m]
        radius = 20  # [m]
        alpha = 0  # 单位：角度

        radii = [radius, radius + StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        offset = 2 * radius

        # 定义掉头前的下方直线车道。
        # 这些车道的 x 坐标从 0 延伸到 length。
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [
                    0,
                    (
                        (2 * StraightLane.DEFAULT_WIDTH + offset)
                        - StraightLane.DEFAULT_WIDTH
                    ),
                ],
                [
                    length,
                    (
                        (2 * StraightLane.DEFAULT_WIDTH + offset)
                        - StraightLane.DEFAULT_WIDTH
                    ),
                ],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                [length, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
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
        有针对性地添加车辆，测试在给定巡航区间内进行掉头时的安全行为边界。

        :return: 自车
        """

        # 这些变量为驾驶行为引入小幅变化。
        position_deviation = 2.0
        speed_deviation = 2.0

        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_lane.position(0, 0), speed=16.0
        )
        # 增强对前方弯道的预判。
        ego_vehicle.PURSUIT_TAU = MDPVehicle.TAU_HEADING
        try:
            ego_vehicle.plan_route_to("d")
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # 原注释说明：若实验需要调整车辆交互的随机程度，
        # 可以考虑注释掉 randomize_behavior() 调用。

        # 车辆 1：阻挡自车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("a", "b", 0),
            longitudinal=25.0 + self.np_random.normal() * position_deviation,
            speed=13.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # 车辆 2：迫使自车进行风险较高的超车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("a", "b", 1),
            longitudinal=56.0 + self.np_random.normal() * position_deviation,
            speed=14.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # 车辆 3：阻挡自车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("b", "c", 1),
            longitudinal=0.5 + self.np_random.normal() * position_deviation,
            speed=4.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # 车辆 4：迫使自车进行风险较高的超车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("b", "c", 0),
            longitudinal=17.5 + self.np_random.normal() * position_deviation,
            speed=5.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # 车辆 5：阻挡自车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("c", "d", 0),
            longitudinal=1.0 + self.np_random.normal() * position_deviation,
            speed=3.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # 车辆 6：迫使自车进行风险较高的超车
        vehicle = vehicles_type.make_on_lane(
            self.road,
            ("c", "d", 1),
            longitudinal=30.0 + self.np_random.normal() * position_deviation,
            speed=5.5 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to("d")
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)


class ConnectedLaneUTurnEnv(ConnectedLaneNeighboursMixin, UTurnEnv):
    pass
