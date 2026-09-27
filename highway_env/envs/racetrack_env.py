from __future__ import annotations

from itertools import repeat

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, ConnectedLaneNeighboursMixin
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


class RacetrackEnv(AbstractEnv):
    """
    连续控制环境。

    智能体需要学习两项技能：
    - 沿赛道行驶；
    - 避免与其他车辆碰撞。

    感谢 @supperted825 提出思路并完成最初实现。
    参见 https://github.com/Farama-Foundation/HighwayEnv/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
            },
        )
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> dict[str, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1
            / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            return self.vehicle.crashed or not self.vehicle.on_road
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # 设置各路段限速，路段顺序为：直线、半径 20 的弯道、直线、
        # 半径 15 的弯道、半径 15 的弯道、直线、两段半径 25 的弯道、半径 18.5 的弯道。
        speedlimits = [None, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

        # 初始化第一条车道。
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # 向道路网络添加车道：直线路段。
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 第 2 段：圆弧 1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 第 3 段：竖直直线
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 第 4 段：圆弧 2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 第 5 段：圆弧 3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 第 6 段：斜线
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 第 7 段：圆弧 4。圆弧过大时会出问题，因此拆成两段。
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 第 8 段：圆弧 5，重新连接起点。
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
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
        在高速公路和汇入车道上放置若干其他车辆及自车。
        """
        rng = self.np_random

        # 受控车辆
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # 前方车辆
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0.0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6.0 + rng.uniform(high=3.0),
            )
            self.road.vehicles.append(vehicle)

            # 其他车辆
            for _ in repeat(None, rng.integers(self.config["other_vehicles"])):
                rand_lane_index = self.road.network.random_lane_index(rng)

                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    rand_lane_index,
                    longitudinal=rng.uniform(
                        low=0.0, high=self.road.network.get_lane(rand_lane_index).length
                    ),
                    speed=6.0 + rng.uniform(high=3.0),
                )
                # 避免刚开始就发生碰撞。
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)


class ConnectedLaneRacetrackEnv(ConnectedLaneNeighboursMixin, RacetrackEnv):
    pass


class RacetrackEnvLarge(RacetrackEnv):
    """
    具有三条车道的更大赛道地图。

    贡献者：@m-walters
    """

    def _make_road(self) -> None:
        net = RoadNetwork()
        w = 5
        w2 = 2 * w
        default_speedlimit = self.config["speed_limit"]

        # 初始化第一条车道。
        lane = StraightLane(
            [42, 0],
            [200, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        )
        self.lane = lane

        # 向道路网络添加车道：直线路段。
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w],
                [200, w],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w2],
                [200, w2],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 第 2 段：圆弧 1
        center1 = [200, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w2,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 第 3 段：竖直直线
        delta_extension = -1.0  # 改善连接处。
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220, -20],
                [220, -60 + delta_extension],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w, -20],
                [220 + w, -60 + delta_extension],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w2, -20],
                [220 + w2, -60 + delta_extension],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 第 4 段：圆弧 2
        center4 = [205, -60]
        radii4 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 第 5 段：圆弧 3
        center5 = [170, -60]
        radii5 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=w,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 - w,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 第 6 段：斜线
        # 向 [-30, -30] 方向延伸。
        extend = np.array([-30, -30])
        start6 = np.array([155.7, -45.7])
        end6 = np.array([135.7, -65.7]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6,
                end6,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_2 = np.array([159.3934, -49.2])
        end6_2 = np.array([139.3934, -69.2]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_2,
                end6_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_3 = np.array(
            [
                start6[0] + 2 * (start6_2[0] - start6[0]),
                start6[1] + 2 * (start6_2[1] - start6[1]),
            ]
        )
        end6_3 = np.array(
            [
                end6[0] + 2 * (end6_2[0] - end6[0]),
                end6[1] + 2 * (end6_2[1] - end6[1]),
            ]
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_3,
                end6_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 第 7 段：圆弧 4
        # 以斜线为参照进行镜像。
        center7 = np.array([118.1, -48.1]) + extend
        radii7 = 25
        theta7 = 317
        # theta7_end = 270 - (theta7 - 270) - 10
        theta7_end = 205
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end - 3),  # 使连接更平顺。
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + w2,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 第 8 段：下一段斜线
        # 相对于上一段圆弧的圆心进行镜像。
        start8 = np.array(
            [
                center7[0] + radii7 * np.cos(np.deg2rad(theta7_end)),
                center7[1] + radii7 * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_2 = np.array(
            [
                center7[0] + (radii7 + w) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w) * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_3 = np.array(
            [
                center7[0] + (radii7 + w2) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w2) * np.sin(np.deg2rad(theta7_end)),
            ]
        )

        # 提前采用第 9 段的半径，使连接更平顺。
        radii9 = 15
        rad = np.deg2rad(30)
        end8 = np.array([42 - radii9 * np.cos(rad), -radii9 - radii9 * np.sin(rad)])
        end8_2 = np.array(
            [42 - (radii9 + w) * np.cos(rad), -radii9 - (radii9 + w) * np.sin(rad)]
        )
        end8_3 = np.array(
            [42 - (radii9 + w2) * np.cos(rad), -radii9 - (radii9 + w2) * np.sin(rad)]
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8,
                end8,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_2,
                end8_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_3,
                end8_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 第 9 段：圆弧 7，终段
        # 由于 y2 = 0……
        center9 = np.array([42, -radii9])
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9,
                np.deg2rad(210),
                np.deg2rad(88),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w,
                np.deg2rad(210),
                np.deg2rad(90),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w2,
                np.deg2rad(212),
                np.deg2rad(88),  # 更平顺的连接
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
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


class ConnectedLaneRacetrackEnvLarge(ConnectedLaneNeighboursMixin, RacetrackEnvLarge):
    pass


class RacetrackEnvOval(RacetrackEnv):
    """
    参数可配置的椭圆形赛道。

    主要扩展功能：
    - 自定义车道数量；
    - 调整水平直线路段的长度；
    - 可选路障，用于要求智能体作出变道决策。

    贡献者：@christophluther
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
                "length": 100,  # 设为 0 时，从 [100, 200] 中随机取值。
                "no_lanes": 3,  # 设为 0 时，从 [2, 7] 中随机取值。
                "block_lane": False,  # 封堵中间车道。
                "force_decision": False,  # 封堵第 1 和第 3 条车道。
            },
        )
        return config

    def _make_road(self) -> None:
        net = RoadNetwork()

        # 定义随机数生成器。
        rng = np.random.default_rng()

        # 设置各路段的限速。
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # 定义长度。
        if self.config["length"] == 0:
            length = rng.integers(100, high=200)
        else:
            length = self.config["length"]

        if self.config["no_lanes"] == 0:
            no_lanes = rng.integers(2, high=7)
        else:
            no_lanes = self.config["no_lanes"]

        # 直线段 1：初始化最内侧车道。
        lane = StraightLane(
            [0, 0],
            [length + 1, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # 依次添加车道。
        net.add_lane("a", "b", lane)

        # 各路段须使用独立循环，以便设置路段名称。
        for i in range(1, no_lanes - 1):
            # 在内侧和外侧车道之间添加车道。
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, i * 5],
                    [length + 1, i * 5],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[1],
                ),
            )

        # 直线段 1：外侧车道
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, (no_lanes - 1) * 5],
                [length + 1, (no_lanes - 1) * 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 弯道 1：内侧车道
        center1 = [length, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )

        for i in range(1, no_lanes - 1):
            # 在内侧和外侧车道之间添加车道。
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + i * 5,
                    np.deg2rad(90),
                    np.deg2rad(0),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[2],
                ),
            )

        # 弯道 1：外侧车道
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + (no_lanes - 1) * 5,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 竖直直线段 1：内侧车道
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length + 20, -20],
                [length + 20, -50],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        for i in range(1, no_lanes - 1):
            # 在内侧和外侧车道之间添加车道。
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [length + 20 + i * 5, -20],
                    [length + 20 + i * 5, -50],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[3],
                ),
            )

        # 竖直直线段 1：外侧车道
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length + 20 + (no_lanes - 1) * 5, -20],
                [length + 20 + (no_lanes - 1) * 5, -50],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 弯道 2：内侧车道
        center2 = [length + 5, -50]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )

        for i in range(1, no_lanes - 1):
            # 在内侧和外侧车道之间添加车道。
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2 + i * 5,
                    np.deg2rad(0),
                    np.deg2rad(-90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[4],
                ),
            )

        # 弯道 2：外侧车道
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + (no_lanes - 1) * 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 水平直线段 2：内侧车道
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length + 5, -65],
                [-5, -65],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[5],
            ),
        )

        for i in range(1, no_lanes - 1):
            # 在内侧和外侧车道之间添加车道。
            net.add_lane(
                "e",
                "f",
                StraightLane(
                    [length + 5, -(65 + i * 5)],
                    [-5, -(65 + i * 5)],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[5],
                ),
            )

        # 水平直线段 2：外侧车道
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length + 5, -(65 + (no_lanes - 1) * 5)],
                [-5, -(65 + (no_lanes - 1) * 5)],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[5],
            ),
        )

        # 弯道 3：内侧车道
        center4 = [-5, -50]
        radii4 = 15
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[6],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "f",
                "g",
                CircularLane(
                    center4,
                    radii4 + i * 5,
                    np.deg2rad(-90),
                    np.deg2rad(-180),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[6],
                ),
            )

        # 弯道 3：外侧车道
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4 + (no_lanes - 1) * 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[6],
            ),
        )

        # 竖直直线段 2：内侧车道
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20, -50],
                [-20, -20],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[7],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "g",
                "h",
                StraightLane(
                    [-20 - i * 5, -50],
                    [-20 - i * 5, -20],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[7],
                ),
            )

        # 竖直直线段 2：外侧车道
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20 - (no_lanes - 1) * 5, -50],
                [-20 - (no_lanes - 1) * 5, -20],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[7],
            ),
        )

        # 弯道 4：内侧车道
        center6 = [0, -20]
        radii6 = 20
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[8],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "h",
                "a",
                CircularLane(
                    center6,
                    radii6 + i * 5,
                    np.deg2rad(180),
                    np.deg2rad(90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimits[8],
                ),
            )

        # 弯道 4：外侧车道
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6 + (no_lanes - 1) * 5,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
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

        # 要求作出二选一决策的场景。
        if self.config["block_lane"]:
            for i in [40.0, 43.0, 46.0, 49.0]:
                road.objects.append(Obstacle(road, [length - i, 3.75]))
                road.objects.append(Obstacle(road, [length - i, 6.25]))

        if self.config["force_decision"]:
            for i in [-1.25, 1.25, 8.85, 11.25]:
                road.objects.append(Obstacle(road, [length - 90.0, i]))

        self.road = road

    # 根据车道数量调整 CL。
    def _make_vehicles(self) -> None:
        """
        在道路上放置若干车辆。
        """
        rng = self.np_random

        # 受控车辆
        self.controlled_vehicles = []
        # 车道数量
        no_lanes = self.config["no_lanes"]

        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(no_lanes))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20.0, 50.0)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # 前方车辆
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0.0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6.0 + rng.uniform(high=3.0),
            )
            self.road.vehicles.append(vehicle)

            # 其他车辆
            for _ in repeat(None, rng.integers(self.config["other_vehicles"])):
                rand_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    rand_lane_index,
                    longitudinal=rng.uniform(
                        low=0.0, high=self.road.network.get_lane(rand_lane_index).length
                    ),
                    speed=6.0 + rng.uniform(high=3.0),
                )
                # 避免刚开始就发生碰撞。
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20.0:
                        break
                else:
                    self.road.vehicles.append(vehicle)


class ConnectedLaneRacetrackEnvOval(ConnectedLaneNeighboursMixin, RacetrackEnvOval):
    pass
