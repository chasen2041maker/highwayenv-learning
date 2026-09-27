import warnings
from itertools import chain
from typing import cast

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv, Observation
from highway_env.envs.common.action import Action, action_factory
from highway_env.envs.common.observation import (
    DictObservation,
    NavigationObservation,
    observation_factory,
)
from highway_env.envs.common.warnings import HighwayEnvExperimentalWarning
from highway_env.road.generation.engine.gen_utils import Lane
from highway_env.road.generation.generator import generate_random_lanes
from highway_env.road.generation.spatial_hash import (
    get_proximal_lanes_wrt_gridpoint,
    point_to_gridpoint,
)
from highway_env.road.lane import PolyLane
from highway_env.road.partitioned_road import PartitionedRoadNetwork
from highway_env.road.road import LineType, Road
from highway_env.vehicle.objects import Landmark, RoadObject


class ParkingSpot(Landmark):
    LENGTH = 7.0
    WIDTH = 3.0


class RandomRoadEnv(AbstractEnv):
    """
    在程序生成的道路网络上进行导航、通行协商和泊车的环境。

    智能体的目标是在不撞上路缘或其他车辆的情况下，尽快到达指定停车位。
    """

    def __init__(
        self, config: dict | None = None, render_mode: str | None = None
    ) -> None:
        super().__init__(config=config, render_mode=render_mode)
        self.lanes = []
        self.vehicle_parked = False

        # TODO：环境稳定后移除此警告。
        warnings.warn(
            HighwayEnvExperimentalWarning.template % self.__class__,
            HighwayEnvExperimentalWarning,
            stacklevel=2,
        )

    @classmethod
    def default_config(cls) -> dict:
        """
        - **max_timesteps**：达到此决策步数后截断回合。
        - **curb_collision_reward**：撞到车道边界时的一次性惩罚。
        - **car_collision_reward**：撞到其他车辆或物体时的一次性惩罚。
        - **parking_reward**：停入目标停车位时的一次性奖励。
        - **parking_score_threshold**：判定泊车完成所需的接近程度阈值。
        - **parking_score_weights**：位置、速度和朝向对齐程度的权重。
        - **route_following_reward_scalar**：朝向或远离下一个路径点行驶时的奖励或惩罚系数。
        - **timestep_reward**：每一步的时间惩罚。
        - **parking_seed**：确定生成道路中停车位位置的伪随机种子。
        - **generation_params**：传递给生成器的自定义参数。
        - **preloaded_lanes**：提供已有道路网络，跳过新网络生成。
        - **lane_partition_gridsize**：将车道划分为空间网格以进行邻近检查时的网格尺寸；
          较小的值可减少密集路网中不必要的检查。
        """
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "DictObservation",
                    "observation_configs": {
                        "lane_lidar": {"type": "LaneLidarObservation"},
                        "navigation": {"type": "NavigationObservation"},
                        "relative_goal": {"type": "RelativeGoalObservation"},
                        "lidar": {"type": "LidarObservation"},
                    },
                },
                "action": {"type": "ContinuousAction"},
                "screen_width": 1200,
                "screen_height": 700,
                "max_timesteps": 1000,
                "curb_collision_reward": -10,
                "car_collision_reward": -20,
                "parking_reward": 10,
                "parking_score_threshold": 0.7,
                "parking_score_weights": [0.5, 1, 3],
                "route_following_reward_scalar": 0.1,
                "timestep_reward": -0.01,
                "parking_seed": 0,
                "generation_params": None,
                "preloaded_lanes": None,
                "lane_partition_gridsize": 30,
            }
        )
        return config

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reset(self) -> None:
        self.lanes = self._make_road()
        assert self.road is not None

        parking_rng = np.random.default_rng(self.config["parking_seed"])
        self.create_parking_spots(
            num_spots=2, spot_width=3, spot_height=6, rng=parking_rng
        )
        spawn_spot = self.road.objects[0]

        self.vehicle = self.action_type.vehicle_class(
            self.road, spawn_spot.position, spawn_spot.heading, 0.0
        )
        self.vehicle.goal = self.road.objects[1]

        self.road.vehicles.append(self.vehicle)

        self.vehicle_parked = False

    def _reward(self, action: Action) -> float:
        """
        奖励组成：
        - 路缘碰撞惩罚；
        - 车辆之间的碰撞惩罚；
        - 一次性泊车奖励；
        - 每步时间惩罚；
        - 路线跟随奖励。
        """
        # 碰撞
        collided_with_curb = self.detect_object_lane_collision(self.vehicle)
        collided_with_car = self.vehicle.crashed

        if collided_with_curb or collided_with_car:
            self.vehicle.crashed = True
            total_timestep_punishment_left = min(
                (self.config["max_timesteps"] + 1 - self.time)
                * self.config["timestep_reward"],
                0,
            )

            if collided_with_curb:
                return (
                    self.config["curb_collision_reward"]
                    + total_timestep_punishment_left
                )
            if collided_with_car:
                return (
                    self.config["car_collision_reward"] + total_timestep_punishment_left
                )

        # 泊车
        parking_score = self.compute_parking_score()
        if parking_score < self.config["parking_score_threshold"]:
            self.vehicle_parked = True
            return self.config["parking_reward"]

        # 路线跟随
        reward_earned = self.config["timestep_reward"]

        if self.config["route_following_reward_scalar"] != 0:
            navigation_observation = None
            if isinstance(self.observation_type, DictObservation):
                for obs in self.observation_type.observation_types.values():
                    if isinstance(obs, NavigationObservation):
                        navigation_observation = obs
            elif isinstance(self.observation_type, NavigationObservation):
                navigation_observation = self.observation_type
            assert (
                navigation_observation is not None
            ), "NavigationObservation must be included as an observation if route_following_reward_scalar is nonzero"

            waypoint_vector = navigation_observation.waypoint - self.vehicle.position
            route_following_score = (
                self.config["route_following_reward_scalar"]
                * np.dot(waypoint_vector, self.vehicle.velocity)
                / np.linalg.norm(waypoint_vector)
            )
            # print("Route following score:", route_following_score)
            reward_earned += route_following_score

        return reward_earned

    def _is_terminated(self) -> bool:
        """
        发生碰撞或成功泊车时终止回合。
        """
        return self.vehicle_parked or self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time > self.config["max_timesteps"]

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        info = super()._info(obs, action)
        info["parked"] = self.vehicle_parked
        return info

    def compute_parking_score(self, p: float = 0.5) -> float:
        # 计算奖励时不使用 RelativeGoalObservation。
        # 这里采用类似 ParkingEnv.compute_reward 的方式。
        # 泊车评分越低越好。

        position_diff = np.linalg.norm(
            self.vehicle.position - self.vehicle.goal.position
        )
        velocity_diff = np.linalg.norm(self.vehicle.velocity)
        alignment_penalty = 1 - abs(
            np.cos(self.vehicle.heading - self.vehicle.goal.heading)
        )  # 完全对齐（同向或反向）时为 0，横向垂直时为 1。

        components = np.array(
            [
                position_diff,
                velocity_diff,
                alignment_penalty,
            ]
        )
        weights = np.array(self.config["parking_score_weights"])
        return np.power(np.dot(np.abs(components), weights), p)

    def _make_road(self) -> list[Lane]:
        if self.config["preloaded_lanes"] is None:
            try:
                lanes = generate_random_lanes(
                    self.np_random, self.config["generation_params"]
                )
            except Exception as e:
                raise RuntimeError(
                    "Fatal error encountered when generating road network."
                    "If this issue persists, try a different seed."
                    f"\n\tOriginal error: {e}"
                ) from e
        else:
            lanes = self.config["preloaded_lanes"]

        net = PartitionedRoadNetwork(
            partition_gridsize=self.config["lane_partition_gridsize"]
        )
        for lane in lanes:
            real_lane = PolyLane(
                lane_points=lane.points,
                left_boundary_points=lane.left_points,
                right_boundary_points=lane.right_points,
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            )
            net.add_lane_bidirectional(lane.start, lane.end, real_lane)

        self.road = Road(net)

        return lanes

    def create_parking_spots(
        self,
        num_spots: int,
        spot_width: float,
        spot_height: float,
        rng: np.random.Generator,
    ) -> bool:
        """
        :param num_spots: 要生成的停车位数量
        :param spot_width: 停车位宽度，必须小于 lane_width
        :param spot_height: 停车位长度，必须小于 forward_speed
        :param rng: 随机数生成器
        :return: 空间是否足够生成指定数量的停车位
        """
        assert self.road is not None
        curb_spot_offset = 0.1
        # 路段索引结构：segment_index: {lane_id, side, pt_id (1-(len-2))}
        segment_indices = []

        for lane_id, lane in enumerate(self.lanes):
            for side in ["left_points", "right_points"]:
                for pt_id in range(1, len(getattr(lane, side)) - 2):
                    segment_indices.append(
                        {"lane_id": lane_id, "side": side, "pt_id": pt_id}
                    )

        rng.shuffle(segment_indices)

        num_parking_spots = 0
        segment_indices_i = 0
        while num_parking_spots < num_spots and segment_indices_i < len(
            segment_indices
        ):
            segment_index = segment_indices[segment_indices_i]
            lane_id = cast(int, segment_index["lane_id"])
            side = cast(str, segment_index["side"])
            pt_id = cast(int, segment_index["pt_id"])

            lane = self.lanes[lane_id]
            lane_side = getattr(lane, side)
            pt0 = lane_side[pt_id]
            pt1 = lane_side[pt_id + 1]

            # 尝试放置一个停车位，
            # 使其与当前车道路段平行。

            # 要求 1：该路段必须足够长，
            # 能够容纳整个停车位。
            seg_dist = np.linalg.norm(pt0 - pt1)
            if seg_dist < spot_height:
                segment_indices_i += 1
                continue

            # 计算新停车位的几何形状。
            vec = pt1 - pt0
            vec /= np.linalg.norm(vec)

            if side == "right_points":
                perp_vec = np.array([vec[1], -vec[0]])
            else:
                perp_vec = np.array([-vec[1], vec[0]])

            center = (pt0 + pt1) / 2 + (perp_vec * (curb_spot_offset + spot_width / 2))
            heading = np.atan2(vec[1], vec[0])

            new_parking_spot = ParkingSpot(self.road, center, heading)
            self.road.objects.append(new_parking_spot)
            num_parking_spots += 1

            # 要求 2：矩形停车区域
            # 不能与其他车道相交。
            if self.detect_object_lane_collision(new_parking_spot):
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue

            # 要求 3：矩形停车区域
            # 不能与任何已有停车位相交。
            collision_detected = False
            for other_object in self.road.objects:
                if other_object is not new_parking_spot:
                    collision_detected, _, _ = new_parking_spot._is_colliding(
                        other_object, 0
                    )
                    if collision_detected:
                        break

            if collision_detected:
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue

            segment_indices_i += 1

        if num_parking_spots < num_spots:
            print(
                "INSUFFICIENT SPOTS FOUND;"
                f" {num_parking_spots} / {num_spots} parking spots generated"
            )
            return False
        return True

    def detect_object_lane_collision(self, obj: RoadObject) -> bool:
        assert self.road is not None and isinstance(
            self.road.network, PartitionedRoadNetwork
        )
        gridpoints = set()
        for pt in obj.polygon():
            gridpoints.add(point_to_gridpoint(pt, self.road.network.partition_gridsize))

        proximal_lanes = set()
        for gpt in gridpoints:
            proximal_lanes.update(
                get_proximal_lanes_wrt_gridpoint(self.road.network.grid_to_lanes, gpt)
            )

        for lane_index in proximal_lanes:
            lane = cast(PolyLane, self.road.network.get_lane(lane_index))

            left_pairs = zip(lane.left_boundary_points, lane.left_boundary_points[1:])
            right_pairs = zip(
                lane.right_boundary_points, lane.right_boundary_points[1:]
            )

            for p0, p1 in chain(left_pairs, right_pairs):
                if obj.intersects_with_line(p0, p1):
                    return True

        return False
