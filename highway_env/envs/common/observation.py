from __future__ import annotations

import math
from collections import OrderedDict
from itertools import chain, product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.generation.engine.gen_utils import line_intersection_t
from highway_env.road.generation.spatial_hash import (
    get_proximal_lanes_wrt_gridpoint,
    point_to_gridpoint,
)
from highway_env.road.lane import AbstractLane, PolyLane
from highway_env.road.partitioned_road import PartitionedRoadNetwork
from highway_env.road.road import LaneIndex, Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """获取观察空间。"""
        raise NotImplementedError()

    def observe(self):
        """获取环境当前状态的观察。"""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        用于观察场景的车辆。

        若未单独设置，默认使用第一辆受控车辆。
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):
    """
    直接采集模拟器渲染画面的观察类型。

    按 Nature DQN 中的方法堆叠连续帧。
    观察形状为 C × W × H（通道、宽、高）。

    传入的配置字典需要包含指定字段。
    环境配置中的观察字典示例::

        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # RGB 转灰度的权重
        }
    """

    def __init__(
        self,
        env: AbstractEnv,
        observation_shape: tuple[int, int],
        stack_size: int,
        weights: list[float],
        scaling: float | None = None,
        centering_position: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size,) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        # 此观察使用的查看器配置可与 env.render() 不同，通常采用较小的画面。
        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config["scaling"],
                "centering_position": centering_position
                or viewer_config["centering_position"],
            }
        )
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Box:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: AbstractEnv, horizon: int = 10, **kwargs) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):
    """观察附近车辆的运动学状态。"""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] | None = None,
        vehicles_count: int = 5,
        features_range: dict[str, list[float]] | None = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = True,
        see_behind: bool = False,
        observe_intentions: bool = False,
        include_obstacles: bool = True,
        **kwargs,
    ) -> None:
        """
        :param env: 要观察的环境
        :param features: 观察中使用的特征名称
        :param vehicles_count: 观察中的车辆数量
        :param features_range: 将特征名称映射到 [min, max] 范围的字典
        :param absolute: 是否采用绝对坐标
        :param order: 观察车辆的排列方式，可取 sorted 或 shuffled
        :param normalize: 是否归一化观察
        :param clip: 是否将数值裁剪到指定范围
        :param see_behind: 是否观察后方车辆
        :param observe_intentions: 是否观察其他车辆的目的地
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Box:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        归一化观察数值。

        当前假设道路沿 x 轴直线延伸。
        :param Dataframe df: 观察数据
        """
        if not self.features_range:
            assert self.env.road is not None
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # 加入自车状态。
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # 加入附近交通参与者的状态。
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)
                    for v in close_vehicles[-self.vehicles_count + 1 :]
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)

        df = df[self.features]

        # 归一化并裁剪数值。
        if self.normalize:
            df = self.normalize_obs(df)
        # 补齐缺少的行。
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # 重新排列。
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # 转换为观察空间要求的数据类型，这里不改变数组形状。
        return obs.astype(self.space().dtype)


class OccupancyGridObservation(ObservationType):
    """以占用网格表示附近车辆。"""

    FEATURES: list[str] = ["presence", "vx", "vy", "on_road"]
    GRID_SIZE: list[list[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]
    GRID_STEP: list[int] = [5, 5]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] | None = None,
        grid_size: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_step: tuple[float, float] | None = None,
        features_range: dict[str, list[float]] | None = None,
        absolute: bool = False,
        align_to_vehicle_axes: bool = False,
        clip: bool = True,
        as_image: bool = False,
        **kwargs,
    ) -> None:
        """
        :param env: 要观察的环境
        :param features: 观察中使用的特征名称
        :param grid_size: 网格在真实世界中的范围 [[min_x, max_x], [min_y, max_y]]
        :param grid_step: 相邻网格单元的间距 [step_x, step_y]
        :param features_range: 将特征名称映射到 [min, max] 范围的字典
        :param absolute: 使用绝对坐标还是相对坐标
        :param align_to_vehicle_axes: 为 True 时，网格坐标轴与车辆对齐；否则与世界坐标轴对齐
        :param clip: 是否将观察裁剪到 [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
            dtype=np.intp,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Box:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(
                shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32
            )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        归一化观察数值。

        当前假设道路沿 x 轴直线延伸。
        :param Dataframe df: 观察数据
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # 初始化空数据。
            self.grid.fill(np.nan)

            # 获取附近交通数据。
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # 归一化。
            df = self.normalize(df)
            assert self.features_range is not None
            # 填入特征。
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # 车辆特征。
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # 还原未归一化的坐标，用于计算网格单元索引。
                        if "x" in self.features_range:
                            x = utils.lmap(
                                x,
                                [-1, 1],
                                [
                                    self.features_range["x"][0],
                                    self.features_range["x"][1],
                                ],
                            )
                        if "y" in self.features_range:
                            y = utils.lmap(
                                y,
                                [-1, 1],
                                [
                                    self.features_range["y"][0],
                                    self.features_range["y"][1],
                                ],
                            )
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer, cell[0], cell[1]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> tuple[int, int]:
        """
        将世界坐标位置转换为网格单元索引。

        启用 align_to_vehicle_axes 时，单元位于车辆坐标系中，否则位于世界坐标系中。

        :param position: 世界坐标位置
        :param relative: 位置是否已经相对于观察车辆表示
        :return: 网格单元索引对 (i, j)
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(
                self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position
        return (
            int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),
            int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1])),
        )

    def index_to_pos(self, index: tuple[int, int]) -> np.ndarray:
        position = np.array(
            [
                (index[0] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
                (index[1] + 0.5) * self.grid_step[1] + self.grid_size[1, 0],
            ]
        )

        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(
        self, layer_index: int, lane_perception_distance: float = 100
    ) -> None:
        """
        用一层网格编码道路内（1）与道路外（0）信息。

        遍历各车道及其上等间距的路径点，填充对应单元。
        网格较大而道路网络较小时，这种方法更快。

        :param layer_index: 该层在网格中的索引
        :param lane_perception_distance: 在车辆位置前后此距离内绘制车道
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = cast(Road, self.env.road)

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(
                        origin - lane_perception_distance,
                        origin + lane_perception_distance,
                        lane_waypoints_spacing,
                    ).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer_index, cell[0], cell[1]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        用一层网格编码道路内（1）与道路外（0）信息。

        此实现遍历网格单元，检查其中心对应的世界坐标位于道路内还是道路外。
        网格较小而道路网络较大时，这种方法更快。
        """
        road = cast(Road, self.env.road)
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: AbstractEnv, scales: list[float], **kwargs) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Dict[str, spaces.Box]:
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float64,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float64,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float64,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:  # ty: ignore[invalid-method-override]
        if not self.observer_vehicle:
            return OrderedDict(
                [
                    ("observation", np.zeros((len(self.features),))),
                    ("achieved_goal", np.zeros((len(self.features),))),
                    ("desired_goal", np.zeros((len(self.features),))),
                ]
            )

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.goal.to_dict()])[
                self.features
            ]
        )
        obs = OrderedDict(
            [
                ("observation", obs / self.scales),
                ("achieved_goal", obs / self.scales),
                ("desired_goal", goal / self.scales),
            ]
        )
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: AbstractEnv, attributes: list[str], **kwargs) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        return OrderedDict(
            [(attribute, getattr(self.env, attribute)) for attribute in self.attributes]
        )


class MultiAgentObservation(ObservationType):
    def __init__(self, env: AbstractEnv, observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Tuple:
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


class TupleObservation(ObservationType):
    """将多个未命名的观察类型组合为一个元组。"""

    def __init__(
        self, env: AbstractEnv, observation_configs: list[dict], **kwargs
    ) -> None:
        super().__init__(env)
        self.observation_types = [
            observation_factory(self.env, obs_config)
            for obs_config in observation_configs
        ]

    def space(self) -> spaces.Tuple:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class DictObservation(ObservationType):
    """将多个带名称的观察类型组合为一个字典。"""

    def __init__(
        self, env: AbstractEnv, observation_configs: dict[str, dict], **kwargs
    ) -> None:
        super().__init__(env)
        self.observation_types = {
            name: observation_factory(self.env, obs_config)
            for name, obs_config in observation_configs.items()
        }

    def space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                name: obs_type.space()
                for name, obs_type in self.observation_types.items()
            }
        )

    def observe(self) -> dict[str, Any]:
        return {
            name: obs_type.observe()
            for name, obs_type in self.observation_types.items()
        }


class ExitObservation(KinematicObservation):
    """专用于 exit_env：在运动学观察中加入到下一个出口车道的距离。"""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # 加入自车状态。
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # 加入附近交通参与者的状态。
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_records(
                        [
                            v.to_dict(
                                origin, observe_intentions=self.observe_intentions
                            )
                            for v in close_vehicles[-self.vehicles_count + 1 :]
                        ]
                    )[self.features],
                ],
                ignore_index=True,
            )
        # 归一化并裁剪数值。
        if self.normalize:
            df = self.normalize_obs(df)
        # 补齐缺少的行。
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # 重新排列。
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # 转换为观察空间要求的数据类型，这里不改变数组形状。
        return obs.astype(self.space().dtype)


class LidarObservation(ObservationType):
    """
    通过模拟 LiDAR 传感器阵列，观察附近车辆和实体物体。

    将车辆周围空间划分为多个角度扇区，返回每个扇区一行、共两列的数组：
    - 到最近可碰撞物体（车辆或障碍物）的距离；
    - 该物体相对速度在当前方向上的分量。

    编号 0 的扇区对应角度 0（东侧），随后扇区角度逐渐增加，
    依次经过东、南、西、北方向。
    """

    DISTANCE = 0
    SPEED = 1

    def __init__(
        self,
        env,
        cells: int = 16,
        maximum_range: float = 60,
        normalize: bool = True,
        **kwargs,
    ):
        """
        :param env: 要观察的环境
        :param cells: 角度扇区数量
        :param maximum_range: 传感器的最大探测范围
        :param normalize: 是否将距离和相对速度除以 ``maximum_range``
        """
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 1)) * float("inf")
        self.origin = None

    def space(self) -> spaces.Box:
        high = 1 if self.normalize else self.maximum_range
        return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)

    def observe(self) -> np.ndarray:
        obs = self.trace(
            self.observer_vehicle.position, self.observer_vehicle.velocity
        ).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2), dtype=np.float32) * self.maximum_range

        assert self.env.road is not None
        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # 障碍物覆盖的角度扇区。
            corners = utils.rect_corners(
                obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading
            )
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            if (
                min_angle < -np.pi / 2 < np.pi / 2 < max_angle
            ):  # 物体的角点跨越 +pi 的角度边界。
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end + 1)
            else:  # 物体的角点跨越 0 的角度边界。
                indexes = np.hstack(
                    [np.arange(start, self.cells), np.arange(0, end + 1)]
                )

            # 计算这些扇区内的实际距离。
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = (origin, origin + self.maximum_range * direction)
                distance = utils.distance_to_rect(ray, corners)
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return (
            np.arctan2(position[1] - origin[1], position[0] - origin[0])
            + self.angle / 2
        )

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])


class LaneLidarObservation(LidarObservation):
    """
    让智能体直接观察周围车道边界，将边界视为墙壁。

    要求使用 PartitionedRoadNetwork。
    忽略非 PolyLane 类型的车道。
    """

    def __init__(
        self,
        env,
        cells: int = 16,
        maximum_range: float = 60,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(env, cells, maximum_range, normalize, **kwargs)
        self.vehicle_heading = 0

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        """
        投射射线，观察到车道的距离。
        """
        self.origin = origin.copy()

        self.vehicle_heading = self.observer_vehicle.heading
        self.grid = np.ones((self.cells, 2), dtype=np.float32) * self.maximum_range

        assert self.env.road is not None
        if not isinstance(self.env.road.network, PartitionedRoadNetwork):
            print("PartitionedRoadNetwork required for LaneLidarObservation")
            return self.grid

        gridsize = self.env.road.network.partition_gridsize

        for index in range(self.cells):
            angle = index * self.angle + self.vehicle_heading  # 根据车辆朝向加入角度偏移。
            vx = math.cos(angle)
            vy = math.sin(angle)

            # 沿空间分区网格追踪射线路径。
            gx, gy = point_to_gridpoint(origin, gridsize)

            lanes_checked = set()
            while True:
                # 检查是否相交。
                proximal_lanes = get_proximal_lanes_wrt_gridpoint(
                    self.env.road.network.grid_to_lanes, (gx, gy)
                )
                lanes_to_check = proximal_lanes - lanes_checked

                closest_distance = self.check_ray_intersection_lanes(
                    lanes_to_check, origin, vx, vy
                )

                if closest_distance < self.maximum_range:
                    self.grid[index, LidarObservation.DISTANCE] = closest_distance
                    break
                lanes_checked.update(lanes_to_check)

                # 计算下一个待搜索的网格分区。
                next_gx = gx + (1 if vx > 0 else 0)
                next_gy = gy + (1 if vy > 0 else 0)
                next_gx_t = (
                    self.maximum_range
                    if vx == 0
                    else ((gridsize * next_gx) - origin[0]) / vx
                )
                next_gy_t = (
                    self.maximum_range
                    if vy == 0
                    else ((gridsize * next_gy) - origin[1]) / vy
                )

                if min(next_gx_t, next_gy_t) > self.maximum_range:
                    break

                if next_gx_t <= next_gy_t:
                    gx = next_gx if vx > 0 else next_gx - 1
                if next_gy_t <= next_gx_t:
                    gy = next_gy if vy > 0 else next_gy - 1

            # 所有车道都静止不动，因此 SPEED 数值
            # 只取决于自车本身的速度。
            self.grid[index, LidarObservation.SPEED] = (
                -origin_velocity[0] * vx - origin_velocity[1] * vy
            )

        return self.grid

    def check_ray_intersection_lanes(
        self,
        lanes_to_check: set[LaneIndex],
        origin: np.ndarray,
        vx: float,
        vy: float,
    ) -> float:
        closest_distance = self.maximum_range
        for lane_index in lanes_to_check:
            lane = cast(Road, self.env.road).network.get_lane(lane_index)
            if not isinstance(lane, PolyLane):
                continue
            left_pairs = zip(lane.left_boundary_points, lane.left_boundary_points[1:])
            right_pairs = zip(
                lane.right_boundary_points, lane.right_boundary_points[1:]
            )

            for p0, p1 in chain(left_pairs, right_pairs):
                t_ray, t_segment = line_intersection_t(
                    origin, np.array([vx, vy]), p0, p1 - p0
                )
                if (
                    t_segment >= 0
                    and t_segment <= 1
                    and t_ray >= 0
                    and t_ray <= self.maximum_range
                ):
                    if t_ray < closest_distance:
                        closest_distance = t_ray

        return closest_distance


class NavigationObservation(ObservationType):
    """
    沿通往目标的最短路径，指引智能体驶向下一个路径点。

    [distance_to_waypoint, cos(delta_heading), sin(delta_heading)]
    """

    waypoint_offset = 0

    def space(self) -> spaces.Box:
        low = np.array([0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([np.inf, 1.0, 1.0], dtype=np.float32)
        return spaces.Box(shape=(3,), low=low, high=high, dtype=np.float32)

    def __init__(
        self, env: AbstractEnv, normalize=True, distance_scale=100, **kwargs
    ) -> None:
        super().__init__(env, **kwargs)

        if (
            self.observer_vehicle is None
            or not hasattr(self.observer_vehicle, "goal")
            or self.observer_vehicle.goal is None
        ):
            return
        assert self.env.road is not None
        self.goal_pos = self.observer_vehicle.goal.position
        self.goal_lane_index = self.env.road.network.get_closest_lane_index(
            self.goal_pos, 0
        )

        self.create_new_path()
        self.node = self.path[0]

        self.cached_paths = []
        self.waypoint = self.get_waypoint()

        self.distance_scale = distance_scale
        self.normalize = normalize

    def observe(self) -> np.ndarray:
        if (
            self.observer_vehicle is None
            or not hasattr(self.observer_vehicle, "goal")
            or self.observer_vehicle.goal is None
        ):
            return np.zeros(3, dtype=np.float32)

        self.update_next_node()
        self.waypoint = self.get_waypoint()

        waypt_offset = self.waypoint - self.observer_vehicle.position
        absolute_heading_to_waypt = np.arctan2(waypt_offset[1], waypt_offset[0])

        # 这里的定义与 RelativeGoalObservation 中的
        # delta_h、cos_dh、sin_dh 完全不同。
        delta_h = absolute_heading_to_waypt - self.observer_vehicle.heading
        cos_dh = np.cos(delta_h)
        sin_dh = np.sin(delta_h)

        distance = np.linalg.norm(waypt_offset)
        if self.normalize:
            distance /= self.distance_scale

        return np.array(
            [distance, cos_dh, sin_dh],
            dtype=np.float32,
        )

    def create_new_path(self) -> None:
        """
        计算起始车道到目标车道的最短路径。
        """
        assert self.env.road is not None
        start_lane_index = self.observer_vehicle.lane_index
        start_node = self.get_next_node(start_lane_index)

        self.path = self.env.road.network.shortest_path(
            start_node, self.goal_lane_index[0]
        )

        # 如果本来就会经过目标车道的另一端，
        # 则无需再完整驶过该车道。
        if self.goal_lane_index[1] in self.path:
            self.path = self.env.road.network.shortest_path(
                start_node, self.goal_lane_index[1]
            )

        # 如果路径没有经过最初偏好的起始节点，
        # 而是经过另一端节点，就改从另一端出发。
        if len(self.path) > 1 and (
            self.path[1] == start_lane_index[0] or self.path[1] == start_lane_index[1]
        ):
            self.path.pop(0)

        if (
            len(self.path) == 0
        ):  # 起点恰好等于目标点时，可能出现这种情况。
            self.path.append(start_node)

    def get_next_node(self, lane_index: LaneIndex) -> str:
        assert self.env.road is not None
        _from, _to, _ = lane_index
        lane = self.env.road.network.get_lane(lane_index)

        # 有两个可选的节点。
        # 优先选择与车辆当前朝向一致的节点。
        lane_heading = lane.heading_at(
            lane.local_coordinates(self.observer_vehicle.position)[0]
        )
        raw_diff = lane_heading - self.observer_vehicle.heading
        shortest_diff = (raw_diff + np.pi) % (2 * np.pi) - np.pi
        heading_offset = np.abs(shortest_diff)

        if (heading_offset > np.pi / 2) == (
            lane_index in self.env.road.network.reversed_lane_indices
        ):
            return _to
        else:
            return _from

    def get_waypoint(self) -> np.ndarray:
        """
        计算用于指示路口行驶方向的路径点。
        """
        if self.node == -1:
            return self.goal_pos

        # 查找从 self.node 出发、
        # 通向路径序列下一个节点的车道。
        index = self.path.index(self.node)  # 可保证 node 位于路径中。
        if index == len(self.path) - 1:
            lane_index = self.goal_lane_index
            if lane_index[0] != self.node:
                lane_index = (lane_index[1], lane_index[0], lane_index[2])

        else:
            next_node = self.path[index + 1]
            lane_index = (self.node, next_node, 0)

        assert self.env.road is not None
        lane = self.env.road.network.get_lane(lane_index)

        if lane_index in self.env.road.network.reversed_lane_indices:
            return lane.curve(lane.length - NavigationObservation.waypoint_offset)
        else:
            return lane.curve(NavigationObservation.waypoint_offset)

    def update_next_node(self) -> None:
        """
        计算下一步应驶向哪个交叉路口。
        """
        current_lane_index = self.observer_vehicle.lane_index
        if current_lane_index == self.goal_lane_index:
            self.node = -1
            return

        # node 表示车辆面向的交叉路口。
        # other_node 表示车辆后方的交叉路口。

        node = self.get_next_node(current_lane_index)
        if node == current_lane_index[0]:
            other_node = current_lane_index[1]
        else:
            other_node = current_lane_index[0]

        if node in self.path and other_node in self.path:
            if self.path.index(node) > self.path.index(other_node):
                self.node = node
            else:
                self.node = other_node
            return

        if node in self.path:
            self.node = node
            return
        if other_node in self.path:
            self.node = other_node
            return

        # 车辆已偏离路线，重新寻找路径。
        self.cached_paths.append(self.path)

        # 检查已经生成的路径。
        for cached_path in self.cached_paths[:-1]:
            if node in cached_path:
                self.path = cached_path
                self.node = node
                return

        # 计算新路径。
        self.create_new_path()
        self.node = self.path[0]


class RelativeGoalObservation(ObservationType):
    """
    观察目标停车位相对于自车的位置和朝向。

    [纵向偏移, 横向偏移, cos(delta_heading), sin(delta_heading)]

    observer_vehicle 必须具有 .goal 属性，
    该属性是带有 .position 和 .heading 的 RoadObject。
    """

    OBS_SIZE = 4  # [纵向偏移, 横向偏移, cos_dh, sin_dh]

    def __init__(
        self,
        env: AbstractEnv,
        normalize: bool = True,
        distance_scale: float = 100.0,
        **kwargs,
    ) -> None:
        """
        :param normalize: 为 True 时，将位置偏移除以 position_scale
        :param position_scale: dx_body 和 dy_body 的归一化除数
        """
        super().__init__(env)
        self.normalize = normalize
        self.distance_scale = distance_scale

    def space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_SIZE,),
            dtype=np.float32,
        )

    def observe(self) -> np.ndarray:
        ego = self.observer_vehicle

        if ego is None or not hasattr(ego, "goal") or ego.goal is None:
            return np.zeros(self.OBS_SIZE, dtype=np.float32)

        goal = ego.goal

        world_offset = goal.position - ego.position

        c, s = np.cos(ego.heading), np.sin(ego.heading)
        R = np.array([[c, s], [-s, c]])
        body_offset = R @ world_offset

        if self.normalize:
            body_offset /= self.distance_scale

        delta_h = goal.heading - ego.heading
        cos_dh = np.cos(delta_h)
        sin_dh = np.sin(delta_h)

        obs = np.array(
            [body_offset[0], body_offset[1], cos_dh, sin_dh],
            dtype=np.float32,
        )
        return obs


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    match config["type"]:
        case "TimeToCollision":
            return TimeToCollisionObservation(env, **config)
        case "Kinematics":
            return KinematicObservation(env, **config)
        case "OccupancyGrid":
            return OccupancyGridObservation(env, **config)
        case "KinematicsGoal":
            return KinematicsGoalObservation(env, **config)
        case "GrayscaleObservation":
            return GrayscaleObservation(env, **config)
        case "AttributesObservation":
            return AttributesObservation(env, **config)
        case "MultiAgentObservation":
            return MultiAgentObservation(env, **config)
        case "TupleObservation":
            return TupleObservation(env, **config)
        case "DictObservation":
            return DictObservation(env, **config)
        case "LidarObservation":
            return LidarObservation(env, **config)
        case "ExitObservation":
            return ExitObservation(env, **config)
        case "LaneLidarObservation":
            return LaneLidarObservation(env, **config)
        case "NavigationObservation":
            return NavigationObservation(env, **config)
        case "RelativeGoalObservation":
            return RelativeGoalObservation(env, **config)
        case _:
            raise ValueError("Unknown observation type")
