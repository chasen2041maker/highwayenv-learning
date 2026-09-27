from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pygame
from numpy.typing import NDArray

from highway_env.envs.common.action import (
    ActionType,
    ContinuousAction,
    DiscreteMetaAction,
)
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.graphics import VehicleGraphics


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv, Action
    from highway_env.envs.common.observation import (
        LaneLidarObservation,
        LidarObservation,
        NavigationObservation,
        ObservationType,
    )


class EnvViewer:
    """用于渲染高速公路驾驶环境的查看器。"""

    SAVE_IMAGES = False
    agent_display = None

    def __init__(self, env: AbstractEnv, config: dict | None = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]
        self.observer_vehicle = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None

        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # 绘图并不一定需要显示窗口。省略 display.set_mode() 调用，
        # 即可直接在绘图表面上绘制，
        # 无需处理屏幕窗口，适用于云端计算等场景。
        if not self.offscreen:
            self.screen = pygame.display.set_mode(
                [self.config["screen_width"], self.config["screen_height"]]
            )
        if self.agent_display:
            self.extend_display()
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get(
            "scaling", self.sim_surface.INITIAL_SCALING
        )
        self.sim_surface.centering_position = self.config.get(
            "centering_position", self.sim_surface.INITIAL_CENTERING
        )
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

    def set_agent_display(self, agent_display: Callable) -> None:
        """
        设置智能体提供的绘制回调。

        智能体可以在专用绘图表面或仿真画面上显示自身行为。

        :param agent_display: 智能体提供的表面绘制回调
        """
        if EnvViewer.agent_display is None:
            self.extend_display()
        EnvViewer.agent_display = agent_display

    def extend_display(self) -> None:
        if not self.offscreen:
            if self.config["screen_width"] > self.config["screen_height"]:
                self.screen = pygame.display.set_mode(
                    (self.config["screen_width"], 2 * self.config["screen_height"])
                )
            else:
                self.screen = pygame.display.set_mode(
                    (2 * self.config["screen_width"], self.config["screen_height"])
                )
        self.agent_surface = pygame.Surface(
            (self.config["screen_width"], self.config["screen_height"])
        )

    def set_agent_action_sequence(self, actions: list[Action]) -> None:
        """
        设置智能体选择的动作序列，以便将其显示出来。

        :param actions: 符合环境动作空间规范的动作列表
        """
        if isinstance(self.env.action_type, DiscreteMetaAction):
            assert self.env.action_type.actions is not None
            mapped_actions = [
                self.env.action_type.actions[a] for a in cast(list[int], actions)
            ]
        elif isinstance(self.env.action_type, ContinuousAction):
            mapped_actions = [
                self.env.action_type.get_action(a) for a in cast(list[NDArray], actions)
            ]
        else:
            mapped_actions = actions
        if len(mapped_actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(
                mapped_actions,
                1 / self.env.config["policy_frequency"],
                1 / 3 / self.env.config["policy_frequency"],
                1 / self.env.config["simulation_frequency"],
            )

    def handle_events(self) -> None:
        """处理 pygame 事件，并将其转交给显示组件和环境中的车辆。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """在 pygame 窗口中显示道路与车辆。"""
        if not self.enabled:
            return

        assert self.env.road is not None
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory, self.sim_surface, offscreen=self.offscreen
            )

        RoadGraphics.display_road_objects(
            self.env.road, self.sim_surface, offscreen=self.offscreen
        )

        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                assert self.agent_surface is not None
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(
                        self.agent_surface, (0, self.config["screen_height"])
                    )
                else:
                    self.screen.blit(
                        self.agent_surface, (self.config["screen_width"], 0)
                    )

        assert self.env.road is not None
        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen,
        )

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(
                self.sim_surface,
                str(self.directory / f"highway-env_{self.frame}.png"),
            )
            self.frame += 1

    def get_image(self) -> np.ndarray:
        """
        以 RGB 数组形式返回渲染图像。

        Gymnasium 采用 H × W × C（高、宽、通道）的排列方式。
        """
        surface = (
            self.screen
            if self.config["render_agent"] and not self.offscreen
            else self.sim_surface
        )
        data = pygame.surfarray.array3d(surface)  # 采用 W × H × C（宽、高、通道）的排列方式
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """显示窗口中心对应的世界坐标位置。"""
        if self.observer_vehicle:
            return self.observer_vehicle.position
        elif self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """关闭 pygame 窗口。"""
        pygame.quit()


class EventHandler:
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.Event) -> None:
        """
        将 pygame 键盘事件映射为控制决策。

        :param action_type: 定义车辆控制方式的 ActionType
        :param event: pygame 事件
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif type(action_type) is ContinuousAction:
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(
        cls, action_type: DiscreteMetaAction, event: pygame.event.Event
    ) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(
        cls, action_type: ContinuousAction, event: pygame.event.Event
    ) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)


class ObservationGraphics:
    LIDAR_COLOR = (0, 0, 0)  # 用于 display_grid 和 display_rays 的显示设置
    NAV_COLOR = (200, 200, 0)  # 导航箭头显示：display_navigation_arrow

    @classmethod
    def display(cls, obs: ObservationType, sim_surface):
        from highway_env.envs.common.observation import (
            DictObservation,
            LaneLidarObservation,
            LidarObservation,
            NavigationObservation,
        )

        if isinstance(obs, NavigationObservation):
            cls.display_navigation_arrow(obs, sim_surface)
        elif isinstance(obs, LaneLidarObservation):
            cls.display_rays(obs, sim_surface)
        elif isinstance(obs, LidarObservation):
            cls.display_grid(obs, sim_surface)
        elif isinstance(obs, DictObservation):
            for obs_type in obs.observation_types.values():
                cls.display(obs_type, sim_surface)

    @classmethod
    def display_grid(cls, lidar_observation: LidarObservation, surface):
        obs_origin = cast(NDArray, lidar_observation.origin)
        cells = lidar_observation.grid.shape[0]
        psi = np.repeat(
            -lidar_observation.angle / 2 + lidar_observation.angle * np.arange(cells),
            2,
        )
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(
            np.minimum(lidar_observation.grid[:, 0], lidar_observation.maximum_range), 2
        )
        points = [
            (
                surface.pos2pix(
                    obs_origin[0] + r[i] * np.cos(psi[i]),
                    obs_origin[1] + r[i] * np.sin(psi[i]),
                )
            )
            for i in range(np.size(psi))
        ]
        pygame.draw.lines(surface, ObservationGraphics.LIDAR_COLOR, True, points, 1)

    @classmethod
    def display_navigation_arrow(cls, nav_observation: NavigationObservation, surface):
        """
        绘制指向下一个路径点的线段。
        """
        origin = nav_observation.observer_vehicle.position

        pygame.draw.line(
            surface,
            ObservationGraphics.NAV_COLOR,
            surface.pos2pix(origin[0], origin[1]),
            surface.pos2pix(nav_observation.waypoint[0], nav_observation.waypoint[1]),
            1,
        )

    @classmethod
    def display_rays(cls, lanelidar_observation: LaneLidarObservation, surface):
        obs_origin = cast(NDArray, lanelidar_observation.origin)
        for index in range(lanelidar_observation.cells):  # [0]:
            angle = (
                index * lanelidar_observation.angle
                + lanelidar_observation.vehicle_heading
            )
            dist = lanelidar_observation.grid[index][0]

            world_x = obs_origin[0] + math.cos(angle) * dist
            world_y = obs_origin[1] + math.sin(angle) * dist
            origin = surface.pos2pix(
                obs_origin[0],
                obs_origin[1],
            )
            point = surface.pos2pix(world_x, world_y)
            pygame.draw.line(surface, ObservationGraphics.LIDAR_COLOR, origin, point, 1)
