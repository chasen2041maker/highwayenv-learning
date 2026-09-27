from __future__ import annotations

import copy
import os
import warnings
from typing import TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import RecordVideo

from highway_env import utils
from highway_env.envs.common.action import Action, ActionType, action_factory
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.observation import ObservationType, observation_factory
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = TypeVar("Observation")


class ConnectedLaneNeighboursMixin:
    """
    在 v1.12 中引入的混入类，用于启用新的邻近车辆检测行为。

    参见 https://github.com/Farama-Foundation/HighwayEnv/pull/667
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(config, {"neighbour_vehicles_connected_lanes": True})
        return config


class AbstractEnv(gym.Env):
    """
    供多种道路驾驶任务使用的通用环境。

    环境包含道路、其他车辆，以及能够变道和改变速度的受控自车。
    动作空间固定；观察空间和奖励函数需要由具体环境实现定义。
    """

    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: RecordVideo | None
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    PERCEPTION_DISTANCE = 5.0 * Vehicle.MAX_SPEED
    """观察中车辆与自车之间允许的最大距离，单位为米。"""

    def __init__(
        self, config: dict | None = None, render_mode: str | None = None
    ) -> None:
        super().__init__()

        # 画面渲染
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.viewer = None
        self._record_video_wrapper = None
        self.enable_auto_render = False

        # 环境配置
        self.config = self.default_config()
        self.configure(config)

        # 场景
        self.road = None
        self.controlled_vehicles = []

        # 观察空间与动作空间
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # 运行状态
        self.time = 0  # 仿真时间
        self.steps = 0  # 已执行的动作数
        self.done = False

        self.reset()

    @property
    def vehicle(self) -> Vehicle:
        """第一辆受控车辆，也是默认受控车辆。"""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """设置唯一的受控车辆。"""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        环境的默认配置。

        具体环境可以重写此配置，也可以调用 configure() 进行修改。
        :return: 配置字典
        """
        return {
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 15,  # 频率，单位为 Hz（赫兹）
            "policy_frequency": 1,  # 频率，单位为 Hz（赫兹）
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # 单位：像素
            "screen_height": 150,  # 单位：像素
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": None,
            "manual_control": False,
            "real_time_rendering": False,
            "neighbour_vehicles_connected_lanes": False,
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

        if "OFFSCREEN_RENDERING" in os.environ:
            suggestion = (
                "rgb_array" if os.getenv("OFFSCREEN_RENDERING") == "1" else "human"
            )
            warnings.warn(
                f"\033[31mhighway_env.{self.__class__.__name__}:\033[0m "
                "The OFFSCREEN_RENDERING environment variable is deprecated "
                f'and ignored. Use render_mode="{suggestion}" instead.',
                DeprecationWarning,
                stacklevel=2,
            )

        if self.config["offscreen_rendering"] is None:
            self.config["offscreen_rendering"] = self.render_mode != "human"

    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = (
            self.config["simulation_frequency"]
            if self._record_video_wrapper
            else self.config["policy_frequency"]
        )
        self.metadata["render_fps"] = video_real_time_ratio * frames_freq

    def define_spaces(self) -> None:
        """
        根据配置设置观察和动作的类型及空间。
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        返回执行指定动作并到达当前状态后获得的奖励。

        :param action: 上一次执行的动作
        :return: 奖励
        """
        raise NotImplementedError

    def _rewards(self, action: Action) -> dict[str, float]:
        """
        返回包含多个目标的奖励向量。

        若实现此方法，应在 _reward() 中把各项奖励汇总为一个标量。
        这个向量本身只应放在 info 字典中返回。

        :param action: 上一次执行的动作
        :return: 格式为 {'reward_name': reward_value} 的字典
        """
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """
        检查当前状态是否为终止状态。

        :return: 当前状态是否终止
        """
        raise NotImplementedError

    def _is_truncated(self) -> bool:
        """
        检查是否应在当前步截断本回合。

        :return: 本回合是否被截断
        """
        raise NotImplementedError

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        """
        返回包含附加信息的字典。

        :param obs: 当前观察
        :param action: 当前动作
        :return: 附加信息字典
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:
        """
        将环境重置到初始配置。

        :param seed: 用于初始化环境伪随机数生成器的种子
        :param options: 可通过 `options["config"]` 指定环境配置
        :return: 重置后状态的观察
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # 第一次设置：根据动作空间确定受控车辆的类。
        self.time = self.steps = 0
        self.done = False
        self._reset()
        if self.road is not None:
            self.road.neighbour_vehicles_connected_lanes = self.config[
                "neighbour_vehicles_connected_lanes"
            ]
        self.define_spaces()  # 第二次设置：场景创建后，将观察和动作关联到车辆。
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _reset(self) -> None:
        """
        重置场景中的道路和车辆。

        具体环境必须重写此方法。
        """
        raise NotImplementedError()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """
        执行一个动作，并推进环境的仿真过程。

        自车执行该动作，道路上的其他车辆按各自默认行为运行；
        经过若干仿真步后，到达下一次决策时刻。

        :param action: 自车执行的动作
        :return: 元组 (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _simulate(self, action: Action | None = None) -> None:
        """在动作保持不变的情况下执行若干仿真步。"""
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            # 将动作转交给车辆。
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # 若已启动查看器，自动渲染中间的仿真帧。
            # 采用离屏渲染时忽略此步骤。
            if (
                frame < frames - 1
            ):  # 最后一帧仍按通常方式由 env.render() 渲染。
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self) -> np.ndarray | None:
        """
        渲染环境。

        如果尚无查看器，则先创建查看器，再用它绘制图像。
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return None
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == "rgb_array":
            return self.viewer.get_image()
        return None

    def close(self) -> None:
        """
        关闭环境。

        若环境查看器存在，也会将其关闭。
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> list[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()
        self._record_video_wrapper.frames_per_sec = self.metadata["render_fps"]

    def _automatic_rendering(self) -> None:
        """
        在一个动作尚未结束时，自动渲染其中的中间帧。

        这样可以渲染完整视频，而不只显示智能体作出决策时的离散画面。
        若设置了 RecordVideo 包装器，则用它采集中间帧。
        """
        if self.viewer is not None and self.enable_auto_render:
            if self._record_video_wrapper:
                self._record_video_wrapper._capture_frame()
            else:
                self.render()

    def simplify(self) -> AbstractEnv:
        """
        返回环境的简化副本，从道路上移除距离较远的车辆。

        这样可以降低策略计算量，同时保留最优动作集合。

        :return: 简化后的环境状态
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [
            state_copy.vehicle
        ] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE
        )

        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> AbstractEnv:
        """
        更改道路上其他车辆的类型。

        :param vehicle_class_path: 其他车辆行为类的导入路径，
            例如 "highway_env.vehicle.behavior.IDMVehicle"
        :return: 已修改其他车辆行为模型的新环境
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> AbstractEnv:
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # 具有车道偏好的车辆也会更不谨慎。
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> AbstractEnv:
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: tuple[str, object]) -> AbstractEnv:
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: tuple[str, tuple[object]]) -> AbstractEnv:
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behavior(self) -> AbstractEnv:
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1 / self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """进行深拷贝，但不复制环境查看器。"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ["viewer", "_record_video_wrapper"]:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result


class MultiAgentWrapper(Wrapper, RecordConstructorArgs):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        RecordConstructorArgs.__init__(self)

    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        reward = info["agents_rewards"]
        terminated = info["agents_terminated"]
        return obs, reward, terminated, truncated, info
