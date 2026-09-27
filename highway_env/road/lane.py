from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import IntEnum

import numpy as np

from highway_env import utils
from highway_env.road.spline import LinearSpline2D
from highway_env.utils import Vector, class_from_path, get_class_path, wrap_to_pi


class AbstractLane:
    """道路上的车道，由其中心曲线描述。"""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: list[LineType]
    speed_limit: float | int

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        将车道局部坐标转换为世界位置。

        :param longitudinal: 车道纵向坐标，单位为米
        :param lateral: 车道横向坐标，单位为米
        :return: 对应的世界位置，单位为米
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        """
        将世界位置转换为车道局部坐标。

        :param position: 世界位置，单位为米
        :return: 车道的（纵向、横向）坐标，单位为米
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        获取给定纵向位置处的车道朝向。

        :param longitudinal: 车道纵向坐标，单位为米
        :return: 车道朝向角，单位为弧度
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        获取给定纵向位置处的车道宽度。

        :param longitudinal: 车道纵向坐标，单位为米
        :return: 车道宽度，单位为米
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict):
        """
        根据配置创建车道实例。

        :param config: 包含车道参数的 JSON 字典
        """
        raise NotImplementedError()

    @abstractmethod
    def to_config(self) -> dict:
        """
        将车道参数写入可序列化为 JSON 的字典。

        :return: 车道参数字典
        """
        raise NotImplementedError()

    def on_lane(
        self,
        position: np.ndarray,
        longitudinal: float = None,
        lateral: float = None,
        margin: float = 0,
    ) -> bool:
        """
        判断给定世界位置是否位于车道上。

        :param position: 世界位置，单位为米
        :param longitudinal: 可选，已知时传入对应的车道纵向坐标，单位为米
        :param lateral: 可选，已知时传入对应的车道横向坐标，单位为米
        :param margin: 可选，在车道宽度两侧额外添加的余量
        :return: 该位置是否在车道上
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = (
            np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin
            and -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        判断从给定世界位置能否到达该车道。

        :param position: 世界位置，单位为米
        :return: 该车道是否可达
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = (
            np.abs(lateral) <= 2 * self.width_at(longitudinal)
            and 0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_close

    def after_end(
        self, position: np.ndarray, longitudinal: float = None, lateral: float = None
    ) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: np.ndarray):
        """计算某位置到车道的 L1 距离，单位为米。"""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(
        self,
        position: np.ndarray,
        heading: float | None,
        heading_weight: float = 1.0,
    ):
        """综合位置和朝向，计算到车道的加权距离。"""
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(self.local_angle(heading, s))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

    def local_angle(self, heading: float, long_offset: float):
        """计算相对车道朝向的未归一化角度。"""
        return wrap_to_pi(heading - self.heading_at(long_offset))


class LineType(IntEnum):
    """车道边线类型。"""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstractLane):
    """沿直线延伸的车道。"""

    def __init__(
        self,
        start: Vector,
        end: Vector,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        """
        创建直线车道。

        :param start: 车道起点，单位为米
        :param end: 车道终点，单位为米
        :param width: 车道宽度，单位为米
        :param line_types: 车道两侧的边线类型
        :param forbidden: 是否禁止变道进入该车道
        :param priority: 车道优先级，用于决定通行权
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.heading = np.arctan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return (
            self.start
            + longitudinal * self.direction
            + lateral * self.direction_lateral
        )

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "start": _to_serializable(self.start),
                "end": _to_serializable(self.end),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class SineLane(StraightLane):
    """正弦曲线车道。"""

    def __init__(
        self,
        start: Vector,
        end: Vector,
        amplitude: float,
        pulsation: float,
        phase: float,
        width: float = StraightLane.DEFAULT_WIDTH,
        line_types: list[LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        """
        创建正弦曲线车道。

        :param start: 车道起点，单位为米
        :param end: 车道终点，单位为米
        :param amplitude: 车道摆动的振幅，单位为米
        :param pulsation: 车道的空间角频率，单位为弧度/米
        :param phase: 车道初始相位，单位为弧度
        """
        super().__init__(
            start, end, width, line_types, forbidden, speed_limit, priority
        )
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return super().position(
            longitudinal,
            lateral
            + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase),
        )

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
            self.amplitude
            * self.pulsation
            * np.cos(self.pulsation * longitudinal + self.phase)
        )

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(
            self.pulsation * longitudinal + self.phase
        )

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        config.update(
            {
                "class_path": get_class_path(self.__class__),
            }
        )
        config["config"].update(
            {
                "amplitude": self.amplitude,
                "pulsation": self.pulsation,
                "phase": self.phase,
            }
        )
        return config


class CircularLane(AbstractLane):
    """沿圆弧延伸的车道。"""

    def __init__(
        self,
        center: Vector,
        radius: float,
        start_phase: float,
        end_phase: float,
        clockwise: bool = True,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: list[LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.clockwise = clockwise
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = radius * (end_phase - start_phase) * self.direction
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction) * np.array(
            [np.cos(phi), np.sin(phi)]
        )

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        lateral = self.direction * (self.radius - r)
        return longitudinal, lateral

    @classmethod
    def from_config(cls, config: dict):
        config["center"] = np.array(config["center"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "center": _to_serializable(self.center),
                "radius": self.radius,
                "start_phase": self.start_phase,
                "end_phase": self.end_phase,
                "clockwise": self.clockwise,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLaneFixedWidth(AbstractLane):
    """
    由一组点定义、使用二维 Hermite 多项式近似的固定宽度车道。
    """

    def __init__(
        self,
        lane_points: list[tuple[float, float]],
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        self.curve = LinearSpline2D(lane_points)
        self.length = self.curve.length
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.speed_limit = speed_limit
        self.priority = priority

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        x, y = self.curve(longitudinal)
        yaw = self.heading_at(longitudinal)
        return np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        lon, lat = self.curve.cartesian_to_frenet(position)
        return lon, lat

    def heading_at(self, longitudinal: float) -> float:
        dx, dy = self.curve.get_dx_dy(longitudinal)
        return np.arctan2(dy, dx)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "lane_points": _to_serializable(
                    [_to_serializable(p.position) for p in self.curve.poses]
                ),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLane(PolyLaneFixedWidth):
    """
    由一组点定义、使用二维 Hermite 多项式近似的车道。
    """

    def __init__(
        self,
        lane_points: list[np.ndarray[(2,), np.floating]],
        left_boundary_points: list[np.ndarray[(2,), np.floating]],
        right_boundary_points: list[np.ndarray[(2,), np.floating]],
        line_types: tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ):
        super().__init__(
            lane_points=lane_points,
            line_types=line_types,
            forbidden=forbidden,
            speed_limit=speed_limit,
            priority=priority,
        )
        self.right_boundary_points = right_boundary_points
        self.left_boundary_points = left_boundary_points
        self.right_boundary = LinearSpline2D(right_boundary_points)
        self.left_boundary = LinearSpline2D(left_boundary_points)
        self._init_width()

    def width_at(self, longitudinal: float) -> float:
        if longitudinal < 0:
            return self.width_samples[0]
        elif longitudinal > len(self.width_samples) - 1:
            return self.width_samples[-1]
        else:
            return self.width_samples[int(longitudinal)]

    def _width_at_s(self, longitudinal: float) -> float:
        """
        在给定 s 值处，根据中心线到两侧边界的最短距离计算宽度，以补偿边界线的凹陷。
        """
        center_x, center_y = self.position(longitudinal, 0)
        right_x, right_y = self.right_boundary(
            self.right_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )
        left_x, left_y = self.left_boundary(
            self.left_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )

        dist_to_center_right = np.linalg.norm(
            np.array([right_x, right_y]) - np.array([center_x, center_y])
        )
        dist_to_center_left = np.linalg.norm(
            np.array([left_x, left_y]) - np.array([center_x, center_y])
        )

        return max(
            min(dist_to_center_right, dist_to_center_left) * 2,
            AbstractLane.DEFAULT_WIDTH,
        )

    def _init_width(self):
        """
        按约 1 米间距预先采样车道宽度，减少运行时计算。
        假设车道宽度在 1～2 米内不会明显变化。
        使用 NumPy 的 linspace 确保采样包含最小和最大的 s 值。
        """
        s_samples = np.linspace(
            0,
            self.curve.length,
            num=int(np.ceil(self.curve.length)) + 1,
        )
        self.width_samples = [self._width_at_s(s) for s in s_samples]

    def to_config(self) -> dict:
        config = super().to_config()

        ordered_boundary_points = _to_serializable(
            [_to_serializable(p.position) for p in reversed(self.left_boundary.poses)]
        )
        ordered_boundary_points += _to_serializable(
            [_to_serializable(p.position) for p in self.right_boundary.poses]
        )

        config["class_name"] = self.__class__.__name__
        config["config"]["ordered_boundary_points"] = ordered_boundary_points
        del config["config"]["width"]

        return config


def _to_serializable(arg: np.ndarray | list) -> list:
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def lane_from_config(cfg: dict) -> AbstractLane:
    return class_from_path(cfg["class_path"])(**cfg["config"])
