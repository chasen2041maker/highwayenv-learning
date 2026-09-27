from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from highway_env import utils


if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
    from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]


class RoadObject(ABC):
    """
    道路上物体的通用接口。

    目前假设所有物体均为矩形。
    """

    LENGTH: float = 2  # 物体长度，单位为米。
    WIDTH: float = 2  # 物体宽度，单位为米。

    @abstractmethod
    def __init__(
        self,
        road: Road,
        position: Sequence[float] | NDArray[np.floating],
        heading: float = 0,
        speed: float = 0,
    ):
        """
        :param road: 放置物体的道路实例
        :param position: 物体在平面上的笛卡尔坐标位置
        :param heading: 相对于水平轴正方向的夹角
        :param speed: 物体在平面上的运动速度
        """
        self.road = road
        self.position = np.array(position, dtype=np.float64)
        self.heading = heading
        self.speed = speed
        self.lane_index = (
            self.road.network.get_closest_lane_index(self.position, self.heading)
            if self.road
            else None
        )
        self.lane = (
            self.road.network.get_lane(self.lane_index) if self.lane_index else None
        )

        # 允许与其他可碰撞物体发生碰撞。
        self.collidable = True

        # 碰撞会产生物理影响。
        self.solid = True

        # 若为 False，此物体不主动检查自身碰撞，
        # 但其他启用碰撞检查的物体仍可检测到与它的碰撞。
        self.check_collisions = True

        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)

    @classmethod
    def make_on_lane(
        cls,
        road: Road,
        lane_index: LaneIndex,
        longitudinal: float,
        speed: float | None = None,
    ) -> RoadObject:
        """
        在指定车道的给定纵向位置创建物体。

        :param road: 包含道路网络的道路对象
        :param lane_index: 物体所在车道的索引
        :param longitudinal: 沿车道的纵向位置
        :param speed: 初始速度，单位为 m/s
        :return: 位于指定位置的 RoadObject
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(
            road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed
        )

    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        """
        检查与另一辆车或物体的碰撞。

        :param other: 另一辆车或物体
        :param dt: 在速度恒定的假设下检查未来碰撞的时间间隔
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                if isinstance(other, Obstacle):
                    self.impact = transition
                elif isinstance(self, Obstacle):
                    other.impact = transition
                else:
                    self.impact = transition / 2
                    other.impact = -transition / 2
        if intersecting:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True

    def _is_colliding(self, other, dt):
        # 用球形包围范围进行快速预检查。
        if (
            np.linalg.norm(other.position - self.position)
            > (self.diagonal + other.diagonal) / 2 + self.speed * dt
        ):
            return (
                False,
                False,
                np.zeros(
                    2,
                ),
            )
        # 进行精确的矩形碰撞检查。
        return utils.are_polygons_intersecting(
            self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt
        )

    # 仅为兼容性而添加。
    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": 0.0,
            "vy": 0.0,
            "cos_h": np.cos(self.heading),
            "sin_h": np.sin(self.heading),
            "cos_d": 0.0,
            "sin_d": 0.0,
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array(
            [
                [-self.LENGTH / 2, -self.WIDTH / 2],
                [-self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, -self.WIDTH / 2],
            ]
        ).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def lane_distance_to(
        self, other: RoadObject, lane: AbstractLane | None = None
    ) -> float:
        """
        计算沿车道到另一物体的带符号距离。

        :param other: 另一物体
        :param lane: 车道
        :return: 到另一物体的距离，单位为米
        """
        if not other:
            return np.nan
        if not lane:
            assert self.lane is not None
            lane = self.lane
        return (
            lane.local_coordinates(other.position)[0]
            - lane.local_coordinates(self.position)[0]
        )

    def intersects_with_line(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """
        判断是否与线段相交。
        """
        line_polygon = np.stack([p0, p1])
        rect_polygon = self.polygon()
        displacement = np.zeros_like(line_polygon, shape=2)

        return utils.are_polygons_intersecting(
            rect_polygon, line_polygon, displacement, displacement
        )[0]

    @property
    def on_road(self) -> bool:
        """判断物体是否位于当前车道内，或已驶出道路。"""
        assert self.lane
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: RoadObject) -> float:
        return self.direction.dot(other.position - self.position)

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):
    """道路上的障碍物。"""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = True


class Landmark(RoadObject):
    """标记道路中必须到达的特定区域的地标。"""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = False
