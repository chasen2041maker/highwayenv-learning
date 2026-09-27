from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import pygame

from highway_env.road.lane import AbstractLane, LineType, PolyLane
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle


if TYPE_CHECKING:
    from highway_env.vehicle.objects import RoadObject

PositionType = Union[Tuple[float, float], np.ndarray]


class WorldSurface(pygame.Surface):
    """带有局部坐标系的 pygame 绘图表面，支持平移和缩放显示区域。"""

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(
        self, size: tuple[int, int], flags: object, surf: pygame.Surface
    ) -> None:
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length: float) -> int:
        """
        将距离（米）转换为像素数。

        :param length: 输入距离，单位为米
        :return: 对应的像素尺寸
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> tuple[int, int]:
        """
        将世界坐标（米）转换为绘图表面上的位置（像素）。

        :param x: 世界坐标 x，单位为米
        :param y: 世界坐标 y，单位为米
        :return: 对应的像素坐标
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> tuple[int, int]:
        """
        将世界位置（米）转换为绘图表面上的位置（像素）。

        :param vec: 世界位置，单位为米
        :return: 对应的像素坐标
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        判断某个位置在绘图表面中是否可见。

        :param vec: 位置
        :param margin: 可见性检查时在画面周围保留的边距
        :return: 该位置是否可见
        """
        x, y = self.vec2pix(vec)
        return (
            -margin < x < self.get_width() + margin
            and -margin < y < self.get_height() + margin
        )

    def move_display_window_to(self, position: PositionType) -> None:
        """
        设置显示区域的原点，使画面以给定世界位置为中心。

        :param position: 世界位置，单位为米
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling,
            ]
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        处理用于平移和缩放显示区域的 pygame 事件。

        :param event: pygame 事件
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position[0] += self.MOVING_FACTOR


class LaneGraphics:
    """车道的可视化绘制。"""

    # 参考 https://www.researchgate.net/figure/French-road-traffic-lane-description-and-specification_fig4_261170641
    STRIPE_SPACING: float = 4.33
    """虚线条纹之间的偏移距离，单位为米。"""

    STRIPE_LENGTH: float = 3
    """一条虚线条纹的长度，单位为米。"""

    STRIPE_WIDTH: float = 0.3
    """一条虚线条纹的宽度，单位为米。"""

    @classmethod
    def display(cls, lane: AbstractLane, surface: WorldSurface) -> None:
        """
        在绘图表面上显示车道。

        :param lane: 要显示的车道
        :param surface: pygame 绘图表面
        """

        # 对具有连续边界的 PolyLane 使用优化绘制流程
        if (
            isinstance(lane, PolyLane)
            and lane.line_types[0] == LineType.CONTINUOUS
            and lane.line_types[1] == LineType.CONTINUOUS
        ):
            thickness = max(surface.pix(cls.STRIPE_WIDTH), 1)

            left_pixels = [surface.vec2pix(pt) for pt in lane.left_boundary_points]
            right_pixels = [surface.vec2pix(pt) for pt in lane.right_boundary_points]

            pygame.draw.lines(surface, surface.WHITE, False, left_pixels, thickness)
            pygame.draw.lines(surface, surface.WHITE, False, right_pixels, thickness)

            return

        stripes_count = int(
            2
            * (surface.get_height() + surface.get_width())
            / (cls.STRIPE_SPACING * surface.scaling)
        )
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (
            int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2
        ) * cls.STRIPE_SPACING
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS_LINE:
                cls.continuous_line(lane, surface, stripes_count, s0, side)

    @classmethod
    def striped_line(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        在绘图表面上绘制车道一侧的虚线。

        :param lane: 车道
        :param surface: pygame 绘图表面
        :param stripes_count: 要绘制的条纹数量
        :param longitudinal: 第一条条纹的纵向位置，单位为米
        :param side: 绘制道路的哪一侧，0 为左侧，1 为右侧
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = (
            longitudinal
            + np.arange(stripes_count) * cls.STRIPE_SPACING
            + cls.STRIPE_LENGTH
        )
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        在绘图表面上绘制车道一侧的虚线。

        :param lane: 车道
        :param surface: pygame 绘图表面
        :param stripes_count: 要绘制的条纹数量
        :param longitudinal: 第一条条纹的纵向位置，单位为米
        :param side: 绘制道路的哪一侧，0 为左侧，1 为右侧
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = (
            longitudinal
            + np.arange(stripes_count) * cls.STRIPE_SPACING
            + cls.STRIPE_SPACING
        )
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        stripes_count: int,
        longitudinal: float,
        side: int,
    ) -> None:
        """
        在绘图表面上绘制车道一侧的实线。

        :param lane: 车道
        :param surface: pygame 绘图表面
        :param stripes_count: 如果画成虚线时所需的条纹数量
        :param longitudinal: 线段起点的纵向位置，单位为米
        :param side: 绘制道路的哪一侧，0 为左侧，1 为右侧
        """
        starts = [longitudinal + 0 * cls.STRIPE_SPACING]
        ends = [longitudinal + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        starts: list[float],
        ends: list[float],
        lats: list[float],
    ) -> None:
        """
        沿车道绘制一组条纹。

        :param lane: 车道
        :param surface: 用于绘制的表面
        :param starts: 各条纹起点的纵向位置列表，单位为米
        :param ends: 各条纹终点的纵向位置列表，单位为米
        :param lats: 各条纹的横向位置列表，单位为米
        """
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for k, _ in enumerate(starts):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(
                    surface,
                    surface.WHITE,
                    (surface.vec2pix(lane.position(starts[k], lats[k]))),
                    (surface.vec2pix(lane.position(ends[k], lats[k]))),
                    max(surface.pix(cls.STRIPE_WIDTH), 1),
                )

    @classmethod
    def draw_ground(
        cls,
        lane: AbstractLane,
        surface: WorldSurface,
        color: tuple[float],
        width: float,
        draw_surface: pygame.Surface = None,
    ) -> None:
        draw_surface = draw_surface or surface
        stripes_count = int(
            2
            * (surface.get_height() + surface.get_width())
            / (cls.STRIPE_SPACING * surface.scaling)
        )
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (
            int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2
        ) * cls.STRIPE_SPACING
        dots = []
        for side in range(2):
            longis = np.clip(
                s0 + np.arange(stripes_count) * cls.STRIPE_SPACING, 0, lane.length
            )
            lats = [2 * (side - 0.5) * width for _ in longis]
            new_dots = [
                surface.vec2pix(lane.position(longi, lat))
                for longi, lat in zip(longis, lats, strict=False)
            ]
            new_dots = reversed(new_dots) if side else new_dots
            dots.extend(new_dots)
        pygame.draw.polygon(draw_surface, color, dots, 0)


class RoadGraphics:
    """道路车道和车辆的可视化绘制。"""

    @staticmethod
    def display(road: Road, surface: WorldSurface) -> None:
        """
        在绘图表面上显示道路的车道。

        :param road: 要显示的道路
        :param surface: pygame 绘图表面
        """
        surface.fill(surface.GREY)
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for i, l in enumerate(road.network.graph[_from][_to]):
                    if (_from, _to, i) not in road.network.reversed_lane_indices:
                        LaneGraphics.display(l, surface)

    @staticmethod
    def display_traffic(
        road: Road,
        surface: WorldSurface,
        simulation_frequency: int = 15,
        offscreen: bool = False,
    ) -> None:
        """
        在绘图表面上显示道路车辆。

        :param road: 要显示的道路
        :param surface: pygame 绘图表面
        :param simulation_frequency: 仿真频率
        :param offscreen: 是否只渲染而不显示到屏幕
        """
        if road.record_history:
            for v in road.vehicles:
                VehicleGraphics.display_history(
                    v, surface, simulation=simulation_frequency, offscreen=offscreen
                )
        for v in road.vehicles:
            VehicleGraphics.display(v, surface, offscreen=offscreen)

    @staticmethod
    def display_road_objects(
        road: Road, surface: WorldSurface, offscreen: bool = False
    ) -> None:
        """
        在绘图表面上显示道路物体。

        :param road: 要显示的道路
        :param surface: pygame 绘图表面
        :param offscreen: 是否使用离屏渲染
        """
        for o in road.objects:
            RoadObjectGraphics.display(o, surface, offscreen=offscreen)


class RoadObjectGraphics:
    """道路物体的可视化绘制。"""

    YELLOW = (200, 200, 0)
    BLUE = (100, 200, 255)
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLACK = (60, 60, 60)
    DEFAULT_COLOR = YELLOW

    @classmethod
    def display(
        cls,
        object_: RoadObject,
        surface: WorldSurface,
        transparent: bool = False,
        offscreen: bool = False,
    ):
        """
        在 pygame 绘图表面上显示道路物体。

        物体表示为带有颜色、可以旋转的矩形。

        :param object_: 要绘制的车辆或物体
        :param surface: 用于绘制物体的表面
        :param transparent: 是否将物体绘制为略微透明
        :param offscreen: 是否使用离屏渲染
        """
        o = object_
        s = pygame.Surface(
            (surface.pix(o.LENGTH), surface.pix(o.LENGTH)), pygame.SRCALPHA
        )  # 逐像素透明度
        rect = (
            0,
            surface.pix(o.LENGTH / 2 - o.WIDTH / 2),
            surface.pix(o.LENGTH),
            surface.pix(o.WIDTH),
        )
        pygame.draw.rect(s, cls.get_color(o, transparent), rect, 0)
        pygame.draw.rect(s, cls.BLACK, rect, 1)
        if (
            not offscreen
        ):  # 离屏模式下 convert_alpha 会报错；TODO：解释原因。
            s = pygame.Surface.convert_alpha(s)
        h = o.heading if abs(o.heading) > 2 * np.pi / 180 else 0
        # 绕中心旋转
        position = surface.pos2pix(o.position[0], o.position[1])
        cls.blit_rotate(surface, s, position, np.rad2deg(-h))

    @staticmethod
    def blit_rotate(
        surf: pygame.Surface,
        image: pygame.Surface,
        pos: Vector,
        angle: float,
        origin_pos: Vector = None,
        show_rect: bool = False,
    ) -> None:
        """感谢 https://stackoverflow.com/a/54714144 提供的方法。"""
        # 计算旋转后图像的轴对齐包围框
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (
            min(box_rotate, key=lambda p: p[0])[0],
            min(box_rotate, key=lambda p: p[1])[1],
        )
        max_box = (
            max(box_rotate, key=lambda p: p[0])[0],
            max(box_rotate, key=lambda p: p[1])[1],
        )

        # 计算旋转中心的平移量
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # 计算旋转后图像左上角的原点
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0],
            pos[1] - origin_pos[1] - max_box[1] + pivot_move[1],
        )
        # 获取旋转后的图像
        rotated_image = pygame.transform.rotate(image, angle)
        # 旋转图像并绘制到表面
        surf.blit(rotated_image, origin)
        # 在图像周围绘制矩形
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def get_color(cls, object_: RoadObject, transparent: bool = False):
        color = cls.DEFAULT_COLOR

        if isinstance(object_, Obstacle):
            if object_.crashed:
                # 表示失败
                color = cls.RED
            else:
                color = cls.YELLOW
        elif isinstance(object_, Landmark):
            if object_.hit:
                # 表示成功
                color = cls.GREEN
            else:
                color = cls.BLUE

        if transparent:
            color = (color[0], color[1], color[2], 30)

        return color
