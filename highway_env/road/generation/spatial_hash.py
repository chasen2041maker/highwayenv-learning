from collections import defaultdict
from itertools import chain
from typing import Any, Sequence

import numpy as np

from .engine.gen_utils import Lane


def point_to_gridpoint(point: np.ndarray, gridsize: int) -> tuple[int, int]:
    """
    将世界坐标转换为空间哈希使用的对应网格坐标。

    :param point: 世界位置
    :param gridsize: 网格单元的边长
    :return: 网格坐标元组
    """
    return tuple(np.floor(point / gridsize).astype(int))


def lanes_spatial_hash(
    lanes: list[Lane], gridsize: int = 100, use_boundaries: bool = True
) -> tuple[defaultdict[Any, set], defaultdict[Any, set]]:
    """
    将车道划分到不同网格中，以显著加快邻近检查。

    :param lanes: 车道列表
    :param gridsize: 网格单元的边长
    :param use_boundaries: 为 True 时使用边界点代替中心线点，默认为 True
    :return: **lane_to_grid**（车道索引到其占用网格点的映射），以及
        **grid_to_lane**（网格点到其中车道索引的映射）
    """

    lane_to_grid = defaultdict(set)
    grid_to_lanes = defaultdict(set)

    for lane_id, lane in enumerate(lanes):
        if use_boundaries:
            pts = chain(lane.left_points, lane.right_points)
        else:
            pts = lane.points

        last_gridpoint = None
        for point in pts:
            gridpoint = point_to_gridpoint(point, gridsize)
            lane_to_grid[lane_id].add(gridpoint)
            grid_to_lanes[gridpoint].add(lane_id)

            # 处理恰好沿对角线穿过的情况，
            # 此时可能跳过一个网格：
            if (
                last_gridpoint is not None
                and np.abs(gridpoint[0] - last_gridpoint[0]) == 1
                and np.abs(gridpoint[1] - last_gridpoint[1]) == 1
            ):
                gp1 = (gridpoint[0], last_gridpoint[1])
                gp2 = (last_gridpoint[0], gridpoint[1])
                lane_to_grid[lane_id].update((gp1, gp2))
                grid_to_lanes[gp1].add(lane_id)
                grid_to_lanes[gp2].add(lane_id)
            last_gridpoint = gridpoint

    return lane_to_grid, grid_to_lanes


gridhash_offsets = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def get_proximal_lanes_wrt_gridpoint(
    grid_to_lanes: defaultdict[Any, set],
    gridpoint: Sequence[int],
    extended: bool = False,
) -> set:
    """
    :param grid_to_lanes: 从网格点到其中车道索引的映射
    :param gridpoint: 网格坐标元组
    :param extended: 是否计入相邻网格中的车道
    :return: 附近车道索引的集合
    """
    proximal_lanes = set()
    for offset in gridhash_offsets if extended else [(0, 0)]:
        offset_gridpoint = (gridpoint[0] + offset[0], gridpoint[1] + offset[1])
        proximal_lanes.update(grid_to_lanes[offset_gridpoint])

    return proximal_lanes


def get_proximal_lanes_wrt_lane(
    lane_id: int,
    lane_to_grid: defaultdict[Any, set],
    grid_to_lanes: defaultdict[Any, set],
    extended: bool = False,
) -> set:
    """
    :param lane_id: 参考车道的索引
    :param lane_to_grid: 从车道到其占用网格点的映射
    :param grid_to_lanes: 从网格点到其中车道的映射
    :param extended: 是否计入相邻网格中的车道
    :return: 附近车道索引的集合
    """
    proximal_lanes = set()
    for gridpoint in lane_to_grid[lane_id]:
        proximal_lanes.update(
            get_proximal_lanes_wrt_gridpoint(
                grid_to_lanes, gridpoint, extended=extended
            )
        )

    proximal_lanes.discard(lane_id)

    return proximal_lanes
