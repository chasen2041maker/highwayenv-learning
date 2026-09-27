from collections import defaultdict
from typing import cast

import numpy as np

from ..spatial_hash import get_proximal_lanes_wrt_lane, lanes_spatial_hash
from .gen_utils import (
    Endpoint,
    Lane,
    do_line_segments_intersect,
    line_intersection_t,
    wrap_with_tqdm,
)


def rectify_map(
    lanes: list[Lane],
    merge_radius: int,
    forward_speed: int,
    disable_prints: bool = False,
) -> None:
    """
    确保邻近端点使用相同的字符串标识，并正确合并相交的车道路径。
    同时移除有缺陷的车道。

    :param lanes: 车道列表
    :param merge_radius: 两个端点合并的距离阈值
    :param forward_speed: 群体生成过程中智能体的速度
    :param disable_prints: 是否关闭进度和状态输出
    """
    rectify_short_lanes(lanes)
    conjoined_nodes = combine_nodes(
        lanes, merge_radius, mark=True, disable_prints=disable_prints
    )
    split_lanes(
        lanes,
        cast(set[str], conjoined_nodes),
        merge_radius=merge_radius,
        forward_speed=forward_speed,
        disable_prints=disable_prints,
    )
    rectify_short_lanes(lanes)  # 再次执行
    combine_nodes(lanes, merge_radius, disable_prints=disable_prints)
    remove_identical_reference_lanes(lanes)
    prune_intersecting_lanes(lanes, disable_prints=disable_prints)


def rectify_short_lanes(lanes: list[Lane]) -> None:
    """
    确保所有车道至少包含 3 个点。

    :param lanes: 车道列表
    """
    lanes_to_remove = []
    for lane in lanes:
        if len(lane.points) <= 1:
            lanes_to_remove.append(lane)
        elif len(lane.points) == 2:
            a = lane.points[0]
            b = lane.points[1]
            lane.points.insert(1, (a + b) / 2)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def combine_nodes(
    lanes: list[Lane],
    merge_radius: int = 20,
    mark: bool = False,
    disable_prints: bool = False,
) -> None | set[str]:
    """
    将邻近节点合并成使用同一标识的逻辑路口。

    :param lanes: 车道列表
    :param merge_radius: 两个端点合并的距离阈值
    :param mark: 若为 True，不修改已有节点，只记录哪些节点将被合并。
        split_lanes 需要 mark=True 时得到的信息；实际的节点合并（mark=False）
        必须在 split_lanes 之后执行。
    :param disable_prints: 是否关闭进度和状态输出
    :return: mark 为 True 时，返回邻近其他节点的节点集合；否则返回 None
    """

    lane_to_grid, grid_to_lanes = lanes_spatial_hash(
        lanes, gridsize=max(merge_radius, 50), use_boundaries=False
    )

    if mark:
        conjoined_nodes = set()
    else:
        # node_power 用来记录
        # 某个字符串标识已经传播到
        # 其他邻近节点的次数
        node_power = defaultdict(int)

    for lane_id, lane in enumerate(
        wrap_with_tqdm(lanes, disabled=disable_prints, desc="Merging nodes")
    ):
        proximal_lanes = get_proximal_lanes_wrt_lane(
            lane_id, lane_to_grid, grid_to_lanes, extended=True
        )
        for other_id in sorted(proximal_lanes):
            other_lane = lanes[other_id]
            # loc 是 location 的缩写，表示车道的哪一端
            for loc in ["start", "end"]:
                for other_loc in ["start", "end"]:
                    p0 = lane.points[Endpoint.l_to_i[loc]]
                    p1 = other_lane.points[Endpoint.l_to_i[other_loc]]
                    if np.linalg.norm(p0 - p1) < merge_radius:
                        lane_loc_id = getattr(lane, loc)
                        other_lane_loc_id = getattr(other_lane, other_loc)
                        if mark and lane_loc_id not in conjoined_nodes:
                            # 首先需要确保
                            # 这两个邻近节点之间没有车道穿过。
                            # 为此，检查是否存在
                            # 某条车道的线段
                            # 与这两个邻近节点
                            # 之间的连线相交。
                            obstruction_found = False
                            for foreign_id in sorted(proximal_lanes):
                                foreign_lane = lanes[foreign_id]
                                pos_pairs = zip(
                                    foreign_lane.points,
                                    foreign_lane.points[1:],
                                )
                                for fp0, fp1 in pos_pairs:
                                    t_a, t_b = line_intersection_t(
                                        p0, p1 - p0, fp0, fp1 - fp0
                                    )
                                    if (
                                        t_a > 0.01
                                        and t_a < 0.99
                                        and t_b > 0.01
                                        and t_b < 0.99
                                    ):
                                        obstruction_found = True
                                        break

                            # 如果发现相交线段，说明
                            # 两个节点被道路分隔，
                            # 不能视为同一个路口。
                            if not obstruction_found:
                                conjoined_nodes.add(lane_loc_id)
                                conjoined_nodes.add(other_lane_loc_id)

                        elif not mark:
                            # node_power 较高的字符串标识
                            # 会覆盖 node_power 较低的节点标识。
                            # 这种强者更强的机制可以防止
                            # 同一个路口出现两个
                            # 相互冲突的字符串标识。

                            if node_power[other_lane_loc_id] > node_power[lane_loc_id]:
                                setattr(lane, loc, other_lane_loc_id)
                                node_power[other_lane_loc_id] += 1
                            else:
                                setattr(other_lane, other_loc, lane_loc_id)
                                node_power[lane_loc_id] += 1

    if mark:
        return conjoined_nodes


def split_lanes(
    lanes: list[Lane],
    conjoined_nodes: set[str],
    merge_radius: int,
    forward_speed: int,
    disable_prints: bool = False,
) -> None:
    """
    当一条车道接入另一条车道时，在它们之间创建路口。

    :param lanes: 车道列表
    :param conjoined_nodes: 邻近其他节点的节点集合
    :param merge_radius: 端点与另一条车道合并的距离阈值
    :param forward_speed: 群体生成过程中智能体的速度
    :param disable_prints: 是否关闭进度和状态输出
    """
    cutoff_length = np.ceil(merge_radius * 2.0 / forward_speed)

    for lane in wrap_with_tqdm(
        lanes,
        disabled=disable_prints,
        desc="Creating intersections between proximal lanes",
    ):
        if len(lane.points) == 0:
            continue
        for loc in ["start", "end"]:
            if getattr(lane, loc) in conjoined_nodes:
                continue
            loc_pos = lane.points[Endpoint.l_to_i[loc]]

            for other_lane in lanes:
                found_index = -1
                closest_dist = None
                for i, pos in enumerate(other_lane.points):
                    dist = np.linalg.norm(pos - loc_pos)
                    if (
                        lane is not other_lane
                        or (i > cutoff_length and i < len(lane.points) - cutoff_length)
                    ) and (closest_dist is None or dist < closest_dist):
                        found_index = i
                        closest_dist = dist

                if closest_dist is not None and closest_dist < merge_radius:
                    if found_index < 2:
                        found_index = 2
                    if found_index > len(other_lane.points) - 2:
                        found_index = len(other_lane.points) - 2

                    lane_loc_id = getattr(lane, loc)
                    # 'old' 表示智能体历史路径中较早的部分
                    # 即点序列中索引较小的那一部分
                    older_half = other_lane.points[:found_index]
                    old_start = other_lane.start
                    other_lane.points = other_lane.points[found_index:]
                    other_lane.start = lane_loc_id

                    new_lane = Lane(start=old_start, end=lane_loc_id, points=older_half)
                    lanes.append(new_lane)
                    conjoined_nodes.add(lane_loc_id)

                    break


def remove_identical_reference_lanes(lanes: list[Lane]) -> None:
    """
    移除起点和终点位置相同的车道。

    :param lanes: 车道列表
    """
    lanes_to_remove = []
    for lane in lanes:
        if lane.start == lane.end:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def prune_intersecting_lanes(lanes: list[Lane], disable_prints: bool = False) -> None:
    """
    删除彼此交叉的车道。

    :param lanes: 车道列表
    :param disable_prints: 是否关闭进度和状态输出
    """
    lane_to_grid, grid_to_lanes = lanes_spatial_hash(
        lanes, gridsize=50, use_boundaries=False
    )

    lanes_to_remove = []
    for lane_id, lane in enumerate(
        wrap_with_tqdm(
            lanes,
            disabled=disable_prints,
            desc="Pruning Intersecting Lanes...",
        )
    ):
        proximal_lanes = get_proximal_lanes_wrt_lane(
            lane_id, lane_to_grid, grid_to_lanes
        )
        collision_detected = False
        for other_id in sorted(proximal_lanes):
            if lane_id < other_id:
                other_lane = lanes[other_id]
                pairs = zip(lane.points, lane.points[1:])
                for p0, p1 in pairs:
                    other_pairs = zip(other_lane.points, other_lane.points[1:])
                    for op0, op1 in other_pairs:
                        if do_line_segments_intersect(p0, p1, op0, op1):
                            collision_detected = True
                            break
                if collision_detected:
                    break
        if collision_detected:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]
