import numpy as np

from .gen_utils import Lane, find_line_intersection, get_radially_sorted_endpoints


def generate_lane_boundaries(lanes: list[Lane], lane_width: int) -> None:
    """
    生成横向的左右边界点。

    :param lanes: 车道列表
    :param lane_width: 期望车道宽度
    """
    for lane in lanes:
        lane.left_points = []
        lane.right_points = []

        for i, point in enumerate(lane.points):
            longitudal_offsets = []
            if i != 0:
                longitudal_offsets.append(lane.points[i - 1] - point)
            if i != len(lane.points) - 1:
                longitudal_offsets.append(lane.points[i + 1] - point)
            # lane.points 的长度不能为 1

            backwards_offset = longitudal_offsets[0].copy()
            if i != 0:
                backwards_offset *= -1

            lat = np.zeros(2)
            for _ in range(2):
                if len(longitudal_offsets) == 1:
                    lat = np.array(
                        [-longitudal_offsets[0][1], longitudal_offsets[0][0]]
                    )
                    break
                elif len(longitudal_offsets) == 2:
                    longitudal_offsets[0] /= np.linalg.norm(longitudal_offsets[0])
                    longitudal_offsets[1] /= np.linalg.norm(longitudal_offsets[1])

                    lat = (longitudal_offsets[0] + longitudal_offsets[1]) / 2.0

                    if np.linalg.norm(lat) == 0:
                        longitudal_offsets.pop()
                        continue
                    break

            mag = np.linalg.norm(lat)
            lat *= (lane_width / 2) / mag

            if (lat[0] * backwards_offset[1] - lat[1] * backwards_offset[0]) < 0:
                lane.right_points.append(point + lat)
                lane.left_points.append(point - lat)
            else:
                lane.right_points.append(point - lat)
                lane.left_points.append(point + lat)


def correct_junction_boundaries(lanes: list[Lane], node: str) -> None:
    """
    对齐路口中按角度相邻车道之间的拐角，使两条边界相交于同一个点。

    :param lanes: 车道列表
    :param node: 路口的字符串标识
    """
    junction = get_radially_sorted_endpoints(lanes, node)
    if len(junction) <= 1:
        return

    # 规则：当前车道左侧应与左邻车道右侧连接
    # 右侧相邻车道：索引加一
    # 左侧相邻车道：索引减一
    # 车道朝向不同时，左右的含义会交换
    for epID, ep in enumerate(junction):
        other_ep = junction[epID - 1]

        self_side = "right_points"
        other_side = "left_points"
        if ep.loc == "start":
            self_side = "left_points"
        if other_ep.loc == "start":
            other_side = "right_points"

        self_side_list = getattr(lanes[ep.id], self_side)
        other_side_list = getattr(lanes[other_ep.id], other_side)

        # 裁剪边界点，直到沿各自前进方向观察时，
        # 两个点都位于对方的后方
        while True:
            pos = self_side_list[ep.point_index()]
            dir = ep.vector(lanes)

            other_pos = other_side_list[other_ep.point_index()]
            other_dir = other_ep.vector(lanes)

            vecToOther = other_pos - pos
            if (
                vecToOther @ dir > 0
                or -(vecToOther @ other_dir) > 0
                or len(self_side_list) <= 3
                or len(other_side_list) <= 3
            ):
                break

            self_side_list.pop(ep.point_index())
            other_side_list.pop(other_ep.point_index())

        # 计算新的公共交点
        new_pos = find_line_intersection(pos, dir, other_pos, other_dir)

        # 如果新点不在原来两个点之间，
        # 则改用这两个点的简单平均位置
        b = other_pos - pos
        a = new_pos - pos

        if b @ b != 0:
            a1 = (a @ b) / (b @ b)
            if a1 <= 0 or a1 >= 1:
                new_pos = (pos + other_pos) / 2

        self_side_list[ep.point_index()] = new_pos
        other_side_list[other_ep.point_index()] = new_pos


def seal_dead_end(lanes: list[Lane], node: str) -> None:
    """
    添加一个额外的右边界点，封闭死胡同路口。

    :param lanes: 车道列表
    :param node: 路口的字符串标识
    """
    # 死胡同是仅有一个端点的路口
    junction = get_radially_sorted_endpoints(lanes, node)
    if len(junction) != 1:
        return

    ep = junction[0]
    lane = lanes[ep.id]

    if ep.loc == "start":
        lane.right_points.insert(0, lane.left_points[0])
    else:
        lane.right_points.append(lane.left_points[-1])

    # 缩短中心点的位置，使其
    # 不会接触新添加的边界线段
    lane.points[ep.point_index()] = (
        lane.points[ep.point_index()] + lane.points[ep.second_point_index()]
    ) / 2
