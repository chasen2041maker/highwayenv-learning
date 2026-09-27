import numpy as np

from .gen_utils import (
    Endpoint,
    Lane,
    get_junction_pos,
    get_nodeset,
    get_radially_sorted_endpoints,
    wrap_with_tqdm,
)


def twist_optimize(
    lanes: list[Lane],
    iterations: int = 40,
    step: float = 0.00001,
    n: int = 3,
    lane_width: int = 10,
    disable_prints: bool = False,
) -> None:
    """
    使用梯度下降扭转、压缩和旋转路口处的车道端点，使它们正确朝向彼此。

    :param lanes: 车道列表
    :param iterations: 梯度下降的迭代次数
    :param step: 梯度下降步长，对结果影响很敏感
    :param n: 每个端点需要扭转的执行器或关节数量
    :param lane_width: 期望车道宽度
    :param disable_prints: 是否关闭进度和状态输出
    """

    rotate_optimize(lanes, n)

    r = 3
    nodeset = get_nodeset(lanes)

    for node in wrap_with_tqdm(
        sorted(nodeset), disabled=disable_prints, desc="Twisting Endpoints..."
    ):
        junction = get_radially_sorted_endpoints(lanes, node)

        for _ in range(iterations):
            for ep in junction:
                length = len(lanes[ep.id].points)
                if length > n:
                    step_twist_gradient(lanes, junction, ep, step=step, n=n, r=r)
                elif length > 2:
                    step_twist_gradient(lanes, junction, ep, step=step, n=2, r=r)

        squish_optimize(lanes, junction, r)

    prune_redundant_lanes(lanes, lane_width)


def step_twist_gradient(
    lanes: list[Lane], junction: list, ep, step=0.0002, n=3, r=3
) -> None:
    """
    扭转端点，使以下损失对扭转角 a 最小：
    (x(a) + r * cos(theta(a)) - x_t)^2 + (y(a) + r * sin(theta(a)) - y_t)^2

    其中：
    - (x_t, y_t) 是路口中心点。
    - (x(a), y(a)) 是施加角度 a 的扭转后端点的位置，用递归函数表示：
        - x(i, a) = x(i-1, a) + r_i * cos(theta_i + a)
        - x(0, a) 为锚点，x(n, a) 为端点。
        - r_i 为线段长度；theta_i 为该线段相对前一段方向的角度偏移。
    - theta(a) 是端点末端的绝对朝向。
    - r 是附着于端点末端的一段假想刚性延伸段的长度。
        - 希望延伸段的末端尽可能靠近 (x_t, y_t)。

    损失接近零时，端点自然指向路口中心。
    扭转过程可以保持车道曲线平滑。

    :param lanes: 车道列表
    :param junction: 构成路口的所有端点列表
    :param ep: 路口中的指定端点
    :param step: 梯度下降步长
    :param n: 需要扭转的执行器或关节数量
    :param r: 端点末端假想延伸段的长度
    """
    if len(junction) <= 1:
        return

    # 计算 (x(a), y(a)) 和 theta(a)
    x_a, y_a = ep.position(lanes)
    vx_a, vy_a = ep.vector(lanes)
    theta_a = np.atan2(vy_a, vx_a)

    # 计算 (x_t, y_t)
    x_t, y_t = get_junction_pos(lanes, junction, excluded_endpoint=ep)
    for endpoint in junction:
        if endpoint is not ep:
            vec = endpoint.vector(lanes)
            x_t += r * vec[0] / (len(junction) - 1)
            y_t += r * vec[1] / (len(junction) - 1)

    # 计算 x'(a, n) 和 y'(a, n)：
    #   x'(i) = -r_i * sin(theta_i) + x'(i-1)
    #   y'(i) = r_i * cos(theta_i) + y'(i-1)
    #   x'(0) = y'(0) = 0
    x_a_derivative = 0
    y_a_derivative = 0
    polar_sequence = get_polar_sequence(lanes, ep, n)
    for i in range(1, n + 1):
        c = polar_sequence[i]
        x_a_derivative += -c[1] * np.sin(c[0])
        y_a_derivative += c[1] * np.cos(c[0])

    # 计算 theta'(a)
    # （端点末端的朝向变化量等于
    # 执行器数量乘以施加的扭转角）
    theta_a_derivative = n

    # 计算损失梯度
    # L' = (x(a) + rcos(theta(a)) - x_t)(x'(a) - (rsin(theta(a)) * theta'(a)))
    #    + (y(a) + rsin(theta(a)) - y_t)(y'(a) + (rcos(theta(a)) * theta'(a)))
    loss_gradient = (x_a + r * np.cos(theta_a) - x_t) * (
        x_a_derivative - (r * np.sin(theta_a) * theta_a_derivative)
    ) + (y_a + r * np.sin(theta_a) - y_t) * (
        y_a_derivative + (r * np.cos(theta_a) * theta_a_derivative)
    )

    twist_endpoint(lanes, ep, -loss_gradient * step, n)


def twist_endpoint(lanes: list[Lane], ep: Endpoint, angle: float, n: int = 3) -> None:
    """
    将各线段视为旋转固定角度的执行器，以此扭转端点。

    :param lanes: 车道列表
    :param ep: 要扭转的端点
    :param angle: 扭转方向和大小
    :param n: 执行器数量
    """
    polar_coord_sequence = get_polar_sequence(lanes, ep, n)

    for i, (theta, r) in enumerate(polar_coord_sequence):
        if i == 0:
            continue
        polar_coord_sequence[i] = (theta + angle * i, r)

    for i in range(1, n + 1):
        index = i_to_index(lanes, ep, n, i)
        base_index = i_to_index(lanes, ep, n, i - 1)

        x_offset = np.cos(polar_coord_sequence[i][0]) * polar_coord_sequence[i][1]
        y_offset = np.sin(polar_coord_sequence[i][0]) * polar_coord_sequence[i][1]
        base_point = lanes[ep.id].points[base_index]

        lanes[ep.id].points[index] = base_point + np.array([x_offset, y_offset])


def get_polar_sequence(lanes: list[Lane], ep: Endpoint, n: int) -> list[tuple]:
    """
    将端点处最前或最后 n 条线段转换为一系列相对极坐标偏移。

    :param lanes: 车道列表
    :param ep: 端点
    :param n: 执行器数量
    """
    polar_coord_sequence = [(-999, -999)]  # 索引 0 对应的元素无效
    for i in range(1, n + 1):
        pos0 = lanes[ep.id].points[i_to_index(lanes, ep, n, i)]
        pos1 = lanes[ep.id].points[
            i_to_index(lanes, ep, n, i - 1)
        ]  # pos1 比 pos0 更靠近基点
        vec = pos0 - pos1

        r = np.linalg.norm(vec)
        theta = np.atan2(vec[1], vec[0])

        polar_coord_sequence.append((theta, r))

    return polar_coord_sequence


def i_to_index(lanes: list[Lane], ep: Endpoint, n: int, i: int) -> int:
    """
    将递归函数 x(i) 或 y(i) 中的 i 映射为点索引。

    - i = 0 表示根部或基点：
        - 对 'start' 端点，索引为 n。
        - 对 'end' 端点，索引为 len(points)-n-1。
    - i = n 表示末端点：
        - 对 'start' 端点，索引为 0。
        - 对 'end' 端点，索引为 len(points)-1。

    :param lanes: 车道列表
    :param ep: 端点
    :param n: 执行器数量
    :param i: 递归函数 x(i) 和 y(i) 中的索引
    """

    if ep.loc == "start":
        return n - i
    else:
        return i + len(lanes[ep.id].points) - n - 1


def rotate_optimize(lanes: list[Lane], n: int = 3) -> None:
    """
    车道太短、无法扭转时，直接旋转车道。

    :param lanes: 车道列表
    :param n: 扭转步骤中的执行器数量，也作为判断车道是否过短的阈值
    """
    for lane_id, lane in enumerate(lanes):
        if len(lane.points) <= n:
            # 计算车道期望的起点和终点

            start_junction = get_radially_sorted_endpoints(lanes, lane.start)
            if len(start_junction) > 1:
                start = get_junction_pos(
                    lanes,
                    start_junction,
                    excluded_endpoint=Endpoint(id=lane_id, loc="start"),
                )
            else:
                start = start_junction[0].position(lanes)

            end_junction = get_radially_sorted_endpoints(lanes, lane.end)
            if len(end_junction) > 1:
                end = get_junction_pos(
                    lanes,
                    end_junction,
                    excluded_endpoint=Endpoint(id=lane_id, loc="end"),
                )
            else:
                end = end_junction[0].position(lanes)

            # 在期望起点和终点之间直接插值，得到各个车道点
            for i in range(len(lane.points)):
                num_pts = len(lane.points)
                lane.points[i] = (end - start) * ((i + 1) / (num_pts + 1)) + start


def squish_optimize(lanes: list[Lane], junction: list[Endpoint], r: int) -> None:
    """
    通过移除点或压缩点间距来缩短端点，确保其不越过路口中心。

    :param lanes: 车道列表
    :param junction: 构成路口的所有端点列表
    :param r: 相对路口中心的期望距离偏移
    """
    if len(junction) <= 1:
        return

    for ep in junction:
        mid = get_junction_pos(lanes, junction, excluded_endpoint=ep)
        b = ep.vector_raw(lanes)

        for _ in range(
            5
        ):  # 重复执行，因为可能需要移除多个车道点
            # 才能正确对齐
            pos = ep.position(lanes)
            second_pos = lanes[ep.id].points[ep.second_point_index()]
            c = mid - second_pos

            if c @ c != 0:
                offset = r * c / np.linalg.norm(c)
            else:
                offset = r * b / np.linalg.norm(b)

            a = mid - offset - pos
            a1 = a @ b / (b @ b)

            if a1 < -1:  # 完全移除最后一个点
                if len(lanes[ep.id].points) > 2:
                    lanes[ep.id].points.pop(ep.point_index())
                    continue
            elif a1 < 0:  # 只调整最后一个点的位置
                new = (lanes[ep.id].points[ep.second_point_index()] + mid) * 0.5
                lanes[ep.id].points[ep.point_index()] = new
            break


def prune_redundant_lanes(lanes: list[Lane], lane_width: int) -> None:
    """
    移除起点和终点相同且相互重叠的车道。

    :param lanes: 车道列表
    :param lane_width: 期望车道宽度
    """

    duplicate_found = True
    while duplicate_found:
        duplicate_found = False
        for lane in lanes:
            for other_lane in lanes:
                if lane is not other_lane:
                    if (
                        lane.start == other_lane.start and lane.end == other_lane.end
                    ) or (
                        lane.end == other_lane.start and lane.start == other_lane.end
                    ):
                        number_of_points_too_close = 0
                        for point in lane.points:
                            closest_distance = None
                            for point2 in other_lane.points:
                                dist = np.linalg.norm(point - point2)
                                if closest_distance is None or dist < closest_distance:
                                    closest_distance = dist
                            if (
                                closest_distance is not None
                                and closest_distance < lane_width * 1.5
                            ):
                                number_of_points_too_close += 1

                        if number_of_points_too_close > 2:
                            duplicate_found = True
                            lanes[:] = [l for l in lanes if l is not other_lane]
                        break
            if duplicate_found:
                break
