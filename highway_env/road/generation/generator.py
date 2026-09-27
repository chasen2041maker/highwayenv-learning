import numpy as np

from .engine import (
    Lane,
    check_lanes_type_validity,
    correct_junction_boundaries,
    generate_lane_boundaries,
    generate_road_network_skeleton,
    get_invalid_lanes,
    get_nodeset,
    kill_lanes,
    rectify_map,
    remove_disjoint_clusters,
    seal_dead_end,
    twist_optimize,
)


def default_params() -> dict:
    """
    :return: 程序化道路生成的参数字典：
        - **target_num_endpoints**：要生成的端点数量。
        - **forward_speed**：每条车道线段的长度。
        - **age_of_maturity**：智能体可以复制或终止之前需要经过的时间步数。
        - **lane_width**：车道宽度，所有车道保持一致。
        - **perlin_variation_params**：以下空间变化属性的 Perlin 噪声范围：
            - **jitteriness**：智能体左右转向的不规则程度。
            - **max_turn_speed**：智能体的大致角速度。
            - **replication_chance**：道路产生分叉的倾向。
            - **spontaneous_death_chance**：道路形成死胡同的倾向。
        - **disable_prints**：是否关闭进度信息输出。
        - **seed**：内部随机数生成器的整数种子；``None`` 表示随机种子。
    """
    return {
        "target_num_endpoints": 2,
        "forward_speed": 10,
        "age_of_maturity": 4,
        "lane_width": 10,
        "perlin_variation_params": {
            "jitteriness": {"upper": 0.1, "lower": 0.0},
            "max_turn_speed": {"upper": 4.0, "lower": 0.01},
            "replication_chance": {"upper": 0.7, "lower": 0.0},
            "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
        },
        "disable_prints": True,
    }


def generate_random_lanes(
    rng: np.random.Generator, provided_params: dict | None = None
) -> list[Lane]:
    """
    通过程序化生成创建车道网络。

    :param rng: 随机数生成器
    :param provided_params: 可选的生成参数字典
    :return: 车道列表
    """
    params = default_params()
    if provided_params is not None:
        params.update(provided_params)

    merge_radius = params["forward_speed"] * 2
    prevent_replication_radius = params["forward_speed"] * params["age_of_maturity"]

    twist_iterations = params["forward_speed"] * 2
    twist_step = 0.0002 / params["forward_speed"]

    # 阶段 1：随机群体生成
    lanes = generate_road_network_skeleton(
        target_num_endpoints=max(2, params["target_num_endpoints"]),
        forward_speed=params["forward_speed"],
        merge_radius=merge_radius,
        prevent_replication_radius=prevent_replication_radius,
        age_of_maturity=params["age_of_maturity"],
        perlin_variation_params=params["perlin_variation_params"],
        rng=rng,
        disable_prints=params["disable_prints"],
    )

    # 阶段 2：校正
    rectify_map(
        lanes,
        merge_radius=merge_radius + params["forward_speed"],
        forward_speed=params["forward_speed"],
        disable_prints=params["disable_prints"],
    )

    # 阶段 3：优化
    twist_optimize(
        lanes,
        iterations=twist_iterations,
        step=twist_step,
        lane_width=params["lane_width"],
        disable_prints=params["disable_prints"],
    )

    # 阶段 4：创建边界
    generate_lane_boundaries(lanes, params["lane_width"])
    for node in sorted(get_nodeset(lanes)):
        correct_junction_boundaries(lanes, node)
        seal_dead_end(lanes, node)

    # 阶段 5：验证
    invalids = get_invalid_lanes(
        lanes, params["forward_speed"], rng=rng, disable_prints=params["disable_prints"]
    )
    if not params["disable_prints"]:
        print(f"Removing {len(invalids)} obstructed lanes")
    kill_lanes(lanes, invalids)
    remove_disjoint_clusters(lanes)

    assert check_lanes_type_validity(lanes)

    return lanes


def serialize_lanes(lanes: list[Lane]) -> list[dict]:
    """
    将 Lane 转换为可写入 JSON 的字典列表。
    """
    lanes_serialized = []

    for lane in lanes:
        lane_serialized = {
            "start": lane.start,
            "end": lane.end,
            "points": [],
            "left_points": [],
            "right_points": [],
        }
        for pt in lane.points:
            lane_serialized["points"].append((pt[0], pt[1]))
        for pt in lane.left_points:
            lane_serialized["left_points"].append((pt[0], pt[1]))
        for pt in lane.right_points:
            lane_serialized["right_points"].append((pt[0], pt[1]))

        lanes_serialized.append(lane_serialized)

    return lanes_serialized


def unserialize_lanes(lanes_serialized: list[dict]) -> list[Lane]:
    """
    将 Lane 字典列表转换为 Lane 对象列表。
    """
    lanes = []

    for lane_serialized in lanes_serialized:
        new_lane = Lane(start=lane_serialized["start"], end=lane_serialized["end"])

        for pt in lane_serialized["points"]:
            new_lane.points.append(np.array([pt[0], pt[1]]))

        if "left_points" in lane_serialized:
            for pt in lane_serialized["left_points"]:
                new_lane.left_points.append(np.array([pt[0], pt[1]]))
            for pt in lane_serialized["right_points"]:
                new_lane.right_points.append(np.array([pt[0], pt[1]]))

        lanes.append(new_lane)

    return lanes


def save_lanes_to_disk(filename: str, lanes: list[Lane]):
    """
    将车道列表直接保存为二进制 .npz 文件。
    """
    data = {}

    for i, lane in enumerate(lanes):
        data[f"lane_{i}_nodes"] = np.array([lane.start, lane.end])
        data[f"lane_{i}_points"] = np.asarray(lane.points)
        data[f"lane_{i}_left"] = np.asarray(lane.left_points)
        data[f"lane_{i}_right"] = np.asarray(lane.right_points)

    np.savez_compressed(filename, allow_pickle=True, **data)


def load_lanes_from_disk(filename: str) -> list[Lane]:
    """
    加载 npz 文件并重新构造 Lane 对象列表。
    """
    with np.load(filename) as data:
        assert len(data.keys()) % 4 == 0
        num_lanes = int(len(data.keys()) / 4)
        lanes = []

        for i in range(num_lanes):
            start, end = data[f"lane_{i}_nodes"]
            new_lane = Lane(start=start, end=end)
            new_lane.points = data[f"lane_{i}_points"]
            new_lane.left_points = data[f"lane_{i}_left"]
            new_lane.right_points = data[f"lane_{i}_right"]

            lanes.append(new_lane)

    return lanes
