from __future__ import annotations

import importlib
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


def finite_mdp(
    env: AbstractEnv, time_quantization: float = 1.0, horizon: float = 10.0
) -> object:
    """
    用碰撞时间（Time-To-Collision，TTC）表示状态。

    状态奖励由不同碰撞时间和车道上的占用网格定义。网格单元表示：
    假设所有已观察车辆（包括自车）保持速度，且其他车辆不变道，
    自车在指定时间处于指定车道时，与另一辆车发生碰撞的概率。

    例如，在三车道道路中，左侧车道有车且预计 5 秒后碰撞，对应网格为：
    [0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0]
    TTC 状态是该网格中的坐标 (lane, time)。

    若自车可以改变速度，则为各个可选速度增加网格层。

    最后将状态展平，以兼容 FiniteMDPEnv 环境。

    :param AbstractEnv env: 所属环境
    :param time_quantization: 状态表示的时间量化间隔，单位为秒
    :param horizon: 预测碰撞的时间范围，单位为秒
    """
    # 计算碰撞时间（TTC）网格。
    grid = compute_ttc_grid(env, time_quantization, horizon)

    # 计算当前状态。
    grid_state = (env.vehicle.speed_index, env.vehicle.lane_index[2], 0)
    state = np.ravel_multi_index(grid_state, grid.shape)

    # 计算状态转移函数。
    transition_model_with_grid = partial(transition_model, grid=grid)
    transition = np.fromfunction(
        transition_model_with_grid, grid.shape + (env.action_space.n,), dtype=int
    )
    transition = np.reshape(transition, (np.size(grid), env.action_space.n))

    # 计算奖励函数。
    v, l, t = grid.shape
    lanes = np.arange(l) / max(l - 1, 1)
    speeds = np.arange(v) / max(v - 1, 1)

    state_reward = (
        +env.config["collision_reward"] * grid
        + env.config["right_lane_reward"]
        * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t))
        + env.config["high_speed_reward"]
        * np.tile(speeds[:, np.newaxis, np.newaxis], (1, l, t))
    )

    state_reward = np.ravel(state_reward)
    action_reward = [
        env.config["lane_change_reward"],
        0,
        env.config["lane_change_reward"],
        0,
        0,
    ]
    reward = np.fromfunction(
        np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
        (np.size(state_reward), np.size(action_reward)),
        dtype=int,
    )

    # 计算终止状态。
    collision = grid == 1
    end_of_horizon = np.fromfunction(
        lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int
    )
    terminal = np.ravel(collision | end_of_horizon)

    # 创建新的有限马尔可夫决策过程（MDP）。
    try:
        module = importlib.import_module("finite_mdp.mdp")
        mdp = module.DeterministicMDP(transition, reward, terminal, state=state)
        mdp.original_shape = grid.shape
        return mdp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"The finite_mdp module is required for conversion. {e}"
        ) from e


def compute_ttc_grid(
    env: AbstractEnv,
    time_quantization: float,
    horizon: float,
    vehicle: Vehicle | None = None,
) -> np.ndarray:
    """
    对每个自车速度和车道，计算与车道内各车辆的预测碰撞时间网格。

    :param env: 所属环境
    :param time_quantization: 每个网格单元对应的时间间隔
    :param horizon: 网格覆盖的时间范围
    :param vehicle: 进行观察的车辆
    :return: 碰撞时间网格，维度顺序为 SPEED × LANES × TIME（速度、车道、时间）
    """
    vehicle = vehicle or env.vehicle
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    grid = np.zeros(
        (vehicle.target_speeds.size, len(road_lanes), int(horizon / time_quantization))
    )
    for speed_index in range(grid.shape[0]):
        ego_speed = vehicle.index_to_speed(speed_index)
        for other in env.road.vehicles:
            if (other is vehicle) or (ego_speed == other.speed):
                continue
            margin = other.LENGTH / 2 + vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]
            for m, cost in collision_points:
                distance = vehicle.lane_distance_to(other) + m
                other_projected_speed = other.speed * np.dot(
                    other.direction, vehicle.direction
                )
                time_to_collision = distance / utils.not_zero(
                    ego_speed - other_projected_speed
                )
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(
                    vehicle.lane_index, other.lane_index, route=vehicle.route, depth=3
                ):
                    # 同一道路，或车道数量相同的相连道路。
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(
                        env.road.network.all_side_lanes(vehicle.lane_index)
                    ):
                        lane = [other.lane_index[2]]
                    # 道路不同且车道数不同：未来所处车道不确定，因此考虑全部车道。
                    else:
                        lane = range(grid.shape[1])
                    # 将碰撞时间分别量化到相邻的上界和下界。
                    for time in [
                        int(time_to_collision / time_quantization),
                        int(np.ceil(time_to_collision / time_quantization)),
                    ]:
                        if 0 <= time < grid.shape[2]:
                            # TODO：检查车道编号越界，例如车辆车道编号超出当前道路的车道数量。
                            grid[speed_index, lane, time] = np.maximum(
                                grid[speed_index, lane, time], cost
                            )
    return grid


def transition_model(h: int, i: int, j: int, a: int, grid: np.ndarray) -> np.ndarray:
    """
    从网格中的一个位置确定性地转移到下一个位置。

    :param h: 速度索引
    :param i: 车道索引
    :param j: 时间索引
    :param a: 动作索引
    :param grid: TTC 网格，规定速度、车道、时间和动作的边界
    """
    # 默认采用保持动作（编号 1）对应的状态转移。
    next_state = clip_position(h, i, j + 1, grid)
    left = a == 0
    right = a == 2
    faster = (a == 3) & (j == 0)
    slower = (a == 4) & (j == 0)
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)
    return next_state


def clip_position(h: int, i: int, j: int, grid: np.ndarray) -> np.ndarray:
    """
    裁剪 TTC 网格中的位置，使其保持在有效范围内。

    :param h: 速度索引
    :param i: 车道索引
    :param j: 时间索引
    :param grid: TTC 网格
    :return: 裁剪后位置对应的展平索引
    """
    h = np.clip(h, 0, grid.shape[0] - 1)
    i = np.clip(i, 0, grid.shape[1] - 1)
    j = np.clip(j, 0, grid.shape[2] - 1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)
    return indexes
