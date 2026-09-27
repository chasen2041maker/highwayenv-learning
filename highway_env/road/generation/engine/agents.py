from collections import defaultdict

import numpy as np
from noise import pnoise2

from ..spatial_hash import get_proximal_lanes_wrt_gridpoint, point_to_gridpoint
from .gen_utils import Lane, wrap_with_tqdm


def generate_road_network_skeleton(
    target_num_endpoints: int,
    forward_speed: int,
    merge_radius: int,
    prevent_replication_radius: int,
    age_of_maturity: int,
    perlin_variation_params: dict,
    rng: np.random.Generator,
    disable_prints: bool = False,
) -> list[Lane]:
    """
    使用一群会移动、复制的智能体，勾画道路连接结构的初稿。

    :param target_num_endpoints: 要生成的路口数量
    :param forward_speed: 智能体每个时间步移动的距离
    :param merge_radius: 智能体接近其他智能体或车道到此距离时终止并合并
    :param prevent_replication_radius: 智能体与其他智能体或车道至少相距此距离，才允许复制
    :param age_of_maturity: 智能体经过多少个时间步后可以复制
    :param perlin_variation_params: 决定转向和复制的速率，以及它们随空间位置的变化
    :param rng: 随机数生成器
    :param disable_prints: 是否关闭进度和状态输出
    :return: 车道列表
    """

    param_getter = PerlinVariation(perlin_variation_params, rng)

    spatial_hash_gridsize = max(prevent_replication_radius, 50)

    successful_generation = False
    MAX_GENERATION_ATTEMPTS = 10
    for generation_attempt in range(MAX_GENERATION_ATTEMPTS):
        lanes: list[Lane] = []
        grid_to_lanes = defaultdict(set)
        agents = [ConstructionAgent(start_location=0)]
        num_locations = 1
        simulation_step = 0
        with wrap_with_tqdm(
            total=target_num_endpoints,
            disabled=disable_prints,
            desc="Generating road network skeleton",
        ) as pbar:
            while num_locations < target_num_endpoints and len(agents) > 0:
                # 移动
                for agent in agents:
                    agent.step(
                        forward_speed=forward_speed, param_getter=param_getter, rng=rng
                    )

                # 复制与终止
                agents_to_add = []
                agents_to_remove = []
                for agent in agents:
                    if agent.is_too_young(age_of_maturity):
                        continue  # 阻止复制和终止

                    true_population = len(agents) - len(agents_to_remove)
                    prevent_replication, kill = agent.death_outcome(
                        agents=agents,
                        true_population=true_population,
                        age_of_maturity=age_of_maturity,
                        merge_radius=merge_radius,
                        prevent_replication_radius=prevent_replication_radius,
                        param_getter=param_getter,
                        rng=rng,
                        lanes=lanes,
                        grid_to_lanes=grid_to_lanes,
                        spatial_hash_gridsize=spatial_hash_gridsize,
                    )

                    if kill:
                        agents_to_remove.append(agent)
                        agent.end_location = str(num_locations)
                        num_locations += 1
                        continue

                    if not prevent_replication:
                        new_agents = agent.replicate(
                            param_getter=param_getter,
                            rng=rng,
                            true_population=true_population,
                            num_locations=num_locations,
                        )
                        if len(new_agents) > 0:
                            agents_to_add += new_agents
                            agents_to_remove.append(agent)
                            num_locations += 1

                # 将历史路径转换为车道
                for dying_agent in agents_to_remove:
                    new_lane = Lane(
                        start=dying_agent.start_location,
                        end=dying_agent.end_location,
                        points=dying_agent.history[:-1],
                    )

                    for point in new_lane.points:
                        gridpoint = point_to_gridpoint(point, spatial_hash_gridsize)
                        grid_to_lanes[gridpoint].add(len(lanes))

                    lanes.append(new_lane)
                    agents.remove(dying_agent)

                # 生成新的智能体
                for new_agent in agents_to_add:
                    agents.append(new_agent)

                simulation_step += 1
                pbar.n = min(num_locations, target_num_endpoints)
                pbar.refresh()

        # 将剩余智能体转换为车道
        for agent in agents:
            new_lane = Lane(
                start=agent.start_location,
                end=str(num_locations),
                points=agent.history,
            )
            lanes.append(new_lane)
            num_locations += 1

        if num_locations >= target_num_endpoints:
            successful_generation = True
            break
        elif not disable_prints:
            print(
                "Swarm generation fault"
                " [insufficient number of nodes reached];"
                " restarting"
            )

    if not successful_generation:
        raise TimeoutError(
            "Generation swarm keeps dying out;"
            " try increasing the replication rate"
            " or decreasing the turn speed."
        )

    return lanes


class ConstructionAgent:
    fork_angles = [-np.pi / 2, 0, np.pi / 2]

    fork_possibilities = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]  # {左转、直行、右转} 的可能子集

    @classmethod
    def random_fork_config(cls, rng):
        return ConstructionAgent.fork_possibilities[
            rng.integers(0, len(ConstructionAgent.fork_possibilities))
        ]

    def __init__(
        self,
        start_location,
        position=None,
        orientation=0,
        angular_velocity=0,
    ):
        if position is None:
            self.position = np.zeros(2)
        else:
            self.position = position

        self.start_location = str(start_location)
        self.end_location = str(-1)
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.history = []

    def step(self, forward_speed, param_getter, rng):
        jitteriness = param_getter.paramAt("jitteriness", self.position)
        max_turn_speed = param_getter.paramAt("max_turn_speed", self.position)

        turn_friction = 1 - jitteriness
        turn_acceleration_range = max_turn_speed * (1 / turn_friction - 1)

        self.position += (
            np.array([np.cos(self.orientation), np.sin(self.orientation)])
            * forward_speed
        )

        self.angular_velocity += turn_acceleration_range * (rng.random() - 0.5)
        self.angular_velocity *= turn_friction
        self.orientation += self.angular_velocity

        self.history.append(self.position.copy())

    def is_too_young(self, age_of_maturity):
        return len(self.history) <= age_of_maturity

    def death_outcome(
        self,
        agents,
        true_population,
        age_of_maturity,
        merge_radius,
        prevent_replication_radius,
        param_getter,
        rng,
        lanes,
        grid_to_lanes,
        spatial_hash_gridsize,
    ):
        """
        :return: prevent_replication（是否阻止复制）、die（是否终止）
        """
        # 根据随机终止概率结束智能体
        spontaneous_death_chance = param_getter.paramAt(
            "spontaneous_death_chance", self.position
        )

        if true_population > 3 and rng.random() < spontaneous_death_chance:
            return True, True

        # 因合并而终止
        prevent_replication = False
        kill = False

        # 检查其他智能体及自身的历史路径
        for other_agent in agents:
            for i, position in enumerate(other_agent.history):
                if self is other_agent and i >= len(self.history) - age_of_maturity:
                    # 靠近自身几个时间步之前经过的位置不应计入
                    # 需要判断的是绕一圈后撞上自身路径的情况，这通常需要更多时间步
                    break

                dist = np.linalg.norm(position - self.position)
                if self is not other_agent and dist < prevent_replication_radius:
                    prevent_replication = True
                if dist < merge_radius:
                    kill = True
                    break

        # 检查已经生成的车道
        if not kill:
            gridpoint = point_to_gridpoint(self.position, spatial_hash_gridsize)
            proximal_lanes = get_proximal_lanes_wrt_gridpoint(
                grid_to_lanes, gridpoint, extended=True
            )
            for lane_id in sorted(proximal_lanes):
                lane = lanes[lane_id]
                for i, position in enumerate(lane.points):
                    dist = np.linalg.norm(position - self.position)
                    if dist < prevent_replication_radius:
                        prevent_replication = True
                    if dist < merge_radius:
                        kill = True
                        break

        return prevent_replication, kill

    def replicate(self, param_getter, rng, true_population, num_locations):
        new_agents = []
        replication_chance = param_getter.paramAt("replication_chance", self.position)
        if rng.random() < replication_chance:
            self.end_location = str(num_locations)

            fork_config = ConstructionAgent.random_fork_config(rng)
            for i, angle in enumerate(ConstructionAgent.fork_angles):
                if fork_config[i] == 1 or true_population < 3:
                    # 如果智能体数量不足，则生成最多的后代（3 个）
                    # 以防智能体全部消失
                    new_agent = ConstructionAgent(
                        start_location=num_locations,
                        position=self.position.copy(),
                        orientation=self.orientation + angle,
                    )
                    new_agents.append(new_agent)

        return new_agents


class PerlinVariation:
    scale = 200
    octaves = 1
    persistence = 0.1
    lacunarity = 2.0

    def __init__(self, perlin_variation_params: dict, rng: np.random.Generator):
        self.perlin_variation_params = perlin_variation_params
        for feature_params_dict in self.perlin_variation_params.values():
            feature_params_dict["x"] = rng.integers(0, 10000)
            feature_params_dict["y"] = rng.integers(0, 10000)

    def paramAt(self, param, pos):
        x = self.perlin_variation_params[param]["x"]
        y = self.perlin_variation_params[param]["y"]
        upper = self.perlin_variation_params[param]["upper"]
        lower = self.perlin_variation_params[param]["lower"]
        noise_val = pnoise2(
            (pos[0] / PerlinVariation.scale) + x,
            (pos[1] / PerlinVariation.scale) + y,
            octaves=PerlinVariation.octaves,
            persistence=PerlinVariation.persistence,
            lacunarity=PerlinVariation.lacunarity,
        )
        # Perlin 噪声的平方
        return (((upper - lower) * noise_val * abs(noise_val)) + upper + lower) / 2.0
