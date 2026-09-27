from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from highway_env.road.lane import AbstractLane, LineType, StraightLane, lane_from_config
from highway_env.vehicle.objects import Landmark


if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = tuple[str, str, int]
Route = list[LaneIndex]


class RoadNetwork:
    graph: dict[str, dict[str, list[AbstractLane]]]

    def __init__(self):
        self.graph = {}
        self.reversed_lane_indices = set()

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        将车道作为道路网络中的一条边。

        :param _from: 车道的起点节点
        :param _to: 车道的终点节点
        :param AbstractLane lane: 车道几何结构
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def add_lane_bidirectional(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        添加允许双向通行的车道。
        即从 _from 到 _to，以及从 _to 到 _from。
        """
        self.add_lane(_from, _to, lane)
        self.add_lane(_to, _from, lane)  # pylint: disable=arguments-out-of-order
        self.reversed_lane_indices.add((_to, _from, len(self.graph[_to][_from]) - 1))

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        获取道路网络中给定索引对应的车道几何结构。

        :param index: 元组（起点节点、终点节点、该道路上的车道编号）
        :return: 对应的车道几何结构
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: float | None = None
    ) -> LaneIndex:
        """
        获取离某个世界位置最近的车道索引。

        :param position: 世界位置，单位为米
        :param heading: 朝向角，单位为弧度
        :return: 最近车道的索引
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route | None = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        获取驶完当前车道后应跟随的下一条车道索引。

        - 如果存在与当前车道匹配的规划路线，则遵循规划。
        - 否则，随机选择下一条道路。
        - 如果下一条道路与当前道路的车道数相同，则保持车道编号。
        - 否则，选择下一条道路上最近的车道。

        :param current_index: 当前目标车道的索引
        :param route: 规划路线，可以为空
        :param position: 车辆位置
        :param np_random: 随机数生成器
        :return: 驶完当前车道后要跟随的下一条车道索引
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # 根据规划路线选择下一条道路
        if route:
            if (
                route[0][:2] == current_index[:2]
            ):  # 刚刚完成路线中的第一段，将其移除。
                route.pop(0)
            if (
                route and route[0][0] == _to
            ):  # 路线中的下一条道路从当前道路的终点开始。
                _, next_to, next_id = route[0]
            elif route:
                logger.warning(
                    "Route {} does not start after current road {}.".format(
                        route[0], current_index
                    )
                )

        # 计算当前投影后的期望位置
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        # 如果下一段路线未知
        if not next_to:
            # 选择车道最接近投影目标位置的道路
            try:
                lanes_dists = [
                    (
                        next_to,
                        *self.next_lane_given_next_road(
                            _from, _to, _id, next_to, next_id, projected_position
                        ),
                    )
                    for next_to in self.graph[_to].keys()
                ]  # 元组内容：(next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # 如果下一段路线已知，则沿该路线选择最近车道
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, _id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id

    def next_lane_given_next_road(
        self,
        _from: str,
        _to: str,
        _id: int,
        next_to: str,
        next_id: int,
        position: np.ndarray,
    ) -> tuple[int, float]:
        # 如果下一条道路的车道数相同，则保持车道编号
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # 否则，选择最近的车道
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(
                lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position)
            )
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> list[list[str]]:
        """
        通过广度优先搜索查找从起点到终点的所有路线。

        :param start: 起点节点
        :param goal: 终点节点
        :return: 从起点到终点的路径列表
        """
        queue = [(start, [start])]
        while queue:
            node, path = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in sorted(
                [key for key in self.graph[node].keys() if key not in path]
            ):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """
        通过广度优先搜索查找从起点到终点的最短路径。

        :param start: 起点节点
        :param goal: 终点节点
        :return: 从起点到终点的最短路径
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
        """
        :param lane_index: 车道索引
        :return: 属于同一条道路的所有车道
        """
        return [
            (lane_index[0], lane_index[1], i)
            for i in range(len(self.graph[lane_index[0]][lane_index[1]]))
        ]

    def side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
        """
        :param lane_index: 车道索引
        :return: 输入车道左侧或右侧相邻车道的索引
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """车道 1 和车道 2 是否属于同一条道路？"""
        return lane_index_1[:2] == lane_index_2[:2] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """车道 1 是否通向车道 2？"""
        return lane_index_1[1] == lane_index_2[0] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
        """
        车道 2 是否通向车道 1 路线中的某条道路？

        碰撞检测需要考虑这些车道上的车辆。
        :param lane_index_1: 起始车道
        :param lane_index_2: 目标车道
        :param route: 从起始车道出发的路线，可以为空
        :param same_lane: 是否比较车道编号
        :param depth: 从车道 1 沿路线搜索的深度
        :return: 两条道路是否连通
        """
        if RoadNetwork.is_same_road(
            lane_index_2, lane_index_1, same_lane
        ) or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # 路线从当前道路开始，跳过这一段
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                # 路线从当前道路继续向前，沿路线搜索
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # 递归搜索路口处的所有道路
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road(
                            (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                        )
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> list[AbstractLane]:
        return [
            lane for to in self.graph.values() for ids in to.values() for lane in ids
        ]

    def lanes_dict(self) -> dict[str, AbstractLane]:
        return {
            (from_, to_, i): lane
            for from_, tos in self.graph.items()
            for to_, ids in tos.items()
            for i, lane in enumerate(ids)
        }

    @staticmethod
    def straight_road_network(
        lanes: int = 4,
        start: float = 0.0,
        length: float = 10000.0,
        angle: float = 0.0,
        speed_limit: float = 30.0,
        nodes_str: tuple[str, str] | None = None,
        net: RoadNetwork | None = None,
    ) -> RoadNetwork:
        net = net or RoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            origin = rotation @ origin
            end = rotation @ end
            line_types = [
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE,
            ]
            net.add_lane(
                *nodes_str,
                StraightLane(
                    origin, end, line_types=line_types, speed_limit=speed_limit
                ),
            )
        return net

    def position_heading_along_route(
        self,
        route: Route,
        longitudinal: float,
        lateral: float,
        current_lane_index: LaneIndex,
    ) -> tuple[np.ndarray, float]:
        """
        在由多条车道组成的路线中，根据局部坐标获取绝对位置和朝向。

        :param route: 规划路线，即车道索引列表
        :param longitudinal: 纵向位置
        :param lateral: 横向位置
        :param current_lane_index: 车辆当前的车道索引
        :return: 位置和朝向
        """

        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # 已知车辆将跟随的道路段，但不知道具体车道。
                # 假设：车辆保持与当前车道相同的 lane_id。
                id_ = (
                    current_lane_index[2]
                    if current_lane_index[2]
                    < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                    else 0
                )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.integers(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict


class Road:
    """道路由一组车道，以及在这些车道上行驶的一组车辆组成。"""

    def __init__(
        self,
        network: RoadNetwork = None,
        vehicles: list[kinematics.Vehicle] = None,
        road_objects: list[objects.RoadObject] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
        neighbour_vehicles_connected_lanes: bool = False,
    ) -> None:
        """
        创建道路。

        :param network: 描述车道的道路网络
        :param vehicles: 在道路上行驶的车辆
        :param road_objects: 道路物体，包括障碍物和地标
        :param np.random.RandomState np_random: 用于车辆行为的随机数生成器
        :param record_history: 是否记录车辆最近的轨迹以供显示
        :param neighbour_vehicles_connected_lanes: 查找邻车时是否搜索相连的车道段
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
        self.neighbour_vehicles_connected_lanes = neighbour_vehicles_connected_lanes

    def close_objects_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        vehicles = [
            v
            for v in self.vehicles
            if np.linalg.norm(v.position - vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]
        obstacles = [
            o
            for o in self.objects
            if np.linalg.norm(o.position - vehicle.position) < distance
            and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)
        ]

        objects_ = vehicles if vehicles_only else vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count is not None:
            objects_ = objects_[:count]
        return objects_

    def close_vehicles_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
    ) -> object:
        return self.close_objects_to(
            vehicle, distance, count, see_behind, sort, vehicles_only=True
        )

    def act(self) -> None:
        """决定道路上各个实体的动作。"""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        推进道路上各个实体的运动。

        :param dt: 时间步长，单位为秒
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1 :]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(
        self, vehicle: kinematics.Vehicle, lane_index: LaneIndex = None
    ) -> tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
        """
        查找给定车辆的前车和后车。

        启用 ``neighbour_vehicles_connected_lanes`` 后，还会搜索相连的前后车道段，
        以检测位于路段边界附近的车辆。

        :param vehicle: 需要查找邻车的车辆
        :param lane_index: 用于查找前后车辆的车道；可以不是该车当前所在的车道。
            若是其他车道，则根据车辆在该车道中的局部坐标进行投影。
        :return: 前车和后车
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None

        lanes_offsets: list[tuple[AbstractLane, float]] = [(lane, 0)]

        if self.neighbour_vehicles_connected_lanes:
            # 通过偏移量，将各个相连车道的纵向坐标
            # 转换到自车所在车道的坐标系。
            _from, _to, _id = lane_index

            for next_lanes in self.network.graph.get(_to, {}).values():
                if _id < len(next_lanes):
                    lanes_offsets.append((next_lanes[_id], lane.length))
                elif next_lanes:
                    lanes_offsets.append((next_lanes[0], lane.length))

            for to_dict in self.network.graph.values():
                if _from in to_dict:
                    prev_lanes = to_dict[_from]
                    if _id < len(prev_lanes):
                        prev_lane = prev_lanes[_id]
                    elif prev_lanes:
                        prev_lane = prev_lanes[0]
                    else:
                        continue
                    lanes_offsets.append((prev_lane, -prev_lane.length))

        for v in self.vehicles + self.objects:
            if v is vehicle or isinstance(v, Landmark):
                continue
            for search_lane, offset in lanes_offsets:
                s_v, lat_v = search_lane.local_coordinates(v.position)
                if not search_lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                s_v += offset
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
                break  # 已在这条车道中匹配，无需检查其他车道

        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()
