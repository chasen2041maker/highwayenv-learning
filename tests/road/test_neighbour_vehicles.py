"""测试 Road.neighbour_vehicles，可选择是否搜索相连的车道段。

覆盖问题 #626：neighbour_vehicles 未考虑相连车道。
"""

import numpy as np
import pytest

from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


def _make_vehicle(
    road: Road,
    net: RoadNetwork,
    lane_index: tuple[str, str, int],
    longitudinal: float,
) -> Vehicle:
    """辅助函数：在车道的指定纵向位置创建车辆。"""
    lane = net.get_lane(lane_index)
    pos = lane.position(longitudinal, 0)
    heading = lane.heading_at(longitudinal)
    v = Vehicle(road, position=pos, heading=heading, speed=10)
    v.lane_index = lane_index
    v.lane = lane
    road.vehicles.append(v)
    return v


def _enable_connected_lane_search(road: Road) -> None:
    road.neighbour_vehicles_connected_lanes = True


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def straight_connected_road() -> tuple[Road, RoadNetwork]:
    """两段相连的直道：a->b（50 米），然后是 b->c（50 米）。"""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def straight_curve_road() -> tuple[Road, RoadNetwork]:
    """直道路段 a->b 后接圆弧路段 b->c。"""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center=[50, -20],
            radius=20,
            start_phase=np.deg2rad(90),
            end_phase=np.deg2rad(0),
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def three_segment_road() -> tuple[Road, RoadNetwork]:
    """三段相连道路：a->b（50 米）、b->c（50 米）、c->d（50 米）。"""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [100, 0], [150, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


@pytest.fixture
def multi_lane_road() -> tuple[Road, RoadNetwork]:
    """双车道的相连道路：a->b 和 b->c，每段都有 2 条车道。"""
    net = RoadNetwork()
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED)
        ),
    )
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 4], [50, 4], line_types=(LineType.STRIPED, LineType.CONTINUOUS)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED)
        ),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane(
            [50, 4], [100, 4], line_types=(LineType.STRIPED, LineType.CONTINUOUS)
        ),
    )
    road = Road(network=net, np_random=np.random.RandomState(42))
    return road, net


# ---------------------------------------------------------------------------
# 测试：同一车道段上的行为（回归测试，原功能必须仍然可用）
# ---------------------------------------------------------------------------


class TestSameSegmentNeighbours:
    """验证原有的同一车道段检测仍然正常。"""

    def test_front_and_rear_on_same_segment(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is front
        assert v_rear is rear

    def test_no_neighbours(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is None

    def test_only_front(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)
        front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is front
        assert v_rear is None

    def test_only_rear(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=40)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=10)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is rear

    def test_connected_segments_ignored_by_default(self, straight_connected_road):
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is None


# ---------------------------------------------------------------------------
# 测试：相连车道段上的邻车（问题 #626 的修复）
# ---------------------------------------------------------------------------


class TestConnectedLaneNeighbours:
    """验证能检测相连的前一段和后一段车道上的车辆。"""

    def test_front_on_next_segment(self, straight_connected_road):
        """下一段 b->c 上的车辆应被检测为前车。"""
        road, net = straight_connected_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front
        ), "Vehicle on connected next segment should be detected as front neighbour"

    def test_rear_on_previous_segment(self, straight_connected_road):
        """前一段 a->b 上的车辆应被检测为后车。"""
        road, net = straight_connected_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert (
            v_rear is rear
        ), "Vehicle on connected previous segment should be detected as rear neighbour"

    def test_front_on_curve_segment(self, straight_curve_road):
        """相连弯道上的车辆应被检测为前车。"""
        road, net = straight_curve_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front
        ), "Vehicle on connected curve should be detected as front neighbour"

    def test_closer_same_segment_preferred_over_next_segment(
        self, straight_connected_road
    ):
        """同一车道段和下一车道段前方都有车辆时，应返回距离更近的一辆。
        """
        road, net = straight_connected_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=30)
        close_front = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)
        _make_vehicle(
            road, net, ("b", "c", 0), longitudinal=10
        )  # 下一车道段上距离更远的车辆

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is close_front
        ), "Closer same-segment vehicle should be preferred over farther next-segment one"

    def test_both_connected_front_and_rear(self, three_segment_road):
        """自车在中间路段，前车在下一段，后车在前一段。"""
        road, net = three_segment_road
        _enable_connected_lane_search(road)
        rear = _make_vehicle(road, net, ("a", "b", 0), longitudinal=45)
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        front = _make_vehicle(road, net, ("c", "d", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert v_front is front
        assert v_rear is rear

    def test_multi_lane_same_lane_id(self, multi_lane_road):
        """在多车道道路中，只应考虑下一路段上车道编号匹配的车辆。
        """
        road, net = multi_lane_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)
        front_lane0 = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)
        # 下一路段的车道 1 上的车辆，车道编号不同
        _make_vehicle(road, net, ("b", "c", 1), longitudinal=3)

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert (
            v_front is front_lane0
        ), "Only vehicles on the same lane id of the next segment should match"


# ---------------------------------------------------------------------------
# 测试：边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """跨相连车道段邻车搜索的边界情况。"""

    def test_no_next_segment(self):
        """当前路段没有下游连接的情况。"""
        net = RoadNetwork()
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0], [50, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
            ),
        )
        road = Road(
            network=net,
            np_random=np.random.RandomState(42),
            neighbour_vehicles_connected_lanes=True,
        )
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=48)

        v_front, v_rear = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is None
        assert v_rear is None

    def test_no_previous_segment(self):
        """当前路段没有上游连接的情况。"""
        net = RoadNetwork()
        net.add_lane(
            "b",
            "c",
            StraightLane(
                [50, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)
            ),
        )
        road = Road(
            network=net,
            np_random=np.random.RandomState(42),
            neighbour_vehicles_connected_lanes=True,
        )
        ego = _make_vehicle(road, net, ("b", "c", 0), longitudinal=5)

        v_front, v_rear = road.neighbour_vehicles(ego, ("b", "c", 0))
        assert v_front is None
        assert v_rear is None

    def test_vehicle_far_on_next_segment_detected(self, straight_connected_road):
        """位于相连下一路段较远处的车辆，仍是有效的前车。"""
        road, net = straight_connected_road
        _enable_connected_lane_search(road)
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        _make_vehicle(road, net, ("b", "c", 0), longitudinal=40)

        v_front, _ = road.neighbour_vehicles(ego, ("a", "b", 0))
        assert v_front is not None

    def test_lane_index_none_returns_none(self, straight_connected_road):
        """车辆没有 lane_index 时，应返回 (None, None)。"""
        road, net = straight_connected_road
        ego = _make_vehicle(road, net, ("a", "b", 0), longitudinal=25)
        ego.lane_index = None

        v_front, v_rear = road.neighbour_vehicles(ego)
        assert v_front is None
        assert v_rear is None
