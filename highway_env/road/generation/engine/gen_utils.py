from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike


# 车道图辅助工具 ----------------------------------------


@dataclass
class Lane:
    """
    车道的原始几何信息和逻辑连接。

    :param start: 车道起始路口或节点的字符串标识
    :param end: 车道终止路口或节点的字符串标识
    :param points: 从起点到终点排列的中心线点列表
    :param left_points: 按顺序排列的左边界点列表
    :param right_points: 按顺序排列的右边界点列表
    """

    start: str
    end: str
    points: list[np.ndarray] = field(default_factory=list)
    left_points: list[np.ndarray] = field(default_factory=list)
    right_points: list[np.ndarray] = field(default_factory=list)

    def __str__(self):
        lines = [f"start: {self.start} end: {self.end}", "left:"]
        for pt in self.left_points:
            lines.append(f"{pt[0]} {pt[1]}")
        lines.append("right:")
        for pt in self.right_points:
            lines.append(f"{pt[0]} {pt[1]}")

        return "\n".join(lines)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class Endpoint:
    """
    表示指定车道的一端。

    :param id: Lane 在 lanes 列表中的索引
    :param loc: location 的缩写，表示车道的哪一端，取值为 'start' 或 'end'
    """

    id: int
    loc: str

    # 将 loc 值转换为列表的首项或末项索引
    l_to_i: ClassVar[dict[str, int]] = {"start": 0, "end": -1}

    def point_index(self) -> int:
        """
        :return: 根据 loc 字段返回 -1 或 0
        """
        return Endpoint.l_to_i[self.loc]

    def second_point_index(self) -> int:
        """
        :return: 根据 loc 字段返回 -2 或 1
        """
        return self.point_index() * 3 + 1  # 将 {-1, 0} 转换为 {-2, 1}

    def position(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: 车道列表
        :return: 车道中心线对应一端的端点位置
        """
        return lanes[self.id].points[self.point_index()]

    def vector_raw(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: 车道列表
        :return: 中心线上从倒数第二个点指向末端点的向量
        """
        pos = self.position(lanes)
        pos2 = lanes[self.id].points[self.second_point_index()]
        return pos - pos2

    def vector(self, lanes: list[Lane]) -> np.ndarray:
        """
        :param lanes: 车道列表
        :return: 表示端点朝向的单位向量
        """
        vec = self.vector_raw(lanes)
        return vec / np.linalg.norm(vec)


def get_radially_sorted_endpoints(lanes: list[Lane], node: str) -> list[Endpoint]:
    """
    :param lanes: 车道列表
    :param node: 路口的字符串标识
    :return: 构成该路口的所有端点，按相对路口中心的角度排序
    """
    endpoints = []

    for i, lane in enumerate(lanes):
        for loc in ["start", "end"]:
            if getattr(lane, loc) == node:
                endpoints.append(Endpoint(id=i, loc=loc))

    if len(endpoints) == 0:
        return []

    midpoint = get_junction_pos(lanes, endpoints)

    def getTheta(ep):
        pos = ep.position(lanes) - midpoint
        return np.arctan2(pos[1], pos[0])

    endpoints.sort(key=getTheta)

    return endpoints


def get_junction_pos(
    lanes: list[Lane],
    junction: list[Endpoint],
    excluded_endpoint: Endpoint | None = None,
    use_boundaries: bool = False,
) -> np.ndarray:
    """
    :param lanes: 车道列表
    :param junction: 同一个路口的所有端点列表
    :param excluded_endpoint: 可选，计算中点时排除的端点
    :param use_boundaries: 是否使用实际边界点代替中心线点
    :return: 所有末端边界点的中点
    """
    if excluded_endpoint is None:
        assert len(junction) > 0
    else:
        assert len(junction) > 1
        assert excluded_endpoint in junction

    pt = np.zeros(2)
    for ep in junction:
        if ep != excluded_endpoint:
            if use_boundaries:
                pt += lanes[ep.id].left_points[ep.point_index()]
                pt += lanes[ep.id].right_points[ep.point_index()]
            else:
                pt += ep.position(lanes)

    pt /= len(junction) - (1 if excluded_endpoint is not None else 0)
    if use_boundaries:
        pt /= 2

    return pt


def get_nodeset(lanes: list[Lane]):
    """
    :param lanes: 车道列表
    :return: 所有不重复的路口字符串标识组成的集合
    """
    nodeset = set()
    for lane in lanes:
        nodeset.add(lane.start)
        nodeset.add(lane.end)
    return nodeset


# 几何辅助工具 ----------------------------------------


def line_intersection_t(
    a: np.ndarray | ArrayLike, av: np.ndarray, b: np.ndarray | ArrayLike, bv: np.ndarray
) -> tuple[float, float]:
    """
    :param a: 直线 A 上的点
    :param av: 直线 A 的方向向量
    :param b: 直线 B 上的点
    :param bv: 直线 B 的方向向量
    :return: 满足 A + t_a*A_v = B + t_b*B_v 的参数 t_a 和 t_b
    """
    A = np.column_stack((av, -bv))
    B = np.asarray(b) - np.asarray(a)

    try:
        t_a, t_b = np.linalg.solve(A, B)
        return t_a, t_b
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def do_line_segments_intersect(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> bool:
    """
    :param a0: 线段 A 的第一个端点
    :param a1: 线段 A 的第二个端点
    :param b0: 线段 B 的第一个端点
    :param b1: 线段 B 的第二个端点
    :return: 两条线段是否相交
    """
    t_a, t_b = line_intersection_t(a0, a1 - a0, b0, b1 - b0)
    return t_a >= 0 and t_a <= 1 and t_b >= 0 and t_b <= 1


def find_line_intersection(
    a: np.ndarray,
    av: np.ndarray,
    b: np.ndarray,
    bv: np.ndarray,
    return_t: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    :param a: 直线 A 上的点
    :param av: 直线 A 的方向向量
    :param b: 直线 B 上的点
    :param bv: 直线 B 的方向向量
    :param return_t: 是否返回满足 A + t_a*A_v = B + t_b*B_v 的 t 值
    :return: 交点；若 return_t 为 True，则返回包含交点和两个 t 值的元组
    """
    t_a, t_b = line_intersection_t(a, av, b, bv)

    pt = a + (av * t_a)

    if return_t:
        return pt, t_a, t_b
    else:
        return pt


# 其他工具 ----------------------------------------


def wrap_with_tqdm(iterable=None, disabled=False, *args, **kwargs):
    """
    封装 tqdm；当 tqdm 不可用或被禁用时，直接透传，不显示进度条。

    :param iterable: 可选，需要包装为进度条的可迭代对象
    :param disabled: 若为 True，即使安装了 tqdm 也跳过进度条
    :return: tqdm 进度条、原可迭代对象或空操作进度条
    """
    if not disabled:
        try:
            from tqdm import tqdm

            return tqdm(iterable, *args, **kwargs)
        except ImportError:
            pass
    if iterable is not None:
        return iterable
    return _DummyPbar()


class _DummyPbar:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def update(self, n: int | float = 1):
        self.n += n

    def refresh(self):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
