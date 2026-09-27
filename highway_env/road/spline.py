from __future__ import annotations

import numpy as np


def numpy_interp1d(x: np.ndarray, y: np.ndarray):
    """
    可直接替代 ``scipy.interpolate.interp1d(x, y, fill_value="extrapolate")``。

    由 https://github.com/Farama-Foundation/HighwayEnv/pull/691 引入，以移除 ``scipy`` 依赖。
    原说明中的验证基准脚本路径为 ``scripts/validate/bench_interp1d.py``。
    """

    def interpolator(x_new):
        x_new = np.asarray(x_new, dtype=float)
        scalar = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)

        result = np.interp(x_new, x, y)

        left = x_new < x[0]
        if np.any(left):
            slope = (y[1] - y[0]) / (x[1] - x[0])
            result[left] = y[0] + slope * (x_new[left] - x[0])

        right = x_new > x[-1]
        if np.any(right):
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            result[right] = y[-1] + slope * (x_new[right] - x[-1])

        return float(result[0]) if scalar else result

    return interpolator


class LinearSpline2D:
    """
    根据一组点拟合的分段线性曲线。
    """

    PARAM_CURVE_SAMPLE_DISTANCE: int = 1  # 曲线采样点之间相距 1 米

    def __init__(self, points: list[tuple[float, float]]):
        x_values = np.array([pt[0] for pt in points])
        y_values = np.array([pt[1] for pt in points])
        x_values_diff = np.diff(x_values)
        x_values_diff = np.hstack((x_values_diff, x_values_diff[-1]))
        y_values_diff = np.diff(y_values)
        y_values_diff = np.hstack((y_values_diff, y_values_diff[-1]))
        arc_length_cumulated = np.hstack(
            (0, np.cumsum(np.sqrt(x_values_diff[:-1] ** 2 + y_values_diff[:-1] ** 2)))
        )
        self.length = arc_length_cumulated[-1]
        self.x_curve = numpy_interp1d(arc_length_cumulated, x_values)
        self.y_curve = numpy_interp1d(arc_length_cumulated, y_values)
        self.dx_curve = numpy_interp1d(arc_length_cumulated, x_values_diff)
        self.dy_curve = numpy_interp1d(arc_length_cumulated, y_values_diff)

        (self.s_samples, self.poses) = self.sample_curve(
            self.x_curve, self.y_curve, self.length, self.PARAM_CURVE_SAMPLE_DISTANCE
        )

    def __call__(self, lon: float) -> tuple[float, float]:
        return self.x_curve(lon), self.y_curve(lon)

    def get_dx_dy(self, lon: float) -> tuple[float, float]:
        idx_pose = self._get_idx_segment_for_lon(lon)
        pose = self.poses[idx_pose]
        return pose.normal

    def cartesian_to_frenet(self, position: tuple[float, float]) -> tuple[float, float]:
        """
        将笛卡尔坐标中的点转换为曲线的 Frenet 坐标。
        """

        pose = self.poses[-1]
        projection = pose.project_onto_normal(position)
        if projection >= 0:
            lon = self.s_samples[-1] + projection
            lat = pose.project_onto_orthonormal(position)
            return lon, lat

        for idx in list(range(1, len(self.s_samples) - 1))[::-1]:
            pose = self.poses[idx]
            projection = pose.project_onto_normal(position)
            if projection >= 0:
                lon = self.s_samples[idx] + projection
                lat = pose.project_onto_orthonormal(position)
                return lon, lat
        pose = self.poses[0]
        lon = pose.project_onto_normal(position)
        lat = pose.project_onto_orthonormal(position)
        return lon, lat

    def frenet_to_cartesian(self, lon: float, lat: float) -> tuple[float, float]:
        """
        将曲线 Frenet 坐标中的点转换为笛卡尔坐标。
        """
        idx_segment = self._get_idx_segment_for_lon(lon)
        s = lon - self.s_samples[idx_segment]
        pose = self.poses[idx_segment]
        point = pose.position + s * pose.normal
        point += lat * pose.orthonormal
        return point

    def _get_idx_segment_for_lon(self, lon: float) -> int:
        """
        返回与给定纵向坐标对应的曲线位姿索引。
        """
        idx_smaller = np.argwhere(lon < self.s_samples)
        if len(idx_smaller) == 0:
            return len(self.s_samples) - 1
        if idx_smaller[0] == 0:
            return 0
        return int(idx_smaller[0].item()) - 1

    @staticmethod
    def sample_curve(x_curve, y_curve, length: float, CURVE_SAMPLE_DISTANCE=1):
        """
        以 CURVE_SAMPLE_DISTANCE 为间距生成曲线采样点。
        这些采样点用于 Frenet 坐标与笛卡尔坐标之间的相互转换。
        """
        num_samples = np.floor(length / CURVE_SAMPLE_DISTANCE)
        s_values = CURVE_SAMPLE_DISTANCE * np.arange(0, int(num_samples) + 1)
        x_values = x_curve(s_values)
        y_values = y_curve(s_values)
        dx_values = np.diff(x_values)
        dx_values = np.hstack((dx_values, dx_values[-1]))
        dy_values = np.diff(y_values)
        dy_values = np.hstack((dy_values, dy_values[-1]))

        poses = [
            CurvePose(x, y, dx, dy)
            for x, y, dx, dy in zip(
                x_values, y_values, dx_values, dy_values, strict=False
            )
        ]

        return s_values, poses


class CurvePose:
    """
    曲线上的采样位姿，用于将 Frenet 坐标转换为笛卡尔坐标。
    """

    def __init__(self, x: float, y: float, dx: float, dy: float):
        self.length = np.sqrt(dx**2 + dy**2)
        self.position = np.array([x, y]).flatten()
        self.normal = np.array([dx, dy]).flatten() / self.length
        self.orthonormal = np.array([-self.normal[1], self.normal[0]]).flatten()

    def distance_to_origin(self, point: tuple[float, float]) -> float:
        """
        计算点 [x, y] 到位姿原点的距离。
        """
        return np.sqrt(np.sum((self.position - point) ** 2))

    def project_onto_normal(self, point: tuple[float, float]) -> float:
        """
        将点投影到位姿的方向向量，计算从位姿原点到该点的纵向距离。
        """
        return self.normal.dot(point - self.position)

    def project_onto_orthonormal(self, point: tuple[float, float]) -> float:
        """
        将点投影到与位姿方向正交的单位向量，计算从位姿原点到该点的横向距离。
        """
        return self.orthonormal.dot(point - self.position)
