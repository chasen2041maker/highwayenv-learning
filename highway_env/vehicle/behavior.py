from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class IDMVehicle(ControlledVehicle):
    """
    同时使用纵向和横向决策策略的车辆。

    - 纵向：IDM 模型根据前车距离和速度计算加速度。
    - 横向：MOBIL 模型通过提高附近车辆的加速度收益，决定何时变道。
    """

    # 纵向策略参数
    ACC_MAX = 6.0  # [m/s2]
    """最大加速度。"""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """期望的最大加速度。"""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """期望的最大减速度。"""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """拥堵时与前车保持的期望距离。"""

    TIME_WANTED = 1.5  # [s]
    """与前车保持的期望时间间隔。"""

    DELTA = 4.0  # []
    """速度项的指数。"""

    DELTA_RANGE = [3.5, 4.5]
    """随机选择 delta 时的取值范围。"""

    # 横向策略参数
    POLITENESS = 0.0  # 取值范围为 [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> IDMVehicle:
        """
        根据已有车辆创建新车辆。

        复制车辆的运动状态和目标运动状态，其他属性使用默认值。

        :param vehicle: 已有车辆
        :return: 具有相同运动状态的新车辆
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
        )
        return v

    def act(self, action: dict | str = None):
        """
        执行动作。

        目前不接受外部动作，因为车辆根据 IDM 和 MOBIL 模型，
        自主决定加速度和变道。

        :param action: 动作
        """
        if self.crashed:
            return
        action = {}
        # 横向：MOBIL 模型
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # 纵向：IDM 模型
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        # 变道期间，同时检查当前车道和目标车道
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        # 跳过 ControlledVehicle.act()，否则这里的控制指令会被覆盖。
        Vehicle.act(self, action)

    def step(self, dt: float):
        """
        推进一步仿真。

        增加决策策略使用的计时器，并推进车辆运动。

        :param dt: 时间步长
        """
        self.timer += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        使用智能驾驶员模型（IDM）计算加速度指令。

        选择加速度时考虑：
        - 达到目标速度；
        - 与前车保持最小安全距离和安全时间间隔。

        :param ego_vehicle: 要计算期望加速度的车辆，不必是 IDM 车辆。
            因此该方法使用类方法，使 IDM 车辆也能推测非 IDM 车辆的行为。
        :param front_vehicle: 自车前方的车辆
        :param rear_vehicle: 自车后方的车辆
        :return: 自车的加速度指令，单位为米/秒²
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        计算车辆与前车之间的期望距离。

        :param ego_vehicle: 被控制的车辆
        :param front_vehicle: 它的前车
        :param projected: 是否将二维速度投影到一维方向
        :return: 两车之间的期望距离，单位为米
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def change_lane_policy(self) -> None:
        """
        决定何时变道。

        依据包括：
        - 决策频率；
        - 目标车道是否足够接近；
        - MOBIL 模型。
        """
        # 如果已经正在变道
        if self.lane_index != self.target_lane_index:
            # 如果道路路线正确但车道不合适，且其他车辆也正驶入同一车道，则取消变道
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if (
                        v is not self
                        and v.lane_index != self.target_lane_index
                        and isinstance(v, ControlledVehicle)
                        and v.target_lane_index == self.target_lane_index
                    ):
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # 否则，按照指定频率
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # 判断是否开始变道
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # 候选车道是否足够接近？
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                self.position
            ):
                continue
            # 仅在车辆运动时变道
            if np.abs(self.speed) < 1:
                continue
            # MOBIL 模型是否建议变道？
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL 变道模型：尽量减少变道引起的整体制动。

        只有满足以下条件才应变道：
        - 变道后，自车和/或后车能够获得更大的加速度；
        - 不会迫使新车道上的后车进行不安全的制动。

        :param lane_index: 候选目标车道
        :return: 是否应执行变道
        """
        # 这次变道对新车道上的后车是否不安全？
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # 规划路线是否指定了某条可以安全驶入的车道？
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # 方向不正确
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # 需要进行不安全的制动
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # 变道是否能为自车和/或后车带来加速度收益？
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
            )
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # 各项检查通过，可以变道。
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        如果停在错误车道上，尝试倒车脱困。

        :param acceleration: IDM 给出的期望加速度
        :return: 为摆脱停滞状态建议的加速度
        """
        stopped_speed = 5
        safe_distance = 200
        # 车辆是否停在错误车道上？
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(
                self, self.road.network.get_lane(self.target_lane_index)
            )
            # 检查两条车道的后方是否都有空余空间
            if (not rear or rear.lane_distance_to(self) > safe_distance) and (
                not new_rear or new_rear.lane_distance_to(self) > safe_distance
            ):
                # 倒车
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class LinearVehicle(IDMVehicle):
    """纵向和横向控制器都对参数呈线性关系的车辆。"""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [
        ControlledVehicle.KP_HEADING,
        ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL,
    ]

    ACCELERATION_RANGE = np.array(
        [
            0.5 * np.array(ACCELERATION_PARAMETERS),
            1.5 * np.array(ACCELERATION_PARAMETERS),
        ]
    )
    STEERING_RANGE = np.array(
        [
            np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
            np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5]),
        ]
    )

    TIME_WANTED = 2.5

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        data: dict = None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: dict | str = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (
            self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0]
        )
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (
            self.STEERING_RANGE[1] - self.STEERING_RANGE[0]
        )
        super().randomize_behavior()

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        使用线性模型计算加速度指令。

        选择加速度时考虑：
        - 达到目标速度；
        - 前车更慢时接近前车速度，后车更快时接近后车速度；
        - 与前车保持最小安全距离。

        :param ego_vehicle: 要计算期望加速度的车辆，不必是 LinearVehicle。
            因此该方法使用类方法，使线性模型车辆也能推测其他类型车辆的行为。
        :param front_vehicle: 自车前方的车辆
        :param rear_vehicle: 自车后方的车辆
        :return: 自车的加速度指令，单位为米/秒²
        """
        return float(
            np.dot(
                self.ACCELERATION_PARAMETERS,
                self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle),
            )
        )

    def acceleration_features(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = (
                getattr(ego_vehicle, "target_speed", ego_vehicle.speed)
                - ego_vehicle.speed
            )
            d_safe = (
                self.DISTANCE_WANTED
                + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            )
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        对参数呈线性关系的控制器。

        覆盖非线性控制器 ControlledVehicle.steering_control()。

        :param target_lane_index: 要跟随的车道索引
        :return: 转向角指令，单位为弧度
        """
        return float(
            np.dot(
                np.array(self.STEERING_PARAMETERS),
                self.steering_features(target_lane_index),
            )
        )

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        用于沿车道行驶的一组特征。

        :param target_lane_index: 要跟随的车道索引
        :return: 特征数组
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array(
            [
                utils.wrap_to_pi(lane_future_heading - self.heading)
                * self.LENGTH
                / utils.not_zero(self.speed),
                -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2),
            ]
        )
        return features

    def longitudinal_structure(self):
        # 标称动力学：对速度积分
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # 目标速度动力学
        phi0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        # 前车速度控制
        phi1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]])
        # 前车位置控制
        phi2 = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [-1, 1, -self.TIME_WANTED, 0], [0, 0, 0, 0]]
        )
        # 关闭速度控制
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # 关闭前车位置控制
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([[0, 1], [0, 0]])
        phi0 = np.array([[0, 0], [0, -1]])
        phi1 = np.array([[0, 0], [-1, 0]])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """保存用于参数回归的特征和输出。"""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        0.5,
    ]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]
