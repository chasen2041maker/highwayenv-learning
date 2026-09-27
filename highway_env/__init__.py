import os
import sys

from gymnasium.envs.registration import register, registry


__version__ = "1.12.2.dev0"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["highway_env"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# 隐藏 pygame 的支持提示。
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _register_highway_envs():
    """导入 envs 模块，使各环境完成注册。

    此函数具有幂等性：多次调用（例如 Gymnasium 在子进程中解析
    ``"highway_env:env-id"`` 规范时）不会引发重复注册错误。
    """
    # 环境已经注册时直接跳过，保证重复调用的结果一致。
    if "highway-v0" in registry:
        return

    from highway_env.envs.common.abstract import MultiAgentWrapper

    # 出口场景：exit_env.py
    register(
        id="exit-v0",
        entry_point="highway_env.envs.exit_env:ExitEnv",
    )
    register(
        id="exit-v1",
        entry_point="highway_env.envs.exit_env:ConnectedLaneExitEnv",
    )

    # 高速公路场景：highway_env.py
    register(
        id="highway-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnv",
    )

    register(
        id="highway-fast-v0",
        entry_point="highway_env.envs.highway_env:HighwayEnvFast",
    )

    # 交叉路口场景：intersection_env.py
    register(
        id="intersection-v0",
        entry_point="highway_env.envs.intersection_env:IntersectionEnv",
    )

    register(
        id="intersection-v1",
        entry_point="highway_env.envs.intersection_env:ContinuousIntersectionEnv",
    )
    register(
        id="intersection-v2",
        entry_point="highway_env.envs.intersection_env:ConnectedLaneIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v0",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
    )

    register(
        id="intersection-multi-agent-v1",
        entry_point="highway_env.envs.intersection_env:MultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )
    register(
        id="intersection-multi-agent-v2",
        entry_point="highway_env.envs.intersection_env:ConnectedLaneMultiAgentIntersectionEnv",
        additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
    )

    # 车道保持场景：lane_keeping_env.py
    register(
        id="lane-keeping-v0",
        entry_point="highway_env.envs.lane_keeping_env:LaneKeepingEnv",
        max_episode_steps=200,
    )

    # 汇入场景：merge_env.py
    register(
        id="merge-v0",
        entry_point="highway_env.envs.merge_env:MergeEnv",
    )
    register(
        id="merge-v1",
        entry_point="highway_env.envs.merge_env:ConnectedLaneMergeEnv",
    )
    register(
        id="merge-generic-v0",
        entry_point="highway_env.envs.merge_env:MergeGenericEnv",
    )
    register(
        id="merge-generic-v1",
        entry_point="highway_env.envs.merge_env:ConnectedLaneMergeGenericEnv",
    )

    # 泊车场景：parking_env.py
    register(
        id="parking-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnv",
    )

    register(
        id="parking-ActionRepeat-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvActionRepeat",
    )

    register(
        id="parking-parked-v0",
        entry_point="highway_env.envs.parking_env:ParkingEnvParkedVehicles",
    )

    # 赛道场景：racetrack_env.py
    register(
        id="racetrack-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnv",
    )
    register(
        id="racetrack-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnv",
    )
    register(
        id="racetrack-large-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvLarge",
    )
    register(
        id="racetrack-large-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnvLarge",
    )
    register(
        id="racetrack-oval-v0",
        entry_point="highway_env.envs.racetrack_env:RacetrackEnvOval",
    )
    register(
        id="racetrack-oval-v1",
        entry_point="highway_env.envs.racetrack_env:ConnectedLaneRacetrackEnvOval",
    )

    # 环岛场景：roundabout_env.py
    register(
        id="roundabout-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutEnv",
    )
    register(
        id="roundabout-v1",
        entry_point="highway_env.envs.roundabout_env:ConnectedLaneRoundaboutEnv",
    )
    register(
        id="roundabout-generic-v0",
        entry_point="highway_env.envs.roundabout_env:RoundaboutGenericEnv",
    )
    register(
        id="roundabout-generic-v1",
        entry_point="highway_env.envs.roundabout_env:ConnectedLaneRoundaboutGenericEnv",
    )

    # 双向道路场景：two_way_env.py
    register(
        id="two-way-v0",
        entry_point="highway_env.envs.two_way_env:TwoWayEnv",
        max_episode_steps=15,
    )

    # 掉头场景：u_turn_env.py
    register(id="u-turn-v0", entry_point="highway_env.envs.u_turn_env:UTurnEnv")
    register(
        id="u-turn-v1",
        entry_point="highway_env.envs.u_turn_env:ConnectedLaneUTurnEnv",
    )

    # 随机道路场景：random_road_env.py
    register(
        id="random-road-v0",
        entry_point="highway_env.envs.random_road_env:RandomRoadEnv",
    )


_register_highway_envs()
