(environments-highway)=

```{eval-rst}
.. currentmodule:: highway_env.envs.highway_env
```

# Highway

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles. The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

```{figure} ../_static/animations/environments/highway.gif
:align: center
:name: fig:highway_env
:width: 80%
```

## Usage

```python
env = gym.make("highway-v0")
```

## Default configuration

```python
{
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,  # 单位：秒
    "ego_spacing": 2,
    "vehicles_density": 1,
    "collision_reward": -1,  # 与车辆发生碰撞时得到的奖励。
    "right_lane_reward": 0.1,  # 在最右侧车道行驶时获得的奖励，向其他车道线性递减到零。
    "high_speed_reward": 0.4,  # 以最高奖励速度行驶时的奖励，较低速度按 config["reward_speed_range"] 线性映射。
    "lane_change_reward": 0,  # 每次变道动作得到的奖励。
    "reward_speed_range": [20, 30],  # 单位为米/秒；将此速度范围线性映射为 [0, 1] 的高速奖励。
    "normalize_reward": True,
    "offroad_terminal": False,
    "simulation_frequency": 15,  # 单位：赫兹
    "policy_frequency": 1,  # 单位：赫兹
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # 单位：像素
    "screen_height": 150,  # 单位：像素
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}
```

More specifically, it is defined in:

```{eval-rst}
.. automethod:: HighwayEnv.default_config
    :no-index:
```

## Faster variant

A faster (x15 speedup) variant is also available with:

```python
env = gym.make("highway-fast-v0")
```

The details of this variant are described [here](https://github.com/Farama-Foundation/HighwayEnv/issues/223).

## API

```{eval-rst}
.. autoclass:: HighwayEnv
    :members:
```
