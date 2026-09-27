import gymnasium as gym
import numpy as np
import pytest

import highway_env


gym.register_envs(highway_env)


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_render(env_spec):
    env = gym.make(env_spec, render_mode="rgb_array").unwrapped
    env.reset()
    img = env.render()
    env.close()
    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.config["screen_height"],
        env.config["screen_width"],
        3,
    )  # (H,W,C)


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_obs_grayscale(env_spec, stack_size=4):
    env = gym.make(env_spec).unwrapped
    env.config.update(
        {
            "offscreen_rendering": True,
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (
                    env.config["screen_width"],
                    env.config["screen_height"],
                ),
                "stack_size": stack_size,
                "weights": [0.2989, 0.5870, 0.1140],
            },
        }
    )
    obs, info = env.reset()
    env.close()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (
        stack_size,
        env.config["screen_width"],
        env.config["screen_height"],
    )


@pytest.mark.parametrize("cells", [3, 16, 60, 61, 62, 123])
def test_render_lidar_observation(cells):
    """渲染激光雷达观察时，每个单元应绘制一个扇区。

    在 v1.12.2 之前，通过累加浮点步长计算角度；对于某些单元数量，
    会多生成一个角度，导致绘图逻辑发生 IndexError。
    """
    env = gym.make(
        "highway-v0",
        render_mode="rgb_array",
        config={"observation": {"type": "LidarObservation", "cells": cells}},
    ).unwrapped
    env.reset(seed=0)
    img = env.render()
    env.close()

    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.config["screen_height"],
        env.config["screen_width"],
        3,
    )
