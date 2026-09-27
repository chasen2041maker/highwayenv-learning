# HighwayEnv｜主项目使用的模拟器源码

基于 [Farama Foundation / HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)，保留上游历史与 [MIT 许可证](LICENSE)，供 vla_basic 的早期驾驶交互与控制阶段使用。

**完整智驾课程、学习者实验、评测证据与唯一进度都在 [vla_basic](https://github.com/chasen2041maker/vla_basic/blob/main/README.md)。** 这里保留 `highway_env/` 模拟器实现，按需读动作、控制器与运动源码。未来视觉、轨迹、VLA 和部署阶段继续由主项目组织，不以本模拟器限定课程范围。

- [当前学习进度](https://github.com/chasen2041maker/vla_basic/blob/main/PROGRESS.md)
- [课程实验与运行方式](https://github.com/chasen2041maker/vla_basic/blob/main/experiments/highway_driving/demos/README.md)
- [章节路线](https://github.com/chasen2041maker/vla_basic/blob/main/learning/BOOK.zh-CN.md)
- [迁移前历史](https://github.com/chasen2041maker/vla_basic/blob/main/archive/notes/2026-09-27-highway-learning-history.md)

原 `demo.py`、`learning/demos/` 与记录保留为历史来源，不再作为另一套活动课程。现有 py310 环境仍通过安装导入这里的源码；用户无需因主项目归属调整重装依赖。

## HighwayEnv 上游说明

[![Python](https://img.shields.io/pypi/pyversions/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![PyPI](https://badge.fury.io/py/highway-env.svg)](https://badge.fury.io/py/highway-env)
[![build](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml/badge.svg)](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build.yml)
[![pre-commit](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/pre-commit.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://highway-env.farama.org/" target="_blank">
        <img src="https://highway-env.farama.org/main/_static/img/highway-text.png" width="500px" />
    </a>
</p>

<p align="center">
    <img src="https://highway-env.farama.org/main/_static/animations/highway-env.gif"><br/>
    <em>An episode of one of the environments available in HighwayEnv.</em>
</p>

A collection of environments for autonomous driving and tactical decision-making tasks. Originally developed by [Edouard Leurent](https://github.com/eleurent) and currently maintained by [Jin Huang](https://github.com/Trenza1ore).

The documentation website is at [highway-env.farama.org](https://highway-env.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Installation

To install HighwayEnv, use:

```bash
pip install highway-env
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add highway-env          # 添加到项目依赖并安装（推荐）
uv pip install highway-env  # 或直接安装，不加入项目依赖（类似 pip install）
```

We support **Linux** and **macOS** primarily, with **Windows** support maintained on a best-effort basis.

## Environments

HighwayEnv includes 10 driving scenario families: `highway`, `intersection`, `exit`, `lane-keeping`, `merge`, `parking`, `racetrack`, `roundabout`, `two-way`, and `u-turn`, with several environments also offering fast, continuous-control, connected-lane, multi-agent, generic, large, or oval variants. The full list with descriptions and configuration options is available in the [documentation](https://highway-env.farama.org/main/environments/).

<details>
<summary>Previews</summary>

| | |
|:---|:---:|
| `highway` | ![highway](https://highway-env.farama.org/main/_static/animations/environments/highway.gif) |
| `merge` | ![merge](https://highway-env.farama.org/main/_static/animations/environments/merge-env.gif) |
| `roundabout` | ![roundabout](https://highway-env.farama.org/main/_static/animations/environments/roundabout-env.gif) |
| `parking` | ![parking](https://highway-env.farama.org/main/_static/animations/environments/parking-env.gif) |
| `intersection` | ![intersection](https://highway-env.farama.org/main/_static/animations/environments/intersection-env.gif) |
| `racetrack` | ![racetrack](https://highway-env.farama.org/main/_static/animations/environments/racetrack-env.gif) |
| `lane-keeping` | ![lane-keeping](https://highway-env.farama.org/main/_static/animations/environments/lane-keeping-env.gif) |
| `two-way` | ![two-way](https://highway-env.farama.org/main/_static/animations/environments/two-way-env.gif) |
| `exit` | ![exit](https://highway-env.farama.org/main/_static/animations/environments/exit-env.gif) |
| `u-turn` | ![u-turn](https://highway-env.farama.org/main/_static/animations/environments/u-turn-env.gif) |

</details>

## Usage

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

# 初始化环境
env = gym.make("highway-v0", config={"lanes_count": 3}, render_mode="human")

# 重置环境，生成第一次观察
obs, info = env.reset()
for _ in range(1000):
    # 在这里加入你的驾驶策略
    action = env.action_space.sample()

    # 用选定动作推进环境一步，
    # 获取下一次观察、奖励，以及回合是否终止或截断
    obs, reward, terminated, truncated, info = env.step(action)

    # 如果本局已结束，则重置环境开始新一局
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See the [documentation](https://highway-env.farama.org/quickstart/) for more examples including how to train agents with Stable Baselines3 and Google Colab notebooks. For examples of trained agents (DQN, DDPG, Value Iteration, MCTS), see the [Agent Examples](https://highway-env.farama.org/content/algorithms/) page.

## Documentation

Read the [documentation online](https://farama-foundation.github.io/HighwayEnv/).

## Development Roadmap

Here is the [roadmap](https://github.com/Farama-Foundation/HighwayEnv/issues/539) for future development work.

## Citating

If you use HighwayEnv in your work, please consider citing it with:

```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Farama-Foundation/HighwayEnv}},
}
```

## Publications

A list of publications using HighwayEnv can be found in the [documentation](https://highway-env.farama.org/main/content/publications/).
