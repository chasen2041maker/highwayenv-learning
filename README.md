# highwayenv-learning

通过 HighwayEnv 仿真，从实践学习自动驾驶决策、车辆控制和强化学习。

这是个人学习仓库，基于 [Farama Foundation / HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) 源码开展练习，保留上游历史和 [MIT 许可证](LICENSE)。当前练习是根据前车距离选择加速、减速与保持速度的规则驾驶程序。

## 与理论仓库怎样配合

**这里负责动手、运行和实验；[vla_basic](https://github.com/chasen2041maker/vla_basic) 负责原理讲解与知识沉淀。两个仓库只走一条学习主线。**

当前课题、运行结果、理解证据和下一步统一记录在本仓库 [PROGRESS.md](PROGRESS.md)，不在理论库复制第二份进度。想知道某个现象背后的原因，从 [理论反向索引](learning/THEORY_LINKS.md) 进入对应讲义；讲完再回到现有代码，不重新做一套旧 H001。

当前驾驶程序、环境和运行方式保持不变。讲义建立不等于实验完成或已经掌握，也不要求先学完所有理论才能动手。

## 学习入口

- [运行脚本](demo.py)：车辆仿真、距离判断、目标速度档位和每局成绩单。
- [当前学习进度](PROGRESS.md)：已做实验、待核对内容和下次接续点。
- [理论反向索引](learning/THEORY_LINKS.md)：从当前代码问题跳到 vla_basic 的对应讲义。
- [学习路线](learning/ROADMAP.md)：从实践读源码，到规则策略、实验评测与强化学习。
- [历史学习记录](learning/LEARNING_LOG.md)：按执行者区分的实验与理解证据。
- [教学约定](AGENTS.md)与[进度维护技能](.agents/skills/highway-learning-progress/SKILL.md)：每个教学小节或新实验结果后的记录方式。

## 运行当前练习

在已安装 Python 3.10 或以上及依赖的环境中，于仓库根目录执行：

```shell
python -m pip install -e .
python demo.py
```

Windows 使用 Conda 时，如果 `noise` 安装报缺少 C++ 编译工具，可先运行 `conda install -c conda-forge noise`，再重新安装项目。当前本机环境与安装记录见 [PROGRESS.md](PROGRESS.md)。

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
uv add highway-env          # adds to project dependencies and installs (preferred)
uv pip install highway-env  # or install without adding to a project (pip install)
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

# Initialise the environment
env = gym.make("highway-v0", config={"lanes_count": 3}, render_mode="human")

# Reset the environment to generate the first observation
obs, info = env.reset()
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    obs, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
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
