---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/highway-text.png
:alt: HighwayEnv Logo
```

```{project-heading}
A collection of environments for autonomous driving and tactical decision-making tasks.
```

```{figure} _static/animations/highway-env.gif
:alt: Highway
:width: 500
```

**HighwayEnv** is a collection of [Gymnasium](https://gymnasium.farama.org/) environments for decision-making in autonomous driving. It features various driving scenarios such as highway cruising, merging, intersection crossing, parking, car racing, and more — all with configurable observations, actions, dynamics, and rewards:

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

This documentation provides:

1. a {ref}`quick start guide <quickstart>` describing the environments and how to get started;
2. a description of the available {ref}`environments <environments>` and their configuration options;
3. a {ref}`detailed guide <user_guide>` covering the nuts and bolts of the project, and how *you* can contribute.

(index-how-to-cite-this-work)=

# How to cite this work?

If you use this package, please consider citing it with this piece of
BibTeX:

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

```{toctree}
:hidden:
:caption: Getting Started

installation
quickstart
```

```{toctree}
:hidden:
:caption: Environments

environments/index
```

```{toctree}
:hidden:
:caption: User Guide

content/algorithms
user_guide
faq
```

```{toctree}
:hidden:
:caption: Reference

List of Publications <content/publications>
bibliography/index
```

```{toctree}
:hidden:
:caption: Development

GitHub <https://github.com/Farama-Foundation/HighwayEnv>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/HighwayEnv/blob/main/CONTRIBUTING.md>
```
