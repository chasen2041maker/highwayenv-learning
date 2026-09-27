% multi_agent:

# The Multi-Agent setting

Most environments can be configured to a multi-agent version. Here is how:

## Increase the number of controlled vehicles

To that end, update the {ref}`environment configuration <configuration>` to increase `controlled_vehicles`

```{eval-rst}
.. jupyter-execute::
  :stderr:

  import gymnasium
  import highway_env

  env = gymnasium.make(
    "highway-v0",
    render_mode="rgb_array",
    config={
      "controlled_vehicles": 2,  # 两辆受控车辆
      "vehicles_count": 1,       # 为便于观察，只放置一辆其他车辆
    }
  )
  env.reset(seed=0)

  from matplotlib import pyplot as plt
  %matplotlib inline
  plt.imshow(env.render())
  plt.title("Controlled vehicles are in green")
  plt.show()
```

## Change the action space

Right now, since the action space has not been changed, only the first vehicle is controlled by `env.step(action)`.
In order for the environment to accept a tuple of actions, its action type must be set to {py:class}`~highway_env.envs.common.action.MultiAgentAction`
The type of actions contained in the tuple must be described by a standard {ref}`action configuration <actions>` in the `action_config` field.

```{eval-rst}
.. jupyter-execute::
  :stderr:

  env.unwrapped.config.update({
    "action": {
      "type": "MultiAgentAction",
      "action_config": {
        "type": "DiscreteMetaAction",
      }
    }
  })
  env.reset()

  _, (ax1, ax2) = plt.subplots(nrows=2)
  ax1.imshow(env.render())
  ax1.set_title("Initial state")

  # 让第一辆车向左变道，第二辆车向右变道
  action_1, action_2 = 0, 2  # 参阅 highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL
  env.step((action_1, action_2))

  ax2.imshow(env.render())
  ax2.set_title("After sending actions to each vehicle")
  plt.show()

```

## Change the observation space

In order to actually decide what `action_1` and `action_2` should be, both vehicles must generate their own observations.
As before, since the observation space has not been changed no far, the observation only includes that of the first vehicle.

In order for the environment to return a tuple of observations -- one for each agent --, its observation type must be set to {py:class}`~highway_env.envs.common.observation.MultiAgentObservation`
The type of observations contained in the tuple must be described by a standard {ref}`observation configuration <observations>` in the `observation_config` field.

```{eval-rst}
.. jupyter-execute::
  :stderr:

  env = gymnasium.make(
    "highway-v0",
    render_mode="rgb_array",
    config={
      "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
          "type": "Kinematics",
        }
      }
    }
  )
  obs, info = env.reset()

  import pprint
  pprint.pprint(obs)
```

## Wrapping it up

Here is a pseudo-code example of how a centralized multi-agent policy could be trained:

```{eval-rst}
.. jupyter-execute::
  :stderr:

  # 多智能体环境配置
  env.unwrapped.config.update({
    "controlled_vehicles": 2,
    "observation": {
      "type": "MultiAgentObservation",
      "observation_config": {
        "type": "Kinematics",
      }
    },
    "action": {
      "type": "MultiAgentAction",
      "action_config": {
        "type": "DiscreteMetaAction",
      }
    }
  })

  # 用于示意的强化学习算法
  class Model:
    """ 强化学习算法的示意代码，根据观察预测动作，
    并根据观测到的状态转移更新模型。"""

    def predict(self, obs):
      return 0

    def update(self, obs, action, next_obs, reward, info, done, truncated):
      pass
  model = Model()

  # 一个训练回合
  obs, info = env.reset()
  done = truncated = False
  while not (done or truncated):
    # 将观察传给模型，得到动作元组
    action = tuple(model.predict(obs_i) for obs_i in obs)
    # 执行动作
    next_obs, reward, done, truncated, info = env.step(action)
    # 根据各个智能体观测到的状态转移更新模型
    for obs_i, action_i, next_obs_i in zip(obs, action, next_obs):
      model.update(obs_i, action_i, next_obs_i, reward, info, done, truncated)
    obs = next_obs

```

For example, this is supported by [eleurent/rl-agents](https://github.com/eleurent/rl-agents)'s DQN implementation, and can be run with

```bash
cd <path/to/rl-agents/scripts>
python experiments.py evaluate configs/IntersectionEnv/env_multi_agent.json \
                               configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json \
                               --train --episodes=3000
```

```{figure} _static/animations/environments/intersection_multi_agent.gif
Video of a multi-agent episode with the trained policy.
```
