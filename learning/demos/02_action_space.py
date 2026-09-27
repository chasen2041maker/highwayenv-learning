import gymnasium as gym
import highway_env

env = gym.make("highway-v0")

try:
    print("动作区间： ", env.action_space)
    print("编号 4 在范围内吗：", env.action_space.contains(4))
    print("编号 5 在范围内吗：", env.action_space.contains(5))
finally:
    env.close()