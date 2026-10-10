"""实验 04：前车太近就减速。"""


import gymnasium as gym
import highway_env

USE_RULE = True
SAFE_GAP = 30

def front_gap(obs):
    """在观察表里找同车道、在我前面、最近的那辆车，返回距离（米）。没有就返回 None。"""
    best = None
    for row in obs[1:]: # 第 0 行是自车，从第 1 行开始是其他车
        presence, x, y, vx, vy = row    # x、y 是相对自车的距离
        if presence == 1 and x > 0 and abs(y) < 2: # 存在、在前面、横向差不到半条车道
            if best is None or x < best:
                best = x
    return best

env = gym.make(
    "highway-fast-v0",
    render_mode = "human",
    config={
        "lanes_count": 4,
        "vehicles_count": 5,
        "initial_lane_id": 3,
        "duration": 400,
        "policy_frequency": 1,
        "real_time_rendering": True,
        "observation": {"type": "Kinematics", "normalize": False},
        #"action": {"type": "DiscreteMetaAction", "target_speeds": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]},
    }
)

obs, info = env.reset(seed=1)
try:
    for step in range(40):
        gap = front_gap(obs)
        safe_gap = 10 + obs[0][3] * 1.5
        if USE_RULE and gap is not None and gap < safe_gap:
            action = 4  #slower
        elif gap is None or gap > safe_gap + 10:
            action = 3  #faster
        else:
            action = 1
        obs,reward,terminated,truncated, info = env.step(action)
        print("第", step + 1, "步  前车距离：", None if gap is None else round(gap, 1),
            "米  想留：", round(safe_gap, 1),
            "米  动作：", action, "  车速：", round(info["speed"], 1), "  奖励：", round(reward, 3))
        front, rear = env.unwrapped.road.neighbour_vehicles(env.unwrapped.vehicle)
        if front:
            print("    前车：", type(front).__name__, " 速度：", round(front.speed, 2), " 目标速度：", round(front.target_speed, 2))
        if terminated:
            print("撞车了！！")
            break
except KeyboardInterrupt:
    print("手动停止")
finally:
    env.close()