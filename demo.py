import gymnasium as gym
import highway_env

# 创建高速公路场景，并显示窗口
#env = gym.make("highway-v0", render_mode="human")

env = gym.make(
    "highway-v0",
    render_mode="human",
    config={
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        }
    },
)

try:
    # 初始化车辆和道路
    obs, info = env.reset()
    episode = 1  # 当前是第几局
    total_reward = 0   # 这一局的累计得分

    for _ in range(300):
        """
        # 随机选择驾驶动作
        action = 1

        # 逐辆检查附近的车，跳过第一行自己的车
        for presense, x, y, vx, vy in obs[1:]:
            distance = x * 200
            side_distance = y * 16

            if (presense == 1 
                and 0 < distance < 40
                and abs(side_distance) < 2
                and vx < 0):
                action = 4
                print("发现同车道出现慢车，发出减速指令！")
                break
        """

        # 先当作没有观察到同车道前车
        nearest_distance = float('inf')

        # 找到观察范围内，同车道最近的前车
        for presense, x, y, vx, vy in obs[1:]:
            distance = x * 200
            side_distance = y * 16

            if presense == 1 and distance > 0 and abs(side_distance) < 2:
                nearest_distance = min(nearest_distance,distance)

        # 根据最近前车的距离，选择动作
        if nearest_distance < 40:
            action = 4
            decision = "前车太近，减速"

        elif nearest_distance > 60:
            action = 3
            decision = "前方距离充足，加速"

        else:
            action = 1
            decision = "保持当前目标速度"

        print(decision)



        # 让车辆执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        print("当前速度：", round(info["speed"] * 3.6, 1), "公里/小时")
        print("目标速度：", env.unwrapped.vehicle.target_speed * 3.6, "公里/小时")
        total_reward += reward

        """"
        # 碰撞或时间到了，就重新开始
        if terminated or truncated:
            obs, info = env.reset()
            print("车辆观察数据：\n", obs)
        """

        if terminated or truncated:
            print("\n========== 本局成绩 ==========")
            print("第", episode, "局")

            if info["crashed"]:
                print("结果：发生碰撞")
            else:
                print("结果：到达本局时间上限")

            print("本局仿真时间：", round(env.unwrapped.time, 1), "秒")
            print("累计奖励：", round(total_reward, 2))
            print("==============================\n")

            # 准备下一局
            episode += 1
            total_reward = 0
            obs, info = env.reset()

except KeyboardInterrupt:
    pass
finally:
    env.close()