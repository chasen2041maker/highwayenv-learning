"""实验 01：发送一次向右变道指令，观察目标车道和实际位置。"""

import gymnasium as gym
import highway_env  # 导入后，Gymnasium 才认识 highway-v0


def main(render_mode="human"):
    # 四车道、无其他车辆，先单独观察变道怎样执行。
    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config={
            "lanes_count": 4,
            "vehicles_count": 5,
            "initial_lane_id": 1,
            "duration": 400,
            "policy_frequency": 1,
            "real_time_rendering": True,  # 新增：按实际时间显示仿真过程
        },
    )

    try:
        obs, info = env.reset(seed=0)
        cfg = env.unwrapped.config
        print("车道、背景车、时长、仿真频率：", cfg["lanes_count"], cfg["vehicles_count"], cfg["duration"], cfg["simulation_frequency"])
        print("动作空间：", env.action_space)
        print("观察空间：", env.observation_space)
        print("obs 形状：", obs.shape)
        print("车辆数配置：", env.unwrapped.config["vehicles_count"])
        print("感知距离：", env.unwrapped.PERCEPTION_DISTANCE, "米")
        print("reset 后 time 和 steps：", env.unwrapped.time, env.unwrapped.steps)

        # 下列内部数据只用于观察模拟器，不参与驾驶决策。
        vehicle = env.unwrapped.vehicle
        print("车道从左到右编号为 0、1、2、3；本次从编号 1 出发。")
        print("编号 1 的车道中心 y=4 米，编号 2 的中心 y=8 米。")
        print("初始横向位置 y：", round(float(vehicle.position[1]), 2), "米")

        for step in range(10):
            # step 从 0 开始：第 3 次决策只发一次向右变道指令。
            if step == 2:
                action = 2
                decision = "向右变道一次"
            else:
                action = 1
                decision = "保持目标车道和目标速度"

            print("\n第", step + 1, "次决策：", decision, "，动作编号：", action)
            obs, reward, terminated, truncated, info = env.step(action)

            # 目标会先改变，实际位置需要随车辆运动逐渐靠近目标。
            print("行动后目标车道编号：", vehicle.target_lane_index[2])
            print("行动后横向位置 y：", round(float(vehicle.position[1]), 2), "米")

            if terminated or truncated:
                if info["crashed"]:
                    print("结束：发生碰撞。")
                elif truncated:
                    print("结束：达到 10 秒仿真时间上限。")
                else:
                    print("结束：环境终止。")
                break

    except KeyboardInterrupt:
        print("\n已手动停止实验。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
