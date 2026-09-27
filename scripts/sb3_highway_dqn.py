import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env  # noqa: F401


TRAIN = True

if __name__ == "__main__":
    # 创建环境
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # 创建模型
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",
    )

    # 训练模型
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_dqn/model")
        del model

    # 运行训练好的模型并录制视频
    model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # 提高渲染帧率
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # 预测动作
            action, _states = model.predict(obs, deterministic=True)
            # 执行动作并获取奖励
            obs, reward, done, truncated, info = env.step(action)
            # 渲染
            env.render()
    env.close()
