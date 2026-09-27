"""测试 highway-env 环境能否在多进程模式（forkserver/spawn）下工作。

使用 stable-baselines3 的 SubprocVecEnv 或类似的向量化环境包装器时，
子进程通过 ``forkserver`` 或 ``spawn`` 启动，**不会**继承父进程的
``import highway_env`` 操作。

Gymnasium 的 ``module:env_name`` 语法，例如 ``"highway_env:highway-v0"``，
会在子进程中触发模块导入，从而按需注册环境。

参考：https://github.com/Farama-Foundation/HighwayEnv/issues/648
"""

import multiprocessing as mp

import gymnasium as gym
import pytest


def _make_env_in_subprocess(env_id: str, result_queue: mp.Queue) -> None:
    """在子进程中创建环境并推进一步，事先不导入 highway_env。"""
    try:
        env = gym.make(env_id)
        obs, _info = env.reset()
        _obs, _reward, _terminated, _truncated, _info = env.step(
            env.action_space.sample()
        )
        env.close()
        result_queue.put(("ok", str(type(obs))))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


@pytest.mark.parametrize(
    "env_id",
    [
        "highway_env:highway-v0",
        "highway_env:highway-fast-v0",
        "highway_env:merge-v0",
        "highway_env:roundabout-v0",
        "highway_env:intersection-v0",
        "highway_env:parking-v0",
    ],
)
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_env_in_subprocess(env_id: str, start_method: str) -> None:
    """使用 module:name 语法，应能在 forkserver/spawn 子进程中创建环境。"""
    if start_method not in mp.get_all_start_methods():
        pytest.skip(f"{start_method} not available on this platform")

    ctx = mp.get_context(start_method)
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_make_env_in_subprocess, args=(env_id, q))
    p.start()
    p.join(timeout=30)

    assert not q.empty(), "Subprocess produced no result (likely crashed)"
    status, detail = q.get()
    assert status == "ok", f"Subprocess failed: {detail}"
