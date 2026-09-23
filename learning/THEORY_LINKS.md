# 理论反向索引｜从实验回到原理

这里是轻量导航，不复制讲义或另建进度。当前做什么、已运行什么、理解到哪里，始终看 [本仓库 PROGRESS.md](../PROGRESS.md)。

理论正文与详细对应关系分别在 [vla_basic 讲义目录](https://github.com/chasen2041maker/vla_basic/blob/main/theory/README.md) 和 [理论—实践对应表](https://github.com/chasen2041maker/vla_basic/blob/main/PRACTICE_MAP.md)。

| 你在程序里遇到的问题 | 去读什么 | 回到哪里验证 |
| --- | --- | --- |
| SLOWER 改了什么，目标与实际为何不同 | [01 目标速度与控制器](https://github.com/chasen2041maker/vla_basic/blob/main/theory/01-target-speed-and-control.md) | [demo.py](../demo.py) 的档位配置和速度打印；[控制器](../highway_env/vehicle/controller.py) |
| 一次 step 过多久，为什么要重新看 obs | [02 step 与反馈](https://github.com/chasen2041maker/vla_basic/blob/main/theory/02-step-and-feedback.md) | [demo 主循环](../demo.py) 与 [AbstractEnv](../highway_env/envs/common/abstract.py) |
| 表格数值怎样换算，只看距离有什么不足 | [03 观察与相对运动](https://github.com/chasen2041maker/vla_basic/blob/main/theory/03-observation-and-relative-motion.md) | [车辆筛选规则](../demo.py) 与 [观察生成](../highway_env/envs/common/observation.py) |

这里的三行是材料对应，不是三个同时激活的实验，也不意味着已经讲完。代码已经有的修改不重复要求；尚无运行或理解证据的部分继续待核对。

以后实践增加新主题，再补对应理论与链接。实际程序、配置、日志、训练和评测留在本仓库；可复用的原理解释写入 vla_basic。旧 vla_basic H001 只作历史参考，不再要求重新通关。

版本注意：本仓库与旧运行器在观察归一化、频率、时限和速度档位上不同。借鉴日志方法可以，复制常量前必须核对 [配置差异说明](https://github.com/chasen2041maker/vla_basic/blob/main/PRACTICE_MAP.md)。
