# 从 HighwayEnv 实践入门自动驾驶

这份路线说明方向和可观察的学习成果。当前走到哪里只看根目录 [PROGRESS.md](../PROGRESS.md)。学习者直接从实践学，Python 和领域概念在实验中补充，各阶段允许交叉推进。

**本仓库负责实际实现与实验；[vla_basic](https://github.com/chasen2041maker/vla_basic) 负责理论讲解。** 通过 [理论反向索引](THEORY_LINKS.md) 找对应讲义，不在两边重复排课，也不改变下面已有实践方向。

## 1. 读懂并控制自己的演示程序

从 `demo.py` 的实际现象理解创建环境、`reset`、`obs`、`action`、`step` 和结束标记。通过固定动作、距离规则、速度档位、日志等小变化，练习条件判断、循环、列表、字典和函数。

可观察成果：能解释一次“观察—决策—执行—重新观察”的过程，自己修改一条已经讲过的规则，并解释运行结果。

需要原理时看：[目标速度与控制](https://github.com/chasen2041maker/vla_basic/blob/main/theory/01-target-speed-and-control.md)、[step 与反馈](https://github.com/chasen2041maker/vla_basic/blob/main/theory/02-step-and-feedback.md)。这是材料对应，不是新增两门先修课。

## 2. 带着问题进入源码

沿实际调用过程查找答案，每次只读本课相关的小段：

| 问题 | 源码入口 |
| --- | --- |
| 环境名称如何找到场景？ | `highway_env/__init__.py` 的环境注册 |
| 道路、车辆、奖励和结束条件从哪里来？ | `highway_env/envs/highway_env.py` |
| reset 和 step 分别做了什么，一步过去多久？ | `highway_env/envs/common/abstract.py` |
| 观察表的单位、坐标和车辆选择是什么？ | `highway_env/envs/common/observation.py` |
| 1、3、4 怎样变成车辆动作？ | `highway_env/envs/common/action.py` |
| 目标速度怎样影响实际运动？ | `highway_env/vehicle/controller.py`、`kinematics.py` |

可观察成果：能为一个现象找到相关函数或配置，并完成一个有依据的小实验。源码顺序由问题决定，不要求逐页读完整仓库。

## 3. 自己构建可解释的驾驶规则

把已有距离判断逐步发展为考虑速度差的跟车规则，理解距离、相对速度、接近时间以及目标速度变化。再学习相邻车道前后车辆检查和变道条件。

变道前先核对观察是否包含后车、道路边界和决策频率；“允许执行某动作”不等于“已经确认该动作安全”。每次修改一个主要因素，保留能够复现失败的条件。

可观察成果：学习者能够独立完成一个有限规则变化，并解释输入、判断、动作以及失败例子。需要拆成函数时再介绍函数和文件组织，不提前建立复杂策略框架。

对应原理：[观察表、相对运动与规则边界](https://github.com/chasen2041maker/vla_basic/blob/main/theory/03-observation-and-relative-motion.md)。实际规则仍只在这里修改。

## 4. 用数据比较策略

把实验改为可复现的条件：固定种子集合和配置，记录碰撞、运行时间、速度与累计奖励。学习区别脚本停止、碰撞、时间上限和未结束回合，比较不同阈值和规则。

可观察成果：对同一组场景给出对比结果，说明收益、代价和证据局限。调参数使用的场景与最终检查场景应分开；不能用少量顺利回合概括所有路况。

## 5. 训练与评估第一个驾驶模型

在已经熟悉的环境上解释状态、动作、奖励、探索、回合和训练过程，再读取项目现有 DQN 示例；到实际需要时核实训练依赖和 API，不提前安装大批库。

运行一个规模合适的训练实验，保存模型，重新加载并在固定场景集合上测试。比较随机动作、手写规则和训练策略，保留失败案例并解释可能原因。

可观察成果：能自己启动和检查训练，加载模型评估，并解释奖励提高与碰撞、速度等指标之间的关系。

## 后续拓展

完成上述实践后，根据学习者兴趣拓展感知、定位、轨迹规划、控制以及驾驶 VLM/VLA。当前获得的状态表是模拟器提供的运动学数据；这些拓展有各自的新输入、任务和验证方法，不能用俯视画面或规则车代替。

理论方向保留于 [vla_basic 知识路线](https://github.com/chasen2041maker/vla_basic/blob/main/ROADMAP.md)。如果具体任务超出 HighwayEnv 的适用范围，届时再选择实际需要的实践环境；不把理论库重新变成第二套实验工程。

历史教学组织参考：[vla_basic 改造前的项目实践方法](https://github.com/chasen2041maker/vla_basic/blob/4503a9cc9d0b67add5e85c28aa5d75e73f2725a7/LEARNING_METHOD.md)。当前内容、节奏与状态仍按本学习者的实际实验安排。
