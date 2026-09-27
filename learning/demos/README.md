> **旧实验来源保留。** 活动版本已归入 [vla_basic/experiments/highway_driving/demos](https://github.com/chasen2041maker/vla_basic/blob/main/experiments/highway_driving/demos/README.md)，今后在那里改写和运行课程实验。下方命令只作旧目录参考，本目录不再增加新课题；唯一进度在 [vla_basic/PROGRESS.md](https://github.com/chasen2041maker/vla_basic/blob/main/PROGRESS.md)。

# 按课题分开的驾驶实验

每个小实验对应一个问题和一小段源码。以后按课题新增文件，不把所有练习不断堆进根目录的 `demo.py`。这里是实验索引和说明，当前学习状态统一记录在 [PROGRESS.md](../../PROGRESS.md)。

按学习者偏好，后续学习代码由助手在对话中分段列出，学习者亲手创建、输入和运行。本目录中助手已经准备的 `01_lane_change.py` 是参考示例，不代表学习者已完成；不要求在源码尚未读到相关内容时提前运行。

| 实验 | 要研究的问题 | 对应源码 |
| --- | --- | --- |
| [已有跟车练习](../../demo.py) | 怎样从观察中找前车，并选择加速、保持或减速？ | [观察](../../highway_env/envs/common/observation.py)、[动作](../../highway_env/envs/common/action.py) |
| [01：单次向右变道](01_lane_change.py) | 变道指令怎样改变目标车道，实际位置怎样跟过去？ | [动作映射与转发](../../highway_env/envs/common/action.py)、[车辆控制器](../../highway_env/vehicle/controller.py) |
| [02：查看动作空间](02_action_space.py) | 环境接受哪些动作编号，怎样检查取值是否在范围内？ | [动作空间定义](../../highway_env/envs/common/action.py) |
| [03：连续动作与单步速度](03_continuous_action.py) | 归一化输入怎样换算成控制量，一步后速度怎样变化？ | [ContinuousAction](../../highway_env/envs/common/action.py) |

## 实验 01：源码读到这些位置时再验证

当前先认识 `common` 目录，再从 `action.py` 文件开头分段读；下面是本实验对应的实现位置，不是要求跳过前面的源码。

1. 打开 `highway_env/envs/common/action.py`，找到 `DiscreteMetaAction.ACTIONS_ALL` 和它的 `act()`：数字 `2` 对应 `LANE_RIGHT`，然后转交车辆执行。
2. 打开 `highway_env/vehicle/controller.py`，找到 `ControlledVehicle.act()` 中的 `elif action == "LANE_RIGHT"`。`_id + 1` 选择右侧目标车道；`np.clip` 把编号限制在道路范围内。随后 `steering_control(self.target_lane_index)` 计算转向，让车辆靠近目标车道中心。

本课只解释“目标车道改变 → 转向 → 实际位置逐渐改变”，暂不展开转向计算的公式。

## 在 VS Code 运行

在项目根目录、已经激活 `py310` 的终端执行：

```powershell
python .\learning\demos\01_lane_change.py
```

也可以在打开该文件时使用 VS Code 的“运行 Python 文件”，解释器选择 `py310`。

实验条件：`highway-v0`，4 条车道、0 辆其他车辆，起始车道编号 1，种子 0，决策频率 1 Hz，仿真时限 10 秒。速度档位沿用环境默认值；实验不主动加减速。`main()` 默认显示窗口，`main(render_mode=None)` 可以用于不显示窗口的独立检查。

## 对着画面和输出看

- 车道从左到右编号为 0、1、2、3，编号 1 是从左数第二条。当前直路的车道中心 y 依次为 0、4、8、12 米。
- 前两次动作都是 1；第 3 次动作是 2，请求向右变道一次；之后恢复动作 1，继续朝已经设置的目标车道行驶。
- 目标车道从 1 变为 2，实际横向位置 y 从 4 米逐渐靠近 8 米。目标改变与实际到位不是同一时刻。
- `env.unwrapped.vehicle` 是模拟器内部诊断信息，仅用于打印。这一实验按固定时机发出动作，没有用内部数据判断交通。
- 动作 1 会继续跟随已有目标，因此转弯不需要每一步都重复发动作 2；重复发动作 2 可能继续把目标推向更右侧车道。

每一组输出是执行该动作、推进约 1 秒仿真后的状态。达到 10 秒时限会结束；提前按 Ctrl+C 是手动停止，不计为跑完实验。

运行后提供第 2～5 次决策的输出，观察目标车道和实际 y 的变化。运行成功可以证明指令被执行；是否理解调用过程，结合你自己的解释记录。

本实验验证单次变道的执行，不包含“遇到慢车，判断是否可以变道”的策略。后续再引入前车与相邻车道前后车辆的观察，并比较跟随和换道的选择。
