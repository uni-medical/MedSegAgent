# nnInteractive 接入与验收

本版本属于 **MIA-Omni**（简称 MIA）的独立开发线。产品目标是统一医学影像的理解、
模型调用、交互修正、测量与报告；当前实现以 nnInteractive 区域交互为起点。

开发分支为 `mia-omni`，仅同步到现有 MedSegAgent GitHub 仓库的同名分支。
服务器及现有公网服务继续使用 `totalseg-agent` 发布线；该分支不混入本版本的
nnInteractive / MedGemma 接入代码，不从 MIA-Omni 开发分支更新现有服务。
两条分支独立维护，不自动合并或切换部署。

这个版本在现有 NiiVue 查看器中增加“区域交互”：前景点、背景点、单层框、
待应用标记撤销/清空、不可变掩膜版本、体积与父版本比较、下载和工作区恢复。
Agent 可以应用用户记录的标记、检查真实版本、解释测量变化；不能自行定位未标出的器官，
不能看见影像像素或判断分割质量。
单个 Agent 任务最多 6 轮工具调用循环、一次修改尝试；体积比较限所选结果及其直接父版本。

MedGemma 已按当前优先级移出接入范围。这里没有报告生成或新的模态判断功能。

## 与现有服务的关系

- 默认关闭。只有 `MEDSEGAGENT_OMNI_ENABLED=1` 才构造扩展存储、注册 Web API、加载界面模块。
- 不改变旧数据库表、TotalSegmentator 工具、CLI/MCP/A2A 的能力注册。
- 扩展工作区和任务位于所选 data root 的 `omni/`；保留冻结原图、提示历史和每版掩膜。
- 模型由独立 Python 子进程执行，环境见 [独立模型环境](../ops/nninteractive/README.md)。
- 同一 GPU 主机必须使用已有调度器的同一个锁目录与全局并发设置。独立目录并不隔离显存。
  测试只应短暂使用空闲卡；没有资源时等待或停止，不修改生产 GPU 白名单。
- 公网切换属于后续发布。已有服务若含未提交代码，应保存实际部署文件及一致性数据快照；
  不能用 Git HEAD 冒充可恢复基线。

## 私有测试启动

在独立 checkout 中准备主服务环境和模型环境，不在公网 checkout 运行安装：

```sh
uv sync --frozen --group dev
uv sync --locked --project ops/nninteractive --python 3.12
```

从官方模型仓库下载固定版本；此步骤与服务启动分开：

```sh
hf download MIC-DKFZ/nnInteractive \
  --revision 3f308d751c00644e4fde6f09c600264b393b21b5 \
  --include 'nnInteractive_v1.0/**' --local-dir /path/to/test-models
```

该 checkpoint 的 SHA256：
`b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9`。
权重采用 CC BY-NC-SA 4.0；软件代码 Apache-2.0 不替代权重许可。

```sh
MEDSEGAGENT_OMNI_ENABLED=1 \
MEDSEGAGENT_DATA_ROOT=/path/to/independent-test-data \
MEDSEGAGENT_PUBLIC_URL=http://127.0.0.1:8878 \
MEDSEGAGENT_NNINTERACTIVE_PYTHON="$PWD/ops/nninteractive/.venv/bin/python" \
MEDSEGAGENT_NNINTERACTIVE_MODEL_PATH=/path/to/test-models/nnInteractive_v1.0 \
uv run --frozen medsegagent serve --host 127.0.0.1 --port 8878
```

直接“应用标记”不需要规划模型。Agent 另需原项目支持的 `OPENAI_BASE_URL` 和
`OPENAI_API_KEY`，只在私有配置中设置。模型不在运行时自动下载。

## 用户流程

1. 上传或打开已有影像，展开“区域交互”，选择“编辑当前影像”。
2. 为一个区域命名，在二维切片上放前景点/背景点，或拖一个框。滚轮仍可切片；
   选择“浏览”或在查看器按 Esc 退出标记模式。
3. 点击“应用标记”，或输入“按我的标记修正并比较体积变化”交给 Agent。
4. 掩膜在原查看器回显，新版本保留父版本。选择旧版再提交会产生新分支，不覆盖旧文件。
5. 刷新后从“继续编辑”选择工作区，恢复冻结原图和版本。游客需保留同一浏览器会话；
   切换账号后看不到上个账号的数据。过期记录按现有保留期清理。

这里的“修正”针对本工作区的 nnInteractive 提示历史。首版没有把既有
TotalSegmentator 的某个标签自动导入为初始掩膜；底层 worker 的 seed 接口已支持，
但尚未向 Web 开放。当前为单图、单区域工作区，多区域管理留待后续。

## 坐标与结果合同

屏幕点先通过 NiiVue `canvasPos2frac` 和 `frac2mm(frac, 0, true)` 变为实际 affine 空间值，
再由后端用原 NIfTI affine 逆变换回原始 XYZ。不能把 NiiVue 的 RAS 重排体素坐标直接
交给 nibabel/nnInteractive。字段名虽然叫 `world`，其单位跟随原 affine，并不强行假定 mm。

框的四角必须对应一个原始体素平面的轴对齐矩形；固定轴吸附到一个切片中心。
不满足时明确拒绝并提示改用点选，不将其扩成错误的 3D 包围盒。
输入保持原强度、原网格；输出校验 shape、affine、spacing、空间单位、二值标签与 hash。
只有空间单位明确时才输出 mL，未知单位显示体素数。

每个工作区一次只运行一个任务；请求编号提供幂等重试，版本比较防止跨标签页覆盖。
取消会沿原有进程组清理链路释放 GPU 租约。规划模型接收用户任务文本、白名单测量与提示数量；
系统不自动附加像素、原始头、标记坐标、文件路径或区域名称。用户任务文本中的绝对路径会打码，
其他文字仍发送给已配置的规划模型，因此手工写入文本的姓名或坐标不在上述排除范围内。

## 首版性能边界

每次“应用”启动新进程、加载模型、按顺序重放全部提示。这个实现优先验证输入、结果、
取消和隔离；它不是常驻会话的低延迟交互。长影像与长提示历史的响应时间需要单独测量。
下一步优化是一个受独占租约管理的会话 worker：载入一次，追加提示，闲置后退出；
须同时验证取消、重连、会话淘汰和资源释放，不能只看第二次点击的速度。

## 验证

```sh
uv run --frozen pytest
uv run --frozen ruff check src tests ops
node --test tests/browser_state.cjs tests/interactive_geometry.mjs
git diff --check
```

自动检查覆盖旧功能回归、所有权、CSRF、过期、丢失 ACK、版本冲突、取消、重启、
原坐标、提示重放、工件校验和 Agent 虚假完成防护。假 worker 的测试不代表 checkpoint 推理。
真实 GPU 测试脚本见 `ops/nninteractive/canary.py`；其合成球体结果仅验证工程链路，
不能用来宣称 CT/MR 器官或病灶分割质量。真实病例质量、长体积显存峰值和连续交互延迟
仍是公开启用前的验收项。

### 本次隔离验收（2026-09-09）

- 开发分支已基于最新的中英文切换和标签本地化代码；区域交互面板复用原语言开关。
- 完整 Python 回归：1139 项通过；Node 浏览器状态与坐标检查：105 项通过；Ruff 与差异检查通过。
  第一次更新基线后的全量运行曾有一个原有遥测超时测试未等到子进程写出 PID；
  单独复查 34 项调度测试及再次完整运行均通过，未为此修改生产调度代码。
- Edge 实际界面验证了前景点、背景点、框选、掩膜回显、旧版分支、刷新恢复、取消保留标记，
  以及切换语言时保留区域名称、输入文本和待应用标记。
- 真实规划模型搭配测试分割器：背景点修正后显示体积从 64.11 mL 变为 32.93 mL，
  Agent 正确报告减少约 31.18 mL，与已保存版本的测量一致。该项不作为 nnInteractive 模型验证。
- 当前服务策略仍为每次请求冷启动并重放提示，尚无常驻会话延迟承诺。

### 真实 RTX 3090 测试

nnInteractive 2.5.1、PyTorch 2.8.0+cu126、Python 3.12.3，在一张 RTX 3090 上完成一次
真实 checkpoint 推理。输入为 `128 × 128 × 96` 合成球体，间距 `1.2 × 1.2 × 2.0 mm`，
带旋转及左手 affine；依次执行前景点、背景点、单层框。

| 测量 | 本次结果 |
| --- | ---: |
| 完整测试 worker 进程 | 29.765 秒 |
| 模型初始化（含上游 warmup） | 6.301 秒 |
| 前景点 / 背景点 / 框选 API | 0.432 / 0.280 / 0.276 秒 |
| Torch 峰值 allocated / reserved | 5.902 / 6.531 GiB |
| 进程 CPU 内存峰值 RSS | 1.583 GiB |
| 三阶段掩膜体素数 | 65,526 → 64,611 → 65,894 |

三个掩膜均为二值，shape、affine、spacing、单位及 qform/sform 的几何 hash 与原图相同；
最终输出与第三阶段一致。模型退出后 GPU 无残留计算进程，生产任务聚合计数前后一致。
首次检查因生产有任务而跳过，待空闲后才执行这一次实际推理，保留两次独立检查记录。

上述交互耗时来自同一进程内的一次模型加载，不能当作当前 Web 每次应用的端到端延迟。
首点还包含异步预处理的剩余等待；三次交互 API 也不等于三次网络 forward。
测试进程包含额外计时与阶段快照开销。这证明该小体积上的工程链路和资源可行性，
不证明真实 CT/MR 的分割质量，也不保证长体积采用同样的显存或耗时。
