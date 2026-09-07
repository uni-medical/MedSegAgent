# 补充分割模型候选与接入门槛

证据核查日期：**2026-09-07**。本文记录基于官方论文、发行包、源码、模型卡、许可证与公开数据元信息的工程选型。随后已接入并在 Mac MPS、pjoffice NVIDIA 和公网完成两个选中模型的真实 canary，见[验证记录](validation.md)；下文的研究阶段描述与真实验收证据分开记录。

## 本轮选择两个 CT 病灶任务

最终选择 **TotalSegmentator `lung_nodules` 与 `liver_lesions`**。二者补足基础器官分割缺少的肺结节与肝病灶能力，复用已有本地推理依赖、隔离目录、任务队列和结果契约。本轮不增加第三个模型。

选型时不以 PyPI 可下载、`device='mps'` 参数存在或论文成绩代替平台支持。实际支持范围由后续真实任务及其边界限定，所有 canary 仅用于工程验收。

| 优先级 | 候选与固定发行 | 实质新增能力 | 权重与资源证据 | 主要边界 |
| --- | --- | --- | --- | --- |
| 1 | `TotalSegmentator==2.18.0` / `lung_nodules` / Dataset913 | CT 肺结节体素掩膜；输出 `1=lung`、`2=lung_nodules` | 官方 2025-01-15 权重 ZIP **765,102,050 B**；1.5 mm 推理；先用肺叶粗分割裁剪；`fast=False` 必须保持 | 不是全身肿瘤模型，不区分良恶性。不能以肺组织非空代替结节检出。小结节可能漏检，正常扫描可为空掩膜。 |
| 2 | `TotalSegmentator==2.18.0` / `liver_lesions` / Dataset591 | CT 肝病灶体素掩膜；输出 `1=liver_lesions` | 官方 2026-03-16 更新权重 ZIP **230,940,114 B**；0.75 × 0.75 × 1 mm，`3d_fullres_high`；3 mm 肝脏定位裁剪；`fast=False` | 只在 CT 肝病灶范围内路由；不提供病理分类、肿瘤分期或良恶性判断。高分辨率裁剪与内存峰值必须实测。 |

版本、ID、标签、裁剪和禁用 `fast` 的依据分别是 [PyPI 2.18.0（2026-08-12 正式发行）][ts-pypi]、[固定版 task 配置][ts-config]、[标签映射][ts-labels]、[官方权重资产][ts-weights]。本机已安装的 `map_tasks_config.py` 与官方 `v2.18.0` 文件逐字节一致。GitHub 的 latest release 当前是 `v2.5.0-weights`，它是权重容器名称，**不是 Python 包版本**；同一权重 release 内的资产还存在 2026 年更新时间，不能只看 release 的 2025 年标题日期。

## 代码、权重、数据许可分别记录

TotalSegmentator 代码使用 [Apache-2.0][ts-license]；固定版本 README 在 “Openly available for any usage (Apache-2.0 license)” 下明确列出 `lung_nodules`、`liver_lesions`、`liver_lesions_mr`、`cerebral_bleed`、`kidney_cysts`。[开放任务列表][ts-readme] 与 [下载代码][ts-download] 一致，候选 ID 不走学术/商业许可服务器。研究阶段仅查询了权重资产元数据，**没有下载 ZIP 检查内部许可证或执行 checkpoint**；实际下载时仍需保存来源、哈希及许可快照。其他 TotalSegmentator 任务不自动继承此开放结论。

公开测试数据拥有独立条款：LIDC-IDRI 与标准化结节 SEG 为 CC BY 3.0；肝病灶 Zenodo 页面为 CC BY 4.0，但其 ZIP 内 `dataset.json` 的 `licence` 字段写 Apache 2.0。此数据声明存在差异，应记录两者并保留来源、署名和版本，不能把代码 Apache 许可当作整个数据集的唯一条款。[LIDC][lidc]、[LIDC 标准化 SEG][lidc-seg]、[肝 CT 数据集][liver-data]。

## MPS 与 NVIDIA：实际可行性来自哪一层

TotalSegmentator 官方文档明确给出 M 系列 Mac 的 `--device mps`，`python_api.py` 与 nnU-Net 的实际设备分支也接收 MPS；本研究核查的是已安装的 TotalSegmentator 2.18.0、nnunetv2 2.8.1、torch 2.14.0。nnU-Net 对非 CUDA 设备关闭 `perform_everything_on_device`，只在 CUDA 上使用 autocast，因此 MPS 路线可能把累计数据保留在 CPU，不能套用 CUDA 的显存与速度数字。[设备 API][ts-api]、[nnU-Net 2.8.1 推理源码][nnunet-infer]。

**没有找到上述两个 specialized task 的官方逐任务 MPS 最低内存或保证可用的显存数值。** README 中 RTX 3090 的通用 `total` 运行表、训练 GPU 数量和 `--fast` 节省内存建议都不是这两个禁用 `fast` 任务的硬件保证。Mac 系统测试和 Ubuntu 子任务测试的存在也不等于每个病灶 checkpoint 已经在 MPS CI 上运行。[通用资源说明及 CI][ts-readme]。

接入应分别记录网络实际设备、裁剪阶段与主模型阶段用时、CPU/RAM 与 GPU 峰值、输出几何、各标签体素数量、退出码和异常。MPS 算子缺失、OOM、超时必须报出真实错误，不可自动转 CPU 后仍标记为 MPS 成功；若明确允许 CPU 回退，应单独记录 `effective_device` 与回退原因。两台机器应各完成阳性病例和失败边界检查后再宣传平台支持。

## 2025–2026 论文与运行权重不能直接画等号

肝模型对应 Nicoli 等的 **Liver Segment and Lesion Segmentation on CT and MRI: An Open-Source Contribution to TotalSegmentator**，2025-10-24 在线发表，收录于 2026 年期刊卷期。[论文 DOI][liver-paper]、[PubMed][liver-pubmed]。摘要报告一个 750 图像 CT/MR 病灶训练模型，CT 病灶 Dice 0.658、MRI 0.337，并报告漏检和每例假阳性；这些是作者特定队列与方法的结果，不是当前平台验收目标或临床承诺。

当前 CT 权重名称与 2026-05-18 官方训练数据记录均标为 **842 CT 病例**，与论文摘要的 750 CT/MR 混合模型不同。未找到足以把论文所有性能数字逐项归到当前 Dataset591 权重的证据，因此不将旧论文数字直接标为当前 `liver_lesions` 的性能。[2026 数据说明][liver-data]、[当前权重资产][ts-weights]。MR 病灶模型另有 Dataset589 权重与约 20.5 GB MR 训练集；在 MRI 序列/采集范围、MPS 推理与结果质量未实测前，不顺带开放 `liver_lesions_mr`。[MR 数据][liver-mr-data]。

`lung_nodules` 的官方任务说明称其由 BLUEMIND AI 提供，训练来自 1,353 例、部分来源 LIDC-IDRI；本次没有找到随当前 checkpoint 发布的独立完整模型论文或逐设备模型卡。该数字作为发布方声明记录，不将来源不明的网络结果补成性能证据。[任务说明][ts-readme]。它仍比需要额外编排的候选更适合先做工程 canary，因为权重、接口和依赖路径都可核验。

## 其他候选的比较与暂缓依据

| 候选 / 已核验可获取版本 | 许可与输入/输出 | 硬件和接口事实 | 本轮处理 |
| --- | --- | --- | --- |
| TS `kidney_cysts` / 2.18.0 / Dataset789 | 开放任务；左右肾囊肿；2025-01-15 权重约 230.6 MB | 同一设备路径，1.5 mm；基础 `total` 已有 `kidney_cyst_left/right`，发布方称 specialized 版本更准确，但本研究未独立测量 | 不是缺失能力，优先级低于肺结节/肝病灶。[配置][ts-config]、[基础标签][ts-labels] |
| TS `cerebral_bleed` / 2.18.0 / Dataset150 | 开放任务；`intracerebral_hemorrhage`；权重 325,569,459 B，2023-09-21 | CT 脑内出血，先脑定位，不是所有颅内出血亚型，也不用于起病时间预测；引用论文主要研究 ICH 起病时间，结果并未支持可靠起病估计 | 可作为以后第三个固定任务，但不是 2025–2026 新病灶模型，本次未准备阳性头颅数据。[权重][bleed-weights]、[原论文][bleed-paper] |
| MONAI `lung_nodule_ct_detection` **0.6.10** | Bundle Apache；CT；输出检测框/分数的字典，而非病灶 NIfTI | 固定模型元数据 MONAI 1.4.0 / torch 2.4.0；配置默认 CUDA 或 CPU，`amp=true`，无 MPS 选择路径 | 与掩膜交付契约不一致，不用框伪造分割。[正式 HF revision][monai-lung] |
| MONAI `pancreas_ct_dints_segmentation` **0.5.2** | Bundle Apache；门静脉期 CT；胰腺与胰腺肿瘤；checkpoint 553,830,837 B | 有正式 HF 模型文件；推理 YAML 的架构 checkpoint `map_location='cuda'`、DiNTS device 也硬编码 CUDA，不能仅改外层 `device` 就声称 MPS；16 GB GPU 文档数字属于训练 | 能补胰腺肿瘤，但需独立兼容性改造及门静脉期样本；本轮暂缓。[固定 bundle][monai-pancreas]、[推理 YAML][pancreas-config] |
| MONAI `brats_mri_segmentation` **0.5.4** | Bundle Apache；4 通道 T1c/T1/T2/FLAIR → 3 类肿瘤区域 | 正式 HF revision 有权重；现接口只有一张单通道 3D 图像，不能补零或复制序列代替四序列；默认 CUDA/CPU | 输入契约不合，暂缓。[模型元数据][monai-brats] |
| MONAI `renalStructures_CECT_segmentation` **0.2.3** | Bundle Apache；数据另为 CC BY-NC-SA 4.0；肾肿瘤等 6 类 | 输入动脉、静脉、排泄三期 CT，需配准和 3 通道；官方训练为 RTX2080Ti，不能拿训练配置当单图推理要求 | 多期输入、配准与当前接口不合，暂缓。[固定 bundle][monai-renal] |
| MONAI `vista3d` **0.5.11** | 代码 Apache；该固定 bundle 权重是 NVIDIA 非商业研究/评估许可；自动类别 ID 或空间点提示 | 132 类含肿瘤，约 872 MB；MONAI 1.4.0 / torch 2.4.0 元数据；CUDA/CPU 默认，不能据此声称 MPS | 不沿用旧权重许可证概括新 NVIDIA 发布，见下文；本轮不安装。[固定 bundle][monai-vista]、[其权重许可][vista-old-license] |
| NVIDIA **NV-Segment-CT**，HF revision `afb5151…`（2026-04-01） | 代码 Apache；权重 **NVIDIA Open Model License**，明确可商用；CT，自动类别 ID + 空间点；约 872 MB，可取 safetensors | 是 2025 CVPR VISTA3D 的当前维护路线，能自动选择肺/肝/胰腺等已定义肿瘤类；官方卡测试 A100/H100，AMP 代码仍调用 `torch.autocast('cuda')`，仓库依赖固定 torch 2.1.2、MONAI 1.4.0、numpy 1.24.4，不能直接覆盖本项目 uv 环境 | 比旧 VISTA 快照更值得保留的广泛 CT 病灶候选，但 MPS 与全链路成本未验证，优先级仍低于两个 TS 任务。[模型卡/文件][nv-ct]、[许可][nv-ct-license]、[固定实现][nv-code] |
| NVIDIA **NV-Segment-CTMR**，HF revision `4fb8b4a…`（2026-03-31） | 权重仍为 NVIDIA 非商业许可；2025-10 发布；345+ 类、CT+MR，自动分割分支 | 不能把 CT 版本的商业友好许可或交互功能套用到 CTMR；主要提升广泛解剖覆盖 | 基础解剖覆盖重复较多，暂缓。[官方比较][vista-new]、[模型卡][nv-ctmr] |
| **nnInteractive 2.5.1**，PyPI 2026-07-07；v1 权重 | 代码 Apache；权重 **CC BY-NC-SA 4.0**；点、二维框、涂画、套索 → 二值目标；checkpoint 411,387,150 B | 2025 原论文；官方推荐 NVIDIA 10 GB VRAM，小物体可低于 6 GB；官方 napari 文档直说 MPS/CPU 的 3D 卷积慢，Mac 推荐远程 GPU；没有文本定位接口 | 留作将来明确空间提示的交互修正；Mac 遥控服务器不能算 Mac 本地 MPS。[PyPI][nn-pypi]、[权重许可][nn-license]、[Mac 说明][nn-mac] |
| **SAM-Med3D**，正式安装入口 **medim 0.1.2**（2025-05-18），turbo 权重 | 代码 Apache；作者 HF README 正文声明 `apache-2.0`，但未放置规范开头 front matter、API license 字段缺失、无独立 LICENSE；约 402 MB | 原论文 2023/2024，2025 有官方 CVPR 分支；示例只选 CUDA/CPU，无 MPS 分支；空间提示二值分割，不是文本目标定位 | 原 demo 使用 **真值选类别、真值裁剪 ROI、真值生成点击**；不能当自动自然语言闭环的证据。与 nnInteractive 功能重复，本轮不接。[正式安装入口][sam-medim]、[作者权重声明][sam-weights]、[真值驱动示例][sam-infer] |

MONAI bundle 版本不仅来自 GitHub `dev` 的 metadata：上表全部核查了对应 Hugging Face revision 存在、`models/model.pt` 文件存在；部分发行是 2025-06-30，VISTA 0.5.11 为 2025-11-05。MONAI Python 包本身已有 [1.6.0（2026-06-22）][monai-pypi]，但这并不把旧 bundle 的 torch/MONAI 配置自动升级为已验证。仓库许可说明要求分别遵守 bundle 与数据许可。[MONAI Model Zoo 说明][model-zoo]。

最新 VISTA 路线值得单独强调：官方维护 README 在 2025-10 发布 CTMR 后，明确建议 **CT 肿瘤或交互修正使用 NV-Segment-CT**；其新版权重许可与旧 MONAI `vista3d` 文件不同。上表保留这个变化，同时保留其 MPS 未测、CUDA AMP 与旧依赖约束，避免因“最新”而忽视安装现实。[官方维护说明][vista-new]。

## 可获取的最小阳性工程 canary

### 肝病灶：可以避免下载整个 38.9 GB ZIP

官方 [Zenodo 20272571][liver-data] 指向固定版本记录 `20272572`，含 842 对 CT/手工标签、nnU-Net NIfTI 格式。固定文件地址：

`https://zenodo.org/api/records/20272572/files/Dataset591_liver_lesions.zip/content`

本次真实执行了 `Range: bytes=-1048576`，服务返回 **206** 与完整 `Content-Range`，只读 1 MiB ZIP 中央目录。目录内找到：

| ZIP 成员 | ZIP 内压缩字节 | 解压后的 `.nii.gz` 字节 | 本次读取情况 |
| --- | ---: | ---: | --- |
| `imagesTr/SzFnYhqzTtnDAHXq_0000.nii.gz` | 7,321,970 | 7,319,742 | 只读取目录元数据，未下载 CT |
| `labelsTr/SzFnYhqzTtnDAHXq.nii.gz` | 9,594 | 14,251 | 精确 Range 提取；大小与 ZIP CRC32 校验通过 |

标签实际解析结果为 **212 × 182 × 142**、**1.5234375 × 1.5234375 × 2 mm**，背景 5,427,780 体素、`liver_lesion=1` **51,148 体素**。这确立阳性标签可用，但尚未验证配对 CT 的内容、推理成功或预测重叠。不要把训练数据 canary 当独立外部验证，不从标签生成模型输入或裁剪病灶 ROI。

后续下载应仅提取这一对成员，拒绝服务器忽略 Range 后返回整个 ZIP，设置最大成员/总下载字节数，校验 CRC 与长度，保存来源和新计算的 SHA-256。若 Range 失效，记录阻塞并转向独立路线，不能悄悄启动 38.9 GB 下载。

### 肺结节：LIDC-IDRI 单个 CT 与同研究 SEG

官方 [LIDC-IDRI][lidc] 和 [标准化结节 SEG][lidc-seg] 提供公开 CC BY 3.0 数据，SEG 记录针对 ≥3 mm 结节。本次真实 [NBIA 查询][lidc-query] 返回患者 `LIDC-IDRI-0001` 的 CT 及同一 `StudyInstanceUID` 的四个读者结节注释。可先选一位读者用于工程对齐核查：

- CT：`1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192`，133 张，API 报告原 DICOM 总计 **70,018,838 B**。[单序列 ZIP][lidc-ct-download] HEAD 实际返回 200 和 ZIP 附件名；不能将原 DICOM 总字节等同 ZIP 大小。
- SEG：`1.2.276.0.7230010.3.1.3.0.89314.1553284067.990548`，说明为 “Segmentation of Nodule 1 - Annotation Nodule 001”，API 原 DICOM 大小 **270,284 B**；[单 SEG ZIP][lidc-seg-download] HEAD 实际返回 200、ZIP 大小 **4,466 B**。本研究未下载影像或 SEG 像素。

下载后需校验 CT 单一 Series、方向、间距、HU 转换；SEG 必须通过引用 SOP Instance UID / 帧空间位置映射回同一 CT 网格，不能按文件名排序或简单重复切片。先保留一个读者掩膜，若以后构建共识则单独记录规则。原始 CT 与 SEG 转 NIfTI 后才进入单张 Web 图像流程。LIDC 与模型训练来源可能重叠，因此只作为工程 canary。

## 接入后应执行的验收与路由

1. 下载权重前固定包版本、模型 ID、精确资产地址；下载后保存 SHA-256、许可快照和实际 checkpoint 信息。推理复用同一核心，不新建 Web/A2A 私有推理实现。
2. 给 `lung_nodules` 与 `liver_lesions` 单独登记 modality、允许标签、`fast=False`、超时和资源限制。CT 肺结节请求只映射前者，CT 肝病灶请求只映射后者；MR、泛全身肿瘤、恶性判断、无目标请求均不得偷偷改选工具。自动器官裁剪只来自输入影像与本地模型，参考标签密封到推理完成后才比较。
3. Mac MPS 和 pjoffice CUDA 各运行同一阳性 CT，并分别记录加载/裁剪/主推理/总时长、显存/RAM、几何、离散标签、病灶非空与可视叠加。肺任务必须检查 label 2，不能以 label 1 肺体素通过验收。记录 Dice/体积/重叠作为本例描述，不把单例或训练集成绩推广为临床性能。
4. 加入阴性、错误模态、错误部位、截断 NIfTI、NaN、超限、任务取消/超时与重启恢复。对合法空病灶掩膜保留明确 `no_target_detected` 语义和原图/日志，不能推论“没有疾病”；模型错误与合法空检出必须可区分。
5. 对相同病例测试 CLI、MCP、Web、A2A 的同核结果，以及公网真实请求、断线 GetTask 和受鉴权结果文件。任一平台未跑通就保留未支持状态。两个优先候选通过之前，不加入第三个大型模型。

研究停止条件：优先候选的正式发行、公开权重声明、任务/设备代码、最小阳性数据路线和主要替代方案排除依据已经有一手证据；剩余关键问题是必须执行的实际安装、MPS/CUDA 推理、转换几何和运行边界，继续堆叠论文无法替代。曾遇到 LNDb 页面与 MDPI 403、旧 TCIA 文档 URL 404，已转向可读取的官方 TCIA API、Crossref/PubMed 与固定源码，没有据不可访问页面推断已完成。

[ts-pypi]: https://pypi.org/project/TotalSegmentator/2.18.0/
[ts-readme]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/README.md
[ts-license]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/LICENSE
[ts-config]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/map_tasks_config.py
[ts-labels]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/map_to_binary.py
[ts-api]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/python_api.py
[ts-download]: https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/totalsegmentator/libs.py
[ts-weights]: https://github.com/wasserth/TotalSegmentator/releases/tag/v2.5.0-weights
[nnunet-infer]: https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/nnunetv2/inference/predict_from_raw_data.py
[liver-paper]: https://doi.org/10.1007/s10278-025-01716-y
[liver-pubmed]: https://pubmed.ncbi.nlm.nih.gov/41136714/
[liver-data]: https://doi.org/10.5281/zenodo.20272571
[liver-mr-data]: https://doi.org/10.5281/zenodo.20272347
[bleed-paper]: https://doi.org/10.3390/jcm12072631
[bleed-weights]: https://github.com/wasserth/TotalSegmentator/releases/tag/v2.0.0-weights
[monai-pypi]: https://pypi.org/project/monai/1.6.0/
[model-zoo]: https://github.com/Project-MONAI/model-zoo
[monai-lung]: https://huggingface.co/MONAI/lung_nodule_ct_detection/tree/0.6.10
[monai-pancreas]: https://huggingface.co/MONAI/pancreas_ct_dints_segmentation/tree/0.5.2
[pancreas-config]: https://huggingface.co/MONAI/pancreas_ct_dints_segmentation/blob/0.5.2/configs/inference.yaml
[monai-brats]: https://huggingface.co/MONAI/brats_mri_segmentation/tree/0.5.4
[monai-renal]: https://huggingface.co/MONAI/renalStructures_CECT_segmentation/tree/0.2.3
[monai-vista]: https://huggingface.co/MONAI/vista3d/tree/0.5.11
[vista-old-license]: https://huggingface.co/MONAI/vista3d/blob/0.5.11/LICENSE
[vista-new]: https://github.com/Project-MONAI/VISTA/tree/main/vista3d
[nv-ct]: https://huggingface.co/nvidia/NV-Segment-CT/tree/afb51518689f71e6abb367ee6301b2cd0225c66a
[nv-ct-license]: https://huggingface.co/nvidia/NV-Segment-CT/blob/afb51518689f71e6abb367ee6301b2cd0225c66a/LICENSE
[nv-code]: https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/cb921f5c58837c0f42a713855d68b32af88e1cdd
[nv-ctmr]: https://huggingface.co/nvidia/NV-Segment-CTMR/tree/4fb8b4a6b2532be9f1c449a3726fe5440ab4213a
[nn-pypi]: https://pypi.org/project/nnInteractive/2.5.1/
[nn-license]: https://huggingface.co/MIC-DKFZ/nnInteractive/blob/3f308d751c00644e4fde6f09c600264b393b21b5/nnInteractive_v1.0/LICENSE
[nn-mac]: https://github.com/MIC-DKFZ/napari-nninteractive/blob/53d004279a72d49b7d6b388caa0275421349bb47/README.md
[sam-medim]: https://pypi.org/project/medim/0.1.2/
[sam-weights]: https://huggingface.co/blueyo0/SAM-Med3D/blob/fc482a040ea69cdc9ae576a1d6bd9db02ab1994c/README.md
[sam-infer]: https://github.com/uni-medical/SAM-Med3D/blob/f3de1fa10da98e46f49f176773d2b1e306ba131f/utils/infer_utils.py
[lidc]: https://www.cancerimagingarchive.net/collection/lidc-idri/
[lidc-seg]: https://www.cancerimagingarchive.net/analysis-result/dicom-lidc-idri-nodules/
[lidc-query]: https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=LIDC-IDRI&PatientID=LIDC-IDRI-0001&format=json
[lidc-ct-download]: https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192
[lidc-seg-download]: https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=1.2.276.0.7230010.3.1.3.0.89314.1553284067.990548
