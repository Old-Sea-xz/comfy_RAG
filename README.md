# Agentic RAG

基于工作流检索的 ComfyUI Agent：**首次 E5 Top-20 检索 → Agent 判断是否追加检索 → 选择工作流链 → 生成、执行与修复时按需检索**。

默认 `agentic_feedback` 流程先用完整请求检索匿名工作流库。Agent 判断候选是否覆盖全部需求；缺少能力时提出针对性查询，最多检索三次，候选足够时立即选择并进入执行。不调用分类树或 JEV。

每次生成/修复规划前，Agent 都可独立判断是否需要追加本地工作流检索，涵盖初始准备、预检报错、执行报错和质量迭代。需要时提供最多三条查询；每条返回 Top-20 描述及前三份参考 API 图，送入当前规划器。参考图用于指导受校验的修复，不会直接覆盖正在执行的图。外部资料搜索仍由原修复诊断 Agent 独立判断。

生成执行最多三轮（首次执行 + 两轮迭代）；追加检索不消耗 GPU 执行次数，也不会重置轮次。同一错误上下文的检索判断与结果保存为检查点，通信恢复复用已完成记录。最后一轮结束后不再做无法用于后续执行的检索。

本包基于此前独立整理版恢复旧代码中的反馈检索分支，并按本次要求新增中途修复的本地 RAG 入口；不是旧实验源码逐字节快照。来源与调整见 [docs/PROVENANCE.md](docs/PROVENANCE.md)。

## 内容

| 路径 | 作用 |
| --- | --- |
| `comfyrouter/agentic_retrieval.py` | 生成/修复中的按需工作流检索与检查点 |
| `comfyrouter/semantic_retrieval.py` | 本地 multilingual-e5-small 编码 |
| `comfyrouter/whole_request_selection.py` | 判断是否追加检索，或选择工作流及依赖 |
| `comfyrouter/case_execution.py` | 共享执行器、输入绑定、检查点及修复 |
| `comfyrouter/agentic.py` | OpenAI-compatible Responses、原生 function calling、请求记录 |
| `comfyrouter/api_serial.py`、`native_queue.py` | 共享 API 锁和 ComfyUI 队列 |
| `library/` | 66 份匿名描述、API 工作流和对应检索向量 |
| `tools/prepare_run.py`、`run.py`、`progress.py` | 准备批次、运行/恢复、简洁进度 |
| `tests/` | 离线回归测试 |
| `integrations/comfyui/` | 默认队列模式所需的服务端扩展及安装器 |

发布副本只保留 RAG 所需模块。分类树、JEV 分类器、独立需求拆分、模型分类与跨任务自进化代码均已移除。工作流组合及依赖在候选足够时由联合选择直接返回。

## 安装

运行环境：Linux、Python 3.12+、已部署的 ComfyUI。建议使用 Python 3.12；视频测试/评测需要 FFmpeg。模型权重与自定义节点由 ComfyUI 环境提供。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写自己的模型、API 地址与密钥。默认模型配置为 `gpt-6-sol`；模型 ID 需与你使用的服务一致。API 支持 Responses 和原生函数调用，也可用 `COMFYROUTER_LLM_RESULT_PROTOCOL=json_schema` 选择结构化 JSON 协议。

本地下载检索编码器：

```bash
hf download intfloat/multilingual-e5-small --local-dir models/multilingual-e5-small
```

官方视觉评测调用独立安装的 [ComfyBench](https://github.com/xxyQwQ/ComfyBench) 的 `script.evaluation`。配置 `COMFYROUTER_COMFYBENCH_ROOT` 为其本地目录；配置 `COMFYROUTER_COMFYUI_ROOT` 为 ComfyUI 本地目录。完整 benchmark 请求、原始媒体、模型权重和评测结果未随仓库发布。

如需要 7890 代理，在 `.env` 设置：

```dotenv
COMFYROUTER_EXTERNAL_PROXY_URL=http://127.0.0.1:7890
COMFYROUTER_LLM_PROXY_URL=http://127.0.0.1:7890
```

本地 ComfyUI 请求使用直连；API 和 GPU 锁的默认位置与原共享执行逻辑兼容。同机其他客户端也必须使用相同锁才能相互协调。ComfyUI 的排队窗口限制待执行 prompt，GPU 工作流按服务端队列依次执行。

默认队列模式需要服务端 `/queue_cleanup` 接口。请按 [ComfyUI 队列扩展说明](integrations/comfyui/README.md) 在独立、停止的 ComfyUI checkout 中安装；包含版本校验、试运行和原文件备份。原版 ComfyUI 不具备此接口。发布整理不会修改现有 ComfyUI 或正在运行的实验。

## 准备任务

任务文件是按 case ID 索引的 JSON 对象，格式见 [examples/tasks.json](examples/tasks.json)。每题包含 `instruction`、`modality`、`category`、`resources`、`input_types`，可附 `evaluation_criteria`。

支持 `t2i`、`i2i`、`i2v`、`t2v`、`v2v`。资源路径相对于 `--resources` 目录；蒙版必须显式标为 `mask`。案例 ID 可以沿用原 benchmark ID；整理代码不会重写请求或评测标准。

```bash
python tools/prepare_run.py \
  --tasks /path/to/tasks.json \
  --resources /path/to/resources \
  --model models/multilingual-e5-small \
  --run runs/rag_batch
```

准备时从运行中的 ComfyUI 读取 `/object_info`；离线准备可传 `--registry /path/to/object_info.json`。新批次会冻结工作流、提示契约、节点接口及哈希；不会复制任何旧路由或成功结果。

默认最多 8 个 case worker，ComfyUI 待执行窗口为 3，资源预算 115 GiB，可通过 `--workers`、`--queue-window`、`--vram-gib` 按机器资源调整。首次执行对照用 `--max-attempts 1`；三轮用默认 `--max-attempts 3`。

## 运行与查看进度

```bash
python -u tools/run.py --run runs/rag_batch
```

在另一终端查看简洁状态：

```bash
python -u tools/progress.py --run runs/rag_batch
```

单次查看追加 `--once`。进度显示 GPU、Comfy 队列、RAG 路由、生成准备、评价/修复、通信重试、待恢复、通过数及失败类型。

正常停止后的恢复使用同一条 `tools/run.py` 命令，已完成结果和检查点会被保留。若 worker 意外消失或 GPU 提交状态不明确，协调器会把该题放入待核对状态并继续其他案例；不会盲目重复提交。详见 [执行与恢复说明](docs/METHOD.md)。

## 验证

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m compileall -q comfyrouter tools integrations
```

测试使用模拟模型和执行器，不请求付费 API，也不启动 GPU 生成。两个 VHS 集成检查需要本地已安装的 ComfyUI-VideoHelperSuite 源码，缺失时跳过。离线测试不代表实际生成准确率；实际运行仍需部署对应节点、模型和评测环境。

## 发布范围

发布文件仅包含代码、提示契约、匿名工作流库、脱敏示例和测试。媒体输入槽已替换为通用占位符。真实 `.env`、凭据、私人服务地址、用户绝对路径、实验日志、HTTP 返回、生成产物、benchmark 媒体和模型文件均不包含在 Git 中。

本地运行会生成请求和响应记录，存放于 Git 忽略的 `runs/` 目录。维护者应继续保留这一隔离方式。发布整理的具体兼容调整见 [docs/PORTABILITY.md](docs/PORTABILITY.md)。
