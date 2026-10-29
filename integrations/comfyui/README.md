# ComfyUI 队列与内存释放接口

默认执行模式依赖本扩展的 `/queue_cleanup` 接口。原版 ComfyUI 不提供此接口，运行器会拒绝缺少该接口的服务。

扩展只增加串行 prompt 边界的模型/缓存释放、RSS 检查和带队列检查点的进程重启，并保留清理前真实的执行状态、错误消息和产物。队列深度不表示 GPU 并行执行。

## 安装到独立 ComfyUI

支持版本及原始文件 SHA-256 见 `upstream.json`。安装器拒绝修改不同版本或已有本地修改的 `main.py`、`server.py`；请先在独立 checkout 中验证，不覆盖其他实验环境。

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git /path/to/ComfyUI
git -C /path/to/ComfyUI checkout 0a33ed6c28f926d14536235771c222f9e6d1026b
# 在 ComfyUI 自己的 Python 环境内安装原项目依赖及 psutil
python -m pip install psutil
python integrations/comfyui/install.py --comfyui /path/to/ComfyUI
# 确保目标 ComfyUI 已停止，再执行写入
python integrations/comfyui/install.py --comfyui /path/to/ComfyUI --apply
```

启动该 ComfyUI 后验证：

```bash
curl --noproxy '*' http://127.0.0.1:8188/queue_cleanup
```

预期包含 `supported: true`、`phase` 和 `boot_id`。仅复制两个 helper 不能替代 `main.py` 与 `server.py` 的集成。

Linux/glibc 环境下，`COMFYROUTER_RELEASE_RSS_LIMIT_GIB` 默认 10 GiB，`COMFYROUTER_RELEASE_TIMEOUT_SECONDS` 默认 30 秒。它们是每次 prompt 前后清理后的进程 RSS 阈值及等待时间，不是 GPU 执行时限或模型 VRAM 限额。超时会保存队列/完成记录并替换当前 Python 进程。请确保启动方式允许进程自重启，且临时目录可写。恢复文件包含本地任务数据，应留在本机；不要提交到仓库。

回退时先停止目标服务，将两个 `.comfy-rag-backup` 恢复为原文件，并删除新增的两个 helper。安装不会自动停止、启动或更改其他 ComfyUI。

离线测试直接执行随包发布的 worker 函数及内存 helper，使用模拟执行器验证清理顺序和失败状态；不需要启动 GPU。派生 worker 的上游许可与出处见 `LICENSE`、`NOTICE`。
